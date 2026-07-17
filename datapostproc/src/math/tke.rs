/// Reynolds-stress components and turbulent kinetic energy.
///
/// Coordinate convention (matches the rest of this crate):
///   array shape (nz, ny, nx): axis 0 = wall-normal z, axis 1 = spanwise y,
///   axis 2 = streamwise x; u = streamwise, v = spanwise, w = wall-normal.
///
/// All inputs are TOTAL second moments ⟨a·b⟩ — either the subavg datasets
/// (uu, vv, …) or ensemble means of instantaneous products.  Everything is
/// spanwise-averaged first and the fluctuation moments are extracted about
/// the span+time mean (same convention as `math/fik.rs`):
///
///     ⟨a'b'⟩ = ⟨ab⟩ − ā·b̄,        k = ½(⟨u'u'⟩ + ⟨v'v'⟩ + ⟨w'w'⟩)
///
/// so spanwise-coherent motion (e.g. from spanwise-periodic control) counts
/// as fluctuation, not mean.

use ndarray::{Array1, ArrayD};
use hdf5::Error;

use super::avg::avg_axis;
use super::ns::{deriv1, deriv2};

// ─── Public result type ────────────────────────────────────────────────────────

/// Spanwise-averaged Reynolds stresses and TKE, each of shape `(nz, nx)`.
pub struct TkeFields {
    /// ⟨u'u'⟩ — streamwise normal stress.
    pub uu: ArrayD<f64>,
    /// ⟨v'v'⟩ — spanwise normal stress.
    pub vv: ArrayD<f64>,
    /// ⟨w'w'⟩ — wall-normal normal stress.
    pub ww: ArrayD<f64>,
    /// ⟨u'v'⟩ — streamwise/spanwise shear stress.
    pub uv: ArrayD<f64>,
    /// ⟨u'w'⟩ — streamwise/wall-normal shear stress.
    pub uw: ArrayD<f64>,
    /// ⟨v'w'⟩ — spanwise/wall-normal shear stress.
    pub vw: ArrayD<f64>,
    /// k = ½(⟨u'u'⟩ + ⟨v'v'⟩ + ⟨w'w'⟩).
    pub tke: ArrayD<f64>,
}

// ─── Main function ─────────────────────────────────────────────────────────────

/// Extract the Reynolds-stress components and TKE from total moments.
///
/// # Arguments
/// * `u`, `v`, `w`                      – mean velocities, shape `(nz, ny, nx)`.
/// * `uu`, `vv`, `ww`, `uv`, `uw`, `vw` – **total** second moments ⟨a·b⟩,
///                                        same shape.
#[allow(clippy::too_many_arguments)]
pub fn tke_fields(
    u:  &ArrayD<f64>,
    v:  &ArrayD<f64>,
    w:  &ArrayD<f64>,
    uu: &ArrayD<f64>,
    vv: &ArrayD<f64>,
    ww: &ArrayD<f64>,
    uv: &ArrayD<f64>,
    uw: &ArrayD<f64>,
    vw: &ArrayD<f64>,
) -> Result<TkeFields, Error> {
    let shape = u.shape().to_vec();
    if shape.len() != 3 {
        return Err(format!("expected 3-D fields (nz, ny, nx), got {shape:?}").into());
    }
    for (name, arr) in [
        ("v", v as &ArrayD<f64>), ("w", w),
        ("uu", uu), ("vv", vv), ("ww", ww),
        ("uv", uv), ("uw", uw), ("vw", vw),
    ] {
        if arr.shape() != shape.as_slice() {
            return Err(format!(
                "'{name}' shape {:?} does not match u's shape {shape:?}",
                arr.shape()
            ).into());
        }
    }

    // ── spanwise average (axis 1) → (nz, nx) ──────────────────────────────────
    let avg = |a: &ArrayD<f64>| avg_axis(a, 1);
    let u2 = avg(u)?;
    let v2 = avg(v)?;
    let w2 = avg(w)?;

    // ── fluctuation moments: ⟨a'b'⟩ = ⟨ab⟩ − ā·b̄ ─────────────────────────────
    let uu_p = &avg(uu)? - &(&u2 * &u2);
    let vv_p = &avg(vv)? - &(&v2 * &v2);
    let ww_p = &avg(ww)? - &(&w2 * &w2);
    let uv_p = &avg(uv)? - &(&u2 * &v2);
    let uw_p = &avg(uw)? - &(&u2 * &w2);
    let vw_p = &avg(vw)? - &(&v2 * &w2);

    let tke = (&(&uu_p + &vv_p) + &ww_p) * 0.5;

    Ok(TkeFields {
        uu: uu_p,
        vv: vv_p,
        ww: ww_p,
        uv: uv_p,
        uw: uw_p,
        vw: vw_p,
        tke,
    })
}

// ─── Reynolds-stress transport budget ─────────────────────────────────────────

/// Velocity component letters, indexed 0/1/2 = u/v/w.
const VEL: [char; 3] = ['u', 'v', 'w'];
/// Direction letters, indexed 0/1/2 = x/y/z (same index as the velocity that
/// points along it: u↔x streamwise, v↔y spanwise, w↔z wall-normal).
const DIR: [char; 3] = ['x', 'y', 'z'];

/// subavg dataset name of the total second moment ⟨u_a·u_b⟩ (uu, uv, …, ww).
fn mom2_name(a: usize, b: usize) -> String {
    let (a, b) = (a.min(b), a.max(b));
    format!("{}{}", VEL[a], VEL[b])
}

/// subavg dataset name of the total third moment ⟨u_a·u_b·u_c⟩.
///
/// Follows the DNS writer's convention (same as the Python `_sortName`):
/// sort ascending, then if the last two letters form the repeated pair, move
/// the single one to the end — e.g. {v,v,u} → `vvu`, {u,u,w} → `uuw`,
/// {u,v,w} → `uvw`.
fn mom3_name(a: usize, b: usize, c: usize) -> String {
    let mut v = [a, b, c];
    v.sort();
    if v[1] == v[2] && v[0] != v[1] {
        format!("{}{}{}", VEL[v[1]], VEL[v[2]], VEL[v[0]])
    } else {
        format!("{}{}{}", VEL[v[0]], VEL[v[1]], VEL[v[2]])
    }
}

/// Budget terms of the ⟨u_i'u_j'⟩ transport equation, each spanwise-averaged
/// to shape `(nz, nx)`.  Signs follow the right-hand side of
///
///   C_ij = P_ij + T_ij + D_ij + Φ_ij + Π_ij − ε_ij
///
/// so that `balance() = prod + turb_trans + visc_trans + press_strain
/// + press_trans − visc_diss − conv ≈ 0` for converged statistics.
pub struct BudgetTerms {
    /// The stress ⟨u_i'u_j'⟩ itself.
    pub stress: ArrayD<f64>,
    /// Production P = −(⟨u_i'u_k'⟩ ∂ū_j/∂x_k + ⟨u_j'u_k'⟩ ∂ū_i/∂x_k).
    pub prod: ArrayD<f64>,
    /// Turbulent transport T = −∂⟨u_i'u_j'u_k'⟩/∂x_k.
    pub turb_trans: ArrayD<f64>,
    /// Viscous transport D = ν ∂²⟨u_i'u_j'⟩/∂x_k².
    pub visc_trans: ArrayD<f64>,
    /// Pressure strain Φ = ⟨p'(∂u_i'/∂x_j + ∂u_j'/∂x_i)⟩ (ρ = 1).
    pub press_strain: ArrayD<f64>,
    /// Pressure transport Π = −(∂⟨p'u_j'⟩/∂x_i + ∂⟨p'u_i'⟩/∂x_j).
    pub press_trans: ArrayD<f64>,
    /// Viscous (pseudo-)dissipation ε = 2ν⟨∂u_i'/∂x_k · ∂u_j'/∂x_k⟩ ≥ 0,
    /// reported positive and SUBTRACTED in the balance.
    pub visc_diss: ArrayD<f64>,
    /// Mean convection C = ū_k ∂⟨u_i'u_j'⟩/∂x_k.
    pub conv: ArrayD<f64>,
}

impl BudgetTerms {
    /// Residual of the budget; ≈ 0 for converged statistics, and a direct
    /// diagnostic of statistical + discretization error otherwise.
    pub fn balance(&self) -> ArrayD<f64> {
        &(&(&(&(&self.prod + &self.turb_trans) + &self.visc_trans)
            + &(&self.press_strain + &self.press_trans))
            - &self.visc_diss)
            - &self.conv
    }

    /// Linear combination `self + w · other` term by term (for assembling the
    /// TKE budget as half the trace of the diagonal budgets).
    pub fn axpy(&self, w: f64, other: &BudgetTerms) -> BudgetTerms {
        let f = |a: &ArrayD<f64>, b: &ArrayD<f64>| a + &(b * w);
        BudgetTerms {
            stress:       f(&self.stress, &other.stress),
            prod:         f(&self.prod, &other.prod),
            turb_trans:   f(&self.turb_trans, &other.turb_trans),
            visc_trans:   f(&self.visc_trans, &other.visc_trans),
            press_strain: f(&self.press_strain, &other.press_strain),
            press_trans:  f(&self.press_trans, &other.press_trans),
            visc_diss:    f(&self.visc_diss, &other.visc_diss),
            conv:         f(&self.conv, &other.conv),
        }
    }

    /// Scale every term by `w`.
    pub fn scale(&self, w: f64) -> BudgetTerms {
        let f = |a: &ArrayD<f64>| a * w;
        BudgetTerms {
            stress:       f(&self.stress),
            prod:         f(&self.prod),
            turb_trans:   f(&self.turb_trans),
            visc_trans:   f(&self.visc_trans),
            press_strain: f(&self.press_strain),
            press_trans:  f(&self.press_trans),
            visc_diss:    f(&self.visc_diss),
            conv:         f(&self.conv),
        }
    }
}

/// Compute the full transport budget of ⟨u_i'u_j'⟩ from a subavg file's
/// stored total moments.
///
/// `load` fetches a dataset by its subavg name and returns it as a 3-D
/// `(nz, ny, nx)` array aligned with the coordinate lengths; all fluctuation
/// moments are extracted about the per-phase time mean, every term is
/// evaluated pointwise in (z, y, x) — the exact stationary transport
/// equation, no spanwise-homogeneity assumption — and spanwise-averaged at
/// the end.
///
/// Production, dissipation and pressure strain use the DNS's stored mean
/// gradients (ux…wz, uxux…wzwz, pux…pwz); convection, turbulent transport,
/// viscous transport and pressure transport differentiate numerically.
/// Numerical ∂/∂y terms need `ny ≥ 3` and are skipped (left zero) otherwise,
/// e.g. for fully span-averaged files with ny = 1.
#[allow(clippy::too_many_arguments)]
pub fn stress_budget(
    load: &mut dyn FnMut(&str) -> Result<ArrayD<f64>, Error>,
    i: usize,
    j: usize,
    x:  &Array1<f64>,
    y:  &Array1<f64>,
    zc: &Array1<f64>,
    nu: f64,
    periodic_x: bool,
) -> Result<BudgetTerms, Error> {
    if i > 2 || j > 2 {
        return Err(format!("component indices must be 0..=2, got ({i}, {j})").into());
    }
    let with_y = y.len() >= 3;

    // numerical derivative along direction k (0/1/2 = x/y/z)
    let d1 = |f: &ArrayD<f64>, k: usize| -> Result<ArrayD<f64>, Error> {
        match k {
            0 => deriv1(f, 2, x, periodic_x),
            1 => deriv1(f, 1, y, true),
            _ => deriv1(f, 0, zc, false),
        }
    };
    let d2 = |f: &ArrayD<f64>, k: usize| -> Result<ArrayD<f64>, Error> {
        match k {
            0 => deriv2(f, 2, x, periodic_x),
            1 => deriv2(f, 1, y, true),
            _ => deriv2(f, 0, zc, false),
        }
    };

    // ── mean fields ───────────────────────────────────────────────────────────
    let ubar = [load("u")?, load("v")?, load("w")?];
    let pbar = load("p")?;
    // dataset names for the stored mean gradients / pressure moments
    let grad_name  = |a: usize, k: usize| format!("{}{}", VEL[a], DIR[k]);
    let pgrad_name = |a: usize, k: usize| format!("p{}{}", VEL[a], DIR[k]);
    let pvel_name  = |a: usize| format!("p{}", VEL[a]);

    // fluctuation second moment ⟨u_a'u_b'⟩ = ⟨u_a u_b⟩ − ū_a ū_b
    macro_rules! rey {
        ($a:expr, $b:expr) => {{
            let m = load(&mom2_name($a, $b))?;
            &m - &(&ubar[$a] * &ubar[$b])
        }};
    }

    let rij = rey!(i, j);
    let zeros = ArrayD::<f64>::zeros(rij.raw_dim());

    // ── production: P = −Σ_k [⟨u_i'u_k'⟩ ∂ū_j/∂x_k + ⟨u_j'u_k'⟩ ∂ū_i/∂x_k] ────
    let mut prod = zeros.clone();
    for k in 0..3 {
        let rik = rey!(i, k);
        prod = &prod - &(&rik * &load(&grad_name(j, k))?);
        let rjk = rey!(j, k);
        prod = &prod - &(&rjk * &load(&grad_name(i, k))?);
    }

    // ── dissipation: ε = 2ν Σ_k [⟨∂u_i/∂x_k ∂u_j/∂x_k⟩ − ∂ū_i/∂x_k ∂ū_j/∂x_k] ─
    let (lo, hi) = (i.min(j), i.max(j));
    let mut diss = zeros.clone();
    for k in 0..3 {
        let name = format!("{}{}{}{}", VEL[lo], DIR[k], VEL[hi], DIR[k]);
        let total = load(&name)?;
        let gg = &load(&grad_name(lo, k))? * &load(&grad_name(hi, k))?;
        diss = &diss + &(&total - &gg);
    }
    let diss = diss * (2.0 * nu);

    // ── pressure strain: Φ = ⟨p ∂u_i/∂x_j⟩ − p̄ ∂ū_i/∂x_j + (i ↔ j) ───────────
    let phi = &(&load(&pgrad_name(i, j))? - &(&pbar * &load(&grad_name(i, j))?))
        + &(&load(&pgrad_name(j, i))? - &(&pbar * &load(&grad_name(j, i))?));

    // ── pressure transport: Π = −∂⟨p'u_j'⟩/∂x_i − ∂⟨p'u_i'⟩/∂x_j ─────────────
    let mut ptr = zeros.clone();
    if i != 1 || with_y {
        let puj = &load(&pvel_name(j))? - &(&pbar * &ubar[j]);
        ptr = &ptr - &d1(&puj, i)?;
    }
    if j != 1 || with_y {
        let pui = &load(&pvel_name(i))? - &(&pbar * &ubar[i]);
        ptr = &ptr - &d1(&pui, j)?;
    }

    // ── turbulent transport: T = −Σ_k ∂⟨u_i'u_j'u_k'⟩/∂x_k ───────────────────
    // ⟨abc⟩' = ⟨abc⟩ − ā⟨bc⟩ − b̄⟨ac⟩ − c̄⟨ab⟩ + 2āb̄c̄  (from total moments)
    let mut turb = zeros.clone();
    for k in 0..3 {
        if k == 1 && !with_y {
            continue;
        }
        let m3 = load(&mom3_name(i, j, k))?;
        let triple = &(&(&m3
            - &(&ubar[i] * &load(&mom2_name(j, k))?))
            - &(&ubar[j] * &load(&mom2_name(i, k))?))
            - &(&ubar[k] * &load(&mom2_name(i, j))?)
            + &(&(&ubar[i] * &ubar[j]) * &ubar[k]) * 2.0;
        turb = &turb - &d1(&triple, k)?;
    }

    // ── viscous transport: D = ν Σ_k ∂²⟨u_i'u_j'⟩/∂x_k² ──────────────────────
    let mut visc = zeros.clone();
    for k in 0..3 {
        if k == 1 && !with_y {
            continue;
        }
        visc = &visc + &d2(&rij, k)?;
    }
    let visc = visc * nu;

    // ── mean convection: C = Σ_k ū_k ∂⟨u_i'u_j'⟩/∂x_k ────────────────────────
    let mut conv = zeros;
    for k in 0..3 {
        if k == 1 && !with_y {
            continue;
        }
        conv = &conv + &(&ubar[k] * &d1(&rij, k)?);
    }

    // ── spanwise average every term → (nz, nx) ────────────────────────────────
    let avg = |a: &ArrayD<f64>| avg_axis(a, 1);
    Ok(BudgetTerms {
        stress:       avg(&rij)?,
        prod:         avg(&prod)?,
        turb_trans:   avg(&turb)?,
        visc_trans:   avg(&visc)?,
        press_strain: avg(&phi)?,
        press_trans:  avg(&ptr)?,
        visc_diss:    avg(&diss)?,
        conv:         avg(&conv)?,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};
    use std::f64::consts::PI;

    /// Steady flow whose only "fluctuation" is a spanwise-periodic wave:
    ///
    ///   u(z,y) = U0(z) + A·sin(2πy/L),   v = B·sin(2πy/L),   w = 0
    ///
    /// With the span+time-mean convention: ū = U0, v̄ = w̄ = 0, and over a
    /// full period ⟨sin²⟩ = ½, so
    ///   ⟨u'u'⟩ = A²/2,  ⟨v'v'⟩ = B²/2,  ⟨w'w'⟩ = 0,
    ///   ⟨u'v'⟩ = A·B/2 (same phase),   k = (A² + B²)/4.
    #[test]
    fn spanwise_wave_variances_and_tke() {
        let (nz, ny, nx) = (6, 64, 5);
        let (a, b) = (0.3_f64, 0.1_f64);
        let u0: Array1<f64> = Array1::linspace(0.5, 1.5, nz);

        let s = |j: usize| (2.0 * PI * j as f64 / ny as f64).sin();
        let u = Array3::from_shape_fn((nz, ny, nx), |(iz, j, _)| u0[iz] + a * s(j)).into_dyn();
        let v = Array3::from_shape_fn((nz, ny, nx), |(_, j, _)| b * s(j)).into_dyn();
        let w = ArrayD::<f64>::zeros(ndarray::IxDyn(&[nz, ny, nx]));

        // total moments of the steady field are the pointwise products
        let uu = &u * &u;
        let vv = &v * &v;
        let ww = &w * &w;
        let uv = &u * &v;
        let uw = &u * &w;
        let vw = &v * &w;

        let tf = tke_fields(&u, &v, &w, &uu, &vv, &ww, &uv, &uw, &vw)
            .expect("tke_fields failed");

        let expect = [
            ("uu", &tf.uu, a * a / 2.0),
            ("vv", &tf.vv, b * b / 2.0),
            ("ww", &tf.ww, 0.0),
            ("uv", &tf.uv, a * b / 2.0),
            ("uw", &tf.uw, 0.0),
            ("vw", &tf.vw, 0.0),
            ("tke", &tf.tke, (a * a + b * b) / 4.0),
        ];
        for (name, field, want) in expect {
            let max_err = field.iter().map(|x| (x - want).abs()).fold(0.0_f64, f64::max);
            assert!(
                max_err < 1e-12,
                "{name}: max |value − {want}| = {max_err:.2e}"
            );
        }
    }

    /// Shape mismatch must be rejected.
    #[test]
    fn rejects_mismatched_shapes() {
        let a = ArrayD::<f64>::zeros(ndarray::IxDyn(&[4, 3, 5]));
        let bad = ArrayD::<f64>::zeros(ndarray::IxDyn(&[4, 3, 6]));
        assert!(tke_fields(&a, &a, &a, &a, &a, &a, &a, &a, &bad).is_err());
    }

    /// Dataset-name rules must match the DNS writer (Python `_sortName`).
    #[test]
    fn moment_name_conventions() {
        assert_eq!(mom2_name(2, 0), "uw");
        assert_eq!(mom2_name(1, 1), "vv");
        assert_eq!(mom3_name(0, 0, 1), "uuv");
        assert_eq!(mom3_name(1, 1, 0), "vvu");
        assert_eq!(mom3_name(2, 2, 1), "wwv");
        assert_eq!(mom3_name(2, 0, 1), "uvw");
        assert_eq!(mom3_name(2, 2, 2), "www");
    }

    /// Homogeneous shear layer with constant stresses:
    ///
    ///   ū = S·z,  v̄ = w̄ = 0,  ⟨u_a'u_b'⟩ = R_ab = const,
    ///   ⟨∂u_a/∂x_k ∂u_b/∂x_k⟩ − ∂ū_a/∂x_k ∂ū_b/∂x_k = E_ab/(2ν) per k... set
    ///   fluctuating-gradient variance G per component, all transport terms 0.
    ///
    /// Expected: prod_uu = −2·R_uw·S, prod_uw = −R_ww·S, ε_ab = 2ν·3·G_ab,
    /// turb/visc/pressure/conv all zero (constant fields, zero stored p-moments).
    #[test]
    fn homogeneous_shear_budget() {
        let (nz, ny, nx) = (8, 4, 6);
        let s_shear = 2.5_f64;
        let nu = 1e-3_f64;
        let r_uw = -1.2e-3_f64;
        let r_ww = 8.0e-4_f64;
        let g_uu = 5.0e-2_f64; // fluctuating-gradient variance per direction

        let zc: Array1<f64> = Array1::linspace(0.1, 1.9, nz);
        let x:  Array1<f64> = Array1::linspace(0.0, 5.0, nx);
        let y:  Array1<f64> = Array1::linspace(0.0, 0.3, ny);

        let full = |v: f64| ArrayD::from_elem(ndarray::IxDyn(&[nz, ny, nx]), v);
        let ubar = Array3::from_shape_fn((nz, ny, nx), |(iz, _, _)| s_shear * zc[iz]).into_dyn();

        let mut load = |name: &str| -> Result<ArrayD<f64>, Error> {
            Ok(match name {
                "u" => ubar.clone(),
                "v" | "w" | "p" => full(0.0),
                // total ⟨uu⟩ = ū² + R_uu (take R_uu = 0 for simplicity)
                "uu" => &ubar * &ubar,
                "uw" => &ubar * &full(0.0) + &full(r_uw),
                "ww" => full(r_ww),
                "uv" | "vv" | "vw" => full(0.0),
                // stored mean gradients: only ∂ū/∂z = S
                "uz" => full(s_shear),
                n if n.len() == 2 && !n.starts_with('p') => full(0.0), // ux, uy, vx…wz
                // gradient products: ⟨u_k u_k⟩ per direction; uzuz includes ū_z²
                "uxux" | "uyuy" => full(g_uu),
                "uzuz" => full(g_uu + s_shear * s_shear),
                n if n.len() == 4 => full(0.0),  // all other gradient products
                // pressure moments all zero
                n if n.starts_with('p') => full(0.0),
                // TOTAL triple moments consistent with zero FLUCTUATION
                // triples: ⟨abc⟩ = ā⟨b'c'⟩ + b̄⟨a'c'⟩ + c̄⟨a'b'⟩ + āb̄c̄
                "uuu" => &(&ubar * &ubar) * &ubar,       // ū³ (R_uu = 0)
                "uuw" => &ubar * (2.0 * r_uw),           // 2ū·R_uw
                "wwu" => &ubar * r_ww,                   // ū·R_ww
                n if n.len() == 3 => full(0.0),          // all other triples
                other => return Err(format!("unexpected dataset '{other}'").into()),
            })
        };

        // uu budget: prod = −2·R_uw·S, diss = 2ν·3G, everything else 0
        let b = stress_budget(&mut load, 0, 0, &x, &y, &zc, nu, false)
            .expect("uu budget failed");
        let expect_prod = -2.0 * r_uw * s_shear;
        let expect_diss = 2.0 * nu * 3.0 * g_uu;
        let check = |name: &str, field: &ArrayD<f64>, want: f64| {
            // interior only: one-sided boundary stencils act on exact
            // constants/linears here, so the whole field should match
            let max_err = field.iter().map(|v| (v - want).abs()).fold(0.0_f64, f64::max);
            assert!(max_err < 1e-10, "{name}: max |value − {want}| = {max_err:.2e}");
        };
        check("prod", &b.prod, expect_prod);
        check("visc_diss", &b.visc_diss, expect_diss);
        check("turb_trans", &b.turb_trans, 0.0);
        check("visc_trans", &b.visc_trans, 0.0);
        check("press_strain", &b.press_strain, 0.0);
        check("press_trans", &b.press_trans, 0.0);
        check("conv", &b.conv, 0.0);
        check("stress", &b.stress, 0.0); // R_uu = 0 by construction
        check("balance", &b.balance(), expect_prod - expect_diss);

        // uw budget: prod = −R_ww·S (only ⟨w'w'⟩∂ū/∂z survives)
        let b = stress_budget(&mut load, 0, 2, &x, &y, &zc, nu, false)
            .expect("uw budget failed");
        check("uw prod", &b.prod, -r_ww * s_shear);
        check("uw stress", &b.stress, r_uw);
        check("uw diss", &b.visc_diss, 0.0);
    }
}
