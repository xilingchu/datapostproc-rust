/// FIK skin-friction decomposition for spatially developing channel flow.
///
/// Reference: Appendix A of "Streamwise Steady Blowing Control of Channel Flow" (Xi Lingchu, 2025).
///
/// Coordinate convention (matches the rest of this crate):
///   Array shape (nz, ny, nx): axis 0 = wall-normal z, axis 1 = spanwise y, axis 2 = streamwise x.
///   Velocity: u = streamwise, v = spanwise, w = wall-normal.
///
/// FIK paper uses x=streamwise, y=wall-normal, z=spanwise.  Mapping:
///   paper y  → code z (axis 0)   integration variable η = z/h ∈ [0, 1]
///   paper z  → code y (axis 1)   spanwise (periodic → all ∂/∂y terms vanish after averaging)
///   paper u'v' (paper wall-normal shear) → code uw (u × wall-normal velocity)
///
/// C_f decomposition (spanwise-homogeneous case, spanwise terms are zero),
/// following Fukagata, Iwamoto & Kasagi (Phys. Fluids 14, L73, 2002), Eq. (11),
/// generalized to a HALF channel that need not be symmetric about z = h
/// (one-wall blowing control breaks the up-down symmetry):
///
///   C_f = C_f^L + C_f^A + C_f^T + C_f^C + C_f^D + C_f^S
///
///   C_f^L   = 6 ũ / Re_b              ũ(x) = (1/h)∫₀ʰ ū dz  (local bulk velocity)
///   C_f^A   = -(1/Re_b) ∂ū/∂z|_{z=h}  (centreline shear; zero for symmetric flow)
///   C_f^T   = C_turb_x + C_turb_y
///             C_turb_x = -3 ∫(1-η)² [∂(u'u')/∂x]″ dη
///             C_turb_y =  6 ∫(1-η)  (-u'w') dη + u'w'|_{z=h}
///   C_f^C   = C_conv_x + C_conv_y
///             C_conv_x = -3 ∫(1-η)² [∂(ū²)/∂x]″ dη
///             C_conv_y =  6 ∫(1-η)  (-ū·w̄) dη + (ū·w̄)|_{z=h}
///   C_f^D   = +(3/Re_b) ∫(1-η)² [∂²ū/∂x²]″ dη
///   C_f^S   = -3 ∫(1-η)² [∂p̄/∂x]″ dη
///
/// where f″(z, x) = f(z, x) − (1/h)∫₀ʰ f dz  is the deviation from the local
/// bulk mean (FIK 2002, Eq. 9).  C_turb_y and C_conv_y are the exact
/// integration-by-parts of  -3∫(1-η)²[∂(u'w')/∂z]″dη  and
/// -3∫(1-η)²[∂(ū·w̄)/∂z]″dη  using only the wall condition ū = u' = 0; the
/// centreline boundary values u'w'(h) and (ū·w̄)(h) survive when control makes
/// the flow asymmetric.
///
/// IMPORTANT — two easy mistakes (both were present in earlier revisions):
///   1. The inhomogeneous terms enter with an overall MINUS sign (FIK Eq. 11:
///      −12∫(1−y)²(I_x″ + ∂p″/∂x)dy in the paper's normalization).
///   2. They must use the double-prime deviations f″, not the raw quantities.
///      FIK first substitutes the integral force balance (their Eq. 5,
///      −∂p̃/∂x = C_f/8 + Ĩ_x) into the momentum equation, so the bulk part of
///      the pressure gradient — which IS the friction — has already been moved
///      to the left-hand side.  Using the full ∂p̄/∂x double-counts it.

use ndarray::{s, Array1, ArrayD, Axis};
use hdf5::Error;

use super::avg::avg_axis;
use super::ns::{deriv1, deriv2};

// ─── Public result type ────────────────────────────────────────────────────────

/// Skin-friction coefficient decomposed into physical contributions.
///
/// All arrays have length `nx` and are indexed by the streamwise position `x`.
pub struct FikDecomposition {
    /// Streamwise cell-centre positions.
    pub x: Array1<f64>,
    /// Laminar contribution: 6·ũ(x)/Re_b with ũ the local bulk velocity.
    pub cf_laminar: Array1<f64>,
    /// Centreline-shear asymmetry term: -(1/Re_b) ∂ū/∂z|_{z=h}.
    /// Vanishes for a symmetric channel; active when one-wall control
    /// breaks the up-down symmetry.
    pub cf_center: Array1<f64>,
    /// Turbulent streamwise term: -3 ∫(1-η)² [∂(u'u')/∂x]″ dη
    pub cf_turb_x: Array1<f64>,
    /// Turbulent wall-normal term: 6 ∫(1-η)(-u'w') dη + u'w'|_{z=h}
    pub cf_turb_y: Array1<f64>,
    /// Turbulent spanwise term (per-plane mode): -3 ∫(1-η)² [∂(u'v')/∂y]″ dη.
    /// Identically zero in spanwise-averaged mode (periodic direction).
    pub cf_turb_z: Array1<f64>,
    /// Mean-convection streamwise term: -3 ∫(1-η)² [∂(ū²)/∂x]″ dη
    pub cf_conv_x: Array1<f64>,
    /// Mean-convection wall-normal term: 6 ∫(1-η)(-ū·w̄) dη + (ū·w̄)|_{z=h}
    pub cf_conv_y: Array1<f64>,
    /// Mean-convection spanwise term (per-plane mode): -3 ∫(1-η)² [∂(ū·v̄)/∂y]″ dη.
    /// Identically zero in spanwise-averaged mode (periodic direction).
    pub cf_conv_z: Array1<f64>,
    /// Streamwise viscous diffusion term: +(3/Re_b) ∫(1-η)² [∂²ū/∂x²]″ dη.
    pub cf_diff_x: Array1<f64>,
    /// Spanwise viscous diffusion term (per-plane mode):
    /// +(3/Re_b) ∫(1-η)² [∂²ū/∂y²]″ dη.
    /// Identically zero in spanwise-averaged mode (periodic direction).
    pub cf_diff_z: Array1<f64>,
    /// Pressure-source term: -3 ∫(1-η)² [∂p̄/∂x]″ dη
    pub cf_source: Array1<f64>,
}

impl FikDecomposition {
    /// Sum of all turbulent contributions.
    pub fn cf_turbulent(&self) -> Array1<f64> {
        &(&self.cf_turb_x + &self.cf_turb_y) + &self.cf_turb_z
    }
    /// Sum of all convective contributions.
    pub fn cf_convection(&self) -> Array1<f64> {
        &(&self.cf_conv_x + &self.cf_conv_y) + &self.cf_conv_z
    }
    /// Sum of all viscous-diffusion contributions.
    pub fn cf_diffusion(&self) -> Array1<f64> {
        &self.cf_diff_x + &self.cf_diff_z
    }
    /// Total C_f reconstructed from all terms.
    pub fn cf_total(&self) -> Array1<f64> {
        &self.cf_laminar
            + &self.cf_center
            + &self.cf_turbulent()
            + &self.cf_convection()
            + &self.cf_diffusion()
            + &self.cf_source
    }
}

// ─── Main function ─────────────────────────────────────────────────────────────

/// Compute the FIK skin-friction decomposition for a spatially developing channel flow.
///
/// # Arguments
/// * `u`, `v`, `w`  – mean velocity components, shape `(nz, ny, nx)`.
/// * `p`            – mean pressure, same shape.
/// * `uu`, `uw`     – **total** second moments ⟨u·u⟩ and ⟨u·w⟩, same shape.
///                    Reynolds stresses are extracted internally:
///                      u'u' = ⟨uu⟩ − ū²,   u'w' = ⟨uw⟩ − ū·w̄.
/// * `x`            – streamwise cell-centre coordinates, length `nx`.
/// * `zc`           – wall-normal cell-centre coordinates, length `nz`, spanning `[0, 2h]`.
/// * `re_b`         – bulk Reynolds number U_b·h/ν (with U_b = 1 in the DNS normalization).
/// * `h`            – half-channel height.  Integration runs from `zc[0]` to `h`.
/// * `periodic_x`   – treat streamwise direction as periodic for derivative stencils.
#[allow(clippy::too_many_arguments)]
pub fn fik_decomposition(
    u:  &ArrayD<f64>,
    _v: &ArrayD<f64>,
    w:  &ArrayD<f64>,
    p:  &ArrayD<f64>,
    uu: &ArrayD<f64>,
    uw: &ArrayD<f64>,
    x:  &Array1<f64>,
    zc: &Array1<f64>,
    re_b: f64,
    h: f64,
    periodic_x: bool,
) -> Result<FikDecomposition, Error> {
    let nx = x.len();
    let nz = zc.len();

    for (name, arr) in [("u", u as &ArrayD<f64>), ("w", w), ("p", p), ("uu", uu), ("uw", uw)] {
        let s = arr.shape();
        if s.len() != 3 || s[0] != nz || s[2] != nx {
            return Err(format!(
                "'{name}' shape {s:?} incompatible with nz={nz} nx={nx}"
            ).into());
        }
    }

    // ── Step 1: average over spanwise y (axis 1) → shape (nz, nx) ────────────
    let avg = |a: &ArrayD<f64>| avg_axis(a, 1);

    let u2  = avg(u)?;
    let w2  = avg(w)?;
    let p2  = avg(p)?;
    let uu2 = avg(uu)?;
    let uw2 = avg(uw)?;

    // ── Step 2: Reynolds stresses (subtract mean products) ────────────────────
    // u'u' = ⟨uu⟩ - ū²
    // u'w' = ⟨uw⟩ - ū·w̄
    let uu_prime = &uu2 - &(&u2 * &u2);
    let uw_prime = &uw2 - &(&u2 * &w2);

    // ── Step 3: mean products needed for convection ───────────────────────────
    let uu_mean = &u2 * &u2;   // ū²
    let uw_mean = &u2 * &w2;   // ū·w̄

    fik_half_channel(
        &u2, &uu_prime, &uw_prime, &uu_mean, &uw_mean, &p2,
        None,
        x, zc, re_b, h, periodic_x,
    )
}

/// Per-plane FIK decomposition: no spanwise averaging, one decomposition per
/// spanwise plane `y[j]`.
///
/// The identity holds plane-by-plane, but the spanwise flux terms no longer
/// vanish: `cf_turb_z` and `cf_conv_z` carry the local spanwise momentum
/// exchange, and the spanwise viscous diffusion +(3/Re_b)∫(1-η)²[∂²ū/∂y²]″dη
/// is reported in `cf_diff_z`.  Reynolds stresses are defined about the pure
/// time mean of each plane (u'v' = ⟨uv⟩ − ū·v̄, etc.), so spanwise-dispersive
/// contributions stay in the mean-convection terms.
///
/// Averaging the returned decompositions over any subset of planes (see
/// [`fik_average`]) is exact by linearity; only an average over the FULL
/// spanwise period makes the spanwise terms cancel.
///
/// # Extra arguments (vs. `fik_decomposition`)
/// * `uv` – **total** second moment ⟨u·v⟩ (u × spanwise velocity).
/// * `y`  – spanwise coordinates, length `ny`, assumed uniform and periodic.
#[allow(clippy::too_many_arguments)]
pub fn fik_decomposition_planes(
    u:  &ArrayD<f64>,
    v:  &ArrayD<f64>,
    w:  &ArrayD<f64>,
    p:  &ArrayD<f64>,
    uu: &ArrayD<f64>,
    uv: &ArrayD<f64>,
    uw: &ArrayD<f64>,
    x:  &Array1<f64>,
    y:  &Array1<f64>,
    zc: &Array1<f64>,
    re_b: f64,
    h: f64,
    periodic_x: bool,
) -> Result<Vec<FikDecomposition>, Error> {
    let nx = x.len();
    let ny = y.len();
    let nz = zc.len();

    for (name, arr) in [("u", u as &ArrayD<f64>), ("v", v), ("w", w), ("p", p),
                        ("uu", uu), ("uv", uv), ("uw", uw)] {
        let s = arr.shape();
        if s.len() != 3 || s[0] != nz || s[1] != ny || s[2] != nx {
            return Err(format!(
                "'{name}' shape {s:?} incompatible with nz={nz} ny={ny} nx={nx}"
            ).into());
        }
    }

    // ── Reynolds stresses & mean products on the full 3-D fields ─────────────
    let uu_m = u * u;
    let uv_m = u * v;
    let uw_m = u * w;
    let uu_p = uu - &uu_m;
    let uv_p = uv - &uv_m;
    let uw_p = uw - &uw_m;

    // ── spanwise derivatives (periodic direction, axis 1) ────────────────────
    let duv_p_dy = deriv1(&uv_p, 1, y, true)?;   // ∂(u'v')/∂y
    let duv_m_dy = deriv1(&uv_m, 1, y, true)?;   // ∂(ū·v̄)/∂y
    let d2u_dy2  = deriv2(u,     1, y, true)?;   // ∂²ū/∂y²

    // ── one decomposition per spanwise plane ─────────────────────────────────
    let plane = |a: &ArrayD<f64>, j: usize| a.index_axis(Axis(1), j).to_owned();

    let mut out = Vec::with_capacity(ny);
    for j in 0..ny {
        let u2j   = plane(u, j);
        let uu_pj = plane(&uu_p, j);
        let uw_pj = plane(&uw_p, j);
        let uu_mj = plane(&uu_m, j);
        let uw_mj = plane(&uw_m, j);
        let p2j   = plane(p, j);

        let duv_pj = plane(&duv_p_dy, j);
        let duv_mj = plane(&duv_m_dy, j);
        let d2u_yj = plane(&d2u_dy2, j);
        let span = SpanwiseTerms {
            duv_p_dy: &duv_pj,
            duv_m_dy: &duv_mj,
            d2u_dy2:  &d2u_yj,
        };

        out.push(fik_half_channel(
            &u2j, &uu_pj, &uw_pj, &uu_mj, &uw_mj, &p2j,
            Some(&span),
            x, zc, re_b, h, periodic_x,
        )?);
    }
    Ok(out)
}

/// Arithmetic average of several decompositions (e.g. a subset of spanwise
/// planes).  Exact by linearity of the FIK identity.
pub fn fik_average(decs: &[&FikDecomposition]) -> Result<FikDecomposition, Error> {
    if decs.is_empty() {
        return Err("fik_average: empty input".into());
    }
    let n = decs.len() as f64;
    let avg = |get: fn(&FikDecomposition) -> &Array1<f64>| -> Array1<f64> {
        let mut s = get(decs[0]).clone();
        for d in &decs[1..] {
            s = s + get(d);
        }
        s / n
    };
    Ok(FikDecomposition {
        x:          decs[0].x.clone(),
        cf_laminar: avg(|d| &d.cf_laminar),
        cf_center:  avg(|d| &d.cf_center),
        cf_turb_x:  avg(|d| &d.cf_turb_x),
        cf_turb_y:  avg(|d| &d.cf_turb_y),
        cf_turb_z:  avg(|d| &d.cf_turb_z),
        cf_conv_x:  avg(|d| &d.cf_conv_x),
        cf_conv_y:  avg(|d| &d.cf_conv_y),
        cf_conv_z:  avg(|d| &d.cf_conv_z),
        cf_diff_x:  avg(|d| &d.cf_diff_x),
        cf_diff_z:  avg(|d| &d.cf_diff_z),
        cf_source:  avg(|d| &d.cf_source),
    })
}

// ─── Core: half-channel triple integration of one (nz, nx) plane/average ─────

/// Spanwise-derivative integrands (already differentiated along y), shape (nz, nx).
struct SpanwiseTerms<'a> {
    duv_p_dy: &'a ArrayD<f64>,  // ∂(u'v')/∂y
    duv_m_dy: &'a ArrayD<f64>,  // ∂(ū·v̄)/∂y
    d2u_dy2:  &'a ArrayD<f64>,  // ∂²ū/∂y²
}

/// FIK triple integration for one 2-D (nz, nx) field set: either the
/// spanwise-averaged fields (`span = None`) or a single spanwise plane
/// (`span = Some(..)`).
#[allow(clippy::too_many_arguments)]
fn fik_half_channel(
    u2:       &ArrayD<f64>,   // ū
    uu_prime: &ArrayD<f64>,   // u'u'
    uw_prime: &ArrayD<f64>,   // u'w'
    uu_mean:  &ArrayD<f64>,   // ū²
    uw_mean:  &ArrayD<f64>,   // ū·w̄
    p2:       &ArrayD<f64>,   // p̄
    span: Option<&SpanwiseTerms>,
    x:  &Array1<f64>,
    zc: &Array1<f64>,
    re_b: f64,
    h: f64,
    periodic_x: bool,
) -> Result<FikDecomposition, Error> {
    let nx = x.len();
    let nz = zc.len();

    // ── Step 4: restrict to the bottom half  zc ≤ h ───────────────────────────
    // The half-channel grid is extended with one row linearly interpolated to
    // z = h exactly, so that centreline boundary values (u'w'|_h, (ū·w̄)|_h,
    // ∂ū/∂z|_h) are available even when control makes the flow asymmetric.
    let n_half = zc.iter().position(|&z| z > h).unwrap_or(nz);
    if n_half < 2 {
        return Err(format!(
            "fewer than 2 wall-normal points below h={h}; check zc and h"
        ).into());
    }
    if n_half == nz {
        return Err(format!(
            "no wall-normal point above h={h}; cannot evaluate centreline values"
        ).into());
    }
    let (i_lo, i_hi) = (n_half - 1, n_half);
    let t = (h - zc[i_lo]) / (zc[i_hi] - zc[i_lo]);

    let mut zc_h = Array1::<f64>::zeros(n_half + 1);
    zc_h.slice_mut(s![..n_half]).assign(&zc.slice(s![..n_half]));
    zc_h[n_half] = h;

    let crop = |a: &ArrayD<f64>| crop_extend(a, n_half, t);

    let u_h      = crop(&u2);
    let uu_p_h   = crop(&uu_prime);
    let uw_p_h   = crop(&uw_prime);
    let uu_m_h   = crop(&uu_mean);
    let uw_m_h   = crop(&uw_mean);
    let p_h      = crop(&p2);

    // centreline shear ∂ū/∂z|_{z=h}: difference across z = h
    let mut uz_h = Array1::<f64>::zeros(nx);
    for ix in 0..nx {
        uz_h[ix] = (u2[[i_hi, ix]] - u2[[i_lo, ix]]) / (zc[i_hi] - zc[i_lo]);
    }

    // ── Step 5: FIK integration weights ──────────────────────────────────────
    // η = z / h,  weight1 = (1-η),  weight2 = (1-η)²  (both vanish at z = h)
    let w1: Array1<f64> = zc_h.mapv(|z| 1.0 - z / h);
    let w2: Array1<f64> = zc_h.mapv(|z| (1.0 - z / h).powi(2));

    // ── Step 6: streamwise derivatives ────────────────────────────────────────
    // Axis 1 of the 2D field = streamwise x.  No wall-normal derivatives are
    // needed: the ∂/∂z terms are handled analytically by integration by parts.
    let dx  = |a: &ArrayD<f64>| deriv1(a, 1, x, periodic_x);
    let d2x = |a: &ArrayD<f64>| deriv2(a, 1, x, periodic_x);

    let d_uu_p_dx = dx(&uu_p_h)?;   // ∂(u'u')/∂x
    let d_uu_m_dx = dx(&uu_m_h)?;   // ∂(ū²)/∂x
    let d2u_dx2   = d2x(&u_h)?;     // ∂²ū/∂x²
    let dp_dx     = dx(&p_h)?;      // ∂p̄/∂x

    // ── Step 7: integrate over wall-normal with FIK weights ──────────────────
    // f″ = f − (1/h)∫₀ʰ f dz  (deviation from local bulk mean, FIK 2002 Eq. 9)
    let dev = |g: &ArrayD<f64>| subtract_bulk(g, &bulk_mean(g, &zc_h, h, false));

    let int2 = |a: &ArrayD<f64>| fik_integrate(a, &w2, &zc_h); // weight (1-η)²
    let int1 = |a: &ArrayD<f64>| fik_integrate(a, &w1, &zc_h); // weight (1-η)

    // centreline boundary values: last row of the extended arrays (z = h)
    let row_h = |a: &ArrayD<f64>| -> Array1<f64> {
        Array1::from_iter((0..nx).map(|ix| a[[n_half, ix]]))
    };
    let uw_p_at_h = row_h(&uw_p_h);
    let uw_m_at_h = row_h(&uw_m_h);

    // local bulk velocity ũ(x) = (1/h)∫₀ʰ ū dz  (ū = 0 at the wall)
    let u_tilde = bulk_mean(&u_h, &zc_h, h, true);

    // ── spanwise terms (per-plane mode only) ─────────────────────────────────
    // Over a full spanwise period these vanish identically; on a single plane
    // they carry the local spanwise momentum exchange.
    let (cf_turb_z, cf_conv_z, cf_diff_span) = match span {
        Some(sp) => {
            let duv_p_h = crop(sp.duv_p_dy);
            let duv_m_h = crop(sp.duv_m_dy);
            let d2u_y_h = crop(sp.d2u_dy2);
            (
                int2(&dev(&duv_p_h))? * (-3.0),
                int2(&dev(&duv_m_h))? * (-3.0),
                int2(&dev(&d2u_y_h))? * (3.0 / re_b),
            )
        }
        None => (
            Array1::zeros(nx),
            Array1::zeros(nx),
            Array1::zeros(nx),
        ),
    };

    Ok(FikDecomposition {
        x:          x.clone(),
        cf_laminar: &u_tilde * (6.0 / re_b),
        cf_center:  &uz_h * (-1.0 / re_b),
        cf_turb_x:  int2(&dev(&d_uu_p_dx))? * (-3.0),
        cf_turb_y:  int1(&(-&uw_p_h))? * 6.0 + &uw_p_at_h,
        cf_turb_z,
        cf_conv_x:  int2(&dev(&d_uu_m_dx))? * (-3.0),
        cf_conv_y:  int1(&(-&uw_m_h))? * 6.0 + &uw_m_at_h,
        cf_conv_z,
        cf_diff_x:  int2(&dev(&d2u_dx2))? * (3.0 / re_b),
        cf_diff_z:  cf_diff_span,
        cf_source:  int2(&dev(&dp_dx))? * (-3.0),
    })
}

// ─── Helpers: half-channel cropping and bulk means ───────────────────────────

/// Crop a 2-D `(nz, nx)` array to its first `n_half` rows and append one row
/// linearly interpolated to z = h (weight `t` between original rows
/// `n_half-1` and `n_half`).
fn crop_extend(a: &ArrayD<f64>, n_half: usize, t: f64) -> ArrayD<f64> {
    let nx = a.shape()[1];
    let mut out = ArrayD::<f64>::zeros(ndarray::IxDyn(&[n_half + 1, nx]));
    for iz in 0..n_half {
        for ix in 0..nx {
            out[[iz, ix]] = a[[iz, ix]];
        }
    }
    for ix in 0..nx {
        out[[n_half, ix]] = (1.0 - t) * a[[n_half - 1, ix]] + t * a[[n_half, ix]];
    }
    out
}

/// Local bulk mean (1/h)∫₀ʰ f dz of a 2-D `(nz, nx)` field: trapezoid over the
/// grid plus the wall strip `[0, z[0]]`.  With `wall_zero = true` the field is
/// assumed to vanish at the wall (no-slip velocity → triangular strip);
/// otherwise it is extrapolated as constant.
fn bulk_mean(f: &ArrayD<f64>, z: &Array1<f64>, h: f64, wall_zero: bool) -> Array1<f64> {
    let (nz, nx) = (f.shape()[0], f.shape()[1]);
    let wall_fac = if wall_zero { 0.5 } else { 1.0 };
    let mut out = Array1::<f64>::zeros(nx);
    for ix in 0..nx {
        let mut s = wall_fac * z[0] * f[[0, ix]];
        for iz in 1..nz {
            s += 0.5 * (f[[iz - 1, ix]] + f[[iz, ix]]) * (z[iz] - z[iz - 1]);
        }
        out[ix] = s / h;
    }
    out
}

/// f″ = f − f̃ : subtract the per-x bulk mean from a 2-D `(nz, nx)` field.
fn subtract_bulk(f: &ArrayD<f64>, fm: &Array1<f64>) -> ArrayD<f64> {
    let mut out = f.clone();
    let (nz, nx) = (f.shape()[0], f.shape()[1]);
    for iz in 0..nz {
        for ix in 0..nx {
            out[[iz, ix]] -= fm[ix];
        }
    }
    out
}

// ─── Helper: trapezoidal integration ─────────────────────────────────────────

/// Integrate `f(z, x) * weight(z)` over the wall-normal direction z using the
/// trapezoidal rule.
///
/// `f`      – shape (nz, nx) as `ArrayD<f64>`
/// `weight` – shape (nz,)
/// `z`      – wall-normal cell-centre positions, shape (nz,)
///
/// Returns an `Array1<f64>` of length nx.
fn fik_integrate(
    f:      &ArrayD<f64>,
    weight: &Array1<f64>,
    z:      &Array1<f64>,
) -> Result<Array1<f64>, Error> {
    let sh = f.shape();
    if sh.len() != 2 {
        return Err("fik_integrate: expected 2-D array (nz, nx)".into());
    }
    let (nz, nx) = (sh[0], sh[1]);
    if weight.len() != nz || z.len() != nz {
        return Err(format!(
            "fik_integrate: weight/z length {} != nz {}",
            weight.len(), nz
        ).into());
    }

    // g[iz, ix] = f[iz, ix] * weight[iz]
    let mut result = Array1::<f64>::zeros(nx);
    for ix in 0..nx {
        let mut s = 0.0_f64;
        for iz in 1..nz {
            let dz  = z[iz] - z[iz - 1];
            let g0  = f[[iz - 1, ix]] * weight[iz - 1];
            let g1  = f[[iz,     ix]] * weight[iz];
            s += 0.5 * (g0 + g1) * dz;
        }
        result[ix] = s;
    }
    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    // Build a uniform 3-D field (nz, ny, nx) from a 1-D z-profile.
    fn broadcast_z(profile: &Array1<f64>, ny: usize, nx: usize) -> ArrayD<f64> {
        let nz = profile.len();
        Array3::from_shape_fn((nz, ny, nx), |(iz, _, _)| profile[iz]).into_dyn()
    }

    fn linspace(a: f64, b: f64, n: usize) -> Array1<f64> {
        Array1::linspace(a, b, n)
    }

    /// Laminar Poiseuille flow in the bottom half (z ∈ [0, h]).
    ///
    /// u(z) = (3 U_b / 2)(2z/h − (z/h)²), U_b = 1, h = 1.
    /// du/dz|_0 = 3, so  C_f = (2/Re_b)*3 = 6/Re_b = C_f^L.
    /// All other FIK terms should vanish.
    #[test]
    fn laminar_poiseuille_cf_equals_laminar_term() {
        let (nz, ny, nx) = (50, 4, 20);
        let h = 1.0_f64;
        let re_b = 2800.0_f64;

        // zc covers full channel [0, 2h]; use a symmetric grid
        let zc = linspace(0.02, 1.98, nz); // cell centres away from walls
        let x  = linspace(0.0, 4.0 * std::f64::consts::PI, nx);
        let y  = linspace(0.0, 2.0 * std::f64::consts::PI, ny);

        // Poiseuille profile (bottom half only; reflected for full channel)
        let u_prof: Array1<f64> = zc.mapv(|z| {
            let zz = if z <= h { z } else { 2.0 * h - z }; // mirror
            1.5 * (2.0 * zz / h - (zz / h).powi(2))
        });
        let zero_prof = Array1::zeros(nz);

        let u  = broadcast_z(&u_prof,    ny, nx);
        let v  = broadcast_z(&zero_prof, ny, nx);
        let w  = broadcast_z(&zero_prof, ny, nx);
        let p  = broadcast_z(&zero_prof, ny, nx);
        // For laminar flow: ⟨uu⟩ = ū², ⟨uw⟩ = 0
        let uu = broadcast_z(&u_prof.mapv(|u| u * u), ny, nx);
        let uw = broadcast_z(&zero_prof, ny, nx);

        let fik = fik_decomposition(
            &u, &v, &w, &p, &uu, &uw,
            &x, &zc, re_b, h,
            false,
        ).expect("FIK failed");

        let cf_L = 6.0 / re_b;

        // Every x-position should give C_f ≈ C_f^L
        let max_total_err = fik.cf_total().iter()
            .map(|&v| (v - cf_L).abs())
            .fold(0.0_f64, f64::max);

        // Non-laminar terms should be near zero
        let max_turb = fik.cf_turbulent().iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        let max_conv = fik.cf_convection().iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        println!("Laminar test: cf_L={cf_L:.4e}  max|total-cf_L|={max_total_err:.2e}  max|turb|={max_turb:.2e}  max|conv|={max_conv:.2e}");

        assert!(max_total_err < 1e-3 * cf_L,
            "total C_f deviates from laminar: {max_total_err:.2e}");
        assert!(max_turb < 1e-10, "turbulent term non-zero for laminar flow: {max_turb:.2e}");
        assert!(max_conv < 1e-10, "convective term non-zero for uniform flow: {max_conv:.2e}");
    }

    /// Periodic turbulent-like channel: only C_f^L and C_f^T_y should survive.
    ///
    /// Construct a synthetic flow with:
    ///   - u(z) = Poiseuille profile (x-uniform)  → all x-derivatives = 0
    ///   - w = 0  (no wall-normal mean)
    ///   - u'w' = −A·(1−η)   (linearly decaying shear stress)
    ///   - u'u' = 0
    ///
    /// Expected:
    ///   C_f^T_y = 6 ∫_0^1 (1−η)(−u'w') dη = 6A ∫_0^1 (1−η)² dη = 6A/3 = 2A
    #[test]
    fn periodic_channel_only_turb_y_and_laminar() {
        let (nz_half, ny, nx) = (400, 4, 16);
        let nz = 2 * nz_half;
        let h  = 1.0_f64;
        let re_b = 2800.0_f64;
        let a  = 1.5e-3_f64; // amplitude of synthetic u'w'

        // Cell centres in [0, 2h]
        let zc: Array1<f64> = Array1::from_iter(
            (0..nz).map(|i| h * (i as f64 + 0.5) / nz_half as f64)
        );
        let x = linspace(0.0, 4.0 * std::f64::consts::PI, nx);
        let y = linspace(0.0, 2.0 * std::f64::consts::PI, ny);

        // u: Poiseuille profile (x-uniform)
        let u_prof: Array1<f64> = zc.mapv(|z| {
            let eta = (z / h).min(2.0 - z / h); // symmetric
            1.5 * (2.0 * eta - eta.powi(2))
        });
        // u'w' = −A·(1−η) for bottom half, mirrored for top
        let uw_prime_prof: Array1<f64> = zc.mapv(|z| {
            let eta = if z <= h { z / h } else { 2.0 - z / h };
            -a * (1.0 - eta)
        });

        let u  = broadcast_z(&u_prof,    ny, nx);
        let v  = broadcast_z(&Array1::zeros(nz), ny, nx);
        let w  = broadcast_z(&Array1::zeros(nz), ny, nx);
        let p  = broadcast_z(&Array1::zeros(nz), ny, nx);
        // ⟨uu⟩ = ū² + u'u' = ū² (u'u' = 0 here)
        let uu = broadcast_z(&u_prof.mapv(|u| u * u), ny, nx);
        // ⟨uw⟩ = ū·w̄ + u'w' = 0 + u'w' = u'w'  (since w̄ = 0)
        let uw = broadcast_z(&uw_prime_prof, ny, nx);

        let fik = fik_decomposition(
            &u, &v, &w, &p, &uu, &uw,
            &x, &zc, re_b, h,
            true, // periodic x
        ).expect("FIK failed");

        // Expected turbulent term: 2A
        let cf_turb_y_expected = 2.0 * a;

        let cf_turb_y_mean: f64 = fik.cf_turb_y.iter().sum::<f64>() / nx as f64;
        let cf_turb_x_max = fik.cf_turb_x.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let cf_conv_max   = fik.cf_convection().iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let cf_diff_max   = fik.cf_diffusion().iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        println!("Periodic test: cf_turb_y_mean={cf_turb_y_mean:.4e}  expected={cf_turb_y_expected:.4e}");
        println!("  max|cf_turb_x|={cf_turb_x_max:.2e}  max|cf_conv|={cf_conv_max:.2e}  max|cf_diff|={cf_diff_max:.2e}");

        let rel_err = (cf_turb_y_mean - cf_turb_y_expected).abs() / cf_turb_y_expected;
        assert!(rel_err < 5e-3, "cf_turb_y error: {rel_err:.2e}");
        assert!(cf_turb_x_max < 1e-10, "cf_turb_x should be zero for x-uniform flow: {cf_turb_x_max:.2e}");
        assert!(cf_conv_max   < 1e-10, "cf_conv should be zero: {cf_conv_max:.2e}");
        assert!(cf_diff_max   < 1e-6  * (6.0 / re_b), "cf_diff too large: {cf_diff_max:.2e}");
    }

    /// Spanwise-uniform flow: every per-plane decomposition must reproduce the
    /// spanwise-averaged one exactly, with zero spanwise terms; and averaging
    /// the planes (fik_average) must be the identity.
    #[test]
    fn per_plane_matches_averaged_for_spanwise_uniform() {
        let (nz_half, ny, nx) = (200, 4, 16);
        let nz = 2 * nz_half;
        let h  = 1.0_f64;
        let re_b = 2800.0_f64;
        let a  = 1.5e-3_f64;

        let zc: Array1<f64> = Array1::from_iter(
            (0..nz).map(|i| h * (i as f64 + 0.5) / nz_half as f64)
        );
        let x = linspace(0.0, 4.0 * std::f64::consts::PI, nx);
        let y = linspace(0.0, 2.0 * std::f64::consts::PI, ny);

        let u_prof: Array1<f64> = zc.mapv(|z| {
            let eta = (z / h).min(2.0 - z / h);
            1.5 * (2.0 * eta - eta.powi(2))
        });
        let uw_prime_prof: Array1<f64> = zc.mapv(|z| {
            let eta = if z <= h { z / h } else { 2.0 - z / h };
            -a * (1.0 - eta)
        });

        let u  = broadcast_z(&u_prof, ny, nx);
        let v  = broadcast_z(&Array1::zeros(nz), ny, nx);
        let w  = broadcast_z(&Array1::zeros(nz), ny, nx);
        let p  = broadcast_z(&Array1::zeros(nz), ny, nx);
        let uu = broadcast_z(&u_prof.mapv(|u| u * u), ny, nx);
        let uv = broadcast_z(&Array1::zeros(nz), ny, nx); // ⟨uv⟩ = ū·v̄ + u'v' = 0
        let uw = broadcast_z(&uw_prime_prof, ny, nx);

        let avg = fik_decomposition(&u, &v, &w, &p, &uu, &uw, &x, &zc, re_b, h, true)
            .expect("averaged FIK failed");
        let decs = fik_decomposition_planes(&u, &v, &w, &p, &uu, &uv, &uw,
                                            &x, &y, &zc, re_b, h, true)
            .expect("per-plane FIK failed");
        assert_eq!(decs.len(), ny);

        let avg_total = avg.cf_total();
        for (j, d) in decs.iter().enumerate() {
            let max_span = d.cf_turb_z.iter().chain(d.cf_conv_z.iter())
                .map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(max_span < 1e-12, "plane {j}: spanwise terms non-zero: {max_span:.2e}");

            let max_diff = d.cf_total().iter().zip(avg_total.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            assert!(max_diff < 1e-12, "plane {j} deviates from averaged: {max_diff:.2e}");
        }

        // subset average (planes 1 and 3) must equal the averaged result too
        let sel = [&decs[1], &decs[3]];
        let sub = fik_average(&sel).expect("fik_average failed");
        let max_diff = sub.cf_total().iter().zip(avg_total.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(max_diff < 1e-12, "subset average deviates: {max_diff:.2e}");
    }
}
