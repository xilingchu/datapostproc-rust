/// Renard–Deck (RD) skin-friction decomposition for spatially developing
/// channel flow.
///
/// Reference: Renard & Deck, "A theoretical decomposition of mean skin
/// friction generation into physical phenomena across the boundary layer",
/// J. Fluid Mech. 790 (2016) 339–367.
///
/// Coordinate convention (matches math/fik.rs):
///   Array shape (nz, ny, nx): axis 0 = wall-normal z, axis 1 = spanwise y,
///   axis 2 = streamwise x.  Velocity: u = streamwise, w = wall-normal.
///
/// Unlike FIK — a MOMENTUM identity obtained by triple integration with the
/// (1-η) weights — RD is an ENERGY identity: multiply the mean streamwise
/// momentum equation by (ū − U_ref) and integrate over the half channel
/// z ∈ [0, h].  The viscous wall term ν(ū−U_ref)∂ū/∂z|₀ = −U_ref·τ_w is the
/// power expended by friction in the frame translating at U_ref, giving
///
///   τ_w U_ref = ν ∫₀ʰ (∂ū/∂z)² dz                            (a) direct dissipation
///             + ∫₀ʰ (−u'w') ∂ū/∂z dz                          (b) TKE production
///             + ∫₀ʰ (ū−U_ref)[ ū ∂ū/∂x + w̄ ∂ū/∂z + ∂p̄/∂x
///                              + ∂(u'u')/∂x − ν ∂²ū/∂x² ] dz  (c) spatial growth
///             + (ū(h)−U_ref)[ u'w'(h) − ν ∂ū/∂z|ₕ ]           boundary (asymmetry)
///
/// and C_f = 2 τ_w / U_ref².  Interpretation: friction power goes to (a) direct
/// viscous dissipation of the mean flow, (b) production of turbulent kinetic
/// energy, and (c) streamwise growth of mean-flow kinetic energy plus pressure
/// work and streamwise transport.  The boundary term survives only when the
/// flow is not symmetric about z = h (e.g. one-wall blowing control).
///
/// For a fully developed symmetric channel with U_ref equal to the bulk
/// velocity, (c) and the boundary term vanish identically and the identity
/// reduces to the standard two-term channel RD:
///   C_f = 2ν∫(∂ū/∂z)² dz + 2∫(−u'w')∂ū/∂z dz     (U_b = 1 normalization).
///
/// Derivation notes (all boundary terms accounted for):
///   * wall (z=0): ū = 0 and u' = 0 (no-slip in u even with wall
///     transpiration), so u'w'(0) = 0 and ū w̄(0) = 0; only the viscous term
///     −U_ref τ_w survives.
///   * The pressure gradient enters RAW (no bulk-balance substitution as in
///     FIK).  With U_ref = bulk velocity, ∫(ū−U_ref)dz ≈ 0 makes the pressure
///     term nearly self-cancelling for a developed flow.
///
/// Numerical notes:
///   * RD needs ∂ū/∂z explicitly (FIK does not).  It is evaluated with the
///     second-order non-uniform stencils of `math/ns.rs` on the FULL grid,
///     then cropped to [0, h], so the value at z = h uses a central stencil.
///   * If the grid does not include the wall (zc[0] > 0), a wall row is
///     prepended with ū = u'u' = u'w' = 0 (no-slip) and w̄, p̄ copied from the
///     first row, so the integrals cover the whole half channel.

use ndarray::{Array1, ArrayD, IxDyn};
use hdf5::Error;

use super::avg::avg_axis;
use super::fik::{crop_extend, fik_integrate, half_grid};
use super::ns::{deriv1, deriv2};

// ─── Public result type ────────────────────────────────────────────────────────

/// Skin-friction coefficient decomposed into energy-budget contributions.
///
/// All arrays have length `nx` and are indexed by the streamwise position `x`.
pub struct RdDecomposition {
    /// Streamwise cell-centre positions.
    pub x: Array1<f64>,
    /// Direct viscous dissipation of the mean flow: (2/U³) ν ∫(∂ū/∂z)² dz.
    pub cf_diss: Array1<f64>,
    /// Turbulent-kinetic-energy production: (2/U³) ∫(−u'w')(∂ū/∂z) dz.
    pub cf_prod: Array1<f64>,
    /// Streamwise mean-KE growth: (2/U³) ∫(ū−U)(ū ∂ū/∂x) dz.
    pub cf_conv_x: Array1<f64>,
    /// Wall-normal mean-KE transport: (2/U³) ∫(ū−U)(w̄ ∂ū/∂z) dz.
    pub cf_conv_y: Array1<f64>,
    /// Streamwise turbulent transport: (2/U³) ∫(ū−U) ∂(u'u')/∂x dz.
    pub cf_turb_x: Array1<f64>,
    /// Streamwise viscous diffusion: −(2/U³) ν ∫(ū−U) ∂²ū/∂x² dz.
    pub cf_diff_x: Array1<f64>,
    /// Pressure-work term: (2/U³) ∫(ū−U) ∂p̄/∂x dz.
    pub cf_source: Array1<f64>,
    /// Centreline boundary term: (2/U³)(ū(h)−U)[u'w'(h) − ν ∂ū/∂z|ₕ].
    /// Vanishes for a symmetric channel; active under one-wall control.
    pub cf_center: Array1<f64>,
}

impl RdDecomposition {
    /// Sum of all spatial-growth contributions (RD's C_f,c).
    pub fn cf_growth(&self) -> Array1<f64> {
        &(&(&self.cf_conv_x + &self.cf_conv_y) + &self.cf_turb_x)
            + &(&self.cf_diff_x + &self.cf_source)
    }
    /// Total C_f reconstructed from all terms.
    pub fn cf_total(&self) -> Array1<f64> {
        &(&self.cf_diss + &self.cf_prod) + &(&self.cf_growth() + &self.cf_center)
    }
}

// ─── Main function ─────────────────────────────────────────────────────────────

/// Compute the RD skin-friction decomposition for a spatially developing
/// channel flow (spanwise-averaged).
///
/// # Arguments
/// * `u`, `w`      – mean streamwise / wall-normal velocity, shape `(nz, ny, nx)`.
/// * `p`           – mean pressure, same shape.
/// * `uu`, `uw`    – **total** second moments ⟨u·u⟩ and ⟨u·w⟩, same shape.
///                   Reynolds stresses are extracted internally:
///                     u'u' = ⟨uu⟩ − ū²,   u'w' = ⟨uw⟩ − ū·w̄.
/// * `x`           – streamwise cell-centre coordinates, length `nx`.
/// * `zc`          – wall-normal coordinates, length `nz`, spanning `[0, 2h]`.
/// * `re_b`        – bulk Reynolds number U_b·h/ν (U_b = 1 in DNS units).
/// * `h`           – half-channel height; integration runs from the wall to h.
/// * `u_ref`       – reference velocity of the RD energy frame; C_f is
///                   normalized as 2τ_w/U_ref².  Use the global bulk velocity
///                   (1.0 in DNS units) to match the FIK/direct C_f.
/// * `periodic_x`  – treat streamwise direction as periodic for derivatives.
#[allow(clippy::too_many_arguments)]
pub fn rd_decomposition(
    u:  &ArrayD<f64>,
    w:  &ArrayD<f64>,
    p:  &ArrayD<f64>,
    uu: &ArrayD<f64>,
    uw: &ArrayD<f64>,
    x:  &Array1<f64>,
    zc: &Array1<f64>,
    re_b: f64,
    h: f64,
    u_ref: f64,
    periodic_x: bool,
) -> Result<RdDecomposition, Error> {
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
    if u_ref <= 0.0 {
        return Err(format!("u_ref must be positive, got {u_ref}").into());
    }
    let nu = 1.0 / re_b;

    // ── Step 1: average over spanwise y (axis 1) → shape (nz, nx) ────────────
    let avg = |a: &ArrayD<f64>| avg_axis(a, 1);
    let u2  = avg(u)?;
    let w2  = avg(w)?;
    let p2  = avg(p)?;
    let uu2 = avg(uu)?;
    let uw2 = avg(uw)?;

    // ── Step 2: Reynolds stresses (subtract mean products) ────────────────────
    let uu_prime = &uu2 - &(&u2 * &u2);
    let uw_prime = &uw2 - &(&u2 * &w2);

    // ── Step 3: ensure the grid starts at the wall ────────────────────────────
    // No-slip: ū = u'u' = u'w' = 0 at z = 0 (u' = 0 even with transpiration);
    // w̄ and p̄ are extrapolated as constant over the thin wall strip.
    let (zc_g, u_g, w_g, p_g, uu_p_g, uw_p_g) = if zc[0] > 1e-12 {
        let mut z = Array1::<f64>::zeros(nz + 1);
        z.slice_mut(ndarray::s![1..]).assign(zc);
        (
            z,
            prepend_row(&u2, false),
            prepend_row(&w2, true),
            prepend_row(&p2, true),
            prepend_row(&uu_prime, false),
            prepend_row(&uw_prime, false),
        )
    } else {
        (zc.clone(), u2, w2, p2, uu_prime, uw_prime)
    };

    // ── Step 4: wall-normal shear on the FULL grid, then crop to [0, h] ──────
    // Computing ∂ū/∂z before cropping keeps a central stencil at z = h.
    let dudz_g = deriv1(&u_g, 0, &zc_g, false)?;

    let (n_half, t, zc_h) = half_grid(&zc_g, h)?;
    let crop = |a: &ArrayD<f64>| crop_extend(a, n_half, t);

    let u_h    = crop(&u_g);
    let w_h    = crop(&w_g);
    let p_h    = crop(&p_g);
    let uu_p_h = crop(&uu_p_g);
    let uw_p_h = crop(&uw_p_g);
    let dudz_h = crop(&dudz_g);

    // ── Step 5: streamwise derivatives on the half-channel fields ────────────
    let dudx   = deriv1(&u_h, 1, x, periodic_x)?;
    let d2udx2 = deriv2(&u_h, 1, x, periodic_x)?;
    let dpdx   = deriv1(&p_h, 1, x, periodic_x)?;
    let duudx  = deriv1(&uu_p_h, 1, x, periodic_x)?;

    // ── Step 6: energy-weighted integrals ────────────────────────────────────
    let eps = u_h.mapv(|v| v - u_ref);              // ū − U_ref
    let ones = Array1::<f64>::from_elem(zc_h.len(), 1.0);
    let int = |a: &ArrayD<f64>| fik_integrate(a, &ones, &zc_h);
    let c = 2.0 / u_ref.powi(3);

    let cf_diss   = int(&(&dudz_h * &dudz_h))? * (c * nu);
    let cf_prod   = int(&(&uw_p_h.mapv(|v| -v) * &dudz_h))? * c;
    let cf_conv_x = int(&(&eps * &(&u_h * &dudx)))? * c;
    let cf_conv_y = int(&(&eps * &(&w_h * &dudz_h)))? * c;
    let cf_turb_x = int(&(&eps * &duudx))? * c;
    let cf_diff_x = int(&(&eps * &d2udx2))? * (-c * nu);
    let cf_source = int(&(&eps * &dpdx))? * c;

    // ── Step 7: centreline boundary term (last row = z = h) ──────────────────
    let last = n_half; // index of the appended row at z = h
    let mut cf_center = Array1::<f64>::zeros(nx);
    for ix in 0..nx {
        let eps_h = u_h[[last, ix]] - u_ref;
        cf_center[ix] = c * eps_h * (uw_p_h[[last, ix]] - nu * dudz_h[[last, ix]]);
    }

    Ok(RdDecomposition {
        x: x.clone(),
        cf_diss,
        cf_prod,
        cf_conv_x,
        cf_conv_y,
        cf_turb_x,
        cf_diff_x,
        cf_source,
        cf_center,
    })
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Prepend a wall row to a 2-D `(nz, nx)` array: zeros (no-slip quantities) or
/// a copy of the first row (constant extrapolation for w̄, p̄).
fn prepend_row(a: &ArrayD<f64>, copy_first: bool) -> ArrayD<f64> {
    let (nz, nx) = (a.shape()[0], a.shape()[1]);
    let mut out = ArrayD::<f64>::zeros(IxDyn(&[nz + 1, nx]));
    for ix in 0..nx {
        out[[0, ix]] = if copy_first { a[[0, ix]] } else { 0.0 };
    }
    for iz in 0..nz {
        for ix in 0..nx {
            out[[iz + 1, ix]] = a[[iz, ix]];
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn broadcast_z(profile: &Array1<f64>, ny: usize, nx: usize) -> ArrayD<f64> {
        let nz = profile.len();
        Array3::from_shape_fn((nz, ny, nx), |(iz, _, _)| profile[iz]).into_dyn()
    }

    fn linspace(a: f64, b: f64, n: usize) -> Array1<f64> {
        Array1::linspace(a, b, n)
    }

    /// Laminar Poiseuille flow: the whole C_f must come from direct
    /// dissipation, C_f^diss = 2ν∫(∂ū/∂z)²dz = 6/Re_b, all other terms ≈ 0.
    /// Grid includes the wall (zc[0] = 0) and the exact centreline point.
    #[test]
    fn laminar_poiseuille_all_in_dissipation() {
        let (nz, ny, nx) = (201, 4, 20);
        let h = 1.0_f64;
        let re_b = 2800.0_f64;

        let zc = linspace(0.0, 2.0, nz);
        let x  = linspace(0.0, 4.0 * std::f64::consts::PI, nx);

        // Poiseuille: u = 1.5(2z − z²) is symmetric about z = 1 as written.
        let u_prof: Array1<f64> = zc.mapv(|z| 1.5 * (2.0 * z - z * z));
        let zero = Array1::<f64>::zeros(nz);

        let u  = broadcast_z(&u_prof, ny, nx);
        let w  = broadcast_z(&zero, ny, nx);
        let p  = broadcast_z(&zero, ny, nx);
        let uu = broadcast_z(&u_prof.mapv(|v| v * v), ny, nx);
        let uw = broadcast_z(&zero, ny, nx);

        let rd = rd_decomposition(&u, &w, &p, &uu, &uw, &x, &zc, re_b, h, 1.0, false)
            .expect("rd_decomposition failed");

        let cf_l = 6.0 / re_b;
        let max_err = rd.cf_total().iter().map(|&v| (v - cf_l).abs()).fold(0.0_f64, f64::max);
        let max_prod = rd.cf_prod.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_growth = rd.cf_growth().iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let max_center = rd.cf_center.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        println!("Laminar RD: cf_L={cf_l:.4e}  max|total-cf_L|={max_err:.2e}  \
                  max|prod|={max_prod:.2e}  max|growth|={max_growth:.2e}  max|center|={max_center:.2e}");

        assert!(max_err < 1e-3 * cf_l, "total deviates from 6/Re_b: {max_err:.2e}");
        assert!(max_prod < 1e-12, "production non-zero for laminar flow");
        assert!(max_growth < 1e-12, "growth non-zero for x-uniform flow");
        assert!(max_center < 1e-12, "centre term non-zero for symmetric flow");
    }

    /// Synthetic turbulent-like channel with u'w' = −A(1−η):
    ///   C_f^prod = 2∫₀¹ A(1−z)·3(1−z) dz = 2A,
    ///   C_f^diss = 6/Re_b (parabolic profile), everything else ≈ 0.
    /// Same expected values as the FIK synthetic test — for a parabolic mean
    /// profile with a linear stress the two decompositions coincide.
    /// Cell-centred grid (zc[0] > 0) exercises the wall-row prepending.
    #[test]
    fn synthetic_turbulent_production_term() {
        let (nz_half, ny, nx) = (400, 4, 16);
        let nz = 2 * nz_half;
        let h  = 1.0_f64;
        let re_b = 2800.0_f64;
        let a  = 1.5e-3_f64;

        let zc: Array1<f64> = Array1::from_iter(
            (0..nz).map(|i| h * (i as f64 + 0.5) / nz_half as f64)
        );
        let x = linspace(0.0, 4.0 * std::f64::consts::PI, nx);

        let u_prof: Array1<f64> = zc.mapv(|z| 1.5 * (2.0 * z - z * z));
        let uw_prime_prof: Array1<f64> = zc.mapv(|z| {
            let eta = if z <= h { z / h } else { 2.0 - z / h };
            -a * (1.0 - eta)
        });

        let u  = broadcast_z(&u_prof, ny, nx);
        let w  = broadcast_z(&Array1::zeros(nz), ny, nx);
        let p  = broadcast_z(&Array1::zeros(nz), ny, nx);
        let uu = broadcast_z(&u_prof.mapv(|v| v * v), ny, nx);
        let uw = broadcast_z(&uw_prime_prof, ny, nx);

        let rd = rd_decomposition(&u, &w, &p, &uu, &uw, &x, &zc, re_b, h, 1.0, true)
            .expect("rd_decomposition failed");

        let mean = |arr: &Array1<f64>| arr.iter().sum::<f64>() / arr.len() as f64;
        let cf_prod_mean = mean(&rd.cf_prod);
        let cf_diss_mean = mean(&rd.cf_diss);
        let max_growth = rd.cf_growth().iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

        let prod_expected = 2.0 * a;
        let diss_expected = 6.0 / re_b;

        println!("Synthetic RD: prod={cf_prod_mean:.4e} (expect {prod_expected:.4e})  \
                  diss={cf_diss_mean:.4e} (expect {diss_expected:.4e})  max|growth|={max_growth:.2e}");

        assert!((cf_prod_mean - prod_expected).abs() / prod_expected < 5e-3,
            "cf_prod error: {:.2e}", (cf_prod_mean - prod_expected).abs() / prod_expected);
        assert!((cf_diss_mean - diss_expected).abs() / diss_expected < 1e-3,
            "cf_diss error: {:.2e}", (cf_diss_mean - diss_expected).abs() / diss_expected);
        assert!(max_growth < 1e-10, "growth terms should vanish: {max_growth:.2e}");
    }
}
