/// Vortex-identification criteria (Q and λ₂) for instantaneous 3-D fields.
///
/// Array convention (matches the rest of this crate): shape (nz, ny, nx)
///   axis 0 = z (wall-normal), axis 1 = y (spanwise), axis 2 = x (streamwise)
/// Velocity components: u = streamwise (x), v = spanwise (y), w = wall-normal (z).
///
/// With the velocity-gradient tensor g_ij = ∂u_i/∂x_j (i, j ∈ {x, y, z} and
/// u_x = u, u_y = v, u_z = w), split into the strain-rate and rotation tensors
///
///   S_ij = (g_ij + g_ji)/2,      Ω_ij = (g_ij − g_ji)/2,
///
/// the two criteria are:
///
/// * Q criterion (Hunt, Wray & Moin 1988):
///     Q = ½(‖Ω‖² − ‖S‖²),  vortex where Q > 0
///   (rotation dominates strain).
///
/// * λ₂ criterion (Jeong & Hussain 1995):
///     λ₂ = middle eigenvalue of the symmetric tensor S² + Ω²,
///   vortex where λ₂ < 0 (pressure sectional minimum in the plane
///   perpendicular to the vortex axis).
///
/// Derivatives use the second-order stencils of `math/ns.rs` (`deriv1`),
/// non-uniform-grid aware; the spanwise direction is normally periodic.
use hdf5::Error;
use ndarray::{Array1, ArrayD};

use super::ns::deriv1;

/// Q and λ₂ fields, same shape as the input velocity fields.
pub struct VortexFields {
    /// Q criterion: ½(‖Ω‖² − ‖S‖²); vortex cores where Q > 0.
    pub q: ArrayD<f64>,
    /// λ₂ criterion: middle eigenvalue of S² + Ω²; vortex cores where λ₂ < 0.
    pub lambda2: ArrayD<f64>,
}

/// Middle eigenvalue of a symmetric 3×3 matrix
/// [[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]]
/// via the trigonometric (analytic) method — no iteration, no allocation.
#[inline]
fn middle_eigenvalue_sym3(
    a11: f64,
    a22: f64,
    a33: f64,
    a12: f64,
    a13: f64,
    a23: f64,
) -> f64 {
    let p1 = a12 * a12 + a13 * a13 + a23 * a23;
    if p1 == 0.0 {
        // Diagonal matrix: middle of the three diagonal entries.
        let (mut lo, mut mid, mut hi) = (a11, a22, a33);
        if lo > mid {
            std::mem::swap(&mut lo, &mut mid);
        }
        if mid > hi {
            std::mem::swap(&mut mid, &mut hi);
        }
        if lo > mid {
            std::mem::swap(&mut lo, &mut mid);
        }
        return mid;
    }
    let q = (a11 + a22 + a33) / 3.0;
    let d11 = a11 - q;
    let d22 = a22 - q;
    let d33 = a33 - q;
    let p2 = d11 * d11 + d22 * d22 + d33 * d33 + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    // r = det(B)/2 with B = (A − qI)/p
    let det_b = d11 * (d22 * d33 - a23 * a23) - a12 * (a12 * d33 - a23 * a13)
        + a13 * (a12 * a23 - d22 * a13);
    let r = (det_b / (p * p * p) / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    // Roots are q + 2p·cos(φ + 2πk/3); with φ ∈ [0, π/3], k = 0 gives the
    // largest and k = 1 the smallest eigenvalue.
    let eig_hi = q + 2.0 * p * phi.cos();
    let eig_lo = q + 2.0 * p * (phi + 2.0 * std::f64::consts::FRAC_PI_3).cos();
    3.0 * q - eig_hi - eig_lo // trace identity → middle eigenvalue
}

/// Compute the Q and λ₂ vortex-identification fields.
///
/// # Arguments
/// * `u`, `v`, `w` – velocity components (streamwise, spanwise, wall-normal),
///   all shaped `(nz, ny, nx)`.
/// * `x`, `y`, `z` – coordinates along axes 2, 1, 0 (lengths nx, ny, nz).
/// * `periodic`    – boundary treatment per axis `[z, y, x]`;
///   DNS channel-flow convention: `[false, true, false]` for a spatially
///   developing simulation (`[false, true, true]` when streamwise-periodic).
pub fn vortex_criteria(
    u: &ArrayD<f64>,
    v: &ArrayD<f64>,
    w: &ArrayD<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
    periodic: [bool; 3],
) -> Result<VortexFields, Error> {
    let shape = u.shape().to_vec();
    if shape.len() != 3 {
        return Err(format!("expected 3-D fields, got shape {shape:?}").into());
    }
    if v.shape() != u.shape() || w.shape() != u.shape() {
        return Err("u, v, w must have the same shape".into());
    }
    let (nz, ny, nx) = (shape[0], shape[1], shape[2]);
    if z.len() != nz || y.len() != ny || x.len() != nx {
        return Err(format!(
            "coordinate lengths ({}, {}, {}) do not match field shape ({nz}, {ny}, {nx})",
            z.len(),
            y.len(),
            x.len()
        )
        .into());
    }

    // Velocity-gradient tensor, physical indexing g[i][j] = ∂u_i/∂x_j.
    // Axis mapping: x → array axis 2, y → axis 1, z → axis 0.
    let dux = deriv1(u, 2, x, periodic[2])?;
    let duy = deriv1(u, 1, y, periodic[1])?;
    let duz = deriv1(u, 0, z, periodic[0])?;
    let dvx = deriv1(v, 2, x, periodic[2])?;
    let dvy = deriv1(v, 1, y, periodic[1])?;
    let dvz = deriv1(v, 0, z, periodic[0])?;
    let dwx = deriv1(w, 2, x, periodic[2])?;
    let dwy = deriv1(w, 1, y, periodic[1])?;
    let dwz = deriv1(w, 0, z, periodic[0])?;

    let mut q_field = ArrayD::<f64>::zeros(u.raw_dim());
    let mut l2_field = ArrayD::<f64>::zeros(u.raw_dim());

    // All arrays are freshly allocated in standard layout → flat slices.
    let (gxx, gxy, gxz) = (
        dux.as_slice().unwrap(),
        duy.as_slice().unwrap(),
        duz.as_slice().unwrap(),
    );
    let (gyx, gyy, gyz) = (
        dvx.as_slice().unwrap(),
        dvy.as_slice().unwrap(),
        dvz.as_slice().unwrap(),
    );
    let (gzx, gzy, gzz) = (
        dwx.as_slice().unwrap(),
        dwy.as_slice().unwrap(),
        dwz.as_slice().unwrap(),
    );
    let qs = q_field.as_slice_mut().unwrap();
    let l2s = l2_field.as_slice_mut().unwrap();

    for i in 0..qs.len() {
        let (g11, g12, g13) = (gxx[i], gxy[i], gxz[i]);
        let (g21, g22, g23) = (gyx[i], gyy[i], gyz[i]);
        let (g31, g32, g33) = (gzx[i], gzy[i], gzz[i]);

        // Strain-rate tensor (symmetric) and rotation tensor (antisymmetric).
        let s11 = g11;
        let s22 = g22;
        let s33 = g33;
        let s12 = 0.5 * (g12 + g21);
        let s13 = 0.5 * (g13 + g31);
        let s23 = 0.5 * (g23 + g32);
        let o12 = 0.5 * (g12 - g21);
        let o13 = 0.5 * (g13 - g31);
        let o23 = 0.5 * (g23 - g32);

        let s_norm2 = s11 * s11
            + s22 * s22
            + s33 * s33
            + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23);
        let o_norm2 = 2.0 * (o12 * o12 + o13 * o13 + o23 * o23);
        qs[i] = 0.5 * (o_norm2 - s_norm2);

        // A = S² + Ω² (symmetric).  Ω is antisymmetric: Ω_ji = −Ω_ij.
        let a11 = s11 * s11 + s12 * s12 + s13 * s13 - o12 * o12 - o13 * o13;
        let a22 = s12 * s12 + s22 * s22 + s23 * s23 - o12 * o12 - o23 * o23;
        let a33 = s13 * s13 + s23 * s23 + s33 * s33 - o13 * o13 - o23 * o23;
        let a12 = s11 * s12 + s12 * s22 + s13 * s23 - o13 * o23;
        let a13 = s11 * s13 + s12 * s23 + s13 * s33 + o12 * o23;
        let a23 = s12 * s13 + s22 * s23 + s23 * s33 - o12 * o13;

        l2s[i] = middle_eigenvalue_sym3(a11, a22, a33, a12, a13, a23);
    }

    Ok(VortexFields {
        q: q_field,
        lambda2: l2_field,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn linspace(a: f64, b: f64, n: usize) -> Array1<f64> {
        Array1::linspace(a, b, n)
    }

    /// Known symmetric matrices with non-zero off-diagonal entries exercise
    /// the trigonometric branch of the analytic eigenvalue solver.
    #[test]
    fn middle_eigenvalue_known_matrices() {
        // [[0,1,1],[1,0,1],[1,1,0]] has eigenvalues (2, −1, −1) → middle −1.
        let m = middle_eigenvalue_sym3(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);
        assert!((m - (-1.0)).abs() < 1e-12, "middle = {m}, expected −1");

        // [[1,1,0],[1,1,0],[0,0,3]]: 2×2 block gives (0, 2), plus 3 → middle 2.
        let m = middle_eigenvalue_sym3(1.0, 1.0, 3.0, 1.0, 0.0, 0.0);
        assert!((m - 2.0).abs() < 1e-12, "middle = {m}, expected 2");

        // Diagonal fast path: diag(5, −3, 1) → middle 1.
        let m = middle_eigenvalue_sym3(5.0, -3.0, 1.0, 0.0, 0.0, 0.0);
        assert!((m - 1.0).abs() < 1e-12, "middle = {m}, expected 1");
    }

    /// Solid-body rotation about the z axis: u = −ω·y, v = ω·x, w = 0.
    /// Exact under second-order stencils (linear field):
    /// Q = ω², λ₂ = −ω² at every interior point.
    #[test]
    fn solid_body_rotation_is_a_vortex() {
        let (nz, ny, nx) = (8, 10, 12);
        let omega = 2.5_f64;
        let x = linspace(0.0, 1.0, nx);
        let y = linspace(0.0, 1.0, ny);
        let z = linspace(0.0, 1.0, nz);

        let u = Array3::from_shape_fn((nz, ny, nx), |(_, iy, _)| -omega * y[iy]).into_dyn();
        let v = Array3::from_shape_fn((nz, ny, nx), |(_, _, ix)| omega * x[ix]).into_dyn();
        let w = ArrayD::<f64>::zeros(u.raw_dim());

        let vf = vortex_criteria(&u, &v, &w, &x, &y, &z, [false, false, false])
            .expect("vortex_criteria failed");

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let q = vf.q[[iz, iy, ix]];
                    let l2 = vf.lambda2[[iz, iy, ix]];
                    // λ₂ tolerance is looser than Q: the trigonometric
                    // eigenvalue method loses a few digits for degenerate
                    // eigenvalue pairs (acos at its ±1 boundary).
                    assert!(
                        (q - omega * omega).abs() < 1e-10,
                        "Q = {q}, expected {}",
                        omega * omega
                    );
                    assert!(
                        (l2 + omega * omega).abs() < 1e-6 * omega * omega,
                        "λ₂ = {l2}, expected {}",
                        -omega * omega
                    );
                }
            }
        }
    }

    /// Pure irrotational plane strain: u = αx, v = −αy, w = 0.
    /// No rotation → Q = −α² < 0 and λ₂ = +α² > 0 (not a vortex).
    #[test]
    fn pure_strain_is_not_a_vortex() {
        let (nz, ny, nx) = (6, 9, 11);
        let alpha = 1.7_f64;
        let x = linspace(0.0, 1.0, nx);
        let y = linspace(0.0, 1.0, ny);
        let z = linspace(0.0, 1.0, nz);

        let u = Array3::from_shape_fn((nz, ny, nx), |(_, _, ix)| alpha * x[ix]).into_dyn();
        let v = Array3::from_shape_fn((nz, ny, nx), |(_, iy, _)| -alpha * y[iy]).into_dyn();
        let w = ArrayD::<f64>::zeros(u.raw_dim());

        let vf = vortex_criteria(&u, &v, &w, &x, &y, &z, [false, false, false])
            .expect("vortex_criteria failed");

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let q = vf.q[[iz, iy, ix]];
                    let l2 = vf.lambda2[[iz, iy, ix]];
                    assert!(
                        (q + alpha * alpha).abs() < 1e-10,
                        "Q = {q}, expected {}",
                        -alpha * alpha
                    );
                    // Degenerate eigenvalues → looser tolerance (see above).
                    assert!(
                        (l2 - alpha * alpha).abs() < 1e-6 * alpha * alpha,
                        "λ₂ = {l2}, expected {}",
                        alpha * alpha
                    );
                }
            }
        }
    }

    /// Simple shear u = γz: strain and rotation balance exactly, Q = 0.
    /// λ₂ for pure shear is also non-negative-definite as a vortex marker:
    /// eigenvalues of S²+Ω² are (0, −γ²/4·0, …) — analytically (γ²/4)·(0, 0, …).
    /// Here we only check Q = 0 and λ₂ = 0 (shear alone is not a vortex).
    #[test]
    fn simple_shear_is_marginal() {
        let (nz, ny, nx) = (12, 6, 7);
        let gamma = 3.0_f64;
        let x = linspace(0.0, 1.0, nx);
        let y = linspace(0.0, 1.0, ny);
        let z = linspace(0.0, 1.0, nz);

        let u = Array3::from_shape_fn((nz, ny, nx), |(iz, _, _)| gamma * z[iz]).into_dyn();
        let v = ArrayD::<f64>::zeros(u.raw_dim());
        let w = ArrayD::<f64>::zeros(u.raw_dim());

        let vf = vortex_criteria(&u, &v, &w, &x, &y, &z, [false, false, false])
            .expect("vortex_criteria failed");

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let q = vf.q[[iz, iy, ix]];
                    let l2 = vf.lambda2[[iz, iy, ix]];
                    assert!(q.abs() < 1e-10, "Q = {q}, expected 0 for simple shear");
                    assert!(l2.abs() < 1e-10, "λ₂ = {l2}, expected 0 for simple shear");
                }
            }
        }
    }
}
