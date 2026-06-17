/// Navier-Stokes momentum residual for incompressible flow.
///
/// Array convention throughout: shape (nz, ny, nx)
///   axis 0 = z (wall-normal), axis 1 = y (spanwise), axis 2 = x (streamwise)
///
/// The residual is defined as:
///   R_i = (u·∇)u_i + ∂p/∂x_i − ν ∇²u_i
///
/// For a steady or time-averaged field R should be near zero.
/// For an instantaneous snapshot R equals ∂u_i/∂t.
///
/// DNS channel flow convention for `periodic`:
///   axis 0 (z, wall-normal) → false
///   axis 1 (y, spanwise)    → true
///   axis 2 (x, streamwise)  → true

use ndarray::{Array1, ArrayD, Axis};
use hdf5::Error;

/// First derivative of `f` along `axis` on a (possibly non-uniform) grid.
///
/// All points use second-order 3-point stencils (requires n >= 3):
/// * Interior points   – central difference (Lagrange).
/// * `periodic = true` – boundary points wrap around (uniform spacing assumed).
/// * `periodic = false`– boundary points use a second-order one-sided stencil.
pub fn deriv1(
    f: &ArrayD<f64>,
    axis: usize,
    coords: &Array1<f64>,
    periodic: bool,
) -> Result<ArrayD<f64>, Error> {
    let n = f.shape()[axis];
    if coords.len() != n {
        return Err(format!(
            "coords length {} != array size {} along axis {}",
            coords.len(),
            n,
            axis
        )
        .into());
    }
    if n < 3 {
        return Err(format!("need at least 3 points along axis {}", axis).into());
    }

    let mut df = f.to_owned();
    for i in 0..n {
        let slice = if i == 0 {
            if periodic {
                // Wrap: left neighbour is f[n-1], right is f[1] (uniform spacing).
                let dl = coords[1] - coords[0];
                let dr = coords[1] - coords[0];
                let fm = f.index_axis(Axis(axis), n - 1);
                let fc = f.index_axis(Axis(axis), 0);
                let fp = f.index_axis(Axis(axis), 1);
                &fm * (-dr / (dl * (dl + dr)))
                    + &fc * ((dr - dl) / (dl * dr))
                    + &fp * (dl / (dr * (dl + dr)))
            } else {
                // Second-order forward stencil through i=0,1,2 (Lagrange).
                // f'(x0) = f0*(-(2h1+h2)/(h1*(h1+h2)))
                //        + f1*((h1+h2)/(h1*h2))
                //        + f2*(-h1/((h1+h2)*h2))
                let f0 = f.index_axis(Axis(axis), 0);
                let f1 = f.index_axis(Axis(axis), 1);
                let f2 = f.index_axis(Axis(axis), 2);
                let h1 = coords[1] - coords[0];
                let h2 = coords[2] - coords[1];
                &f0 * (-(2.0 * h1 + h2) / (h1 * (h1 + h2)))
                    + &f1 * ((h1 + h2) / (h1 * h2))
                    + &f2 * (-h1 / ((h1 + h2) * h2))
            }
        } else if i == n - 1 {
            if periodic {
                // Wrap: left neighbour is f[n-2], right is f[0] (uniform spacing).
                let dl = coords[n - 1] - coords[n - 2];
                let dr = coords[n - 1] - coords[n - 2];
                let fm = f.index_axis(Axis(axis), n - 2);
                let fc = f.index_axis(Axis(axis), n - 1);
                let fp = f.index_axis(Axis(axis), 0);
                &fm * (-dr / (dl * (dl + dr)))
                    + &fc * ((dr - dl) / (dl * dr))
                    + &fp * (dl / (dr * (dl + dr)))
            } else {
                // Second-order backward stencil through i=n-3,n-2,n-1 (Lagrange).
                // f'(x_{n-1}) = f_{n-3}*(h2/(h1*(h1+h2)))
                //             + f_{n-2}*(-(h1+h2)/(h1*h2))
                //             + f_{n-1}*((h1+2h2)/((h1+h2)*h2))
                let f0 = f.index_axis(Axis(axis), n - 3);
                let f1 = f.index_axis(Axis(axis), n - 2);
                let f2 = f.index_axis(Axis(axis), n - 1);
                let h1 = coords[n - 2] - coords[n - 3];
                let h2 = coords[n - 1] - coords[n - 2];
                &f0 * (h2 / (h1 * (h1 + h2)))
                    + &f1 * (-(h1 + h2) / (h1 * h2))
                    + &f2 * ((h1 + 2.0 * h2) / ((h1 + h2) * h2))
            }
        } else {
            let fm = f.index_axis(Axis(axis), i - 1);
            let fc = f.index_axis(Axis(axis), i);
            let fp = f.index_axis(Axis(axis), i + 1);
            let dl = coords[i] - coords[i - 1];
            let dr = coords[i + 1] - coords[i];
            // Second-order non-uniform central difference (Lagrange 3-point weights).
            &fm * (-dr / (dl * (dl + dr)))
                + &fc * ((dr - dl) / (dl * dr))
                + &fp * (dl / (dr * (dl + dr)))
        };
        df.index_axis_mut(Axis(axis), i).assign(&slice);
    }
    Ok(df)
}

/// Second derivative of `f` along `axis` on a (possibly non-uniform) grid.
///
/// * `periodic = false` – one-sided 3-point Lagrange stencil at the boundaries
///   (requires n >= 3).
/// * `periodic = true`  – wraps around at boundaries, uniform spacing assumed.
pub fn deriv2(
    f: &ArrayD<f64>,
    axis: usize,
    coords: &Array1<f64>,
    periodic: bool,
) -> Result<ArrayD<f64>, Error> {
    let n = f.shape()[axis];
    if coords.len() != n {
        return Err(format!(
            "coords length {} != array size {} along axis {}",
            coords.len(),
            n,
            axis
        )
        .into());
    }
    if n < 3 {
        return Err(format!(
            "need at least 3 points along axis {} for second derivative",
            axis
        )
        .into());
    }

    let mut d2f = f.to_owned();
    for i in 0..n {
        // Lagrange second-derivative weights for 3 points with spacings h1, h2:
        //   f''(x_k) = 2*f0/(h1*(h1+h2)) − 2*f1/(h1*h2) + 2*f2/(h2*(h1+h2))
        let slice = if i == 0 {
            if periodic {
                let h = coords[1] - coords[0]; // uniform → h1 = h2 = h
                let fm = f.index_axis(Axis(axis), n - 1);
                let fc = f.index_axis(Axis(axis), 0);
                let fp = f.index_axis(Axis(axis), 1);
                (&fm + &fp - &fc * 2.0) / (h * h)
            } else {
                let f0 = f.index_axis(Axis(axis), 0);
                let f1 = f.index_axis(Axis(axis), 1);
                let f2 = f.index_axis(Axis(axis), 2);
                let h1 = coords[1] - coords[0];
                let h2 = coords[2] - coords[1];
                &f0 * (2.0 / (h1 * (h1 + h2)))
                    + &f1 * (-2.0 / (h1 * h2))
                    + &f2 * (2.0 / (h2 * (h1 + h2)))
            }
        } else if i == n - 1 {
            if periodic {
                let h = coords[n - 1] - coords[n - 2];
                let fm = f.index_axis(Axis(axis), n - 2);
                let fc = f.index_axis(Axis(axis), n - 1);
                let fp = f.index_axis(Axis(axis), 0);
                (&fm + &fp - &fc * 2.0) / (h * h)
            } else {
                let f0 = f.index_axis(Axis(axis), n - 3);
                let f1 = f.index_axis(Axis(axis), n - 2);
                let f2 = f.index_axis(Axis(axis), n - 1);
                let h1 = coords[n - 2] - coords[n - 3];
                let h2 = coords[n - 1] - coords[n - 2];
                &f0 * (2.0 / (h1 * (h1 + h2)))
                    + &f1 * (-2.0 / (h1 * h2))
                    + &f2 * (2.0 / (h2 * (h1 + h2)))
            }
        } else {
            let fm = f.index_axis(Axis(axis), i - 1);
            let fc = f.index_axis(Axis(axis), i);
            let fp = f.index_axis(Axis(axis), i + 1);
            let h1 = coords[i] - coords[i - 1];
            let h2 = coords[i + 1] - coords[i];
            &fm * (2.0 / (h1 * (h1 + h2)))
                + &fc * (-2.0 / (h1 * h2))
                + &fp * (2.0 / (h2 * (h1 + h2)))
        };
        d2f.index_axis_mut(Axis(axis), i).assign(&slice);
    }
    Ok(d2f)
}

/// Continuity residual: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z.
///
/// `periodic[axis]` sets the boundary treatment per axis (0=z, 1=y, 2=x).
/// Typical DNS channel flow: `[false, true, true]`.
pub fn divergence(
    u: &ArrayD<f64>,
    v: &ArrayD<f64>,
    w: &ArrayD<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
    periodic: [bool; 3],
) -> Result<ArrayD<f64>, Error> {
    if v.shape() != u.shape() || w.shape() != u.shape() {
        return Err("u, v, w must have the same shape".into());
    }
    Ok(deriv1(u, 2, x, periodic[2])?
        + deriv1(v, 1, y, periodic[1])?
        + deriv1(w, 0, z, periodic[0])?)
}

/// Navier-Stokes momentum residual at every grid point for incompressible flow.
///
/// Computes the three momentum-equation residuals:
///   R_x = u ∂u/∂x + v ∂u/∂y + w ∂u/∂z + ∂p/∂x − ν ∇²u
///   R_y = u ∂v/∂x + v ∂v/∂y + w ∂v/∂z + ∂p/∂y − ν ∇²v
///   R_z = u ∂w/∂x + v ∂w/∂y + w ∂w/∂z + ∂p/∂z − ν ∇²w
///
/// # Arguments
/// * `u`, `v`, `w`  – velocity components (x, y, z); all shaped `(nz, ny, nx)`
/// * `p`            – pressure, same shape
/// * `x`, `y`, `z`  – coordinate arrays with lengths `nx`, `ny`, `nz`
/// * `nu`           – kinematic viscosity
/// * `periodic`     – `[z_periodic, y_periodic, x_periodic]` (axis 0/1/2).
///                    Typical channel DNS: `[false, true, true]`.
///
/// # Returns
/// `(res_x, res_y, res_z)` – residual arrays, same shape as input.
pub fn ns_momentum_residual(
    u: &ArrayD<f64>,
    v: &ArrayD<f64>,
    w: &ArrayD<f64>,
    p: &ArrayD<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
    nu: f64,
    periodic: [bool; 3],
) -> Result<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>), Error> {
    let shape = u.shape();
    if v.shape() != shape || w.shape() != shape || p.shape() != shape {
        return Err("u, v, w, p must all have the same shape".into());
    }

    let (pz, py, px) = (periodic[0], periodic[1], periodic[2]);

    // Momentum flux tensors (element-wise products), matching the Fortran staggered-grid
    // divergence form: conv = ∂(u_i u)/∂x + ∂(u_i v)/∂y + ∂(u_i w)/∂z
    // On the collocated post-processing grid this is equivalent to the Fortran's
    // 0.25*(ue*ue - uw*uw)*d1xp + ... stencil for cell-centred data.
    let uu = u * u;
    let uv = u * v;
    let uw_f = u * w;
    let vv = v * v;
    let vw_f = v * w;
    let ww = w * w;

    // Divergence-form convection (matches Fortran lineStep)
    let conv_u = deriv1(&uu,   2, x, px)? + deriv1(&uv,   1, y, py)? + deriv1(&uw_f, 0, z, pz)?;
    let conv_v = deriv1(&uv,   2, x, px)? + deriv1(&vv,   1, y, py)? + deriv1(&vw_f, 0, z, pz)?;
    let conv_w = deriv1(&uw_f, 2, x, px)? + deriv1(&vw_f, 1, y, py)? + deriv1(&ww,   0, z, pz)?;

    // Pressure gradients
    let dp_dx = deriv1(p, 2, x, px)?;
    let dp_dy = deriv1(p, 1, y, py)?;
    let dp_dz = deriv1(p, 0, z, pz)?;

    // Viscous term: ν ∇²u_i  (matches Fortran's (due*d2xp - duw*d2xm + ...)*nu)
    let visc_u = (deriv2(u, 2, x, px)? + deriv2(u, 1, y, py)? + deriv2(u, 0, z, pz)?) * nu;
    let visc_v = (deriv2(v, 2, x, px)? + deriv2(v, 1, y, py)? + deriv2(v, 0, z, pz)?) * nu;
    let visc_w = (deriv2(w, 2, x, px)? + deriv2(w, 1, y, py)? + deriv2(w, 0, z, pz)?) * nu;

    // R = ∂(u_i u_j)/∂x_j + ∂p/∂x_i − ν ∇²u_i
    // (Fortran sign: rsdu = visc - conv, then ∂u/∂t = rsdu - ∂p/∂x + headx)
    let res_x = conv_u + dp_dx - visc_u;
    let res_y = conv_v + dp_dy - visc_v;
    let res_z = conv_w + dp_dz - visc_w;

    Ok((res_x, res_y, res_z))
}

/// L2 norm of the N-S residual: `sqrt(mean(R_x² + R_y² + R_z²))`.
///
/// A single scalar quantifying how well momentum is conserved across the domain.
pub fn ns_residual_l2(
    res_x: &ArrayD<f64>,
    res_y: &ArrayD<f64>,
    res_z: &ArrayD<f64>,
) -> Result<f64, Error> {
    let n = res_x.len();
    if res_y.len() != n || res_z.len() != n {
        return Err("residual arrays must have the same number of elements".into());
    }
    let sum_sq: f64 = res_x
        .iter()
        .zip(res_y.iter())
        .zip(res_z.iter())
        .map(|((rx, ry), rz)| rx * rx + ry * ry + rz * rz)
        .sum();
    Ok((sum_sq / n as f64).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, IxDyn};

    fn linspace(start: f64, end: f64, n: usize) -> Array1<f64> {
        Array1::linspace(start, end, n)
    }

    // ── derivative accuracy ───────────────────────────────────────────────────

    #[test]
    fn deriv1_nonperiodic_sin() {
        let n = 200;
        let x = linspace(0.0, 2.0 * std::f64::consts::PI, n);
        let f = x.mapv(f64::sin).into_dyn();
        let df = deriv1(&f, 0, &x, false).unwrap();
        // All points (including boundaries) should be second-order accurate.
        let max_err = (0..n)
            .map(|i| (df[IxDyn(&[i])] - x[i].cos()).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max error (incl. boundaries) = {}", max_err);
    }

    #[test]
    fn deriv2_nonperiodic_sin() {
        let n = 100;
        let x = linspace(0.0, 2.0 * std::f64::consts::PI, n);
        let f = x.mapv(f64::sin).into_dyn();
        let d2f = deriv2(&f, 0, &x, false).unwrap();
        let max_err = (2..n - 2)
            .map(|i| (d2f[IxDyn(&[i])] + x[i].sin()).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max interior error = {}", max_err);
    }

    /// With periodic BC the boundary points should also reach O(dx²) accuracy.
    #[test]
    fn deriv1_periodic_recovers_boundary() {
        let n = 128;
        // [0, 2π) — exclude the endpoint so the grid is truly periodic
        let dx = 2.0 * std::f64::consts::PI / n as f64;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 * dx));
        let f = x.mapv(f64::sin).into_dyn();
        let df = deriv1(&f, 0, &x, true).unwrap();

        // All points (including boundaries) should recover cos(x)
        let max_err = (0..n)
            .map(|i| (df[IxDyn(&[i])] - x[i].cos()).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max error (periodic) = {:.2e}", max_err);
    }

    #[test]
    fn deriv2_periodic_recovers_boundary() {
        let n = 128;
        let dx = 2.0 * std::f64::consts::PI / n as f64;
        let x: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 * dx));
        let f = x.mapv(f64::sin).into_dyn();
        let d2f = deriv2(&f, 0, &x, true).unwrap();

        let max_err = (0..n)
            .map(|i| (d2f[IxDyn(&[i])] + x[i].sin()).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max error (periodic) = {:.2e}", max_err);
    }

    // ── N-S residual ─────────────────────────────────────────────────────────

    /// Couette flow: u(z) = z, v = w = p = 0, non-periodic in z.
    /// Exact steady solution → residual must be zero everywhere.
    #[test]
    fn ns_residual_couette_is_zero() {
        let (nz, ny, nx) = (20, 4, 4);
        let x = linspace(0.0, 1.0, nx);
        let y = linspace(0.0, 1.0, ny);
        let z = linspace(0.0, 1.0, nz);

        let u = Array3::from_shape_fn((nz, ny, nx), |(iz, _, _)| z[iz]).into_dyn();
        let zero = ArrayD::zeros(u.raw_dim());

        let (rx, ry, rz) = ns_momentum_residual(
            &u, &zero, &zero, &zero,
            &x, &y, &z, 1e-3,
            [false, false, false],
        ).unwrap();

        let l2 = ns_residual_l2(&rx, &ry, &rz).unwrap();
        assert!(l2 < 1e-10, "Couette residual L2 = {:.2e}", l2);
    }

    // ── continuity ────────────────────────────────────────────────────────────

    /// ∇·(x, y, −2z) = 1 + 1 − 2 = 0 everywhere.
    #[test]
    fn divergence_linear_field_is_zero() {
        let n = 10usize;
        let x = linspace(0.0, 1.0, n);
        let y = linspace(0.0, 1.0, n);
        let z = linspace(0.0, 1.0, n);

        let u = Array3::from_shape_fn((n, n, n), |(_, _, ix)| x[ix]).into_dyn();
        let v = Array3::from_shape_fn((n, n, n), |(_, iy, _)| y[iy]).into_dyn();
        let w = Array3::from_shape_fn((n, n, n), |(iz, _, _)| -2.0 * z[iz]).into_dyn();

        let div = divergence(&u, &v, &w, &x, &y, &z, [false, false, false]).unwrap();
        let max_err = div.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_err < 1e-10, "max divergence error = {:.2e}", max_err);
    }
}
