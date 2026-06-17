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

use ndarray::{Array1, ArrayD, Axis};
use hdf5::Error;

/// First derivative of `f` along `axis` on a non-uniform grid defined by `coords`.
///
/// Interior: second-order central difference (Lagrange 3-point).
/// Boundaries: first-order one-sided difference.
pub fn deriv1(f: &ArrayD<f64>, axis: usize, coords: &Array1<f64>) -> Result<ArrayD<f64>, Error> {
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
    if n < 2 {
        return Err(format!("need at least 2 points along axis {}", axis).into());
    }

    let mut df = f.to_owned();
    for i in 0..n {
        let slice = if i == 0 {
            let f0 = f.index_axis(Axis(axis), 0);
            let f1 = f.index_axis(Axis(axis), 1);
            let dx = coords[1] - coords[0];
            (&f1 - &f0) / dx
        } else if i == n - 1 {
            let fn1 = f.index_axis(Axis(axis), n - 1);
            let fn2 = f.index_axis(Axis(axis), n - 2);
            let dx = coords[n - 1] - coords[n - 2];
            (&fn1 - &fn2) / dx
        } else {
            let fm = f.index_axis(Axis(axis), i - 1);
            let fc = f.index_axis(Axis(axis), i);
            let fp = f.index_axis(Axis(axis), i + 1);
            let dl = coords[i] - coords[i - 1];
            let dr = coords[i + 1] - coords[i];
            // Second-order non-uniform central difference weights from Lagrange interpolation.
            &fm * (-dr / (dl * (dl + dr))) + &fc * ((dr - dl) / (dl * dr)) + &fp * (dl / (dr * (dl + dr)))
        };
        df.index_axis_mut(Axis(axis), i).assign(&slice);
    }
    Ok(df)
}

/// Second derivative of `f` along `axis` on a non-uniform grid defined by `coords`.
///
/// Interior: second-order 3-point Lagrange stencil.
/// Boundaries: one-sided 3-point stencil (requires n >= 3).
pub fn deriv2(f: &ArrayD<f64>, axis: usize, coords: &Array1<f64>) -> Result<ArrayD<f64>, Error> {
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
        return Err(format!("need at least 3 points along axis {} for second derivative", axis).into());
    }

    let mut d2f = f.to_owned();
    for i in 0..n {
        // Lagrange second-derivative weights for 3 points with spacings h1, h2:
        //   f''(x_k) = 2*f0/(h1*(h1+h2)) − 2*f1/(h1*h2) + 2*f2/(h2*(h1+h2))
        let slice = if i == 0 {
            let f0 = f.index_axis(Axis(axis), 0);
            let f1 = f.index_axis(Axis(axis), 1);
            let f2 = f.index_axis(Axis(axis), 2);
            let h1 = coords[1] - coords[0];
            let h2 = coords[2] - coords[1];
            &f0 * (2.0 / (h1 * (h1 + h2)))
                + &f1 * (-2.0 / (h1 * h2))
                + &f2 * (2.0 / (h2 * (h1 + h2)))
        } else if i == n - 1 {
            let f0 = f.index_axis(Axis(axis), n - 3);
            let f1 = f.index_axis(Axis(axis), n - 2);
            let f2 = f.index_axis(Axis(axis), n - 1);
            let h1 = coords[n - 2] - coords[n - 3];
            let h2 = coords[n - 1] - coords[n - 2];
            &f0 * (2.0 / (h1 * (h1 + h2)))
                + &f1 * (-2.0 / (h1 * h2))
                + &f2 * (2.0 / (h2 * (h1 + h2)))
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
/// Should be near zero everywhere for incompressible DNS data.
pub fn divergence(
    u: &ArrayD<f64>,
    v: &ArrayD<f64>,
    w: &ArrayD<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
) -> Result<ArrayD<f64>, Error> {
    if v.shape() != u.shape() || w.shape() != u.shape() {
        return Err("u, v, w must have the same shape".into());
    }
    Ok(deriv1(u, 2, x)? + deriv1(v, 1, y)? + deriv1(w, 0, z)?)
}

/// Navier-Stokes momentum residual at every grid point for incompressible flow.
///
/// Computes the three momentum-equation residuals:
///   R_x = u ∂u/∂x + v ∂u/∂y + w ∂u/∂z + ∂p/∂x − ν ∇²u
///   R_y = u ∂v/∂x + v ∂v/∂y + w ∂v/∂z + ∂p/∂y − ν ∇²v
///   R_z = u ∂w/∂x + v ∂w/∂y + w ∂w/∂z + ∂p/∂z − ν ∇²w
///
/// # Arguments
/// * `u`, `v`, `w` – velocity components (x, y, z); all shaped `(nz, ny, nx)`
/// * `p`           – pressure, same shape
/// * `x`, `y`, `z` – coordinate arrays with lengths `nx`, `ny`, `nz` respectively
/// * `nu`          – kinematic viscosity
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
) -> Result<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>), Error> {
    let shape = u.shape();
    if v.shape() != shape || w.shape() != shape || p.shape() != shape {
        return Err("u, v, w, p must all have the same shape".into());
    }

    // Velocity gradients (axis 0=z, 1=y, 2=x)
    let du_dx = deriv1(u, 2, x)?;
    let du_dy = deriv1(u, 1, y)?;
    let du_dz = deriv1(u, 0, z)?;

    let dv_dx = deriv1(v, 2, x)?;
    let dv_dy = deriv1(v, 1, y)?;
    let dv_dz = deriv1(v, 0, z)?;

    let dw_dx = deriv1(w, 2, x)?;
    let dw_dy = deriv1(w, 1, y)?;
    let dw_dz = deriv1(w, 0, z)?;

    // Pressure gradients
    let dp_dx = deriv1(p, 2, x)?;
    let dp_dy = deriv1(p, 1, y)?;
    let dp_dz = deriv1(p, 0, z)?;

    // Viscous term: ν ∇²u_i
    let visc_u = (deriv2(u, 2, x)? + deriv2(u, 1, y)? + deriv2(u, 0, z)?) * nu;
    let visc_v = (deriv2(v, 2, x)? + deriv2(v, 1, y)? + deriv2(v, 0, z)?) * nu;
    let visc_w = (deriv2(w, 2, x)? + deriv2(w, 1, y)? + deriv2(w, 0, z)?) * nu;

    // R = convection + pressure gradient − viscous diffusion
    let res_x = (u * &du_dx) + (v * &du_dy) + (w * &du_dz) + dp_dx - visc_u;
    let res_y = (u * &dv_dx) + (v * &dv_dy) + (w * &dv_dz) + dp_dy - visc_v;
    let res_z = (u * &dw_dx) + (v * &dw_dy) + (w * &dw_dz) + dp_dz - visc_w;

    Ok((res_x, res_y, res_z))
}

/// L2 norm of the N-S residual: `sqrt(mean(R_x² + R_y² + R_z²))`.
///
/// A single scalar that quantifies how well momentum is conserved across
/// the entire domain. Useful for a quick sanity check.
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

    #[test]
    fn deriv1_uniform_sin() {
        let n = 100;
        let x = linspace(0.0, 2.0 * std::f64::consts::PI, n);
        let f = x.mapv(f64::sin).into_dyn();
        let df = deriv1(&f, 0, &x).unwrap();
        // Interior error should be O(dx²) ≈ 1e-4 for n=100
        let max_err = (2..n - 2)
            .map(|i| (df[IxDyn(&[i])] - x[i].cos()).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max interior error = {}", max_err);
    }

    #[test]
    fn deriv2_uniform_sin() {
        let n = 100;
        let x = linspace(0.0, 2.0 * std::f64::consts::PI, n);
        let f = x.mapv(f64::sin).into_dyn();
        let d2f = deriv2(&f, 0, &x).unwrap();
        // d²sin/dx² = -sin; interior error O(dx²)
        let max_err = (2..n - 2)
            .map(|i| (d2f[IxDyn(&[i])] + x[i].sin()).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 1e-3, "max interior error = {}", max_err);
    }

    /// Couette flow is an exact steady N-S solution:
    ///   u(z) = z, v = w = 0, p = 0  →  residual must be identically zero.
    #[test]
    fn ns_residual_couette_is_zero() {
        let (nz, ny, nx) = (20, 4, 4);
        let x = linspace(0.0, 1.0, nx);
        let y = linspace(0.0, 1.0, ny);
        let z = linspace(0.0, 1.0, nz);
        let nu = 1e-3;

        let u = Array3::from_shape_fn((nz, ny, nx), |(iz, _, _)| z[iz]).into_dyn();
        let zero = ArrayD::zeros(u.raw_dim());

        let (rx, ry, rz) =
            ns_momentum_residual(&u, &zero, &zero, &zero, &x, &y, &z, nu).unwrap();

        let l2 = ns_residual_l2(&rx, &ry, &rz).unwrap();
        assert!(l2 < 1e-10, "Couette residual L2 = {:.2e}", l2);
    }

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

        let div = divergence(&u, &v, &w, &x, &y, &z).unwrap();
        let max_err = div.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(max_err < 1e-10, "max divergence error = {:.2e}", max_err);
    }
}
