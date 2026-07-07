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

use ndarray::{Array1, Array3, ArrayD, Axis};
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

/// RANS momentum residual using the stored second-moment tensor ⟨u_i u_j⟩.
///
/// For a time-averaged (or phase-averaged) DNS field the full momentum balance is:
///   ∂⟨u_i u_j⟩/∂x_j + ∂⟨p⟩/∂x_i − ν ∇²⟨u_i⟩ = 0  (steady mean flow)
///
/// where ⟨u_i u_j⟩ = ⟨u_i⟩⟨u_j⟩ + ⟨u_i' u_j'⟩ already contains both the
/// mean-flow convection and the Reynolds-stress divergence.  Using the stored
/// second moments directly avoids the need to split them.
///
/// # Arguments
/// * `u`, `v`, `w`  – mean velocity components, shape `(nz, ny, nx)`
/// * `p`            – mean pressure
/// * `uu`,`uv`,`uw`,`vv`,`vw`,`ww` – total second moments ⟨u_i u_j⟩, same shape
/// * `x`, `y`, `z`  – coordinate arrays
/// * `nu`           – kinematic viscosity
/// * `periodic`     – `[z_periodic, y_periodic, x_periodic]`
///
/// # Returns
/// `(res_x, res_y, res_z)` – RANS residual arrays; near-zero means the
/// time-averaged momentum budget closes.
#[allow(clippy::too_many_arguments)]
pub fn rans_momentum_residual(
    u:  &ArrayD<f64>,
    v:  &ArrayD<f64>,
    w:  &ArrayD<f64>,
    p:  &ArrayD<f64>,
    uu: &ArrayD<f64>,
    uv: &ArrayD<f64>,
    uw: &ArrayD<f64>,
    vv: &ArrayD<f64>,
    vw: &ArrayD<f64>,
    ww: &ArrayD<f64>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
    nu: f64,
    periodic: [bool; 3],
) -> Result<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>), Error> {
    let shape = u.shape();
    for (name, arr) in [("v",v),("w",w),("p",p),
                        ("uu",uu),("uv",uv),("uw",uw),
                        ("vv",vv),("vw",vw),("ww",ww)] {
        if arr.shape() != shape {
            return Err(format!("'{}' shape {:?} != u shape {:?}", name, arr.shape(), shape).into());
        }
    }

    let (pz, py, px) = (periodic[0], periodic[1], periodic[2]);

    // Divergence of total stress tensor ∂⟨u_i u_j⟩/∂x_j
    let conv_u = deriv1(uu, 2, x, px)? + deriv1(uv, 1, y, py)? + deriv1(uw, 0, z, pz)?;
    let conv_v = deriv1(uv, 2, x, px)? + deriv1(vv, 1, y, py)? + deriv1(vw, 0, z, pz)?;
    let conv_w = deriv1(uw, 2, x, px)? + deriv1(vw, 1, y, py)? + deriv1(ww, 0, z, pz)?;

    // Pressure gradients
    let dp_dx = deriv1(p, 2, x, px)?;
    let dp_dy = deriv1(p, 1, y, py)?;
    let dp_dz = deriv1(p, 0, z, pz)?;

    // Viscous diffusion of mean flow
    let visc_u = (deriv2(u, 2, x, px)? + deriv2(u, 1, y, py)? + deriv2(u, 0, z, pz)?) * nu;
    let visc_v = (deriv2(v, 2, x, px)? + deriv2(v, 1, y, py)? + deriv2(v, 0, z, pz)?) * nu;
    let visc_w = (deriv2(w, 2, x, px)? + deriv2(w, 1, y, py)? + deriv2(w, 0, z, pz)?) * nu;

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

/// N-S momentum residual using the **exact Fortran `lineStep` stencil**.
///
/// Matches `mod_solver.f90 :: lineStep` exactly:
/// - Convection: face-interpolated skew-symmetric form
///     `conv_u = 0.25*((u_i+u_{i+1})^2 − (u_{i-1}+u_i)^2)*d1xp + …`
/// - Viscosity: 3-point non-uniform Laplacian
///     `visc_u = (due*d2xp − duw*d2xm + …)*ν`
/// - Pressure: one-sided face difference
///     `dp/dx = (p_{i+1}−p_i)*d1x`
///
/// # Grid layout (inst_400000 files)
/// ```text
///   u, v, w, p : shape (nz=276, ny=384, nx_f=801)
///   x          : 800 cell-centre x-coords  (uniform, Δx = x[0])
///   y          : 384 cell-centre y-coords  (uniform, periodic, Δy = y[0])
///   zc         : 276 z-face positions      (zc[0]=0=bottom wall, zc[275]=2=top wall)
///   zd         : 551 interleaved z dual-grid (zd[2k]=zc[k], zd[2k+1]=cell-centre_k)
/// ```
///
/// # Returns
/// `(res_u, res_v, res_w)` with shape `(nz_int=273, ny=384, nx_int=799)`.
/// Interior only: `iz = 1..273`, `ix = 1..799`, all `iy` (periodic y).
/// `res_w[iz,…]` is the w-residual at z-face `iz+1`.
///
/// # Sign convention
/// `R = conv + grad_p − ν∇²u = headx − ∂u/∂t`
pub fn ns_residual_fortran_stencil(
    u:  &ArrayD<f64>, v: &ArrayD<f64>, w: &ArrayD<f64>, p: &ArrayD<f64>,
    x:  &Array1<f64>, y: &Array1<f64>,
    zc: &Array1<f64>, zd: &Array1<f64>,
    nu: f64,
) -> Result<(Array3<f64>, Array3<f64>, Array3<f64>), Error> {
    let nz  = u.shape()[0];   // 276
    let ny  = u.shape()[1];   // 384
    let nxf = u.shape()[2];   // 801 (includes right ghost)
    let nx  = x.len();        // 800

    if nxf < nx + 1 {
        return Err("u x-dimension must be nx+1 (one right ghost cell)".into());
    }
    if zc.len() != nz || zd.len() != 2 * (nz - 1) + 1 {
        return Err(format!(
            "zc must have {} entries, zd must have {} entries; got {} and {}",
            nz, 2*(nz-1)+1, zc.len(), zd.len()
        ).into());
    }

    // Uniform x spacing
    let dx  = x[1] - x[0];
    let d1x = 1.0 / dx;           // d1x = d1xp = d1xm  (uniform grid)
    let d2x = d1x * d1x;          // d2xp = d2xm

    // Uniform y spacing (periodic)
    let dy  = y[1] - y[0];
    let d1y = 1.0 / dy;
    let d2y = d1y * d1y;

    // z-coefficients for cells iz = 0..nz-2  (= 0..274, nz-1 = 275 cells)
    // zc[k] = face k (k=0..275), zd[2k] = zc[k], zd[2k+1] = cell-centre k
    let ncells = nz - 1;   // 275

    // d1z[k]  = 1 / cell_width[k]  = 1 / (zc[k+1] - zc[k])
    let d1z: Vec<f64> = (0..ncells).map(|k| 1.0 / (zc[k + 1] - zc[k])).collect();

    // d1zp_uv[k] = 1 / (cellctr[k+1] - cellctr[k])  for u/v viscosity (k=0..ncells-2)
    // d1zm_uv[k] = 1 / (cellctr[k]   - cellctr[k-1]) for u/v viscosity (k=1..ncells-1)
    // Stored d1zm_uv[j] = d1zm at cell j+1 so j=0..ncells-2 covers k=1..ncells-1
    let d1zp_uv: Vec<f64> = (0..ncells - 1).map(|k| 1.0 / (zd[2*k + 3] - zd[2*k + 1])).collect();
    let d1zm_uv: Vec<f64> = (1..ncells    ).map(|k| 1.0 / (zd[2*k + 1] - zd[2*k - 1])).collect();
    // d2zp_uv[k] = d1z[k] * d1zp_uv[k]  (k=0..ncells-2)
    // d2zm_uv[k] = d1z[k] * d1zm_uv[k-1] (k=1..ncells-1, so index shift)
    let d2zp_uv: Vec<f64> = (0..ncells - 1).map(|k| d1z[k] * d1zp_uv[k]).collect();
    let d2zm_uv: Vec<f64> = (1..ncells    ).map(|k| d1z[k] * d1zm_uv[k - 1]).collect();

    // z-coefficients for w at face iz (iz=1..ncells-1 = 1..274):
    // d1zp_w[iz] = 1 / (cellctr[iz] - cellctr[iz-1])  = 1 / (zd[2iz+1] - zd[2iz-1])
    // d2zpp_w[iz] = d1zp_w[iz] * d1z[iz]      (cell above face iz)
    // d2zpm_w[iz] = d1zp_w[iz] * d1z[iz-1]    (cell below face iz)
    // stored at index iz-1 → range 0..ncells-2

    // Interior output region: iz=1..273, iy=0..ny-1, ix=1..799
    let nz_out = 273usize;   // iz = 1..273 (inclusive)
    let nx_out = nx - 1;     // ix = 1..799

    let mut rx = Array3::<f64>::zeros((nz_out, ny, nx_out));
    let mut ry = Array3::<f64>::zeros((nz_out, ny, nx_out));
    let mut rz = Array3::<f64>::zeros((nz_out, ny, nx_out));

    for iz in 1usize..=nz_out {          // iz = 1..273  (cell index for u/v, face index for w)
        let oz = iz - 1;                 // output z index 0..272

        // u/v z-coefficients at cell iz (iz=1..273, safe: d2zp_uv goes 0..273, d2zm_uv goes 0..273)
        let d2zp_uv_k = d2zp_uv[iz];        // iz <= 273 < ncells-1=274 ✓
        let d2zm_uv_k = d2zm_uv[iz - 1];    // iz-1 <= 272, d2zm_uv has ncells-1=274 entries ✓

        // w z-coefficients at face iz (stored at index iz-1)
        let d1zp_w  = 1.0 / (zd[2 * iz + 1] - zd[2 * iz - 1]);
        let d2zpp_w = d1zp_w * d1z[iz];
        let d2zpm_w = d1zp_w * d1z[iz - 1];
        let d1z_w   = d1zp_w;  // conv_w z-term uses d1zp (face-to-face)

        for iy in 0..ny {
            let iy_p = (iy + 1) % ny;
            let iy_m = (iy + ny - 1) % ny;

            for ix in 1..nx {            // ix = 1..799
                let ox = ix - 1;         // output x index 0..798

                // ── u-equation ──────────────────────────────────────────────────
                // Face interpolated velocities (each is sum of two adjacent values = 2×face_value)
                let ue  = u[[iz, iy,   ix    ]] + u[[iz, iy,   ix + 1]];
                let uw_ = u[[iz, iy,   ix - 1]] + u[[iz, iy,   ix    ]];
                let uh  = u[[iz, iy,   ix    ]] + u[[iz, iy_p, ix    ]];
                let uq  = u[[iz, iy_m, ix    ]] + u[[iz, iy,   ix    ]];
                let un  = u[[iz, iy,   ix    ]] + u[[iz + 1, iy, ix  ]];
                let us_ = u[[iz, iy,   ix    ]] + u[[iz - 1, iy, ix  ]];

                let vh  = v[[iz, iy,   ix    ]] + v[[iz, iy,   ix + 1]];
                let vq  = v[[iz, iy_m, ix    ]] + v[[iz, iy_m, ix + 1]];
                let wn  = w[[iz, iy,   ix    ]] + w[[iz, iy,   ix + 1]];
                let ws_ = w[[iz - 1, iy, ix  ]] + w[[iz - 1, iy, ix + 1]];

                let conv_u = 0.25 * (ue * ue - uw_ * uw_) * d1x
                           + 0.25 * (uh * vh - uq * vq)   * d1y
                           + 0.25 * (un * wn - us_ * ws_) * d1z[iz];

                let due = u[[iz, iy,   ix + 1]] - u[[iz, iy,   ix    ]];
                let duw = u[[iz, iy,   ix    ]] - u[[iz, iy,   ix - 1]];
                let duh = u[[iz, iy_p, ix    ]] - u[[iz, iy,   ix    ]];
                let duq = u[[iz, iy,   ix    ]] - u[[iz, iy_m, ix    ]];
                let dun = u[[iz + 1, iy, ix  ]] - u[[iz, iy,   ix    ]];
                let dus = u[[iz, iy,   ix    ]] - u[[iz - 1, iy, ix  ]];

                let visc_u = (due * d2x - duw * d2x
                            + duh * d2y - duq * d2y
                            + dun * d2zp_uv_k - dus * d2zm_uv_k) * nu;

                let dp_dx = (p[[iz, iy, ix + 1]] - p[[iz, iy, ix]]) * d1x;

                rx[[oz, iy, ox]] = conv_u + dp_dx - visc_u;

                // ── v-equation ──────────────────────────────────────────────────
                let ve  = v[[iz, iy,   ix    ]] + v[[iz, iy,   ix + 1]];
                let vw_ = v[[iz, iy,   ix - 1]] + v[[iz, iy,   ix    ]];
                let vh2 = v[[iz, iy,   ix    ]] + v[[iz, iy_p, ix    ]];
                let vq2 = v[[iz, iy_m, ix    ]] + v[[iz, iy,   ix    ]];
                let vn  = v[[iz, iy,   ix    ]] + v[[iz + 1, iy, ix  ]];
                let vs_ = v[[iz, iy,   ix    ]] + v[[iz - 1, iy, ix  ]];

                // Cross terms for v-conv: u interpolated to top y-face, w interpolated to top y-face
                let ue_v = u[[iz, iy,   ix    ]] + u[[iz, iy_p, ix    ]];
                let uw_v = u[[iz, iy,   ix - 1]] + u[[iz, iy_p, ix - 1]];
                let wn_v = w[[iz, iy,   ix    ]] + w[[iz, iy_p, ix    ]];
                let ws_v = w[[iz - 1, iy, ix  ]] + w[[iz - 1, iy_p, ix]];

                let conv_v = 0.25 * (ue_v * ve - uw_v * vw_) * d1x
                           + 0.25 * (vh2 * vh2 - vq2 * vq2)  * d1y
                           + 0.25 * (wn_v * vn - ws_v * vs_) * d1z[iz];

                let dve = v[[iz, iy,   ix + 1]] - v[[iz, iy,   ix    ]];
                let dvw = v[[iz, iy,   ix    ]] - v[[iz, iy,   ix - 1]];
                let dvh = v[[iz, iy_p, ix    ]] - v[[iz, iy,   ix    ]];
                let dvq = v[[iz, iy,   ix    ]] - v[[iz, iy_m, ix    ]];
                let dvn = v[[iz + 1, iy, ix  ]] - v[[iz, iy,   ix    ]];
                let dvs = v[[iz, iy,   ix    ]] - v[[iz - 1, iy, ix  ]];

                let visc_v = (dve * d2x - dvw * d2x
                            + dvh * d2y - dvq * d2y
                            + dvn * d2zp_uv_k - dvs * d2zm_uv_k) * nu;

                let dp_dy = (p[[iz, iy_p, ix]] - p[[iz, iy, ix]]) * d1y;

                ry[[oz, iy, ox]] = conv_v + dp_dy - visc_v;

                // ── w-equation  (w at z-face iz, between cells iz-1 and iz) ────
                let we  = w[[iz, iy,   ix    ]] + w[[iz, iy,   ix + 1]];
                let ww_ = w[[iz, iy,   ix - 1]] + w[[iz, iy,   ix    ]];
                let wh  = w[[iz, iy,   ix    ]] + w[[iz, iy_p, ix    ]];
                let wq  = w[[iz, iy_m, ix    ]] + w[[iz, iy,   ix    ]];
                let wn2 = w[[iz, iy,   ix    ]] + w[[iz + 1, iy, ix  ]];
                let ws2 = w[[iz, iy,   ix    ]] + w[[iz - 1, iy, ix  ]];

                // Cross terms for w: u and v interpolated to z-face iz
                let ue_w = u[[iz, iy,   ix    ]] + u[[iz + 1, iy,   ix    ]];
                let uw_w = u[[iz, iy,   ix - 1]] + u[[iz + 1, iy,   ix - 1]];
                let vh_w = v[[iz, iy,   ix    ]] + v[[iz + 1, iy,   ix    ]];
                let vq_w = v[[iz, iy_m, ix    ]] + v[[iz + 1, iy_m, ix    ]];

                // NOTE: w-conv uses d1xm, d1ym, d1zp (different from u-conv)
                // For uniform x,y: d1xm=d1x, d1ym=d1y; so only d1zp differs
                let conv_w = 0.25 * (ue_w * we - uw_w * ww_) * d1x
                           + 0.25 * (vh_w * wh - vq_w * wq)  * d1y
                           + 0.25 * (wn2 * wn2 - ws2 * ws2)  * d1z_w;

                let dwe = w[[iz, iy,   ix + 1]] - w[[iz, iy,   ix    ]];
                let dww = w[[iz, iy,   ix    ]] - w[[iz, iy,   ix - 1]];
                let dwh = w[[iz, iy_p, ix    ]] - w[[iz, iy,   ix    ]];
                let dwq = w[[iz, iy,   ix    ]] - w[[iz, iy_m, ix    ]];
                let dwn = w[[iz + 1, iy, ix  ]] - w[[iz, iy,   ix    ]];
                let dws = w[[iz, iy,   ix    ]] - w[[iz - 1, iy, ix  ]];

                // w viscosity uses d2zpp/d2zpm (staggered z second-derivative)
                let visc_w = (dwe * d2x   - dww * d2x
                            + dwh * d2y   - dwq * d2y
                            + dwn * d2zpp_w - dws * d2zpm_w) * nu;

                let dp_dz = (p[[iz + 1, iy, ix]] - p[[iz, iy, ix]]) * d1zp_w;

                rz[[oz, iy, ox]] = conv_w + dp_dz - visc_w;
            }
        }
    }

    Ok((rx, ry, rz))
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
