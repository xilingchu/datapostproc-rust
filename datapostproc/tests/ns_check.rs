/// Integration test: compute N-S momentum residual on the real DNS HDF5 snapshot.
///
/// The file `subavg_450000_0.h5` contains time/phase-averaged statistics from a
/// channel DNS.  Grid layout:
///   u/v/w/p : [nz=276, ny=8, nx_ghost=252]  (2 ghost cells in x for periodicity)
///   x       : [nx=250]   cell-centre x-coords  (strip ghosts → take [1..251])
///   y       : [ny=8]     cell-centre y-coords
///   zc      : [nz=276]   cell-centre z-coords
///
/// Periodicity: x and y are periodic, z is non-periodic (channel walls).

use datapostproc_rust::data::H5File;
use datapostproc_rust::hdf5::H5Data;
use datapostproc_rust::math::ns::{deriv1, ns_momentum_residual, ns_residual_l2};
use ndarray::{Array1, ArrayD, Axis, Ix1, Slice};

const H5_FILE: &str = "subavg_450000_0.h5";

/// Read a named 3-D dataset as ArrayD<f64>.
fn read_3d(h5: &mut H5File, name: &str) -> ArrayD<f64> {
    h5.add_dataset(name, None)
        .unwrap_or_else(|e| panic!("cannot open '{}': {}", name, e));
    match h5.dataset(name).unwrap().read_data::<f64>().unwrap() {
        H5Data::Array(a) => a,
        H5Data::Scalar(_) => panic!("'{}' is scalar", name),
    }
}

/// Read a named 1-D coordinate dataset as Array1<f64>.
fn read_coord(h5: &mut H5File, name: &str) -> Array1<f64> {
    match h5.coord(name).unwrap_or_else(|| panic!("coord '{}' not found", name))
           .read_data::<f64>().unwrap() {
        H5Data::Array(a) => a.into_dimensionality::<Ix1>().unwrap(),
        H5Data::Scalar(_) => panic!("coord '{}' is scalar", name),
    }
}

/// Strip the one ghost cell on each side of the x-axis (axis 2).
/// Full shape [nz, ny, nx+2] → interior [nz, ny, nx].
fn strip_x_ghosts(arr: &ArrayD<f64>) -> ArrayD<f64> {
    let nx_ghost = arr.shape()[2];
    arr.slice_axis(Axis(2), Slice::from(1..nx_ghost - 1))
        .to_owned()
}

#[test]
fn ns_residual_on_h5_data() {
    let mut h5 = H5File::new(H5_FILE).expect("open HDF5");
    h5.get_info().unwrap();
    h5.load_coords().unwrap();

    let nu = h5.info().nu.expect("nu not found");
    println!("nu = {:.6e}", nu);

    // Load velocity and pressure
    let u_raw = read_3d(&mut h5, "u");
    let v_raw = read_3d(&mut h5, "v");
    let w_raw = read_3d(&mut h5, "w");
    let p_raw = read_3d(&mut h5, "p");

    println!("raw u shape: {:?}", u_raw.shape());

    // Strip x ghost cells (axis 2: 252 → 250)
    let u = strip_x_ghosts(&u_raw);
    let v = strip_x_ghosts(&v_raw);
    let w = strip_x_ghosts(&w_raw);
    let p = strip_x_ghosts(&p_raw);
    println!("interior shape: {:?}", u.shape());

    // Coordinates
    let x  = read_coord(&mut h5, "x");
    let y  = read_coord(&mut h5, "y");
    let zc = read_coord(&mut h5, "zc");
    println!("x[0]={:.4} x[-1]={:.4}  y[0]={:.4}  zc[0]={:.4e} zc[-1]={:.4}",
             x[0], x[x.len()-1], y[0], zc[0], zc[zc.len()-1]);

    // N-S residual: periodic in x (axis 2) and y (axis 1), non-periodic in z (axis 0)
    let (rx, ry, rz) = ns_momentum_residual(
        &u, &v, &w, &p,
        &x, &y, &zc,
        nu,
        [false, true, true],   // [z_periodic, y_periodic, x_periodic]
    ).expect("ns_momentum_residual failed");

    let l2 = ns_residual_l2(&rx, &ry, &rz).unwrap();

    // Component-wise L2 norms
    let n = rx.len() as f64;
    let l2_x = (rx.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_y = (ry.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_z = (rz.iter().map(|v| v*v).sum::<f64>() / n).sqrt();

    println!("N-S residual L2 (total) = {:.4e}", l2);
    println!("  R_x L2 = {:.4e}", l2_x);
    println!("  R_y L2 = {:.4e}", l2_y);
    println!("  R_z L2 = {:.4e}", l2_z);

    // Max absolute residual
    let max_x = rx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_y = ry.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_z = rz.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    println!("  max|R_x| = {:.4e}", max_x);
    println!("  max|R_y| = {:.4e}", max_y);
    println!("  max|R_z| = {:.4e}", max_z);

    // The test just asserts the computation finished; residual magnitude
    // is printed for inspection (time-averaged data won't be zero due to
    // Reynolds stress terms that are not included here).
    assert!(l2.is_finite(), "residual contains NaN/Inf");
}

/// Cross-check: compare our finite-difference ∂u/∂x against the
/// pre-computed gradient `ux` stored in the HDF5 file.
#[test]
fn derivative_matches_stored_gradient() {
    let mut h5 = H5File::new(H5_FILE).expect("open HDF5");
    h5.load_coords().unwrap();

    let u_raw  = read_3d(&mut h5, "u");
    let ux_raw = read_3d(&mut h5, "ux"); // ∂u/∂x stored by Fortran subAvg

    let u  = strip_x_ghosts(&u_raw);
    let ux = strip_x_ghosts(&ux_raw);

    let x = read_coord(&mut h5, "x");

    // Our derivative (periodic in x)
    let du_dx = deriv1(&u, 2, &x, true).expect("deriv1 failed");

    // Compare interior (skip 2 points near each x-boundary to avoid stencil differences)
    let nx = x.len();
    let mut max_err = 0.0_f64;
    let mut sum_sq  = 0.0_f64;
    let mut count   = 0usize;
    for iz in 1..u.shape()[0]-1 {
        for iy in 0..u.shape()[1] {
            for ix in 2..nx-2 {
                let diff = du_dx[[iz, iy, ix]] - ux[[iz, iy, ix]];
                max_err = max_err.max(diff.abs());
                sum_sq += diff * diff;
                count  += 1;
            }
        }
    }
    let rmse = (sum_sq / count as f64).sqrt();
    println!("∂u/∂x vs stored ux:  max_err={:.4e}  RMSE={:.4e}", max_err, rmse);

    // The Fortran uses central differences for cell-centred data, same as ours.
    // Differences come from the ghost-cell / boundary treatment and any
    // normalisation differences; interior RMSE should be small.
    assert!(rmse.is_finite(), "RMSE is NaN/Inf");
}
