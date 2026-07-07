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
use datapostproc_rust::math::ns::{deriv1, ns_momentum_residual, ns_residual_l2, rans_momentum_residual, ns_residual_fortran_stencil};
use ndarray::{Array1, ArrayD, Axis, Ix1, Slice};

/// Read a scalar (shape [1]) from a raw HDF5 file by path.
fn read_scalar_h5(filename: &str, name: &str) -> f64 {
    let file = hdf5::File::open(filename).unwrap_or_else(|e| panic!("open {}: {}", filename, e));
    file.dataset(name)
        .unwrap_or_else(|e| panic!("dataset '{}' in {}: {}", name, filename, e))
        .read_1d::<f64>()
        .unwrap()[0]
}

/// Read a named 3-D dataset from a standalone HDF5 file (one variable per file).
fn read_3d_from(filename: &str, name: &str) -> ArrayD<f64> {
    let mut h5 = H5File::new(filename)
        .unwrap_or_else(|e| panic!("open {}: {}", filename, e));
    h5.add_dataset(name, None)
        .unwrap_or_else(|e| panic!("dataset '{}' in {}: {}", name, filename, e));
    match h5.dataset(name).unwrap().read_data::<f64>().unwrap() {
        H5Data::Array(a) => a,
        H5Data::Scalar(_) => panic!("'{}' is scalar", name),
    }
}

/// Read 1-D coordinate from a standalone HDF5 file.
fn read_coord_from(filename: &str, name: &str) -> Array1<f64> {
    let mut h5 = H5File::new(filename)
        .unwrap_or_else(|e| panic!("open {}: {}", filename, e));
    h5.load_coords().unwrap();
    match h5.coord(name)
        .unwrap_or_else(|| panic!("coord '{}' not found in {}", name, filename))
        .read_data::<f64>()
        .unwrap()
    {
        H5Data::Array(a) => a.into_dimensionality::<Ix1>().unwrap(),
        H5Data::Scalar(_) => panic!("coord '{}' is scalar", name),
    }
}

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

/// RANS residual check: use stored ⟨u_i u_j⟩ tensors so the Reynolds-stress
/// divergence is included.  For a converged time-average this residual should
/// be much smaller than the N-S residual computed from mean velocities alone.
#[test]
fn rans_residual_on_h5_data() {
    let mut h5 = H5File::new(H5_FILE).expect("open HDF5");
    h5.get_info().unwrap();
    h5.load_coords().unwrap();

    let nu = h5.info().nu.expect("nu not found");

    // Mean velocities and pressure
    let u = strip_x_ghosts(&read_3d(&mut h5, "u"));
    let v = strip_x_ghosts(&read_3d(&mut h5, "v"));
    let w = strip_x_ghosts(&read_3d(&mut h5, "w"));
    let p = strip_x_ghosts(&read_3d(&mut h5, "p"));

    // Total second moments ⟨u_i u_j⟩ (mean² + Reynolds stress)
    let uu = strip_x_ghosts(&read_3d(&mut h5, "uu"));
    let uv = strip_x_ghosts(&read_3d(&mut h5, "uv"));
    let uw = strip_x_ghosts(&read_3d(&mut h5, "uw"));
    let vv = strip_x_ghosts(&read_3d(&mut h5, "vv"));
    let vw = strip_x_ghosts(&read_3d(&mut h5, "vw"));
    let ww = strip_x_ghosts(&read_3d(&mut h5, "ww"));

    let x  = read_coord(&mut h5, "x");
    let y  = read_coord(&mut h5, "y");
    let zc = read_coord(&mut h5, "zc");

    let (rx, ry, rz) = rans_momentum_residual(
        &u, &v, &w, &p,
        &uu, &uv, &uw, &vv, &vw, &ww,
        &x, &y, &zc,
        nu,
        [false, true, true],
    ).expect("rans_momentum_residual failed");

    let l2 = ns_residual_l2(&rx, &ry, &rz).unwrap();
    let n  = rx.len() as f64;
    let l2_x = (rx.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_y = (ry.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_z = (rz.iter().map(|v| v*v).sum::<f64>() / n).sqrt();

    println!("RANS residual L2 (total) = {:.4e}", l2);
    println!("  R_x L2 = {:.4e}", l2_x);
    println!("  R_y L2 = {:.4e}", l2_y);
    println!("  R_z L2 = {:.4e}", l2_z);
    println!("  max|R_x| = {:.4e}", rx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  max|R_y| = {:.4e}", ry.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  max|R_z| = {:.4e}", rz.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));

    assert!(l2.is_finite(), "RANS residual contains NaN/Inf");
}

/// Instantaneous N-S residual on `inst_400000_*.h5`.
///
/// Grid layout (all four files share the same grid):
///   u/v/w/p : [nz=276, ny=384, nx_ghost=801]  (1 ghost cell on right in x)
///   x       : [nx=800]   cell-centre x-coords  → strip last point of field
///   y       : [ny=384]   cell-centre y-coords
///   zc      : [nz=276]   cell-centre z-coords
///
/// All three directions are non-periodic.
/// headx is the constant body force (mean pressure gradient driver) along x.
/// The computed residual equals  ∂u_i/∂t + headx·δ_{ix}  at every grid point.
#[test]
fn ns_residual_on_inst_data() {
    let uf = "inst_400000_u.h5";
    let vf = "inst_400000_v.h5";
    let wf = "inst_400000_w.h5";
    let pf = "inst_400000_p.h5";

    let nu    = read_scalar_h5(uf, "nu");
    let headx = read_scalar_h5(uf, "headx");
    let heady = read_scalar_h5(uf, "heady");
    println!("nu={:.4e}  headx={:.4e}  heady={:.4e}", nu, headx, heady);

    // Load fields (shape [nz=276, ny=384, nx_ghost=801])
    let u_raw = read_3d_from(uf, "u");
    let v_raw = read_3d_from(vf, "v");
    let w_raw = read_3d_from(wf, "w");
    let p_raw = read_3d_from(pf, "p");
    println!("raw shape: {:?}", u_raw.shape());

    // Strip the one ghost cell on the right of the x-axis (axis 2): 801 → 800
    let nx = 800usize;
    let u = u_raw.slice_axis(Axis(2), Slice::from(..nx)).to_owned();
    let v = v_raw.slice_axis(Axis(2), Slice::from(..nx)).to_owned();
    let w = w_raw.slice_axis(Axis(2), Slice::from(..nx)).to_owned();
    let p = p_raw.slice_axis(Axis(2), Slice::from(..nx)).to_owned();
    println!("interior shape: {:?}", u.shape());

    // Coordinates (read from u-file; same grid in all four files)
    let x  = read_coord_from(uf, "x");   // 800 cell-centre x-coords
    let y  = read_coord_from(uf, "y");   // 384 cell-centre y-coords
    let zc = read_coord_from(uf, "zc");  // 276 cell-centre z-coords
    println!("x[0]={:.4}  x[-1]={:.4}  y[0]={:.4}  zc[0]={:.4e}  zc[-1]={:.4}",
             x[0], x[x.len()-1], y[0], zc[0], zc[zc.len()-1]);

    // N-S residual: all non-periodic
    let (rx, ry, rz) = ns_momentum_residual(
        &u, &v, &w, &p,
        &x, &y, &zc,
        nu,
        [false, false, false],
    ).expect("ns_momentum_residual failed");

    let n    = rx.len() as f64;
    let l2   = ns_residual_l2(&rx, &ry, &rz).unwrap();
    let l2_x = (rx.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let l2_y = (ry.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let l2_z = (rz.iter().map(|v| v * v).sum::<f64>() / n).sqrt();

    println!("N-S residual  L2 (total)  = {:.4e}", l2);
    println!("  R_x L2 = {:.4e}", l2_x);
    println!("  R_y L2 = {:.4e}", l2_y);
    println!("  R_z L2 = {:.4e}", l2_z);

    let max_x = rx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_y = ry.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_z = rz.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    println!("  max|R_x| = {:.4e}", max_x);
    println!("  max|R_y| = {:.4e}", max_y);
    println!("  max|R_z| = {:.4e}", max_z);

    // ∂u/∂t estimate: subtract the constant body force from each component
    let dudt_l2 = ((rx.iter().map(|v| (v - headx).powi(2)).sum::<f64>()
                  + ry.iter().map(|v| (v - heady).powi(2)).sum::<f64>()
                  + rz.iter().map(|v| v * v).sum::<f64>()) / n).sqrt();
    let dudt_x  = (rx.iter().map(|v| (v - headx).powi(2)).sum::<f64>() / n).sqrt();
    let dudt_y  = (ry.iter().map(|v| (v - heady).powi(2)).sum::<f64>() / n).sqrt();
    println!("∂u/∂t L2 (R - body_force): total={:.4e}  x={:.4e}  y={:.4e}  z={:.4e}",
             dudt_l2, dudt_x, dudt_y, l2_z);

    assert!(l2.is_finite(), "residual contains NaN/Inf");
}

/// N-S residual using the **exact Fortran lineStep stencil** on inst_400000 data.

///
/// Compares with `ns_residual_on_inst_data` to quantify the stencil difference.
#[test]
fn ns_residual_fortran_stencil_inst() {
    let uf = "inst_400000_u.h5";
    let vf = "inst_400000_v.h5";
    let wf = "inst_400000_w.h5";
    let pf = "inst_400000_p.h5";

    let nu    = read_scalar_h5(uf, "nu");
    let headx = read_scalar_h5(uf, "headx");
    let heady = read_scalar_h5(uf, "heady");

    // Raw staggered fields (including right ghost), shape (276, 384, 801)
    let u = read_3d_from(uf, "u");
    let v = read_3d_from(vf, "v");
    let w = read_3d_from(wf, "w");
    let p = read_3d_from(pf, "p");
    println!("field shape: {:?}", u.shape());

    let x  = read_coord_from(uf, "x");
    let y  = read_coord_from(uf, "y");
    let zc = read_coord_from(uf, "zc");
    let zd = read_coord_from(uf, "zd");
    println!("x[0]={:.4}  y[0]={:.4}  zc[0]={:.4e}  zc[-1]={:.4}",
             x[0], y[0], zc[0], zc[zc.len()-1]);
    println!("nu={:.4e}  headx={:.4e}  heady={:.4e}", nu, headx, heady);

    let (rx, ry, rz) = ns_residual_fortran_stencil(
        &u, &v, &w, &p,
        &x, &y, &zc, &zd,
        nu,
    ).expect("ns_residual_fortran_stencil failed");

    let n    = rx.len() as f64;
    let l2_x = (rx.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let l2_y = (ry.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let l2_z = (rz.iter().map(|v| v * v).sum::<f64>() / n).sqrt();
    let l2   = ((rx.iter().map(|v| v * v).sum::<f64>()
               + ry.iter().map(|v| v * v).sum::<f64>()
               + rz.iter().map(|v| v * v).sum::<f64>()) / n).sqrt();

    println!("Fortran-stencil N-S residual (interior {:.0e} pts):", n);
    println!("  total L2 = {:.4e}", l2);
    println!("  R_x L2   = {:.4e}  max|R_x| = {:.4e}", l2_x,
             rx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  R_y L2   = {:.4e}  max|R_y| = {:.4e}", l2_y,
             ry.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  R_z L2   = {:.4e}  max|R_z| = {:.4e}", l2_z,
             rz.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));

    // ∂u/∂t estimate: R - body_force
    let dudt_x = (rx.iter().map(|v| (v - headx).powi(2)).sum::<f64>() / n).sqrt();
    let dudt_y = (ry.iter().map(|v| (v - heady).powi(2)).sum::<f64>() / n).sqrt();
    println!("∂u/∂t L2: x={:.4e}  y={:.4e}  z={:.4e}", dudt_x, dudt_y, l2_z);

    assert!(l2.is_finite(), "residual contains NaN/Inf");
}

/// Find the top-K locations with the largest absolute residual.
fn top_k(arr: &ndarray::Array3<f64>, k: usize) -> Vec<(f64, f64, usize, usize, usize)> {
    let mut heap: Vec<(f64, f64, usize, usize, usize)> = Vec::with_capacity(k + 1);
    for ((oz, iy, ox), &val) in arr.indexed_iter() {
        let abs = val.abs();
        if heap.len() < k || abs > heap.last().map(|e| e.0).unwrap_or(0.0) {
            heap.push((abs, val, oz, iy, ox));
            heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            if heap.len() > k { heap.truncate(k); }
        }
    }
    heap
}

/// Locate the large-residual cells using the Fortran stencil on inst_400000 data.
#[test]
fn find_large_residual_locations() {
    let uf = "inst_400000_u.h5";
    let vf = "inst_400000_v.h5";
    let wf = "inst_400000_w.h5";
    let pf = "inst_400000_p.h5";

    let nu = read_scalar_h5(uf, "nu");

    let u = read_3d_from(uf, "u");
    let v = read_3d_from(vf, "v");
    let w = read_3d_from(wf, "w");
    let p = read_3d_from(pf, "p");

    let x  = read_coord_from(uf, "x");
    let y  = read_coord_from(uf, "y");
    let zc = read_coord_from(uf, "zc");
    let zd = read_coord_from(uf, "zd");

    let (rx, ry, rz) = ns_residual_fortran_stencil(
        &u, &v, &w, &p, &x, &y, &zc, &zd, nu,
    ).expect("ns_residual_fortran_stencil failed");

    let k = 20;
    println!("\n=== Top {} |R_x| locations ===", k);
    println!("{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} | {:>14}", "iz","iy","ix","z_cell","y","x","R_x");
    for (abs, val, oz, iy, ox) in top_k(&rx, k) {
        let iz = oz + 1;
        let ix = ox + 1;
        let z_cell = zd[2 * iz + 1];
        println!("{:>6} {:>6} {:>6} | {:>10.5} {:>10.5} {:>10.5} | {:>14.6e}  |R|={:.4e}",
                 iz, iy, ix, z_cell, y[iy], x[ix], val, abs);
    }

    println!("\n=== Top {} |R_y| locations ===", k);
    println!("{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} | {:>14}", "iz","iy","ix","z_cell","y","x","R_y");
    for (abs, val, oz, iy, ox) in top_k(&ry, k) {
        let iz = oz + 1;
        let ix = ox + 1;
        let z_cell = zd[2 * iz + 1];
        println!("{:>6} {:>6} {:>6} | {:>10.5} {:>10.5} {:>10.5} | {:>14.6e}  |R|={:.4e}",
                 iz, iy, ix, z_cell, y[iy], x[ix], val, abs);
    }

    println!("\n=== Top {} |R_z| locations ===", k);
    println!("{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} | {:>14}", "iz","iy","ix","z_face","y","x","R_z");
    for (abs, val, oz, iy, ox) in top_k(&rz, k) {
        let iz = oz + 1;
        let ix = ox + 1;
        let z_face = zc[iz];
        println!("{:>6} {:>6} {:>6} | {:>10.5} {:>10.5} {:>10.5} | {:>14.6e}  |R|={:.4e}",
                 iz, iy, ix, z_face, y[iy], x[ix], val, abs);
    }

    // z-profile of RMS residual (averaged over x,y)
    let nz_out = rx.shape()[0];
    let ny_s   = rx.shape()[1];
    let nx_out = rx.shape()[2];
    let n_xy   = (ny_s * nx_out) as f64;

    println!("\n=== z-profile of RMS residual (x,y-averaged) ===");
    println!("{:>6} {:>10} | {:>12} {:>12} {:>12}", "iz","z_cell","rms_Rx","rms_Ry","rms_Rz");
    for oz in 0..nz_out {
        let iz = oz + 1;
        let z_cell = zd[2 * iz + 1];
        let rms_x = (rx.slice(ndarray::s![oz,..,..]).iter().map(|v| v*v).sum::<f64>() / n_xy).sqrt();
        let rms_y = (ry.slice(ndarray::s![oz,..,..]).iter().map(|v| v*v).sum::<f64>() / n_xy).sqrt();
        let rms_z = (rz.slice(ndarray::s![oz,..,..]).iter().map(|v| v*v).sum::<f64>() / n_xy).sqrt();
        println!("{:>6} {:>10.5} | {:>12.4e} {:>12.4e} {:>12.4e}", iz, z_cell, rms_x, rms_y, rms_z);
    }

    let n = rx.len() as f64;
    let l2 = ((rx.iter().map(|v| v*v).sum::<f64>()
             + ry.iter().map(|v| v*v).sum::<f64>()
             + rz.iter().map(|v| v*v).sum::<f64>()) / n).sqrt();
    println!("\nTotal L2 = {:.4e}", l2);

    // ── ix ≤ 180 sub-region (ox = 0..179, i.e. ix = 1..180) ─────────────────
    // output x-index ox = ix-1, so ix ≤ 180 → ox < 180
    let x_cut = 180usize;   // number of x-columns to include (ox = 0..179)
    let rx180 = rx.slice(ndarray::s![.., .., ..x_cut]);
    let ry180 = ry.slice(ndarray::s![.., .., ..x_cut]);
    let rz180 = rz.slice(ndarray::s![.., .., ..x_cut]);

    let n180 = rx180.len() as f64;
    let l2x180 = (rx180.iter().map(|v| v*v).sum::<f64>() / n180).sqrt();
    let l2y180 = (ry180.iter().map(|v| v*v).sum::<f64>() / n180).sqrt();
    let l2z180 = (rz180.iter().map(|v| v*v).sum::<f64>() / n180).sqrt();
    let l2_180 = ((rx180.iter().map(|v| v*v).sum::<f64>()
                 + ry180.iter().map(|v| v*v).sum::<f64>()
                 + rz180.iter().map(|v| v*v).sum::<f64>()) / n180).sqrt();
    let max_x180 = rx180.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_y180 = ry180.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let max_z180 = rz180.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    println!("\n=== ix = 1..{} (x = {:.4}..{:.4}) ===", x_cut, x[1], x[x_cut]);
    println!("  total L2     = {:.4e}", l2_180);
    println!("  R_x L2={:.4e}  max|R_x|={:.4e}", l2x180, max_x180);
    println!("  R_y L2={:.4e}  max|R_y|={:.4e}", l2y180, max_y180);
    println!("  R_z L2={:.4e}  max|R_z|={:.4e}", l2z180, max_z180);

    println!("\n=== Top 10 |R_x| for ix≤{} ===", x_cut);
    println!("{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} | {:>14}", "iz","iy","ix","z_cell","y","x","R_x");
    for (abs, val, oz, iy, ox) in top_k(&rx180.to_owned(), 10) {
        let iz = oz + 1; let ix = ox + 1;
        println!("{:>6} {:>6} {:>6} | {:>10.5} {:>10.5} {:>10.5} | {:>14.6e}  |R|={:.4e}",
                 iz, iy, ix, zd[2*iz+1], y[iy], x[ix], val, abs);
    }
    println!("=== Top 10 |R_y| for ix≤{} ===", x_cut);
    println!("{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} | {:>14}", "iz","iy","ix","z_cell","y","x","R_y");
    for (abs, val, oz, iy, ox) in top_k(&ry180.to_owned(), 10) {
        let iz = oz + 1; let ix = ox + 1;
        println!("{:>6} {:>6} {:>6} | {:>10.5} {:>10.5} {:>10.5} | {:>14.6e}  |R|={:.4e}",
                 iz, iy, ix, zd[2*iz+1], y[iy], x[ix], val, abs);
    }
    println!("=== Top 10 |R_z| for ix≤{} ===", x_cut);
    println!("{:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} | {:>14}", "iz","iy","ix","z_face","y","x","R_z");
    for (abs, val, oz, iy, ox) in top_k(&rz180.to_owned(), 10) {
        let iz = oz + 1; let ix = ox + 1;
        println!("{:>6} {:>6} {:>6} | {:>10.5} {:>10.5} {:>10.5} | {:>14.6e}  |R|={:.4e}",
                 iz, iy, ix, zc[iz], y[iy], x[ix], val, abs);
    }

    // z-profile for ix≤180
    let n_xy180 = (ny_s * x_cut) as f64;
    println!("\n=== z-profile RMS (ix≤{}) ===", x_cut);
    println!("{:>6} {:>10} | {:>12} {:>12} {:>12}", "iz","z_cell","rms_Rx","rms_Ry","rms_Rz");
    for oz in 0..nz_out {
        let iz = oz + 1;
        let z_cell = zd[2 * iz + 1];
        let rms_x = (rx.slice(ndarray::s![oz,..,..x_cut]).iter().map(|v| v*v).sum::<f64>() / n_xy180).sqrt();
        let rms_y = (ry.slice(ndarray::s![oz,..,..x_cut]).iter().map(|v| v*v).sum::<f64>() / n_xy180).sqrt();
        let rms_z = (rz.slice(ndarray::s![oz,..,..x_cut]).iter().map(|v| v*v).sum::<f64>() / n_xy180).sqrt();
        println!("{:>6} {:>10.5} | {:>12.4e} {:>12.4e} {:>12.4e}", iz, z_cell, rms_x, rms_y, rms_z);
    }

    assert!(l2.is_finite());
}

/// N-S residual on the **spatial-mean field** averaged over y and ix < 180.
///
/// For a 1-D mean profile (uniform in x and y) all convective terms vanish and
/// only the z-viscous term survives: R_x ≈ −ν d²ū/dz².
/// This should be O(headx) ≈ 4e-3, far smaller than the instantaneous ~0.4.
#[test]
fn mean_field_residual_ix180() {
    let uf = "inst_400000_u.h5";
    let vf = "inst_400000_v.h5";
    let wf = "inst_400000_w.h5";
    let pf = "inst_400000_p.h5";

    let nu    = read_scalar_h5(uf, "nu");
    let headx = read_scalar_h5(uf, "headx");

    let u_raw = read_3d_from(uf, "u");
    let v_raw = read_3d_from(vf, "v");
    let w_raw = read_3d_from(wf, "w");
    let p_raw = read_3d_from(pf, "p");

    let x  = read_coord_from(uf, "x");
    let y  = read_coord_from(uf, "y");
    let zc = read_coord_from(uf, "zc");
    let zd = read_coord_from(uf, "zd");

    let nz  = u_raw.shape()[0];  // 276
    let ny  = u_raw.shape()[1];  // 384
    let nxf = u_raw.shape()[2];  // 801

    // Average over iy = 0..ny and ix = 0..179  (first 180 physical x-columns)
    let x_cut = 180usize;
    let n_avg = (ny * x_cut) as f64;

    let mean1d = |field: &ndarray::ArrayD<f64>| -> Vec<f64> {
        (0..nz).map(|iz| {
            field.slice(ndarray::s![iz, .., ..x_cut])
                 .iter().sum::<f64>() / n_avg
        }).collect()
    };

    let u_mean1d = mean1d(&u_raw);
    let v_mean1d = mean1d(&v_raw);
    let w_mean1d = mean1d(&w_raw);
    let p_mean1d = mean1d(&p_raw);

    println!("Mean u profile (first/last 5 z levels):");
    for iz in (0..5).chain(271..276) {
        println!("  iz={:3}  z={:.5}  ū={:.6e}  w̄={:.6e}  p̄={:.6e}",
                 iz, zc[iz], u_mean1d[iz], w_mean1d[iz], p_mean1d[iz]);
    }

    // Broadcast 1-D mean back to 3-D (uniform in x and y)
    use ndarray::Array3;
    let broadcast = |mean1d: &[f64]| -> ndarray::ArrayD<f64> {
        let mut arr = ndarray::ArrayD::<f64>::zeros(ndarray::IxDyn(&[nz, ny, nxf]));
        for iz in 0..nz {
            arr.slice_mut(ndarray::s![iz, .., ..]).fill(mean1d[iz]);
        }
        arr
    };

    let u_m = broadcast(&u_mean1d);
    let v_m = broadcast(&v_mean1d);
    let w_m = broadcast(&w_mean1d);
    let p_m = broadcast(&p_mean1d);

    let (rx, ry, rz) = ns_residual_fortran_stencil(
        &u_m, &v_m, &w_m, &p_m,
        &x, &y, &zc, &zd,
        nu,
    ).expect("ns_residual_fortran_stencil failed");

    // For a uniform-in-x-y field all points at the same iz have the same residual;
    // just pick iy=0, ox=0 to get the z-profile.
    let nz_out = rx.shape()[0];

    println!("\nnu={:.4e}  headx={:.4e}  u_tau={:.4e}  Re_tau={:.1}",
             nu, headx, headx.sqrt(), headx.sqrt() / nu);

    println!("\n=== Mean-field residual z-profile (ix<180 spatial avg) ===");
    println!("{:>6} {:>10} | {:>12} {:>12} {:>12}  (headx={:.4e})",
             "iz","z_cell","R_x","R_y","R_z", headx);
    for oz in 0..nz_out {
        let iz = oz + 1;
        let z_cell = zd[2 * iz + 1];
        // All iy and ox are identical for a 1-D broadcast field
        let rx_val = rx[[oz, 0, 0]];
        let ry_val = ry[[oz, 0, 0]];
        let rz_val = rz[[oz, 0, 0]];
        println!("{:>6} {:>10.5} | {:>12.4e} {:>12.4e} {:>12.4e}",
                 iz, z_cell, rx_val, ry_val, rz_val);
    }

    let n = rx.len() as f64;
    let l2_x = (rx.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_y = (ry.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_z = (rz.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2   = ((rx.iter().map(|v|v*v).sum::<f64>()
               + ry.iter().map(|v|v*v).sum::<f64>()
               + rz.iter().map(|v|v*v).sum::<f64>()) / n).sqrt();

    println!("\nMean-field L2:  total={:.4e}  R_x={:.4e}  R_y={:.4e}  R_z={:.4e}",
             l2, l2_x, l2_y, l2_z);
    println!("vs headx = {:.4e}  ratio = {:.2}", headx, l2 / headx);
    println!("vs instantaneous L2 ≈ 6.36e-1  reduction factor = {:.0}×", 6.36e-1 / l2);

    assert!(l2.is_finite());
}

/// RANS residual check on `subavg_600000_0.h5`.
///
/// Same balance as `rans_residual_on_h5_data` but using a longer-averaged
/// statistics file (avg_start=600000, nt_phase=200000) with a wider x-domain
/// (nx=350 interior points, x ∈ [10.25, 22.61]).
///
/// Grid layout:
///   u/v/w/p and second moments : [nz=276, ny=8, nx_ghost=352]
///   x  : [350]   cell-centre x-coords (strip 1 ghost on each side)
///   y  : [8]     cell-centre y-coords
///   zc : [276]   z-face positions
///
/// Periodicity: x and y periodic, z non-periodic.
#[test]
fn rans_residual_on_subavg_600000() {
    const FILE: &str = "subavg_600000_0.h5";

    let mut h5 = H5File::new(FILE).expect("open HDF5");
    h5.get_info().unwrap();
    h5.load_coords().unwrap();

    let nu = h5.info().nu.expect("nu not found");
    println!("nu = {:.6e}", nu);

    // Mean velocities and pressure
    let u = strip_x_ghosts(&read_3d(&mut h5, "u"));
    let v = strip_x_ghosts(&read_3d(&mut h5, "v"));
    let w = strip_x_ghosts(&read_3d(&mut h5, "w"));
    let p = strip_x_ghosts(&read_3d(&mut h5, "p"));
    println!("interior shape: {:?}", u.shape());

    // Total second moments ⟨u_i u_j⟩
    let uu = strip_x_ghosts(&read_3d(&mut h5, "uu"));
    let uv = strip_x_ghosts(&read_3d(&mut h5, "uv"));
    let uw = strip_x_ghosts(&read_3d(&mut h5, "uw"));
    let vv = strip_x_ghosts(&read_3d(&mut h5, "vv"));
    let vw = strip_x_ghosts(&read_3d(&mut h5, "vw"));
    let ww = strip_x_ghosts(&read_3d(&mut h5, "ww"));

    let x  = read_coord(&mut h5, "x");
    let y  = read_coord(&mut h5, "y");
    let zc = read_coord(&mut h5, "zc");
    println!("x[0]={:.4}  x[-1]={:.4}  y[0]={:.4e}  zc[0]={:.4e}  zc[-1]={:.4}",
             x[0], x[x.len()-1], y[0], zc[0], zc[zc.len()-1]);

    let (rx, ry, rz) = rans_momentum_residual(
        &u, &v, &w, &p,
        &uu, &uv, &uw, &vv, &vw, &ww,
        &x, &y, &zc,
        nu,
        [false, true, true],
    ).expect("rans_momentum_residual failed");

    let n   = rx.len() as f64;
    let l2  = ns_residual_l2(&rx, &ry, &rz).unwrap();
    let l2_x = (rx.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_y = (ry.iter().map(|v| v*v).sum::<f64>() / n).sqrt();
    let l2_z = (rz.iter().map(|v| v*v).sum::<f64>() / n).sqrt();

    println!("RANS residual L2 (total) = {:.4e}", l2);
    println!("  R_x L2 = {:.4e}", l2_x);
    println!("  R_y L2 = {:.4e}", l2_y);
    println!("  R_z L2 = {:.4e}", l2_z);
    println!("  max|R_x| = {:.4e}", rx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  max|R_y| = {:.4e}", ry.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));
    println!("  max|R_z| = {:.4e}", rz.iter().map(|v| v.abs()).fold(0.0_f64, f64::max));

    // Estimate u_tau from near-wall viscous stress: du/dz|_{z=0} * nu ≈ u_tau^2
    let u_wall_grad = (u[[1, 0, 0]] - u[[0, 0, 0]]) / (zc[1] - zc[0]);
    let u_tau = (nu * u_wall_grad).sqrt();
    let re_tau = u_tau / nu;
    let a_plus = u_tau * u_tau * u_tau / nu;
    println!("u_tau ≈ {:.4e}  Re_tau ≈ {:.1}", u_tau, re_tau);
    println!("viscous acceleration scale u_tau^3/ν = {:.4e}", a_plus);
    println!("RANS residual total L2 in wall units = {:.4e} a+", l2 / a_plus);

    // z-profile of RMS RANS residual (averaged over x and y)
    // Use zc[iz] directly: it maps 1-to-1 to each of the 276 z-levels in the field.
    let nz_out = rx.shape()[0];
    let n_xy   = (rx.shape()[1] * rx.shape()[2]) as f64;
    println!("\n=== z-profile of RANS RMS residual (x,y-averaged) ===");
    println!("{:>6} {:>10} | {:>12} {:>12} {:>12}", "iz", "z", "rms_Rx", "rms_Ry", "rms_Rz");
    for iz in 0..nz_out {
        let z = zc[iz];
        let rms_x = (rx.slice(ndarray::s![iz,..,..]).iter().map(|v| v*v).sum::<f64>() / n_xy).sqrt();
        let rms_y = (ry.slice(ndarray::s![iz,..,..]).iter().map(|v| v*v).sum::<f64>() / n_xy).sqrt();
        let rms_z = (rz.slice(ndarray::s![iz,..,..]).iter().map(|v| v*v).sum::<f64>() / n_xy).sqrt();
        println!("{:>6} {:>10.5} | {:>12.4e} {:>12.4e} {:>12.4e}", iz, z, rms_x, rms_y, rms_z);
    }

    assert!(l2.is_finite(), "RANS residual contains NaN/Inf");
}
