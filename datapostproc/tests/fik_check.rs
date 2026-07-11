/// Integration test: FIK skin-friction decomposition on real DNS statistics.
///
/// File: subavg_600000_0.h5  (raw file — uu/uw are total second moments ⟨u_i u_j⟩)
/// Grid layout:
///   u/v/w/p/uu/uw : [nz=276, ny=8, nx_ghost=352]  (1 ghost on each side of x)
///   x             : [350]  cell-centre x-coords
///   zc            : [276]  wall-normal coords, zc[0]=0 (wall), zc[-1]=2 (top wall)
///
/// The FIK decomposition is compared with the direct wall-gradient C_f:
///   C_f_direct(x) = 2 * nu * mean_y(u[iz=1, :, ix]) / zc[1]
///
/// For a spatially developing channel after a blowing-control actuator we expect
/// the sum of all FIK terms to agree with the direct measurement to within the
/// finite-difference and averaging errors.

use datapostproc_rust::data::H5File;
use datapostproc_rust::hdf5::H5Data;
use datapostproc_rust::math::fik::{fik_average, fik_decomposition, fik_decomposition_planes};
use ndarray::{s, Array1, ArrayD, Axis, Ix1};

const FILE: &str = "subavg_600000_0.h5";

// ── helpers (same as ns_check.rs) ────────────────────────────────────────────

fn read_3d(h5: &mut H5File, name: &str) -> ArrayD<f64> {
    h5.add_dataset(name, None)
        .unwrap_or_else(|e| panic!("dataset '{name}': {e}"));
    match h5.dataset(name).unwrap().read_data::<f64>().unwrap() {
        H5Data::Array(a) => a,
        H5Data::Scalar(_) => panic!("'{name}' is scalar"),
    }
}

fn read_coord(h5: &mut H5File, name: &str) -> Array1<f64> {
    match h5.coord(name)
        .unwrap_or_else(|| panic!("coord '{name}' not found"))
        .read_data::<f64>()
        .unwrap()
    {
        H5Data::Array(a) => a.into_dimensionality::<Ix1>().unwrap(),
        H5Data::Scalar(_) => panic!("coord '{name}' is scalar"),
    }
}

/// Strip one ghost cell on each side of the x-axis (axis 2).
fn strip_x_ghosts(arr: &ArrayD<f64>) -> ArrayD<f64> {
    let nx_ghost = arr.shape()[2];
    arr.slice_axis(Axis(2), (1..nx_ghost - 1).into()).to_owned()
}

// ── test ─────────────────────────────────────────────────────────────────────

#[test]
fn fik_on_subavg_600000() {
    let mut h5 = H5File::new(FILE).expect("open HDF5");
    h5.get_info().unwrap();
    h5.load_coords().unwrap();

    let nu = h5.info().nu.expect("nu not found in file");
    println!("nu = {nu:.6e}   Re_b = {:.1}", 1.0 / nu);

    // ── load mean fields and total second moments ─────────────────────────────
    // uu, uw are TOTAL second moments: ⟨u·u⟩ = ū² + u'u', etc.
    // Reynolds stresses are extracted internally by fik_decomposition.
    let u  = strip_x_ghosts(&read_3d(&mut h5, "u"));
    let v  = strip_x_ghosts(&read_3d(&mut h5, "v"));
    let w  = strip_x_ghosts(&read_3d(&mut h5, "w"));
    let p  = strip_x_ghosts(&read_3d(&mut h5, "p"));
    let uu = strip_x_ghosts(&read_3d(&mut h5, "uu"));
    let uw = strip_x_ghosts(&read_3d(&mut h5, "uw"));

    println!("interior shape: {:?}", u.shape());

    // ── coordinates ──────────────────────────────────────────────────────────
    let x  = read_coord(&mut h5, "x");
    let zc = read_coord(&mut h5, "zc");
    println!("x  : [{:.4}, {:.4}]  nx={}",  x[0],  x[x.len()-1],  x.len());
    println!("zc : [{:.4e}, {:.4}]  nz={}", zc[0], zc[zc.len()-1], zc.len());

    let nx = x.len();
    let h  = 1.0_f64;          // half-channel height
    let re_b = 1.0 / nu;

    // ── direct wall C_f from viscous sublayer: C_f = 2ν u[iz=1] / zc[1] ─────
    // iz=0 is the wall (zc[0]=0, u=0); iz=1 is the first interior cell.
    let iz_wall = 1usize;
    let z_wall  = zc[iz_wall];
    // average over spanwise y (axis 1), extract iz_wall row
    let cf_direct: Array1<f64> = {
        let u_slice = u.slice(s![iz_wall, .., ..]);   // (ny, nx)
        let u_ymean: Array1<f64> = u_slice.mean_axis(ndarray::Axis(0))
            .expect("mean over y");
        u_ymean.mapv(|u_val| 2.0 * nu * u_val / z_wall)
    };

    println!("\nDirect C_f (viscous sublayer, iz={iz_wall}, z={z_wall:.4e}):");
    println!("  mean = {:.4e}  min = {:.4e}  max = {:.4e}",
             cf_direct.mean().unwrap(),
             cf_direct.iter().cloned().fold(f64::INFINITY, f64::min),
             cf_direct.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // ── FIK decomposition ────────────────────────────────────────────────────
    let fik = fik_decomposition(
        &u, &v, &w, &p, &uu, &uw,
        &x, &zc,
        re_b,
        h,
        false,  // domain is NOT periodic in x (convective-BC channel)
    ).expect("fik_decomposition failed");

    let cf_total   = fik.cf_total();
    let cf_turb    = fik.cf_turbulent();
    let cf_conv    = fik.cf_convection();

    // ── summary statistics ───────────────────────────────────────────────────
    let mean = |a: &Array1<f64>| a.mean().unwrap_or(f64::NAN);
    let amax = |a: &Array1<f64>| a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);

    println!("\nFIK decomposition (x-averaged values):");
    println!("  C_f^L     = {:.4e}  (6ũ/Re_b; 6/Re_b = {:.4e})", mean(&fik.cf_laminar), 6.0/re_b);
    println!("  C_f^A     = {:.4e}  (centreline shear, asymmetry)", mean(&fik.cf_center));
    println!("  C_f^T_x   = {:.4e}", mean(&fik.cf_turb_x));
    println!("  C_f^T_y   = {:.4e}  (standard FIK shear term)", mean(&fik.cf_turb_y));
    println!("  C_f^C_x   = {:.4e}", mean(&fik.cf_conv_x));
    println!("  C_f^C_y   = {:.4e}  (wall-normal mean convection)", mean(&fik.cf_conv_y));
    println!("  C_f^D     = {:.4e}", mean(&fik.cf_diffusion()));
    println!("  C_f^S     = {:.4e}  (pressure source)", mean(&fik.cf_source));
    println!("  ─────────────────────────────────────────");
    println!("  C_f (FIK) = {:.4e}", mean(&cf_total));
    println!("  C_f (direct) = {:.4e}", mean(&cf_direct));

    // ── per-x table (print every ~10th point) ────────────────────────────────
    let step = (nx / 20).max(1);
    println!("\n{:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
             "x", "Cf_direct", "Cf_FIK", "CfL", "CfTx", "CfTy", "CfCx", "CfCy", "CfS");
    for ix in (0..nx).step_by(step) {
        println!("{:8.4}  {:10.4e}  {:10.4e}  {:10.4e}  {:10.4e}  {:10.4e}  {:10.4e}  {:10.4e}  {:10.4e}",
                 x[ix],
                 cf_direct[ix],
                 cf_total[ix],
                 fik.cf_laminar[ix],
                 fik.cf_turb_x[ix],
                 fik.cf_turb_y[ix],
                 fik.cf_conv_x[ix],
                 fik.cf_conv_y[ix],
                 fik.cf_source[ix]);
    }

    // ── residual: difference between FIK total and direct ────────────────────
    let diff: Array1<f64> = &cf_total - &cf_direct;
    let rms_diff = (diff.iter().map(|v| v*v).sum::<f64>() / nx as f64).sqrt();
    let rel_rms  = rms_diff / mean(&cf_direct);

    println!("\nFIK vs direct: rms_diff = {rms_diff:.4e}  rel_rms = {rel_rms:.4e}");
    println!("  max|turb_x| = {:.4e}  max|conv_x| = {:.4e}  max|diff| = {:.4e}  max|source| = {:.4e}",
             amax(&fik.cf_turb_x), amax(&fik.cf_conv_x), amax(&fik.cf_diffusion()), amax(&fik.cf_source));

    // ── assertions ───────────────────────────────────────────────────────────
    assert!(cf_total.iter().all(|v| v.is_finite()), "FIK total contains NaN/Inf");
    assert!(cf_direct.iter().all(|v| v.is_finite()), "direct C_f contains NaN/Inf");

    // C_f^T_y should be the dominant turbulent contribution and positive
    assert!(mean(&fik.cf_turb_y) > 0.0, "cf_turb_y should be positive");
    // Laminar term: 6ũ/Re_b with ũ ≈ 1 (global constant flow rate)
    let cf_l_rel = (mean(&fik.cf_laminar) - 6.0/re_b).abs() / (6.0/re_b);
    assert!(cf_l_rel < 0.05, "cf_laminar deviates from 6/Re_b by {cf_l_rel:.2e}");
}

/// Per-plane FIK on the same file: each spanwise plane must reproduce its own
/// local wall C_f, and the all-plane average must recover the spanwise-averaged
/// result (spanwise terms cancel over the full period).
#[test]
fn fik_per_plane_on_subavg_600000() {
    let mut h5 = H5File::new(FILE).expect("open HDF5");
    h5.get_info().unwrap();
    h5.load_coords().unwrap();

    let nu = h5.info().nu.expect("nu not found in file");

    let u  = strip_x_ghosts(&read_3d(&mut h5, "u"));
    let v  = strip_x_ghosts(&read_3d(&mut h5, "v"));
    let w  = strip_x_ghosts(&read_3d(&mut h5, "w"));
    let p  = strip_x_ghosts(&read_3d(&mut h5, "p"));
    let uu = strip_x_ghosts(&read_3d(&mut h5, "uu"));
    let uv = strip_x_ghosts(&read_3d(&mut h5, "uv"));
    let uw = strip_x_ghosts(&read_3d(&mut h5, "uw"));

    let x  = read_coord(&mut h5, "x");
    let y  = read_coord(&mut h5, "y");
    let zc = read_coord(&mut h5, "zc");

    let nx = x.len();
    let ny = y.len();
    let re_b = 1.0 / nu;
    let h = 1.0_f64;
    let iz1 = 1usize;

    let decs = fik_decomposition_planes(
        &u, &v, &w, &p, &uu, &uv, &uw,
        &x, &y, &zc,
        re_b, h, false,
    ).expect("fik_decomposition_planes failed");
    assert_eq!(decs.len(), ny);

    let mean = |a: &Array1<f64>| a.mean().unwrap_or(f64::NAN);

    // per-plane totals vs per-plane direct C_f.
    // Single-plane statistics are noisier (~√ny fewer samples) and the
    // spanwise FD uses only ny=8 points, so the tolerance is looser than in
    // the spanwise-averaged test.
    println!("\nPer-plane FIK vs local direct C_f:");
    let mut worst = 0.0_f64;
    for (j, d) in decs.iter().enumerate() {
        let cf_dir: Array1<f64> = u.slice(s![iz1, j, ..])
            .mapv(|uv| 2.0 * nu * uv / zc[iz1]);
        let total = d.cf_total();
        let rms = (total.iter().zip(cf_dir.iter())
            .map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / nx as f64).sqrt();
        let rel = rms / mean(&cf_dir);
        println!("  plane {j}: mean_direct={:.4e}  mean_fik={:.4e}  rel_rms={rel:.2e}  \
                  turb_z={:.2e}  conv_z={:.2e}",
                 mean(&cf_dir), mean(&total), mean(&d.cf_turb_z), mean(&d.cf_conv_z));
        assert!(total.iter().all(|v| v.is_finite()), "plane {j}: NaN/Inf");
        worst = worst.max(rel);
    }
    assert!(worst < 0.15, "worst per-plane FIK vs direct rel_rms {worst:.2e} > 15%");

    // all-plane average must recover the spanwise-averaged decomposition
    let all: Vec<_> = decs.iter().collect();
    let avg_planes = fik_average(&all).expect("fik_average failed");
    let avg_span = fik_decomposition(&u, &v, &w, &p, &uu, &uw, &x, &zc, re_b, h, false)
        .expect("spanwise-averaged FIK failed");

    let t_planes = avg_planes.cf_total();
    let t_span   = avg_span.cf_total();
    let rms = (t_planes.iter().zip(t_span.iter())
        .map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / nx as f64).sqrt();
    let rel = rms / mean(&t_span);
    println!("\nAll-plane average vs spanwise-averaged: rel_rms = {rel:.2e}");
    println!("  residual spanwise terms: turb_z={:.2e}  conv_z={:.2e}",
             mean(&avg_planes.cf_turb_z), mean(&avg_planes.cf_conv_z));
    assert!(rel < 0.02, "all-plane average deviates from spanwise average: {rel:.2e}");
    // spanwise terms must cancel over the full period (up to FD error)
    assert!(mean(&avg_planes.cf_turb_z).abs() < 1e-4,
        "turb_z does not cancel over full period");
}
