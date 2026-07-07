//! Integration test: spanwise spectra / two-point correlation on a real
//! instantaneous DNS snapshot.  Skips (passes trivially) if the data file
//! is not present.

use datapostproc_rust::data::H5File;
use datapostproc_rust::hdf5::H5Data;
use datapostproc_rust::math::spectrum::SpanwiseSpectrum;
use ndarray::{Array1, ArrayD, Axis, Ix1};
use std::path::Path;

const SNAPSHOT: &str = "inst_400000_u.h5";

#[test]
fn spanwise_spectrum_on_real_snapshot() {
    let path = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), SNAPSHOT);
    if !Path::new(&path).exists() {
        eprintln!("skipping: {SNAPSHOT} not found");
        return;
    }

    let mut h5 = H5File::new(&path).expect("open");
    h5.get_info().expect("info");
    h5.load_coords().expect("coords");

    let read_coord = |h5: &H5File, name: &str| -> Array1<f64> {
        match h5.coord(name).unwrap().read_data::<f64>().unwrap() {
            H5Data::Array(a) => a.into_dimensionality::<Ix1>().unwrap(),
            H5Data::Scalar(_) => panic!("coord '{name}' is scalar"),
        }
    };
    let y = read_coord(&h5, "y");
    let x = read_coord(&h5, "x");
    let zc = read_coord(&h5, "zc");
    let ny = y.len();
    let nxc = x.len();
    let dy = y[1] - y[0];
    let ly = dy * ny as f64;

    h5.add_dataset("u", None).expect("dataset u");
    let raw: ArrayD<f64> = match h5.dataset("u").unwrap().read_data::<f64>().unwrap() {
        H5Data::Array(a) => a,
        H5Data::Scalar(_) => panic!("'u' is scalar"),
    };
    let nxf = raw.shape()[2];
    let u = match nxf - nxc {
        0 => raw,
        1 => raw.slice_axis(Axis(2), (0..nxc).into()).to_owned(),
        2 => raw.slice_axis(Axis(2), (1..nxc + 1).into()).to_owned(),
        _ => panic!("x extent mismatch"),
    };
    let nz = u.shape()[0];

    let mut spec = SpanwiseSpectrum::new(nz, ny, ly).expect("new");
    spec.accumulate(&u).expect("accumulate");

    // ── Parseval: spectrum-reconstructed variance == direct variance ─────────
    let var = spec.variance().expect("variance");
    let iz_test = (0..nz)
        .min_by(|&a, &b| (zc[a] - 0.2).abs().partial_cmp(&(zc[b] - 0.2).abs()).unwrap())
        .unwrap();
    let mut direct = 0.0;
    for ix in 0..nxc {
        let line: Vec<f64> = (0..ny).map(|j| u[[iz_test, j, ix]]).collect();
        let mean = line.iter().sum::<f64>() / ny as f64;
        direct += line.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ny as f64;
    }
    direct /= nxc as f64;
    let rel = (var[iz_test] - direct).abs() / direct;
    println!("z = {:.4}: var(spec) = {:.6e}  var(direct) = {:.6e}  rel = {rel:.2e}",
             zc[iz_test], var[iz_test], direct);
    assert!(rel < 1e-10, "Parseval violated: rel = {rel:.2e}");

    // ── spectrum: non-negative, mean removed (k=0 mode ≈ 0) ──────────────────
    let (k, e) = spec.energy_spectrum().expect("spectrum");
    assert!(k[0] == 0.0 && k[1] > 0.0);
    for iz in 0..nz {
        assert!(e[[iz, 0]].abs() < 1e-20, "k=0 mode not removed at z index {iz}");
        for m in 0..e.shape()[1] {
            assert!(e[[iz, m]] >= 0.0 && e[[iz, m]].is_finite());
        }
    }

    // ── correlation: rho(0) = 1, |rho| <= 1 (+ tolerance), finite ────────────
    let (r, rho) = spec.correlation().expect("correlation");
    assert!(r[0] == 0.0);
    assert!((r[r.len() - 1] - ly / 2.0).abs() < 1e-12);
    for iz in 0..nz {
        assert!((rho[[iz, 0]] - 1.0).abs() < 1e-12, "rho(0) != 1 at z index {iz}");
        for j in 0..rho.shape()[1] {
            assert!(rho[[iz, j]].is_finite() && rho[[iz, j]].abs() <= 1.0 + 1e-12);
        }
    }

    // ── physical sanity: turbulence decorrelates across the half-span ────────
    let nsep = rho.shape()[1];
    let tail = rho[[iz_test, nsep - 1]].abs();
    println!("rho at max separation (z = {:.4}): {tail:.3e}", zc[iz_test]);
    assert!(tail < 0.5, "no decorrelation across half the span: {tail}");
}
