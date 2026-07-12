// src/main.rs
use clap::{Args, Parser, Subcommand};

use datapostproc_rust::data::H5File;
use datapostproc_rust::hdf5::{Block, BlockValue, H5Data};
use datapostproc_rust::math::avg::avg_to_profile;
use datapostproc_rust::math::fik::{
    fik_average, fik_decomposition, fik_decomposition_planes, FikDecomposition,
};
use datapostproc_rust::math::rd::rd_decomposition;
use datapostproc_rust::math::spectrum::SpanwiseSpectrum;
use datapostproc_rust::math::wall::{friction_velocity, wall_shear_stress};
use datapostproc_rust::output::dat::write_dat;
use datapostproc_rust::output::normalize::Profiles;
use datapostproc_rust::output::xdmf::{write_xdmf, VarSpec};

use ndarray::{s, Array1, ArrayD, Axis, Ix1};
use std::collections::HashMap;
use std::path::Path;

// ─── CLI definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "datapostproc", about = "DNS data post-processor")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Average and write statistics to a .dat file.
    Output(OutputArgs),
    /// Generate an XDMF file for Paraview.
    Hdfview(HdfviewArgs),
    /// Spanwise energy spectra and two-point correlations from
    /// instantaneous snapshots.
    Spectra(SpectraArgs),
    /// FIK skin-friction decomposition from a subavg HDF5 file.
    Fik(FikArgs),
    /// Renard–Deck (energy-budget) skin-friction decomposition from a
    /// subavg HDF5 file.
    Rd(RdArgs),
}

/// Hyperslab [start stride count block] for output (4 values, HDF5 convention).
#[derive(Args, Clone)]
struct BlockArgs {
    /// x-direction hyperslab: start stride count block
    #[arg(short = 'x', long, num_args = 4, value_name = "INT")]
    blockx: Option<Vec<usize>>,
    /// y-direction hyperslab: start stride count block
    #[arg(short = 'y', long, num_args = 4, value_name = "INT")]
    blocky: Option<Vec<usize>>,
    /// z-direction hyperslab: start stride count block
    #[arg(short = 'z', long, num_args = 4, value_name = "INT")]
    blockz: Option<Vec<usize>>,
}

/// Hyperslab [start stride count] for hdfview (3 values, XDMF convention).
#[derive(Args, Clone)]
struct ViewBlockArgs {
    /// x-direction hyperslab: start stride count
    #[arg(short = 'x', long, num_args = 3, value_name = "INT")]
    blockx: Option<Vec<usize>>,
    /// y-direction hyperslab: start stride count
    #[arg(short = 'y', long, num_args = 3, value_name = "INT")]
    blocky: Option<Vec<usize>>,
    /// z-direction hyperslab: start stride count
    #[arg(short = 'z', long, num_args = 3, value_name = "INT")]
    blockz: Option<Vec<usize>>,
}

#[derive(Args)]
struct OutputArgs {
    /// Input HDF5 file.
    #[arg(short, long, value_name = "FILE")]
    file: String,
    /// Output .dat file (default: output.dat).
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,
    /// Variables to process.
    #[arg(short, long, num_args = 1.., value_name = "VAR")]
    variables: Vec<String>,
    /// Wall-normal direction: x, y, or z.
    #[arg(short, long, value_name = "DIR")]
    dire: String,
    /// Normalize to wall units (u+, z+, …).
    #[arg(short, long, default_value_t = false)]
    normalize: bool,
    /// Separate file containing u for computing uτ (default: same as --file).
    #[arg(long, value_name = "FILE")]
    uout: Option<String>,
    #[command(flatten)]
    block: BlockArgs,
}

#[derive(Args)]
struct HdfviewArgs {
    /// Input HDF5 file.
    #[arg(short, long, value_name = "FILE")]
    file: String,
    /// Output .xdmf file (default: same stem as input).
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,
    /// Variables to include.
    #[arg(short, long, num_args = 1.., value_name = "VAR")]
    variables: Vec<String>,
    #[command(flatten)]
    block: ViewBlockArgs,
}

#[derive(Args)]
struct SpectraArgs {
    /// Instantaneous snapshot HDF5 files; all are ensemble-averaged together.
    #[arg(short, long, num_args = 1.., value_name = "FILE")]
    files: Vec<String>,
    /// Dataset name to analyse (u, v, w, p, …).
    #[arg(short, long, default_value = "u")]
    variable: String,
    /// Output stem: writes <stem>_spec.dat and <stem>_corr.dat.
    #[arg(short, long, default_value = "spectra")]
    output: String,
    /// x-index range START END (0-based, half-open, indices of the x
    /// coordinate array) to average over; default: full streamwise extent.
    #[arg(long, num_args = 2, value_name = "INT")]
    xrange: Option<Vec<usize>>,
    /// Wall-normal locations (physical z, wall at 0) written as columns;
    /// each is snapped to the nearest grid point.
    #[arg(long, value_delimiter = ',', default_value = "0.05,0.1,0.2,0.5,1.0")]
    zloc: Vec<f64>,
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Output(args) => run_output(args),
        Command::Hdfview(args) => run_hdfview(args),
        Command::Spectra(args) => run_spectra(args),
        Command::Fik(args) => run_fik(args),
        Command::Rd(args) => run_rd(args),
    }
}

// ─── output sub-command ───────────────────────────────────────────────────────

fn run_output(args: OutputArgs) {
    // Wall-normal axis index (DNS layout: dim-0 = z, dim-1 = y, dim-2 = x)
    let wall_axis = dire_to_axis(&args.dire);

    // Optionally build a Block for hyperslab reading
    let block = build_block(&args.block);

    // Open and initialise the HDF5 file
    let var_refs: Vec<&str> = args.variables.iter().map(String::as_str).collect();
    let mut h5 = H5File::new(&args.file).expect("Failed to open HDF5 file");
    h5.init(&var_refs).expect("Failed to initialise H5File");

    // ── Average each variable to a 1-D wall-normal profile ───────────────────
    let mut profiles: HashMap<String, Array1<f64>> = HashMap::new();

    for var in &args.variables {
        let arr = {
            let data = h5.dataset(var)
                .unwrap_or_else(|| panic!("Variable '{}' not found", var));
            match data.read_data::<f64>().expect("read failed") {
                H5Data::Array(a) => a,
                H5Data::Scalar(_) => panic!("'{}' is a scalar – cannot average", var),
            }
        };
        let profile = avg_to_profile(&arr, wall_axis)
            .expect("averaging failed")
            .into_dimensionality::<Ix1>()
            .expect("dimension error");
        profiles.insert(var.clone(), profile);
    }

    // ── Wall-normal coordinate ────────────────────────────────────────────────
    let coord_arr: Array1<f64> = {
        let data = h5.coord(&args.dire)
            .unwrap_or_else(|| panic!("Coordinate '{}' not found", args.dire));
        match data.read_data::<f64>().expect("coord read failed") {
            H5Data::Array(a) => a.into_dimensionality::<Ix1>().expect("dim error"),
            H5Data::Scalar(_) => panic!("Coordinate '{}' is scalar", args.dire),
        }
    };
    profiles.insert(args.dire.clone(), coord_arr.clone());

    // ── Read u for τ_w computation ────────────────────────────────────────────
    let nu = h5.info().nu.expect("'nu' not found in HDF5 file");

    let u_profile: Array1<f64> = if profiles.contains_key("u") {
        profiles["u"].clone()
    } else {
        // u not in the variable list – load it separately
        let u_path = args.uout.as_deref().unwrap_or(&args.file);
        let mut u_h5 = H5File::new(u_path).expect("Failed to open u file");
        u_h5.add_dataset("u", block.clone()).expect("Cannot open 'u' dataset");
        let arr = {
            let data = u_h5.dataset("u").expect("'u' not found");
            match data.read_data::<f64>().expect("read u failed") {
                H5Data::Array(a) => a,
                H5Data::Scalar(_) => panic!("'u' is scalar"),
            }
        };
        avg_to_profile(&arr, wall_axis)
            .expect("avg u failed")
            .into_dimensionality::<Ix1>()
            .expect("dim error")
    };

    let tau = wall_shear_stress(
        &u_profile.clone().into_dyn(),
        &coord_arr.clone().into_dyn(),
        nu,
    ).expect("τ_w computation failed");
    let utau = friction_velocity(tau);
    // In non-dimensional DNS: Re_τ = uτ / ν
    let ret = utau / nu;

    // ── Optional normalization ────────────────────────────────────────────────
    let final_profiles = if args.normalize {
        let p = Profiles::new(&args.dire, nu, tau, profiles)
            .expect("Profiles::new failed");
        p.to_wall_units().expect("normalization failed")
    } else {
        profiles
    };

    // ── Write .dat ────────────────────────────────────────────────────────────
    let output_path = args.output.unwrap_or_else(|| "output".to_string());
    write_dat(Path::new(&output_path), &args.dire, ret, &final_profiles)
        .expect("Failed to write .dat file");
}

// ─── hdfview sub-command ──────────────────────────────────────────────────────

fn run_hdfview(args: HdfviewArgs) {
    let var_refs: Vec<&str> = args.variables.iter().map(String::as_str).collect();

    let mut h5 = H5File::new(&args.file).expect("Failed to open HDF5 file");
    h5.get_info().expect("Failed to read DNS info");
    h5.load_coords().expect("Failed to load coordinates");
    for v in &var_refs {
        h5.add_dataset(v, None)
            .unwrap_or_else(|_| eprintln!("Warning: dataset '{}' not found, skipping", v));
    }

    let info = h5.info();
    let sx = info.nx.expect("nx not available") as usize;
    let sy = info.ny.expect("ny not available") as usize;
    let sz = info.nz.expect("nz not available") as usize;

    // Find coordinate names actually present in the file
    let varx = find_coord(&h5, 'x').expect("x coordinate not found (tried 'x', 'xc')");
    let vary = find_coord(&h5, 'y').expect("y coordinate not found (tried 'y', 'yc')");
    let varz = find_coord(&h5, 'z').expect("z coordinate not found (tried 'z', 'zc')");

    // Default blocks: full extent
    let blockx = view_block(&args.block.blockx, sx);
    let blocky = view_block(&args.block.blocky, sy);
    let blockz = view_block(&args.block.blockz, sz);

    // Build per-variable specs (with periodicity detection)
    let var_specs: Vec<VarSpec> = args.variables.iter().filter_map(|var| {
        let shape = h5.dataset(var)?.shape();
        if shape.len() < 3 { return None; }

        // Periodicity: ghost layers add 2 to y and/or x
        let y_per = usize::from(shape[1] == sy + 2);
        let x_per = usize::from(shape[2] == sx + 2);

        let mut vbx = blockx; vbx[0] += x_per;
        let mut vby = blocky; vby[0] += y_per;

        Some(VarSpec {
            name: var.clone(),
            full_shape: [shape[0], shape[1], shape[2]],
            blockz,
            blocky: vby,
            blockx: vbx,
        })
    }).collect();

    // Use absolute path so Paraview can find the HDF5 regardless of working dir
    let hdf5_abs = std::fs::canonicalize(&args.file)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| args.file.clone());

    let output_path = args.output.unwrap_or_else(|| args.file.clone());
    write_xdmf(
        Path::new(&output_path),
        &hdf5_abs,
        &varx, &vary, &varz,
        sx, sy, sz,
        blockx, blocky, blockz,
        &var_specs,
    ).expect("Failed to write XDMF file");
}

// ─── spectra sub-command ──────────────────────────────────────────────────────

fn run_spectra(args: SpectraArgs) {
    assert!(!args.files.is_empty(), "need at least one snapshot file");

    let read_coord = |h5: &H5File, name: &str| -> Array1<f64> {
        match h5
            .coord(name)
            .unwrap_or_else(|| panic!("coord '{name}' not found"))
            .read_data::<f64>()
            .unwrap()
        {
            H5Data::Array(a) => a.into_dimensionality::<Ix1>().unwrap(),
            H5Data::Scalar(_) => panic!("coord '{name}' is scalar"),
        }
    };

    // ── coordinates / metadata from the first snapshot ────────────────────────
    let (y, zc, x, nu) = {
        let mut h5 = H5File::new(&args.files[0]).expect("failed to open first file");
        h5.get_info().expect("failed to read DNS info");
        h5.load_coords().expect("failed to load coordinates");
        let nu = h5.info().nu.expect("'nu' not in HDF5 file");
        (
            read_coord(&h5, "y"),
            read_coord(&h5, "zc"),
            read_coord(&h5, "x"),
            nu,
        )
    };
    let ny = y.len();
    let nxc = x.len();

    // spanwise grid must be uniform (plain FFT, no windowing)
    let dy = y[1] - y[0];
    for j in 1..ny {
        assert!(
            ((y[j] - y[j - 1]) - dy).abs() < 1e-8 * dy,
            "spanwise grid is not uniform at j={j}"
        );
    }
    let ly = dy * ny as f64;

    let (x0, x1) = match &args.xrange {
        Some(v) => (v[0], v[1]),
        None => (0, nxc),
    };
    assert!(x0 < x1 && x1 <= nxc, "invalid --xrange {x0} {x1} (nx = {nxc})");

    // ── accumulate all snapshots ──────────────────────────────────────────────
    let mut spec: Option<SpanwiseSpectrum> = None;

    for path in &args.files {
        let mut h5 = H5File::new(path).expect("failed to open HDF5 file");
        h5.add_dataset(&args.variable, None)
            .unwrap_or_else(|e| panic!("dataset '{}' in {path}: {e}", args.variable));
        let raw = match h5
            .dataset(&args.variable)
            .unwrap()
            .read_data::<f64>()
            .unwrap()
        {
            H5Data::Array(a) => a,
            H5Data::Scalar(_) => panic!("'{}' is scalar", args.variable),
        };

        // Align the streamwise extent with the x coordinate array:
        //   nxf == nxc      already aligned
        //   nxf == nxc + 1  inst convention: one extra outflow column at the end
        //   nxf == nxc + 2  subavg convention: one ghost column on each side
        let nxf = raw.shape()[2];
        let aligned: ArrayD<f64> = match nxf.checked_sub(nxc) {
            Some(0) => raw,
            Some(1) => raw.slice_axis(Axis(2), (0..nxc).into()).to_owned(),
            Some(2) => raw.slice_axis(Axis(2), (1..nxc + 1).into()).to_owned(),
            _ => panic!("field x extent {nxf} incompatible with x coord length {nxc}"),
        };
        let window = aligned.slice_axis(Axis(2), (x0..x1).into()).to_owned();

        let acc = spec.get_or_insert_with(|| {
            SpanwiseSpectrum::new(window.shape()[0], ny, ly).expect("SpanwiseSpectrum::new")
        });
        acc.accumulate(&window).expect("accumulate failed");
        eprintln!("accumulated {path}");
    }

    let spec = spec.unwrap();
    let nz = zc.len();

    // ── map requested z locations to grid points (deduplicated) ───────────────
    let mut z_indices: Vec<usize> = Vec::new();
    for &zq in &args.zloc {
        let iz = (0..nz)
            .min_by(|&a, &b| {
                (zc[a] - zq).abs().partial_cmp(&(zc[b] - zq).abs()).unwrap()
            })
            .unwrap();
        if !z_indices.contains(&iz) {
            z_indices.push(iz);
        }
    }

    // ── write outputs ─────────────────────────────────────────────────────────
    let stem = args.output.strip_suffix(".dat").unwrap_or(&args.output);
    let ret = 1.0 / nu;

    let (k, e) = spec.energy_spectrum().expect("energy_spectrum");
    let mut cols: HashMap<String, Array1<f64>> = HashMap::new();
    cols.insert("yk".into(), k);
    for &iz in &z_indices {
        cols.insert(
            format!("e{}_{:.3}", args.variable, zc[iz]),
            e.row(iz).to_owned(),
        );
    }
    let spec_path = format!("{stem}_spec.dat");
    write_dat(Path::new(&spec_path), "yk", ret, &cols).expect("failed to write spectrum");
    eprintln!("wrote {spec_path}");

    let (r, rho) = spec.correlation().expect("correlation");
    let mut cols: HashMap<String, Array1<f64>> = HashMap::new();
    cols.insert("ydel".into(), r);
    for &iz in &z_indices {
        cols.insert(
            format!("r{}_{:.3}", args.variable, zc[iz]),
            rho.row(iz).to_owned(),
        );
    }
    let corr_path = format!("{stem}_corr.dat");
    write_dat(Path::new(&corr_path), "ydel", ret, &cols).expect("failed to write correlation");
    eprintln!("wrote {corr_path}");

    eprintln!(
        "ensemble: {} snapshot(s) x {} x-columns = {} y-lines per z",
        args.files.len(),
        x1 - x0,
        spec.samples()
    );
}

// ─── fik sub-command ──────────────────────────────────────────────────────────

#[derive(Args)]
struct FikArgs {
    /// Input subavg HDF5 file (raw: uu/uw are total second moments).
    #[arg(short, long, value_name = "FILE")]
    file: String,
    /// Output .dat file (default: fik.dat).
    #[arg(short, long, value_name = "FILE", default_value = "fik.dat")]
    output: String,
    /// Half-channel height h (default: 1.0).
    #[arg(long, default_value_t = 1.0)]
    half_height: f64,
    /// Number of ghost cells to strip from each x-boundary (default: 1).
    #[arg(long, default_value_t = 1usize)]
    ghost_x: usize,
    /// Number of (post-ghost-strip) points to drop from the start of the
    /// x-range, e.g. to discard a corrupted lead-in.
    #[arg(long, default_value_t = 0usize)]
    trim_start: usize,
    /// Number of (post-ghost-strip) points to drop from the end of the
    /// x-range, e.g. to discard a corrupted tail.
    #[arg(long, default_value_t = 0usize)]
    trim_end: usize,
    /// Treat streamwise direction as periodic for derivative stencils.
    #[arg(long, default_value_t = false)]
    periodic_x: bool,
    /// Per-plane mode: one decomposition (and one output file `<output>_p<j>.dat`)
    /// per spanwise plane, with the spanwise terms cf_turb_z/cf_conv_z retained.
    #[arg(long, default_value_t = false)]
    per_plane: bool,
    /// Average over the given spanwise plane indices (comma-separated, e.g.
    /// `--planes 2,5`).  Exact by linearity; spanwise terms do NOT cancel for
    /// a subset of planes.  Writes a single output file.
    #[arg(long, value_delimiter = ',')]
    planes: Option<Vec<usize>>,
}

fn run_fik(args: FikArgs) {
    let mut h5 = H5File::new(&args.file).expect("failed to open HDF5 file");
    h5.get_info().expect("failed to read DNS info");
    h5.load_coords().expect("failed to load coordinates");

    let nu   = h5.info().nu.expect("'nu' not in HDF5 file");
    let re_b = 1.0 / nu;

    // Load 3-D fields helper
    let load = |h5: &mut H5File, name: &str| -> ArrayD<f64> {
        h5.add_dataset(name, None)
          .unwrap_or_else(|e| panic!("dataset '{name}': {e}"));
        match h5.dataset(name).unwrap().read_data::<f64>().unwrap() {
            H5Data::Array(a) => a,
            H5Data::Scalar(_) => panic!("'{name}' is scalar"),
        }
    };

    let u_raw  = load(&mut h5, "u");
    let v_raw  = load(&mut h5, "v");
    let w_raw  = load(&mut h5, "w");
    let p_raw  = load(&mut h5, "p");
    let uu_raw = load(&mut h5, "uu");
    let uw_raw = load(&mut h5, "uw");

    // Strip ghost cells along x (axis 2)
    let g = args.ghost_x;
    let nx_ghost = u_raw.shape()[2];
    let strip = |a: ArrayD<f64>| -> ArrayD<f64> {
        a.slice_axis(Axis(2), (g..nx_ghost - g).into()).to_owned()
    };
    let u  = strip(u_raw);
    let v  = strip(v_raw);
    let w  = strip(w_raw);
    let p  = strip(p_raw);
    let uu = strip(uu_raw);
    let uw = strip(uw_raw);

    // Coordinates
    let load_coord = |h5: &mut H5File, name: &str| -> Array1<f64> {
        match h5.coord(name)
            .unwrap_or_else(|| panic!("coord '{name}' not found"))
            .read_data::<f64>().unwrap()
        {
            H5Data::Array(a) => a.into_dimensionality::<Ix1>().unwrap(),
            H5Data::Scalar(_) => panic!("coord '{name}' is scalar"),
        }
    };
    let x_full  = load_coord(&mut h5, "x");
    let zc = load_coord(&mut h5, "zc");

    // Trim a corrupted lead-in / tail from the streamwise range.
    let nx_full = x_full.len();
    let lo = args.trim_start;
    let hi = nx_full - args.trim_end;
    assert!(lo < hi, "trim_start ({lo}) must be < nx - trim_end ({hi})");
    let trim = |a: ArrayD<f64>| -> ArrayD<f64> {
        a.slice_axis(Axis(2), (lo..hi).into()).to_owned()
    };
    let u  = trim(u);
    let v  = trim(v);
    let w  = trim(w);
    let p  = trim(p);
    let uu = trim(uu);
    let uw = trim(uw);
    let x  = x_full.slice(s![lo..hi]).to_owned();

    // Direct C_f from viscous sublayer: 2ν·u(iz=1) / zc[1]
    let iz1 = 1usize;
    // per spanwise plane j
    let cf_direct_plane = |u: &ArrayD<f64>, j: usize| -> Array1<f64> {
        u.slice(s![iz1, j, ..]).mapv(|uv| 2.0 * nu * uv / zc[iz1])
    };
    // spanwise-averaged
    let cf_direct_mean: Array1<f64> = {
        let u_ymean: Array1<f64> = u.slice(s![iz1, .., ..])
            .mean_axis(ndarray::Axis(0))
            .expect("mean over y");
        u_ymean.mapv(|uv| 2.0 * nu * uv / zc[iz1])
    };

    // Write one decomposition (+ direct C_f) to a .dat file
    let write_fik = |fik: &FikDecomposition, cf_direct: Array1<f64>, path: &str| {
        let mut cols: HashMap<String, Array1<f64>> = HashMap::new();
        cols.insert("x".into(),         x.clone());
        cols.insert("cf_direct".into(), cf_direct);
        cols.insert("cf_fik".into(),    fik.cf_total());
        cols.insert("cf_lam".into(),    fik.cf_laminar.clone());
        cols.insert("cf_center".into(), fik.cf_center.clone());
        cols.insert("cf_turb_x".into(), fik.cf_turb_x.clone());
        cols.insert("cf_turb_y".into(), fik.cf_turb_y.clone());
        cols.insert("cf_turb_z".into(), fik.cf_turb_z.clone());
        cols.insert("cf_conv_x".into(), fik.cf_conv_x.clone());
        cols.insert("cf_conv_y".into(), fik.cf_conv_y.clone());
        cols.insert("cf_conv_z".into(), fik.cf_conv_z.clone());
        cols.insert("cf_diff_x".into(), fik.cf_diff_x.clone());
        cols.insert("cf_diff_z".into(), fik.cf_diff_z.clone());
        cols.insert("cf_source".into(), fik.cf_source.clone());
        write_dat(Path::new(path), "x", re_b, &cols)
            .expect("failed to write .dat file");
        eprintln!("wrote {path}");
    };

    if args.per_plane || args.planes.is_some() {
        // ── per-plane mode: spanwise terms retained ──────────────────────────
        let uv = trim(strip(load(&mut h5, "uv")));
        let y  = load_coord(&mut h5, "y");
        let decs = fik_decomposition_planes(
            &u, &v, &w, &p, &uu, &uv, &uw,
            &x, &y, &zc,
            re_b, args.half_height, args.periodic_x,
        ).expect("fik_decomposition_planes failed");

        if let Some(planes) = &args.planes {
            // exact subset average over the selected planes
            for &j in planes {
                assert!(j < decs.len(), "plane index {j} out of range (ny={})", decs.len());
            }
            let sel: Vec<&FikDecomposition> = planes.iter().map(|&j| &decs[j]).collect();
            let fik = fik_average(&sel).expect("fik_average failed");
            let mut cf_dir = Array1::<f64>::zeros(x.len());
            for &j in planes {
                cf_dir = cf_dir + &cf_direct_plane(&u, j);
            }
            cf_dir /= planes.len() as f64;
            write_fik(&fik, cf_dir, &args.output);
        } else {
            let stem = args.output.strip_suffix(".dat").unwrap_or(&args.output).to_string();
            for (j, d) in decs.iter().enumerate() {
                write_fik(d, cf_direct_plane(&u, j), &format!("{stem}_p{j}.dat"));
            }
        }
    } else {
        // ── spanwise-averaged mode (default) ─────────────────────────────────
        let fik = fik_decomposition(
            &u, &v, &w, &p, &uu, &uw,
            &x, &zc,
            re_b,
            args.half_height,
            args.periodic_x,
        ).expect("fik_decomposition failed");
        write_fik(&fik, cf_direct_mean, &args.output);
    }
}

// ─── rd sub-command ───────────────────────────────────────────────────────────

#[derive(Args)]
struct RdArgs {
    /// Input subavg HDF5 file (raw: uu/uw are total second moments).
    #[arg(short, long, value_name = "FILE")]
    file: String,
    /// Output .dat file (default: rd.dat).
    #[arg(short, long, value_name = "FILE", default_value = "rd.dat")]
    output: String,
    /// Half-channel height h (default: 1.0).
    #[arg(long, default_value_t = 1.0)]
    half_height: f64,
    /// Reference velocity U_ref of the RD energy frame; C_f = 2τ_w/U_ref².
    /// Default 1.0 = global bulk velocity in DNS units (matches direct C_f).
    #[arg(long, default_value_t = 1.0)]
    u_ref: f64,
    /// Number of ghost cells to strip from each x-boundary (default: 1).
    #[arg(long, default_value_t = 1usize)]
    ghost_x: usize,
    /// Number of (post-ghost-strip) points to drop from the start of the
    /// x-range, e.g. to discard a corrupted lead-in.
    #[arg(long, default_value_t = 0usize)]
    trim_start: usize,
    /// Number of (post-ghost-strip) points to drop from the end of the
    /// x-range, e.g. to discard a corrupted tail.
    #[arg(long, default_value_t = 0usize)]
    trim_end: usize,
    /// Treat streamwise direction as periodic for derivative stencils.
    #[arg(long, default_value_t = false)]
    periodic_x: bool,
}

fn run_rd(args: RdArgs) {
    let mut h5 = H5File::new(&args.file).expect("failed to open HDF5 file");
    h5.get_info().expect("failed to read DNS info");
    h5.load_coords().expect("failed to load coordinates");

    let nu   = h5.info().nu.expect("'nu' not in HDF5 file");
    let re_b = 1.0 / nu;

    let load = |h5: &mut H5File, name: &str| -> ArrayD<f64> {
        h5.add_dataset(name, None)
          .unwrap_or_else(|e| panic!("dataset '{name}': {e}"));
        match h5.dataset(name).unwrap().read_data::<f64>().unwrap() {
            H5Data::Array(a) => a,
            H5Data::Scalar(_) => panic!("'{name}' is scalar"),
        }
    };

    let u_raw  = load(&mut h5, "u");
    let w_raw  = load(&mut h5, "w");
    let p_raw  = load(&mut h5, "p");
    let uu_raw = load(&mut h5, "uu");
    let uw_raw = load(&mut h5, "uw");

    // Strip ghost cells along x (axis 2), then trim a corrupted lead-in/tail.
    let g = args.ghost_x;
    let nx_ghost = u_raw.shape()[2];
    let load_coord = |h5: &mut H5File, name: &str| -> Array1<f64> {
        match h5.coord(name)
            .unwrap_or_else(|| panic!("coord '{name}' not found"))
            .read_data::<f64>().unwrap()
        {
            H5Data::Array(a) => a.into_dimensionality::<Ix1>().unwrap(),
            H5Data::Scalar(_) => panic!("coord '{name}' is scalar"),
        }
    };
    let x_full = load_coord(&mut h5, "x");
    let zc     = load_coord(&mut h5, "zc");

    let nx_full = x_full.len();
    let lo = args.trim_start;
    let hi = nx_full - args.trim_end;
    assert!(lo < hi, "trim_start ({lo}) must be < nx - trim_end ({hi})");

    let cut = |a: ArrayD<f64>| -> ArrayD<f64> {
        let stripped = a.slice_axis(Axis(2), (g..nx_ghost - g).into()).to_owned();
        stripped.slice_axis(Axis(2), (lo..hi).into()).to_owned()
    };
    let u  = cut(u_raw);
    let w  = cut(w_raw);
    let p  = cut(p_raw);
    let uu = cut(uu_raw);
    let uw = cut(uw_raw);
    let x  = x_full.slice(s![lo..hi]).to_owned();

    // Direct C_f from viscous sublayer: 2ν·u(iz=1) / zc[1]
    let iz1 = 1usize;
    let cf_direct: Array1<f64> = {
        let u_ymean: Array1<f64> = u.slice(s![iz1, .., ..])
            .mean_axis(ndarray::Axis(0))
            .expect("mean over y");
        u_ymean.mapv(|uv| 2.0 * nu * uv / zc[iz1])
    };

    let rd = rd_decomposition(
        &u, &w, &p, &uu, &uw,
        &x, &zc,
        re_b,
        args.half_height,
        args.u_ref,
        args.periodic_x,
    ).expect("rd_decomposition failed");

    let mut cols: HashMap<String, Array1<f64>> = HashMap::new();
    cols.insert("x".into(),         x.clone());
    cols.insert("cf_direct".into(), cf_direct);
    cols.insert("cf_rd".into(),     rd.cf_total());
    cols.insert("cf_diss".into(),   rd.cf_diss.clone());
    cols.insert("cf_prod".into(),   rd.cf_prod.clone());
    cols.insert("cf_growth".into(), rd.cf_growth());
    cols.insert("cf_conv_x".into(), rd.cf_conv_x.clone());
    cols.insert("cf_conv_y".into(), rd.cf_conv_y.clone());
    cols.insert("cf_turb_x".into(), rd.cf_turb_x.clone());
    cols.insert("cf_diff_x".into(), rd.cf_diff_x.clone());
    cols.insert("cf_source".into(), rd.cf_source.clone());
    cols.insert("cf_center".into(), rd.cf_center.clone());
    write_dat(Path::new(&args.output), "x", re_b, &cols)
        .expect("failed to write .dat file");
    eprintln!("wrote {}", args.output);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn dire_to_axis(dire: &str) -> usize {
    match dire.to_lowercase().as_str() {
        "z" => 0,
        "y" => 1,
        "x" => 2,
        d   => panic!("Unknown direction '{}' – expected x, y, or z", d),
    }
}

fn build_block(args: &BlockArgs) -> Option<Block> {
    if args.blockx.is_none() && args.blocky.is_none() && args.blockz.is_none() {
        return None;
    }
    let bv = |v: &Vec<usize>| BlockValue::new([v[0], v[1], v[2], v[3]]).expect("invalid block");
    let bz = args.blockz.as_ref().map(bv);
    let by = args.blocky.as_ref().map(bv);
    let bx = args.blockx.as_ref().map(bv);
    Some(Block::new(vec![bz, by, bx]))
}

/// Default [start, stride, count] = [0, 1, full_extent].
fn view_block(arg: &Option<Vec<usize>>, full: usize) -> [usize; 3] {
    match arg {
        Some(v) => [v[0], v[1], v[2]],
        None    => [0, 1, full],
    }
}

fn find_coord(h5: &H5File, axis: char) -> Option<String> {
    let plain = axis.to_string();
    if h5.coord(&plain).is_some() { return Some(plain); }
    let with_c = format!("{}c", axis);
    if h5.coord(&with_c).is_some() { return Some(with_c); }
    None
}
