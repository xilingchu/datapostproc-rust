// src/main.rs
use clap::{Args, Parser, Subcommand};

use datapostproc_rust::data::H5File;
use datapostproc_rust::hdf5::{Block, BlockValue, H5Data};
use datapostproc_rust::math::avg::avg_to_profile;
use datapostproc_rust::math::wall::{friction_velocity, wall_shear_stress};
use datapostproc_rust::output::dat::write_dat;
use datapostproc_rust::output::normalize::Profiles;
use datapostproc_rust::output::xdmf::{write_xdmf, VarSpec};

use ndarray::{Array1, Ix1};
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

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Output(args) => run_output(args),
        Command::Hdfview(args) => run_hdfview(args),
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
