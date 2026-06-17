/// Generate an XDMF file for visualization in Paraview.
///
/// Mirrors Python `hdfView._output()`.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub struct VarSpec {
    pub name:       String,
    pub full_shape: [usize; 3],   // [nz_total, ny_total, nx_total]
    pub blockz:     [usize; 3],   // [start, stride, count]
    pub blocky:     [usize; 3],
    pub blockx:     [usize; 3],
}

/// Write an XDMF file describing a 3-D rectilinear grid.
///
/// # Arguments
/// * `out_path` – output path (`.xdmf` appended if missing)
/// * `hdf5_path` – absolute path to the HDF5 file referenced in the XDMF
/// * `varx/y/z`  – coordinate dataset names in the HDF5 (e.g. "x", "xc")
/// * `sx/sy/sz`  – total sizes of the coordinate arrays
/// * `blockx/y/z` – `[start, stride, count]` for each dimension
/// * `vars`      – per-variable shape and hyperslab specs
pub fn write_xdmf(
    out_path: &Path,
    hdf5_path: &str,
    varx: &str,
    vary: &str,
    varz: &str,
    sx: usize,
    sy: usize,
    sz: usize,
    blockx: [usize; 3],
    blocky: [usize; 3],
    blockz: [usize; 3],
    vars: &[VarSpec],
) -> std::io::Result<()> {
    let final_path: PathBuf = if out_path.extension().and_then(|e| e.to_str()) == Some("xdmf") {
        out_path.to_path_buf()
    } else {
        out_path.with_extension("xdmf")
    };

    let file = File::create(&final_path)?;
    let mut w = BufWriter::new(file);

    let [bx_start, bx_stride, bx_count] = blockx;
    let [by_start, by_stride, by_count] = blocky;
    let [bz_start, bz_stride, bz_count] = blockz;

    writeln!(w, "<?xml version=\"1.0\" ?>")?;
    writeln!(w, "<Xdmf Version=\"2.0\">")?;
    writeln!(w, "  <Domain>")?;
    writeln!(w, "    <Grid Name=\"Structured Grid\" GridType=\"Uniform\">")?;
    writeln!(w,
        "      <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"{} {} {}\"/>",
        bz_count, by_count, bx_count
    )?;
    writeln!(w, "      <Geometry GeometryType=\"VXVYVZ\">")?;

    // ── X coordinate ──────────────────────────────────────────────────────────
    write_coord_item(&mut w, bx_count, bx_start, bx_stride, bx_count, sx, hdf5_path, varx)?;
    // ── Y coordinate ──────────────────────────────────────────────────────────
    write_coord_item(&mut w, by_count, by_start, by_stride, by_count, sy, hdf5_path, vary)?;
    // ── Z coordinate ──────────────────────────────────────────────────────────
    write_coord_item(&mut w, bz_count, bz_start, bz_stride, bz_count, sz, hdf5_path, varz)?;

    writeln!(w, "      </Geometry>")?;

    // ── Variables ─────────────────────────────────────────────────────────────
    for var in vars {
        let [vbz_s, vbz_st, vbz_c] = var.blockz;
        let [vby_s, vby_st, vby_c] = var.blocky;
        let [vbx_s, vbx_st, vbx_c] = var.blockx;
        let [fsz, fsy, fsx] = var.full_shape;

        writeln!(w,
            "      <Attribute Name=\"{}\" AttributeType=\"Scalar\" Center=\"Node\">",
            var.name
        )?;
        writeln!(w,
            "        <DataItem ItemType=\"HyperSlab\" Dimensions=\"{} {} {}\">",
            vbz_c, vby_c, vbx_c
        )?;
        writeln!(w, "          <DataItem Dimensions=\"3 3\" Format=\"XML\">")?;
        writeln!(w,
            "                    {} {} {}\n                    {} {} {}\n                    {} {} {}",
            vbz_s, vby_s, vbx_s,
            vbz_st, vby_st, vbx_st,
            vbz_c, vby_c, vbx_c
        )?;
        writeln!(w, "          </DataItem>")?;
        writeln!(w,
            "          <DataItem Dimensions=\"{} {} {}\" Precision=\"8\" Format=\"HDF\">{}:/{}</DataItem>",
            fsz, fsy, fsx, hdf5_path, var.name
        )?;
        writeln!(w, "        </DataItem>")?;
        writeln!(w, "      </Attribute>")?;
    }

    writeln!(w, "    </Grid>")?;
    writeln!(w, "  </Domain>")?;
    writeln!(w, "</Xdmf>")?;

    println!("XDMF written: {}", final_path.display());
    Ok(())
}

fn write_coord_item(
    w: &mut BufWriter<File>,
    dim: usize,
    start: usize,
    stride: usize,
    count: usize,
    total: usize,
    hdf5_path: &str,
    varname: &str,
) -> std::io::Result<()> {
    writeln!(w, "        <DataItem ItemType=\"HyperSlab\" Dimensions=\"{}\">", dim)?;
    writeln!(w, "          <DataItem Dimensions=\"3 1\" Format=\"XML\">")?;
    writeln!(w,
        "                    {}\n                    {}\n                    {}",
        start, stride, count
    )?;
    writeln!(w, "          </DataItem>")?;
    writeln!(w,
        "          <DataItem Dimensions=\"{}\" Precision=\"8\" Format=\"HDF\">{}:/{}</DataItem>",
        total, hdf5_path, varname
    )?;
    writeln!(w, "        </DataItem>")?;
    Ok(())
}
