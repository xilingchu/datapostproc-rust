/// Write wall-normal statistics to a formatted `.dat` text file.
///
/// Output format mirrors the Python `_output()` in `outputBase.py`:
///
/// ```text
/// Statistics of the data along with z, Ret=XXX
///             C1            C2            C3
///             zc            uu            vv
/// ────────────────────────────────────────────────────────────────────────────────
/// ────────────────────────────────────────────────────────────────────────────────
///       0.123456  1.234568e-01  ...
/// ```
///
/// Column ordering rules (same as Python `cmp_zmax`):
///   1. Coordinate columns (name starts with x/y/z) come first.
///   2. Shorter names come before longer names.
///   3. "balance" is always last.
use ndarray::Array1;
use hdf5::Error;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write as IoWrite};
use std::path::Path;

/// Column width used throughout (matches Python `{:>14s}` / `{:14.6e}`).
const COL_WIDTH: usize = 14;
const SEP_WIDTH: usize = 80;

/// Set of variable names that are printed with fewer decimal places in Python.
/// (`rethe`, `redel`, `rex`, `retau`, `redeldi`, `redelen` → `{:14.2e}`)
fn is_reynolds_number(name: &str) -> bool {
    matches!(
        name,
        "rethe" | "redel" | "rex" | "retau" | "redeldi" | "redelen"
    )
}

/// Set of variable names printed as plain floats (`{:14.6f}` in Python).
fn is_coordinate_plus(name: &str) -> bool {
    name.ends_with("plus") || name == "zc"
}

/// Sort column names using the same rules as Python's `cmp_zmax`:
///   1. Coordinate names (x/y/z prefix) first.
///   2. Shorter before longer.
///   3. "balance" last.
fn sort_columns(names: &mut Vec<String>) {
    names.sort_by(|a, b| {
        let a_coord = a.starts_with(['x', 'y', 'z']);
        let b_coord = b.starts_with(['x', 'y', 'z']);

        if a_coord && !b_coord {
            return std::cmp::Ordering::Less;
        }
        if !a_coord && b_coord {
            return std::cmp::Ordering::Greater;
        }
        if a == "balance" {
            return std::cmp::Ordering::Greater;
        }
        if b == "balance" {
            return std::cmp::Ordering::Less;
        }
        // shorter name first, then lexicographic
        a.len().cmp(&b.len()).then(a.cmp(b))
    });
}

/// Write a statistics `.dat` file.
///
/// # Arguments
/// * `path`   – output file path (`.dat` extension will be appended if missing).
/// * `dire`   – wall-normal direction label (`"z"`, `"y"`, …).
/// * `ret`    – friction Reynolds number (uτ/ν × half-channel height), for the header.
/// * `data`   – named 1-D profile arrays (all must have the same length).
pub fn write_dat<P: AsRef<Path>>(
    path: P,
    dire: &str,
    ret: f64,
    data: &HashMap<String, Array1<f64>>,
) -> Result<(), Error> {
    // Determine output path
    let path = path.as_ref();
    let path_str = path.to_string_lossy();
    let final_path = if path_str.ends_with(".dat") {
        path.to_path_buf()
    } else {
        path.with_extension("dat")
    };

    // Validate: all columns must have the same length
    let lengths: Vec<usize> = data.values().map(|a| a.len()).collect();
    let n = *lengths.first().ok_or_else(|| Error::from("write_dat: no data columns"))?;
    if lengths.iter().any(|&l| l != n) {
        return Err("write_dat: all columns must have the same length".into());
    }

    // Sort columns
    let mut col_names: Vec<String> = data.keys().cloned().collect();
    sort_columns(&mut col_names);

    // Build header strings
    let title = format!(
        "{:<40}\n",
        format!("Statistics of the data along with {}, Ret={:.4}", dire, ret)
    );
    let sep = "-".repeat(SEP_WIDTH) + "\n";

    let header1: String = col_names
        .iter()
        .enumerate()
        .map(|(i, _)| format!("{:>COL_WIDTH$}", format!("C{}", i + 1), COL_WIDTH = COL_WIDTH))
        .collect::<Vec<_>>()
        .join("")
        + "\n";

    let header2: String = col_names
        .iter()
        .map(|name| format!("{:>COL_WIDTH$}", name, COL_WIDTH = COL_WIDTH))
        .collect::<Vec<_>>()
        .join("")
        + "\n";

    // Write file
    let file = File::create(&final_path)
        .map_err(|e| Error::from(format!("write_dat: cannot create file: {}", e)))?;
    let mut writer = BufWriter::new(file);

    writer
        .write_all((title + &header1 + &header2 + &sep + &sep).as_bytes())
        .map_err(|e| Error::from(format!("write_dat: write error: {}", e)))?;

    for row in 0..n {
        let line: String = col_names
            .iter()
            .map(|name| {
                let v = data[name][row];
                if is_reynolds_number(name) {
                    format!("{:>14.2e}", v)
                } else if is_coordinate_plus(name) {
                    format!("{:>14.6}", v)
                } else {
                    format!("{:>14.6e}", v)
                }
            })
            .collect::<Vec<_>>()
            .join("")
            + "\n";

        writer
            .write_all(line.as_bytes())
            .map_err(|e| Error::from(format!("write_dat: write error: {}", e)))?;
    }

    println!("File generated: {}", final_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn sort_columns_ordering() {
        let mut names = vec![
            "balance".to_string(),
            "uu".to_string(),
            "zc".to_string(),
            "uv".to_string(),
            "u".to_string(),
        ];
        sort_columns(&mut names);
        // "zc" first (coord), then "u" < "uu" < "uv" (shorter first), "balance" last
        assert_eq!(names[0], "zc");
        assert_eq!(names[names.len() - 1], "balance");
    }

    #[test]
    fn write_dat_smoke() {
        let mut data = HashMap::new();
        data.insert("zc".to_string(),  array![0.5, 1.5, 2.5]);
        data.insert("uu".to_string(),  array![0.01, 0.02, 0.03]);
        let tmp = std::env::temp_dir().join("test_output.dat");
        write_dat(&tmp, "z", 180.0, &data).unwrap();
        assert!(tmp.exists());
        std::fs::remove_file(tmp).ok();
    }
}
