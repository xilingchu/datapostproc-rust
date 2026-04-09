use datapostproc_rust::data::H5File;
use datapostproc_rust::hdf5::H5Data;

const H5_FILE: &str = "subavg_450000_0.h5";

#[test]
fn test_list_variables() {
    let f = H5File::new(H5_FILE).unwrap();
    let vars = f.list_variables().expect("Failed to list variables");
    println!("variables ({}):", vars.len());
    for v in &vars {
        println!("  {v}");
    }
    assert!(!vars.is_empty());
}

#[test]
fn test_open_file() {
    H5File::new(H5_FILE).expect("Failed to open HDF5 file");
}

#[test]
fn test_dns_info() {
    let mut f = H5File::new(H5_FILE).unwrap();
    f.get_info().expect("Failed to read DNSInfo");

    let info = f.info();
    println!("nx={:?}, ny={:?}, nz={:?}", info.nx, info.ny, info.nz);
    println!("nu={:?}, re={:?}", info.nu, info.re);

    assert!(info.is_defined);
    assert!(info.nx.is_some());
    assert!(info.ny.is_some());
    assert!(info.nz.is_some());
    assert!(info.nu.unwrap() > 0.0);
    assert!((info.re.unwrap() - 1.0 / info.nu.unwrap()).abs() < 1e-12);
}

#[test]
fn test_load_coords() {
    let mut f = H5File::new(H5_FILE).unwrap();
    f.load_coords().expect("Failed to load coords");

    for name in &["x", "xd", "x_zero", "y", "yd", "zc", "zd"] {
        let coord = f.coord(name).unwrap_or_else(|| panic!("Missing coord: {name}"));
        let data = coord.read_data::<f64>().unwrap_or_else(|e| panic!("Failed to read {name}: {e}"));
        match &data {
            H5Data::Array(arr) => println!("{name}: shape={:?}, first={:.6}", arr.shape(), arr[[0]]),
            H5Data::Scalar(v)  => println!("{name}: scalar={v:.6}"),
        }
    }
}

#[test]
fn test_add_datasets() {
    let mut f = H5File::new(H5_FILE).unwrap();
    f.add_datasets(&["u", "v", "w"]).expect("Failed to add datasets");

    for name in &["u", "v", "w"] {
        let data = f.dataset(name)
            .unwrap_or_else(|| panic!("Missing dataset: {name}"))
            .read_data::<f64>()
            .unwrap_or_else(|e| panic!("Failed to read {name}: {e}"));
        match &data {
            H5Data::Array(arr) => println!("{name}: shape={:?}", arr.shape()),
            H5Data::Scalar(v)  => println!("{name}: scalar={v}"),
        }
    }
}
