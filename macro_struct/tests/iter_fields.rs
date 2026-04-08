use macro_struct::IterFields;

#[derive(IterFields)]
struct Point {
    x: f64,
    y: f64,
}

#[derive(IterFields)]
struct DNSInfo {
    nx: i32,
    ny: i32,
    nz: i32,
    re: f64,
}

#[test]
fn test_field_names() {
    let p = Point { x: 1.0, y: 2.0 };
    let names: Vec<&str> = p.iter_fields().map(|(name, _)| name).collect();
    assert_eq!(names, vec!["x", "y"]);
}

#[test]
fn test_field_values_downcast() {
    let p = Point { x: 3.14, y: -1.0 };
    let fields: Vec<_> = p.iter_fields().collect();

    let x_val = fields[0].1.downcast_ref::<f64>().expect("x should be f64");
    let y_val = fields[1].1.downcast_ref::<f64>().expect("y should be f64");

    assert!((x_val - 3.14).abs() < f64::EPSILON);
    assert!((y_val - (-1.0)).abs() < f64::EPSILON);
}

#[test]
fn test_field_count() {
    let info = DNSInfo { nx: 64, ny: 128, nz: 32, re: 180.0 };
    let count = info.iter_fields().count();
    assert_eq!(count, 4);
}

#[test]
fn test_mixed_types_downcast() {
    let info = DNSInfo { nx: 64, ny: 128, nz: 32, re: 180.0 };
    let fields: Vec<_> = info.iter_fields().collect();

    assert_eq!(*fields[0].1.downcast_ref::<i32>().unwrap(), 64);
    assert_eq!(*fields[1].1.downcast_ref::<i32>().unwrap(), 128);
    assert_eq!(*fields[2].1.downcast_ref::<i32>().unwrap(), 32);
    assert!((fields[3].1.downcast_ref::<f64>().unwrap() - 180.0).abs() < f64::EPSILON);
}

#[test]
fn test_wrong_type_downcast_returns_none() {
    let p = Point { x: 1.0, y: 2.0 };
    let fields: Vec<_> = p.iter_fields().collect();
    // x is f64, not i32
    assert!(fields[0].1.downcast_ref::<i32>().is_none());
}
