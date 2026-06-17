use ndarray::{ArrayD, Axis, IxDyn};
use hdf5::Error;

/// Average a dynamic array along one axis, reducing its rank by 1.
/// Equivalent to Python's `_avg.avg3d(var, direction)` / `avg2d`.
pub fn avg_axis(var: &ArrayD<f64>, axis: usize) -> Result<ArrayD<f64>, Error> {
    if axis >= var.ndim() {
        return Err(format!(
            "axis {} is out of bounds for array with {} dimensions",
            axis,
            var.ndim()
        )
        .into());
    }
    var.mean_axis(Axis(axis))
        .ok_or_else(|| Error::from("avg_axis: array has zero length along the given axis"))
}

/// Reduce a 3-D array to a 1-D profile by averaging over the two non-wall directions.
///
/// `wall_axis` is the axis you want to **keep** (0 = z, 1 = y, 2 = x in DNS convention).
/// The two remaining axes are averaged away in the order that preserves the wall-normal profile.
///
/// Equivalent to Python's `_avg.nor_all(var, dire)`.
pub fn avg_to_profile(var: &ArrayD<f64>, wall_axis: usize) -> Result<ArrayD<f64>, Error> {
    if var.ndim() != 3 {
        return Err(format!(
            "avg_to_profile expects a 3-D array, got {} dimensions",
            var.ndim()
        )
        .into());
    }
    if wall_axis >= 3 {
        return Err(format!("wall_axis {} is out of bounds for a 3-D array", wall_axis).into());
    }

    // Average out the two axes that are NOT wall_axis.
    // After the first avg the rank drops to 2; the surviving axes are renumbered,
    // so we must recompute which axis to remove next.
    let other_axes: Vec<usize> = (0..3).filter(|&a| a != wall_axis).collect();

    // First reduction: remove other_axes[0].
    let first = avg_axis(var, other_axes[0])?;

    // After removing other_axes[0] the wall_axis shifts if wall_axis > other_axes[0].
    // The second axis to remove was other_axes[1]; after the first removal it becomes:
    let second_axis_in_2d = if other_axes[1] > other_axes[0] {
        other_axes[1] - 1
    } else {
        other_axes[1]
    };

    avg_axis(&first, second_axis_in_2d)
}

/// Convenience: average a 2-D slice (e.g. an x-z plane) to a 1-D profile along `wall_axis`.
pub fn avg2d_to_profile(var: &ArrayD<f64>, wall_axis: usize) -> Result<ArrayD<f64>, Error> {
    if var.ndim() != 2 {
        return Err(format!(
            "avg2d_to_profile expects a 2-D array, got {} dimensions",
            var.ndim()
        )
        .into());
    }
    if wall_axis >= 2 {
        return Err(format!("wall_axis {} is out of bounds for a 2-D array", wall_axis).into());
    }
    let other_axis = 1 - wall_axis;
    avg_axis(var, other_axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn make_test_array() -> ArrayD<f64> {
        // Shape (2, 3, 4): z=2, y=3, x=4
        // value[z][y][x] = z as f64
        Array3::from_shape_fn((2, 3, 4), |(z, _y, _x)| z as f64).into_dyn()
    }

    #[test]
    fn avg_axis_reduces_rank() {
        let a = make_test_array();
        let result = avg_axis(&a, 0).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
        // mean over z: (0 + 1) / 2 = 0.5 for all elements
        assert!((result[[0, 0]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn avg_to_profile_keeps_wall_axis() {
        let a = make_test_array();
        // Keep axis 0 (z), average over y and x → shape [2]
        let profile = avg_to_profile(&a, 0).unwrap();
        assert_eq!(profile.shape(), &[2]);
        // z=0 slice is all 0.0, z=1 slice is all 1.0
        assert!((profile[IxDyn(&[0])] - 0.0).abs() < 1e-12);
        assert!((profile[IxDyn(&[1])] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn avg_axis_out_of_bounds() {
        let a = make_test_array();
        assert!(avg_axis(&a, 5).is_err());
    }
}
