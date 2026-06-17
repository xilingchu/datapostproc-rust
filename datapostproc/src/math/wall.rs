/// Wall shear stress and friction velocity.
///
/// Python equivalent (`outputBase.py`):
///   tau  = nu * u1 / z1      (u1: spatially-averaged u at first wall-normal cell)
///   utau = sqrt(tau)
use ndarray::{ArrayD, IxDyn};
use hdf5::Error;

/// Compute wall shear stress from the mean streamwise velocity at the first
/// wall-normal cell centre.
///
/// # Arguments
/// * `u_profile` – 1-D array of mean u averaged over the homogeneous directions,
///                 indexed by wall-normal position (output of `avg_to_profile`).
/// * `z`         – 1-D array of wall-normal cell-centre coordinates.
/// * `nu`        – kinematic viscosity.
///
/// Returns a scalar `tau = nu * u[0] / z[0]`.
pub fn wall_shear_stress(u_profile: &ArrayD<f64>, z: &ArrayD<f64>, nu: f64) -> Result<f64, Error> {
    if u_profile.ndim() != 1 {
        return Err(format!(
            "wall_shear_stress: u_profile must be 1-D, got {} dimensions",
            u_profile.ndim()
        )
        .into());
    }
    if z.ndim() != 1 {
        return Err("wall_shear_stress: z must be 1-D".into());
    }
    let u1 = u_profile[IxDyn(&[0])];
    let z1 = z[IxDyn(&[0])];
    if z1 == 0.0 {
        return Err("wall_shear_stress: z[0] is zero, cannot divide".into());
    }
    Ok(nu * u1 / z1)
}

/// Compute friction velocity `utau = sqrt(tau)`.
/// Returns 0.0 if `tau` is negative (avoids NaN on non-physical inputs).
pub fn friction_velocity(tau: f64) -> f64 {
    if tau <= 0.0 { 0.0 } else { tau.sqrt() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tau_basic() {
        let u = array![1.0, 2.0, 3.0].into_dyn();
        let z = array![0.5, 1.5, 2.5].into_dyn();
        let tau = wall_shear_stress(&u, &z, 1e-3).unwrap();
        // nu * u[0] / z[0] = 1e-3 * 1.0 / 0.5 = 2e-3
        assert!((tau - 2e-3).abs() < 1e-15);
    }

    #[test]
    fn utau_positive() {
        assert!((friction_velocity(4.0) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn utau_non_physical() {
        assert_eq!(friction_velocity(-1.0), 0.0);
        assert_eq!(friction_velocity(0.0), 0.0);
    }

    #[test]
    fn tau_zero_z_is_err() {
        let u = array![1.0].into_dyn();
        let z = array![0.0].into_dyn();
        assert!(wall_shear_stress(&u, &z, 1e-3).is_err());
    }
}
