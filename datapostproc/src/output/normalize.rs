/// Wall-unit normalization.
///
/// Mirrors Python `outputData.normalize()`:
///   - coordinates (other than the wall-normal `dire`) → multiply by utau/nu
///   - wall-normal coordinate (`dire`)                 → kept as original; `{dire}plus` added
///   - velocities  (keys containing u/v/w)             → divide by utau
///   - pressure    (keys containing p)                 → multiply by 1/tau
use ndarray::Array1;
use hdf5::Error;
use std::collections::HashMap;

pub struct Profiles {
    pub dire: String,
    pub nu:   f64,
    pub tau:  f64,
    pub utau: f64,
    pub data: HashMap<String, Array1<f64>>,
}

impl Profiles {
    pub fn new(dire: &str, nu: f64, tau: f64, data: HashMap<String, Array1<f64>>) -> Result<Self, Error> {
        if tau <= 0.0 {
            return Err("normalize: tau must be positive".into());
        }
        let utau = tau.sqrt();
        Ok(Self { dire: dire.to_string(), nu, tau, utau, data })
    }

    /// Normalize profiles to wall units and add `{dire}plus`.
    ///
    /// Python behavior (from `outputData.normalize`):
    ///   1. Normalize all coord keys in-place (incl. the wall-normal coord).
    ///   2. Save the normalized wall-normal coord as `{dire}plus`.
    ///   3. Restore the original (physical) wall-normal coord under `dire`.
    pub fn to_wall_units(&self) -> Result<HashMap<String, Array1<f64>>, Error> {
        let tau_inv = 1.0 / self.tau;
        let mut out: HashMap<String, Array1<f64>> = HashMap::new();
        let mut wall_coord_plus: Option<Array1<f64>> = None;

        for (key, arr) in &self.data {
            let normed = if key.starts_with(['x', 'y', 'z']) {
                let normalized = arr.mapv(|v| v * self.utau / self.nu);
                if *key == self.dire {
                    // Save wall-unit version for `{dire}plus`; restore original below.
                    wall_coord_plus = Some(normalized);
                    arr.clone()  // keep original physical coord under `dire`
                } else {
                    normalized
                }
            } else if key.chars().any(|c| matches!(c, 'u' | 'v' | 'w')) {
                arr.mapv(|v| v / self.utau)
            } else if key.contains('p') {
                arr.mapv(|v| v * tau_inv)
            } else {
                arr.clone()
            };
            out.insert(key.clone(), normed);
        }

        // Insert `{dire}plus` (wall-unit wall-normal coordinate)
        if let Some(plus) = wall_coord_plus {
            let plus_key = format!("{}plus", self.dire.chars().next().unwrap_or('z'));
            out.insert(plus_key, plus);
        }

        Ok(out)
    }
}
