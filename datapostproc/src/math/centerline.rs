/// Control may change the position of the center line.
/// Centerline was defined through dudz

use ndarray::{Array1};

pub fn get_centerline<T>(u_slice: &Array1<f64>, uz_slice: &Array1<f64>, z: &Array1<f64>) -> f64 
        {
        let len = uz_slice.len();
        let mut locate = 0usize;
        for i in 0..(len-1) {
            if uz_slice[i] * uz_slice[i+1] < 0f64 {
                locate = i;
                break;
            }
        }
    // Locate the location where split uz>0 and uz<0.
    let result = (z[locate+1]-z[locate])*2f64*u_slice[locate]/(u_slice[locate+1] - u_slice[locate]) + z[locate];
    return result
}
