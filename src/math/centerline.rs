/// Control may change the position of the center line.
/// Centerline was defined through dudz

use ndarray::Array1;

pub fn get_centerline<T>(&u_slice: &Array1<T>, &uz_slice: &Array1<T>, &z: &Array1<T>) -> T {
        let len = &uz_slice.len();
        let mut locate = 0usize;
        for i in 0..len {
            if &uz_slice[i] * &uz_slice(i+1) < 0 {
                let locate = i;
                break;
            }
        }
    // Locate the location where split uz>0 and uz<0.
    let k = (&u_slice[locate+1] - &u_slice[locate])/(&z[locate+1] - &z[locate]);
    let result = (&z[locate+1]-&z[locate])*2*&u_slice[locate]/(&u_slice[locate+1] - &u_slice[locate]) + &z[locate];
    return result
}
