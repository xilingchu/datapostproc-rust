/// Reynolds-stress components and turbulent kinetic energy.
///
/// Coordinate convention (matches the rest of this crate):
///   array shape (nz, ny, nx): axis 0 = wall-normal z, axis 1 = spanwise y,
///   axis 2 = streamwise x; u = streamwise, v = spanwise, w = wall-normal.
///
/// All inputs are TOTAL second moments ⟨a·b⟩ — either the subavg datasets
/// (uu, vv, …) or ensemble means of instantaneous products.  Everything is
/// spanwise-averaged first and the fluctuation moments are extracted about
/// the span+time mean (same convention as `math/fik.rs`):
///
///     ⟨a'b'⟩ = ⟨ab⟩ − ā·b̄,        k = ½(⟨u'u'⟩ + ⟨v'v'⟩ + ⟨w'w'⟩)
///
/// so spanwise-coherent motion (e.g. from spanwise-periodic control) counts
/// as fluctuation, not mean.

use ndarray::ArrayD;
use hdf5::Error;

use super::avg::avg_axis;

// ─── Public result type ────────────────────────────────────────────────────────

/// Spanwise-averaged Reynolds stresses and TKE, each of shape `(nz, nx)`.
pub struct TkeFields {
    /// ⟨u'u'⟩ — streamwise normal stress.
    pub uu: ArrayD<f64>,
    /// ⟨v'v'⟩ — spanwise normal stress.
    pub vv: ArrayD<f64>,
    /// ⟨w'w'⟩ — wall-normal normal stress.
    pub ww: ArrayD<f64>,
    /// ⟨u'v'⟩ — streamwise/spanwise shear stress.
    pub uv: ArrayD<f64>,
    /// ⟨u'w'⟩ — streamwise/wall-normal shear stress.
    pub uw: ArrayD<f64>,
    /// ⟨v'w'⟩ — spanwise/wall-normal shear stress.
    pub vw: ArrayD<f64>,
    /// k = ½(⟨u'u'⟩ + ⟨v'v'⟩ + ⟨w'w'⟩).
    pub tke: ArrayD<f64>,
}

// ─── Main function ─────────────────────────────────────────────────────────────

/// Extract the Reynolds-stress components and TKE from total moments.
///
/// # Arguments
/// * `u`, `v`, `w`                      – mean velocities, shape `(nz, ny, nx)`.
/// * `uu`, `vv`, `ww`, `uv`, `uw`, `vw` – **total** second moments ⟨a·b⟩,
///                                        same shape.
#[allow(clippy::too_many_arguments)]
pub fn tke_fields(
    u:  &ArrayD<f64>,
    v:  &ArrayD<f64>,
    w:  &ArrayD<f64>,
    uu: &ArrayD<f64>,
    vv: &ArrayD<f64>,
    ww: &ArrayD<f64>,
    uv: &ArrayD<f64>,
    uw: &ArrayD<f64>,
    vw: &ArrayD<f64>,
) -> Result<TkeFields, Error> {
    let shape = u.shape().to_vec();
    if shape.len() != 3 {
        return Err(format!("expected 3-D fields (nz, ny, nx), got {shape:?}").into());
    }
    for (name, arr) in [
        ("v", v as &ArrayD<f64>), ("w", w),
        ("uu", uu), ("vv", vv), ("ww", ww),
        ("uv", uv), ("uw", uw), ("vw", vw),
    ] {
        if arr.shape() != shape.as_slice() {
            return Err(format!(
                "'{name}' shape {:?} does not match u's shape {shape:?}",
                arr.shape()
            ).into());
        }
    }

    // ── spanwise average (axis 1) → (nz, nx) ──────────────────────────────────
    let avg = |a: &ArrayD<f64>| avg_axis(a, 1);
    let u2 = avg(u)?;
    let v2 = avg(v)?;
    let w2 = avg(w)?;

    // ── fluctuation moments: ⟨a'b'⟩ = ⟨ab⟩ − ā·b̄ ─────────────────────────────
    let uu_p = &avg(uu)? - &(&u2 * &u2);
    let vv_p = &avg(vv)? - &(&v2 * &v2);
    let ww_p = &avg(ww)? - &(&w2 * &w2);
    let uv_p = &avg(uv)? - &(&u2 * &v2);
    let uw_p = &avg(uw)? - &(&u2 * &w2);
    let vw_p = &avg(vw)? - &(&v2 * &w2);

    let tke = (&(&uu_p + &vv_p) + &ww_p) * 0.5;

    Ok(TkeFields {
        uu: uu_p,
        vv: vv_p,
        ww: ww_p,
        uv: uv_p,
        uw: uw_p,
        vw: vw_p,
        tke,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array3};
    use std::f64::consts::PI;

    /// Steady flow whose only "fluctuation" is a spanwise-periodic wave:
    ///
    ///   u(z,y) = U0(z) + A·sin(2πy/L),   v = B·sin(2πy/L),   w = 0
    ///
    /// With the span+time-mean convention: ū = U0, v̄ = w̄ = 0, and over a
    /// full period ⟨sin²⟩ = ½, so
    ///   ⟨u'u'⟩ = A²/2,  ⟨v'v'⟩ = B²/2,  ⟨w'w'⟩ = 0,
    ///   ⟨u'v'⟩ = A·B/2 (same phase),   k = (A² + B²)/4.
    #[test]
    fn spanwise_wave_variances_and_tke() {
        let (nz, ny, nx) = (6, 64, 5);
        let (a, b) = (0.3_f64, 0.1_f64);
        let u0: Array1<f64> = Array1::linspace(0.5, 1.5, nz);

        let s = |j: usize| (2.0 * PI * j as f64 / ny as f64).sin();
        let u = Array3::from_shape_fn((nz, ny, nx), |(iz, j, _)| u0[iz] + a * s(j)).into_dyn();
        let v = Array3::from_shape_fn((nz, ny, nx), |(_, j, _)| b * s(j)).into_dyn();
        let w = ArrayD::<f64>::zeros(ndarray::IxDyn(&[nz, ny, nx]));

        // total moments of the steady field are the pointwise products
        let uu = &u * &u;
        let vv = &v * &v;
        let ww = &w * &w;
        let uv = &u * &v;
        let uw = &u * &w;
        let vw = &v * &w;

        let tf = tke_fields(&u, &v, &w, &uu, &vv, &ww, &uv, &uw, &vw)
            .expect("tke_fields failed");

        let expect = [
            ("uu", &tf.uu, a * a / 2.0),
            ("vv", &tf.vv, b * b / 2.0),
            ("ww", &tf.ww, 0.0),
            ("uv", &tf.uv, a * b / 2.0),
            ("uw", &tf.uw, 0.0),
            ("vw", &tf.vw, 0.0),
            ("tke", &tf.tke, (a * a + b * b) / 4.0),
        ];
        for (name, field, want) in expect {
            let max_err = field.iter().map(|x| (x - want).abs()).fold(0.0_f64, f64::max);
            assert!(
                max_err < 1e-12,
                "{name}: max |value − {want}| = {max_err:.2e}"
            );
        }
    }

    /// Shape mismatch must be rejected.
    #[test]
    fn rejects_mismatched_shapes() {
        let a = ArrayD::<f64>::zeros(ndarray::IxDyn(&[4, 3, 5]));
        let bad = ArrayD::<f64>::zeros(ndarray::IxDyn(&[4, 3, 6]));
        assert!(tke_fields(&a, &a, &a, &a, &a, &a, &a, &a, &bad).is_err());
    }
}
