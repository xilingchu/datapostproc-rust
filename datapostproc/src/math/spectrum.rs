//! Spanwise energy spectra and two-point correlations.
//!
//! Coordinate convention (matches the rest of this crate):
//!   Array shape (nz, ny, nx): axis 0 = wall-normal z, axis 1 = spanwise y,
//!   axis 2 = streamwise x.
//!
//! The spanwise direction y is the PERIODIC, UNIFORM direction, so spectra and
//! correlations are computed along axis 1 with a plain FFT (no windowing).
//! The streamwise direction is spatially developing (non-periodic, non-uniform
//! spacing), so x positions only enter as extra ensemble samples: every y-line
//! at every (z, x) of every snapshot is one realization.
//!
//! Definitions (per wall-normal location z, ensemble ⟨·⟩ over x and snapshots):
//!
//!   fluctuation:   u'(y) = u(y) − (1/N) Σ_n u(y_n)      (spanwise mean removed,
//!                  i.e. the k=0 mode is zeroed per line)
//!   DFT:           ĉ_m = Σ_n u'_n e^{−2πi m n / N}
//!   stored power:  P_m = ⟨|ĉ_m|²⟩ / N²                  (two-sided, m = 0..N/2)
//!
//!   variance:      ⟨u'²⟩ = Σ_m w_m P_m                  (Parseval)
//!   mode energy:   E_m   = w_m P_m       with  w_m = 1 for m = 0 and m = N/2
//!                                        (N even), w_m = 2 otherwise
//!   spectral density:  E(k_m) = E_m / Δk,  k_m = 2π m / L_y,  Δk = 2π / L_y
//!                  so that Σ_m E(k_m) Δk = ⟨u'²⟩.
//!   correlation:   R(r_j) = ⟨u'(y) u'(y+r_j)⟩ / ⟨u'²⟩
//!                        = Σ_m w_m P_m cos(2π m j / N) / ⟨u'²⟩,   r_j = j L_y/N
//!
//! Note for spanwise-periodic control (slit arrays with a spanwise period):
//! only the uniform (k=0) mode is removed per line, so a steady spanwise-varying
//! mean shows up as sharp peaks at the control wavenumber and its harmonics.
//! That is physically meaningful (dispersive component) — do not be surprised.

use hdf5::Error;
use ndarray::{Array1, Array2, ArrayD, Axis};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

/// Accumulator for spanwise spectra: feed it snapshots with [`accumulate`],
/// then read out the ensemble-averaged spectrum / correlation.
///
/// [`accumulate`]: SpanwiseSpectrum::accumulate
pub struct SpanwiseSpectrum {
    nz: usize,
    ny: usize,
    ly: f64,
    nk: usize, // ny/2 + 1
    /// Accumulated two-sided power ⟨|ĉ_m|²⟩/N², shape (nz, nk); divide by `lines`.
    power: Array2<f64>,
    /// Number of y-lines accumulated per z (n_x × n_snapshots).
    lines: usize,
    fft: Arc<dyn Fft<f64>>,
}

impl SpanwiseSpectrum {
    /// `nz` — wall-normal points, `ny` — spanwise points (uniform, periodic),
    /// `ly` — spanwise period length.
    pub fn new(nz: usize, ny: usize, ly: f64) -> Result<Self, Error> {
        if ny < 2 {
            return Err(format!("ny = {ny} too small for a spectrum").into());
        }
        if ly <= 0.0 {
            return Err(format!("ly = {ly} must be positive").into());
        }
        let fft = FftPlanner::new().plan_fft_forward(ny);
        Ok(Self {
            nz,
            ny,
            ly,
            nk: ny / 2 + 1,
            power: Array2::zeros((nz, ny / 2 + 1)),
            lines: 0,
            fft,
        })
    }

    /// Add one snapshot, shape `(nz, ny, nx_sel)`.  Every y-line (each z, x)
    /// counts as one realization; the spanwise mean of each line is removed
    /// before the FFT.
    pub fn accumulate(&mut self, u: &ArrayD<f64>) -> Result<(), Error> {
        let s = u.shape();
        if s.len() != 3 || s[0] != self.nz || s[1] != self.ny {
            return Err(format!(
                "snapshot shape {s:?} incompatible with nz={} ny={}",
                self.nz, self.ny
            )
            .into());
        }
        let nx = s[2];
        let ny_f = self.ny as f64;
        let norm = 1.0 / (ny_f * ny_f);

        let mut buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); self.ny];

        // lanes(Axis(1)) iterates row-major over (z, x): lane l → z = l / nx.
        for (l, lane) in u.lanes(Axis(1)).into_iter().enumerate() {
            let iz = l / nx;
            let mean = lane.sum() / ny_f;
            for (b, &v) in buf.iter_mut().zip(lane.iter()) {
                *b = Complex::new(v - mean, 0.0);
            }
            self.fft.process(&mut buf);
            let mut row = self.power.row_mut(iz);
            for m in 0..self.nk {
                row[m] += buf[m].norm_sqr() * norm;
            }
        }
        self.lines += nx;
        Ok(())
    }

    /// Number of y-lines accumulated so far (per z).
    pub fn samples(&self) -> usize {
        self.lines
    }

    /// One-sided weight w_m: 1 at m = 0 and at the Nyquist mode (even N), else 2.
    fn weight(&self, m: usize) -> f64 {
        if m == 0 || (self.ny % 2 == 0 && m == self.nk - 1) {
            1.0
        } else {
            2.0
        }
    }

    /// Spanwise fluctuation variance ⟨u'²⟩(z) reconstructed from the spectrum.
    pub fn variance(&self) -> Result<Array1<f64>, Error> {
        self.check_samples()?;
        let n = self.lines as f64;
        let mut var = Array1::zeros(self.nz);
        for iz in 0..self.nz {
            let mut s = 0.0;
            for m in 0..self.nk {
                s += self.weight(m) * self.power[[iz, m]] / n;
            }
            var[iz] = s;
        }
        Ok(var)
    }

    /// One-sided energy spectral density.
    ///
    /// Returns `(k, e)` with `k[m] = 2π m / L_y` (length nk) and `e` of shape
    /// `(nz, nk)`, normalized so that `Σ_m e[z,m] Δk = ⟨u'²⟩(z)`.
    pub fn energy_spectrum(&self) -> Result<(Array1<f64>, Array2<f64>), Error> {
        self.check_samples()?;
        let n = self.lines as f64;
        let dk = 2.0 * std::f64::consts::PI / self.ly;
        let k = Array1::from_iter((0..self.nk).map(|m| m as f64 * dk));
        let mut e = Array2::zeros((self.nz, self.nk));
        for iz in 0..self.nz {
            for m in 0..self.nk {
                e[[iz, m]] = self.weight(m) * self.power[[iz, m]] / n / dk;
            }
        }
        Ok((k, e))
    }

    /// Normalized two-point correlation coefficient.
    ///
    /// Returns `(r, rho)` with separations `r[j] = j L_y / N` for
    /// `j = 0..=N/2` and `rho` of shape `(nz, N/2+1)`; `rho[z,0] = 1`.
    /// At a z where the variance is zero, the row is set to 0 (with rho[0]=1).
    pub fn correlation(&self) -> Result<(Array1<f64>, Array2<f64>), Error> {
        self.check_samples()?;
        let n = self.lines as f64;
        let nsep = self.ny / 2 + 1;
        let dy = self.ly / self.ny as f64;
        let r = Array1::from_iter((0..nsep).map(|j| j as f64 * dy));
        let mut rho = Array2::zeros((self.nz, nsep));
        let two_pi = 2.0 * std::f64::consts::PI;

        for iz in 0..self.nz {
            let mut r0 = 0.0;
            for m in 0..self.nk {
                r0 += self.weight(m) * self.power[[iz, m]] / n;
            }
            if r0 <= f64::EPSILON {
                rho[[iz, 0]] = 1.0;
                continue;
            }
            for j in 0..nsep {
                let mut rj = 0.0;
                for m in 0..self.nk {
                    let phase = two_pi * (m * j) as f64 / self.ny as f64;
                    rj += self.weight(m) * self.power[[iz, m]] / n * phase.cos();
                }
                rho[[iz, j]] = rj / r0;
            }
        }
        Ok((r, rho))
    }

    fn check_samples(&self) -> Result<(), Error> {
        if self.lines == 0 {
            return Err("no snapshots accumulated".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    const PI: f64 = std::f64::consts::PI;

    /// u'(y) = A cos(2π m0 y / Ly): all energy in mode m0, ρ(r) = cos(2π m0 r / Ly).
    #[test]
    fn single_mode_spectrum_and_correlation() {
        let (nz, ny, nx) = (3, 64, 5);
        let ly = 2.0 * PI;
        let m0 = 4usize;
        let a = 0.7;

        let u = Array3::from_shape_fn((nz, ny, nx), |(_, j, _)| {
            let y = ly * j as f64 / ny as f64;
            a * (2.0 * PI * m0 as f64 * y / ly).cos()
        })
        .into_dyn();

        let mut spec = SpanwiseSpectrum::new(nz, ny, ly).unwrap();
        spec.accumulate(&u).unwrap();

        // variance = A²/2
        let var = spec.variance().unwrap();
        for iz in 0..nz {
            assert!((var[iz] - a * a / 2.0).abs() < 1e-12, "var = {}", var[iz]);
        }

        // all mode energy at m0
        let (k, e) = spec.energy_spectrum().unwrap();
        let dk = 2.0 * PI / ly;
        assert!((k[1] - dk).abs() < 1e-12);
        for m in 0..e.shape()[1] {
            let expected = if m == m0 { a * a / 2.0 / dk } else { 0.0 };
            assert!(
                (e[[1, m]] - expected).abs() < 1e-10,
                "mode {m}: {} vs {expected}",
                e[[1, m]]
            );
        }

        // correlation is a pure cosine
        let (r, rho) = spec.correlation().unwrap();
        for j in 0..rho.shape()[1] {
            let expected = (2.0 * PI * m0 as f64 * r[j] / ly).cos();
            assert!(
                (rho[[2, j]] - expected).abs() < 1e-10,
                "sep {j}: {} vs {expected}",
                rho[[2, j]]
            );
        }
        assert!((rho[[0, 0]] - 1.0).abs() < 1e-12);
    }

    /// Parseval: spectrum-reconstructed variance equals the direct variance,
    /// for an irregular (deterministic pseudo-random) field, ny not a power of 2.
    #[test]
    fn parseval_matches_direct_variance() {
        let (nz, ny, nx) = (2, 48, 7);
        let ly = 3.0;

        // deterministic "noise"
        let u = Array3::from_shape_fn((nz, ny, nx), |(i, j, k)| {
            ((i * 7919 + j * 104729 + k * 1299709) % 10007) as f64 / 10007.0
        })
        .into_dyn();

        let mut spec = SpanwiseSpectrum::new(nz, ny, ly).unwrap();
        spec.accumulate(&u).unwrap();
        let var = spec.variance().unwrap();

        for iz in 0..nz {
            // direct variance: average over x of per-line spanwise variance
            let mut direct = 0.0;
            for ix in 0..nx {
                let line: Vec<f64> = (0..ny).map(|j| u[[iz, j, ix]]).collect();
                let mean = line.iter().sum::<f64>() / ny as f64;
                direct += line.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ny as f64;
            }
            direct /= nx as f64;
            assert!(
                (var[iz] - direct).abs() < 1e-12 * direct.max(1.0),
                "z {iz}: {} vs {direct}",
                var[iz]
            );
        }
    }

    /// Ensemble averaging: two snapshots with amplitudes A and B give the
    /// mean of the two individual spectra.
    #[test]
    fn ensemble_average_over_snapshots() {
        let (nz, ny, nx) = (1, 32, 1);
        let ly = 1.0;
        let m0 = 3usize;

        let make = |amp: f64| {
            Array3::from_shape_fn((nz, ny, nx), |(_, j, _)| {
                let y = ly * j as f64 / ny as f64;
                amp * (2.0 * PI * m0 as f64 * y / ly).sin()
            })
            .into_dyn()
        };

        let mut spec = SpanwiseSpectrum::new(nz, ny, ly).unwrap();
        spec.accumulate(&make(1.0)).unwrap();
        spec.accumulate(&make(2.0)).unwrap();
        assert_eq!(spec.samples(), 2);

        let var = spec.variance().unwrap();
        let expected = 0.5 * (1.0 * 1.0 / 2.0 + 2.0 * 2.0 / 2.0);
        assert!((var[0] - expected).abs() < 1e-12, "{} vs {expected}", var[0]);
    }
}
