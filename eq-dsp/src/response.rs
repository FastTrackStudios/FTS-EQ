//! Frequency response calculation for GUI display.
//!
//! Uses the analog prototype magnitude-squared formula directly,
//! avoiding the cost of running the actual digital filter.
//! This runs on the GUI thread, not the audio thread.

use std::f64::consts::PI;

use crate::band::Band;
use crate::coeff;

/// Calculate the magnitude response in dB at a given frequency for one band.
///
/// This uses the digital filter coefficients to compute the exact magnitude
/// response at the given frequency, suitable for drawing the EQ curve.
pub fn band_magnitude_db(band: &Band, freq_hz: f64, sample_rate: f64) -> f64 {
    if !band.enabled || (band.gain_db.abs() < 1e-6 && band.filter_type.has_gain()) {
        return 0.0;
    }

    let w = 2.0 * PI * freq_hz / sample_rate;
    let c = coeff::calculate(
        band.filter_type,
        band.freq_hz,
        band.q,
        band.gain_db,
        sample_rate,
    );

    magnitude_squared_db(c, w)
}

/// Compute |H(e^jw)|^2 in dB from biquad coefficients.
fn magnitude_squared_db(c: [f64; 6], w: f64) -> f64 {
    let cos_w = w.cos();
    let cos_2w = (2.0 * w).cos();

    // |H(e^jw)|^2 = |B(e^jw)|^2 / |A(e^jw)|^2
    let num = c[3] * c[3]
        + c[4] * c[4]
        + c[5] * c[5]
        + 2.0 * (c[3] * c[4] + c[4] * c[5]) * cos_w
        + 2.0 * c[3] * c[5] * cos_2w;

    let den = c[0] * c[0]
        + c[1] * c[1]
        + c[2] * c[2]
        + 2.0 * (c[0] * c[1] + c[1] * c[2]) * cos_w
        + 2.0 * c[0] * c[2] * cos_2w;

    if den < 1e-30 {
        return 0.0;
    }

    10.0 * (num / den).log10()
}

/// Calculate the total EQ response in dB at a frequency across all bands.
pub fn total_magnitude_db(bands: &[Band], freq_hz: f64, sample_rate: f64) -> f64 {
    let mut total_db = 0.0;
    for band in bands {
        total_db += band_magnitude_db(band, freq_hz, sample_rate);
    }
    total_db
}

/// Generate a frequency response curve as (frequency_hz, magnitude_db) pairs.
///
/// Returns `num_points` logarithmically-spaced points from 20Hz to 20kHz.
pub fn response_curve(bands: &[Band], sample_rate: f64, num_points: usize) -> Vec<(f64, f64)> {
    let log_min = 20.0_f64.ln();
    let log_max = 20000.0_f64.ln();

    (0..num_points)
        .map(|i| {
            let t = i as f64 / (num_points - 1) as f64;
            let freq = (log_min + t * (log_max - log_min)).exp();
            let db = total_magnitude_db(bands, freq, sample_rate);
            (freq, db)
        })
        .collect()
}
