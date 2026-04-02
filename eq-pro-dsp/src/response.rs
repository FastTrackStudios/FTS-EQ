//! Frequency response computation (dual-path: biquad and ZPK).
//!
//! Provides magnitude response, phase response, and group delay evaluation
//! for cascaded biquad sections. Used for display rendering and
//! anti-cramping delay computation.

use std::f64::consts::PI;

use crate::biquad::{eval_sos, mag_db_sos, Coeffs};
use crate::zpk::Zpk;

/// Compute magnitude response in dB at the given frequencies.
///
/// Evaluates the cascade of biquad sections at each frequency and returns
/// the magnitude in decibels.
///
/// # Arguments
/// * `sections` - Cascade of biquad coefficient arrays
/// * `frequencies` - Frequencies in Hz to evaluate at
/// * `sample_rate` - Sample rate in Hz
pub fn compute_magnitude_response(
    sections: &[Coeffs],
    frequencies: &[f64],
    sample_rate: f64,
) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&freq| {
            let w = 2.0 * PI * freq / sample_rate;
            mag_db_sos(sections, w)
        })
        .collect()
}

/// Compute phase response in radians at the given frequencies.
///
/// Evaluates the cascade of biquad sections at each frequency and returns
/// the phase angle in radians (range: -pi to pi).
///
/// # Arguments
/// * `sections` - Cascade of biquad coefficient arrays
/// * `frequencies` - Frequencies in Hz to evaluate at
/// * `sample_rate` - Sample rate in Hz
pub fn compute_phase_response(
    sections: &[Coeffs],
    frequencies: &[f64],
    sample_rate: f64,
) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&freq| {
            let w = 2.0 * PI * freq / sample_rate;
            let h = eval_sos(sections, w);
            h.arg()
        })
        .collect()
}

/// Compute group delay in samples at a given frequency.
///
/// Group delay is the negative derivative of the phase response:
///   tau(w) = -d(phase)/dw
///
/// Computed using central finite differences for numerical stability.
///
/// # Arguments
/// * `sections` - Cascade of biquad coefficient arrays
/// * `freq_hz` - Frequency in Hz to evaluate at
/// * `sample_rate` - Sample rate in Hz
pub fn compute_group_delay(sections: &[Coeffs], freq_hz: f64, sample_rate: f64) -> f64 {
    let w = 2.0 * PI * freq_hz / sample_rate;

    // Central difference step size. Small enough for accuracy,
    // large enough to avoid floating-point noise.
    let dw = 1e-6;

    let w_lo = (w - dw).max(1e-10);
    let w_hi = (w + dw).min(PI - 1e-10);
    let actual_dw = w_hi - w_lo;

    if actual_dw < 1e-15 {
        return 0.0;
    }

    let phase_lo = eval_sos(sections, w_lo).arg();
    let phase_hi = eval_sos(sections, w_hi).arg();

    // Unwrap phase difference: handle wrapping around +/-pi
    let mut dphi = phase_hi - phase_lo;
    while dphi > PI {
        dphi -= 2.0 * PI;
    }
    while dphi < -PI {
        dphi += 2.0 * PI;
    }

    // Group delay = -d(phase)/dw
    -dphi / actual_dw
}

/// Compute magnitude response from a ZPK representation.
///
/// Directly evaluates the ZPK transfer function without converting
/// to biquad sections. Useful for display before coefficient computation.
pub fn compute_magnitude_response_zpk(
    zpk: &Zpk,
    frequencies: &[f64],
    sample_rate: f64,
) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&freq| {
            let w = 2.0 * PI * freq / sample_rate;
            zpk.mag_db(w)
        })
        .collect()
}

/// Compute phase response from a ZPK representation.
pub fn compute_phase_response_zpk(
    zpk: &Zpk,
    frequencies: &[f64],
    sample_rate: f64,
) -> Vec<f64> {
    frequencies
        .iter()
        .map(|&freq| {
            let w = 2.0 * PI * freq / sample_rate;
            zpk.eval_z(w).arg()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biquad::PASSTHROUGH;
    use crate::design::{self, FilterType};

    #[test]
    fn passthrough_magnitude_is_zero_db() {
        let freqs = vec![100.0, 1000.0, 10000.0];
        let mags = compute_magnitude_response(&[PASSTHROUGH], &freqs, 48000.0);
        for (i, &mag) in mags.iter().enumerate() {
            assert!(
                mag.abs() < 1e-10,
                "Passthrough at {} Hz should be 0 dB, got {mag}",
                freqs[i]
            );
        }
    }

    #[test]
    fn passthrough_phase_is_zero() {
        let freqs = vec![100.0, 1000.0, 10000.0];
        let phases = compute_phase_response(&[PASSTHROUGH], &freqs, 48000.0);
        for (i, &phase) in phases.iter().enumerate() {
            assert!(
                phase.abs() < 1e-10,
                "Passthrough at {} Hz should have 0 phase, got {phase}",
                freqs[i]
            );
        }
    }

    #[test]
    fn passthrough_group_delay_is_zero() {
        let gd = compute_group_delay(&[PASSTHROUGH], 1000.0, 48000.0);
        assert!(
            gd.abs() < 1e-6,
            "Passthrough group delay should be ~0, got {gd}"
        );
    }

    #[test]
    fn lowpass_magnitude_decreases() {
        let sos = design::design_filter(FilterType::Lowpass, 1000.0, 0.707, 0.0, 48000.0, 4);
        let freqs = vec![100.0, 1000.0, 10000.0, 20000.0];
        let mags = compute_magnitude_response(&sos, &freqs, 48000.0);

        // Below cutoff should be ~0 dB
        assert!(mags[0].abs() < 1.0, "100 Hz should be ~0 dB, got {}", mags[0]);
        // Above cutoff should be attenuated
        assert!(
            mags[2] < -20.0,
            "10 kHz should be attenuated, got {} dB",
            mags[2]
        );
        // Well above cutoff should be very attenuated
        assert!(
            mags[3] < mags[2],
            "20 kHz should be more attenuated than 10 kHz"
        );
    }

    #[test]
    fn peak_group_delay_nonzero() {
        let sos = design::design_filter(FilterType::Peak, 1000.0, 2.0, 12.0, 48000.0, 2);
        let gd = compute_group_delay(&sos, 1000.0, 48000.0);
        // Peak filter should have nonzero group delay at center
        assert!(
            gd.abs() > 0.01,
            "Peak filter should have nonzero group delay at center, got {gd}"
        );
    }

    #[test]
    fn magnitude_response_length_matches_input() {
        let freqs: Vec<f64> = (1..100).map(|i| i as f64 * 100.0).collect();
        let mags = compute_magnitude_response(&[PASSTHROUGH], &freqs, 48000.0);
        assert_eq!(mags.len(), freqs.len());
    }

    #[test]
    fn phase_response_bounded() {
        let sos = design::design_filter(FilterType::Lowpass, 1000.0, 0.707, 0.0, 48000.0, 4);
        let freqs: Vec<f64> = (1..100).map(|i| i as f64 * 200.0).collect();
        let phases = compute_phase_response(&sos, &freqs, 48000.0);

        for (i, &phase) in phases.iter().enumerate() {
            assert!(
                phase >= -PI && phase <= PI,
                "Phase at {} Hz out of range: {phase}",
                freqs[i]
            );
        }
    }

    #[test]
    fn zpk_magnitude_matches_sos() {
        use crate::prototype;
        use crate::transform;
        use crate::biquad;

        let proto = prototype::butterworth_lp_prewarped(4, 1000.0, 48000.0);
        let digital = transform::bilinear(&proto, 48000.0);
        let sos = biquad::zpk_to_sos(&digital);

        let freqs = vec![100.0, 500.0, 1000.0, 5000.0, 10000.0];
        let mags_sos = compute_magnitude_response(&sos, &freqs, 48000.0);
        let mags_zpk = compute_magnitude_response_zpk(&digital, &freqs, 48000.0);

        for (i, (&ms, &mz)) in mags_sos.iter().zip(mags_zpk.iter()).enumerate() {
            assert!(
                (ms - mz).abs() < 0.5,
                "At {} Hz: SOS={ms:.2} dB, ZPK={mz:.2} dB",
                freqs[i]
            );
        }
    }
}
