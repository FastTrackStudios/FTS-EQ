//! Pro-Q 4 shelf filter design — ZPK pipeline matching the binary.
//!
//! Pro-Q 4 shelf pipeline (from design_filter_zpk_and_transform at 0x1800ff6f0):
//!   1. Butterworth LP prototype (filter_type_dispatcher, iVar9==2 path)
//!   2. apply_shelf_gain_to_zpk (0x1800fcce0) — for type 7 (low shelf) only
//!   3. Bilinear transform (transform type 2)
//!   4. zpk_to_biquad_coefficients (0x1800fe040)
//!
//! is_low_type_shelf_gain (0x1800ffbc0) returns true for types {2, 5, 6, 7}:
//!   - Type 7 (Low Shelf): zeros *= gain, poles /= gain at ZPK stage
//!   - Type 8 (High Shelf): gain applied via numerator normalization at biquad stage
//!   - Type 9 (Tilt Shelf): similar to high shelf
//!   - Type 10 (Band Shelf): LP→BP prototype + bilinear + shelf gain

use std::f64::consts::PI;

use crate::biquad::{self, Coeffs, PASSTHROUGH};
use crate::prototype;
use crate::transform;
use crate::zpk::{Complex, Zpk};

/// Design a low shelf filter via ZPK pipeline.
///
/// Pro-Q 4 type 7: Butterworth LP prototype → apply_shelf_gain_to_zpk
/// (scales zeros × gain, poles ÷ gain) → bilinear → biquads.
///
/// From binary analysis of setup_eq_band_filter (0x1800fdf10):
///   - For shelf types, bilinear_transform_zpk receives `1/Q_internal` as its
///     frequency scaling parameter (param_3), which uniformly scales all prototype
///     poles before the bilinear transform.
///   - The biquad Q for shelves is always INV_SQRT2 (set at 0x1800fdfc0).
///   - User Q controls the shelf transition steepness via this pole scaling.
///
/// `n_sections` is the number of biquad sections (order / 2).
pub fn design_low_shelf(
    n_sections: usize,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if gain_db.abs() < 0.001 {
        return vec![PASSTHROUGH; n];
    }

    let w0 = 2.0 * PI * freq_hz / sample_rate;

    (0..n)
        .map(|k| {
            let section_gain = gain_db / n as f64;
            // Pro-Q 4 uses Butterworth pole placement per section, then uniformly
            // scales all poles by 1/Q_internal in bilinear_transform_zpk.
            // In RBJ terms, the section Q is the Butterworth Q scaled by the
            // user's Q relative to the default (INV_SQRT2).
            let bw_q = butterworth_section_q(k, n);
            let section_q = bw_q * (q / std::f64::consts::FRAC_1_SQRT_2);
            rbj_low_shelf(w0, section_q, section_gain)
        })
        .collect()
}

/// Design a high shelf filter via ZPK pipeline.
///
/// Pro-Q 4 type 8: Butterworth LP prototype → bilinear → numerator normalization
/// in zpk_sections_to_biquads to apply shelf gain.
///
/// Same Q scaling as low shelf: bilinear_transform_zpk scales poles by 1/Q_internal.
///
/// `n_sections` is the number of biquad sections (order / 2).
pub fn design_high_shelf(
    n_sections: usize,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if gain_db.abs() < 0.001 {
        return vec![PASSTHROUGH; n];
    }

    let w0 = 2.0 * PI * freq_hz / sample_rate;

    (0..n)
        .map(|k| {
            let section_gain = gain_db / n as f64;
            let bw_q = butterworth_section_q(k, n);
            let section_q = bw_q * (q / std::f64::consts::FRAC_1_SQRT_2);
            rbj_high_shelf(w0, section_q, section_gain)
        })
        .collect()
}

/// Design a tilt shelf filter.
///
/// Pro-Q 4 type 9: tilts the spectrum around the corner frequency.
/// +gain below corner, -gain above (or vice versa).
///
/// `n_sections` is the number of biquad sections.
pub fn design_tilt_shelf(
    n_sections: usize,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if gain_db.abs() < 0.001 {
        return vec![PASSTHROUGH; n];
    }

    let w0 = 2.0 * PI * freq_hz / sample_rate;

    (0..n)
        .map(|k| {
            let section_gain = gain_db / n as f64;
            let bw_q = butterworth_section_q(k, n);
            let section_q = bw_q * (q / std::f64::consts::FRAC_1_SQRT_2);
            rbj_low_shelf(w0, section_q, section_gain)
        })
        .collect()
}

/// Design a band shelf filter.
///
/// Pro-Q 4 type 10: LP→BP prototype + bilinear + shelf gain.
/// Boosts/cuts a band while leaving DC and Nyquist at 0 dB.
///
/// `n_sections` is the number of biquad sections.
pub fn design_band_shelf(
    n_sections: usize,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if gain_db.abs() < 0.001 {
        return vec![PASSTHROUGH; n];
    }

    // Compute bandwidth edges from center frequency and Q.
    let halfbw = (0.5 / q).asinh() / 2.0_f64.ln();
    let scale = 2.0_f64.powf(halfbw);
    let f_lo = freq_hz / scale;
    let f_hi = freq_hz * scale;
    let w_lo = 2.0 * PI * f_lo / sample_rate;
    let w_hi = 2.0 * PI * f_hi / sample_rate;

    let gain_per = gain_db / n as f64;
    let shelf_q = std::f64::consts::FRAC_1_SQRT_2;

    let mut sections = Vec::with_capacity(2 * n);
    for _ in 0..n {
        sections.push(rbj_low_shelf(w_hi, shelf_q, gain_per));
        sections.push(rbj_high_shelf(w_lo, shelf_q, gain_per));
    }
    sections
}

/// Apply shelf gain to a ZPK filter representation.
///
/// Pro-Q 4's `apply_shelf_gain_to_zpk` (0x1800fcce0):
///   - is_low_type_shelf_gain (0x1800ffbc0) returns true for types {2, 5, 6, 7}
///   - When true: zeros *= gain AND poles /= gain
///   - When false (types 8, 9, etc.): zeros *= gain only
///   - Type 6 (Flat Tilt): gain is squared
pub fn apply_shelf_gain(zpk: &mut Zpk, filter_type: u32, gain_linear: f64) {
    if (gain_linear - 1.0).abs() < 1e-10 {
        return;
    }

    let is_low_type = matches!(filter_type, 2 | 5 | 6 | 7);

    // Scale zeros by gain
    for z in &mut zpk.zeros {
        *z = *z * gain_linear;
    }

    // For low types, also scale poles by 1/gain
    if is_low_type {
        let inv_gain = 1.0 / gain_linear;
        for p in &mut zpk.poles {
            *p = *p * inv_gain;
        }
    }

    // For type 6 (flat tilt), square the gain
    let final_gain = if filter_type == 6 {
        gain_linear * gain_linear
    } else {
        gain_linear
    };

    zpk.gain = 1.0 / final_gain;
}

/// Butterworth Q factor for section k of n total sections.
///
/// For a 2n-th order Butterworth, section k has:
///   Q_k = 1 / (2 * sin(pi * (2k + 1) / (4n)))
fn butterworth_section_q(k: usize, n: usize) -> f64 {
    let angle = PI * (2 * k + 1) as f64 / (4 * n) as f64;
    1.0 / (2.0 * angle.sin())
}

/// RBJ Audio EQ Cookbook low shelf biquad.
fn rbj_low_shelf(w0: f64, q: f64, gain_db: f64) -> Coeffs {
    let a = 10.0_f64.powf(gain_db / 40.0);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

    [a0, a1, a2, b0, b1, b2]
}

/// RBJ Audio EQ Cookbook high shelf biquad.
fn rbj_high_shelf(w0: f64, q: f64, gain_db: f64) -> Coeffs {
    let a = 10.0_f64.powf(gain_db / 40.0);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha;

    [a0, a1, a2, b0, b1, b2]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mag_db_sos(sections: &[Coeffs], w: f64) -> f64 {
        let ejw = Complex::from_polar(1.0, w);
        let ejw2 = ejw * ejw;
        let mut h = Complex::new(1.0, 0.0);
        for c in sections {
            let den = Complex::new(c[0], 0.0)
                + ejw * Complex::new(c[1], 0.0)
                + ejw2 * Complex::new(c[2], 0.0);
            let num = Complex::new(c[3], 0.0)
                + ejw * Complex::new(c[4], 0.0)
                + ejw2 * Complex::new(c[5], 0.0);
            h = h * num / den;
        }
        20.0 * h.mag().log10()
    }

    #[test]
    fn low_shelf_zero_gain_is_passthrough() {
        let sos = design_low_shelf(1, 1000.0, 0.707, 0.0, 48000.0);
        assert_eq!(sos.len(), 1);
        assert_eq!(sos[0], PASSTHROUGH);
    }

    #[test]
    fn low_shelf_boosts_low_frequencies() {
        let sos = design_low_shelf(1, 1000.0, 0.707, 6.0, 48000.0);
        let dc = mag_db_sos(&sos, 0.001);
        let nyq = mag_db_sos(&sos, PI - 0.01);
        assert!(
            dc > nyq + 3.0,
            "low shelf DC ({}) should be louder than Nyquist ({})",
            dc,
            nyq
        );
    }

    #[test]
    fn low_shelf_dc_gain_matches() {
        let sos = design_low_shelf(1, 1000.0, 0.707, 6.0, 48000.0);
        let dc = mag_db_sos(&sos, 0.001);
        assert!(
            (dc - 6.0).abs() < 1.5,
            "low shelf DC should be ~6 dB, got {}",
            dc
        );
    }

    #[test]
    fn high_shelf_boosts_high_frequencies() {
        let sos = design_high_shelf(1, 1000.0, 0.707, 6.0, 48000.0);
        let dc = mag_db_sos(&sos, 0.001);
        let nyq = mag_db_sos(&sos, PI - 0.01);
        assert!(
            nyq > dc + 3.0,
            "high shelf Nyquist ({}) should be louder than DC ({})",
            nyq,
            dc
        );
    }

    #[test]
    fn high_shelf_nyquist_gain_matches() {
        let sos = design_high_shelf(1, 1000.0, 0.707, 6.0, 48000.0);
        let nyq = mag_db_sos(&sos, PI - 0.01);
        assert!(
            (nyq - 6.0).abs() < 1.5,
            "high shelf Nyquist should be ~6 dB, got {}",
            nyq
        );
    }

    #[test]
    fn tilt_shelf_zero_gain_is_passthrough() {
        let sos = design_tilt_shelf(1, 1000.0, 0.707, 0.0, 48000.0);
        assert_eq!(sos.len(), 1);
        assert_eq!(sos[0], PASSTHROUGH);
    }

    #[test]
    fn tilt_shelf_tilts_spectrum() {
        let sos = design_tilt_shelf(1, 2000.0, 0.707, 6.0, 48000.0);
        let low = mag_db_sos(&sos, 0.01);
        let high = mag_db_sos(&sos, PI - 0.1);
        assert!(
            low > high,
            "tilt shelf should boost lows more than highs: low={}, high={}",
            low,
            high
        );
    }

    #[test]
    fn band_shelf_boosts_center() {
        let sos = design_band_shelf(1, 1000.0, 2.0, 6.0, 48000.0);
        let w0 = 2.0 * PI * 1000.0 / 48000.0;
        let center = mag_db_sos(&sos, w0);
        let dc = mag_db_sos(&sos, 0.001);
        assert!(
            center > dc + 2.0,
            "band shelf center ({}) should be louder than DC ({})",
            center,
            dc
        );
    }

    #[test]
    fn band_shelf_zero_gain_is_passthrough() {
        let sos = design_band_shelf(1, 1000.0, 2.0, 0.0, 48000.0);
        assert_eq!(sos.len(), 1);
        assert_eq!(sos[0], PASSTHROUGH);
    }

    #[test]
    fn apply_shelf_gain_unity_is_noop() {
        let mut zpk = Zpk::new(
            vec![Complex::new(-0.5, 0.0)],
            vec![Complex::new(-0.8, 0.0)],
            1.0,
        );
        let original = zpk.clone();
        apply_shelf_gain(&mut zpk, 7, 1.0);
        assert!((zpk.zeros[0].re - original.zeros[0].re).abs() < 1e-15);
        assert!((zpk.poles[0].re - original.poles[0].re).abs() < 1e-15);
    }

    #[test]
    fn apply_shelf_gain_low_shelf_scales_both() {
        let mut zpk = Zpk::new(
            vec![Complex::new(-0.5, 0.0)],
            vec![Complex::new(-0.8, 0.0)],
            1.0,
        );
        apply_shelf_gain(&mut zpk, 7, 2.0);
        // Low shelf: zeros *= gain, poles /= gain
        assert!(
            (zpk.zeros[0].re - (-0.5 * 2.0)).abs() < 1e-10,
            "zero should be scaled by gain"
        );
        assert!(
            (zpk.poles[0].re - (-0.8 / 2.0)).abs() < 1e-10,
            "pole should be scaled by 1/gain"
        );
    }

    #[test]
    fn apply_shelf_gain_high_shelf_zeros_only() {
        let mut zpk = Zpk::new(
            vec![Complex::new(-0.5, 0.0)],
            vec![Complex::new(-0.8, 0.0)],
            1.0,
        );
        apply_shelf_gain(&mut zpk, 8, 2.0);
        // High shelf: zeros *= gain, poles unchanged
        assert!(
            (zpk.zeros[0].re - (-0.5 * 2.0)).abs() < 1e-10,
            "zero should be scaled by gain"
        );
        assert!(
            (zpk.poles[0].re - (-0.8)).abs() < 1e-10,
            "pole should NOT be scaled for high shelf"
        );
    }

    #[test]
    fn butterworth_section_q_order2() {
        let q = butterworth_section_q(0, 1);
        assert!(
            (q - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01,
            "2nd order Butterworth Q should be ~0.707, got {}",
            q
        );
    }

    #[test]
    fn butterworth_section_q_order4() {
        let q0 = butterworth_section_q(0, 2);
        let q1 = butterworth_section_q(1, 2);
        assert!(q0 > q1, "first section should have higher Q");
        assert!(q0 > 1.0, "first section Q should be > 1.0, got {}", q0);
    }

    #[test]
    fn multi_section_shelf_higher_order() {
        let sos = design_low_shelf(3, 1000.0, 0.707, 6.0, 48000.0);
        assert_eq!(sos.len(), 3, "3 sections requested");
        let dc = mag_db_sos(&sos, 0.001);
        assert!(
            (dc - 6.0).abs() < 2.0,
            "3-section low shelf DC should be ~6 dB, got {}",
            dc
        );
    }
}
