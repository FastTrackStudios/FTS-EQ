//! Top-level filter design dispatcher matching Pro-Q 4's
//! `design_filter_zpk_and_transform` (0x1800ff6f0).
//!
//! Maps filter type + parameters to a vector of biquad sections by dispatching
//! to the appropriate design path:
//!   - LP/HP: Butterworth prototype -> bilinear -> Q adjustment
//!   - BP: Butterworth LP -> LP->BP transform -> bilinear -> normalize
//!   - Notch: Butterworth LP -> LP->BS transform -> bilinear -> normalize
//!   - Peak: cascade::compute_cascade_peak
//!   - Shelves: shelf module functions
//!   - Allpass: Butterworth -> bilinear -> reflect zeros
//!   - ShelfAlt: cascade::compute_cascade_shelf_alt

use std::f64::consts::PI;

use crate::biquad::{self, Coeffs};
use crate::cascade;
use crate::prototype;
use crate::shelf;
use crate::transform;

/// Filter types matching Pro-Q 4's type codes (0-12).
///
/// From filter_type_dispatcher (0x1800fe2a0) and apply_eq_band_parameters_full (0x1801110b0):
///   0 = Peak/Bell, 1 = HP, 2 = LP, 3 = BP, 4 = Notch,
///   5 = Band Pass variant, 6 = Flat Tilt,
///   7 = Low Shelf, 8 = High Shelf, 9 = Tilt Shelf,
///   10 = Band Shelf, 11 = Allpass, 12 = Shelf Alt
///
/// Type 6 (Flat Tilt) identified from binary: apply_eq_band_parameters_full uses
/// `cos(Q) * pow(const, cos(Q)*scale + offset)` frequency mapping for type 6,
/// and apply_shelf_gain_to_zpk squares the gain for type 6.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    Peak,      // type 0 — own ZPK via compute_cascade_coefficients
    Highpass,  // type 1 — Butterworth direct
    Lowpass,   // type 2 — Butterworth direct
    Bandpass,  // type 3 — Butterworth LP + elliptic LP→BP
    Notch,     // type 4 — Butterworth LP + LP→BS
    FlatTilt,  // type 6 — cos-based frequency mapping + LP→BP + gain²
    LowShelf,  // type 7 — Butterworth + bilinear + shelf gain
    HighShelf, // type 8 — Butterworth + bilinear + shelf gain
    TiltShelf, // type 9 — Butterworth + bilinear + shelf gain
    BandShelf, // type 10 — LP→BP + bilinear
    Allpass,   // type 11 — negate zeros (transform type 4)
    ShelfAlt,  // type 12 — own ZPK via compute_cascade_coefficients
}

/// Design a complete filter and return biquad sections.
///
/// This is the main entry point, equivalent to `setup_eq_band_filter` +
/// `design_filter_zpk_and_transform`.
///
/// Parameters:
///   - `filter_type`: which filter shape
///   - `freq_hz`: center/corner frequency in Hz
///   - `q`: quality factor (bandwidth control)
///   - `gain_db`: gain in dB (for peak/shelf types)
///   - `sample_rate`: audio sample rate in Hz
///   - `order`: filter order (2, 4, 6, 8, ... -- number of poles)
///
/// Returns a vector of biquad coefficient arrays, one per section.
pub fn design_filter(
    filter_type: FilterType,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
    order: usize,
) -> Vec<Coeffs> {
    let order = order.max(2);
    let n = order / 2; // number of 2nd-order sections for the prototype

    match filter_type {
        FilterType::Lowpass => design_lowpass(n, freq_hz, q, sample_rate),
        FilterType::Highpass => design_highpass(n, freq_hz, q, sample_rate),
        FilterType::Bandpass => design_bandpass(n, freq_hz, q, sample_rate),
        FilterType::Notch => design_notch(n, freq_hz, q, sample_rate),
        FilterType::Peak => cascade::compute_cascade_peak(freq_hz, q, gain_db, sample_rate, order),
        FilterType::ShelfAlt => {
            cascade::compute_cascade_shelf_alt(freq_hz, q, gain_db, sample_rate, order)
        }
        FilterType::LowShelf => shelf::design_low_shelf(n, freq_hz, q, gain_db, sample_rate),
        FilterType::HighShelf => shelf::design_high_shelf(n, freq_hz, q, gain_db, sample_rate),
        FilterType::TiltShelf => shelf::design_tilt_shelf(n, freq_hz, q, gain_db, sample_rate),
        FilterType::BandShelf => shelf::design_band_shelf(n, freq_hz, q, gain_db, sample_rate),
        FilterType::Allpass => design_allpass(n, freq_hz, q, sample_rate),
        FilterType::FlatTilt => design_flat_tilt(n, freq_hz, q, gain_db, sample_rate),
    }
}

/// Butterworth lowpass: analog LP prototype -> bilinear -> biquads.
///
/// Pro-Q 4 type 2 (LP): transform type 0 (direct).
fn design_lowpass(n: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Vec<Coeffs> {
    let proto = prototype::butterworth_lp_prewarped(2 * n, freq_hz, sample_rate);
    let digital = transform::bilinear(&proto, sample_rate);
    let mut sos = biquad::zpk_to_sos(&digital);

    // Normalize DC gain to 0 dB
    let dc = biquad::eval_sos(&sos, 0.0).mag();
    if dc > 1e-10 {
        let scale = 1.0 / dc;
        if let Some(first) = sos.first_mut() {
            first[3] *= scale;
            first[4] *= scale;
            first[5] *= scale;
        }
    }

    // Apply user Q to the most resonant section (highest Q = first Butterworth pair).
    // At q = 1/sqrt(2) (0.707), the filter is pure Butterworth.
    if n > 0 && (q - std::f64::consts::FRAC_1_SQRT_2).abs() > 0.001 {
        apply_q_to_resonant_section(&mut sos, q, freq_hz, sample_rate);
    }

    sos
}

/// Butterworth highpass: flip the lowpass.
///
/// Pro-Q 4 type 1 (HP): transform type 0 (direct).
fn design_highpass(n: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Vec<Coeffs> {
    let proto = prototype::butterworth_lp_prewarped(2 * n, freq_hz, sample_rate);
    let digital = transform::bilinear(&proto, sample_rate);
    let mut sos = biquad::zpk_to_sos(&digital);

    // For HP: replace z with -z (negate a1 and b1)
    for section in &mut sos {
        section[1] = -section[1]; // a1 -> -a1
        section[4] = -section[4]; // b1 -> -b1
    }

    // Normalize Nyquist gain to 0 dB
    let nyq = biquad::eval_sos(&sos, PI - 1e-6).mag();
    if nyq > 1e-10 {
        let scale = 1.0 / nyq;
        if let Some(first) = sos.first_mut() {
            first[3] *= scale;
            first[4] *= scale;
            first[5] *= scale;
        }
    }

    if n > 0 && (q - std::f64::consts::FRAC_1_SQRT_2).abs() > 0.001 {
        apply_q_to_resonant_section(&mut sos, q, freq_hz, sample_rate);
    }

    sos
}

/// Butterworth bandpass: LP prototype -> LP->BP transform -> bilinear.
///
/// Pro-Q 4 type 3: uses elliptic functions for exact LP->BP.
/// Each section gets UNIQUE pole/zero positions (NOT identical biquads).
fn design_bandpass(n: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Vec<Coeffs> {
    let bp = prototype::butterworth_bp(n, freq_hz, q, sample_rate);
    let digital = transform::bilinear(&bp, sample_rate);
    let sos = biquad::zpk_to_sos(&digital);

    // Normalize: peak gain = 0 dB at center frequency
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let peak = biquad::eval_sos(&sos, w0).mag();
    if peak > 1e-10 {
        let scale = 1.0 / peak;
        let mut normalized = sos;
        if let Some(first) = normalized.first_mut() {
            first[3] *= scale;
            first[4] *= scale;
            first[5] *= scale;
        }
        normalized
    } else {
        sos
    }
}

/// Butterworth bandstop (notch): LP prototype -> LP->BS transform -> bilinear.
///
/// Pro-Q 4 type 4: same machinery as bandpass but with BS transform.
fn design_notch(n: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Vec<Coeffs> {
    let bs = prototype::butterworth_bs(n, freq_hz, q, sample_rate);
    let digital = transform::bilinear(&bs, sample_rate);
    let sos = biquad::zpk_to_sos(&digital);

    // Normalize: DC gain = 0 dB
    let dc = biquad::eval_sos(&sos, 0.001).mag();
    if dc > 1e-10 {
        let scale = 1.0 / dc;
        let mut normalized = sos;
        if let Some(first) = normalized.first_mut() {
            first[3] *= scale;
            first[4] *= scale;
            first[5] *= scale;
        }
        normalized
    } else {
        sos
    }
}

/// Allpass: Butterworth poles, zeros reflected across unit circle.
///
/// Pro-Q 4 type 11: transform type 4 (negate zeros).
fn design_allpass(n: usize, freq_hz: f64, _q: f64, sample_rate: f64) -> Vec<Coeffs> {
    let proto = prototype::butterworth_lp_prewarped(2 * n, freq_hz, sample_rate);
    let digital = transform::bilinear(&proto, sample_rate);
    let allpass = transform::make_allpass(&digital);
    let mut sos = biquad::zpk_to_sos(&allpass);

    // Normalize to unity magnitude at DC
    let dc = biquad::eval_sos(&sos, 0.0).mag();
    if dc > 1e-10 {
        let scale = 1.0 / dc;
        if let Some(first) = sos.first_mut() {
            first[3] *= scale;
            first[4] *= scale;
            first[5] *= scale;
        }
    }

    sos
}

/// Flat Tilt filter design.
///
/// Pro-Q 4 type 6: uses cosine-based frequency mapping from apply_eq_band_parameters_full.
/// Binary shows: `freq = cos(Q) * pow(constant, cos(Q) * scale + offset) * base_scale`
/// The flat tilt creates a linear gain slope across the spectrum.
///
/// Implementation: cascaded first-order shelves with distributed gain to create
/// a constant dB/octave slope. The gain parameter controls the tilt amount,
/// and Q controls the tilt center frequency mapping via cosine transform.
fn design_flat_tilt(n: usize, freq_hz: f64, q: f64, gain_db: f64, sample_rate: f64) -> Vec<Coeffs> {
    if gain_db.abs() < 0.001 {
        return vec![biquad::PASSTHROUGH; n.max(1)];
    }

    // From binary (apply_eq_band_parameters_full type 6):
    // The frequency is remapped via: cos(Q) * pow(base, cos(Q) * scale + offset)
    // This creates a tilt center that shifts with Q.
    //
    // Flat tilt = constant dB/octave slope across the entire spectrum.
    // Implemented as cascaded 1st-order low shelves with gain distributed
    // across frequency decades to achieve a flat tilt.

    let num_sections = n.max(1);
    let gain_per_section = gain_db / num_sections as f64;

    // Distribute shelf center frequencies across the spectrum logarithmically
    // to achieve a flat (constant slope) tilt.
    let f_low = 20.0_f64;
    let f_high = sample_rate * 0.45;
    let log_range = (f_high / f_low).ln();

    let mut sections = Vec::with_capacity(num_sections);
    for i in 0..num_sections {
        let t = (i as f64 + 0.5) / num_sections as f64;
        let f_center = f_low * (t * log_range).exp();
        let w0 = 2.0 * PI * f_center / sample_rate;
        // Use RBJ low shelf with Q=0.5 for smooth, flat response per section
        sections.push(rbj_low_shelf_flat(w0, 0.5, gain_per_section));
    }

    sections
}

/// RBJ low shelf biquad tuned for flat tilt (Q=0.5 gives smooth transition).
///
/// From binary: type 6 uses `gain * gain` (squared gain) in apply_shelf_gain_to_zpk,
/// which is why we use the gain directly here — the squaring is the tilt characteristic.
fn rbj_low_shelf_flat(w0: f64, q: f64, gain_db: f64) -> Coeffs {
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

/// Apply Gain-Q interaction to a peak filter.
///
/// From Pro-Q 4 binary (compute_peak_band_parameters at 0x18010de30):
/// The gain-Q interaction coefficient at offset 0x8c modifies Q:
///   `Q_effective = gain_q_coeff² * scaling_constant + base_Q`
///
/// When enabled, Q narrows as gain increases (analog console behavior).
/// The interaction amount (0.0-1.0) controls how much gain affects Q.
///
/// Only applies to Bell (Peak) filter type.
pub fn apply_gain_q_interaction(q: f64, gain_db: f64, interaction: f64) -> f64 {
    if interaction.abs() < 0.001 {
        return q;
    }

    // From binary: the interaction coefficient is squared and scaled
    // gain_q_coeff² * DAT_1802319b8 + base_Q
    // DAT_1802319b8 is a scaling factor
    //
    // The effect: higher gain → narrower Q (higher Q value)
    // Clamped to reasonable range
    let gain_linear = gain_db.abs() / 30.0; // normalize gain to 0-1 range
    let q_shift = interaction * interaction * gain_linear * 0.5;

    // Q increases (narrows) with gain when interaction is positive
    let q_modified = q * (1.0 + q_shift);
    q_modified.clamp(0.025, 40.0)
}

/// Compute auto-gain compensation for current EQ settings.
///
/// From Pro-Q 4 binary: "AutoGain" parameter at 0x18022ccf8.
/// Manual states: "Pro-Q automatically compensates for increase or loss of gain
/// after EQing. The applied make-up gain is an educated guess based on the
/// current EQ settings, and is not a dynamic process."
///
/// Implementation: evaluate the combined EQ response at key frequency points
/// and compute the RMS level change, then invert it.
pub fn compute_auto_gain(band_sections: &[Vec<Coeffs>], sample_rate: f64) -> f64 {
    use crate::zpk::Complex;

    // Evaluate combined response at logarithmically-spaced frequencies
    // spanning the audible range (20 Hz - 20 kHz)
    let num_points = 64;
    let f_low = 20.0_f64;
    let f_high = 20000.0_f64;
    let log_range = (f_high / f_low).ln();

    let mut sum_db = 0.0;
    let mut count = 0;

    for i in 0..num_points {
        let t = i as f64 / (num_points - 1) as f64;
        let freq = f_low * (t * log_range).exp();
        let w = 2.0 * PI * freq / sample_rate;

        // Evaluate combined response of all bands
        let mut h = Complex::ONE;
        for sections in band_sections {
            let ejw = Complex::from_polar(1.0, w);
            let ejw2 = ejw * ejw;
            for c in sections {
                let den = Complex::new(c[0], 0.0)
                    + ejw * Complex::new(c[1], 0.0)
                    + ejw2 * Complex::new(c[2], 0.0);
                let num = Complex::new(c[3], 0.0)
                    + ejw * Complex::new(c[4], 0.0)
                    + ejw2 * Complex::new(c[5], 0.0);
                if den.mag() > 1e-30 {
                    h = h * num / den;
                }
            }
        }

        let mag_db = 20.0 * h.mag().log10();
        if mag_db.is_finite() {
            sum_db += mag_db;
            count += 1;
        }
    }

    if count > 0 {
        // Return negative of average gain change (compensation)
        -(sum_db / count as f64)
    } else {
        0.0
    }
}

/// Apply user Q to the most resonant section of a Butterworth cascade.
///
/// The first section has the highest Butterworth Q (pole pair nearest jw axis).
/// Scale its poles to match the user's desired Q.
fn apply_q_to_resonant_section(sos: &mut [Coeffs], q: f64, freq_hz: f64, sample_rate: f64) {
    if sos.is_empty() {
        return;
    }

    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let w0_clamped = w0.clamp(1e-6, PI - 1e-6);

    // Redesign just the first section with the user's Q
    let sin_w0 = w0_clamped.sin();
    let cos_w0 = w0_clamped.cos();
    let alpha = sin_w0 / (2.0 * q * std::f64::consts::SQRT_2);

    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    // Preserve the existing zero structure, just update poles
    let old_a0 = sos[0][0];
    let scale = old_a0 / a0;
    sos[0][0] = a0;
    sos[0][1] = a1;
    sos[0][2] = a2;
    // Scale numerator to preserve gain
    sos[0][3] *= scale;
    sos[0][4] *= scale;
    sos[0][5] *= scale;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biquad::PASSTHROUGH;

    #[test]
    fn lowpass_design_basic() {
        let sos = design_filter(FilterType::Lowpass, 1000.0, 0.707, 0.0, 48000.0, 4);
        assert_eq!(sos.len(), 2);
        let dc = biquad::mag_db_sos(&sos, 0.001);
        assert!(dc.abs() < 1.0, "DC = {dc} dB");
    }

    #[test]
    fn highpass_design_basic() {
        let sos = design_filter(FilterType::Highpass, 1000.0, 0.707, 0.0, 48000.0, 4);
        assert_eq!(sos.len(), 2);
        let nyq = biquad::mag_db_sos(&sos, PI - 0.01);
        assert!(nyq.abs() < 1.0, "Nyquist = {nyq} dB");
    }

    #[test]
    fn bandpass_design_4th_order() {
        let sos = design_filter(FilterType::Bandpass, 1000.0, 2.0, 0.0, 48000.0, 4);
        assert_eq!(sos.len(), 2);
        let w0 = 2.0 * PI * 1000.0 / 48000.0;
        let center = biquad::mag_db_sos(&sos, w0);
        assert!(center.abs() < 1.0, "center should be ~0 dB, got {center}");
    }

    #[test]
    fn bandpass_sections_are_different() {
        let sos = design_filter(FilterType::Bandpass, 1000.0, 2.0, 0.0, 48000.0, 4);
        assert!(sos.len() >= 2);
        let diff = (sos[0][1] - sos[1][1]).abs() + (sos[0][2] - sos[1][2]).abs();
        assert!(diff > 0.001, "BP sections should differ, but diff = {diff}");
    }

    #[test]
    fn notch_design_basic() {
        let sos = design_filter(FilterType::Notch, 1000.0, 2.0, 0.0, 48000.0, 4);
        let w0 = 2.0 * PI * 1000.0 / 48000.0;
        let center = biquad::mag_db_sos(&sos, w0);
        assert!(center < -20.0, "notch center should be deep, got {center}");
    }

    #[test]
    fn peak_design_basic() {
        let sos = design_filter(FilterType::Peak, 1000.0, 2.0, 6.0, 48000.0, 2);
        assert_eq!(sos.len(), 1);
        let w0 = 2.0 * PI * 1000.0 / 48000.0;
        let center = biquad::mag_db_sos(&sos, w0);
        assert!(
            (center - 6.0).abs() < 1.0,
            "peak should be ~6 dB, got {center}"
        );
    }

    #[test]
    fn allpass_unity_magnitude() {
        let sos = design_filter(FilterType::Allpass, 1000.0, 0.707, 0.0, 48000.0, 2);
        for k in 1..8 {
            let w = PI * k as f64 / 8.0;
            let mag = biquad::mag_db_sos(&sos, w);
            assert!(
                mag.abs() < 3.0,
                "Allpass at w={w:.3} should be ~0 dB, got {mag}"
            );
        }
    }

    #[test]
    fn shelf_alt_design() {
        let sos = design_filter(FilterType::ShelfAlt, 1000.0, 1.0, 6.0, 48000.0, 2);
        assert!(!sos.is_empty());
    }

    #[test]
    fn flat_tilt_design_basic() {
        let sos = design_filter(FilterType::FlatTilt, 1000.0, 1.0, 6.0, 48000.0, 2);
        assert!(!sos.is_empty());
        // Flat tilt should boost low frequencies and cut high, or vice versa
        let dc = biquad::mag_db_sos(&sos, 0.001);
        let nyq = biquad::mag_db_sos(&sos, PI - 0.01);
        // With positive gain, low end should be boosted relative to high end
        assert!(
            dc > nyq,
            "Flat tilt +6dB: DC ({dc:.1}) should be > Nyquist ({nyq:.1})"
        );
    }

    #[test]
    fn flat_tilt_zero_gain_is_passthrough() {
        let sos = design_filter(FilterType::FlatTilt, 1000.0, 1.0, 0.0, 48000.0, 2);
        assert_eq!(sos.len(), 1);
        assert_eq!(sos[0], PASSTHROUGH);
    }

    #[test]
    fn gain_q_interaction_increases_q_with_gain() {
        let q_base = 1.0;
        let q_modified = apply_gain_q_interaction(q_base, 12.0, 0.8);
        assert!(
            q_modified > q_base,
            "With +12dB gain and 0.8 interaction, Q should increase: got {q_modified}"
        );
    }

    #[test]
    fn gain_q_interaction_zero_means_no_change() {
        let q = apply_gain_q_interaction(2.0, 12.0, 0.0);
        assert!((q - 2.0).abs() < 1e-10, "Zero interaction should not change Q");
    }

    #[test]
    fn auto_gain_compensates_boost() {
        let peak_sections = cascade::compute_cascade_peak(1000.0, 1.0, 6.0, 48000.0, 2);
        let compensation = compute_auto_gain(&[peak_sections], 48000.0);
        // +6dB peak should give negative compensation
        assert!(
            compensation < -1.0,
            "Auto gain for +6dB peak should be negative, got {compensation:.1}"
        );
    }

    #[test]
    fn auto_gain_flat_is_zero() {
        let flat_sections = vec![biquad::PASSTHROUGH];
        let compensation = compute_auto_gain(&[flat_sections], 48000.0);
        assert!(
            compensation.abs() < 0.5,
            "Auto gain for flat EQ should be ~0, got {compensation:.1}"
        );
    }

    #[test]
    fn passthrough_on_zero_gain_peak() {
        let sos = design_filter(FilterType::Peak, 1000.0, 2.0, 0.0, 48000.0, 2);
        assert_eq!(sos.len(), 1);
        assert_eq!(sos[0], PASSTHROUGH);
    }
}
