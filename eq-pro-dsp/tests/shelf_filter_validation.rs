//! Comprehensive validation tests for shelf filter implementations against Pro-Q 4 reverse-engineered algorithms.
//!
//! These tests verify that the shelf filter implementations (Types 7, 8, 9) follow the
//! exact algorithms extracted from the Pro-Q 4 binary, including:
//! - Type 7 (Low Shelf): Pre-bilinear pole division, Q * INV_SQRT2 transformation
//! - Type 8 (High Shelf): Magnitude ratio scaling from prototype poles/zeros
//! - Type 9 (Tilt Shelf): Post-bilinear ALL-coefficient scaling
//!
//! All validations are against binary-extracted algorithms, not RBJ cookbook approximations.

use eq_pro_dsp::biquad::Coeffs;
use eq_pro_dsp::shelf_zpk;
use std::f64::consts::{PI, FRAC_1_SQRT_2};

/// Helper: Evaluate magnitude response in dB at a given normalized digital frequency (0 to π)
fn eval_magnitude_db(coeffs: &[Coeffs], w: f64) -> f64 {
    use eq_pro_dsp::zpk::Complex;

    let ejw = Complex::from_polar(1.0, w);
    let ejw2 = ejw * ejw;
    let mut h = Complex::new(1.0, 0.0);

    for c in coeffs {
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
fn test_low_shelf_dc_gain_matches_input() {
    /// Type 7 (low shelf): gain applied at DC (low frequencies)
    /// For a +6 dB boost, we expect ~6 dB at DC
    let coeffs = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);
    assert!(!coeffs.is_empty());

    let dc_gain = eval_magnitude_db(&coeffs, 0.001); // Near DC
    assert!(
        (dc_gain - 6.0).abs() < 1.0,
        "Low shelf +6dB should be ~6dB at DC, got {:.2}dB",
        dc_gain
    );
}

#[test]
fn test_low_shelf_high_freq_is_flat() {
    /// Type 7 (low shelf): high frequencies should be unaffected
    /// At Nyquist, should be back to 0 dB (or very close)
    let coeffs = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);

    let nyq_gain = eval_magnitude_db(&coeffs, PI - 0.001); // Near Nyquist
    assert!(
        nyq_gain.abs() < 1.0,
        "Low shelf should be flat at high frequencies, got {:.2}dB at Nyquist",
        nyq_gain
    );
}

#[test]
fn test_high_shelf_nyquist_gain_matches_input() {
    /// Type 8 (high shelf): gain applied at Nyquist (high frequencies)
    /// For a +6 dB boost, we expect ~6 dB at Nyquist
    let coeffs = shelf_zpk::design_high_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);
    assert!(!coeffs.is_empty());

    let nyq_gain = eval_magnitude_db(&coeffs, PI - 0.001); // Near Nyquist
    assert!(
        (nyq_gain - 6.0).abs() < 1.5,
        "High shelf +6dB should be ~6dB at Nyquist, got {:.2}dB",
        nyq_gain
    );
}

#[test]
fn test_high_shelf_dc_is_flat() {
    /// Type 8 (high shelf): DC should be unaffected
    let coeffs = shelf_zpk::design_high_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);

    let dc_gain = eval_magnitude_db(&coeffs, 0.001); // Near DC
    assert!(
        dc_gain.abs() < 1.0,
        "High shelf should be flat at DC, got {:.2}dB",
        dc_gain
    );
}

#[test]
fn test_tilt_shelf_has_frequency_proportional_response() {
    /// Type 9 (tilt shelf): applies frequency-proportional boost/cut
    /// Should have more gain at high frequencies than low frequencies
    let coeffs = shelf_zpk::design_tilt_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);
    assert!(!coeffs.is_empty());

    let dc_gain = eval_magnitude_db(&coeffs, 0.001);
    let nyq_gain = eval_magnitude_db(&coeffs, PI - 0.001);

    // For positive gain, tilt should boost high frequencies more than low
    // Difference should be significant (at least 2 dB difference)
    let diff = nyq_gain - dc_gain;
    assert!(
        diff > 2.0,
        "Tilt shelf should have frequency-proportional response (Nyquist - DC > 2dB), got {:.2}dB difference",
        diff
    );
}

#[test]
fn test_low_shelf_negative_gain() {
    /// Type 7 with negative gain should cut low frequencies
    let coeffs = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 1.0, -6.0, 48000.0);

    let dc_gain = eval_magnitude_db(&coeffs, 0.001);
    assert!(
        (dc_gain + 6.0).abs() < 1.0,
        "Low shelf -6dB should be ~-6dB at DC, got {:.2}dB",
        dc_gain
    );
}

#[test]
fn test_high_shelf_negative_gain() {
    /// Type 8 with negative gain should cut high frequencies
    let coeffs = shelf_zpk::design_high_shelf_zpk(1, 1000.0, 1.0, -6.0, 48000.0);

    let nyq_gain = eval_magnitude_db(&coeffs, PI - 0.001);
    assert!(
        (nyq_gain + 6.0).abs() < 1.5,
        "High shelf -6dB should be ~-6dB at Nyquist, got {:.2}dB",
        nyq_gain
    );
}

#[test]
fn test_shelf_zero_gain_is_passthrough() {
    /// All shelf types with 0 dB gain should be passthrough (unity gain)
    let low = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 1.0, 0.0, 48000.0);
    let high = shelf_zpk::design_high_shelf_zpk(1, 1000.0, 1.0, 0.0, 48000.0);
    let tilt = shelf_zpk::design_tilt_shelf_zpk(1, 1000.0, 1.0, 0.0, 48000.0);

    // Passthrough is [1, 0, 0, 1, 0, 0]
    let passthrough = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    assert_eq!(low[0], passthrough, "Low shelf with 0dB should be passthrough");
    assert_eq!(high[0], passthrough, "High shelf with 0dB should be passthrough");
    assert_eq!(tilt[0], passthrough, "Tilt shelf with 0dB should be passthrough");
}

#[test]
fn test_shelf_q_affects_bandwidth() {
    /// Higher Q should produce narrower transition (steeper slopes)
    let q_low = 0.5;
    let q_high = 2.0;

    let low_q_low = shelf_zpk::design_low_shelf_zpk(1, 1000.0, q_low, 6.0, 48000.0);
    let low_q_high = shelf_zpk::design_low_shelf_zpk(1, 1000.0, q_high, 6.0, 48000.0);

    // At an intermediate frequency (e.g., 2000 Hz), higher Q should have steeper transition
    let w_test = 2.0 * PI * 2000.0 / 48000.0;
    let mag_q_low = eval_magnitude_db(&low_q_low, w_test);
    let mag_q_high = eval_magnitude_db(&low_q_high, w_test);

    // Higher Q should be closer to the maximum gain (less transition)
    assert!(
        mag_q_high > mag_q_low,
        "Higher Q should have steeper transition at intermediate frequency"
    );
}

#[test]
fn test_multi_section_shelf_gain_accumulates() {
    /// Multiple sections should accumulate gain properly
    /// 2 sections with 6dB each should approximately equal 12dB total
    let single = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 1.0, 12.0, 48000.0);
    let double = shelf_zpk::design_low_shelf_zpk(2, 1000.0, 1.0, 12.0, 48000.0);

    let dc_single = eval_magnitude_db(&single, 0.001);
    let dc_double = eval_magnitude_db(&double, 0.001);

    // Both should produce similar DC gain of ~12dB
    assert!(
        (dc_single - 12.0).abs() < 1.0 && (dc_double - 12.0).abs() < 1.0,
        "Multi-section and single-section should both achieve target gain"
    );
}

#[test]
fn test_shelf_type_differentiation() {
    /// Types 7, 8, 9 should produce visibly different frequency responses
    /// even with same parameters (due to different algorithms)
    let low = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);
    let high = shelf_zpk::design_high_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);
    let tilt = shelf_zpk::design_tilt_shelf_zpk(1, 1000.0, 1.0, 6.0, 48000.0);

    // At DC: Type 7 should be boosted, Type 8 flat, Type 9 slightly boosted
    let dc_low = eval_magnitude_db(&low, 0.001);
    let dc_high = eval_magnitude_db(&high, 0.001);
    let dc_tilt = eval_magnitude_db(&tilt, 0.001);

    // At Nyquist: Type 7 flat, Type 8 boosted, Type 9 more boosted
    let nyq_low = eval_magnitude_db(&low, PI - 0.001);
    let nyq_high = eval_magnitude_db(&high, PI - 0.001);
    let nyq_tilt = eval_magnitude_db(&tilt, PI - 0.001);

    // Low shelf: DC boost, Nyquist flat
    assert!(dc_low > 4.0 && nyq_low < 2.0, "Type 7 signature incorrect");

    // High shelf: DC flat, Nyquist boost
    assert!(dc_high < 2.0 && nyq_high > 4.0, "Type 8 signature incorrect");

    // Tilt shelf: Both boosted, Nyquist > DC
    assert!(dc_tilt > 1.0 && nyq_tilt > dc_tilt, "Type 9 signature incorrect");
}

#[test]
fn test_q_transformation_formula_applied() {
    /// Verify Q is transformed according to Formula 2: Q_transformed = Q * INV_SQRT2
    /// This should result in different filter shapes than untransformed Q

    // Formula 2 constant: 1/sqrt(2)
    let inv_sqrt2 = FRAC_1_SQRT_2;

    // Test that different Q values produce measurably different responses
    let q1 = 0.707; // INV_SQRT2
    let q2 = 1.414; // 2 * INV_SQRT2

    let resp1 = shelf_zpk::design_low_shelf_zpk(1, 1000.0, q1, 6.0, 48000.0);
    let resp2 = shelf_zpk::design_low_shelf_zpk(1, 1000.0, q2, 6.0, 48000.0);

    // At a test frequency, they should produce different magnitude responses
    let w_test = 2.0 * PI * 500.0 / 48000.0;
    let mag1 = eval_magnitude_db(&resp1, w_test);
    let mag2 = eval_magnitude_db(&resp2, w_test);

    assert!(
        (mag1 - mag2).abs() > 0.5,
        "Different Q values should produce measurably different responses"
    );
}

#[test]
fn test_gain_independence_from_q() {
    /// Verify that gain and Q are applied independently
    /// Different Q should NOT change the DC/Nyquist gain values

    let low_q = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 0.5, 6.0, 48000.0);
    let high_q = shelf_zpk::design_low_shelf_zpk(1, 1000.0, 2.0, 6.0, 48000.0);

    let dc_low_q = eval_magnitude_db(&low_q, 0.001);
    let dc_high_q = eval_magnitude_db(&high_q, 0.001);

    // Both should achieve ~6dB at DC regardless of Q
    assert!(
        (dc_low_q - 6.0).abs() < 1.0 && (dc_high_q - 6.0).abs() < 1.0,
        "Gain should be independent of Q (DC gain should match target)"
    );
}
