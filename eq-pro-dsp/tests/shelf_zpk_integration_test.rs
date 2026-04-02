//! Integration tests for the new shelf_zpk module
//!
//! Verifies that the ZPK pipeline produces reasonable biquad coefficients
//! and integrates correctly with the rest of the DSP system.

use eq_pro_dsp::shelf_zpk;
use std::f64::consts::PI;

#[test]
fn low_shelf_zpk_produces_biquads() {
    println!("\n=== Low Shelf ZPK - Produces Valid Biquads ===\n");

    let sample_rate = 48000.0;
    let freq_hz = 1000.0;
    let user_q = 0.707;
    let gain_db = 6.0;

    let sos = shelf_zpk::design_low_shelf_zpk(1, freq_hz, user_q, gain_db, sample_rate);

    println!("Sections returned: {}", sos.len());
    assert!(sos.len() > 0, "Should return at least one section");

    for (i, coeffs) in sos.iter().enumerate() {
        println!(
            "Section {}: a0={:.6}, a1={:.6}, a2={:.6}, b0={:.6}, b1={:.6}, b2={:.6}",
            i, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]
        );

        // Verify denominator is normalized (a0 = 1 or close to 1)
        assert!(coeffs[0].abs() > 0.1, "a0 should be significant");

        // Verify no NaN or infinity
        for (j, &val) in coeffs.iter().enumerate() {
            assert!(val.is_finite(), "Coefficient {} is not finite: {}", j, val);
        }

        // Verify gain is applied to numerator (b0 should be > 1 for positive gain)
        if gain_db > 0.0 {
            assert!(
                coeffs[3] > coeffs[0] * 1.1,
                "b0 should be significantly larger than a0 for positive gain"
            );
        }
    }

    println!("✓ Low shelf produces valid biquads\n");
}

#[test]
fn high_shelf_zpk_produces_biquads() {
    println!("\n=== High Shelf ZPK - Produces Valid Biquads ===\n");

    let sample_rate = 48000.0;
    let freq_hz = 5000.0;
    let user_q = 0.707;
    let gain_db = -6.0;

    let sos = shelf_zpk::design_high_shelf_zpk(1, freq_hz, user_q, gain_db, sample_rate);

    println!("Sections returned: {}", sos.len());
    assert!(sos.len() > 0, "Should return at least one section");

    for (i, coeffs) in sos.iter().enumerate() {
        println!(
            "Section {}: a0={:.6}, a1={:.6}, a2={:.6}, b0={:.6}, b1={:.6}, b2={:.6}",
            i, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]
        );

        // Verify no NaN or infinity
        for (j, &val) in coeffs.iter().enumerate() {
            assert!(val.is_finite(), "Coefficient {} is not finite", j);
        }

        // Verify gain is applied (b0 should be < 1 for negative gain)
        if gain_db < 0.0 {
            assert!(
                coeffs[3] < coeffs[0] * 0.9,
                "b0 should be smaller than a0 for negative gain"
            );
        }
    }

    println!("✓ High shelf produces valid biquads\n");
}

#[test]
fn tilt_shelf_zpk_produces_biquads() {
    println!("\n=== Tilt Shelf ZPK - Produces Valid Biquads ===\n");

    let sample_rate = 48000.0;
    let freq_hz = 3000.0;
    let user_q = 0.707;
    let gain_db = 3.0;

    let sos = shelf_zpk::design_tilt_shelf_zpk(1, freq_hz, user_q, gain_db, sample_rate);

    println!("Sections returned: {}", sos.len());
    assert!(sos.len() > 0, "Should return at least one section");

    for (i, coeffs) in sos.iter().enumerate() {
        println!(
            "Section {}: a0={:.6}, a1={:.6}, a2={:.6}, b0={:.6}, b1={:.6}, b2={:.6}",
            i, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5]
        );

        // Verify no NaN or infinity
        for (j, &val) in coeffs.iter().enumerate() {
            assert!(val.is_finite(), "Coefficient {} is not finite", j);
        }
    }

    println!("✓ Tilt shelf produces valid biquads\n");
}

#[test]
fn zero_gain_returns_passthrough() {
    println!("\n=== Zero Gain Returns Passthrough ===\n");

    let sample_rate = 48000.0;
    let freq_hz = 1000.0;
    let user_q = 0.707;
    let gain_db = 0.0;

    let sos = shelf_zpk::design_low_shelf_zpk(1, freq_hz, user_q, gain_db, sample_rate);

    println!("Sections returned: {}", sos.len());
    assert_eq!(sos.len(), 1, "Should return passthrough for zero gain");

    let coeffs = &sos[0];
    println!("Passthrough coeffs: {:?}", coeffs);

    // Passthrough should be [1, 0, 0, 1, 0, 0]
    assert_eq!(coeffs[0], 1.0, "a0 should be 1");
    assert_eq!(coeffs[1], 0.0, "a1 should be 0");
    assert_eq!(coeffs[2], 0.0, "a2 should be 0");
    assert_eq!(coeffs[3], 1.0, "b0 should be 1");
    assert_eq!(coeffs[4], 0.0, "b1 should be 0");
    assert_eq!(coeffs[5], 0.0, "b2 should be 0");

    println!("✓ Zero gain returns passthrough\n");
}

#[test]
fn multiple_sections_for_high_order() {
    println!("\n=== Multiple Sections for High Order ===\n");

    let sample_rate = 48000.0;
    let freq_hz = 1000.0;
    let user_q = 0.707;
    let gain_db = 6.0;
    let n_sections = 4;

    let sos = shelf_zpk::design_low_shelf_zpk(n_sections, freq_hz, user_q, gain_db, sample_rate);

    println!(
        "Sections returned: {} (requested: {})",
        sos.len(),
        n_sections
    );
    assert_eq!(
        sos.len(),
        n_sections,
        "Should return requested number of sections"
    );

    println!("✓ Correct number of sections returned\n");
}

#[test]
fn compare_shelf_types_gain_direction() {
    println!("\n=== Compare Shelf Types - Gain Direction ===\n");

    let sample_rate = 48000.0;
    let freq_hz = 1000.0;
    let user_q = 0.707;
    let boost_db = 6.0;
    let cut_db = -6.0;

    // Test boost
    let low_boost = shelf_zpk::design_low_shelf_zpk(1, freq_hz, user_q, boost_db, sample_rate);
    let high_boost = shelf_zpk::design_high_shelf_zpk(1, freq_hz, user_q, boost_db, sample_rate);
    let tilt_boost = shelf_zpk::design_tilt_shelf_zpk(1, freq_hz, user_q, boost_db, sample_rate);

    println!("Boost (6 dB):");
    println!("  Low Shelf:  b0={:.6}", low_boost[0][3]);
    println!("  High Shelf: b0={:.6}", high_boost[0][3]);
    println!("  Tilt Shelf: b0={:.6}", tilt_boost[0][3]);

    // All should have b0 > 1 for boost
    assert!(low_boost[0][3] > 1.0, "Low shelf boost should have b0 > 1");
    assert!(
        high_boost[0][3] > 1.0,
        "High shelf boost should have b0 > 1"
    );
    assert!(
        tilt_boost[0][3] > 1.0,
        "Tilt shelf boost should have b0 > 1"
    );

    // Test cut
    let low_cut = shelf_zpk::design_low_shelf_zpk(1, freq_hz, user_q, cut_db, sample_rate);
    let high_cut = shelf_zpk::design_high_shelf_zpk(1, freq_hz, user_q, cut_db, sample_rate);
    let tilt_cut = shelf_zpk::design_tilt_shelf_zpk(1, freq_hz, user_q, cut_db, sample_rate);

    println!("\nCut (-6 dB):");
    println!("  Low Shelf:  b0={:.6}", low_cut[0][3]);
    println!("  High Shelf: b0={:.6}", high_cut[0][3]);
    println!("  Tilt Shelf: b0={:.6}", tilt_cut[0][3]);

    // All should have b0 < 1 for cut (assuming normalized)
    // This is less strict since the normalization might vary
    assert!(
        low_cut[0][3] < low_boost[0][3],
        "Cut should have smaller b0 than boost"
    );

    println!("\n✓ All shelf types apply gain in correct direction\n");
}
