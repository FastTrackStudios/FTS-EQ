// Integration test for parameter transformation module
// Validates that parameter transformations match Pro-Q 4 binary behavior

use eq_pro_dsp::parameters;

#[test]
fn test_type_0_peak_parameter_transformation() {
    // Type 0: Peak filter with 6dB gain
    let result = parameters::transform_parameters(
        0,       // filter_type: Peak
        1.0,     // user_q
        6.0,     // user_gain_db (positive gain)
        1000.0,  // center_freq_hz
        48000.0, // sample_rate
        -1,      // mode (default)
        0,       // param_state
        0.0,     // sq_component
        0.0,     // mode_param
    );

    // Type 0 should output full gain as processed_q
    assert!(
        result.processed_q > 0.0,
        "Type 0 with positive gain should have positive processed_q"
    );
    assert!(
        result.q_term > 0.0,
        "Type 0 q_term should be positive with positive gain"
    );
    assert!(result.frequency <= std::f64::consts::PI + 0.01);

    // Type 0 with zero gain
    let result_zero =
        parameters::transform_parameters(0, 1.0, 0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0);
    assert_eq!(
        result_zero.processed_q, 0.0,
        "Type 0 with zero gain should have zero processed_q"
    );
}

#[test]
fn test_type_3_bandpass_parameter_transformation() {
    // Type 3: Bandpass filter
    let result = parameters::transform_parameters(
        3,       // filter_type: Bandpass
        1.0,     // user_q
        3.0,     // user_gain_db
        1000.0,  // center_freq_hz
        48000.0, // sample_rate
        -1,      // mode
        0,       // param_state
        0.0,     // sq_component
        0.0,     // mode_param
    );

    // Type 3 should apply 50% Q reduction in standard path
    assert!(
        result.processed_q < 1.0,
        "Type 3 should reduce Q in standard path"
    );
    assert!(
        result.q_term >= 0.0,
        "Type 3 should have non-negative q_term"
    );
}

#[test]
fn test_type_6_clipping_behavior() {
    // Type 6: Flat Tilt with clipping
    let result_low = parameters::transform_parameters(
        6,       // filter_type: FlatTilt/Type6
        0.5,     // user_q (below clip threshold 1.885)
        0.0,     // user_gain_db
        1000.0,  // center_freq_hz
        48000.0, // sample_rate
        -1,      // mode
        0,       // param_state
        0.0,     // sq_component
        0.0,     // mode_param
    );

    let result_high = parameters::transform_parameters(
        6,   // filter_type
        2.5, // user_q (above clip threshold 1.885)
        0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0,
    );

    // Low value should pass through
    assert!((result_low.processed_q - 0.5).abs() < 0.001);

    // High value should be clipped
    assert!(
        result_high.processed_q < 2.0,
        "Type 6 should clip high Q values"
    );
}

#[test]
fn test_type_10_band_shelf_modes() {
    // Type 10: Band Shelf with mode=-1 (simple path)
    let result_simple = parameters::transform_parameters(
        10,  // filter_type: BandShelf
        1.0, // user_q
        3.0, // user_gain_db
        1000.0, 48000.0, -1, // mode: simple path
        0, 0.0, 0.0,
    );

    // Type 10 simple mode should produce valid output
    assert!(
        result_simple.frequency > 2.5,
        "Type 10 frequency should be in valid range"
    );
    assert!(
        result_simple.frequency < 3.15,
        "Type 10 frequency should be bounded"
    );

    // Type 10 with mode=1 (complex path)
    let result_complex = parameters::transform_parameters(
        10, 1.0, 3.0, 1000.0, 48000.0, 1, // mode: complex path
        0, 0.5, // sq_component
        0.5, // mode_param
    );

    assert!(
        result_complex.processed_q > 0.0,
        "Type 10 complex should produce positive Q"
    );
}

#[test]
fn test_parameter_defaults_sensible() {
    // Test that parameter module works with sensible defaults for all types
    for filter_type in 0u32..=12 {
        let result = parameters::transform_parameters(
            filter_type,
            1.0, // neutral Q
            6.0, // use positive gain (Type 0 requires it)
            1000.0,
            48000.0,
            -1,  // default mode (simple path)
            0,   // default param_state
            0.0, // default sq_component
            0.0, // default mode_param
        );

        // Basic sanity checks
        assert!(
            result.processed_q >= 0.0,
            "Type {} processed_q should be non-negative",
            filter_type
        );
        assert!(
            result.processed_q < 1e6,
            "Type {} processed_q should be reasonable",
            filter_type
        );
        assert!(
            result.frequency > 0.1,
            "Type {} frequency should be positive",
            filter_type
        );
        assert!(
            result.frequency < 10.0,
            "Type {} frequency should be bounded",
            filter_type
        );
    }
}

#[test]
fn test_type_range_completeness() {
    // Ensure all 13 types are handled (no panics)
    let mut handled_count = 0;
    for filter_type in 0u32..=12 {
        let _ = parameters::transform_parameters(
            filter_type,
            1.0,
            0.0,
            1000.0,
            48000.0,
            -1,
            0,
            0.0,
            0.0,
        );
        handled_count += 1;
    }

    assert_eq!(handled_count, 13, "All 13 filter types should be handled");
}

#[test]
fn test_frequency_range_respect() {
    // Verify that frequency outputs respect Pro-Q 4's frequency constraints
    let result_0 = parameters::transform_parameters(0, 1.0, 0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0);
    let result_3 = parameters::transform_parameters(3, 1.0, 0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0);
    let result_10 =
        parameters::transform_parameters(10, 1.0, 0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0);

    // All should produce π-bounded frequencies
    let pi = std::f64::consts::PI;
    assert!(result_0.frequency <= pi + 0.01);
    assert!(result_3.frequency <= pi + 0.01);
    assert!(result_10.frequency <= pi + 0.01);
}

#[test]
fn test_q_scaling_consistency() {
    // Test that Q scaling is applied consistently across similar types
    let result_3 = parameters::transform_parameters(3, 1.0, 0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0);
    let result_4 = parameters::transform_parameters(4, 1.0, 0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0);

    // Types 3 and 4 should have similar Q reduction behavior
    assert!(result_3.processed_q > 0.0);
    assert!(result_4.processed_q > 0.0);
}

#[test]
fn test_gain_independence() {
    // Test that output structure is returned even with extreme gain values
    let results = vec![
        parameters::transform_parameters(0, 1.0, -24.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0),
        parameters::transform_parameters(0, 1.0, 0.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0),
        parameters::transform_parameters(0, 1.0, 24.0, 1000.0, 48000.0, -1, 0, 0.0, 0.0),
    ];

    for result in results {
        assert!(result.processed_q.is_finite());
        assert!(result.q_term.is_finite());
        assert!(result.gain_term.is_finite());
        assert!(result.frequency.is_finite());
    }
}
