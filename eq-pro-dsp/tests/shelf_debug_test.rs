/// Debug test for shelf filter implementation
/// Trace through the ZPK pipeline to understand gain application
use eq_pro_dsp::biquad;
use eq_pro_dsp::prototype;
use eq_pro_dsp::zpk::Complex;
use std::f64::consts::PI;

#[test]
fn debug_shelf_gain_parameter() {
    // Test case 1: Low shelf type 7
    // Parameters as would come from setup_eq_band_filter

    let order = 2_usize; // 2nd order Butterworth
    let freq_hz = 1000.0_f64;
    let q = 0.707_f64; // 1/sqrt(2), the default shelf Q
    let gain_db = 6.0_f64; // +6 dB boost
    let sample_rate = 48000.0_f64;

    println!("\n=== Low Shelf (Type 7) Debug ===");
    println!("Order: {}", order);
    println!("Frequency: {} Hz", freq_hz);
    println!("Q: {}", q);
    println!("Gain: {} dB", gain_db);
    println!("Sample rate: {} Hz", sample_rate);

    // Step 1: Butterworth LP prototype
    let lp_zpk = prototype::butterworth_lp(order);
    println!("\n--- Butterworth LP Prototype ---");
    println!("Poles count: {}", lp_zpk.poles.len());
    println!("Zeros count: {}", lp_zpk.zeros.len());
    println!("Gain: {}", lp_zpk.gain);
    for (i, pole) in lp_zpk.poles.iter().enumerate() {
        println!("  Pole {}: {:?}", i, pole);
    }

    // Step 2: Understand the Q-derived gain parameter
    // From disassembly: pow(Q, 0.5 / order) for low shelf
    let q_factor = q.powf(0.5 / order as f64);
    println!("\n--- Q-Derived Parameter ---");
    println!("Q: {}", q);
    println!("Exponent: 0.5 / {} = {}", order, 0.5 / order as f64);
    println!("pow(Q, exponent): {}", q_factor);
    println!("1/pow(Q, exponent): {}", 1.0 / q_factor);

    // Step 3: What we're trying to achieve (shelf gain in dB)
    let shelf_gain_linear = 10.0_f64.powf(gain_db / 20.0);
    println!("\n--- Expected Shelf Gain ---");
    println!("Shelf gain (dB): {}", gain_db);
    println!("Shelf gain (linear): {}", shelf_gain_linear);
    println!("Shelf gain (reciprocal): {}", 1.0 / shelf_gain_linear);

    // The question: how does q_factor relate to shelf_gain_linear?
    println!("\n--- Relationship Analysis ---");
    println!(
        "q_factor == shelf_gain? {}",
        (q_factor - shelf_gain_linear).abs() < 0.01
    );
    println!(
        "q_factor == 1/shelf_gain? {}",
        (q_factor - 1.0 / shelf_gain_linear).abs() < 0.01
    );
    println!(
        "Ratio q_factor / shelf_gain: {}",
        q_factor / shelf_gain_linear
    );

    // Step 4: Frequency analysis
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    println!("\n--- Frequency Analysis ---");
    println!("Normalized frequency w0: {} rad", w0);
    println!("w0 in degrees: {} °", w0 * 180.0 / PI);
}

#[test]
fn debug_high_shelf_parameter() {
    // Test case 2: High shelf type 8 (NOT using apply_shelf_gain_to_zpk)

    let order = 2;
    let freq_hz = 1000.0;
    let q = 0.707;
    let gain_db = 6.0;
    let sample_rate = 48000.0;

    println!("\n=== High Shelf (Type 8) Debug ===");
    println!("Order: {}", order);
    println!("Frequency: {} Hz", freq_hz);
    println!("Q: {}", q);
    println!("Gain: {} dB", gain_db);

    // High shelf uses elliptic constant (1.0), so:
    // pow(Q, 1.0 / order) = pow(0.707, 0.5) = sqrt(0.707) ≈ 0.841
    let q_factor = q.powf(1.0 / order as f64);
    println!("\n--- Q-Derived Parameter (High Shelf) ---");
    println!("pow(Q, 1.0/{}) = {}", order, q_factor);

    // But high shelf doesn't call apply_shelf_gain_to_zpk!
    // So where is the gain applied?
    println!("\n--- High Shelf Gain Application ---");
    println!("NOTE: Type 8 does NOT call apply_shelf_gain_to_zpk");
    println!("Gain must be applied elsewhere (in zpk_to_biquad_coefficients?)");
}

#[test]
fn debug_tilt_shelf_parameter() {
    // Test case 3: Tilt shelf type 9

    let order = 2;
    let freq_hz = 2000.0;
    let q = 0.707;
    let gain_db = 6.0;
    let sample_rate = 48000.0;

    println!("\n=== Tilt Shelf (Type 9) Debug ===");
    println!("Order: {}", order);
    println!("Frequency: {} Hz", freq_hz);
    println!("Q: {}", q);
    println!("Gain: {} dB", gain_db);

    // Tilt also uses elliptic constant
    let q_factor = q.powf(1.0 / order as f64);
    println!("\n--- Q-Derived Parameter (Tilt) ---");
    println!("pow(Q, 1.0/{}) = {}", order, q_factor);
    println!("Like high shelf, tilt shelf doesn't call apply_shelf_gain_to_zpk");
}

#[test]
fn analyze_gain_db_vs_q_relationship() {
    // Try to find if there's a relationship between gain_db and Q

    println!("\n=== Gain vs Q Relationship ===");

    // Various Q and gain combinations
    let test_cases = vec![
        (0.5, 6.0),
        (0.707, 6.0),
        (1.0, 6.0),
        (2.0, 6.0),
        (0.707, 3.0),
        (0.707, 12.0),
    ];

    for (q, gain_db) in test_cases {
        let gain_linear = 10.0_f64.powf(gain_db / 20.0);
        let q_factor_low = q.powf(0.5 / 2.0); // Assuming order=2
        let q_factor_high = q.powf(1.0 / 2.0);

        println!("\nQ={}, Gain={}dB:", q, gain_db);
        println!("  gain_linear = {}", gain_linear);
        println!("  pow(Q, 0.5/2) = {}", q_factor_low);
        println!("  pow(Q, 1.0/2) = {}", q_factor_high);
        println!(
            "  gain_linear == q_factor_low? {}",
            (gain_linear - q_factor_low).abs() < 0.01
        );
        println!(
            "  gain_linear == q_factor_high? {}",
            (gain_linear - q_factor_high).abs() < 0.01
        );
    }
}

#[test]
fn debug_parameter_flow_from_setup_eq_band_filter() {
    // Trace what parameters are actually passed from setup_eq_band_filter

    println!("\n=== Parameter Flow Analysis ===");
    println!("From setup_eq_band_filter disassembly (0x1800fdf10):");
    println!("- param_5 (XMM1) = Q (from user input)");
    println!("- param_6 (XMM4) = sample_rate");
    println!("- param_7 = INV_SQRT2 (0.707..., override Q for shelves)");
    println!("- param_8 = 1.0/Q_internal");
    println!("");
    println!("Then calls design_filter_zpk_and_transform which:");
    println!("1. Calls filter_type_dispatcher");
    println!("2. Processes ZPK based on transform_type");
    println!("3. Calls zpk_to_biquad_coefficients");
    println!("");
    println!("The actual shelf gain (dB) is NOT directly passed to apply_shelf_gain_to_zpk!");
    println!("Instead, Q-derived values are used for pole/zero scaling.");
}
