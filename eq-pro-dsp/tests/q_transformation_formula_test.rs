/// Test the complete Q transformation and gain formulas from Pro-Q 4
/// Extracted via binary reverse engineering from apply_eq_band_parameters_full

#[test]
fn test_q_transformation_formula() {
    println!("\n=== COMPLETE Q TRANSFORMATION FORMULA ===\n");

    // Constants extracted from Pro-Q 4 binary:
    // - 0x180231988: ln(10)/20 ≈ 0.1151292546
    // - 0x180231764: 0.1355425119 (c1 for power formula)
    // - 0x180231804: 0.5 (c2 for power formula)
    // - 0x180231760: 0.125 (c3 for power formula)
    // - 0x180231df4: 32.0 (base for power formula)
    // - 0x18028737c: INV_SQRT2 ≈ 0.7071067812

    let ln10_over_20 = 0.115129254649702_f64;
    let c1 = 0.1355425119_f64;
    let c2 = 0.5_f64;
    let c3 = 0.125_f64;
    let base = 32.0_f64;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

    println!("Constants extracted from binary:");
    println!("  ln(10)/20 = {:.15}", ln10_over_20);
    println!("  c1 = {:.10}", c1);
    println!("  c2 = {:.10}", c2);
    println!("  c3 = {:.10}", c3);
    println!("  base = {:.1}", base);
    println!("  INV_SQRT2 = {:.10}\n", inv_sqrt2);

    // Formula 1: For types 1, 2, 4, 5, 6, 12 (peak, notch, some filters)
    println!("FORMULA 1 - Type 1,2,4,5,6,12 (Peak, Notch, etc):");
    println!("  Q_transformed = base^(cos(Q) * c1 + c2) * c3");
    println!("  Q_transformed = 32^(cos(Q) * 0.1355 + 0.5) * 0.125\n");

    let test_qs: Vec<f64> = vec![0.5, 0.707, 1.0, 2.0];
    for &q in &test_qs {
        let cos_q = q.cos();
        let exponent = cos_q * c1 + c2;
        let q_transformed = base.powf(exponent) * c3;
        println!(
            "  Q={:.3}: cos(Q)={:.6}, exp={:.6}, Q_eff={:.6}",
            q, cos_q, exponent, q_transformed
        );
    }

    // Formula 2: For types 8, 9, 11 and others (high shelf, tilt, allpass, etc)
    println!("\nFORMULA 2 - Type 8,9,11+ (High Shelf, Tilt, Allpass):");
    println!("  Q_transformed = Q * INV_SQRT2");
    println!("  Q_transformed = Q * 0.7071067812\n");

    for &q in &test_qs {
        let q_transformed = q * inv_sqrt2;
        println!("  Q={:.3}: Q_eff={:.6}", q, q_transformed);
    }

    // Formula 3: Gain conversion (ALL types)
    println!("\nFORMULA 3 - Gain Conversion (All Types):");
    println!("  linear_gain = exp(gain_db * ln(10)/20)");
    println!("  linear_gain = 10^(gain_db / 20)\n");

    let test_gains: Vec<f64> = vec![0.0, 3.0, 6.0, 12.0];
    for &gain_db in &test_gains {
        let linear_via_exp = (gain_db * ln10_over_20).exp();
        let linear_direct = 10.0_f64.powf(gain_db / 20.0);
        let db_check = 20.0 * linear_via_exp.log10();
        println!(
            "  gain={:5.1} dB: linear={:.10} (check: {:.1} dB)",
            gain_db, linear_via_exp, db_check
        );
        assert!(
            (linear_via_exp - linear_direct).abs() < 1e-10,
            "Formula mismatch for gain {}",
            gain_db
        );
    }

    // Complete pipeline example
    println!("\n=== COMPLETE EXAMPLE: Type 7 (Low Shelf) ===");
    println!("  User Q: 0.707");
    println!("  User gain: 6 dB");
    println!("  Filter type: 7 (low shelf)\n");

    let user_q = 0.707_f64;
    let user_gain_db = 6.0_f64;

    // Type 7 is NOT in {1,2,4,5,6,12}, so it uses Formula 2
    let q_transformed = user_q * inv_sqrt2;
    let linear_gain = (user_gain_db * ln10_over_20).exp();

    println!("Results:");
    println!("  Q_transformed = {:.10}", q_transformed);
    println!(
        "  linear_gain = {:.10} ({:.1} dB)",
        linear_gain,
        20.0 * linear_gain.log10()
    );

    println!("\nThese are passed to setup_eq_band_filter:");
    println!("  - Butterworth LP prototype is created");
    println!("  - apply_shelf_gain_to_zpk scales poles by 1/pow(Q_eff, 0.5/order)");
    println!("  - linear_gain is applied in biquad coefficients");
    println!("  - Result: low shelf with proper frequency response");
}

#[test]
fn verify_formula_against_known_values() {
    println!("\n=== VERIFICATION AGAINST KNOWN VALUES ===\n");

    // For a 6dB boost:
    let gain_db = 6.0_f64;
    let expected_linear = 1.9952623149688795_f64; // 10^(6/20)

    let ln10_over_20 = 0.115129254649702_f64;
    let computed_linear = (gain_db * ln10_over_20).exp();

    println!("Expected linear gain (10^(6/20)): {:.15}", expected_linear);
    println!(
        "Computed linear gain (exp formula): {:.15}",
        computed_linear
    );
    println!(
        "Difference: {:.2e}",
        (expected_linear - computed_linear).abs()
    );

    assert!((expected_linear - computed_linear).abs() < 1e-13);
    println!("✓ Formula verified!\n");

    // For Q transformation of type 8 (high shelf):
    let user_q = 0.707_f64;
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let q_transformed = user_q * inv_sqrt2;

    println!("For type 8 (high shelf):");
    println!("  User Q: {:.6}", user_q);
    println!("  Q_transformed: {:.6}", q_transformed);
    println!("  Expected (Q * 0.707): {:.6}", user_q * 0.707);
}

#[test]
fn formula_summary() {
    println!("\n=== FORMULA SUMMARY ===\n");

    println!("PRO-Q 4 PARAMETER TRANSFORMATION (from apply_eq_band_parameters_full):\n");

    println!("Input: user_Q, user_gain_db, filter_type\n");

    println!("Step 1: Transform Q based on filter type");
    println!("  If filter_type in {{1, 2, 4, 5, 6, 12}} (peak, notch):");
    println!("    Q_eff = 32^(cos(Q) * 0.1355425119 + 0.5) * 0.125");
    println!("  Else (shelf, tilt, allpass, etc):");
    println!("    Q_eff = Q * 0.7071067812\n");

    println!("Step 2: Convert gain_db to linear gain (ALL types):");
    println!("    linear_gain = exp(gain_db * 0.115129254649702)");
    println!("               = 10^(gain_db / 20)\n");

    println!("Step 3: Pass to setup_eq_band_filter:");
    println!("    setup_eq_band_filter(state, flag, filter_type, freq, Q_eff, linear_gain)\n");

    println!("Step 4: In setup_eq_band_filter:");
    println!("    - Create Butterworth LP prototype");
    println!("    - Apply apply_shelf_gain_to_zpk with pow(Q_eff, 0.5/order)");
    println!("    - Bilinear transform");
    println!("    - Convert ZPK to biquad coefficients");
    println!("    - Use linear_gain in numerator normalization\n");

    println!("CRITICAL: gain_db is NOT encoded into Q!");
    println!("Both are applied independently:");
    println!("  - Q affects pole/zero positions (affects transition shape)");
    println!("  - gain affects biquad numerator coefficients (affects DC magnitude)");
}
