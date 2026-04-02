/// Test to verify the relationship between Q and dB gain
/// Hypothesis: The dB gain is encoded INTO the Q parameter, not applied separately
use eq_pro_dsp::prototype;

#[test]
fn verify_q_encodes_gain() {
    println!("\n=== Q Parameter Encodes Gain Theory ===\n");

    // Theory: For Pro-Q 4 shelves, the effective Q that reaches apply_shelf_gain_to_zpk
    // must be computed from user_Q and gain_db BEFORE calling setup_eq_band_filter.
    //
    // When the binary receives Q in filter_type_dispatcher, it's already the "effective Q"
    // that incorporates gain information.

    // Test with different gain values while keeping other parameters constant
    let user_q = 0.707_f64; // User's Q input
    let order = 2_usize;
    let const_low = 0.5_f64;

    println!("Testing the hypothesis that gain is encoded in Q:");
    println!("User Q (constant): {}", user_q);
    println!("Order: {}", order);
    println!("Low-shelf constant: {}\n", const_low);

    // For different user gain_db values, try to find the pattern
    let gain_values = vec![0.0, 3.0, 6.0, 12.0];

    for gain_db in gain_values {
        let gain_linear = 10.0_f64.powf(gain_db / 20.0);

        println!("Gain: {} dB = {:.6} linear", gain_db, gain_linear);

        // Hypothesis 1: effective_Q = user_Q * gain_linear
        {
            let eff_q1 = user_q * gain_linear;
            let gain_param = eff_q1.powf(const_low / order as f64);
            println!("  Hypothesis 1 (Q * gain_linear):");
            println!("    effective_Q = {:.6}", eff_q1);
            println!("    pow(Q, 0.5/2) = {:.6}", gain_param);
        }

        // Hypothesis 2: effective_Q = user_Q * gain_linear.sqrt()
        {
            let eff_q2 = user_q * gain_linear.sqrt();
            let gain_param = eff_q2.powf(const_low / order as f64);
            println!("  Hypothesis 2 (Q * sqrt(gain_linear)):");
            println!("    effective_Q = {:.6}", eff_q2);
            println!("    pow(Q, 0.5/2) = {:.6}", gain_param);
        }

        // Hypothesis 3: effective_Q = user_Q^gain_linear
        {
            let eff_q3 = user_q.powf(gain_linear);
            let gain_param = eff_q3.powf(const_low / order as f64);
            println!("  Hypothesis 3 (Q^gain_linear):");
            println!("    effective_Q = {:.6}", eff_q3);
            println!("    pow(Q, 0.5/2) = {:.6}", gain_param);
        }

        // Hypothesis 4: Check what gain_param value would produce the desired gain
        // If we want pow(Q, exp) to equal gain_linear, then:
        // exp = ln(gain_linear) / ln(Q) = log_Q(gain_linear)
        {
            if user_q != 1.0 {
                let required_exp = gain_linear.ln() / user_q.ln();
                println!("  Hypothesis 4 (what exponent would give desired gain?):");
                println!("    Required exponent: {:.6}", required_exp);
                println!(
                    "    Actual exponent (0.5/order): {:.6}",
                    const_low / order as f64
                );
                let gain_param = user_q.powf(required_exp);
                println!(
                    "    pow(Q, required_exp) = {:.6} (target: {:.6})",
                    gain_param, gain_linear
                );
            }
        }

        println!("");
    }

    // Key insight from Butterworth
    {
        println!("Butterworth pole analysis:");
        let lpzpk = prototype::butterworth_lp(2);
        println!(
            "LP order 2 has {} poles (frequency-normalized at -1)",
            lpzpk.poles.len()
        );
        println!("DC gain: {}", lpzpk.gain);

        // When we scale poles by 1/Q:
        // pole at -0.707 becomes -0.707/Q
        // This makes the filter resonant (less attenuation around DC for low shelf)
        println!("\nWhen apply_shelf_gain_to_zpk scales poles by 1/Q:");
        println!("  pole: -0.707 → -0.707/Q");
        println!("  This changes the DC cutoff frequency!");
        println!("  Lower Q → higher DC cutoff → more gain at DC");
        println!("  Higher Q → lower DC cutoff → less gain at DC");
    }
}

#[test]
fn analyze_shelf_design_difference() {
    println!("\n=== How Pro-Q 4 Shelves Differ from RBJ ===\n");

    println!("RBJ Shelf Design:");
    println!("  1. Get Butterworth prototype");
    println!("  2. Apply gain formula: A = 10^(dB/40) for shelves");
    println!("  3. Design biquad with explicit gain factor");
    println!("  4. Gain applied uniformly to output\n");

    println!("Pro-Q 4 Shelf Design (from binary analysis):");
    println!("  1. Convert user (Q, gain_db) → effective_Q (where?)");
    println!("  2. Get Butterworth prototype LP");
    println!("  3. In filter_type_dispatcher:");
    println!("     apply_shelf_gain_to_zpk(pow(effective_Q, 0.5/order))");
    println!("     - Zeros *= pow(effective_Q, 0.5/order)");
    println!("     - Poles /= pow(effective_Q, 0.5/order)");
    println!("  4. Bilinear transform");
    println!("  5. Convert ZPK to biquads\n");

    println!("Key Differences:");
    println!("  - RBJ: Gain is a separate multiplicative factor");
    println!("  - Pro-Q: Gain is achieved by moving pole/zero positions!");
    println!("  - RBJ: Q controls transition width");
    println!("  - Pro-Q: Q encodes BOTH transition width AND gain magnitude!");
    println!("  - This requires pre-computing effective Q from user parameters\n");
}
