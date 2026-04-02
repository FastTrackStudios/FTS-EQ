/// Comprehensive trace of shelf gain parameter flow through the Pro-Q 4 pipeline
/// This test analyzes how gain is applied for different shelf types
use eq_pro_dsp::prototype;
use std::f64::consts::PI;

#[test]
fn trace_shelf_gain_application_by_type() {
    println!("\n=== Shelf Gain Application Analysis ===\n");

    // From binary analysis:
    // Type 7 (Low Shelf): gain applied via apply_shelf_gain_to_zpk
    //   - Parameter: pow(Q, 0.5/order)
    //   - Application: zeros *= gain, poles /= gain
    //
    // Type 8 (High Shelf): gain NOT applied via apply_shelf_gain_to_zpk
    //   - Where is gain applied?
    //
    // Type 9 (Tilt Shelf): gain applied via scale_zpk_coefficients_by_gain AFTER bilinear
    //   - Parameter: Q / sqrt(2) per the formula
    //   - Application: BOTH zeros and poles *= gain
    //
    // Type 10 (Band Shelf): Similar to type 8?

    // Constants from binary at disassembly offsets
    let const_low_type: f64 = 0.5; // 0x180231a00
    let const_high_type: f64 = 1.0; // 0x180231ab8

    println!("Constants:");
    println!("  Low-type shelf (0x180231a00): {}", const_low_type);
    println!("  High-type shelf (0x180231ab8): {}", const_high_type);
    println!("");

    // Test case: order=2, Q=0.707, gain_db=6dB
    let order = 2_usize;
    let q = 0.707_f64;
    let gain_db = 6.0_f64;
    let gain_linear = 10.0_f64.powf(gain_db / 20.0);

    println!("Test Parameters:");
    println!("  Order: {}", order);
    println!("  Q: {}", q);
    println!("  Gain (dB): {}", gain_db);
    println!("  Gain (linear): {}", gain_linear);
    println!("");

    // Type 7: Low Shelf parameter computation
    {
        println!("TYPE 7 (LOW SHELF):");
        let gain_param = q.powf(const_low_type / order as f64);
        println!(
            "  Gain parameter: pow(Q, {}/{}) = {}",
            const_low_type, order, gain_param
        );
        println!(
            "  Q-derived param ≠ dB gain: {} ≠ {}",
            gain_param, gain_linear
        );
        println!(
            "  Ratio: {} / {} = {}",
            gain_param,
            gain_linear,
            gain_param / gain_linear
        );
        println!("  ZPK: zeros *= {}, poles /= {}", gain_param, gain_param);
        println!("");
    }

    // Type 8: High Shelf parameter computation
    {
        println!("TYPE 8 (HIGH SHELF):");
        let gain_param = q.powf(const_high_type / order as f64);
        println!(
            "  Gain parameter: pow(Q, {}/{}) = {}",
            const_high_type, order, gain_param
        );
        println!(
            "  Q-derived param ≠ dB gain: {} ≠ {}",
            gain_param, gain_linear
        );
        println!("  NOTE: apply_shelf_gain_to_zpk is NOT called!");
        println!("  Gain applied in zpk_to_biquad_coefficients?");
        println!("");
    }

    // Type 9: Tilt Shelf parameter computation
    {
        println!("TYPE 9 (TILT SHELF):");
        println!("  Gain applied AFTER ZPK→bilinear pipeline!");
        let const_1_0 = 1.0_f64;
        let tilt_gain_factor =
            const_1_0 / ((const_1_0 / q) * (1.0_f64 / std::f64::consts::FRAC_1_SQRT_2));
        println!(
            "  scale_zpk_coefficients_by_gain parameter: 1.0 / (1.0/Q * INV_SQRT2) = {}",
            tilt_gain_factor
        );
        println!(
            "  Simplified: Q * sqrt(2) = {}",
            q * std::f64::consts::SQRT_2
        );
        println!("  Application: BOTH zeros and poles *= gain_scale");
        println!("");
    }

    // Hypothesis: Where does the actual dB gain go?
    {
        println!("KEY QUESTION: Where is the actual dB gain applied?");
        println!("");
        println!("Possibilities:");
        println!("1. Type 7: Is the dB gain encoded in the Q parameter?");
        println!("   - User sets: Q=0.707, gain_db=6");
        println!("   - Binary passes: pow(Q, 0.5/order) to apply_shelf_gain_to_zpk");
        println!("   - But gain_db is not directly passed!");
        println!("");
        println!("2. Type 9: The formula 'Q * sqrt(2)' could encode the gain");
        println!("   - For gain_db=6: linear_gain = 1.995");
        println!(
            "   - If Q = sqrt(gain), then Q would be {}",
            gain_linear.sqrt()
        );
        println!("   - Actual Q = 0.707, so this doesn't match!");
        println!("");
        println!("3. The dB gain might be applied in zpk_to_biquad_coefficients");
        println!("   - Which receives a gain parameter via CONCAT44()");
        println!("   - Need to investigate that function");
        println!("");
    }
}

#[test]
fn analyze_butterworth_prototype() {
    println!("\n=== Butterworth Prototype Analysis ===\n");

    // Create Butterworth LP prototypes and examine their properties
    for order in [1, 2, 3, 4] {
        let lpzpk = prototype::butterworth_lp(order);
        println!("Butterworth LP order {}:", order);
        println!("  Poles: {}", lpzpk.poles.len());
        println!("  Zeros: {}", lpzpk.zeros.len());
        println!("  Gain: {}", lpzpk.gain);

        for (i, pole) in lpzpk.poles.iter().enumerate() {
            println!("    Pole {}: ({:.6}, {:.6})", i, pole.re, pole.im);
        }
        println!("");
    }

    // Key insight: Butterworth LP has NO zeros! Only poles at -0.707 ± 0.707j for order=2
    // When apply_shelf_gain_to_zpk scales "zeros *= gain", it's operating on an empty array!
    // But the poles are scaled by 1/gain, which DOES affect the frequency response.
}
