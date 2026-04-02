/// Determine the correct formula for converting user gain_db to effective_Q
/// Hypothesis: effective_Q = user_Q * (gain_linear)^(some_power)
use std::f64::consts::PI;

#[test]
fn find_effective_q_formula() {
    println!("\n=== Finding Effective Q Formula ===\n");

    // From binary analysis:
    // apply_shelf_gain_to_zpk receives: pow(effective_Q, 0.5/order)
    // And uses this to scale poles by: 1 / pow(effective_Q, 0.5/order)
    //
    // For order=2, that's: 1 / pow(effective_Q, 0.25) = effective_Q^(-0.25)
    //
    // If we want this pole scaling to produce a certain DC gain boost,
    // we need to understand the relationship between pole scaling and frequency response.

    let user_q = 0.707_f64;
    let gain_db = 6.0_f64;
    let gain_linear = 10.0_f64.powf(gain_db / 20.0);
    let order = 2_usize;
    let const_low = 0.5_f64;

    println!("Given:");
    println!("  User Q: {}", user_q);
    println!("  Gain dB: {}", gain_db);
    println!("  Gain linear: {}", gain_linear);
    println!("  Order: {}", order);
    println!("");

    // Theory: The pole scaling factor (1/pow(Q, 0.25)) must relate to the gain
    // For a shelf filter, the DC gain is approximately the product of all pole scalings
    //
    // If we scale poles by k = 1/pow(effective_Q, 0.25), then:
    // DC response ≈ k^(number of poles) = [1/pow(effective_Q, 0.25)]^2 (for 2 poles)
    //             = 1 / pow(effective_Q, 0.5)
    //
    // So we want: 1 / pow(effective_Q, 0.5) ≈ gain_linear
    // Therefore: effective_Q ≈ 1 / (gain_linear^2)
    // Or: effective_Q ≈ gain_linear^(-2)

    println!("Hypothesis: effective_Q = gain_linear^(-2)");
    {
        let eff_q = gain_linear.powf(-2.0);
        let pole_scale = 1.0 / eff_q.powf(const_low);
        println!("  effective_Q = {:.6}", eff_q);
        println!("  pole scale = 1/pow(Q, 0.5) = {:.6}", pole_scale);
        println!(
            "  (for 2 poles: {:.6}^2 = {:.6})",
            pole_scale,
            pole_scale.powi(2)
        );
        println!("");
    }

    // But wait - we also have the user's Q value!
    // Maybe the formula combines both: effective_Q = user_Q * gain_linear^(something)

    println!("Alternative: effective_Q = user_Q * gain_linear^(power)");
    println!("Testing different powers:\n");

    for power in [0.0, 0.5, 1.0, 2.0, 4.0] {
        let eff_q = user_q * gain_linear.powf(power);
        let gain_param = eff_q.powf(const_low / order as f64);
        let pole_scale = 1.0 / gain_param;
        let dc_boost = pole_scale.powi(2); // 2 poles

        println!(
            "Power {}: effective_Q = {} * {:.6}^{} = {:.6}",
            power, user_q, gain_linear, power, eff_q
        );
        println!(
            "         pole scale = {:.6}, DC boost = {:.6}",
            pole_scale, dc_boost
        );
        println!("");
    }

    // The key insight: for a 2nd-order section with poles moved by scaling k:
    // The DC magnitude is approximately: |H(0)| ≈ k^(num_poles)
    //
    // But actually, the exact formula for a low shelf depends on the pole positions!
    // Butterworth poles are on the unit circle normalized to -1, so at angle 45°
    // When we scale them by k, they become: -k * cos(45°) ± j*k*sin(45°)
    //
    // The magnitude response at DC involves evaluating H(0) which requires
    // computing the product of distances from 0 to the poles and zeros.

    println!("For Butterworth poles at 45° (-0.707 ± 0.707j):");
    println!("When scaled by k: (-0.707k ± 0.707jk)");
    println!("Distance from 0: |pole| = sqrt((0.707k)^2 + (0.707k)^2) = k");
    println!("Product of 2 pole distances: k * k = k^2");
    println!("");
    println!("So DC magnitude ∝ k^(-2) where k = pole_scale");
    println!("If pole_scale = 1.090 (from our test), DC ∝ 1.090^2 ≈ 1.19 ≈ 1.76 dB");
    println!("But we want 6 dB! So pole scaling alone doesn't explain the gain.");
    println!("");
    println!("CONCLUSION: The dB gain must be applied as a separate gain factor");
    println!("in the biquad numerator, not just through pole/zero positioning!");
}

#[test]
fn verify_parameter_passing() {
    println!("\n=== Parameter Passing Verification ===\n");

    // The mystery: where does the 6 dB gain parameter end up in the biquad coefficients?
    //
    // From zpk_to_biquad_coefficients disassembly, it seems the gain parameter
    // might be applied to the biquad numerator coefficients for allpass filters (type 11+)
    // but not for shelves (types 7-10).
    //
    // For shelves, maybe the gain is implicitly handled by the formula itself!

    println!("Looking at the shelf gain application mechanisms:\n");

    println!("Type 7 (Low Shelf):");
    println!("  1. apply_shelf_gain_to_zpk called with pow(Q, 0.5/order)");
    println!("  2. Zeros *= pow(Q, 0.5/order)");
    println!("  3. Poles /= pow(Q, 0.5/order)");
    println!("  4. => Pole scaling by 1/pow(Q, 0.5/order)");
    println!("  5. Bilinear transform");
    println!("  6. Convert to biquads");
    println!("");

    println!("Type 9 (Tilt Shelf):");
    println!("  1. Design LP normally with user_Q");
    println!("  2. Bilinear transform");
    println!("  3. Convert to biquads");
    println!("  4. Call scale_zpk_coefficients_by_gain with Q*sqrt(2)");
    println!("  5. => Coefficients multiplied by this factor");
    println!("");

    println!("Key Insight:");
    println!("Type 7 changes POLE POSITIONS before bilinear");
    println!("Type 9 scales COEFFICIENTS after bilinear");
    println!("");
    println!("Maybe Type 8 is similar to Type 9?");
    println!("Or maybe the dB gain is truly embedded in the Q value by the caller!");
}
