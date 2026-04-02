/// Test implementation of Pro-Q 4 shelf filters using the actual ZPK pipeline
/// This shows what the correct implementation should look like
use eq_pro_dsp::prototype;
use eq_pro_dsp::transform;
use eq_pro_dsp::zpk::Zpk;
use std::f64::consts::PI;

#[test]
fn implement_low_shelf_zpk_pipeline() {
    println!("\n=== Low Shelf ZPK Pipeline Implementation ===\n");

    // Parameters
    let order = 2_usize; // 2nd order = 1 biquad section
    let freq_hz = 1000.0_f64;
    let q = 0.707_f64; // User's Q parameter
    let gain_db = 6.0_f64;
    let sample_rate = 48000.0_f64;

    println!("Input parameters:");
    println!("  Order: {}", order);
    println!("  Frequency: {} Hz", freq_hz);
    println!("  Q: {}", q);
    println!("  Gain: {} dB", gain_db);
    println!("  Sample rate: {} Hz\n", sample_rate);

    // Step 1: Create Butterworth LP prototype
    let mut zpk = prototype::butterworth_lp(order);
    println!("Step 1: Butterworth LP prototype");
    println!("  Poles: {}", zpk.poles.len());
    println!("  Zeros: {}", zpk.zeros.len());
    println!("  Gain: {}\n", zpk.gain);

    // Step 2: Apply shelf gain scaling (apply_shelf_gain_to_zpk behavior)
    // For type 7 (low shelf):
    //   - Zeros are scaled by pow(Q, 0.5/order)
    //   - Poles are scaled by 1/pow(Q, 0.5/order)
    //
    // BUT: The Q here is still a mystery! We don't know how user_Q and gain_db
    // combine to create the effective Q that produces the right gain response.

    let const_low = 0.5_f64; // From binary at 0x180231a00
    let gain_param = q.powf(const_low / order as f64);

    println!("Step 2: Apply shelf gain scaling");
    println!("  Constant for low shelf: {}", const_low);
    println!(
        "  Gain parameter: pow(Q, {}/{}) = {}",
        const_low, order, gain_param
    );
    println!("  Pole scaling: 1/{} = {}\n", gain_param, 1.0 / gain_param);

    // Scale zeros by gain_param (though Butterworth LP has no zeros)
    for zero in zpk.zeros.iter_mut() {
        *zero = *zero * gain_param;
    }

    // Scale poles by 1/gain_param
    for pole in zpk.poles.iter_mut() {
        *pole = *pole / gain_param;
    }

    println!("Step 3: Apply bilinear transform");
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    println!("  Normalized frequency w0: {} rad\n", w0);

    // Apply bilinear transform (would be done here)
    // transform::bilinear_transform_zpk(&mut zpk, w0);

    println!("Step 4: Convert ZPK to biquad coefficients");
    println!("  (Would call zpk_to_biquad_coefficients here)\n");

    println!("KEY MYSTERY UNRESOLVED:");
    println!(
        "  The gain_db parameter ({} dB) is NOT used in this pipeline!",
        gain_db
    );
    println!(
        "  Pole scaling alone produces ~1.76 dB, but we need {} dB!",
        gain_db
    );
    println!("  Where is the actual dB gain applied?");
}

#[test]
fn document_missing_gain_formula() {
    println!("\n=== The Missing Gain Formula ===\n");

    println!("What we know:");
    println!("1. apply_shelf_gain_to_zpk receives pow(Q, 0.5/order)");
    println!("2. This parameter scales poles by 1/pow(Q, 0.5/order)");
    println!("3. For Q=0.707 and order=2: pole scaling is 1.09x");
    println!("4. This produces ~1.76 dB DC gain change");
    println!("5. But we need to produce 6 dB gain!\n");

    println!("What we don't know:");
    println!("1. How does user_Q relate to effective_Q passed to filter_type_dispatcher?");
    println!("2. How does user_gain_db affect the parameters?");
    println!("3. Where is the remaining 4.24 dB of gain applied?\n");

    println!("Hypotheses:");
    println!("A. Gain is encoded into effective Q before setup_eq_band_filter");
    println!("   - Formula: effective_Q = f(user_Q, user_gain_db, type)");
    println!("   - This function is likely in the plugin parameter processing\n");

    println!("B. Gain is applied in zpk_to_biquad_coefficients");
    println!("   - The dB gain parameter that arrives at design_filter_zpk_and_transform");
    println!("   - might be used in the numerator normalization for shelves\n");

    println!("C. Gain is applied in scale_zpk_coefficients_by_gain (for type 9)");
    println!("   - Type 9 calls this function AFTER bilinear transform");
    println!("   - Formula involves Q * sqrt(2), might extend to other types\n");

    println!("Next investigation:");
    println!("1. Examine the plugin parameter interface");
    println!("2. Find where user parameters are converted to DSP parameters");
    println!("3. Look for the actual gain formula implementation");
}
