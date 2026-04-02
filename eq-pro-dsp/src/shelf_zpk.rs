//! Pro-Q 4 shelf filter design — actual ZPK pipeline (not RBJ approximation)
//!
//! CRITICAL: This replaces the RBJ-based implementation because Pro-Q 4 applies
//! shelf gain through pole/zero repositioning, NOT as a separate multiplicative factor!
//!
//! Pipeline from binary analysis (0x1800fdf10 setup_eq_band_filter):
//!   1. Get Butterworth LP prototype
//!   2. Apply gain scaling via apply_shelf_gain_to_zpk:
//!      - Scales zeros by pow(Q_effective, constant/order)
//!      - Scales poles by 1/pow(Q_effective, constant/order)
//!   3. Apply bilinear transform
//!   4. Convert ZPK to biquad coefficients
//!   5. For type 9 (tilt): apply scale_zpk_coefficients_by_gain post-bilinear
//!
//! DISCOVERED FORMULAS (from apply_eq_band_parameters_full at 0x1801110b0):
//!
//! For types 1, 2, 4, 5, 6, 12 (peak, notch filters):
//!   Q_transformed = 32^(cos(user_Q) * 0.1355425119 + 0.5) * 0.125
//!
//! For types 8, 9, 11+ (shelf, tilt, allpass):
//!   Q_transformed = user_Q * 0.7071067812 (INV_SQRT2)
//!
//! For ALL types (gain application):
//!   linear_gain = exp(gain_db * 0.115129254649702)
//!                = 10^(gain_db / 20)  [mathematically equivalent]
//!
//! KEY DISCOVERY: gain_db and Q are applied INDEPENDENTLY!
//! Gain is NOT encoded into Q, but applied separately via biquad numerator coefficients.

use crate::biquad::{self, Coeffs};
use crate::prototype;
use crate::transform;
use crate::zpk::Zpk;

// Constants extracted from Pro-Q 4 binary via reverse engineering
const LN10_OVER_20: f64 = 0.115129254649702; // At 0x180231988
const BASE_POWER: f64 = 32.0; // At 0x180231df4
const C1_POWER: f64 = 0.1355425119; // At 0x180231764
const C2_POWER: f64 = 0.5; // At 0x180231804
const C3_POWER: f64 = 0.125; // At 0x180231760
const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2; // 0x18028737c
const CONST_LOW_SHELF: f64 = 0.5; // At 0x180231a00

/// Design a low shelf filter (type 7) via actual ZPK pipeline.
///
/// Type 7 is low shelf: gain is applied by scaling poles via apply_shelf_gain_to_zpk.
/// CRITICAL: Poles are DIVIDED by gain_param (not multiplied)
///
/// Pipeline:
/// 1. Create Butterworth LP prototype (order 2 for 1 biquad section)
/// 2. Transform Q: Q * INV_SQRT2 (Formula 2 for shelf types)
/// 3. Apply shelf gain scaling: divide poles, multiply zeros
/// 4. Apply bilinear transform
/// 5. Convert ZPK to biquad coefficients
/// 6. Apply linear gain to numerator coefficients
pub fn design_low_shelf_zpk(
    n_sections: usize,
    _freq_hz: f64,
    user_q: f64,
    user_gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if user_gain_db.abs() < 0.001 {
        return vec![crate::biquad::PASSTHROUGH; n];
    }

    // Apply Formula 2: Q * INV_SQRT2 for shelf types
    let q_transformed = user_q * INV_SQRT2;

    // Divide gain across sections
    let section_gain_db = user_gain_db / n as f64;
    // Convert section gain_db to linear gain
    let linear_gain = (section_gain_db * LN10_OVER_20).exp();

    // For apply_shelf_gain_to_zpk, use square root of linear gain
    let gain_param = linear_gain.sqrt();

    let mut sections = Vec::new();

    for _k in 0..n {
        // Create order-2 Butterworth prototype
        let mut zpk = prototype::butterworth_lp(2);

        // Apply shelf gain scaling (Type 7: divide poles, multiply zeros)
        apply_shelf_gain_to_zpk_type7(&mut zpk, gain_param);

        // Apply bilinear transform
        zpk = transform::bilinear(&zpk, sample_rate);

        // Convert to biquad and apply gain
        let mut sos = biquad::zpk_to_sos(&zpk);

        // Apply linear gain to numerator (shelf gain in biquad domain)
        // Coeffs format: [a0, a1, a2, b0, b1, b2]
        for coeffs in sos.iter_mut() {
            coeffs[3] *= linear_gain; // b0
            coeffs[4] *= linear_gain; // b1
            coeffs[5] *= linear_gain; // b2
        }

        sections.extend(sos);
    }

    sections
}

/// Design a high shelf filter (type 8) via ZPK pipeline.
///
/// Type 8 is high shelf: gain is applied via pole/zero magnitude ratio in biquad domain.
/// CRITICAL: Type 8 does NOT call apply_shelf_gain_to_zpk (unlike type 7)
/// Instead, gain scaling happens AFTER biquad conversion via magnitude ratio.
///
/// Pipeline:
/// 1. Create Butterworth LP prototype (order 2)
/// 2. Transform Q: Q * INV_SQRT2 (Formula 2 for shelf types)
/// 3. Apply bilinear transform directly (NO pole scaling)
/// 4. Convert ZPK to biquad coefficients
/// 5. SPECIAL TYPE 8: Scale numerator by 1 / sqrt(|zero|² / |pole|²)
/// 6. Apply linear gain to numerator coefficients
pub fn design_high_shelf_zpk(
    n_sections: usize,
    _freq_hz: f64,
    user_q: f64,
    user_gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if user_gain_db.abs() < 0.001 {
        return vec![crate::biquad::PASSTHROUGH; n];
    }

    // Apply Formula 2: Q * INV_SQRT2 for shelf types
    // TODO: q_transformed needs to be used for Type 8 magnitude ratio scaling once
    // biquad conversion supports it
    let _q_transformed = user_q * INV_SQRT2;

    // Divide gain across sections
    let section_gain_db = user_gain_db / n as f64;
    let linear_gain = (section_gain_db * LN10_OVER_20).exp();

    let mut sections = Vec::new();

    for _k in 0..n {
        // Create order-2 Butterworth prototype (BEFORE bilinear)
        let zpk_proto = prototype::butterworth_lp(2);

        // Type 8 MAGNITUDE RATIO SCALING (from binary at 0x1800fcbde-0x1800fcc63):
        // We need to compute the pole/zero magnitude ratio BEFORE bilinear transform
        // because we lose access to complex values after biquad conversion
        //
        // For Butterworth LP prototype with complex conjugate poles and zeros at infinity:
        // The ratio is computed from the magnitudes in the frequency domain.
        // After bilinear transform and biquad conversion, we have:
        //   b_coeffs = (b0, b1, b2) from (zero_a + zero_b*z^-1 + zero_a*z^-2) / gain_zpk
        //   a_coeffs = (1, a1, a2) from pole positions
        //
        // Type 8 applies: numerator *= 1.0 / sqrt(|zero|² / |pole|²)

        // Compute Type 8 magnitude ratio scaling from prototype poles/zeros
        // Binary disassembly (0x1800fcbde-0x1800fcc63) shows:
        // scaling = 1.0 / sqrt(|zero|² / |pole|²) = |pole| / |zero|
        let mut type8_scale = 1.0;
        if !zpk_proto.poles.is_empty() && !zpk_proto.zeros.is_empty() {
            let pole = zpk_proto.poles[0];
            let zero = zpk_proto.zeros[0];

            let pole_mag = pole.mag();
            let zero_mag = zero.mag();

            // Compute scaling: |pole| / |zero|
            if zero_mag > 1e-12 {
                type8_scale = pole_mag / zero_mag;
            } else if pole_mag > 1e-12 {
                // Avoid division by zero if zero is at infinity
                type8_scale = 1.0 / pole_mag;
            }
        }

        // Type 8 does NOT call apply_shelf_gain_to_zpk!
        // Poles and zeros remain at prototype positions

        // Apply bilinear transform
        let zpk = transform::bilinear(&zpk_proto, sample_rate);

        // Convert to biquad
        let mut sos = biquad::zpk_to_sos(&zpk);

        // Apply Type 8 magnitude ratio scaling to numerator
        // Coeffs format: [a0, a1, a2, b0, b1, b2]
        for coeffs in sos.iter_mut() {
            coeffs[3] *= type8_scale; // b0
            coeffs[4] *= type8_scale; // b1
            coeffs[5] *= type8_scale; // b2
        }

        // Apply linear gain to numerator
        for coeffs in sos.iter_mut() {
            coeffs[3] *= linear_gain; // b0
            coeffs[4] *= linear_gain; // b1
            coeffs[5] *= linear_gain; // b2
        }

        sections.extend(sos);
    }

    sections
}

/// Design a tilt shelf filter (type 9) via ZPK pipeline.
///
/// Type 9 is tilt: applies frequency-proportional response shaping via post-bilinear scaling.
/// Scaling formula: post_bilinear_scale = Q_transformed * sqrt(2)
/// CRITICAL: This scales BOTH numerator AND denominator coefficients (unlike other types)
///
/// Pipeline:
/// 1. Create Butterworth LP prototype (order 2)
/// 2. Transform Q: Q * INV_SQRT2 (Formula 2 for shelf types)
/// 3. Apply bilinear transform
/// 4. Convert ZPK to biquad coefficients
/// 5. Apply post-bilinear scaling: all coeffs *= (Q_transformed * sqrt(2))
/// 6. Apply linear gain to numerator
pub fn design_tilt_shelf_zpk(
    n_sections: usize,
    _freq_hz: f64,
    user_q: f64,
    user_gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if user_gain_db.abs() < 0.001 {
        return vec![crate::biquad::PASSTHROUGH; n];
    }

    // Apply Formula 2: Q * INV_SQRT2 for shelf types
    let q_transformed = user_q * INV_SQRT2;

    // Divide gain across sections
    let section_gain_db = user_gain_db / n as f64;
    let linear_gain = (section_gain_db * LN10_OVER_20).exp();

    // Post-bilinear scaling for type 9: Q_transformed * sqrt(2)
    let post_bilinear_scale = q_transformed * std::f64::consts::SQRT_2;

    let mut sections = Vec::new();

    for _k in 0..n {
        // Create order-2 Butterworth prototype
        let mut zpk = prototype::butterworth_lp(2);

        // Apply bilinear transform
        zpk = transform::bilinear(&zpk, sample_rate);

        // Convert to biquad
        let mut sos = biquad::zpk_to_sos(&zpk);

        // Apply post-bilinear scaling (type 9 specific)
        // Coeffs format: [a0, a1, a2, b0, b1, b2]
        for coeffs in sos.iter_mut() {
            coeffs[3] *= post_bilinear_scale; // b0
            coeffs[4] *= post_bilinear_scale; // b1
            coeffs[5] *= post_bilinear_scale; // b2
            coeffs[1] *= post_bilinear_scale; // a1
            coeffs[2] *= post_bilinear_scale; // a2

            // Apply linear gain to numerator
            coeffs[3] *= linear_gain; // b0
            coeffs[4] *= linear_gain; // b1
            coeffs[5] *= linear_gain; // b2
        }

        sections.extend(sos);
    }

    sections
}

/// Design a band shelf filter (type 10) via ZPK pipeline.
///
/// Type 10 is band shelf: similar to type 8 but with narrow banding.
/// Band Shelf (Type 10): Elliptic LP→BP transformation with gain applied to poles/zeros.
///
/// Pro-Q 4's band shelf combines a bandpass filter with shelf-like gain application.
/// Algorithm:
/// 1. Design bandpass using elliptic LP→BP transformation
/// 2. Apply shelf gain by scaling poles and zeros
/// 3. Convert to biquad sections
pub fn design_band_shelf_zpk(
    n_sections: usize,
    freq_hz: f64,
    user_q: f64,
    user_gain_db: f64,
    sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if user_gain_db.abs() < 0.001 {
        return vec![crate::biquad::PASSTHROUGH; n];
    }

    // Step 1: Design elliptic bandpass prototype
    let mut bp = crate::prototype::butterworth_bp_elliptic(n, freq_hz, user_q, sample_rate);

    // Step 2: Apply shelf gain via pole/zero scaling
    // Convert gain_db to linear and then compute the scale factor
    let gain_linear = 10.0_f64.powf(user_gain_db / 20.0);

    // For band shelf, scale poles and zeros symmetrically
    // Higher order sections use geometric mean for smoother response
    let scale_factor = gain_linear.powf(0.5 / bp.poles.len().max(1) as f64);

    // Scale zeros by gain_factor (boost zeros outward for gain)
    for zero in &mut bp.zeros {
        *zero = *zero * scale_factor;
    }

    // Scale poles inversely (move inward for boost)
    for pole in &mut bp.poles {
        *pole = *pole / scale_factor;
    }

    // Adjust overall gain
    bp.gain = bp.gain * gain_linear.powf(bp.poles.len() as f64 / 2.0);

    // Step 3: Apply bilinear transform and convert to biquads
    let digital = crate::transform::bilinear(&bp, sample_rate);
    let mut sos = crate::biquad::zpk_to_sos(&digital);

    // Normalize: peak gain at center frequency should match requested gain_db
    let w0 = 2.0 * std::f64::consts::PI * freq_hz / sample_rate;
    let peak = crate::biquad::eval_sos(&sos, w0).mag();
    if peak > 1e-10 {
        let gain_error = user_gain_db - 20.0 * peak.log10();
        let scale = 10.0_f64.powf(gain_error / 20.0);
        if let Some(first) = sos.first_mut() {
            first[3] *= scale;
            first[4] *= scale;
            first[5] *= scale;
        }
    }

    sos
}

/// Apply shelf gain scaling to ZPK (type 7, low shelf only).
///
/// From binary analysis at 0x1800fcce0 (apply_shelf_gain_to_zpk):
/// This function is ONLY called for type 7 (low shelf).
/// It uses is_low_type_shelf_gain() to determine if poles should be divided.
///
/// Operations:
/// - Zeros are multiplied by gain_param
/// - Poles are DIVIDED by gain_param (type 7 specific!)
/// - ZPK gain is set to 1/gain_param
///
/// gain_param = sqrt(linear_gain) passed as effective_q parameter
/// This scaling reshapes the frequency response for proper shelf boost/cut.
fn apply_shelf_gain_to_zpk_type7(zpk: &mut Zpk, gain_param: f64) {
    let order = zpk.poles.len();

    if order == 0 {
        return;
    }

    // Compute the exponent: 0.5 / order
    let exponent = CONST_LOW_SHELF / order as f64;
    let scale_factor = gain_param.powf(exponent);

    // Scale zeros (Butterworth LP prototype has no zeros, but preserve structure)
    for zero in zpk.zeros.iter_mut() {
        *zero = *zero * scale_factor;
    }

    // Scale poles (DIVIDED, not multiplied - this is critical for type 7!)
    for pole in zpk.poles.iter_mut() {
        *pole = *pole / scale_factor;
    }

    // Set ZPK gain to compensate for pole/zero scaling
    zpk.gain = 1.0 / scale_factor;
}

/// Compute effective Q based on filter type using discovered Pro-Q 4 formulas.
///
/// From apply_eq_band_parameters_full (0x1801110b0) in Pro-Q 4 binary:
///
/// **Formula 1** (types 1, 2, 4, 5, 6, 12 - peak, notch, etc):
/// ```
/// Q_transformed = 32^(cos(user_Q) * 0.1355425119 + 0.5) * 0.125
/// ```
///
/// **Formula 2** (types 8, 9, 11+ - shelf, tilt, allpass, etc):
/// ```
/// Q_transformed = user_Q * 0.7071067812  (INV_SQRT2)
/// ```
fn compute_effective_q(user_q: f64, filter_type: u32) -> f64 {
    match filter_type {
        // Formula 1: Power formula for peak and notch filters
        1 | 2 | 4 | 5 | 6 | 12 => {
            let cos_q = user_q.cos();
            let exponent = cos_q * C1_POWER + C2_POWER;
            BASE_POWER.powf(exponent) * C3_POWER
        }
        // Formula 2: For shelf/tilt/allpass
        // The binary shows Q_transformed = Q * INV_SQRT2, but this is the final parameter.
        // For the ZPK apply_shelf_gain_to_zpk stage, we need to scale differently.
        // Based on RBJ equivalence: section_q = bw_q * (user_q / INV_SQRT2)
        // This means effective Q for ZPK should be inverted
        8 | 9 | 11 | _ => user_q / INV_SQRT2,
    }
}
