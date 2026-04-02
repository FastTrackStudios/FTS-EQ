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
///
/// Pipeline:
/// 1. Create Butterworth LP prototype (order 2 for 1 biquad section)
/// 2. Transform Q based on filter type (type 7 uses INV_SQRT2 scaling)
/// 3. Apply shelf gain scaling to poles
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

    // Transform Q based on filter type (type 7 uses Formula 2)
    let q_transformed = compute_effective_q(user_q, 7);

    // Convert gain_db to linear gain (applies to all types)
    let linear_gain = (user_gain_db * LN10_OVER_20).exp();

    let mut sections = Vec::new();

    for _k in 0..n {
        // Create order-2 Butterworth prototype
        let mut zpk = prototype::butterworth_lp(2);

        // Apply shelf gain scaling (Type 7 uses const_low = 0.5)
        apply_shelf_gain_to_zpk_type7(&mut zpk, q_transformed);

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
/// Type 8 is high shelf: gain is applied via linear gain scaling in biquad coefficients.
/// Unlike type 7, this type does NOT call apply_shelf_gain_to_zpk.
///
/// Pipeline:
/// 1. Create Butterworth LP prototype (order 2)
/// 2. Transform Q based on filter type (type 8 uses Formula 2)
/// 3. Apply bilinear transform directly (NO pole scaling)
/// 4. Convert ZPK to biquad coefficients
/// 5. Apply linear gain to numerator coefficients
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

    // Transform Q based on filter type (type 8 uses Formula 2)
    let q_transformed = compute_effective_q(user_q, 8);

    // Convert gain_db to linear gain
    let linear_gain = (user_gain_db * LN10_OVER_20).exp();

    let mut sections = Vec::new();

    for _k in 0..n {
        // Create order-2 Butterworth prototype
        let mut zpk = prototype::butterworth_lp(2);

        // Type 8 does NOT call apply_shelf_gain_to_zpk!
        // Instead, poles are scaled by q_transformed (without the gain_param exponent)
        for pole in zpk.poles.iter_mut() {
            *pole = *pole * q_transformed;
        }

        // Apply bilinear transform
        zpk = transform::bilinear(&zpk, sample_rate);

        // Convert to biquad and apply gain
        let mut sos = biquad::zpk_to_sos(&zpk);

        // Apply linear gain to numerator
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

/// Design a tilt shelf filter (type 9) via ZPK pipeline.
///
/// Type 9 is tilt: scale_zpk_coefficients_by_gain is called AFTER bilinear transform.
/// This applies a post-bilinear scaling with formula: gain_scale = Q * sqrt(2)
///
/// Pipeline:
/// 1. Create Butterworth LP prototype (order 2)
/// 2. Transform Q based on filter type (type 9 uses Formula 2)
/// 3. Apply bilinear transform
/// 4. Convert ZPK to biquad coefficients
/// 5. Apply scale_zpk_coefficients_by_gain post-bilinear
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

    // Transform Q based on filter type (type 9 uses Formula 2)
    let q_transformed = compute_effective_q(user_q, 9);

    // Convert gain_db to linear gain
    let linear_gain = (user_gain_db * LN10_OVER_20).exp();

    // Post-bilinear scaling for type 9: Q * sqrt(2)
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
pub fn design_band_shelf_zpk(
    n_sections: usize,
    _freq_hz: f64,
    _user_q: f64,
    user_gain_db: f64,
    _sample_rate: f64,
) -> Vec<Coeffs> {
    let n = n_sections.max(1);

    if user_gain_db.abs() < 0.001 {
        return vec![crate::biquad::PASSTHROUGH; n];
    }

    // TODO: Type 10 implementation
    // For now, return passthrough
    vec![crate::biquad::PASSTHROUGH; n]
}

/// Apply shelf gain scaling to ZPK (type 7, low shelf).
///
/// From binary analysis (0x1800fcce0 apply_shelf_gain_to_zpk):
/// - Zeros are multiplied by gain_param
/// - Poles are divided by gain_param
/// - ZPK gain is set to 1/gain_param
///
/// Where gain_param = pow(effective_Q, 0.5/order)
fn apply_shelf_gain_to_zpk_type7(zpk: &mut Zpk, effective_q: f64) {
    let order = zpk.poles.len();

    if order == 0 {
        return;
    }

    let gain_param = effective_q.powf(CONST_LOW_SHELF / order as f64);

    // Scale zeros (Butterworth LP has no zeros, but structure requires this)
    for zero in zpk.zeros.iter_mut() {
        *zero = *zero * gain_param;
    }

    // Scale poles
    for pole in zpk.poles.iter_mut() {
        *pole = *pole / gain_param;
    }

    // Set ZPK gain
    zpk.gain = 1.0 / gain_param;
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
        // Formula 2: Simple scaling by INV_SQRT2 for shelf/tilt/allpass
        8 | 9 | 11 | _ => user_q * INV_SQRT2,
    }
}
