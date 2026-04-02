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
//! MYSTERY: The relationship between (user_Q, user_gain_dB) → effective_Q is unknown!
//! The dB gain parameter is NOT directly used in apply_shelf_gain_to_zpk.
//! Hypothesis: Gain is pre-computed into Q by the plugin parameter interface.

use std::f64::consts::PI;

use crate::biquad::Coeffs;
use crate::prototype;
use crate::transform;
use crate::zpk::Zpk;

/// Design a low shelf filter (type 7) via actual ZPK pipeline.
///
/// Input: user_Q and user_gain_dB
/// Output: Biquad coefficients
///
/// # WARNING
/// The actual dB gain formula (user_Q, user_gain_dB) → effective_Q is not yet known!
/// This implementation currently uses user_Q directly, which will NOT produce correct gain.
/// The missing formula is likely in the plugin parameter processing layer.
pub fn design_low_shelf_zpk(
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

    // TODO: This is the missing piece!
    // We need to find: effective_Q = f(user_Q, user_gain_dB, shelf_type=7)
    // For now, use user_Q directly (WRONG but shows the structure)
    let effective_q = user_q;

    let mut sections = Vec::new();

    for k in 0..n {
        // Create order-2 Butterworth prototype
        let mut zpk = prototype::butterworth_lp(2);

        // Apply shelf gain scaling (Type 7 uses const_low = 0.5)
        apply_shelf_gain_to_zpk_type7(&mut zpk, effective_q);

        // Apply bilinear transform
        let w0 = 2.0 * PI * freq_hz / sample_rate;
        // TODO: apply bilinear transform

        // Convert to biquad
        // TODO: convert ZPK to biquad coefficients

        // Placeholder
        sections.push(crate::biquad::PASSTHROUGH);
    }

    sections
}

/// Design a high shelf filter (type 8) via ZPK pipeline.
///
/// # WARNING
/// This implementation is incomplete - the actual gain application mechanism is unknown!
pub fn design_high_shelf_zpk(
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

    // For high shelf, apply_shelf_gain_to_zpk is NOT called!
    // Gain is applied differently - either in zpk_to_biquad_coefficients
    // or through a different mechanism entirely.

    vec![crate::biquad::PASSTHROUGH; n]
}

/// Design a tilt shelf filter (type 9) via ZPK pipeline.
///
/// Type 9 is special: scale_zpk_coefficients_by_gain is called AFTER bilinear transform.
///
/// # WARNING
/// The relationship between user parameters and the actual gain formula is unknown!
pub fn design_tilt_shelf_zpk(
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

    // Type 9 uses scale_zpk_coefficients_by_gain with formula:
    // gain_scale = 1.0 / ((1.0 / Q) * (1.0 / INV_SQRT2))
    //            = Q * sqrt(2)
    //
    // But this doesn't directly use user_gain_db either!
    // The dB gain must be encoded into the Q parameter somehow.

    vec![crate::biquad::PASSTHROUGH; n]
}

/// Design a band shelf filter (type 10) via ZPK pipeline.
///
/// # WARNING
/// Unknown implementation details!
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
    let const_low = 0.5_f64;  // From binary at 0x180231a00

    if order == 0 {
        return;
    }

    let gain_param = effective_q.powf(const_low / order as f64);

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

/// MISSING FORMULA
///
/// This is the critical missing piece! We need to find:
/// ```
/// fn compute_effective_q(user_q: f64, user_gain_db: f64, shelf_type: u32) -> f64
/// ```
///
/// Current hypotheses:
/// 1. effective_Q = user_Q * (gain_linear)^power
/// 2. effective_Q = user_Q^(gain_linear)
/// 3. effective_Q is computed by the plugin parameter layer before calling DSP
///
/// Testing against Pro-Q 4 reference would show which formula is correct.
/// The formula likely differs for type 7 (low), type 8 (high), type 9 (tilt), type 10 (band).
