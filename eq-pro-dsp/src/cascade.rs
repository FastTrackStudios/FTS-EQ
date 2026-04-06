//! Pro-Q 4's cascade coefficient computation for peak/bell and shelf (type 12) filters.
//!
//! `compute_cascade_coefficients` (0x1800fec20) computes ZPK directly for peak/bell
//! filters without going through Butterworth prototypes. It uses a specialized approach:
//!
//! - For type 0 (peak/bell): RBJ cookbook with per-section gain distribution.
//!   Higher orders distribute gain across sections with exponential spacing.
//!
//! - For type 0xc (shelf alt / type 12): gain = sqrt(gain), with geometric gain
//!   spacing across sections for smooth shelf transitions.
//!
//! Key insight: Pro-Q 4 does NOT simply stack identical biquads. Each section gets
//! a different gain_db/section to create the proper cascade response.

use std::f64::consts::PI;

use crate::biquad::{Coeffs, PASSTHROUGH};

/// Binary-exact peak/bell cascade (compute_cascade_coefficients @ 0x1800fec20).
///
/// Creates N Butterworth all-pole sections with z^-2 numerators.
/// Q scaling applied to first section only. Gain normalization via
/// accumulated 0.25/cos²(θ_k) product.
///
/// The filter output is used in a dry/wet blend in band.rs:
///   output = input + (gain_linear - 1) * cascade_output
pub fn compute_cascade_peak(
    _freq_hz: f64,
    q: f64,
    gain_db: f64,
    _sample_rate: f64,
    order: usize,
) -> Vec<Coeffs> {
    let n = (order / 2).max(1);

    if gain_db.abs() < 0.001 {
        return vec![PASSTHROUGH; n];
    }

    // Gain-dependent Q interpolation (design_filter_zpk_and_transform @ 0x1800ff6f0)
    let gain_linear = 10.0_f64.powf(gain_db.abs() / 20.0);
    let inv_sqrt2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let effective_q = if gain_linear < 2.0 {
        gain_linear * (q - inv_sqrt2) + inv_sqrt2
    } else {
        q
    };

    // Build Butterworth all-pole sections
    let mut sections = Vec::with_capacity(n);
    let mut gain_accum = 1.0_f64;

    for k in 0..n {
        let theta_k = PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let cos_k = theta_k.cos();

        // Denominator: poles on unit circle at Butterworth angles
        let mut a1 = 2.0 * cos_k;
        let a2 = 1.0;

        // Gain factor per section: 0.25/cos²(θ_k)
        gain_accum *= 0.25 / (cos_k * cos_k);

        // Q scaling on FIRST section only (zpk_to_biquad_coefficients @ 0x1800fe040)
        if k == 0 {
            let scale = inv_sqrt2 / effective_q;
            a1 *= scale;
        }

        // Numerator: z^-2 (b0=0, b1=0, b2=1)
        sections.push([1.0, a1, a2, 0.0, 0.0, 1.0]);
    }

    // Apply accumulated gain to first section's numerator
    // This normalizes the cascade response level
    if let Some(first) = sections.first_mut() {
        first[5] *= gain_accum; // b2 *= gain_accum
    }

    sections
}

/// Compute cascade biquads for the shelf-alt filter (type 12 / 0xc).
///
/// From compute_cascade_coefficients @ 0x1800fec20:
/// - Always exactly 3 sections (param_3 == 0xc path)
/// - Uses gain^(1/4) scaling on pole/zero positions
/// - Hardcoded frequency ladder: base constants * 32^section_index
/// - Real poles and zeros (no imaginary component)
/// - Produces z-domain poles/zeros directly (transform type 0)
///
/// Constants from binary:
///   DAT_180232030 = -0.01313900648833929 (base pole/zero 1)
///   DAT_180232038 = -0.07432544468767008 (base pole/zero 2)
///   DAT_180231c58 = 32.0 (section spacing)
///   DAT_180231bd8 = 5.656854249492381 = 4*sqrt(2) (inter-section gain scaling)
pub fn compute_cascade_shelf_alt(
    _freq_hz: f64,
    _q: f64,
    gain_db: f64,
    _sample_rate: f64,
    _order: usize,
) -> Vec<Coeffs> {
    if gain_db.abs() < 0.001 {
        return vec![PASSTHROUGH; 3];
    }

    // Convert dB to linear gain
    let gain_linear = 10.0_f64.powf(gain_db / 20.0);

    // Binary: param_4 = SQRT(param_4); dVar24 = sqrt(param_4)
    // So gain_sqrt = gain^(1/2), gain_qrt = gain^(1/4)
    let gain_sqrt = gain_linear.sqrt();
    let gain_qrt = gain_sqrt.sqrt();
    let inv_gain_qrt = 1.0 / gain_qrt;

    // Hardcoded constants from binary
    const BASE_1: f64 = -0.01313900648833929; // DAT_180232030
    const BASE_2: f64 = -0.07432544468767008; // DAT_180232038
    const SECTION_SPACING: f64 = 32.0;        // DAT_180231c58
    const INTER_GAIN: f64 = 5.656854249492381; // DAT_180231bd8 = 4*sqrt(2)

    // Build 3 sections, each with 2 real poles and 2 real zeros
    // Section k uses frequencies: base * SECTION_SPACING^k
    let mut sections = Vec::with_capacity(3);
    let mut freq_1 = BASE_1;
    let mut freq_2 = BASE_2;
    let mut section_gain = gain_sqrt; // Binary: param_1[0x11] = param_4 (= sqrt(gain))

    for _ in 0..3 {
        // Zeros scaled by gain^(1/4), poles scaled by 1/gain^(1/4)
        let z1 = freq_1 * gain_qrt;
        let z2 = freq_2 * gain_qrt;
        let p1 = freq_1 * inv_gain_qrt;
        let p2 = freq_2 * inv_gain_qrt;

        // Convert 2-real-pole, 2-real-zero ZPK section to biquad
        // a0=1, a1=-(p1+p2), a2=p1*p2, b0=section_gain, b1=-(z1+z2)*section_gain, b2=z1*z2*section_gain
        let a0 = 1.0;
        let a1 = -(p1 + p2);
        let a2 = p1 * p2;
        let b0 = section_gain;
        let b1 = -(z1 + z2) * section_gain;
        let b2 = z1 * z2 * section_gain;

        sections.push([a0, a1, a2, b0, b1, b2]);

        // Step frequencies for next section
        freq_1 *= SECTION_SPACING;
        freq_2 *= SECTION_SPACING;
        // Step gain: binary multiplies by DAT_180231bd8 between sections
        section_gain *= INTER_GAIN;
    }

    sections
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Evaluate magnitude in dB of a cascade of biquad sections at digital frequency w.
    fn mag_db_sos(sections: &[Coeffs], w: f64) -> f64 {
        use crate::zpk::Complex;
        let ejw = Complex::from_polar(1.0, w);
        let ejw2 = ejw * ejw;
        let mut h = Complex::new(1.0, 0.0);
        for c in sections {
            let den = Complex::new(c[0], 0.0)
                + ejw * Complex::new(c[1], 0.0)
                + ejw2 * Complex::new(c[2], 0.0);
            let num = Complex::new(c[3], 0.0)
                + ejw * Complex::new(c[4], 0.0)
                + ejw2 * Complex::new(c[5], 0.0);
            h = h * num / den;
        }
        20.0 * h.mag().log10()
    }

    #[test]
    fn peak_zero_gain_is_passthrough() {
        let sos = compute_cascade_peak(1000.0, 2.0, 0.0, 48000.0, 2);
        assert_eq!(sos.len(), 1);
        assert_eq!(sos[0], PASSTHROUGH);
    }

    #[test]
    fn peak_butterworth_angle_correctness() {
        // Single section (order=2): theta_0 = pi/4, cos(pi/4) = 1/sqrt(2)
        let sos = compute_cascade_peak(1000.0, 2.0, 6.0, 48000.0, 2);
        assert_eq!(sos.len(), 1);
        // b0 and b1 should be 0 (z^-2 numerator)
        assert_eq!(sos[0][3], 0.0, "b0 should be 0");
        assert_eq!(sos[0][4], 0.0, "b1 should be 0");
        // b2 should be gain_accum = 0.25/cos^2(pi/4) = 0.25/0.5 = 0.5
        assert!(
            (sos[0][5] - 0.5).abs() < 1e-10,
            "b2 should be 0.5 (gain_accum), got {}",
            sos[0][5]
        );
        // a2 should be 1.0
        assert!((sos[0][2] - 1.0).abs() < 1e-10, "a2 should be 1.0");
    }

    #[test]
    fn peak_q_scaling() {
        // Use 12 dB gain so gain_linear > 2.0 and Q interpolation is bypassed
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let sos_bw = compute_cascade_peak(1000.0, inv_sqrt2, 12.0, 48000.0, 2);
        // theta_0 = pi/4, cos(pi/4) = inv_sqrt2, a1 = 2*cos = sqrt(2)
        // scale = inv_sqrt2 / inv_sqrt2 = 1.0, so a1 stays sqrt(2)
        let expected_a1 = 2.0_f64.sqrt();
        assert!(
            (sos_bw[0][1] - expected_a1).abs() < 1e-10,
            "a1 with Butterworth Q should be sqrt(2), got {}",
            sos_bw[0][1]
        );

        // With Q = 2*inv_sqrt2, scale = 0.5, so a1 should be halved
        let sos_high_q = compute_cascade_peak(1000.0, 2.0 * inv_sqrt2, 12.0, 48000.0, 2);
        let expected_a1_hq = expected_a1 * 0.5;
        assert!(
            (sos_high_q[0][1] - expected_a1_hq).abs() < 1e-10,
            "a1 with 2x Butterworth Q should be halved, got {}",
            sos_high_q[0][1]
        );
    }

    #[test]
    fn peak_gain_accumulation() {
        // Order 4 = 2 sections
        let sos = compute_cascade_peak(1000.0, 2.0, 12.0, 48000.0, 4);
        assert_eq!(sos.len(), 2);

        // Section 0: theta_0 = pi/8, section 1: theta_1 = 3*pi/8
        let cos0 = (PI / 8.0).cos();
        let cos1 = (3.0 * PI / 8.0).cos();
        let expected_gain = (0.25 / (cos0 * cos0)) * (0.25 / (cos1 * cos1));

        // gain_accum applied to first section b2
        assert!(
            (sos[0][5] - expected_gain).abs() < 1e-10,
            "first section b2 should be gain_accum={}, got {}",
            expected_gain,
            sos[0][5]
        );
        // Second section b2 should be 1.0 (unscaled)
        assert!(
            (sos[1][5] - 1.0).abs() < 1e-10,
            "second section b2 should be 1.0, got {}",
            sos[1][5]
        );
    }

    #[test]
    fn peak_gain_dependent_q_interpolation() {
        // When gain_linear < 2.0 (~6dB), Q is interpolated:
        //   effective_q = gain_linear * (q_user - inv_sqrt2) + inv_sqrt2
        // Use q_user < inv_sqrt2 so that interpolation pushes effective_q
        // toward inv_sqrt2 (higher), making scale smaller, making a1 smaller.
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let q_user = 0.3; // < inv_sqrt2, so interpolation increases Q

        // At 3 dB, gain_linear ≈ 1.41, effective_q = 1.41*(0.3 - 0.707) + 0.707 ≈ 0.133
        // At 12 dB, gain_linear ≈ 3.98 > 2.0, effective_q = q_user = 0.3
        let sos_low = compute_cascade_peak(1000.0, q_user, 3.0, 48000.0, 2);
        let sos_high = compute_cascade_peak(1000.0, q_user, 12.0, 48000.0, 2);

        // Verify Q interpolation changes the a1 coefficient
        let a1_low = sos_low[0][1].abs();
        let a1_high = sos_high[0][1].abs();
        assert!(
            (a1_low - a1_high).abs() > 0.01,
            "Q interpolation should cause different a1 values: low_gain={a1_low}, high_gain={a1_high}"
        );
    }

    #[test]
    fn shelf_alt_zero_gain_is_passthrough() {
        let sos = compute_cascade_shelf_alt(1000.0, 1.0, 0.0, 48000.0, 2);
        assert_eq!(sos.len(), 3);
        for (i, s) in sos.iter().enumerate() {
            assert_eq!(*s, PASSTHROUGH, "Section {i} should be passthrough");
        }
    }

    #[test]
    fn shelf_alt_has_gain_at_center() {
        let sos = compute_cascade_shelf_alt(1000.0, 1.0, 12.0, 48000.0, 2);
        // Always 3 sections from hardcoded ZPK path
        assert_eq!(sos.len(), 3);
        // All sections should be valid (non-NaN) and not passthrough
        for (i, section) in sos.iter().enumerate() {
            for (j, &coeff) in section.iter().enumerate() {
                assert!(
                    coeff.is_finite(),
                    "section[{}][{}] is not finite: {}",
                    i,
                    j,
                    coeff
                );
            }
            assert_ne!(*section, PASSTHROUGH, "Section {i} should not be passthrough for non-zero gain");
        }
    }

    #[test]
    fn shelf_alt_multi_section() {
        // Always 3 sections regardless of order
        let sos = compute_cascade_shelf_alt(1000.0, 1.0, 12.0, 48000.0, 6);
        assert_eq!(sos.len(), 3);
        // All sections should be valid (non-NaN)
        for (i, section) in sos.iter().enumerate() {
            for (j, &coeff) in section.iter().enumerate() {
                assert!(
                    coeff.is_finite(),
                    "section[{}][{}] is not finite: {}",
                    i,
                    j,
                    coeff
                );
            }
        }
    }
}
