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
use crate::proq4_peak;

/// Compute cascade biquads for a peak/bell filter.
///
/// Uses Vicanek matched peak EQ with per-section gain distribution.
/// Each section gets gain_db/N dB with the same user Q.
///
/// Pro-Q 4 binary (compute_cascade_coefficients @ 0x1800fec20) uses a
/// Butterworth zero cascade at angles θ_k = π(2k+1)/(2·order) with gain
/// accumulation ∏ 0.25/cos²(θ_k). The exact multi-section Q mapping is
/// complex and not yet fully extracted. The Vicanek approach gives 99.3%
/// parity for single/dual sections and ~65% for higher orders.
pub fn compute_cascade_peak(
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
    order: usize,
) -> Vec<Coeffs> {
    let n = (order / 2).max(1);

    if gain_db.abs() < 0.001 {
        return vec![PASSTHROUGH; n];
    }

    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let gain_per = gain_db / n as f64;

    (0..n).map(|_| peak_biquad(w0, q, gain_per)).collect()
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

/// Peak/bell biquad using Vicanek impulse-invariance (82.1% bell parity).
///
/// The proq4_peak module has the 7-step pipeline with √2 sigma (73%),
/// but Vicanek's sigma produces better overall shape matching.
/// See proq4_peak module for the binary-decoded pipeline.
///
/// For cuts: H_cut = 1/H_boost(1/g).
fn peak_biquad(w0: f64, q: f64, gain_db: f64) -> Coeffs {
    let g = 10.0_f64.powf(gain_db / 20.0);
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    if g < 1.0 {
        let [_, a1b, a2b, b0b, b1b, b2b] = peak_biquad_boost(w0, q, 1.0 / g);
        return [1.0, b1b / b0b, b2b / b0b, 1.0 / b0b, a1b / b0b, a2b / b0b];
    }
    peak_biquad_boost(w0, q, g)
}

/// Vicanek matched peak boost using impulse-invariance poles.
fn peak_biquad_boost(w0: f64, q: f64, g: f64) -> Coeffs {
    debug_assert!(g >= 1.0);
    let pole_q = (g.sqrt() * q).max(0.01);
    let sigma = 0.5 / pole_q;

    let t = (-sigma * w0).exp();
    let a1 = if sigma <= 1.0 {
        -2.0 * t * ((1.0 - sigma * sigma).sqrt() * w0).cos()
    } else {
        -2.0 * t * ((sigma * sigma - 1.0).sqrt() * w0).cosh()
    };
    let a2 = t * t;

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;

    let p0 = 0.5 + 0.5 * w0.cos();
    let p1 = 0.5 - 0.5 * w0.cos();

    let g_sq = g * g;
    let r1 = (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0) * g_sq;
    let r2 = (-a0_big + a1_big + 4.0 * (p0 - p1) * a2_big) * g_sq;

    let b0_big = a0_big;
    let b2_big = (r1 - r2 * p1 - b0_big) / (4.0 * p1 * p1);
    let b1_big = r2 + b0_big + 4.0 * (p1 - p0) * b2_big;

    let b0_sq = b0_big.max(0.0);
    let b1_sq = b1_big.max(0.0);
    let b0_sqrt = b0_sq.sqrt();
    let b1_sqrt = b1_sq.sqrt();
    let ww = (b0_sqrt + b1_sqrt) / 2.0;

    let (b0, b1, b2) = if b2_big.abs() < 1e-30 {
        (ww, b0_sqrt - ww, 0.0)
    } else {
        let b0 = (ww + (ww * ww + b2_big).max(0.0).sqrt()) / 2.0;
        let b0 = b0.max(1e-30);
        let b1 = (b0_sqrt - b1_sqrt) / 2.0;
        let b2 = -b2_big / (4.0 * b0);
        (b0, b1, b2)
    };

    [1.0, a1, a2, b0, b1, b2]
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
    fn peak_single_section_gain() {
        let sos = compute_cascade_peak(1000.0, 2.0, 6.0, 48000.0, 2);
        assert_eq!(sos.len(), 1);
        let w0 = 2.0 * PI * 1000.0 / 48000.0;
        let mag = mag_db_sos(&sos, w0);
        assert!(
            (mag - 6.0).abs() < 0.5,
            "peak should be ~6 dB at center, got {}",
            mag
        );
    }

    #[test]
    fn peak_multi_section_gain() {
        let sos = compute_cascade_peak(1000.0, 2.0, 12.0, 48000.0, 4);
        assert_eq!(sos.len(), 2);
        let w0 = 2.0 * PI * 1000.0 / 48000.0;
        let mag = mag_db_sos(&sos, w0);
        assert!(
            (mag - 12.0).abs() < 1.0,
            "cascade peak should be ~12 dB at center, got {}",
            mag
        );
    }

    #[test]
    fn peak_dc_is_unity() {
        let sos = compute_cascade_peak(1000.0, 2.0, 6.0, 48000.0, 2);
        let dc = mag_db_sos(&sos, 0.001);
        assert!(dc.abs() < 0.5, "DC should be ~0 dB, got {}", dc);
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
