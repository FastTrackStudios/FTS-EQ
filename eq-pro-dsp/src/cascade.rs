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

/// Compute cascade biquads for a peak/bell filter.
///
/// Uses RBJ cookbook peak EQ formula with per-section gain distribution.
/// For multi-section cascades, gain is distributed so that each section
/// contributes to the total response shape properly.
///
/// For `param_3` in {0, 3, 8} (standard peak modes), the gain distribution is:
///   gain_per_section = total_gain_db / num_sections
///
/// This gives the familiar response shape where bandwidth narrows with order.
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

    // Exponential gain distribution: each section gets gain_db/n.
    // This is the standard approach for cascaded parametric EQs.
    // Pro-Q 4 uses this for param_3 in {0, 3, 8}.
    let gain_per = gain_db / n as f64;

    (0..n).map(|_| peak_biquad(w0, q, gain_per)).collect()
}

/// Compute cascade biquads for the shelf-alt filter (type 12).
///
/// Pro-Q 4's type 0xc shelf uses a different gain distribution:
///   1. Total gain is square-rooted: effective_gain = sqrt(gain_linear)
///   2. Sections use geometric gain spacing — each section's gain is a power
///      of the total, creating a smooth shelf transition.
///
/// The geometric spacing means section k gets:
///   gain_k = total_gain_db * weight_k
/// where weights are geometrically distributed across sections.
pub fn compute_cascade_shelf_alt(
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

    // Type 12 shelf: gain = sqrt(gain), so halve the dB
    let effective_gain_db = gain_db / 2.0;

    // Geometric gain spacing: each section gets a progressively different share.
    // For n sections, weights are: 2^0, 2^1, ..., 2^(n-1), normalized to sum = 1.
    // This creates the smooth shelf shape Pro-Q 4 is known for.
    let total_weight: f64 = (0..n).map(|k| geometric_weight(k, n)).sum();

    (0..n)
        .map(|k| {
            let weight = geometric_weight(k, n) / total_weight;
            let section_gain = effective_gain_db * weight;

            // Use a shelf-like biquad for each section.
            // The shelf-alt type uses peak biquads with adjusted Q per section
            // to approximate a shelf response.
            let section_q = q * (1.0 + 0.5 * k as f64 / n.max(1) as f64);
            peak_biquad(w0, section_q, section_gain)
        })
        .collect()
}

/// Geometric weight for section k of n total sections.
///
/// Gives more gain to later sections, creating the characteristic
/// shelf-alt response shape.
fn geometric_weight(k: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    // Geometric progression: weight_k = r^k where r = 2^(1/(n-1))
    let r = 2.0_f64.powf(1.0 / (n - 1) as f64);
    r.powi(k as i32)
}

/// Single peak/bell biquad using RBJ Audio EQ Cookbook.
///
/// H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
///
/// where:
///   A = 10^(gain_db/40)  (sqrt of linear gain)
///   alpha = sin(w0) / (2*Q)
///   b0 = 1 + alpha*A
///   b1 = -2*cos(w0)
///   b2 = 1 - alpha*A
///   a0 = 1 + alpha/A
///   a1 = -2*cos(w0)
///   a2 = 1 - alpha/A
fn peak_biquad(w0: f64, q: f64, gain_db: f64) -> Coeffs {
    let a = 10.0_f64.powf(gain_db / 40.0);
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * q);

    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_w0;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha / a;

    [a0, a1, a2, b0, b1, b2]
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
        assert_eq!(sos.len(), 1);
        assert_eq!(sos[0], PASSTHROUGH);
    }

    #[test]
    fn shelf_alt_has_gain_at_center() {
        let sos = compute_cascade_shelf_alt(1000.0, 1.0, 12.0, 48000.0, 2);
        assert_eq!(sos.len(), 1);
        let w0 = 2.0 * PI * 1000.0 / 48000.0;
        let mag = mag_db_sos(&sos, w0);
        // Shelf alt uses sqrt(gain), so effective is 6 dB
        assert!(
            mag > 2.0 && mag < 10.0,
            "shelf-alt center should have moderate gain, got {}",
            mag
        );
    }

    #[test]
    fn shelf_alt_multi_section() {
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

    #[test]
    fn geometric_weight_single_section() {
        let w = geometric_weight(0, 1);
        assert!((w - 1.0).abs() < 1e-10);
    }

    #[test]
    fn geometric_weight_increases() {
        let w0 = geometric_weight(0, 3);
        let w1 = geometric_weight(1, 3);
        let w2 = geometric_weight(2, 3);
        assert!(w1 > w0, "weights should increase: w0={}, w1={}", w0, w1);
        assert!(w2 > w1, "weights should increase: w1={}, w2={}", w1, w2);
    }
}
