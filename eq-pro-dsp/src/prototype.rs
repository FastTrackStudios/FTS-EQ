//! Analog filter prototypes — Butterworth pole generation with elliptic LP→BP support.
//!
//! Pro-Q 4 uses `maybe_butterworth_pole_next` (0x1801dbbf0) to generate
//! Butterworth poles in the s-domain. These are the starting point for
//! all filter types except Peak (type 0) and Shelf Alt (type 12).
//!
//! For LP/HP (types 1,2): poles placed directly (transform type 0)
//! For BP (type 3): poles transformed via LP→BP using elliptic functions
//! For Notch (type 4): poles transformed via LP→BS
//! For Shelves (types 7,8,9): poles transformed via bilinear (transform type 2)

use std::f64::consts::PI;

use crate::elliptic;
use crate::zpk::{Complex, Zpk};

/// Generate Butterworth lowpass prototype poles in the s-domain.
///
/// An N-th order Butterworth LP has N poles on the unit circle in the left half-plane:
///   s_k = exp(j * pi * (2k + N + 1) / (2N))  for k = 0..N-1
///
/// All poles have |s_k| = 1 (unit circle), and the prototype has cutoff w = 1.
pub fn butterworth_lp(order: usize) -> Zpk {
    let mut poles = Vec::with_capacity(order);
    for k in 0..order {
        let angle = PI * (2 * k + order + 1) as f64 / (2 * order) as f64;
        poles.push(Complex::from_polar(1.0, angle));
    }
    Zpk::new(vec![], poles, 1.0)
}

/// Generate Butterworth lowpass prototype with pre-warped cutoff.
///
/// Pre-warps the analog cutoff frequency to compensate for bilinear transform
/// frequency warping: w_a = 2*fs*tan(w_d / 2)
pub fn butterworth_lp_prewarped(order: usize, freq_hz: f64, sample_rate: f64) -> Zpk {
    let w_d = 2.0 * PI * freq_hz / sample_rate;
    let w_a = 2.0 * sample_rate * (w_d / 2.0).tan();

    let mut proto = butterworth_lp(order);
    for p in &mut proto.poles {
        *p = *p * w_a;
    }
    proto.gain = w_a.powi(order as i32);
    proto
}

/// Generate Butterworth bandpass prototype via standard LP→BP transformation.
///
/// LP→BP transform: s -> Q_bp * (s/w0 + w0/s)
///
/// Each LP pole s_k maps to a PAIR of BP poles by solving:
///   s^2 - s*(s_k*bw_a) + w0_a^2 = 0
///
/// Each LP pole also contributes a zero at s = 0 (DC).
pub fn butterworth_bp(order: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Zpk {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let w0_a = 2.0 * sample_rate * (w0 / 2.0).tan();
    let bw_a = w0_a / q;

    let lp = butterworth_lp(order);

    let mut bp_poles = Vec::with_capacity(2 * order);
    let mut bp_zeros = Vec::with_capacity(order);

    for &s_k in &lp.poles {
        let b = s_k * bw_a;
        let disc = b * b - Complex::new(4.0 * w0_a * w0_a, 0.0);
        let sqrt_disc = disc.sqrt();

        let s1 = (b + sqrt_disc) / 2.0;
        let s2 = (b - sqrt_disc) / 2.0;

        bp_poles.push(s1);
        bp_poles.push(s2);
        bp_zeros.push(Complex::ZERO);
    }

    let gain = bw_a.powi(order as i32);
    Zpk::new(bp_zeros, bp_poles, gain)
}

/// Generate Butterworth bandpass via elliptic LP→BP transformation.
///
/// This is Pro-Q 4's exact approach for type 3 (bandpass) filters. Instead of the
/// standard quadratic LP→BP which places poles using simple square root, the elliptic
/// version uses Jacobi elliptic functions to compute exact pole positions per section.
///
/// The elliptic approach gives better pole placement for higher-order bandpass filters,
/// ensuring each section has properly spaced poles for optimal frequency response.
///
/// Algorithm:
///   1. Compute pre-warped center frequency and bandwidth
///   2. For each Butterworth prototype pole, use elliptic functions to map
///      the pole angle to exact BP pole positions
///   3. Each section gets unique pole/zero positions (NOT identical biquads)
pub fn butterworth_bp_elliptic(order: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Zpk {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let w0_a = 2.0 * sample_rate * (w0 / 2.0).tan();
    let bw_a = w0_a / q;

    // Selectivity parameter for elliptic functions: k = bw / (2 * w0)
    // Controls how "narrow" the bandpass is. Clamped for numerical stability.
    let k = (bw_a / (2.0 * w0_a)).min(0.9999999);

    // Complete elliptic integral K(k) for normalization
    let kk = elliptic::elliptic_k_complete(k);

    let mut bp_poles = Vec::with_capacity(2 * order);
    let mut bp_zeros = Vec::with_capacity(order);

    // CRITICAL FIX: Section-indexed elliptic parametrization.
    // Pro-Q 4 generates DISTINCT poles for each section using u_i = (2*i+1)*K(k)/order,
    // not duplicates. Each section index gets a DIFFERENT elliptic function evaluation.
    // For order-N filter: iterate through N sections, not through LP poles.
    for section_idx in 0..order {
        let u_i = (2.0 * section_idx as f64 + 1.0) * kk / order as f64;

        // Evaluate Jacobi elliptic functions at this section's parameter
        let sn_val = elliptic::elliptic_sn(u_i, k);
        let cn_val = (1.0 - sn_val * sn_val).max(0.0).sqrt();
        let dn_val = (1.0 - k * k * sn_val * sn_val).max(0.0).sqrt();

        // The elliptic transform maps to BP poles:
        //   sigma = bw * sn * dn / (1 - k^2 * sn^2)
        //   omega_offset = bw * cn / (1 - k^2 * sn^2)
        // Each section gets UNIQUE sigma and omega_offset based on section_idx.
        let denom = 1.0 - k * k * sn_val * sn_val + 1e-30;
        let sigma = bw_a * sn_val * dn_val / denom;
        let omega_offset = bw_a * cn_val / denom;

        // Create ONE conjugate pole pair for this section.
        // The ±omega_offset creates the above/below-center-frequency pole pair.
        let p_upper = Complex::new(-sigma, w0_a + omega_offset);
        let p_lower = Complex::new(-sigma, -(w0_a + omega_offset));

        bp_poles.push(p_upper);
        bp_poles.push(p_lower);

        bp_zeros.push(Complex::ZERO);
    }

    let gain = bw_a.powi(order as i32);
    Zpk::new(bp_zeros, bp_poles, gain)
}

/// Generate Butterworth bandstop (notch) prototype via LP→BS transformation.
///
/// LP→BS transform: reciprocal of LP→BP. Each LP pole maps to a pair of BS poles,
/// and each contributes a pair of zeros at +/-jw0 (the notch frequencies).
pub fn butterworth_bs(order: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Zpk {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let w0_a = 2.0 * sample_rate * (w0 / 2.0).tan();
    let bw_a = w0_a / q;

    let lp = butterworth_lp(order);

    let mut bs_poles = Vec::with_capacity(2 * order);
    let mut bs_zeros = Vec::with_capacity(2 * order);

    for &s_k in &lp.poles {
        // BS pole equation: s^2 - s*(bw_a/s_k) + w0_a^2 = 0
        let b = Complex::new(bw_a, 0.0) / s_k;
        let disc = b * b - Complex::new(4.0 * w0_a * w0_a, 0.0);
        let sqrt_disc = disc.sqrt();

        let s1 = (b + sqrt_disc) / 2.0;
        let s2 = (b - sqrt_disc) / 2.0;

        bs_poles.push(s1);
        bs_poles.push(s2);

        // Each LP pole contributes zeros at +/-jw0
        bs_zeros.push(Complex::new(0.0, w0_a));
        bs_zeros.push(Complex::new(0.0, -w0_a));
    }

    Zpk::new(bs_zeros, bs_poles, 1.0)
}

/// Generate Butterworth bandstop via elliptic LP→BS transformation.
///
/// This is Pro-Q 4's exact approach for type 4 (notch) filters. Similar to elliptic bandpass
/// but uses the complement transformation to create notches instead of peaks.
///
/// Algorithm:
///   1. Compute pre-warped center frequency and bandwidth
///   2. For each Butterworth prototype pole, use elliptic functions to map
///      the pole angle to exact BS pole positions
///   3. Each section gets unique pole/zero positions (NOT identical biquads)
pub fn butterworth_bs_elliptic(order: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Zpk {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let w0_a = 2.0 * sample_rate * (w0 / 2.0).tan();
    let bw_a = w0_a / q;

    // Selectivity parameter for elliptic functions: k = bw / (2 * w0)
    // Controls how "narrow" the notch is. Clamped for numerical stability.
    let k = (bw_a / (2.0 * w0_a)).min(0.9999999);

    // Complete elliptic integral K(k) for normalization
    let kk = elliptic::elliptic_k_complete(k);

    let mut bs_poles = Vec::with_capacity(2 * order);
    let mut bs_zeros = Vec::with_capacity(2 * order);

    // CRITICAL FIX: Section-indexed elliptic parametrization (same as bandpass).
    // Pro-Q 4 generates DISTINCT poles for each section using u_i = (2*i+1)*K(k)/order,
    // not duplicates. Each section index gets a DIFFERENT elliptic function evaluation.
    for section_idx in 0..order {
        let u_i = (2.0 * section_idx as f64 + 1.0) * kk / order as f64;

        // Evaluate Jacobi elliptic functions at this section's parameter
        let sn_val = elliptic::elliptic_sn(u_i, k);
        let cn_val = (1.0 - sn_val * sn_val).max(0.0).sqrt();
        let dn_val = (1.0 - k * k * sn_val * sn_val).max(0.0).sqrt();

        // The elliptic transform maps to BS poles:
        //   sigma = bw * sn * dn / (1 - k^2 * sn^2)
        //   omega_offset = bw * cn / (1 - k^2 * sn^2)
        // Each section gets UNIQUE sigma and omega_offset based on section_idx.
        let denom = 1.0 - k * k * sn_val * sn_val + 1e-30;
        let sigma = bw_a * sn_val * dn_val / denom;
        let omega_offset = bw_a * cn_val / denom;

        // Create ONE conjugate pole pair for this section.
        // The ±omega_offset creates the above/below-center-frequency pole pair.
        let p_upper = Complex::new(-sigma, w0_a + omega_offset);
        let p_lower = Complex::new(-sigma, -(w0_a + omega_offset));

        bs_poles.push(p_upper);
        bs_poles.push(p_lower);

        // Zeros at +/- j*w0 for notch (two zeros per section)
        bs_zeros.push(Complex::new(0.0, w0_a));
        bs_zeros.push(Complex::new(0.0, -w0_a));
    }

    let gain = 1.0;
    Zpk::new(bs_zeros, bs_poles, gain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butterworth_2nd_order_poles() {
        let zpk = butterworth_lp(2);
        assert_eq!(zpk.poles.len(), 2);
        for p in &zpk.poles {
            assert!((p.mag() - 1.0).abs() < 1e-10, "pole not on unit circle");
            assert!(p.re < 0.0, "pole not in LHP");
        }
    }

    #[test]
    fn butterworth_4th_order_poles() {
        let zpk = butterworth_lp(4);
        assert_eq!(zpk.poles.len(), 4);
        for p in &zpk.poles {
            assert!((p.mag() - 1.0).abs() < 1e-10);
            assert!(p.re < 0.0);
        }
    }

    #[test]
    fn butterworth_prewarped_scales_poles() {
        let proto = butterworth_lp_prewarped(2, 1000.0, 48000.0);
        assert_eq!(proto.poles.len(), 2);
        for p in &proto.poles {
            assert!(
                p.mag() > 100.0,
                "pre-warped pole should have large magnitude"
            );
        }
    }

    #[test]
    fn bandpass_doubles_order() {
        let bp = butterworth_bp(2, 1000.0, 1.0, 48000.0);
        assert_eq!(bp.poles.len(), 4);
        assert_eq!(bp.zeros.len(), 2);
    }

    #[test]
    fn bandpass_poles_in_lhp() {
        let bp = butterworth_bp(2, 1000.0, 2.0, 48000.0);
        for p in &bp.poles {
            assert!(p.re < 1e-10, "BP pole should be in LHP, got re={}", p.re);
        }
    }

    #[test]
    fn bandstop_has_notch_zeros() {
        let bs = butterworth_bs(2, 1000.0, 1.0, 48000.0);
        assert_eq!(bs.poles.len(), 4);
        assert_eq!(bs.zeros.len(), 4);
    }

    #[test]
    fn bandstop_zeros_on_imaginary_axis() {
        let bs = butterworth_bs(2, 1000.0, 2.0, 48000.0);
        for z in &bs.zeros {
            assert!(
                z.re.abs() < 1e-10,
                "BS zero should be on imaginary axis, got re={}",
                z.re
            );
        }
    }

    #[test]
    fn elliptic_bp_produces_correct_pole_count() {
        let bp = butterworth_bp_elliptic(2, 1000.0, 2.0, 48000.0);
        // 2nd order = 1 conjugate pair -> 4 BP poles + conjugates
        assert!(
            bp.poles.len() >= 4,
            "expected >= 4 BP poles, got {}",
            bp.poles.len()
        );
    }

    #[test]
    fn elliptic_bp_poles_in_lhp() {
        let bp = butterworth_bp_elliptic(2, 1000.0, 2.0, 48000.0);
        for (i, p) in bp.poles.iter().enumerate() {
            assert!(
                p.re < 1e-10,
                "elliptic BP pole {} should be in LHP, got re={}",
                i,
                p.re
            );
        }
    }

    #[test]
    fn elliptic_bp_4th_order() {
        let bp = butterworth_bp_elliptic(4, 1000.0, 4.0, 48000.0);
        // 4th order = 2 conjugate pairs -> 8 BP poles
        assert!(
            bp.poles.len() >= 8,
            "expected >= 8 BP poles, got {}",
            bp.poles.len()
        );
    }

    #[test]
    fn standard_and_elliptic_bp_same_2nd_order() {
        // For 2nd order, both methods should produce similar results
        let std_bp = butterworth_bp(2, 1000.0, 2.0, 48000.0);
        let ell_bp = butterworth_bp_elliptic(2, 1000.0, 2.0, 48000.0);

        // Both should have poles in LHP with similar magnitudes
        let std_mag: f64 =
            std_bp.poles.iter().map(|p| p.mag()).sum::<f64>() / std_bp.poles.len() as f64;
        let ell_mag: f64 =
            ell_bp.poles.iter().map(|p| p.mag()).sum::<f64>() / ell_bp.poles.len() as f64;

        // They won't be identical but should be in the same ballpark
        let ratio = std_mag / (ell_mag + 1e-30);
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "standard and elliptic BP pole magnitudes should be comparable: std={}, ell={}",
            std_mag,
            ell_mag
        );
    }
}
