//! Analog filter prototypes — Butterworth pole generation.
//!
//! Pro-Q 4 uses `maybe_butterworth_pole_next` (0x1801dbbf0) to generate
//! Butterworth poles in the s-domain. These are the starting point for
//! all filter types except Peak (type 0) and Shelf (type 12).
//!
//! For LP/HP (types 1,2): poles placed directly (transform type 0)
//! For BP/Notch (types 3,4): poles transformed via LP→BP (transform type 1)
//! For Shelves (types 7,8,9): poles transformed via bilinear (transform type 2)

use std::f64::consts::PI;

use crate::zpk::{Complex, Zpk};

/// Generate Butterworth lowpass prototype poles in the s-domain.
///
/// An N-th order Butterworth LP has N poles on the unit circle in the left half-plane:
///   s_k = exp(j * π * (2k + N + 1) / (2N))  for k = 0..N-1
///
/// All poles have |s_k| = 1 (unit circle), and the prototype has cutoff ω = 1.
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
/// frequency warping: ω_a = 2·fs·tan(ω_d / 2)
pub fn butterworth_lp_prewarped(order: usize, freq_hz: f64, sample_rate: f64) -> Zpk {
    let w_d = 2.0 * PI * freq_hz / sample_rate;
    let w_a = 2.0 * sample_rate * (w_d / 2.0).tan();

    let mut proto = butterworth_lp(order);
    // Scale poles by analog cutoff frequency
    for p in &mut proto.poles {
        *p = *p * w_a;
    }
    proto.gain = w_a.powi(order as i32);
    proto
}

/// Generate Butterworth bandpass prototype via LP→BP transformation.
///
/// LP→BP transform: s → Q_bp * (s/ω₀ + ω₀/s)
///
/// Each LP pole s_k maps to a PAIR of BP poles by solving:
///   Q_bp * (s/ω₀ + ω₀/s) = s_k
///   → s² - s·(s_k·ω₀/Q_bp) + ω₀² = 0
///
/// This is the core of Pro-Q 4's bandpass (type 3) using elliptic functions,
/// but the standard quadratic approach gives equivalent results for Butterworth.
///
/// Each LP pole also contributes a zero at s = 0 (DC).
pub fn butterworth_bp(order: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Zpk {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    // Pre-warp center frequency for bilinear transform
    let w0_a = 2.0 * sample_rate * (w0 / 2.0).tan();
    // Bandwidth from Q
    let bw_a = w0_a / q;

    let lp = butterworth_lp(order);

    let mut bp_poles = Vec::with_capacity(2 * order);
    let mut bp_zeros = Vec::with_capacity(order);

    for &s_k in &lp.poles {
        // LP→BP: solve s² - s·(s_k·bw_a) + w0_a² = 0
        // s = (s_k·bw_a ± sqrt((s_k·bw_a)² - 4·w0_a²)) / 2
        let b = s_k * bw_a;
        let disc = b * b - Complex::new(4.0 * w0_a * w0_a, 0.0);
        let sqrt_disc = complex_sqrt(disc);

        let s1 = (b + sqrt_disc) / 2.0;
        let s2 = (b - sqrt_disc) / 2.0;

        bp_poles.push(s1);
        bp_poles.push(s2);

        // Each LP pole adds a zero at origin
        bp_zeros.push(Complex::ZERO);
    }

    // Gain: normalize so |H(jω₀)| = 1
    let gain = bw_a.powi(order as i32);

    Zpk::new(bp_zeros, bp_poles, gain)
}

/// Generate Butterworth bandstop (notch) prototype via LP→BS transformation.
///
/// LP→BS transform: s → Q_bs * ω₀ / (s + ω₀²/s) = Q_bs * ω₀ * s / (s² + ω₀²)
///
/// Each LP pole maps to a pair of BS poles, and each also contributes
/// a pair of zeros at ±jω₀ (the notch frequencies).
pub fn butterworth_bs(order: usize, freq_hz: f64, q: f64, sample_rate: f64) -> Zpk {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let w0_a = 2.0 * sample_rate * (w0 / 2.0).tan();
    let bw_a = w0_a / q;

    let lp = butterworth_lp(order);

    let mut bs_poles = Vec::with_capacity(2 * order);
    let mut bs_zeros = Vec::with_capacity(2 * order);

    for &s_k in &lp.poles {
        // LP→BS: s → bw_a * s_k / (s²/w0_a + w0_a)... actually:
        // Reciprocal of LP→BP: replace s_k with 1/s_k in the BP formula,
        // then the zeros go to ±jω₀ instead of 0.
        //
        // BS pole equation: s² - s·(bw_a/s_k) + w0_a² = 0
        let b = Complex::new(bw_a, 0.0) / s_k;
        let disc = b * b - Complex::new(4.0 * w0_a * w0_a, 0.0);
        let sqrt_disc = complex_sqrt(disc);

        let s1 = (b + sqrt_disc) / 2.0;
        let s2 = (b - sqrt_disc) / 2.0;

        bs_poles.push(s1);
        bs_poles.push(s2);

        // Each LP pole contributes zeros at ±jω₀
        bs_zeros.push(Complex::new(0.0, w0_a));
        bs_zeros.push(Complex::new(0.0, -w0_a));
    }

    Zpk::new(bs_zeros, bs_poles, 1.0)
}

/// Complex square root.
fn complex_sqrt(z: Complex) -> Complex {
    let r = z.mag();
    let theta = z.arg();
    Complex::from_polar(r.sqrt(), theta / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butterworth_2nd_order_poles() {
        let zpk = butterworth_lp(2);
        assert_eq!(zpk.poles.len(), 2);
        // 2nd order Butterworth: poles at 135° and 225°
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
    fn bandpass_doubles_order() {
        let bp = butterworth_bp(2, 1000.0, 1.0, 48000.0);
        assert_eq!(bp.poles.len(), 4); // 2 LP poles → 4 BP poles
        assert_eq!(bp.zeros.len(), 2); // 2 zeros at origin
    }

    #[test]
    fn bandstop_has_notch_zeros() {
        let bs = butterworth_bs(2, 1000.0, 1.0, 48000.0);
        assert_eq!(bs.poles.len(), 4);
        assert_eq!(bs.zeros.len(), 4); // pairs at ±jω₀
    }
}
