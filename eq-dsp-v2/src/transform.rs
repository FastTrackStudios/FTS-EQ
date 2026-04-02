//! Frequency domain transforms: s-domain → z-domain.
//!
//! Pro-Q 4's transform pipeline (from design_filter_zpk_and_transform):
//!   Transform type 0: Direct (LP/HP) — bilinear s→z
//!   Transform type 1: LP→BP warp (BP/Notch) — already in s-domain, bilinear s→z
//!   Transform type 2: Bilinear (Shelves) — bilinear with gain manipulation
//!   Transform type 3: LP→BP + bilinear (Band Shelf)
//!   Transform type 4: Negate zeros (Allpass)

use crate::zpk::{Complex, Zpk};

/// Bilinear transform: map s-domain ZPK to z-domain ZPK.
///
/// Uses the substitution s = 2·fs·(z-1)/(z+1).
///
/// Each s-domain pole s_k maps to z-domain pole:
///   z_k = (1 + s_k/(2·fs)) / (1 - s_k/(2·fs))
///
/// Zeros at infinity map to z = -1 (Nyquist).
///
/// This is Pro-Q 4's `bilinear_transform_zpk` (0x1800fc550).
pub fn bilinear(zpk: &Zpk, sample_rate: f64) -> Zpk {
    let fs2 = 2.0 * sample_rate;

    let mut z_poles = Vec::with_capacity(zpk.poles.len());
    let mut z_zeros = Vec::with_capacity(zpk.poles.len());

    // Transform poles: s_k → z_k = (1 + s_k/2fs) / (1 - s_k/2fs)
    for &p in &zpk.poles {
        let num = Complex::ONE + p / fs2;
        let den = Complex::ONE - p / fs2;
        z_poles.push(num / den);
    }

    // Transform explicit zeros
    for &z in &zpk.zeros {
        let num = Complex::ONE + z / fs2;
        let den = Complex::ONE - z / fs2;
        z_zeros.push(num / den);
    }

    // Implicit zeros at s = ∞ map to z = -1
    let extra_zeros = zpk.poles.len() - zpk.zeros.len();
    for _ in 0..extra_zeros {
        z_zeros.push(Complex::new(-1.0, 0.0));
    }

    // Compute gain by evaluating at DC: H_analog(s=0) should equal H_digital(z=1).
    // H_analog(0) = gain * prod(-z_k) / prod(-p_k)
    let mut h_analog_dc = Complex::new(zpk.gain, 0.0);
    for &z in &zpk.zeros {
        h_analog_dc = h_analog_dc * (Complex::ZERO - z);
    }
    for &p in &zpk.poles {
        h_analog_dc = h_analog_dc / (Complex::ZERO - p);
    }

    // H_digital(z=1) without gain = prod(1 - z_k) / prod(1 - p_k)
    let z_one = Complex::ONE;
    let mut h_digital_dc = Complex::ONE;
    for &z in &z_zeros {
        h_digital_dc = h_digital_dc * (z_one - z);
    }
    for &p in &z_poles {
        h_digital_dc = h_digital_dc / (z_one - p);
    }

    // gain = H_analog(0) / H_digital_no_gain(1)
    let gain = if h_digital_dc.mag() > 1e-30 {
        (h_analog_dc / h_digital_dc).re
    } else {
        zpk.gain
    };

    Zpk::new(z_zeros, z_poles, gain)
}

/// Create allpass from existing ZPK: negate (reflect) zeros.
///
/// Pro-Q 4 transform type 4: for each zero z, replace with 1/z*.
/// This creates an allpass filter with the same pole structure.
pub fn make_allpass(zpk: &Zpk) -> Zpk {
    let mut new_zeros = Vec::with_capacity(zpk.poles.len());
    // For allpass: zeros are reflections of poles across unit circle
    for &p in &zpk.poles {
        new_zeros.push(Complex::ONE / p.conj());
    }
    Zpk::new(new_zeros, zpk.poles.clone(), zpk.gain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prototype;

    #[test]
    fn bilinear_preserves_pole_count() {
        let proto = prototype::butterworth_lp_prewarped(4, 1000.0, 48000.0);
        let digital = bilinear(&proto, 48000.0);
        assert_eq!(digital.poles.len(), 4);
        assert_eq!(digital.zeros.len(), 4);
    }

    #[test]
    fn bilinear_poles_inside_unit_circle() {
        let proto = prototype::butterworth_lp_prewarped(4, 1000.0, 48000.0);
        let digital = bilinear(&proto, 48000.0);
        for p in &digital.poles {
            assert!(p.mag() < 1.0 + 1e-10, "pole outside unit circle: {:?}", p);
        }
    }

    #[test]
    fn bilinear_lp_dc_gain_unity() {
        let proto = prototype::butterworth_lp_prewarped(2, 1000.0, 48000.0);
        let digital = bilinear(&proto, 48000.0);
        // Evaluate at DC (z = 1, w = 0)
        let h = digital.eval_z(0.0);
        let mag = h.mag();
        assert!(
            (mag - 1.0).abs() < 0.01,
            "DC gain should be ~1.0, got {mag}"
        );
    }
}
