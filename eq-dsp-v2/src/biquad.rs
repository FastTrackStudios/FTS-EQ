//! ZPK → biquad coefficient conversion.
//!
//! Pro-Q 4's `zpk_to_biquad_coefficients` (0x1800fe040) converts the internal
//! ZPK representation (20 doubles per section) into standard biquad [a0,a1,a2,b0,b1,b2].
//!
//! Each second-order section has a conjugate pole pair and a conjugate zero pair:
//!   H(z) = gain * (z - z0)(z - z0*) / ((z - p0)(z - p0*))
//!        = gain * (1 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)

use crate::zpk::{Complex, Zpk, pair_conjugates};

/// Standard biquad coefficients: [a0, a1, a2, b0, b1, b2].
/// Convention: H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
pub type Coeffs = [f64; 6];

pub const PASSTHROUGH: Coeffs = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Convert a z-domain ZPK to a vector of second-order section biquad coefficients.
///
/// Each section is a conjugate pole/zero pair converted to real biquad coefficients.
pub fn zpk_to_sos(zpk: &Zpk) -> Vec<Coeffs> {
    let sections = pair_conjugates(zpk);
    let mut result = Vec::with_capacity(sections.len());

    for (poles, zeros, gain) in sections {
        let (a1, a2) = poles_to_den(&poles);
        let (b0, b1, b2) = zeros_to_num(&zeros, gain);
        result.push([1.0, a1, a2, b0, b1, b2]);
    }

    result
}

/// Convert a conjugate pole pair to denominator coefficients.
///
/// For conjugate pair p, p*:
///   (z - p)(z - p*) = z² - 2·Re(p)·z + |p|²
///   In z⁻¹ form: 1 - 2·Re(p)·z⁻¹ + |p|²·z⁻²
fn poles_to_den(poles: &[Complex]) -> (f64, f64) {
    match poles.len() {
        0 => (0.0, 0.0),
        1 => {
            // Single real pole: (z - p) = 1 - p·z⁻¹
            debug_assert!(poles[0].im.abs() < 1e-10);
            (-poles[0].re, 0.0)
        }
        2 => {
            // Conjugate pair: 1 - 2·Re(p)·z⁻¹ + |p|²·z⁻²
            let re = poles[0].re;
            let mag_sq = poles[0].mag_sq();
            (-2.0 * re, mag_sq)
        }
        _ => panic!("poles_to_den: expected 0, 1, or 2 poles"),
    }
}

/// Convert a conjugate zero pair to numerator coefficients (with gain).
fn zeros_to_num(zeros: &[Complex], gain: f64) -> (f64, f64, f64) {
    match zeros.len() {
        0 => (gain, 0.0, 0.0),
        1 => {
            debug_assert!(zeros[0].im.abs() < 1e-10);
            (gain, -gain * zeros[0].re, 0.0)
        }
        2 => {
            let re = zeros[0].re;
            let mag_sq = zeros[0].mag_sq();
            (gain, -2.0 * gain * re, gain * mag_sq)
        }
        _ => panic!("zeros_to_num: expected 0, 1, or 2 zeros"),
    }
}

/// Evaluate a cascade of biquad sections at digital frequency w.
///
/// Returns complex H(e^{jw}).
pub fn eval_sos(sections: &[Coeffs], w: f64) -> Complex {
    let ejw = Complex::from_polar(1.0, w);
    let ejw2 = ejw * ejw;
    let mut h = Complex::ONE;
    for c in sections {
        let den = Complex::new(c[0], 0.0) + ejw * Complex::new(c[1], 0.0) + ejw2 * Complex::new(c[2], 0.0);
        let num = Complex::new(c[3], 0.0) + ejw * Complex::new(c[4], 0.0) + ejw2 * Complex::new(c[5], 0.0);
        h = h * num / den;
    }
    h
}

/// Evaluate magnitude in dB of a cascade of biquad sections.
pub fn mag_db_sos(sections: &[Coeffs], w: f64) -> f64 {
    20.0 * eval_sos(sections, w).mag().log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prototype;
    use crate::transform;
    use std::f64::consts::PI;

    #[test]
    fn butterworth_lp4_sos() {
        let proto = prototype::butterworth_lp_prewarped(4, 1000.0, 48000.0);
        let digital = transform::bilinear(&proto, 48000.0);
        let sos = zpk_to_sos(&digital);
        assert_eq!(sos.len(), 2, "4th order = 2 biquad sections");

        // DC gain should be ~1.0 (0 dB)
        let dc_db = mag_db_sos(&sos, 0.0);
        assert!(dc_db.abs() < 0.5, "DC gain = {dc_db} dB, expected ~0");

        // Well above cutoff should be attenuated
        let w_10k = 2.0 * PI * 10000.0 / 48000.0;
        let mag_10k = mag_db_sos(&sos, w_10k);
        assert!(mag_10k < -20.0, "10kHz should be attenuated, got {mag_10k} dB");
    }

    #[test]
    fn butterworth_lp2_corner_frequency() {
        let proto = prototype::butterworth_lp_prewarped(2, 1000.0, 48000.0);
        let digital = transform::bilinear(&proto, 48000.0);
        let sos = zpk_to_sos(&digital);

        // At corner frequency, Butterworth should be -3 dB
        let w_1k = 2.0 * PI * 1000.0 / 48000.0;
        let mag_1k = mag_db_sos(&sos, w_1k);
        assert!(
            (mag_1k - (-3.0)).abs() < 0.5,
            "corner frequency should be ~-3 dB, got {mag_1k}"
        );
    }

    #[test]
    fn bandpass_has_peak_at_center() {
        let bp = prototype::butterworth_bp(2, 1000.0, 2.0, 48000.0);
        let digital = transform::bilinear(&bp, 48000.0);
        let sos = zpk_to_sos(&digital);

        let w_1k = 2.0 * PI * 1000.0 / 48000.0;
        let mag_center = mag_db_sos(&sos, w_1k);
        let mag_dc = mag_db_sos(&sos, 0.001);
        let mag_nyq = mag_db_sos(&sos, PI - 0.001);

        assert!(mag_center > mag_dc + 10.0, "center should be louder than DC");
        assert!(mag_center > mag_nyq + 10.0, "center should be louder than Nyquist");
    }

    #[test]
    fn bandstop_has_notch_at_center() {
        let bs = prototype::butterworth_bs(2, 1000.0, 2.0, 48000.0);
        let digital = transform::bilinear(&bs, 48000.0);
        let sos = zpk_to_sos(&digital);

        let w_1k = 2.0 * PI * 1000.0 / 48000.0;
        let mag_center = mag_db_sos(&sos, w_1k);
        let mag_dc = mag_db_sos(&sos, 0.001);

        assert!(mag_center < -20.0, "center should be deeply attenuated, got {mag_center}");
        assert!(mag_dc.abs() < 1.0, "DC should be ~0 dB, got {mag_dc}");
    }
}
