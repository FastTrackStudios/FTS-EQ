//! ZPK to biquad coefficient conversion matching Pro-Q 4's
//! `zpk_to_biquad_coefficients` (0x1800fe040).

use crate::zpk::{pair_conjugates, Complex, Zpk};

/// Biquad coefficients: [a0, a1, a2, b0, b1, b2].
pub type Coeffs = [f64; 6];

/// Identity / passthrough biquad.
pub const PASSTHROUGH: Coeffs = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// Convert a ZPK representation to second-order sections (biquad coefficients).
///
/// Each section is a `Coeffs` array [a0, a1, a2, b0, b1, b2] where the
/// transfer function is H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2).
pub fn zpk_to_sos(zpk: &Zpk) -> Vec<Coeffs> {
    let sections = pair_conjugates(zpk);
    let mut sos = Vec::with_capacity(sections.len());

    for (poles, zeros, gain) in sections {
        let den = poles_to_den(&poles);
        let num = zeros_to_num(&zeros, gain);
        sos.push([den[0], den[1], den[2], num[0], num[1], num[2]]);
    }

    if sos.is_empty() {
        sos.push([1.0, 0.0, 0.0, zpk.gain, 0.0, 0.0]);
    }

    sos
}

/// Evaluate a cascade of second-order sections at normalized frequency w.
///
/// w is in radians: w = 2*pi*f/fs.
pub fn eval_sos(sections: &[Coeffs], w: f64) -> Complex {
    let ejw = Complex::from_polar(1.0, w);
    let ejw2 = ejw * ejw;

    let mut result = Complex::ONE;
    for s in sections {
        let den = Complex::new(s[0], 0.0) + ejw * s[1] + ejw2 * s[2];
        let num = Complex::new(s[3], 0.0) + ejw * s[4] + ejw2 * s[5];
        result = result * (num / den);
    }
    result
}

/// Magnitude in dB of a cascade of second-order sections at frequency w.
pub fn mag_db_sos(sections: &[Coeffs], w: f64) -> f64 {
    20.0 * eval_sos(sections, w).mag().log10()
}

// ─── Internal Helpers ─────────────────────────────────────────────────────────

/// Convert up to 2 poles into denominator coefficients [a0, a1, a2].
fn poles_to_den(poles: &[Complex]) -> [f64; 3] {
    match poles.len() {
        0 => [1.0, 0.0, 0.0],
        1 => {
            let p = poles[0];
            // (1 - p*z^-1) expanded: real coefficients only for real or paired poles.
            [1.0, -p.re, 0.0]
        }
        _ => {
            // Two poles: (1 - p0*z^-1)(1 - p1*z^-1)
            let p0 = poles[0];
            let p1 = poles[1];
            let sum = p0 + p1;
            let prod = p0 * p1;
            [1.0, -sum.re, prod.re]
        }
    }
}

/// Convert up to 2 zeros into numerator coefficients [b0, b1, b2], scaled by gain.
fn zeros_to_num(zeros: &[Complex], gain: f64) -> [f64; 3] {
    match zeros.len() {
        0 => [gain, 0.0, 0.0],
        1 => {
            let z = zeros[0];
            [gain, -gain * z.re, 0.0]
        }
        _ => {
            let z0 = zeros[0];
            let z1 = zeros[1];
            let sum = z0 + z1;
            let prod = z0 * z1;
            [gain, -gain * sum.re, gain * prod.re]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn passthrough_is_unity() {
        let mag = mag_db_sos(&[PASSTHROUGH], PI / 4.0);
        assert!(
            mag.abs() < 1e-10,
            "Passthrough should be 0 dB, got {:.6}",
            mag
        );
    }

    #[test]
    fn zpk_to_sos_empty_gives_passthrough() {
        let zpk = Zpk::new(vec![], vec![], 1.0);
        let sos = zpk_to_sos(&zpk);
        assert_eq!(sos.len(), 1);
        // With gain=1.0 and no poles/zeros, should act as passthrough.
        let mag = mag_db_sos(&sos, std::f64::consts::PI / 4.0);
        assert!(mag.abs() < 1e-10, "Expected 0 dB, got {:.6}", mag);
    }

    #[test]
    fn zpk_to_sos_preserves_pole_count() {
        let zpk = Zpk::new(
            vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)],
            vec![Complex::new(0.5, 0.3), Complex::new(0.5, -0.3)],
            2.0,
        );
        let sos = zpk_to_sos(&zpk);
        // 2 poles = 1 second-order section.
        assert_eq!(sos.len(), 1);
    }

    #[test]
    fn zpk_to_sos_four_poles() {
        let zpk = Zpk::new(
            vec![
                Complex::new(-1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(0.0, -1.0),
            ],
            vec![
                Complex::new(0.5, 0.3),
                Complex::new(0.5, -0.3),
                Complex::new(0.3, 0.7),
                Complex::new(0.3, -0.7),
            ],
            1.0,
        );
        let sos = zpk_to_sos(&zpk);
        assert_eq!(sos.len(), 2);
    }

    #[test]
    fn eval_sos_matches_zpk_eval() {
        let zpk = Zpk::new(
            vec![Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0)],
            vec![Complex::new(0.5, 0.3), Complex::new(0.5, -0.3)],
            2.0,
        );
        let sos = zpk_to_sos(&zpk);

        // Compare at several frequencies.
        for k in 1..8 {
            let w = PI * k as f64 / 8.0;
            let from_zpk = zpk.eval_z(w).mag();
            let from_sos = eval_sos(&sos, w).mag();
            let diff = (from_zpk - from_sos).abs();
            assert!(
                diff < 1e-8,
                "Mismatch at w={:.3}: zpk={:.6}, sos={:.6}",
                w,
                from_zpk,
                from_sos
            );
        }
    }

    #[test]
    fn gain_only_filter() {
        let zpk = Zpk::new(vec![], vec![], 4.0);
        let sos = zpk_to_sos(&zpk);
        let mag = mag_db_sos(&sos, PI / 4.0);
        let expected = 20.0 * 4.0_f64.log10();
        assert!(
            (mag - expected).abs() < 1e-8,
            "Expected {:.4} dB, got {:.4} dB",
            expected,
            mag
        );
    }
}
