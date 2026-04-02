//! Frequency domain transforms matching Pro-Q 4's transform type table at 0x18020eff0.
//!
//! Transform types:
//!   0 = Direct (no transform) — LP/HP
//!   1 = LP→BP frequency warp — BP/Notch
//!   2 = Bilinear transform — Shelves
//!   3 = LP→BP + bilinear — Band Shelf
//!   4 = Negate zeros — Allpass

use crate::zpk::{Complex, Zpk};

/// Bilinear s→z transform (transform type 2).
///
/// Maps analog prototype poles/zeros from the s-plane to the z-plane using:
///   z = (1 + s/2fs) / (1 - s/2fs)
///
/// Unmapped zeros are placed at z = -1 (Nyquist) to preserve filter order.
pub fn bilinear(zpk: &Zpk, sample_rate: f64) -> Zpk {
    let fs2 = 2.0 * sample_rate;

    let mut z_zeros: Vec<Complex> = zpk
        .zeros
        .iter()
        .map(|&z| {
            let num = Complex::ONE + z / fs2;
            let den = Complex::ONE - z / fs2;
            num / den
        })
        .collect();

    let z_poles: Vec<Complex> = zpk
        .poles
        .iter()
        .map(|&p| {
            let num = Complex::ONE + p / fs2;
            let den = Complex::ONE - p / fs2;
            num / den
        })
        .collect();

    // Pad zeros to match pole count — extra zeros go to Nyquist (z = -1).
    while z_zeros.len() < z_poles.len() {
        z_zeros.push(Complex::new(-1.0, 0.0));
    }

    // Adjust gain: product of (fs2 - z) / product of (fs2 - p).
    let mut gain = Complex::new(zpk.gain, 0.0);
    for &z in &zpk.zeros {
        gain = gain * (Complex::new(fs2, 0.0) - z);
    }
    for &p in &zpk.poles {
        gain = gain / (Complex::new(fs2, 0.0) - p);
    }
    // Account for added Nyquist zeros.
    let extra = z_poles.len().saturating_sub(zpk.zeros.len());
    for _ in 0..extra {
        gain = gain * Complex::new(fs2, 0.0);
    }

    Zpk::new(z_zeros, z_poles, gain.re)
}

/// Allpass transform (transform type 4): negate zeros across the unit circle.
///
/// For each pole p, creates a zero at 1/conj(p), making |H(e^jw)| = 1
/// for all frequencies while preserving the phase response of the poles.
pub fn make_allpass(zpk: &Zpk) -> Zpk {
    let zeros: Vec<Complex> = zpk
        .poles
        .iter()
        .map(|&p| {
            // Reflect pole across unit circle: z = 1/conj(p)
            p.conj().inv()
        })
        .collect();

    // Gain adjustment to normalize: product |p_i| for stable poles.
    let mut gain = zpk.gain;
    for &p in &zpk.poles {
        let m = p.mag();
        if m > 1e-15 {
            gain *= m;
        }
    }

    Zpk::new(zeros, zpk.poles.clone(), gain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn bilinear_preserves_pole_count() {
        let zpk = Zpk::new(
            vec![Complex::new(-1.0, 0.0)],
            vec![Complex::new(-0.5, 0.5), Complex::new(-0.5, -0.5)],
            1.0,
        );
        let z = bilinear(&zpk, 44100.0);
        assert_eq!(z.poles.len(), 2);
        // Zeros padded to match pole count.
        assert_eq!(z.zeros.len(), 2);
    }

    #[test]
    fn bilinear_poles_inside_unit_circle() {
        // Stable analog poles (negative real part) must map to |z| < 1.
        let zpk = Zpk::new(
            vec![],
            vec![
                Complex::new(-1000.0, 2000.0),
                Complex::new(-1000.0, -2000.0),
            ],
            1.0,
        );
        let z = bilinear(&zpk, 44100.0);
        for p in &z.poles {
            assert!(
                p.mag() < 1.0,
                "Pole {:?} has magnitude {} >= 1.0",
                p,
                p.mag()
            );
        }
    }

    #[test]
    fn bilinear_dc_maps_to_z1() {
        // s = 0 should map to z = 1.
        let zpk = Zpk::new(vec![Complex::ZERO], vec![Complex::new(-1.0, 0.0)], 1.0);
        let z = bilinear(&zpk, 44100.0);
        let z0 = z.zeros[0];
        assert!(
            (z0.re - 1.0).abs() < 1e-10 && z0.im.abs() < 1e-10,
            "s=0 should map to z=1, got {:?}",
            z0
        );
    }

    #[test]
    fn allpass_zero_count_matches_poles() {
        let zpk = Zpk::new(
            vec![Complex::new(0.5, 0.0)],
            vec![Complex::new(0.5, 0.3), Complex::new(0.5, -0.3)],
            1.0,
        );
        let ap = make_allpass(&zpk);
        assert_eq!(ap.zeros.len(), ap.poles.len());
    }

    #[test]
    fn allpass_unit_magnitude() {
        // A simple allpass from a single real pole inside the unit circle.
        let zpk = Zpk::new(vec![], vec![Complex::new(0.5, 0.0)], 1.0);
        let ap = make_allpass(&zpk);

        // Check that |H(e^jw)| ≈ 1 at several frequencies.
        for k in 0..8 {
            let w = PI * k as f64 / 8.0;
            let mag = ap.eval_z(w).mag();
            assert!(
                (mag - 1.0).abs() < 0.05,
                "Allpass magnitude at w={:.3} is {:.4}, expected ~1.0",
                w,
                mag
            );
        }
    }
}
