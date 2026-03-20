//! Vicanek matched filter coefficient calculation.
//!
//! Based on Martin Vicanek, "Matched Second Order Digital Filters" (2016)
//! and the ZLEqualizer implementation by ZL Audio.
//!
//! Unlike the standard Audio EQ Cookbook (bilinear transform), this uses
//! impulse-invariance for poles and frequency-domain matching for zeros,
//! giving no frequency cramping near Nyquist.

use std::f64::consts::PI;

use crate::filter_type::FilterType;

/// Raw biquad coefficients: [a0, a1, a2, b0, b1, b2].
/// Convention: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
pub type Coeffs = [f64; 6];

/// Passthrough (identity) coefficients.
pub const PASSTHROUGH: Coeffs = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

// r[impl dsp.biquad.coefficients]
/// Calculate matched filter coefficients for the given parameters.
///
/// - `filter_type`: which filter shape
/// - `freq_hz`: center/corner frequency in Hz
/// - `q`: resonance (0.707 = Butterworth)
/// - `gain_db`: boost/cut in dB (ignored for pass/notch types)
/// - `sample_rate`: sample rate in Hz
///
/// Returns `[a0, a1, a2, b0, b1, b2]`.
pub fn calculate(
    filter_type: FilterType,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
) -> Coeffs {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let g = 10.0_f64.powf(gain_db / 20.0); // linear gain

    // Clamp w0 to avoid instability at extremes
    let w0 = w0.clamp(1e-6, PI - 1e-6);

    match filter_type {
        FilterType::Peak => peak_2(w0, q, g),
        FilterType::LowShelf => low_shelf_2(w0, q, g),
        FilterType::HighShelf => high_shelf_2(w0, q, g),
        FilterType::TiltShelf => tilt_shelf_2(w0, q, g),
        FilterType::Lowpass => lowpass_2(w0, q),
        FilterType::Highpass => highpass_2(w0, q),
        FilterType::Bandpass => bandpass_2(w0, q),
        FilterType::Notch => notch_2(w0, q),
        FilterType::BandShelf => {
            // Band shelf is handled at the band level by cascading shelves
            PASSTHROUGH
        }
    }
}

/// Solve for poles using impulse invariance.
/// `b` = damping = 0.5/Q, `c` = 1.0 (standard) or sqrt(sqrt(g)) for shelves.
/// Returns (a1, a2).
fn solve_poles(w0: f64, b: f64, c: f64) -> (f64, f64) {
    let t = (-b * w0).exp();
    let a1 = if b <= c {
        -2.0 * t * ((c * c - b * b).sqrt() * w0).cos()
    } else {
        -2.0 * t * ((b * b - c * c).sqrt() * w0).cosh()
    };
    let a2 = t * t;
    (a1, a2)
}

/// Phi basis functions for magnitude-squared matching.
#[inline]
fn phi0(w: f64) -> f64 {
    0.5 + 0.5 * w.cos()
}

#[inline]
fn phi1(w: f64) -> f64 {
    0.5 - 0.5 * w.cos()
}

/// Convert magnitude-squared B coefficients to b coefficients (minimum phase).
/// `big_b = [B0, B1, B2]` → returns `(b0, b1, b2)`.
fn mag_sq_to_b(big_b: [f64; 3]) -> (f64, f64, f64) {
    let b0_sq = big_b[0].max(0.0);
    let b1_sq = big_b[1].max(0.0);
    let b0_sqrt = b0_sq.sqrt();
    let b1_sqrt = b1_sq.sqrt();
    let w = (b0_sqrt + b1_sqrt) / 2.0;

    if big_b[2].abs() < 1e-30 {
        // b2 = 0 case (lowpass)
        let b0 = w;
        let b1 = b0_sqrt - b0;
        return (b0, b1, 0.0);
    }

    let b0 = (w + (w * w + big_b[2]).max(0.0).sqrt()) / 2.0;
    let b0 = b0.max(1e-30);
    let b1 = (b0_sqrt - b1_sqrt) / 2.0;
    let b2 = -big_b[2] / (4.0 * b0);
    (b0, b1, b2)
}

/// 2nd-order matched lowpass.
fn lowpass_2(w0: f64, q: f64) -> Coeffs {
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    // Magnitude-squared denominator coefficients
    let a0_big = (1.0 + a1 + a2) * (1.0 + a1 + a2);
    let a1_big = (1.0 - a1 + a2) * (1.0 - a1 + a2);
    let a2_big = -4.0 * a2;

    // Match at DC (unity) and at w0 (gain = Q)
    let p0 = phi0(w0);
    let p1 = phi1(w0);
    let r1 = (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0) * q * q;

    let b0_big = a0_big; // Unity at DC
    let b1_big = (r1 - b0_big * p0) / p1;
    let (b0, b1, _) = mag_sq_to_b([b0_big, b1_big.max(0.0), 0.0]);

    [1.0, a1, a2, b0, b1, 0.0]
}

/// 2nd-order matched highpass.
fn highpass_2(w0: f64, q: f64) -> Coeffs {
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    let a0_big = (1.0 + a1 + a2) * (1.0 + a1 + a2);
    let a1_big = (1.0 - a1 + a2) * (1.0 - a1 + a2);
    let a2_big = -4.0 * a2;

    // Double zero at DC, match at w0 (gain = Q)
    let p0 = phi0(w0);
    let p1 = phi1(w0);
    let r1 = a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0;
    let b0 = q * r1.sqrt() / (4.0 * p1);
    let b1 = -2.0 * b0;
    let b2 = b0;

    [1.0, a1, a2, b0, b1, b2]
}

/// 2nd-order matched bandpass.
fn bandpass_2(w0: f64, q: f64) -> Coeffs {
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    let a0_big = (1.0 + a1 + a2) * (1.0 + a1 + a2);
    let a1_big = (1.0 - a1 + a2) * (1.0 - a1 + a2);
    let a2_big = -4.0 * a2;

    let p0 = phi0(w0);
    let p1 = phi1(w0);

    // Zero at DC (B0=0), unity at w0
    let r1 = a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0;
    let r2 = -a0_big + a1_big + 4.0 * (p0 - p1) * a2_big;

    let b2_big = (r1 - r2 * p1) / (4.0 * p1 * p1);
    let b1_big = r2 + 4.0 * (p1 - p0) * b2_big;

    let (b0, b1, b2) = mag_sq_to_b([0.0, b1_big.max(0.0), b2_big]);

    [1.0, a1, a2, b0, b1, b2]
}

/// 2nd-order matched notch.
fn notch_2(w0: f64, q: f64) -> Coeffs {
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    // Zeros placed exactly at w0
    let b0 = 1.0;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0;

    // Normalize for unity at DC
    let dc_num = b0 + b1 + b2;
    let dc_den = 1.0 + a1 + a2;
    let scale = dc_den / dc_num;

    [1.0, a1, a2, b0 * scale, b1 * scale, b2 * scale]
}

/// 2nd-order matched peak EQ.
fn peak_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }

    // Poles use modified damping that depends on gain
    let pole_q = (g.sqrt() * q).max(0.01);
    let (a1, a2) = solve_poles(w0, 0.5 / pole_q, 1.0);

    let a0_big = (1.0 + a1 + a2) * (1.0 + a1 + a2);
    let a1_big = (1.0 - a1 + a2) * (1.0 - a1 + a2);
    let a2_big = -4.0 * a2;

    let p0 = phi0(w0);
    let p1 = phi1(w0);

    let g_sq = g * g;

    // Unity at DC, match G^2 at w0
    let r1 = (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0) * g_sq;
    let r2 = (-a0_big + a1_big + 4.0 * (p0 - p1) * a2_big) * g_sq;

    let b0_big = a0_big;
    let b2_big = (r1 - r2 * p1 - b0_big) / (4.0 * p1 * p1);
    let b1_big = r2 + b0_big + 4.0 * (p1 - p0) * b2_big;

    let (b0, b1, b2) = mag_sq_to_b([b0_big, b1_big.max(0.0), b2_big]);

    [1.0, a1, a2, b0, b1, b2]
}

/// 2nd-order matched tilt shelf.
fn tilt_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }

    let g_work;
    let invert;

    // For g > 1, we compute the inverse and swap a/b at the end
    if g > 1.0 {
        g_work = 1.0 / g;
        invert = true;
    } else {
        g_work = g;
        invert = false;
    }

    let g4 = g_work.sqrt().sqrt(); // g^(1/4)
    let (a1, a2) = solve_poles(w0, g4 / (2.0 * q), g4);

    let a0_big = (1.0 + a1 + a2) * (1.0 + a1 + a2);
    let a1_big = (1.0 - a1 + a2) * (1.0 - a1 + a2);
    let a2_big = -4.0 * a2;

    let p0 = phi0(w0);
    let p1 = phi1(w0);

    // Match at DC: gain = sqrt(g), at Nyquist: gain = 1/sqrt(g), at w0: gain = 1
    let dc_gain_sq = g_work;
    let ny_gain_sq = 1.0 / g_work;

    let b0_big = a0_big * dc_gain_sq;
    let b1_big = a1_big * ny_gain_sq;

    // Solve for B2 using match at w0
    let target = a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0; // gain = 1 at w0
    let b2_big = (target - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);

    if invert {
        // Swap numerator and denominator
        [b0, b1, b2, 1.0, a1, a2]
    } else {
        [1.0, a1, a2, b0, b1, b2]
    }
}

/// 2nd-order matched low shelf = tilt shelf with gain, then scale numerator.
///
/// DC gain = sqrt(g) * sqrt(g) = g. Nyquist gain = (1/sqrt(g)) * sqrt(g) = 1.
fn low_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    let mut c = tilt_shelf_2(w0, q, g);
    let scale = g.sqrt();
    c[3] *= scale;
    c[4] *= scale;
    c[5] *= scale;
    c
}

/// 2nd-order matched high shelf = tilt shelf with inverted gain, then scale numerator.
///
/// DC gain = sqrt(1/g) * sqrt(g) = 1. Nyquist gain = sqrt(g) * sqrt(g) = g.
fn high_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    let mut c = tilt_shelf_2(w0, q, 1.0 / g);
    let scale = g.sqrt();
    c[3] *= scale;
    c[4] *= scale;
    c[5] *= scale;
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    // r[verify dsp.biquad.coefficients]
    #[test]
    fn peak_unity_gain_is_passthrough() {
        let c = calculate(FilterType::Peak, 1000.0, 0.707, 0.0, 44100.0);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 1.0).abs() < 1e-10);
        assert!(c[1].abs() < 1e-10);
        assert!(c[4].abs() < 1e-10);
    }

    // r[verify dsp.biquad.coefficients]
    #[test]
    fn lowpass_has_unity_dc_gain() {
        let c = lowpass_2(2.0 * PI * 1000.0 / 44100.0, 0.707);
        // DC gain = (b0+b1+b2)/(a0+a1+a2) should be ~1.0
        let dc = (c[3] + c[4] + c[5]) / (c[0] + c[1] + c[2]);
        assert!((dc - 1.0).abs() < 0.01, "DC gain = {dc}, expected ~1.0");
    }

    // r[verify dsp.biquad.coefficients]
    #[test]
    fn highpass_has_zero_dc_gain() {
        let c = highpass_2(2.0 * PI * 1000.0 / 44100.0, 0.707);
        let dc = (c[3] + c[4] + c[5]) / (c[0] + c[1] + c[2]);
        assert!(dc.abs() < 0.01, "DC gain = {dc}, expected ~0.0");
    }

    // r[verify dsp.biquad.coefficients]
    #[test]
    fn coefficients_are_finite() {
        let types = [
            FilterType::Peak,
            FilterType::Lowpass,
            FilterType::Highpass,
            FilterType::Bandpass,
            FilterType::Notch,
            FilterType::LowShelf,
            FilterType::HighShelf,
            FilterType::TiltShelf,
        ];
        for ft in types {
            let c = calculate(ft, 1000.0, 0.707, 6.0, 44100.0);
            for (i, v) in c.iter().enumerate() {
                assert!(v.is_finite(), "{ft:?} coeff[{i}] = {v} is not finite");
            }
        }
    }
}
