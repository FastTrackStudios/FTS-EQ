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
        FilterType::Allpass => allpass_2(w0, q),
        FilterType::BandShelf | FilterType::FlatTilt => {
            // These are handled at the band level
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

/// 2nd-order lowpass (RBJ cookbook / bilinear transform).
///
/// Uses the standard bilinear transform which frequency-cramps near Nyquist,
/// matching Pro-Q 4's LP behavior.
fn lowpass_2(w0: f64, q: f64) -> Coeffs {
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * q);

    let b1 = 1.0 - cos_w0;
    let b0 = b1 / 2.0;
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    let a0_inv = 1.0 / a0;
    [
        1.0,
        a1 * a0_inv,
        a2 * a0_inv,
        b0 * a0_inv,
        b1 * a0_inv,
        b2 * a0_inv,
    ]
}

/// 2nd-order highpass (RBJ cookbook / bilinear transform).
///
/// Uses the standard bilinear transform which frequency-cramps near Nyquist,
/// matching Pro-Q 4's HP behavior.
fn highpass_2(w0: f64, q: f64) -> Coeffs {
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * q);

    let b1 = -(1.0 + cos_w0);
    let b0 = -b1 / 2.0;
    let b2 = b0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    let a0_inv = 1.0 / a0;
    [
        1.0,
        a1 * a0_inv,
        a2 * a0_inv,
        b0 * a0_inv,
        b1 * a0_inv,
        b2 * a0_inv,
    ]
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

/// 2nd-order allpass filter.
///
/// Unity magnitude at all frequencies, phase shift around w0.
/// Uses the same pole placement as the bandpass, with zeros mirrored.
fn allpass_2(w0: f64, q: f64) -> Coeffs {
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    // Allpass: numerator = reversed denominator
    // H(z) = (a2 + a1*z^-1 + z^-2) / (1 + a1*z^-1 + a2*z^-2)
    [1.0, a1, a2, a2, a1, 1.0]
}

/// 1st-order lowpass filter coefficients (bilinear transform).
///
/// True 1st-order (6 dB/oct) with frequency cramping matching Pro-Q 4.
pub fn lowpass_1(freq_hz: f64, sample_rate: f64) -> Coeffs {
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let wc = (w0 / 2.0).tan();
    let a0_inv = 1.0 / (1.0 + wc);
    let b0 = wc * a0_inv;
    let b1 = b0;
    let a1 = (wc - 1.0) * a0_inv;
    [1.0, a1, 0.0, b0, b1, 0.0]
}

/// 1st-order highpass filter coefficients (bilinear transform).
///
/// True 1st-order (6 dB/oct) with frequency cramping matching Pro-Q 4.
pub fn highpass_1(freq_hz: f64, sample_rate: f64) -> Coeffs {
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let wc = (w0 / 2.0).tan();
    let a0_inv = 1.0 / (1.0 + wc);
    let b0 = a0_inv;
    let b1 = -b0;
    let a1 = (wc - 1.0) * a0_inv;
    [1.0, a1, 0.0, b0, b1, 0.0]
}

/// 1st-order low shelf filter coefficients.
///
/// Pro-Q 4 convention: freq_hz is the half-gain-in-dB point (where gain = sqrt(G)).
/// For a 1st-order analog shelf, the half-gain point is at sqrt(G) * corner_freq,
/// so we shift the corner: corner = freq / sqrt(G).
pub fn low_shelf_1(freq_hz: f64, gain_db: f64, sample_rate: f64) -> Coeffs {
    let g = 10.0_f64.powf(gain_db / 20.0);

    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }

    // Shift corner frequency so that freq_hz is the half-gain-in-dB point.
    // Boost (g > 1): half-gain at sqrt(g)*corner → corner = freq/sqrt(g) (shift down)
    // Cut (g < 1): half-gain at corner/sqrt(g) → corner = freq*sqrt(g) (shift down)
    let corner_hz = if g > 1.0 {
        freq_hz / g.sqrt()
    } else {
        freq_hz * g.sqrt()
    };
    let w0 = (2.0 * PI * corner_hz / sample_rate).clamp(1e-6, PI - 1e-6);

    // Bilinear transform 1st-order shelf (prewarp with w0/2)
    let wc = (w0 / 2.0).tan();
    if g > 1.0 {
        // Boost
        let gwc = g * wc;
        let a0_inv = 1.0 / (1.0 + wc);
        let b0 = (1.0 + gwc) * a0_inv;
        let b1 = (-1.0 + gwc) * a0_inv;
        let a1 = (-1.0 + wc) * a0_inv;
        [1.0, a1, 0.0, b0, b1, 0.0]
    } else {
        // Cut
        let wc_g = wc / g;
        let a0_inv = 1.0 / (1.0 + wc_g);
        let b0 = (1.0 + wc) * a0_inv;
        let b1 = (-1.0 + wc) * a0_inv;
        let a1 = (-1.0 + wc_g) * a0_inv;
        [1.0, a1, 0.0, b0, b1, 0.0]
    }
}

/// 1st-order high shelf filter coefficients.
///
/// Pro-Q 4 convention: freq_hz is the half-gain-in-dB point.
/// Analog prototype: H(s) = (g*s + wc) / (s + wc)
/// DC gain = 1, Nyquist gain = g.
pub fn high_shelf_1(freq_hz: f64, gain_db: f64, sample_rate: f64) -> Coeffs {
    let g = 10.0_f64.powf(gain_db / 20.0);

    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }

    // Shift corner frequency so that freq_hz is the half-gain-in-dB point.
    // Half-gain at corner/sqrt(g), so corner = freq * sqrt(g) for boost,
    // corner = freq / sqrt(g) for cut. Both shift the corner AWAY from freq.
    let corner_hz = if g > 1.0 {
        freq_hz * g.sqrt()
    } else {
        freq_hz / g.sqrt()
    };
    let w0 = (2.0 * PI * corner_hz / sample_rate).clamp(1e-6, PI - 1e-6);

    // Bilinear transform 1st-order high shelf (prewarp with w0/2)
    let wc = (w0 / 2.0).tan();
    if g > 1.0 {
        // Boost: H(s) = (g*s + wc) / (s + wc)
        let a0_inv = 1.0 / (1.0 + wc);
        let b0 = (g + wc) * a0_inv;
        let b1 = (-g + wc) * a0_inv;
        let a1 = (-1.0 + wc) * a0_inv;
        [1.0, a1, 0.0, b0, b1, 0.0]
    } else {
        // Cut: H(s) = (s + wc) / ((1/g)*s + wc)
        let g_inv = 1.0 / g;
        let a0_inv = 1.0 / (g_inv + wc);
        let b0 = (1.0 + wc) * a0_inv;
        let b1 = (-1.0 + wc) * a0_inv;
        let a1 = (-g_inv + wc) * a0_inv;
        [1.0, a1, 0.0, b0, b1, 0.0]
    }
}

/// 1st-order allpass filter coefficients.
pub fn allpass_1(freq_hz: f64, sample_rate: f64) -> Coeffs {
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let p = (-w0).exp();
    // Allpass: H(z) = (-p + z^-1) / (1 - p*z^-1)
    [1.0, -p, 0.0, -p, 1.0, 0.0]
}

/// Compute alpha for shelving filters using the RBJ S (shelf slope) parameter.
///
/// `alpha = sin(w0)/2 * sqrt((A + 1/A) * (1/S - 1) + 2)`
///
/// When S = 1: Butterworth (maximally flat transition).
/// S < 1: gentler slope. S > 1: steeper, possibly resonant.
fn shelf_alpha(sin_w0: f64, a_amp: f64, s: f64) -> f64 {
    let s = s.max(0.01); // avoid division by zero
    let val = (a_amp + 1.0 / a_amp) * (1.0 / s - 1.0) + 2.0;
    sin_w0 / 2.0 * val.max(0.001).sqrt()
}

/// 2nd-order low shelf (RBJ cookbook / bilinear transform).
///
/// Uses the RBJ Audio EQ Cookbook shelf formulas with the S (shelf slope)
/// parameter. Pro-Q 4 convention: Q_internal = Q_display / sqrt(2),
/// and S = Q_display = Q_internal * sqrt(2).
/// When S=1 (Q_display=1): Butterworth slope.
fn low_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }

    let a_amp = g.sqrt(); // A = sqrt(g) = 10^(dBgain/40)
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    // Convert internal Q to shelf slope S: S = Q * sqrt(2) = Q_display
    let s = q * std::f64::consts::SQRT_2;
    let alpha = shelf_alpha(sin_w0, a_amp, s);
    let two_sqrt_a_alpha = 2.0 * a_amp.sqrt() * alpha;

    let a0 = (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = -2.0 * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0);
    let a2 = (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha;
    let b0 = a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = 2.0 * a_amp * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0);
    let b2 = a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha);

    let a0_inv = 1.0 / a0;
    [
        1.0,
        a1 * a0_inv,
        a2 * a0_inv,
        b0 * a0_inv,
        b1 * a0_inv,
        b2 * a0_inv,
    ]
}

/// 2nd-order high shelf (RBJ cookbook / bilinear transform).
///
/// Uses the RBJ Audio EQ Cookbook shelf formulas with the S (shelf slope)
/// parameter, same convention as low_shelf_2.
fn high_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }

    let a_amp = g.sqrt(); // A = sqrt(g) = 10^(dBgain/40)
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let s = q * std::f64::consts::SQRT_2;
    let alpha = shelf_alpha(sin_w0, a_amp, s);
    let two_sqrt_a_alpha = 2.0 * a_amp.sqrt() * alpha;

    let a0 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0);
    let a2 = (a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha;
    let b0 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = -2.0 * a_amp * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0);
    let b2 = a_amp * ((a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha);

    let a0_inv = 1.0 / a0;
    [
        1.0,
        a1 * a0_inv,
        a2 * a0_inv,
        b0 * a0_inv,
        b1 * a0_inv,
        b2 * a0_inv,
    ]
}

/// First-order section for Julius Smith's spectral tilt filter.
///
/// Each section is a first-order pole-zero pair, digitized via bilinear transform
/// with prewarping to the pivot frequency.
///
/// `pole_hz` and `zero_hz`: analog pole/zero frequencies.
/// `pivot_hz`: the frequency around which the tilt pivots (0 dB crossing).
/// `sample_rate`: sample rate in Hz.
pub fn flat_tilt_section(pole_hz: f64, zero_hz: f64, pivot_hz: f64, sample_rate: f64) -> Coeffs {
    // Use bilinear transform with prewarping to the pivot frequency
    let t = 1.0 / sample_rate;
    let wp = 2.0 * PI * pivot_hz;
    let wp_prewarp = (wp * t / 2.0).tan();

    // Prewarp pole and zero frequencies
    let pole_analog = 2.0 * PI * pole_hz;
    let zero_analog = 2.0 * PI * zero_hz;

    let prewarp = |w: f64| -> f64 {
        if wp_prewarp.abs() < 1e-30 || (wp * t / 2.0).tan().abs() < 1e-30 {
            return w;
        }
        wp_prewarp * (w * t / 2.0).tan() / (wp * t / 2.0).tan()
    };

    let p = prewarp(pole_analog);
    let z = prewarp(zero_analog);

    // Bilinear transform: s -> 2/T * (1 - z^-1)/(1 + z^-1)
    // H(s) = (s + z) / (s + p) → H(z) with bilinear
    let c = 2.0 / t;

    let a0 = c + p;
    let a1 = -c + p;
    let b0 = c + z;
    let b1 = -c + z;

    // Normalize by a0
    let a0_inv = 1.0 / a0;

    [1.0, a1 * a0_inv, 0.0, b0 * a0_inv, b1 * a0_inv, 0.0]
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
