//! Filter coefficient calculation — hybrid matched/RBJ approach.
//!
//! Peak, notch, allpass, and 1st-order shelves use Vicanek matched design (no Nyquist cramping).
//! LP, HP, bandpass, 2nd-order shelves, and tilt shelf use RBJ cookbook (bilinear transform).

use std::f64::consts::PI;

use crate::filter_type::FilterType;

/// Raw biquad coefficients: [a0, a1, a2, b0, b1, b2].
/// Convention: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
pub type Coeffs = [f64; 6];

/// Passthrough (identity) coefficients.
pub const PASSTHROUGH: Coeffs = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

// r[impl dsp.biquad.coefficients]
/// Calculate filter coefficients for the given parameters.
pub fn calculate(
    filter_type: FilterType,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sample_rate: f64,
) -> Coeffs {
    let w0 = 2.0 * PI * freq_hz / sample_rate;
    let g = 10.0_f64.powf(gain_db / 20.0);
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
        FilterType::BandShelf | FilterType::FlatTilt => PASSTHROUGH,
    }
}

// ── Vicanek matched filter helpers ──────────────────────────────────

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

#[inline]
fn phi0(w: f64) -> f64 {
    0.5 + 0.5 * w.cos()
}

#[inline]
fn phi1(w: f64) -> f64 {
    0.5 - 0.5 * w.cos()
}

fn mag_sq_to_b(big_b: [f64; 3]) -> (f64, f64, f64) {
    let b0_sq = big_b[0].max(0.0);
    let b1_sq = big_b[1].max(0.0);
    let b0_sqrt = b0_sq.sqrt();
    let b1_sqrt = b1_sq.sqrt();
    let w = (b0_sqrt + b1_sqrt) / 2.0;

    if big_b[2].abs() < 1e-30 {
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

// ── RBJ cookbook filters ─────────────────────────────────────────────

fn lowpass_2(w0: f64, q: f64) -> Coeffs {
    // Vicanek matched lowpass: impulse-invariance poles + magnitude matching.
    // Avoids bilinear transform cramping near Nyquist.
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;
    let p0 = phi0(w0);
    let p1 = phi1(w0);

    // DC gain = 1
    let b0_big = a0_big;
    // Nyquist gain = analog value (not 0 — avoids cramping)
    // Analog 2nd-order LP: |H(jw)|² = 1/((1-r²)² + r²/Q²)  where r = w/w0
    let r = PI / w0; // Nyquist / cutoff ratio
    let r2 = r * r;
    let nyq_mag_sq = 1.0 / ((1.0 - r2).powi(2) + r2 / (q * q));
    let b1_big = a1_big * nyq_mag_sq;

    // Match at corner: |H(w0)|² = Q² (exact for 2nd-order LP)
    let q_sq = q * q;
    let target_at_corner = q_sq * (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0);
    let b2_big = (target_at_corner - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
}

fn highpass_2(w0: f64, q: f64) -> Coeffs {
    // Vicanek matched highpass: impulse-invariance poles + magnitude matching.
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;
    let p0 = phi0(w0);
    let p1 = phi1(w0);

    // Nyquist gain = 1
    let b1_big = a1_big;
    // DC gain = analog value (not 0 — symmetric with LP approach)
    // Analog 2nd-order HP: |H(jw)|² = r⁴/((1-r²)² + r²/Q²) where r = w/w0
    // At DC (w→0): r→0, so |H|² → 0. Use 0 for DC.
    let b0_big = 0.0;

    // Match at corner: |H(w0)|² = Q²
    let q_sq = q * q;
    let target_at_corner = q_sq * (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0);
    let b2_big = (target_at_corner - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
}

fn bandpass_2(w0: f64, q: f64) -> Coeffs {
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * q);
    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
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

// ── Vicanek matched filters ─────────────────────────────────────────

fn notch_2(w0: f64, q: f64) -> Coeffs {
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);
    let b0 = 1.0;
    let b1 = -2.0 * w0.cos();
    let b2 = 1.0;
    let dc_num = b0 + b1 + b2;
    let dc_den = 1.0 + a1 + a2;
    let scale = dc_den / dc_num;
    [1.0, a1, a2, b0 * scale, b1 * scale, b2 * scale]
}

fn peak_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    let pole_q = (g.sqrt() * q).max(0.01);
    let (a1, a2) = solve_poles(w0, 0.5 / pole_q, 1.0);
    let a0_big = (1.0 + a1 + a2) * (1.0 + a1 + a2);
    let a1_big = (1.0 - a1 + a2) * (1.0 - a1 + a2);
    let a2_big = -4.0 * a2;
    let p0 = phi0(w0);
    let p1 = phi1(w0);
    let g_sq = g * g;
    let r1 = (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0) * g_sq;
    let r2 = (-a0_big + a1_big + 4.0 * (p0 - p1) * a2_big) * g_sq;
    let b0_big = a0_big;
    let b2_big = (r1 - r2 * p1 - b0_big) / (4.0 * p1 * p1);
    let b1_big = r2 + b0_big + 4.0 * (p1 - p0) * b2_big;
    let (b0, b1, b2) = mag_sq_to_b([b0_big, b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
}

fn tilt_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    // Tilt shelf: DC gain = √g, Nyquist gain = 1/√g.
    // Implemented as RBJ low shelf with gain=g, scaled by 1/√g to center
    // the tilt around unity. RBJ shelf is numerically robust at all frequencies.
    let a_amp = g.sqrt(); // RBJ uses A = √gain
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let s = q * std::f64::consts::SQRT_2;
    let alpha = shelf_alpha(sin_w0, a_amp, s);
    let two_sqrt_a_alpha = 2.0 * a_amp.sqrt() * alpha;
    let a0 = (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha;
    let a1 = -2.0 * ((a_amp - 1.0) + (a_amp + 1.0) * cos_w0);
    let a2 = (a_amp + 1.0) + (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha;
    let b0 = a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 + two_sqrt_a_alpha);
    let b1 = 2.0 * a_amp * ((a_amp - 1.0) - (a_amp + 1.0) * cos_w0);
    let b2 = a_amp * ((a_amp + 1.0) - (a_amp - 1.0) * cos_w0 - two_sqrt_a_alpha);
    // Scale by 1/√g to center tilt: low shelf has DC=g, Nyquist=1;
    // after scaling: DC=g/√g=√g, Nyquist=1/√g.
    let scale = 1.0 / g.sqrt();
    let a0_inv = scale / a0;
    [1.0, a1 / a0, a2 / a0, b0 * a0_inv, b1 * a0_inv, b2 * a0_inv]
}

fn allpass_2(w0: f64, q: f64) -> Coeffs {
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);
    [1.0, a1, a2, a2, a1, 1.0]
}

// ── 1st-order filters ───────────────────────────────────────────────

pub fn lowpass_1(freq_hz: f64, sample_rate: f64) -> Coeffs {
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let wc = (w0 / 2.0).tan();
    let a0_inv = 1.0 / (1.0 + wc);
    let b0 = wc * a0_inv;
    let b1 = b0;
    let a1 = (wc - 1.0) * a0_inv;
    [1.0, a1, 0.0, b0, b1, 0.0]
}

pub fn highpass_1(freq_hz: f64, sample_rate: f64) -> Coeffs {
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let wc = (w0 / 2.0).tan();
    let a0_inv = 1.0 / (1.0 + wc);
    let b0 = a0_inv;
    let b1 = -b0;
    let a1 = (wc - 1.0) * a0_inv;
    [1.0, a1, 0.0, b0, b1, 0.0]
}

pub fn low_shelf_1(freq_hz: f64, gain_db: f64, sample_rate: f64) -> Coeffs {
    // Vicanek matched 1-pole low shelf (no Nyquist cramping).
    // Low shelf = high shelf with 1/G, then scale b coefficients by G.
    let g = 10.0_f64.powf(gain_db / 20.0);
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    let c = high_shelf_1_matched(freq_hz, 1.0 / g, sample_rate);
    // Scale numerator by gain to get low-shelf DC=G, Nyquist=1
    [c[0], c[1], c[2], c[3] * g, c[4] * g, c[5]]
}

pub fn high_shelf_1(freq_hz: f64, gain_db: f64, sample_rate: f64) -> Coeffs {
    let g = 10.0_f64.powf(gain_db / 20.0);
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    high_shelf_1_matched(freq_hz, g, sample_rate)
}

/// Vicanek matched 1-pole high shelf filter.
/// fc is in Hz, g is linear gain (>1 boost, <1 cut).
fn high_shelf_1_matched(freq_hz: f64, g: f64, sample_rate: f64) -> Coeffs {
    let fc = freq_hz / (sample_rate / 2.0); // normalize to Nyquist
    let fc = fc.max(1e-6);
    let fc_sq = fc * fc;
    let fm = 0.9_f64; // matching point slightly below Nyquist
    let phi_m = 1.0 - (PI * fm).cos();

    // Use eq. 12 with matching point fm = 0.9
    let alpha = (2.0 / (PI * PI)) * (1.0 / (fm * fm) + 1.0 / (g * fc_sq)) - 1.0 / phi_m;
    let beta = (2.0 / (PI * PI)) * (1.0 / (fm * fm) + g / fc_sq) - 1.0 / phi_m;

    // Recover a1 and b ratio from α and β (eq. 10)
    let a1 = -alpha / (1.0 + alpha + (1.0 + 2.0 * alpha).max(0.0).sqrt());
    let b_ratio = -beta / (1.0 + beta + (1.0 + 2.0 * beta).max(0.0).sqrt());

    // High shelf normalization: DC gain = 1, so b0 + b1 = 1 + a1 (eq. 6/11)
    let b0 = (1.0 + a1) / (1.0 + b_ratio);
    let b1 = b_ratio * b0;

    [1.0, a1, 0.0, b0, b1, 0.0]
}

pub fn allpass_1(freq_hz: f64, sample_rate: f64) -> Coeffs {
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let p = (-w0).exp();
    [1.0, -p, 0.0, -p, 1.0, 0.0]
}

/// 1st-order tilt shelf using Vicanek matched 1-pole shelf.
/// DC gain = 1/√g, Nyquist gain = √g (tilts around pivot at freq_hz).
pub fn tilt_shelf_1(freq_hz: f64, gain_db: f64, sample_rate: f64) -> Coeffs {
    let g = 10.0_f64.powf(gain_db / 20.0);
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    // Low shelf with gain=g, then scale by 1/√g to center tilt.
    let c = low_shelf_1(freq_hz, gain_db, sample_rate);
    let inv_sqrt_g = 1.0 / g.sqrt();
    [c[0], c[1], c[2], c[3] * inv_sqrt_g, c[4] * inv_sqrt_g, c[5]]
}

// ── Matched 2nd-order shelves with Q ─────────────────────────────────

fn shelf_alpha(sin_w0: f64, a_amp: f64, s: f64) -> f64 {
    let s = s.max(0.01);
    let val = (a_amp + 1.0 / a_amp) * (1.0 / s - 1.0) + 2.0;
    sin_w0 / 2.0 * val.max(0.001).sqrt()
}

fn low_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    // Matched low shelf = matched high shelf with 1/G, scale numerator by G
    let c = high_shelf_2_matched_q(w0, q, 1.0 / g);
    [c[0], c[1], c[2], c[3] * g, c[4] * g, c[5] * g]
}

fn high_shelf_2(w0: f64, q: f64, g: f64) -> Coeffs {
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    high_shelf_2_matched_q(w0, q, g)
}

/// Vicanek-style matched 2nd-order high shelf with arbitrary Q.
/// Uses impulse-invariance for poles + magnitude matching at DC/Nyquist/corner.
/// The analog prototype is: H(s) = A·[A·s²+(√A/Q)·s+1] / [s²+(√A/Q)·s+A]
/// where A = √G.
fn high_shelf_2_matched_q(w0: f64, q: f64, g: f64) -> Coeffs {
    let a = g.sqrt(); // A = √G (RBJ convention)
    let sqrt_a = a.sqrt(); // √A = G^(1/4)

    // Map analog shelf denominator poles to digital via impulse invariance.
    // Denominator: s² + (√A/Q)·ωc·s + A·ωc² = 0
    // Standard form poles at s = ωc·(-b ± j·√(c²-b²))
    // where b = √A/(2Q), c = √A
    let (a1, a2) = solve_poles(w0, sqrt_a / (2.0 * q), sqrt_a);

    // Squared-magnitude parameters for denominator
    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;
    let p0 = phi0(w0);
    let p1 = phi1(w0);

    // DC: high shelf has |H(0)|² = 1
    let b0_big = a0_big;

    // Nyquist: compute from analog prototype
    // |H(jf)|² = A²·[(1-A·f²)² + A·f²/Q²] / [(A-f²)² + A·f²/Q²]
    let f_ny = PI / w0;
    let f_ny_sq = f_ny * f_ny;
    let a_f_ny_sq = a * f_ny_sq;
    let a_over_q_sq = a * f_ny_sq / (q * q);
    let num_ny = (1.0 - a_f_ny_sq).powi(2) + a_over_q_sq;
    let den_ny = (a - f_ny_sq).powi(2) + a_over_q_sq;
    let h_ny_sq = if den_ny.abs() > 1e-30 {
        a * a * num_ny / den_ny
    } else {
        g // fallback to full gain
    };
    let b1_big = a1_big * h_ny_sq;

    // Corner: |H(w0)|² = G (half-gain point in dB)
    let target = g * (a0_big * p0 + a1_big * p1 + a2_big * 4.0 * p0 * p1);
    let b2_big = (target - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
}

/// Zoelzer parametric low shelf with resonance support.
/// Uses Q directly (not S). Q=1/√2 gives Butterworth. Q > 1/√2 creates resonance.
pub fn low_shelf_resonant(freq_hz: f64, q: f64, gain_db: f64, sample_rate: f64) -> Coeffs {
    let g = 10.0_f64.powf(gain_db / 20.0);
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let v = g;
    let k = 1.0 / (w0 / 2.0).tan();
    let k2 = k * k;
    let sqrt_2v = (2.0 * v).sqrt();
    let sqrt_2 = std::f64::consts::SQRT_2;
    let a0 = k2 + sqrt_2 * k / q + 1.0;
    let a1 = 2.0 * (1.0 - k2);
    let a2 = k2 - sqrt_2 * k / q + 1.0;
    let b0 = k2 + sqrt_2v * k / q + v;
    let b1 = 2.0 * (v - k2);
    let b2 = k2 - sqrt_2v * k / q + v;
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

/// Zoelzer parametric high shelf with resonance support.
pub fn high_shelf_resonant(freq_hz: f64, q: f64, gain_db: f64, sample_rate: f64) -> Coeffs {
    let g = 10.0_f64.powf(gain_db / 20.0);
    if (g - 1.0).abs() < 1e-6 {
        return PASSTHROUGH;
    }
    let w0 = (2.0 * PI * freq_hz / sample_rate).clamp(1e-6, PI - 1e-6);
    let v = g;
    let k = 1.0 / (w0 / 2.0).tan();
    let k2 = k * k;
    let sqrt_2v = (2.0 * v).sqrt();
    let sqrt_2 = std::f64::consts::SQRT_2;
    let a0 = k2 + sqrt_2 * k / q + 1.0;
    let a1 = 2.0 * (1.0 - k2);
    let a2 = k2 - sqrt_2 * k / q + 1.0;
    let b0 = v * k2 + sqrt_2v * k / q + 1.0;
    let b1 = 2.0 * (1.0 - v * k2);
    let b2 = v * k2 - sqrt_2v * k / q + 1.0;
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

// ── Flat tilt ───────────────────────────────────────────────────────

pub fn flat_tilt_section(pole_hz: f64, zero_hz: f64, pivot_hz: f64, sample_rate: f64) -> Coeffs {
    let t = 1.0 / sample_rate;
    let wp = 2.0 * PI * pivot_hz;
    let wp_prewarp = (wp * t / 2.0).tan();
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
    let c = 2.0 / t;
    let a0 = c + p;
    let a1 = -c + p;
    let b0 = c + z;
    let b1 = -c + z;
    let a0_inv = 1.0 / a0;
    [1.0, a1 * a0_inv, 0.0, b0 * a0_inv, b1 * a0_inv, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_unity_gain_is_passthrough() {
        let c = calculate(FilterType::Peak, 1000.0, 0.707, 0.0, 44100.0);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[3] - 1.0).abs() < 1e-10);
        assert!(c[1].abs() < 1e-10);
        assert!(c[4].abs() < 1e-10);
    }

    #[test]
    fn lowpass_has_unity_dc_gain() {
        let c = lowpass_2(2.0 * PI * 1000.0 / 44100.0, 0.707);
        let dc = (c[3] + c[4] + c[5]) / (c[0] + c[1] + c[2]);
        assert!((dc - 1.0).abs() < 0.01, "DC gain = {dc}, expected ~1.0");
    }

    #[test]
    fn highpass_has_zero_dc_gain() {
        let c = highpass_2(2.0 * PI * 1000.0 / 44100.0, 0.707);
        let dc = (c[3] + c[4] + c[5]) / (c[0] + c[1] + c[2]);
        assert!(dc.abs() < 0.01, "DC gain = {dc}, expected ~0.0");
    }

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
