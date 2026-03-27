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
    // Matches magnitude at DC (1), Nyquist (analog value), and corner (Q^2).
    // Better than BLT for passband accuracy near Nyquist.
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;
    let p0 = phi0(w0);
    let p1 = phi1(w0);

    // DC gain = 1
    let b0_big = a0_big;
    // Nyquist gain = analog value (avoids cramping)
    let r = PI / w0;
    let r2 = r * r;
    let nyq_mag_sq = 1.0 / ((1.0 - r2).powi(2) + r2 / (q * q));
    let b1_big = a1_big * nyq_mag_sq;

    // Match at corner: |H(w0)|^2 = Q^2
    let q_sq = q * q;
    let target_at_corner = q_sq * (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0);
    let b2_big = (target_at_corner - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
}

fn highpass_2(w0: f64, q: f64) -> Coeffs {
    // Hybrid: Vicanek impulse-invariance poles (good passband, no cramping)
    // + exact zeros at z=1 (deep DC stopband).
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);
    // Numerator: proportional to (1 - z^-1)^2 = 1 - 2z^-1 + z^-2
    // Scale for unity Nyquist gain: H(-1) = (b0-b1+b2)/(1-a1+a2) = 1
    let nyq_den = 1.0 - a1 + a2;
    let scale = nyq_den / 4.0; // (1-(-2)+1) = 4
    [1.0, a1, a2, scale, -2.0 * scale, scale]
}

fn bandpass_2(w0: f64, q: f64) -> Coeffs {
    // Vicanek matched bandpass: impulse-invariance poles + 3-point magnitude matching.
    // Matches: DC=0 (exact zero), Nyquist=analog value, center=unity peak.
    let (a1, a2) = solve_poles(w0, 0.5 / q, 1.0);

    // Analog bandpass magnitude squared at Nyquist (frequency ratio = π/w0).
    let r = PI / w0;
    let r2 = r * r;
    let nyq_mag_sq = r2 / (q * q) / ((r2 - 1.0).powi(2) + r2 / (q * q));

    // Denominator magnitude squared at Nyquist
    let den_nyq_sq = (1.0 - a1 + a2).powi(2);

    let cw = w0.cos();
    let sw = w0.sin();
    let c2w = (2.0 * cw * cw) - 1.0;
    let s2w = 2.0 * sw * cw;
    let den_re = 1.0 + a1 * cw + a2 * c2w;
    let den_im = -a1 * sw - a2 * s2w;
    let den_w0_sq = den_re * den_re + den_im * den_im;

    // Constraint 1: b0+b1+b2 = 0 (DC zero), so b1 = -(b0+b2)
    // Constraint 2: |H(π)|² = nyq_mag_sq → (2*(b0+b2))² / den_nyq_sq = nyq_mag_sq
    let s_val = nyq_mag_sq.sqrt() * den_nyq_sq.sqrt() / 2.0; // b0+b2

    // Constraint 3: |H(w0)|² = 1 → |N(w0)|² = den_w0_sq
    // With b1 = -(b0+b2) = -S, b2 = S - b0:
    //   N_re = b0*(1-c2w) + S*(c2w-cw)
    //   N_im = b0*s2w + S*(sw-s2w)
    // |N|² = P*b0² + 2*S*R*b0 + S²*T = den_w0_sq
    let aa = 1.0 - c2w; // coefficient of b0 in N_re
    let bb = c2w - cw; // coefficient of S in N_re
    let cc = s2w; // coefficient of b0 in N_im
    let dd = sw - s2w; // coefficient of S in N_im

    let p_coeff = aa * aa + cc * cc; // 4*sin²(w0)
    let r_coeff = aa * bb + cc * dd;
    let t_coeff = bb * bb + dd * dd;

    // Quadratic: P*b0² + 2*S*R*b0 + (S²*T - den_w0_sq) = 0
    let qa = p_coeff;
    let qb = 2.0 * s_val * r_coeff;
    let qc = s_val * s_val * t_coeff - den_w0_sq;

    let disc = (qb * qb - 4.0 * qa * qc).max(0.0);
    // Pick the root that gives positive b0
    let b0_a = (-qb + disc.sqrt()) / (2.0 * qa);
    let b0_b = (-qb - disc.sqrt()) / (2.0 * qa);
    let b0 = if b0_a > 0.0 { b0_a } else { b0_b };
    let b2 = s_val - b0;
    let b1 = -s_val; // -(b0+b2)

    [1.0, a1, a2, b0, b1, b2]
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
    // Use Vicanek matched low shelf (gain=g), scaled by 1/√g to center
    // the tilt around unity: DC=g/√g=√g, Nyquist=1/√g.
    let c = low_shelf_2(w0, q, g);
    let scale = 1.0 / g.sqrt();
    [c[0], c[1], c[2], c[3] * scale, c[4] * scale, c[5] * scale]
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
    let fm = 0.8_f64; // matching point
    let phi_m = 1.0 - (PI * fm).cos();
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

/// Vicanek matched 2nd-order high shelf via tilt shelf proxy (ZL approach).
///
/// High shelf = tilt_shelf(g) with numerator scaled by √g.
/// DC=1, Nyquist=g.
fn high_shelf_2_matched_q(w0: f64, q: f64, g: f64) -> Coeffs {
    let c = tilt_shelf_2_matched(w0, q, g);
    let scale = g.sqrt();
    [c[0], c[1], c[2], c[3] * scale, c[4] * scale, c[5] * scale]
}

/// Vicanek matched 2nd-order tilt shelf with optimal matching frequencies.
///
/// For g > 1 (boost), works with 1/g and swaps numerator/denominator.
/// Matching frequencies derived analytically from the analog prototype's
/// error-minimizing points (Vicanek's quadratic).
fn tilt_shelf_2_matched(w0: f64, q: f64, g: f64) -> Coeffs {
    let reverse = g > 1.0;
    let g = if reverse { 1.0 / g } else { g };

    let sqrt_g = g.sqrt();
    let ssqrt_g = sqrt_g.sqrt(); // g^(1/4)

    // Poles via impulse invariance (same as ZL's solve_a with 3 args)
    let (a1, a2) = solve_poles(w0, ssqrt_g / (2.0 * q), ssqrt_g);

    // Denominator squared-magnitude parameters: A = get_AB(a)
    let a_big = [(1.0 + a1 + a2).powi(2), (1.0 - a1 + a2).powi(2), -4.0 * a2];

    // Compute optimal matching frequencies from Vicanek's quadratic
    let w02 = w0 * w0;
    let c2 = sqrt_g * (-1.0 + 2.0 * q * q);
    let c0 = c2 * w02 * w02;
    let c1 = -2.0 * (1.0 + g) * (q * w0) * (q * w0);
    let discriminant = c1 * c1 - 4.0 * c0 * c2;

    let mut ws = if discriminant <= 0.0 {
        [0.0, w0 * 0.5, w0]
    } else {
        let delta = discriminant.sqrt();
        let inv_2c2 = 0.5 / c2;
        let sol1 = (-c1 + delta) * inv_2c2;
        let sol2 = (-c1 - delta) * inv_2c2;
        if sol1 < 0.0 || sol2 < 0.0 {
            [0.0, w0 * 0.5, w0]
        } else {
            let w1 = sol1.sqrt();
            let w2 = sol2.sqrt();
            if w1 < PI || w2 < PI {
                [0.0, w1.min(w2), w1.max(w2).min(PI)]
            } else {
                [0.0, PI / 2.0, PI]
            }
        }
    };

    // Tilt shelf analog magnitude squared: H(s) = N(s)/D(s) where
    //   N(s) = sqrt_g·s² + ssqrt_g·w0/q·s + w0²
    //   D(s) = s² + ssqrt_g·w0/q·s + sqrt_g·w0²
    // |H(jw)|² = |N(jw)|²/|D(jw)|²
    let w0_over_q = w0 / q;
    let tilt_mag2 = |w: f64| -> f64 {
        let w2 = w * w;
        // N(jw) = (w0² - sqrt_g·w²) + j·(ssqrt_g·w0/q·w)
        let nr = w02 - sqrt_g * w2;
        let ni = ssqrt_g * w0_over_q * w;
        // D(jw) = (sqrt_g·w0² - w²) + j·(ssqrt_g·w0/q·w)
        let dr = sqrt_g * w02 - w2;
        let di = ssqrt_g * w0_over_q * w;
        let num_sq = nr * nr + ni * ni;
        let den_sq = dr * dr + di * di;
        if den_sq < 1e-30 {
            return 1.0;
        }
        num_sq / den_sq
    };

    // Retry loop matches ZL: B starts invalid, first solve uses original ws
    let _ws = ws;
    let mut b_big = [-1.0, -1.0, -1.0_f64];
    let mut trial = 0;
    while !check_b_valid(&b_big) && trial < 20 {
        trial += 1;
        let mut phi = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for i in 0..3 {
            let p = phi_vec(ws[i]);
            phi[i] = p;
            rhs[i] = tilt_mag2(ws[i]) * dot3(&p, &a_big);
        }
        b_big = linear_solve_3x3(&phi, &rhs);
        ws[2] = 0.5 * (ws[2] + PI);
    }
    // Fallback if still invalid after 20 trials: revert to original ws
    if trial == 20 {
        let ws_fallback = _ws;
        let mut phi = [[0.0; 3]; 3];
        let mut rhs = [0.0; 3];
        for i in 0..3 {
            let p = phi_vec(ws_fallback[i]);
            phi[i] = p;
            rhs[i] = tilt_mag2(ws_fallback[i]) * dot3(&p, &a_big);
        }
        b_big = linear_solve_3x3(&phi, &rhs);
    }

    let (b0, b1, b2) = mag_sq_to_b([b_big[0].max(0.0), b_big[1].max(0.0), b_big[2]]);

    if reverse {
        [b0, b1, b2, 1.0, a1, a2]
    } else {
        [1.0, a1, a2, b0, b1, b2]
    }
}

fn phi_vec(w: f64) -> [f64; 3] {
    let c = 0.5 * w.cos();
    let p0 = 0.5 + c;
    let p1 = 0.5 - c;
    [p0, p1, 4.0 * p0 * p1]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn check_b_valid(b: &[f64; 3]) -> bool {
    b[0] > 0.0 && b[1] > 0.0 && (0.5 * (b[0].sqrt() + b[1].sqrt())).powi(2) + b[2] > 0.0
}

/// Solve 3×3 system exploiting phi(0)=[1,0,0] structure.
fn linear_solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    if a[0][0].abs() > a[0][1].abs() {
        let x0 = b[0] / a[0][0];
        let denom = -(a[1][2] * a[2][1] - a[1][1] * a[2][2]);
        if denom.abs() < 1e-30 {
            return [-1.0; 3];
        }
        let x1 = (a[2][2] * b[1] - a[1][2] * b[2] + a[1][2] * a[2][0] * x0
            - a[1][0] * a[2][2] * x0)
            / denom;
        let x2 = (-a[2][1] * b[1] + a[1][1] * b[2] - a[1][1] * a[2][0] * x0
            + a[1][0] * a[2][1] * x0)
            / denom;
        [x0, x1, x2]
    } else {
        let x1 = b[0] / a[0][1];
        let denom = -(a[1][2] * a[2][0] - a[1][0] * a[2][2]);
        if denom.abs() < 1e-30 {
            return [-1.0; 3];
        }
        let x0 = (a[1][2] * a[2][1] * b[0] - a[1][1] * a[2][2] * b[0] + a[2][2] * b[1]
            - a[1][2] * b[2])
            / denom;
        let x2 = (a[1][1] * a[2][0] * b[0] - a[1][0] * a[2][1] * b[0] - a[2][0] * b[1]
            + a[1][0] * b[2])
            / denom;
        [x0, x1, x2]
    }
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
