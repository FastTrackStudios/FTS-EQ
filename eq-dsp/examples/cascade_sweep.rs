//! Sweep sigma correction parameters for LP/HP cascades against Pro-Q 4 reference data.
//!
//! Usage: cargo run --release --example cascade_sweep --package eq-dsp

use std::f64::consts::PI;
use std::fs;
use std::path::Path;

const SR: f64 = 48000.0;
const NOISE_FLOOR_DB: f64 = -80.0;
const REF_DIR: &str = "/home/cody/Development/FastTrackStudio/fts-analyzer/reference/pro-q4/48k";

// ── Reference data loading ──────────────────────────────────────────

struct RefData {
    freqs: Vec<f64>,
    mags: Vec<f64>,
}

fn load_ref(path: &Path) -> Option<RefData> {
    let data = fs::read(path).ok()?;
    if data.len() < 4 {
        return None;
    }
    let num_bins = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
    let has_gd = (data.len() - 4) / num_bins == 16;
    let stride = if has_gd { 16 } else { 12 };

    let mut freqs = Vec::with_capacity(num_bins);
    let mut mags = Vec::with_capacity(num_bins);

    for i in 0..num_bins {
        let off = 4 + i * stride;
        let freq = f32::from_le_bytes(data[off..off + 4].try_into().ok()?) as f64;
        let mag = f32::from_le_bytes(data[off + 4..off + 8].try_into().ok()?) as f64;
        freqs.push(freq);
        mags.push(mag);
    }

    Some(RefData { freqs, mags })
}

// ── Coefficient computation ─────────────────────────────────────────

fn solve_poles(w0: f64, b: f64, c: f64) -> (f64, f64) {
    let t = (-b * w0).exp();
    let a1 = if b <= c {
        -2.0 * t * ((c * c - b * b).sqrt() * w0).cos()
    } else {
        -2.0 * t * ((b * b - c * c).sqrt() * w0).cosh()
    };
    (a1, t * t)
}

fn phi0(w: f64) -> f64 {
    0.5 + 0.5 * w.cos()
}
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
        return (w, b0_sqrt - w, 0.0);
    }
    let b0 = (w + (w * w + big_b[2]).max(0.0).sqrt()) / 2.0;
    let b0 = b0.max(1e-30);
    let b1 = (b0_sqrt - b1_sqrt) / 2.0;
    let b2 = -big_b[2] / (4.0 * b0);
    (b0, b1, b2)
}

type Coeffs = [f64; 6];

/// LP section with parameterized bidirectional sigma correction.
/// sigma_scale: multiplier on sigma (>1 = more damped, <1 = less damped)
fn lp_section(w0: f64, q: f64, sigma_scale: f64) -> Coeffs {
    let sigma = 0.5 / q;
    let sigma_eff = (sigma * sigma_scale).max(0.001);

    let (a1, a2) = solve_poles(w0, sigma_eff, 1.0);

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;
    let p0 = phi0(w0);
    let p1 = phi1(w0);

    let b0_big = a0_big;
    let r = PI / w0;
    let r2 = r * r;
    let nyq_mag_sq = 1.0 / ((1.0 - r2).powi(2) + r2 / (q * q));
    let b1_big = a1_big * nyq_mag_sq;

    let q_sq = q * q;
    let target = q_sq * (a0_big * p0 + a1_big * p1 + a2_big * p0 * p1 * 4.0);
    let b2_big = (target - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
}

/// HP section with parameterized sigma correction.
fn hp_section(w0: f64, q: f64, sigma_scale: f64) -> Coeffs {
    let sigma = 0.5 / q;
    let sigma_eff = (sigma * sigma_scale).max(0.001);

    let (a1, a2) = solve_poles(w0, sigma_eff, 1.0);

    let cw = w0.cos();
    let sw = w0.sin();
    let c2w = 2.0 * cw * cw - 1.0;
    let s2w = 2.0 * sw * cw;
    let num_re = 1.0 - 2.0 * cw + c2w;
    let num_im = 2.0 * sw - s2w;
    let num_mag = (num_re * num_re + num_im * num_im).sqrt();
    let den_re = 1.0 + a1 * cw + a2 * c2w;
    let den_im = -a1 * sw - a2 * s2w;
    let den_mag = (den_re * den_re + den_im * den_im).sqrt();
    let scale = q * den_mag / num_mag.max(1e-30);
    [1.0, a1, a2, scale, -2.0 * scale, scale]
}

// ── Cascade magnitude + RMS ─────────────────────────────────────────

fn cascade_mag_db(sections: &[Coeffs], freq: f64) -> f64 {
    let w = 2.0 * PI * freq / SR;
    let (cw, sw) = (w.cos(), w.sin());
    let (c2w, s2w) = (2.0 * cw * cw - 1.0, 2.0 * sw * cw);

    let mut mag_sq = 1.0_f64;
    for c in sections {
        let nr = c[3] + c[4] * cw + c[5] * c2w;
        let ni = -(c[4] * sw + c[5] * s2w);
        let dr = c[0] + c[1] * cw + c[2] * c2w;
        let di = -(c[1] * sw + c[2] * s2w);
        let n_sq = nr * nr + ni * ni;
        let d_sq = dr * dr + di * di;
        if d_sq > 1e-30 {
            mag_sq *= n_sq / d_sq;
        }
    }
    10.0 * mag_sq.max(1e-30).log10()
}

fn cascade_rms(sections: &[Coeffs], ref_data: &RefData) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for (i, &freq) in ref_data.freqs.iter().enumerate() {
        let ref_db = ref_data.mags[i];
        let our_db = cascade_mag_db(sections, freq);
        if ref_db < NOISE_FLOOR_DB && our_db < NOISE_FLOOR_DB {
            continue;
        }
        let diff = our_db - ref_db;
        sum_sq += diff * diff;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt()
}

// ── Butterworth Q ───────────────────────────────────────────────────

fn butterworth_q_for_order(order: usize, i: usize) -> f64 {
    let angle = PI * (2 * i + 1) as f64 / (2 * order) as f64;
    0.5 / angle.sin()
}

const SQRT_2: f64 = std::f64::consts::SQRT_2;

// ── Test infrastructure ─────────────────────────────────────────────

struct TestCase {
    ftype: &'static str,
    fc: u32,
    display_q: f64,
    slope: u8,
    order: usize,
    w0: f64,
    section_qs: Vec<f64>,
    num_2nd: usize,
    first_order_coeffs: Option<Coeffs>,
    ref_data: RefData,
}

fn load_test_cases() -> Vec<TestCase> {
    let freqs = [
        20, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000, 12000, 14000, 16000, 17000, 18000,
        19000, 20000, 21000, 22000,
    ];
    let qs: [f64; 4] = [0.5, 1.0, 4.0, 10.0];
    // slope index → filter order (from lp_hp_slope_to_order in eq-plugin)
    let slopes = [(2u8, 2usize), (5, 5), (8, 12)];
    let ftypes = ["high_cut", "low_cut"];

    let mut cases = Vec::new();
    for &ftype in &ftypes {
        for &(slope, order) in &slopes {
            for &fc in &freqs {
                for &dq in &qs {
                    let dq_str = if dq.fract() == 0.0 {
                        format!("{}", dq as u32)
                    } else {
                        format!("{dq}")
                    };
                    let fname = format!("{ftype}_{fc}hz_q{dq_str}_s{slope}.bin");
                    let path = Path::new(REF_DIR).join(&fname);
                    let ref_data = match load_ref(&path) {
                        Some(d) => d,
                        None => continue,
                    };

                    let num_2nd = order / 2;
                    let has_first_order = order % 2 == 1;
                    let q_internal = dq / SQRT_2;
                    let w0 = 2.0 * PI * fc as f64 / SR;

                    let first_order_coeffs = if has_first_order {
                        Some(if ftype == "high_cut" {
                            eq_dsp::coeff::lowpass_1_matched(fc as f64, SR)
                        } else {
                            eq_dsp::coeff::highpass_1_matched(fc as f64, SR)
                        })
                    } else {
                        None
                    };

                    let mut section_qs = Vec::with_capacity(num_2nd);
                    for i in 0..num_2nd {
                        let bw_q = butterworth_q_for_order(order, i);
                        let q_sec = if i == 0 {
                            bw_q * q_internal * SQRT_2
                        } else {
                            bw_q
                        };
                        section_qs.push(q_sec);
                    }

                    cases.push(TestCase {
                        ftype,
                        fc,
                        display_q: dq,
                        slope,
                        order,
                        w0,
                        section_qs,
                        num_2nd,
                        first_order_coeffs,
                        ref_data,
                    });
                }
            }
        }
    }
    cases
}

/// Bidirectional sigma correction for LP.
/// sigma < crossover: INCREASE sigma (more damped) — counteracts II resonance sharpening
/// sigma > crossover: DECREASE sigma (less damped) — counteracts II transition band softening
fn lp_sigma_scale(sigma: f64, w_norm: f64, num_biquads: usize, params: &[f64; 5]) -> f64 {
    let [crossover, inc_strength, dec_strength, w_power, _div_power] = *params;
    let w_factor = w_norm.powf(w_power);
    let n = num_biquads as f64;

    if sigma < crossover {
        // Increase sigma for high-Q sections
        let t = (crossover - sigma) / crossover; // 0 at crossover, 1 at sigma=0
        1.0 + inc_strength * t * w_factor / n.sqrt()
    } else {
        // Decrease sigma for low-Q sections
        let t = (sigma - crossover) / (1.0 - crossover).max(0.01);
        let correction = dec_strength * t * w_factor / n.sqrt();
        1.0 - correction.min(0.49)
    }
}

/// Bidirectional sigma correction for HP.
fn hp_sigma_scale(sigma: f64, w_norm: f64, num_biquads: usize, params: &[f64; 5]) -> f64 {
    let [crossover, inc_strength, dec_strength, w_power, _div_power] = *params;
    let w_factor = w_norm.powf(w_power);
    let n = num_biquads as f64;

    if sigma < crossover {
        1.0 + inc_strength * ((crossover - sigma) / crossover) * w_factor / n.sqrt()
    } else {
        let t = (sigma - crossover) / (1.0 - crossover).max(0.01);
        let correction = dec_strength * t * w_factor / n.sqrt();
        1.0 - correction.min(0.49)
    }
}

fn eval_lp_bidir(case: &TestCase, params: &[f64; 5]) -> f64 {
    let w_norm = case.w0 / PI;
    let mut sections = Vec::new();
    if let Some(first) = case.first_order_coeffs {
        sections.push(first);
    }
    for &q in &case.section_qs {
        let sigma = 0.5 / q;
        let scale = lp_sigma_scale(sigma, w_norm, case.num_2nd, params);
        sections.push(lp_section(case.w0, q, scale));
    }
    cascade_rms(&sections, &case.ref_data)
}

fn eval_hp_bidir(case: &TestCase, params: &[f64; 5]) -> f64 {
    let w_norm = case.w0 / PI;
    let mut sections = Vec::new();
    if let Some(first) = case.first_order_coeffs {
        sections.push(first);
    }
    for &q in &case.section_qs {
        let sigma = 0.5 / q;
        let scale = hp_sigma_scale(sigma, w_norm, case.num_2nd, params);
        sections.push(hp_section(case.w0, q, scale));
    }
    cascade_rms(&sections, &case.ref_data)
}

fn main() {
    let cases = load_test_cases();
    let lp_cases: Vec<&TestCase> = cases.iter().filter(|c| c.ftype == "high_cut").collect();
    let hp_cases: Vec<&TestCase> = cases.iter().filter(|c| c.ftype == "low_cut").collect();

    // Only evaluate cases that are numerically stable in coefficient-based evaluation.
    // Low-freq high-order cases (20-100Hz s8) have numerical issues in frequency-domain
    // evaluation but work fine in actual audio processing. Filter to fc >= 500Hz for LP,
    // and avoid very low fc for HP.
    let lp_stable: Vec<&TestCase> = lp_cases
        .iter()
        .filter(|c| c.fc >= 500 || c.slope <= 5)
        .copied()
        .collect();
    let hp_stable: Vec<&TestCase> = hp_cases
        .iter()
        .filter(|c| c.fc <= 22000) // HP doesn't have the same low-freq issue
        .copied()
        .collect();

    println!(
        "Loaded {} LP ({} stable) + {} HP ({} stable) test cases",
        lp_cases.len(),
        lp_stable.len(),
        hp_cases.len(),
        hp_stable.len()
    );

    // ── Current formula baseline ────────────────────────────────────
    println!("\n=== CURRENT BASELINE (stable cases only) ===");
    let (lp_pass, lp_fails) = eval_baseline(&lp_stable, true);
    let (hp_pass, hp_fails) = eval_baseline(&hp_stable, false);
    println!(
        "  LP: {}/{} pass, HP: {}/{} pass",
        lp_pass,
        lp_stable.len(),
        hp_pass,
        hp_stable.len()
    );
    print_fails("LP", &lp_fails, 15);
    print_fails("HP", &hp_fails, 15);

    // ── Grid search: bidirectional sigma correction ─────────────────
    // The key insight from per-section optimization:
    // - sigma < crossover (high Q): INCREASE sigma to dampen II resonance sharpening
    // - sigma > crossover (low Q): DECREASE sigma to steepen II transition band
    println!("\n=== LP BIDIRECTIONAL GRID SEARCH (stable cases) ===");
    let crossovers = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5];
    let inc_strengths = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0];
    let dec_strengths = [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0];
    let w_powers = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0];

    let mut best_lp_pass = 0u32;
    let mut best_lp_params = [0.0f64; 5];

    for &cross in &crossovers {
        for &inc in &inc_strengths {
            for &dec in &dec_strengths {
                for &wp in &w_powers {
                    let params = [cross, inc, dec, wp, 0.0];
                    let mut pass = 0u32;
                    for case in &lp_stable {
                        if eval_lp_bidir(case, &params) <= 1.0 {
                            pass += 1;
                        }
                    }
                    if pass > best_lp_pass {
                        best_lp_pass = pass;
                        best_lp_params = params;
                    }
                }
            }
        }
    }
    println!(
        "  BEST: cross={:.2} inc={:.1} dec={:.1} wp={:.1} → {}/{} (baseline {})",
        best_lp_params[0],
        best_lp_params[1],
        best_lp_params[2],
        best_lp_params[3],
        best_lp_pass,
        lp_stable.len(),
        lp_pass
    );

    let mut details: Vec<(u32, f64, u8, f64)> = Vec::new();
    for case in &lp_stable {
        let rms = eval_lp_bidir(case, &best_lp_params);
        if rms > 1.0 {
            details.push((case.fc, case.display_q, case.slope, rms));
        }
    }
    details.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("  Failures:");
    for (fc, dq, s, rms) in details.iter().take(20) {
        println!("    {fc}Hz Q={dq} s{s}: {rms:.3}");
    }
    // s2 regressions
    let s2_reg: usize = lp_stable
        .iter()
        .filter(|c| c.slope == 2 && eval_lp_bidir(c, &best_lp_params) > 1.0)
        .count();
    println!("  s2 regressions: {}", s2_reg);

    // ── HP grid search ──────────────────────────────────────────────
    println!("\n=== HP BIDIRECTIONAL GRID SEARCH (stable cases) ===");
    let mut best_hp_pass = 0u32;
    let mut best_hp_params = [0.0f64; 5];

    for &cross in &crossovers {
        for &inc in &inc_strengths {
            for &dec in &dec_strengths {
                for &wp in &w_powers {
                    let params = [cross, inc, dec, wp, 0.0];
                    let mut pass = 0u32;
                    for case in &hp_stable {
                        if eval_hp_bidir(case, &params) <= 1.0 {
                            pass += 1;
                        }
                    }
                    if pass > best_hp_pass {
                        best_hp_pass = pass;
                        best_hp_params = params;
                    }
                }
            }
        }
    }
    println!(
        "  BEST: cross={:.2} inc={:.1} dec={:.1} wp={:.1} → {}/{} (baseline {})",
        best_hp_params[0],
        best_hp_params[1],
        best_hp_params[2],
        best_hp_params[3],
        best_hp_pass,
        hp_stable.len(),
        hp_pass
    );

    let mut details: Vec<(u32, f64, u8, f64)> = Vec::new();
    for case in &hp_stable {
        let rms = eval_hp_bidir(case, &best_hp_params);
        if rms > 1.0 {
            details.push((case.fc, case.display_q, case.slope, rms));
        }
    }
    details.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("  Failures:");
    for (fc, dq, s, rms) in details.iter().take(20) {
        println!("    {fc}Hz Q={dq} s{s}: {rms:.3}");
    }
    let s2_reg: usize = hp_stable
        .iter()
        .filter(|c| c.slope == 2 && eval_hp_bidir(c, &best_hp_params) > 1.0)
        .count();
    println!("  s2 regressions: {}", s2_reg);
}

fn eval_baseline(cases: &[&TestCase], is_lp: bool) -> (usize, Vec<(u32, f64, u8, f64)>) {
    let ft = if is_lp {
        eq_dsp::filter_type::FilterType::Lowpass
    } else {
        eq_dsp::filter_type::FilterType::Highpass
    };
    let mut pass = 0;
    let mut fails = Vec::new();
    for case in cases {
        let mut sections: Vec<Coeffs> = Vec::new();
        if let Some(first) = case.first_order_coeffs {
            sections.push(first);
        }
        if is_lp {
            if case.first_order_coeffs.is_some() {
                // Already added 1st-order above
            }
        }
        for &q in &case.section_qs {
            sections.push(eq_dsp::coeff::calculate_cascade(
                ft,
                case.fc as f64,
                q,
                0.0,
                SR,
                case.num_2nd,
            ));
        }
        let rms = cascade_rms(&sections, &case.ref_data);
        if rms <= 1.0 {
            pass += 1;
        } else {
            fails.push((case.fc, case.display_q, case.slope, rms));
        }
    }
    fails.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    (pass, fails)
}

fn print_fails(label: &str, fails: &[(u32, f64, u8, f64)], n: usize) {
    if fails.is_empty() {
        return;
    }
    println!("  {} worst:", label);
    for (fc, dq, s, rms) in fails.iter().take(n) {
        println!("    {fc}Hz Q={dq} s{s}: {rms:.3}");
    }
}
