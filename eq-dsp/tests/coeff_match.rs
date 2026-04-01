//! Test coefficient designs against Pro-Q 4 reference data.
//!
//! Reads detail CSVs from /tmp/eq-test2/details/ and compares our filter
//! response against the Pro-Q reference to find the best coefficient approach.

use std::f64::consts::PI;

/// Read Pro-Q reference response from a detail CSV.
fn read_ref_response(path: &str) -> Vec<(f64, f64)> {
    let content = std::fs::read_to_string(path).expect("failed to read CSV");
    let mut data = Vec::new();
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() >= 2 {
            let freq: f64 = fields[0].parse().unwrap_or(0.0);
            let ref_mag: f64 = fields[1].parse().unwrap_or(0.0);
            if freq > 10.0 && freq < 23000.0 {
                data.push((freq, ref_mag));
            }
        }
    }
    data
}

/// Compute magnitude response of a 1st-order filter at given frequency.
fn mag_db_1st(b0: f64, b1: f64, a1: f64, freq: f64, sr: f64) -> f64 {
    let w = 2.0 * PI * freq / sr;
    let cw = w.cos();
    let num = b0 * b0 + b1 * b1 + 2.0 * b0 * b1 * cw;
    let den = 1.0 + a1 * a1 + 2.0 * a1 * cw;
    if den < 1e-30 || num < 1e-30 {
        return 0.0;
    }
    10.0 * (num / den).log10()
}

/// Compute magnitude response of a biquad at given frequency.
fn mag_db_biquad(coeffs: &[f64; 6], freq: f64, sr: f64) -> f64 {
    let w = 2.0 * PI * freq / sr;
    let cw = w.cos();
    let sw = w.sin();
    let c2w = 2.0 * cw * cw - 1.0;
    let s2w = 2.0 * sw * cw;

    let nr = coeffs[3] + coeffs[4] * cw + coeffs[5] * c2w;
    let ni = -coeffs[4] * sw - coeffs[5] * s2w;
    let dr = coeffs[0] + coeffs[1] * cw + coeffs[2] * c2w;
    let di = -coeffs[1] * sw - coeffs[2] * s2w;

    let num_sq = nr * nr + ni * ni;
    let den_sq = dr * dr + di * di;
    if den_sq < 1e-30 || num_sq < 1e-30 {
        return 0.0;
    }
    10.0 * (num_sq / den_sq).log10()
}

/// Compute RMS error of a 1st-order filter against reference.
fn rms_1st(b0: f64, b1: f64, a1: f64, ref_data: &[(f64, f64)], sr: f64) -> f64 {
    let mut sum_sq = 0.0;
    let n = ref_data.len();
    for &(freq, ref_mag) in ref_data {
        let test_mag = mag_db_1st(b0, b1, a1, freq, sr);
        let diff = test_mag - ref_mag;
        sum_sq += diff * diff;
    }
    (sum_sq / n as f64).sqrt()
}

/// Vicanek matched 1st-order high shelf coefficients.
fn vicanek_1st(freq_hz: f64, g: f64, sr: f64, fm: f64) -> (f64, f64, f64) {
    let fc = freq_hz / (sr / 2.0);
    let fc_sq = fc * fc;
    let phi_m = 1.0 - (PI * fm).cos();

    let alpha = (2.0 / (PI * PI)) * (1.0 / (fm * fm) + 1.0 / (g * fc_sq)) - 1.0 / phi_m;
    let beta = (2.0 / (PI * PI)) * (1.0 / (fm * fm) + g / fc_sq) - 1.0 / phi_m;

    let a1 = -alpha / (1.0 + alpha + (1.0 + 2.0 * alpha).max(0.0).sqrt());
    let b_ratio = -beta / (1.0 + beta + (1.0 + 2.0 * beta).max(0.0).sqrt());

    let b0 = (1.0 + a1) / (1.0 + b_ratio);
    let b1 = b_ratio * b0;
    (b0, b1, a1)
}

/// BLT 1st-order high shelf with pre-warping.
fn blt_1st(freq_hz: f64, g: f64, sr: f64) -> (f64, f64, f64) {
    let w0 = (2.0 * PI * freq_hz / sr).clamp(1e-6, PI - 1e-6);
    let k = (w0 / 2.0).tan();
    let sqrt_g = g.sqrt();
    let a = k / sqrt_g;
    let b = k * sqrt_g;
    let inv = 1.0 / (1.0 + b);
    let a1 = (b - 1.0) * inv;
    let b0 = g * (1.0 + a) * inv;
    let b1 = g * (a - 1.0) * inv;
    (b0, b1, a1)
}

/// Brute-force optimal 1st-order high shelf (sweep a1, compute b0/b1 from DC=1 constraint).
fn optimal_1st(g: f64, ref_data: &[(f64, f64)], sr: f64) -> (f64, f64, f64, f64) {
    let mut best_a1 = 0.0;
    let mut best_b0 = 0.0;
    let mut best_rms = f64::MAX;

    // For each a1, find best b0 that minimizes RMS
    // DC constraint: b0 + b1 = 1 + a1, so b1 = (1+a1) - b0
    for a1_i in -500..100 {
        let a1 = a1_i as f64 / 1000.0;
        let dc = 1.0 + a1;
        if dc <= 0.0 {
            continue;
        }

        // Sweep b0 around the Vicanek range
        let b0_center = (dc + g * (1.0 - a1)) / 2.0; // DC+Nyquist midpoint
        let b0_lo = (b0_center * 0.5 * 1000.0) as i64;
        let b0_hi = (b0_center * 1.5 * 1000.0) as i64;
        for b0_i in b0_lo..=b0_hi {
            let b0 = b0_i as f64 / 1000.0;
            let b1 = dc - b0;
            let rms = rms_1st(b0, b1, a1, ref_data, sr);
            if rms < best_rms {
                best_rms = rms;
                best_a1 = a1;
                best_b0 = b0;
            }
        }
    }

    let b1 = (1.0 + best_a1) - best_b0;
    (best_b0, b1, best_a1, best_rms)
}

#[test]
fn compare_1st_order_approaches() {
    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q1_s0.csv";
    if !std::path::Path::new(csv_path).exists() {
        eprintln!("Skipping: detail CSV not found at {csv_path}");
        return;
    }

    let ref_data = read_ref_response(csv_path);
    let sr = 48000.0;
    let g = 10.0_f64.powf(6.0 / 20.0); // +6 dB

    println!("\n=== 1st-order high shelf: 10kHz +6dB Q=1 s0 ===\n");

    // Test different Vicanek matching points
    for &fm in &[0.3, 0.5, 0.7, 0.8, 0.9, 0.95] {
        let (b0, b1, a1) = vicanek_1st(10000.0, g, sr, fm);
        let rms = rms_1st(b0, b1, a1, &ref_data, sr);
        let nyq = (b0 - b1) / (1.0 - a1);
        println!(
            "Vicanek fm={fm:.2}: a1={a1:.6} b0={b0:.6} b1={b1:.6} Nyq={:.3}dB RMS={rms:.4}dB",
            20.0 * nyq.log10()
        );
    }

    // BLT
    {
        let (b0, b1, a1) = blt_1st(10000.0, g, sr);
        let rms = rms_1st(b0, b1, a1, &ref_data, sr);
        let nyq = (b0 - b1) / (1.0 - a1);
        println!(
            "BLT:            a1={a1:.6} b0={b0:.6} b1={b1:.6} Nyq={:.3}dB RMS={rms:.4}dB",
            20.0 * nyq.log10()
        );
    }

    // Optimal (brute force)
    let (b0, b1, a1, rms) = optimal_1st(g, &ref_data, sr);
    let nyq = (b0 - b1) / (1.0 - a1);
    println!(
        "Optimal:        a1={a1:.6} b0={b0:.6} b1={b1:.6} Nyq={:.3}dB RMS={rms:.4}dB",
        20.0 * nyq.log10()
    );

    // Show Pro-Q Nyquist gain
    let last = ref_data.last().unwrap();
    println!("\nPro-Q at {:.0}Hz: {:.3} dB", last.0, last.1);
    println!("Target gain: {:.3} dB", 6.0);

    // Show error profiles at key frequencies for best approaches
    println!("\nFrequency response comparison:");
    println!(
        "{:>8}  {:>8}  {:>10}  {:>10}  {:>10}",
        "Freq", "Pro-Q", "Vic0.8", "Vic0.95", "Optimal"
    );
    let (b0_v8, b1_v8, a1_v8) = vicanek_1st(10000.0, g, sr, 0.8);
    let (b0_v95, b1_v95, a1_v95) = vicanek_1st(10000.0, g, sr, 0.95);

    for target_freq in [
        100, 1000, 5000, 8000, 10000, 12000, 15000, 18000, 20000, 22000,
    ] {
        let (_, ref_mag) = *ref_data
            .iter()
            .min_by_key(|(f, _)| ((f - target_freq as f64).abs() * 100.0) as i64)
            .unwrap();
        let v8 = mag_db_1st(b0_v8, b1_v8, a1_v8, target_freq as f64, sr);
        let v95 = mag_db_1st(b0_v95, b1_v95, a1_v95, target_freq as f64, sr);
        let opt = mag_db_1st(b0, b1, a1, target_freq as f64, sr);
        println!(
            "{:8}  {:+8.3}  {:+10.3}  {:+10.3}  {:+10.3}",
            target_freq, ref_mag, v8, v95, opt
        );
    }
}

#[test]
fn compare_2nd_order_shelf_approaches() {
    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q1_s5.csv";
    if !std::path::Path::new(csv_path).exists() {
        eprintln!("Skipping: detail CSV not found at {csv_path}");
        return;
    }

    let ref_data = read_ref_response(csv_path);
    let sr = 48000.0;
    let g = 10.0_f64.powf(6.0 / 20.0);

    println!("\n=== 2nd-order shelf cascade: 10kHz +6dB Q=1 s5 (order=6, 3 biquads) ===\n");

    // Compute our current cascade response
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;

    let q_user = 1.0; // display Q=1 → internal Q = 1/√2 (but this is applied in band.rs)
    let internal_q = q_user / std::f64::consts::SQRT_2;
    let gain_per_section = 6.0 / 3.0; // 2 dB each
    let g_per = 10.0_f64.powf(gain_per_section / 20.0);

    // Butterworth Q values for 3 sections
    let bw_qs: Vec<f64> = (0..3)
        .map(|i| {
            let angle = PI * (2 * i + 1) as f64 / (4 * 3) as f64;
            0.5 / angle.cos()
        })
        .collect();

    println!("Butterworth Q values: {:?}", bw_qs);
    println!(
        "Gain per section: {:.2} dB (G={:.4})",
        gain_per_section, g_per
    );

    // Current approach: Vicanek matched for each biquad
    let sections: Vec<[f64; 6]> = bw_qs
        .iter()
        .map(|&q| coeff::calculate(FilterType::HighShelf, 10000.0, q, gain_per_section, sr))
        .collect();

    // Compute cascade response
    let mut rms = 0.0;
    let mut max_err = 0.0f64;
    let mut n = 0;
    println!(
        "\n{:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Freq", "Pro-Q", "FTS", "Diff", "Sec0", "Sec1", "Sec2"
    );
    for target_freq in [
        100, 1000, 5000, 7000, 8000, 9000, 10000, 11000, 12000, 14000, 16000, 18000, 20000, 22000,
    ] {
        let (_, ref_mag) = *ref_data
            .iter()
            .min_by_key(|(f, _)| ((f - target_freq as f64).abs() * 100.0) as i64)
            .unwrap();

        let mut cascade_db = 0.0;
        let mut per_section = Vec::new();
        for s in &sections {
            let db = mag_db_biquad(s, target_freq as f64, sr);
            cascade_db += db;
            per_section.push(db);
        }

        let diff = cascade_db - ref_mag;
        rms += diff * diff;
        max_err = max_err.max(diff.abs());
        n += 1;

        println!(
            "{:8}  {:+8.3}  {:+8.3}  {:+8.4}  {:+8.4}  {:+8.4}  {:+8.4}",
            target_freq, ref_mag, cascade_db, diff, per_section[0], per_section[1], per_section[2]
        );
    }
    let rms_err = (rms / n as f64).sqrt();
    println!(
        "\nCascade RMS error: {:.4} dB, Max error: {:.4} dB",
        rms_err, max_err
    );

    // Now try RBJ cookbook shelf for comparison
    println!("\n--- RBJ cookbook shelf comparison ---");
    let rbj_sections: Vec<[f64; 6]> = bw_qs
        .iter()
        .map(|&q| rbj_high_shelf(10000.0, q, gain_per_section, sr))
        .collect();

    let mut rms_rbj = 0.0;
    let mut n = 0;
    println!(
        "\n{:>8}  {:>8}  {:>8}  {:>8}",
        "Freq", "Pro-Q", "RBJ", "Diff"
    );
    for target_freq in [
        100, 1000, 5000, 8000, 10000, 12000, 15000, 18000, 20000, 22000,
    ] {
        let (_, ref_mag) = *ref_data
            .iter()
            .min_by_key(|(f, _)| ((f - target_freq as f64).abs() * 100.0) as i64)
            .unwrap();
        let mut cascade_db = 0.0;
        for s in &rbj_sections {
            cascade_db += mag_db_biquad(s, target_freq as f64, sr);
        }
        let diff = cascade_db - ref_mag;
        rms_rbj += diff * diff;
        n += 1;
        println!(
            "{:8}  {:+8.3}  {:+8.3}  {:+8.4}",
            target_freq, ref_mag, cascade_db, diff
        );
    }
    println!(
        "RBJ cascade RMS error: {:.4} dB",
        (rms_rbj / n as f64).sqrt()
    );
}

#[test]
fn find_optimal_shelf_matching() {
    // For the s5 case (order=6, 3 biquads), explore what corner matching
    // target gives the best cascade match to Pro-Q.
    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q1_s5.csv";
    if !std::path::Path::new(csv_path).exists() {
        eprintln!("Skipping: detail CSV not found");
        return;
    }

    let ref_data = read_ref_response(csv_path);
    let sr = 48000.0;
    let gain_per = 2.0; // dB
    let g_per = 10.0_f64.powf(gain_per / 20.0);

    let bw_qs: Vec<f64> = (0..3)
        .map(|i| {
            let angle = PI * (2 * i + 1) as f64 / (4 * 3) as f64;
            0.5 / angle.cos()
        })
        .collect();

    println!("\n=== Sweep corner gain target ===");
    println!("Standard target: |H(w0)|^2 = G = {:.4}", g_per);

    // Sweep the corner gain target multiplier
    for mult_pct in [95, 97, 98, 99, 100, 101, 102, 103, 105, 110] {
        let mult = mult_pct as f64 / 100.0;
        let sections: Vec<[f64; 6]> = bw_qs
            .iter()
            .map(|&q| high_shelf_2_with_corner_mult(10000.0, q, gain_per, sr, mult))
            .collect();

        let mut rms = 0.0;
        let mut n = 0;
        for &(freq, ref_mag) in &ref_data {
            let mut cascade_db = 0.0;
            for s in &sections {
                cascade_db += mag_db_biquad(s, freq, sr);
            }
            let diff = cascade_db - ref_mag;
            rms += diff * diff;
            n += 1;
        }
        let rms = (rms / n as f64).sqrt();
        println!("corner_mult={mult:.2}: cascade RMS={rms:.4} dB");
    }

    // Sweep Nyquist gain target multiplier
    println!("\n=== Sweep Nyquist gain target ===");
    for nyq_mult_1000 in [1000, 1002, 1005, 1007, 1010, 1012, 1015, 1020, 1025, 1030] {
        let nyq_mult = nyq_mult_1000 as f64 / 1000.0;
        let sections: Vec<[f64; 6]> = bw_qs
            .iter()
            .map(|&q| high_shelf_2_with_nyq_mult(10000.0, q, gain_per, sr, nyq_mult))
            .collect();

        let mut rms = 0.0;
        let mut n = 0;
        for &(freq, ref_mag) in &ref_data {
            let mut cascade_db = 0.0;
            for s in &sections {
                cascade_db += mag_db_biquad(s, freq, sr);
            }
            rms += (cascade_db - ref_mag).powi(2);
            n += 1;
        }
        println!(
            "nyq_mult={nyq_mult:.2}: cascade RMS={:.4} dB",
            (rms / n as f64).sqrt()
        );
    }

    // Also try shifting the corner frequency
    println!("\n=== Sweep corner frequency offset ===");
    for freq_shift_pct in [95, 97, 98, 99, 100, 101, 102, 103, 105] {
        let freq = 10000.0 * freq_shift_pct as f64 / 100.0;
        let sections: Vec<[f64; 6]> = bw_qs
            .iter()
            .map(|&q| {
                use eq_dsp::coeff;
                use eq_dsp::filter_type::FilterType;
                coeff::calculate(FilterType::HighShelf, freq, q, gain_per, sr)
            })
            .collect();

        let mut rms = 0.0;
        let mut n = 0;
        for &(freq_bin, ref_mag) in &ref_data {
            let mut cascade_db = 0.0;
            for s in &sections {
                cascade_db += mag_db_biquad(s, freq_bin, sr);
            }
            rms += (cascade_db - ref_mag).powi(2);
            n += 1;
        }
        println!(
            "freq_offset={freq_shift_pct}%: cascade RMS={:.4} dB",
            (rms / n as f64).sqrt()
        );
    }
}

/// Modified Vicanek matched shelf with adjustable corner gain target.
fn high_shelf_2_with_corner_mult(
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sr: f64,
    corner_mult: f64,
) -> [f64; 6] {
    let w0 = (2.0 * PI * freq_hz / sr).clamp(1e-6, PI - 1e-6);
    let g = 10.0_f64.powf(gain_db / 20.0);
    let a = g.sqrt();
    let sqrt_a = a.sqrt();

    // Impulse invariance poles
    let b_param = sqrt_a / (2.0 * q);
    let c_param = sqrt_a;
    let t = (-b_param * w0).exp();
    let (a1, a2) = if b_param <= c_param {
        let a1 = -2.0 * t * ((c_param * c_param - b_param * b_param).sqrt() * w0).cos();
        (a1, t * t)
    } else {
        let a1 = -2.0 * t * ((b_param * b_param - c_param * c_param).sqrt() * w0).cosh();
        (a1, t * t)
    };

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;

    let phi0 = |w: f64| 0.5 + 0.5 * w.cos();
    let phi1 = |w: f64| 0.5 - 0.5 * w.cos();
    let p0 = phi0(w0);
    let p1 = phi1(w0);

    // DC = 1
    let b0_big = a0_big;

    // Nyquist from analog
    let f_ny = PI / w0;
    let f_ny_sq = f_ny * f_ny;
    let a_over_q_sq = a * f_ny_sq / (q * q);
    let num_ny = (1.0 - a * f_ny_sq).powi(2) + a_over_q_sq;
    let den_ny = (a - f_ny_sq).powi(2) + a_over_q_sq;
    let h_ny_sq = if den_ny.abs() > 1e-30 {
        a * a * num_ny / den_ny
    } else {
        g
    };
    let b1_big = a1_big * h_ny_sq;

    // Corner: |H(w0)|^2 = G * corner_mult (adjustable!)
    let target = g * corner_mult * (a0_big * p0 + a1_big * p1 + a2_big * 4.0 * p0 * p1);
    let b2_big = (target - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    // Extract b coefficients from magnitude-squared
    let b0_sq = b0_big.max(0.0);
    let b1_sq = b1_big.max(0.0);
    let b0_sqrt = b0_sq.sqrt();
    let b1_sqrt = b1_sq.sqrt();
    let w = (b0_sqrt + b1_sqrt) / 2.0;

    let (b0, b1, b2) = if b2_big.abs() < 1e-30 {
        let b0 = w;
        let b1 = b0_sqrt - b0;
        (b0, b1, 0.0)
    } else {
        let b0 = (w + (w * w + b2_big).max(0.0).sqrt()) / 2.0;
        let b0 = b0.max(1e-30);
        let b1 = (b0_sqrt - b1_sqrt) / 2.0;
        let b2 = -b2_big / (4.0 * b0);
        (b0, b1, b2)
    };

    [1.0, a1, a2, b0, b1, b2]
}

/// Modified Vicanek matched shelf with adjustable Nyquist gain target.
fn high_shelf_2_with_nyq_mult(
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    sr: f64,
    nyq_mult: f64,
) -> [f64; 6] {
    let w0 = (2.0 * PI * freq_hz / sr).clamp(1e-6, PI - 1e-6);
    let g = 10.0_f64.powf(gain_db / 20.0);
    let a = g.sqrt();
    let sqrt_a = a.sqrt();

    let b_param = sqrt_a / (2.0 * q);
    let c_param = sqrt_a;
    let t = (-b_param * w0).exp();
    let (a1, a2) = if b_param <= c_param {
        (
            -2.0 * t * ((c_param * c_param - b_param * b_param).sqrt() * w0).cos(),
            t * t,
        )
    } else {
        (
            -2.0 * t * ((b_param * b_param - c_param * c_param).sqrt() * w0).cosh(),
            t * t,
        )
    };

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;
    let phi0 = |w: f64| 0.5 + 0.5 * w.cos();
    let phi1 = |w: f64| 0.5 - 0.5 * w.cos();
    let p0 = phi0(w0);
    let p1 = phi1(w0);

    let b0_big = a0_big; // DC = 1

    // Nyquist from analog * multiplier
    let f_ny = PI / w0;
    let f_ny_sq = f_ny * f_ny;
    let a_over_q_sq = a * f_ny_sq / (q * q);
    let num_ny = (1.0 - a * f_ny_sq).powi(2) + a_over_q_sq;
    let den_ny = (a - f_ny_sq).powi(2) + a_over_q_sq;
    let h_ny_sq = if den_ny.abs() > 1e-30 {
        a * a * num_ny / den_ny * nyq_mult
    } else {
        g * nyq_mult
    };
    let b1_big = a1_big * h_ny_sq;

    // Corner = G
    let target = g * (a0_big * p0 + a1_big * p1 + a2_big * 4.0 * p0 * p1);
    let b2_big = (target - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let b0_sq = b0_big.max(0.0);
    let b1_sq = b1_big.max(0.0);
    let b0_sqrt = b0_sq.sqrt();
    let b1_sqrt = b1_sq.sqrt();
    let w = (b0_sqrt + b1_sqrt) / 2.0;
    let (b0, b1, b2) = if b2_big.abs() < 1e-30 {
        (w, b0_sqrt - w, 0.0)
    } else {
        let b0 = (w + (w * w + b2_big).max(0.0).sqrt()) / 2.0;
        let b0 = b0.max(1e-30);
        let b1 = (b0_sqrt - b1_sqrt) / 2.0;
        let b2 = -b2_big / (4.0 * b0);
        (b0, b1, b2)
    };

    [1.0, a1, a2, b0, b1, b2]
}

/// Compute cascade RMS error vs reference data for a set of biquad sections.
fn cascade_rms(sections: &[[f64; 6]], ref_data: &[(f64, f64)], sr: f64) -> f64 {
    let mut sum_sq = 0.0;
    let mut n = 0;
    for &(freq, ref_mag) in ref_data {
        let mut cascade_db = 0.0;
        for s in sections {
            cascade_db += mag_db_biquad(s, freq, sr);
        }
        let diff = cascade_db - ref_mag;
        sum_sq += diff * diff;
        n += 1;
    }
    (sum_sq / n as f64).sqrt()
}

#[test]
fn sweep_cascade_params_s5() {
    // Sweep cascade parameters for s5 (order=6, 3 biquads) to find optimal
    // Q scaling and gain distribution that matches Pro-Q.
    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q1_s5.csv";
    if !std::path::Path::new(csv_path).exists() {
        eprintln!("Skipping: detail CSV not found");
        return;
    }

    let ref_data = read_ref_response(csv_path);
    let sr = 48000.0;

    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;

    let bw_qs: Vec<f64> = (0..3)
        .map(|i| {
            let angle = PI * (2 * i + 1) as f64 / (4 * 3) as f64;
            0.5 / angle.cos()
        })
        .collect();

    println!("\n=== Current baseline (equal gain, Butterworth Q) ===");
    let gain_per = 6.0 / 3.0;
    let sections: Vec<[f64; 6]> = bw_qs
        .iter()
        .map(|&q| coeff::calculate(FilterType::HighShelf, 10000.0, q, gain_per, sr))
        .collect();
    let baseline_rms = cascade_rms(&sections, &ref_data, sr);
    println!("Baseline: RMS={baseline_rms:.6} dB");

    // 1. Sweep uniform Q scale factor applied to all sections
    println!("\n=== Sweep uniform Q scale ===");
    let mut best_q_scale = 1.0;
    let mut best_q_rms = baseline_rms;
    for q_scale_pct in 80..=130 {
        let q_scale = q_scale_pct as f64 / 100.0;
        let sections: Vec<[f64; 6]> = bw_qs
            .iter()
            .map(|&q| coeff::calculate(FilterType::HighShelf, 10000.0, q * q_scale, gain_per, sr))
            .collect();
        let rms = cascade_rms(&sections, &ref_data, sr);
        if rms < best_q_rms {
            best_q_rms = rms;
            best_q_scale = q_scale;
        }
    }
    println!("Best uniform Q scale: {best_q_scale:.2} → RMS={best_q_rms:.6} dB");

    // 2. Sweep gain distribution: first section gets more/less, rest share remainder
    println!("\n=== Sweep gain distribution (sec0 vs sec1+sec2) ===");
    let mut best_gain0 = gain_per;
    let mut best_gain_rms = baseline_rms;
    for gain0_pct in 50..=200 {
        let gain0 = gain_per * gain0_pct as f64 / 100.0;
        let remaining = 6.0 - gain0;
        let gain_rest = remaining / 2.0;
        if gain_rest <= 0.0 {
            continue;
        }
        let sections: Vec<[f64; 6]> = bw_qs
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let g = if i == 0 { gain0 } else { gain_rest };
                coeff::calculate(FilterType::HighShelf, 10000.0, q, g, sr)
            })
            .collect();
        let rms = cascade_rms(&sections, &ref_data, sr);
        if rms < best_gain_rms {
            best_gain_rms = rms;
            best_gain0 = gain0;
        }
    }
    println!(
        "Best sec0 gain: {best_gain0:.3} dB (rest: {:.3} each) → RMS={best_gain_rms:.6} dB",
        (6.0 - best_gain0) / 2.0
    );

    // 3. Sweep per-section Q independently: [q0_scale, q1_scale, q2_scale]
    println!("\n=== Sweep per-section Q scales (coarse) ===");
    let mut best_qs = [1.0, 1.0, 1.0];
    let mut best_per_q_rms = baseline_rms;
    for q0 in (80..=130).step_by(5) {
        for q1 in (80..=130).step_by(5) {
            for q2 in (80..=130).step_by(5) {
                let scales = [q0 as f64 / 100.0, q1 as f64 / 100.0, q2 as f64 / 100.0];
                let sections: Vec<[f64; 6]> = bw_qs
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| {
                        coeff::calculate(
                            FilterType::HighShelf,
                            10000.0,
                            q * scales[i],
                            gain_per,
                            sr,
                        )
                    })
                    .collect();
                let rms = cascade_rms(&sections, &ref_data, sr);
                if rms < best_per_q_rms {
                    best_per_q_rms = rms;
                    best_qs = scales;
                }
            }
        }
    }
    println!(
        "Best per-section Q: [{:.2}, {:.2}, {:.2}] → RMS={best_per_q_rms:.6} dB",
        best_qs[0], best_qs[1], best_qs[2]
    );

    // Fine-tune around best
    let mut best_fine_qs = best_qs;
    let mut best_fine_rms = best_per_q_rms;
    for dq0 in -10..=10 {
        for dq1 in -10..=10 {
            for dq2 in -10..=10 {
                let scales = [
                    best_qs[0] + dq0 as f64 / 100.0,
                    best_qs[1] + dq1 as f64 / 100.0,
                    best_qs[2] + dq2 as f64 / 100.0,
                ];
                let sections: Vec<[f64; 6]> = bw_qs
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| {
                        coeff::calculate(
                            FilterType::HighShelf,
                            10000.0,
                            q * scales[i],
                            gain_per,
                            sr,
                        )
                    })
                    .collect();
                let rms = cascade_rms(&sections, &ref_data, sr);
                if rms < best_fine_rms {
                    best_fine_rms = rms;
                    best_fine_qs = scales;
                }
            }
        }
    }
    println!(
        "Fine-tuned per-section Q: [{:.2}, {:.2}, {:.2}] → RMS={best_fine_rms:.6} dB",
        best_fine_qs[0], best_fine_qs[1], best_fine_qs[2]
    );

    // 4. Combined: best Q scales + gain distribution
    println!("\n=== Combined Q + gain sweep ===");
    let mut best_combo_rms = best_fine_rms;
    let mut best_combo = (best_fine_qs, [gain_per; 3]);
    for gain0_pct in 80..=120 {
        let gain0 = gain_per * gain0_pct as f64 / 100.0;
        let remaining = 6.0 - gain0;
        for gain1_pct in 80..=120 {
            let gain1 = (remaining / 2.0) * gain1_pct as f64 / 100.0;
            let gain2 = 6.0 - gain0 - gain1;
            if gain2 <= 0.0 {
                continue;
            }
            let gains = [gain0, gain1, gain2];
            let sections: Vec<[f64; 6]> = bw_qs
                .iter()
                .enumerate()
                .map(|(i, &q)| {
                    coeff::calculate(
                        FilterType::HighShelf,
                        10000.0,
                        q * best_fine_qs[i],
                        gains[i],
                        sr,
                    )
                })
                .collect();
            let rms = cascade_rms(&sections, &ref_data, sr);
            if rms < best_combo_rms {
                best_combo_rms = rms;
                best_combo = (best_fine_qs, gains);
            }
        }
    }
    println!("Best combo: Q=[{:.2}, {:.2}, {:.2}] Gain=[{:.3}, {:.3}, {:.3}] → RMS={best_combo_rms:.6} dB",
             best_combo.0[0], best_combo.0[1], best_combo.0[2],
             best_combo.1[0], best_combo.1[1], best_combo.1[2]);

    // Show frequency comparison for best combo
    println!(
        "\n{:>8}  {:>8}  {:>8}  {:>8}",
        "Freq", "Pro-Q", "Best", "Diff"
    );
    let sections: Vec<[f64; 6]> = bw_qs
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            coeff::calculate(
                FilterType::HighShelf,
                10000.0,
                q * best_combo.0[i],
                best_combo.1[i],
                sr,
            )
        })
        .collect();
    for target_freq in [
        100, 1000, 5000, 7000, 8000, 9000, 10000, 11000, 12000, 14000, 16000, 18000, 20000, 22000,
    ] {
        let (_, ref_mag) = *ref_data
            .iter()
            .min_by_key(|(f, _)| ((f - target_freq as f64).abs() * 100.0) as i64)
            .unwrap();
        let mut cascade_db = 0.0;
        for s in &sections {
            cascade_db += mag_db_biquad(s, target_freq as f64, sr);
        }
        println!(
            "{:8}  {:+8.3}  {:+8.3}  {:+8.4}",
            target_freq,
            ref_mag,
            cascade_db,
            cascade_db - ref_mag
        );
    }
}

#[test]
fn sweep_cascade_multi_scenario() {
    // Test best cascade params across multiple scenarios to see if they generalize.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let scenarios = [
        ("high_shelf_10000hz_+6db_q1_s5", 10000.0, 6.0, 1.0, 6),
        ("high_shelf_10000hz_-12db_q1_s5", 10000.0, -12.0, 1.0, 6),
        ("high_shelf_1000hz_+6db_q1_s5", 1000.0, 6.0, 1.0, 6),
        ("high_shelf_10000hz_+6db_q4_s5", 10000.0, 6.0, 4.0, 6),
        ("high_shelf_10000hz_+6db_q1_s8", 10000.0, 6.0, 1.0, 12),
        ("high_shelf_10000hz_+6db_q1_s2", 10000.0, 6.0, 1.0, 3),
    ];

    for (name, freq, gain_db, q_display, order) in scenarios {
        let csv_path = format!("/tmp/eq-test2/details/{name}.csv");
        if !std::path::Path::new(&csv_path).exists() {
            continue;
        }
        let ref_data = read_ref_response(&csv_path);

        // Reconstruct cascade exactly as band.rs does it
        let internal_q = q_display / std::f64::consts::SQRT_2;
        let q_user = internal_q * std::f64::consts::SQRT_2; // = q_display
        let effective_gain = gain_db;

        let has_first_order = order % 2 == 1;
        let num_2nd: usize = order / 2;

        // Gain distribution
        let (gain_1st, gain_2nd) = if order == 3 {
            (effective_gain * 0.20, effective_gain * 0.80)
        } else {
            let gain_per_pole = effective_gain / order as f64;
            (gain_per_pole, gain_per_pole * 2.0)
        };

        let bw_qs: Vec<f64> = (0..num_2nd)
            .map(|i| {
                let angle = PI * (2 * i + 1) as f64 / (4 * num_2nd) as f64;
                0.5 / angle.cos()
            })
            .collect();

        let mut sections = Vec::new();
        if has_first_order {
            sections.push(coeff::high_shelf_1(freq, gain_1st, sr));
        }
        for i in 0..num_2nd {
            let is_last = i == num_2nd - 1;
            let is_second_last = i == num_2nd.saturating_sub(2) && num_2nd > 1;
            let bw_q = bw_qs[i];
            let q_section = if is_last {
                let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                let scale = if q_user > 1.0 {
                    1.0 + (q_user.ln() * 1.03 * blend)
                } else {
                    q_user.powf(blend * 0.75)
                };
                bw_q * scale
            } else if is_second_last && order >= 6 {
                let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                let scale = if q_user > 1.0 {
                    1.0 + (q_user.ln() * 0.3 * blend)
                } else {
                    1.0
                };
                bw_q * scale
            } else {
                bw_q
            };
            sections.push(coeff::calculate(
                FilterType::HighShelf,
                freq,
                q_section,
                gain_2nd,
                sr,
            ));
        }

        let rms = cascade_rms(&sections, &ref_data, sr);
        println!("{name}: RMS={rms:.4} dB ({} sections)", sections.len());
    }
}

#[test]
fn sweep_last_q_scale_multi() {
    // For each scenario, find the optimal Q scale for the LAST biquad section.
    // This tests whether a simple "reduce last Q" rule generalizes.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let scenarios: Vec<(&str, f64, f64, f64, usize)> = vec![
        ("high_shelf_10000hz_+6db_q1_s5", 10000.0, 6.0, 1.0, 6),
        ("high_shelf_10000hz_-12db_q1_s5", 10000.0, -12.0, 1.0, 6),
        ("high_shelf_1000hz_+6db_q1_s5", 1000.0, 6.0, 1.0, 6),
        ("high_shelf_5000hz_+6db_q1_s5", 5000.0, 6.0, 1.0, 6),
        ("high_shelf_10000hz_+6db_q0.5_s5", 10000.0, 6.0, 0.5, 6),
        ("high_shelf_10000hz_+6db_q4_s5", 10000.0, 6.0, 4.0, 6),
        ("high_shelf_10000hz_+6db_q10_s5", 10000.0, 6.0, 10.0, 6),
        ("high_shelf_10000hz_+6db_q1_s8", 10000.0, 6.0, 1.0, 12),
        ("high_shelf_10000hz_-12db_q1_s8", 10000.0, -12.0, 1.0, 12),
        ("high_shelf_5000hz_+6db_q1_s8", 5000.0, 6.0, 1.0, 12),
    ];

    println!(
        "\n{:<45} {:>8} {:>10} {:>10}",
        "Scenario", "Baseline", "BestLastQ", "BestRMS"
    );
    for (name, freq, gain_db, q_display, order) in scenarios {
        let csv_path = format!("/tmp/eq-test2/details/{name}.csv");
        if !std::path::Path::new(&csv_path).exists() {
            continue;
        }
        let ref_data = read_ref_response(&csv_path);

        let num_2nd: usize = order / 2;
        let gain_per = gain_db / num_2nd as f64; // equal gain per biquad (even order only for this test)

        let bw_qs: Vec<f64> = (0..num_2nd)
            .map(|i| {
                let angle = PI * (2 * i + 1) as f64 / (4 * num_2nd) as f64;
                0.5 / angle.cos()
            })
            .collect();

        // Baseline
        let sections: Vec<[f64; 6]> = bw_qs
            .iter()
            .map(|&q| coeff::calculate(FilterType::HighShelf, freq, q, gain_per, sr))
            .collect();
        let baseline = cascade_rms(&sections, &ref_data, sr);

        // Sweep last Q scale
        let mut best_scale = 1.0;
        let mut best_rms = baseline;
        for s in 50..=150 {
            let scale = s as f64 / 100.0;
            let sections: Vec<[f64; 6]> = bw_qs
                .iter()
                .enumerate()
                .map(|(i, &q)| {
                    let qs = if i == num_2nd - 1 { q * scale } else { q };
                    coeff::calculate(FilterType::HighShelf, freq, qs, gain_per, sr)
                })
                .collect();
            let rms = cascade_rms(&sections, &ref_data, sr);
            if rms < best_rms {
                best_rms = rms;
                best_scale = scale;
            }
        }

        println!(
            "{:<45} {:>8.4} {:>10.2} {:>10.4}",
            name, baseline, best_scale, best_rms
        );
    }
}

#[test]
fn analyze_q4_s5_error() {
    // Deep dive: what does the Q=4 s5 error look like? Where is the mismatch?
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q4_s5.csv";
    if !std::path::Path::new(&csv_path).exists() {
        eprintln!("Skipping: no Q=4 s5 CSV");
        return;
    }
    let ref_data = read_ref_response(csv_path);

    let freq = 10000.0;
    let gain_db = 6.0;
    let order = 6usize;
    let num_2nd = 3usize;
    let gain_per = gain_db / num_2nd as f64;

    let bw_qs: Vec<f64> = (0..num_2nd)
        .map(|i| {
            let angle = PI * (2 * i + 1) as f64 / (4 * num_2nd) as f64;
            0.5 / angle.cos()
        })
        .collect();

    // Current band.rs behavior for Q=4:
    // internal_q = 4/√2 = 2.828, q_user = internal_q * √2 = 4.0
    let q_user = 4.0_f64;
    let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0); // = 0.7
    println!("blend = {blend:.2}");
    println!("Butterworth Qs: {:?}", bw_qs);

    // Last section scale
    let last_scale = 1.0 + (q_user.ln() * 1.03 * blend);
    println!("Last section scale: {last_scale:.4}");
    println!("Last section Q: {:.4}", bw_qs[2] * last_scale);

    // Second-to-last scale
    let s2l_scale = 1.0 + (q_user.ln() * 0.3 * blend);
    println!("Second-to-last scale: {s2l_scale:.4}");
    println!("Second-to-last Q: {:.4}", bw_qs[1] * s2l_scale);

    // Build current cascade
    let q_secs: Vec<f64> = bw_qs
        .iter()
        .enumerate()
        .map(|(i, &bw_q)| {
            if i == num_2nd - 1 {
                bw_q * last_scale
            } else if i == num_2nd - 2 {
                bw_q * s2l_scale
            } else {
                bw_q
            }
        })
        .collect();

    let sections: Vec<[f64; 6]> = q_secs
        .iter()
        .map(|&q| coeff::calculate(FilterType::HighShelf, freq, q, gain_per, sr))
        .collect();
    let current_rms = cascade_rms(&sections, &ref_data, sr);
    println!("\nCurrent cascade Q values: {:?}", q_secs);
    println!("Current RMS: {current_rms:.4} dB");

    // Pure Butterworth
    let pure_sections: Vec<[f64; 6]> = bw_qs
        .iter()
        .map(|&q| coeff::calculate(FilterType::HighShelf, freq, q, gain_per, sr))
        .collect();
    let pure_rms = cascade_rms(&pure_sections, &ref_data, sr);
    println!("Pure Butterworth RMS: {pure_rms:.4} dB");

    // Show frequency profile for both
    println!(
        "\n{:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "Freq", "Pro-Q", "Current", "PureBW", "CurrDiff"
    );
    for target_freq in [
        100, 1000, 5000, 7000, 8000, 9000, 9500, 10000, 10500, 11000, 12000, 14000, 16000, 18000,
        20000, 22000,
    ] {
        let (_, ref_mag) = *ref_data
            .iter()
            .min_by_key(|(f, _)| ((f - target_freq as f64).abs() * 100.0) as i64)
            .unwrap();
        let mut cur_db = 0.0;
        let mut pure_db = 0.0;
        for s in &sections {
            cur_db += mag_db_biquad(s, target_freq as f64, sr);
        }
        for s in &pure_sections {
            pure_db += mag_db_biquad(s, target_freq as f64, sr);
        }
        println!(
            "{:8}  {:+8.3}  {:+8.3}  {:+8.3}  {:+8.4}",
            target_freq,
            ref_mag,
            cur_db,
            pure_db,
            cur_db - ref_mag
        );
    }

    // Sweep all 3 Q scales for Q=4
    println!("\n=== Full 3D Q sweep for Q=4 s5 ===");
    let mut best_qs = [1.0; 3];
    let mut best_rms = f64::MAX;
    for q0 in (80..=150).step_by(5) {
        for q1 in (80..=200).step_by(5) {
            for q2 in (80..=250).step_by(5) {
                let scales = [q0 as f64 / 100.0, q1 as f64 / 100.0, q2 as f64 / 100.0];
                let sections: Vec<[f64; 6]> = bw_qs
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| {
                        coeff::calculate(FilterType::HighShelf, freq, q * scales[i], gain_per, sr)
                    })
                    .collect();
                let rms = cascade_rms(&sections, &ref_data, sr);
                if rms < best_rms {
                    best_rms = rms;
                    best_qs = scales;
                }
            }
        }
    }
    println!(
        "Best Q scales: [{:.2}, {:.2}, {:.2}] → RMS={best_rms:.4} dB",
        best_qs[0], best_qs[1], best_qs[2]
    );

    // Also check if the error is from the biquad coefficient design, not the cascade
    // Try equal gain with best Q scales
    let sections: Vec<[f64; 6]> = bw_qs
        .iter()
        .enumerate()
        .map(|(i, &q)| coeff::calculate(FilterType::HighShelf, freq, q * best_qs[i], gain_per, sr))
        .collect();
    println!(
        "\n{:>8}  {:>8}  {:>8}  {:>8}",
        "Freq", "Pro-Q", "BestQ", "Diff"
    );
    for target_freq in [
        100, 1000, 5000, 8000, 9000, 9500, 10000, 10500, 11000, 12000, 14000, 20000, 22000,
    ] {
        let (_, ref_mag) = *ref_data
            .iter()
            .min_by_key(|(f, _)| ((f - target_freq as f64).abs() * 100.0) as i64)
            .unwrap();
        let mut cascade_db = 0.0;
        for s in &sections {
            cascade_db += mag_db_biquad(s, target_freq as f64, sr);
        }
        println!(
            "{:8}  {:+8.3}  {:+8.3}  {:+8.4}",
            target_freq,
            ref_mag,
            cascade_db,
            cascade_db - ref_mag
        );
    }
}

#[test]
fn sweep_optimal_q_scales_all_s5() {
    // For every available s5 scenario, find the optimal per-section Q scales.
    // Goal: find a pattern that can be expressed as a formula.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let base_dir = "/tmp/eq-test2/details";
    let files: Vec<String> = std::fs::read_dir(base_dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_str()?.to_string();
            if name.contains("_s5.csv") && name.starts_with("high_shelf") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    let mut results = Vec::new();
    println!(
        "\n{:<50} {:>6} {:>6} {:>6} {:>8} {:>8}",
        "Scenario", "Q0", "Q1", "Q2", "Base", "Best"
    );

    for file in &files {
        let csv_path = format!("{base_dir}/{file}");
        let ref_data = read_ref_response(&csv_path);

        // Parse scenario name: high_shelf_{freq}hz_{gain}db_q{q}_s5.csv
        let name = file.strip_suffix(".csv").unwrap();
        let parts: Vec<&str> = name.split('_').collect();
        // high_shelf_10000hz_+6db_q1_s5
        let freq_str = parts[2].strip_suffix("hz").unwrap_or("10000");
        let freq: f64 = freq_str.parse().unwrap_or(10000.0);
        let gain_str = parts[3].strip_suffix("db").unwrap_or("+6");
        let gain_db: f64 = gain_str.parse().unwrap_or(6.0);
        let q_str = parts[4].strip_prefix('q').unwrap_or("1");
        let _q_display: f64 = q_str.parse().unwrap_or(1.0);

        let num_2nd = 3usize; // s5 = order 6 = 3 biquads
        let gain_per = gain_db / num_2nd as f64;

        let bw_qs: Vec<f64> = (0..num_2nd)
            .map(|i| {
                let angle = PI * (2 * i + 1) as f64 / (4 * num_2nd) as f64;
                0.5 / angle.cos()
            })
            .collect();

        // Baseline
        let sections: Vec<[f64; 6]> = bw_qs
            .iter()
            .map(|&q| coeff::calculate(FilterType::HighShelf, freq, q, gain_per, sr))
            .collect();
        let baseline = cascade_rms(&sections, &ref_data, sr);

        // Coarse sweep
        let mut best_qs = [1.0; 3];
        let mut best_rms = baseline;
        for q0 in (70..=140).step_by(5) {
            for q1 in (70..=200).step_by(5) {
                for q2 in (50..=250).step_by(5) {
                    let scales = [q0 as f64 / 100.0, q1 as f64 / 100.0, q2 as f64 / 100.0];
                    let sections: Vec<[f64; 6]> = bw_qs
                        .iter()
                        .enumerate()
                        .map(|(i, &q)| {
                            coeff::calculate(
                                FilterType::HighShelf,
                                freq,
                                q * scales[i],
                                gain_per,
                                sr,
                            )
                        })
                        .collect();
                    let rms = cascade_rms(&sections, &ref_data, sr);
                    if rms < best_rms {
                        best_rms = rms;
                        best_qs = scales;
                    }
                }
            }
        }

        println!(
            "{:<50} {:>6.2} {:>6.2} {:>6.2} {:>8.4} {:>8.4}",
            name, best_qs[0], best_qs[1], best_qs[2], baseline, best_rms
        );
        results.push((
            name.to_string(),
            freq,
            gain_db,
            _q_display,
            best_qs,
            baseline,
            best_rms,
        ));
    }

    // Print sorted by Q for analysis
    println!("\n=== Sorted by Q display value ===");
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| {
        a.3.partial_cmp(&b.3)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    for (name, _freq, _gain, q, qs, base, best) in &sorted {
        println!(
            "Q={q:<4}  [{:.2}, {:.2}, {:.2}]  base={base:.4}  best={best:.4}  {name}",
            qs[0], qs[1], qs[2]
        );
    }
}

#[test]
fn sweep_s2_gain_split_and_q() {
    // For s2 (order=3, 1st-order + biquad), sweep gain split and biquad Q.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let scenarios = [
        ("high_shelf_10000hz_+6db_q1_s2", 10000.0, 6.0, 1.0),
        ("high_shelf_10000hz_-12db_q1_s2", 10000.0, -12.0, 1.0),
        ("high_shelf_1000hz_+6db_q1_s2", 1000.0, 6.0, 1.0),
        ("high_shelf_10000hz_+6db_q4_s2", 10000.0, 6.0, 4.0),
        ("high_shelf_10000hz_+6db_q0.5_s2", 10000.0, 6.0, 0.5),
    ];

    for (name, freq, gain_db, _q_display) in scenarios {
        let csv_path = format!("/tmp/eq-test2/details/{name}.csv");
        if !std::path::Path::new(&csv_path).exists() {
            continue;
        }
        let ref_data = read_ref_response(&csv_path);

        // Current: 20/80 split, BW Q
        let bw_q = 0.5 / (PI / 4.0).cos(); // n=1, i=0: 1/√2
        let gain_1st = gain_db * 0.20;
        let gain_2nd = gain_db * 0.80;
        let s1 = coeff::high_shelf_1(freq, gain_1st, sr);
        let s2 = coeff::calculate(FilterType::HighShelf, freq, bw_q, gain_2nd, sr);
        let baseline = cascade_rms(&[s1, s2], &ref_data, sr);

        // Sweep gain split AND Q scale
        let mut best = (0.20_f64, 1.0_f64, baseline);
        for split_pct in 0..=50 {
            let split = split_pct as f64 / 100.0;
            let g1 = gain_db * split;
            let g2 = gain_db * (1.0 - split);
            for q_pct in 50..=200 {
                let q = bw_q * q_pct as f64 / 100.0;
                let s1 = coeff::high_shelf_1(freq, g1, sr);
                let s2 = coeff::calculate(FilterType::HighShelf, freq, q, g2, sr);
                let rms = cascade_rms(&[s1, s2], &ref_data, sr);
                if rms < best.2 {
                    best = (split, q_pct as f64 / 100.0, rms);
                }
            }
        }
        println!(
            "{name}: base={baseline:.4}  best_split={:.2} best_q_scale={:.2} best_rms={:.4}",
            best.0, best.1, best.2
        );
    }
}

#[test]
fn sweep_s2_split_universal() {
    // For every s2 scenario, find the optimal 1st-order gain split.
    // Tests a range of fixed splits to find the best universal value.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let base_dir = "/tmp/eq-test2/details";
    let files: Vec<String> = std::fs::read_dir(base_dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_str()?.to_string();
            if name.contains("_s2.csv") && name.starts_with("high_shelf") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    // Count pass/fail at 1.0 dB tolerance for each split value
    let bw_q = 0.5 / (PI / 4.0).cos(); // Butterworth for n=1

    println!(
        "\n=== Universal split sweep (all {} s2 high_shelf scenarios) ===",
        files.len()
    );
    println!(
        "{:>6} {:>8} {:>8} {:>8}",
        "Split", "Passes", "AvgRMS", "MaxRMS"
    );

    for split_pct in [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50] {
        let split = split_pct as f64 / 100.0;
        let mut total_rms = 0.0;
        let mut max_rms = 0.0f64;
        let mut passes = 0;
        let mut total = 0;

        for file in &files {
            let csv_path = format!("{base_dir}/{file}");
            let ref_data = read_ref_response(&csv_path);

            let name = file.strip_suffix(".csv").unwrap();
            let parts: Vec<&str> = name.split('_').collect();
            let freq_str = parts[2].strip_suffix("hz").unwrap_or("10000");
            let freq: f64 = freq_str.parse().unwrap_or(10000.0);
            let gain_str = parts[3].strip_suffix("db").unwrap_or("+6");
            let gain_db: f64 = gain_str.parse().unwrap_or(6.0);

            let gain_1st = gain_db * split;
            let gain_2nd = gain_db * (1.0 - split);
            let s1 = coeff::high_shelf_1(freq, gain_1st, sr);
            let s2 = coeff::calculate(FilterType::HighShelf, freq, bw_q, gain_2nd, sr);
            let rms = cascade_rms(&[s1, s2], &ref_data, sr);

            total_rms += rms;
            max_rms = max_rms.max(rms);
            if rms < 1.0 {
                passes += 1;
            }
            total += 1;
        }

        let avg_rms = total_rms / total as f64;
        println!(
            "{split_pct:>5}% {:>8}/{:<8} {:>8.4} {:>8.4}",
            passes, total, avg_rms, max_rms
        );
    }

    // Also test split as function of normalized frequency
    println!("\n=== Frequency-dependent split: split = k / (1 + w0/pi) ===");
    for k_pct in [0, 5, 10, 15, 20, 30] {
        let k = k_pct as f64 / 100.0;
        let mut total_rms = 0.0;
        let mut passes = 0;
        let mut total = 0;

        for file in &files {
            let csv_path = format!("{base_dir}/{file}");
            let ref_data = read_ref_response(&csv_path);

            let name = file.strip_suffix(".csv").unwrap();
            let parts: Vec<&str> = name.split('_').collect();
            let freq_str = parts[2].strip_suffix("hz").unwrap_or("10000");
            let freq: f64 = freq_str.parse().unwrap_or(10000.0);
            let gain_str = parts[3].strip_suffix("db").unwrap_or("+6");
            let gain_db: f64 = gain_str.parse().unwrap_or(6.0);

            let w0_norm = 2.0 * freq / sr; // freq / Nyquist
            let split = (k * (1.0 - w0_norm)).max(0.0);
            let gain_1st = gain_db * split;
            let gain_2nd = gain_db * (1.0 - split);
            let s1 = coeff::high_shelf_1(freq, gain_1st, sr);
            let s2 = coeff::calculate(FilterType::HighShelf, freq, bw_q, gain_2nd, sr);
            let rms = cascade_rms(&[s1, s2], &ref_data, sr);

            total_rms += rms;
            if rms < 1.0 {
                passes += 1;
            }
            total += 1;
        }
        println!(
            "k={k:.2}: {passes}/{total} passes, avg RMS={:.4}",
            total_rms / total as f64
        );
    }
}

#[test]
fn s2_worst_cases() {
    // Show the worst s2 cases to understand what's driving the errors.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;
    let bw_q = 0.5 / (PI / 4.0).cos();

    let base_dir = "/tmp/eq-test2/details";
    let files: Vec<String> = std::fs::read_dir(base_dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_str()?.to_string();
            if name.contains("_s2.csv") && name.starts_with("high_shelf") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    let mut results: Vec<(String, f64, f64)> = Vec::new();
    for file in &files {
        let csv_path = format!("{base_dir}/{file}");
        let ref_data = read_ref_response(&csv_path);
        let name = file.strip_suffix(".csv").unwrap();
        let parts: Vec<&str> = name.split('_').collect();
        let freq: f64 = parts[2]
            .strip_suffix("hz")
            .unwrap_or("10000")
            .parse()
            .unwrap_or(10000.0);
        let gain_db: f64 = parts[3]
            .strip_suffix("db")
            .unwrap_or("+6")
            .parse()
            .unwrap_or(6.0);

        // Current: 20% split
        let gain_1st = gain_db * 0.20;
        let gain_2nd = gain_db * 0.80;
        let s1 = coeff::high_shelf_1(freq, gain_1st, sr);
        let s2 = coeff::calculate(FilterType::HighShelf, freq, bw_q, gain_2nd, sr);
        let current = cascade_rms(&[s1, s2], &ref_data, sr);

        // Best split (0% = no 1st-order gain influence)
        let s1 = coeff::high_shelf_1(freq, 0.0, sr);
        let s2 = coeff::calculate(FilterType::HighShelf, freq, bw_q, gain_db, sr);
        let zero_split = cascade_rms(&[s1, s2], &ref_data, sr);

        results.push((name.to_string(), current, zero_split));
    }

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n=== Top 30 worst s2 cases ===");
    println!("{:<50} {:>8} {:>8}", "Scenario", "Current", "Split0%");
    for (name, current, zero_split) in results.iter().take(30) {
        println!("{:<50} {:>8.4} {:>8.4}", name, current, zero_split);
    }

    // Also show some that improved
    println!("\n=== Top 20 most improved with 0% split ===");
    let mut improved = results.clone();
    improved.sort_by(|a, b| (b.1 - b.2).partial_cmp(&(a.1 - a.2)).unwrap());
    for (name, current, zero_split) in improved.iter().take(20) {
        println!(
            "{:<50} {:>8.4} → {:>8.4} (Δ={:.4})",
            name,
            current,
            zero_split,
            current - zero_split
        );
    }
}

#[test]
fn analyze_shelf_q_mapping() {
    // For s2 (order=3 = 1st+biquad), sweep the BIQUAD Q to find what value
    // best matches Pro-Q at different display Q values.
    // This reveals Pro-Q's internal Q mapping for shelves.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let scenarios = [
        ("high_shelf_10000hz_+6db_q0.5_s2", 10000.0, 6.0, 0.5),
        ("high_shelf_10000hz_+6db_q1_s2", 10000.0, 6.0, 1.0),
        ("high_shelf_10000hz_+6db_q4_s2", 10000.0, 6.0, 4.0),
        ("high_shelf_10000hz_+6db_q10_s2", 10000.0, 6.0, 10.0),
        ("high_shelf_1000hz_+6db_q0.5_s2", 1000.0, 6.0, 0.5),
        ("high_shelf_1000hz_+6db_q1_s2", 1000.0, 6.0, 1.0),
        ("high_shelf_1000hz_+6db_q4_s2", 1000.0, 6.0, 4.0),
        ("high_shelf_1000hz_+6db_q10_s2", 1000.0, 6.0, 10.0),
        ("high_shelf_5000hz_+6db_q0.5_s2", 5000.0, 6.0, 0.5),
        ("high_shelf_5000hz_+6db_q1_s2", 5000.0, 6.0, 1.0),
        ("high_shelf_5000hz_+6db_q4_s2", 5000.0, 6.0, 4.0),
        ("high_shelf_5000hz_+6db_q10_s2", 5000.0, 6.0, 10.0),
    ];

    println!(
        "\n{:<45} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Scenario", "Q_disp", "Q_curr", "Q_best", "RMS_cur", "RMS_best"
    );

    for (name, freq, gain_db, q_display) in scenarios {
        let csv_path = format!("/tmp/eq-test2/details/{name}.csv");
        if !std::path::Path::new(&csv_path).exists() {
            continue;
        }
        let ref_data = read_ref_response(&csv_path);

        // Current Q: q_display / √2
        let q_current = q_display / std::f64::consts::SQRT_2;

        // Current cascade: 20/80 split
        let bw_q = 0.5 / (PI / 4.0).cos();
        let gain_1st = gain_db * 0.20;
        let gain_2nd = gain_db * 0.80;
        let s1 = coeff::high_shelf_1(freq, gain_1st, sr);
        let s2 = coeff::calculate(FilterType::HighShelf, freq, bw_q, gain_2nd, sr);
        let rms_current = cascade_rms(&[s1, s2], &ref_data, sr);

        // Sweep biquad Q AND gain split simultaneously
        let mut best_q = q_current;
        let mut best_rms = rms_current;
        let mut best_split = 0.20;
        for q_pct in 5..=2000 {
            let q = q_pct as f64 / 100.0;
            for split_pct in [0, 2, 5, 10, 20, 33] {
                let split = split_pct as f64 / 100.0;
                let g1 = gain_db * split;
                let g2 = gain_db * (1.0 - split);
                let s1 = coeff::high_shelf_1(freq, g1, sr);
                let s2 = coeff::calculate(FilterType::HighShelf, freq, q, g2, sr);
                let rms = cascade_rms(&[s1, s2], &ref_data, sr);
                if rms < best_rms {
                    best_rms = rms;
                    best_q = q;
                    best_split = split;
                }
            }
        }

        println!(
            "{:<45} {:>8.1} {:>8.4} {:>8.4} {:>8.4} {:>8.4}  split={:.2}",
            name, q_display, q_current, best_q, rms_current, best_rms, best_split
        );
    }
}

#[test]
fn verify_shelf_q_formula() {
    // Test the q_internal = q_display^k / √2 formula for shelves.
    // Sweep k to find optimal value across all s2 scenarios.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let base_dir = "/tmp/eq-test2/details";
    let files: Vec<String> = std::fs::read_dir(base_dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_str()?.to_string();
            if name.contains("_s2.csv") && name.starts_with("high_shelf") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    println!("\n=== Sweep Q exponent k for q_internal = q_display^k / √2 (all s2 shelves) ===");
    println!("{:>6} {:>8} {:>8} {:>8}", "k", "Passes", "AvgRMS", "MaxRMS");

    for k_100 in 30..=100 {
        let k = k_100 as f64 / 100.0;
        let mut total_rms = 0.0;
        let mut max_rms = 0.0f64;
        let mut passes = 0;
        let mut total = 0;

        for file in &files {
            let csv_path = format!("{base_dir}/{file}");
            let ref_data = read_ref_response(&csv_path);

            let name = file.strip_suffix(".csv").unwrap();
            let parts: Vec<&str> = name.split('_').collect();
            let freq: f64 = parts[2]
                .strip_suffix("hz")
                .unwrap_or("10000")
                .parse()
                .unwrap_or(10000.0);
            let gain_str = parts[3].strip_suffix("db").unwrap_or("+6");
            let gain_db: f64 = gain_str.parse().unwrap_or(6.0);
            let q_str = parts[4].strip_prefix('q').unwrap_or("1");
            let q_display: f64 = q_str.parse().unwrap_or(1.0);

            let q_internal = q_display.powf(k) / std::f64::consts::SQRT_2;

            // For order=3: use gain_per_pole formula (1st gets 1/3, biquad gets 2/3)
            let gain_1st = gain_db / 3.0;
            let gain_2nd = gain_db * 2.0 / 3.0;
            let bw_q = 0.5 / (PI / 4.0).cos(); // Butterworth Q for n=1
                                               // Use bw_q * (q_internal / 0.707) as the section Q? No, let's test
                                               // the formula as the Q passed to coeff::calculate.

            let s1 = coeff::high_shelf_1(freq, gain_1st, sr);
            let s2 = coeff::calculate(FilterType::HighShelf, freq, q_internal, gain_2nd, sr);
            let rms = cascade_rms(&[s1, s2], &ref_data, sr);

            total_rms += rms;
            max_rms = max_rms.max(rms);
            if rms < 1.0 {
                passes += 1;
            }
            total += 1;
        }

        let avg_rms = total_rms / total as f64;
        if k_100 % 2 == 0 {
            println!(
                "{k:>6.2} {:>8}/{:<8} {:>8.4} {:>8.4}",
                passes, total, avg_rms, max_rms
            );
        }
    }

    // Fine-tune around k=0.58 with different gain splits
    println!("\n=== Fine-tune k + gain split ===");
    for k_100 in (50..=64).step_by(1) {
        let k = k_100 as f64 / 100.0;
        for &split_pct in &[0, 5, 10, 20, 33] {
            let split = split_pct as f64 / 100.0;
            let mut total_rms = 0.0;
            let mut passes = 0;
            let mut total = 0;

            for file in &files {
                let csv_path = format!("{base_dir}/{file}");
                let ref_data = read_ref_response(&csv_path);
                let name = file.strip_suffix(".csv").unwrap();
                let parts: Vec<&str> = name.split('_').collect();
                let freq: f64 = parts[2]
                    .strip_suffix("hz")
                    .unwrap_or("10000")
                    .parse()
                    .unwrap_or(10000.0);
                let gain_db: f64 = parts[3]
                    .strip_suffix("db")
                    .unwrap_or("+6")
                    .parse()
                    .unwrap_or(6.0);
                let q_display: f64 = parts[4]
                    .strip_prefix('q')
                    .unwrap_or("1")
                    .parse()
                    .unwrap_or(1.0);

                let q_internal = q_display.powf(k) / std::f64::consts::SQRT_2;
                let g1 = gain_db * split;
                let g2 = gain_db * (1.0 - split);
                let s1 = coeff::high_shelf_1(freq, g1, sr);
                let s2 = coeff::calculate(FilterType::HighShelf, freq, q_internal, g2, sr);
                let rms = cascade_rms(&[s1, s2], &ref_data, sr);

                total_rms += rms;
                if rms < 1.0 {
                    passes += 1;
                }
                total += 1;
            }
            if passes >= 280 {
                let avg_rms = total_rms / total as f64;
                println!("k={k:.2} split={split_pct}%: {passes}/{total}, avg={avg_rms:.4}");
            }
        }
    }

    // Compare with current approach (Q/√2 + 20/80 split)
    {
        let mut total_rms = 0.0;
        let mut max_rms = 0.0f64;
        let mut passes = 0;
        let mut total = 0;

        for file in &files {
            let csv_path = format!("{base_dir}/{file}");
            let ref_data = read_ref_response(&csv_path);
            let name = file.strip_suffix(".csv").unwrap();
            let parts: Vec<&str> = name.split('_').collect();
            let freq: f64 = parts[2]
                .strip_suffix("hz")
                .unwrap_or("10000")
                .parse()
                .unwrap_or(10000.0);
            let gain_str = parts[3].strip_suffix("db").unwrap_or("+6");
            let gain_db: f64 = gain_str.parse().unwrap_or(6.0);

            let bw_q = 0.5 / (PI / 4.0).cos();
            let s1 = coeff::high_shelf_1(freq, gain_db * 0.20, sr);
            let s2 = coeff::calculate(FilterType::HighShelf, freq, bw_q, gain_db * 0.80, sr);
            let rms = cascade_rms(&[s1, s2], &ref_data, sr);

            total_rms += rms;
            max_rms = max_rms.max(rms);
            if rms < 1.0 {
                passes += 1;
            }
            total += 1;
        }
        println!(
            "\nCurrent (Q/√2, 20/80): {passes}/{total}, avg={:.4}, max={:.4}",
            total_rms / total as f64,
            max_rms
        );
    }
}

#[test]
fn verify_shelf_q_all_slopes() {
    // Test q^0.5/√2 formula across ALL shelf slopes (s0, s2, s5, s8).
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let base_dir = "/tmp/eq-test2/details";
    let all_files: Vec<String> = std::fs::read_dir(base_dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.ok()?.file_name().to_str()?.to_string();
            if name.starts_with("high_shelf") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    for slope_tag in ["s0", "s2", "s5", "s8"] {
        let files: Vec<&String> = all_files
            .iter()
            .filter(|f| f.contains(&format!("_{slope_tag}.csv")))
            .collect();

        // Need to compute cascade for each slope
        let order = match slope_tag {
            "s0" => 1,
            "s2" => 3,
            "s5" => 6,
            "s8" => 12,
            _ => continue,
        };

        let has_first_order = order % 2 == 1;
        let num_2nd: usize = order / 2;

        let bw_qs: Vec<f64> = (0..num_2nd)
            .map(|i| {
                let angle = PI * (2 * i + 1) as f64 / (4 * num_2nd) as f64;
                0.5 / angle.cos()
            })
            .collect();

        // Count passes for current and new Q formula
        let mut current_passes = 0;
        let mut new_passes = 0;
        let mut current_sum = 0.0;
        let mut new_sum = 0.0;
        let mut total = 0;

        for file in &files {
            let csv_path = format!("{base_dir}/{file}");
            let ref_data = read_ref_response(&csv_path);
            let name = file.strip_suffix(".csv").unwrap();
            let parts: Vec<&str> = name.split('_').collect();
            let freq: f64 = parts[2]
                .strip_suffix("hz")
                .unwrap_or("10000")
                .parse()
                .unwrap_or(10000.0);
            let gain_db: f64 = parts[3]
                .strip_suffix("db")
                .unwrap_or("+6")
                .parse()
                .unwrap_or(6.0);
            let q_display: f64 = parts[4]
                .strip_prefix('q')
                .unwrap_or("1")
                .parse()
                .unwrap_or(1.0);

            // Current approach
            let current_rms = {
                let q_internal = q_display / std::f64::consts::SQRT_2;
                let q_user = q_internal * std::f64::consts::SQRT_2; // = q_display
                let effective_gain = gain_db;

                let (gain_1st, gain_2nd) = if order == 3 {
                    (effective_gain * 0.20, effective_gain * 0.80)
                } else {
                    let gain_per_pole = effective_gain / order as f64;
                    (gain_per_pole, gain_per_pole * 2.0)
                };

                let mut sections = Vec::new();
                if has_first_order {
                    sections.push(coeff::high_shelf_1(freq, gain_1st, sr));
                }
                for i in 0..num_2nd {
                    let is_last = i == num_2nd - 1;
                    let is_second_last = i == num_2nd.saturating_sub(2) && num_2nd > 1;
                    let bw_q_val = bw_qs[i];
                    let q_section = if is_last {
                        let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                        let scale = if q_user > 1.0 {
                            1.0 + (q_user.ln() * 1.03 * blend)
                        } else {
                            q_user.powf(blend * 0.75)
                        };
                        bw_q_val * scale
                    } else if is_second_last && order >= 6 {
                        let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                        let scale = if q_user > 1.0 {
                            1.0 + (q_user.ln() * 0.3 * blend)
                        } else {
                            1.0
                        };
                        bw_q_val * scale
                    } else {
                        bw_q_val
                    };
                    sections.push(coeff::calculate(
                        FilterType::HighShelf,
                        freq,
                        q_section,
                        gain_2nd,
                        sr,
                    ));
                }
                cascade_rms(&sections, &ref_data, sr)
            };

            // New approach (matching actual code changes):
            // 1. lib.rs: Q unchanged (q_display / √2)
            // 2. band.rs: 5/95 split for order=3
            // 3. band.rs: single biquad uses √(q_display)/√2 (shelf Q compression)
            // 4. Multi-biquad cascade: unchanged (Butterworth + scale with original q_user)
            let new_rms = {
                let q_internal = q_display / std::f64::consts::SQRT_2;
                let q_user_orig = q_display; // q_internal * √2

                let (gain_1st, gain_2nd) = if order == 3 {
                    (gain_db * 0.05, gain_db * 0.95)
                } else {
                    let gain_per_pole = gain_db / order as f64;
                    (gain_per_pole, gain_per_pole * 2.0)
                };

                let mut sections = Vec::new();
                if has_first_order {
                    sections.push(coeff::high_shelf_1(freq, gain_1st, sr));
                }
                for i in 0..num_2nd {
                    let is_last = i == num_2nd - 1;
                    let is_second_last = i == num_2nd.saturating_sub(2) && num_2nd > 1;
                    let bw_q_val = bw_qs[i];
                    let q_section = if num_2nd == 1 {
                        // Single biquad: shelf Q compression
                        q_display.sqrt() / std::f64::consts::SQRT_2
                    } else if is_last {
                        let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                        let scale = if q_user_orig > 1.0 {
                            1.0 + (q_user_orig.ln() * 1.03 * blend)
                        } else {
                            q_user_orig.powf(blend * 0.75)
                        };
                        bw_q_val * scale
                    } else if is_second_last && order >= 6 {
                        let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                        let scale = if q_user_orig > 1.0 {
                            1.0 + (q_user_orig.ln() * 0.3 * blend)
                        } else {
                            1.0
                        };
                        bw_q_val * scale
                    } else {
                        bw_q_val
                    };
                    sections.push(coeff::calculate(
                        FilterType::HighShelf,
                        freq,
                        q_section,
                        gain_2nd,
                        sr,
                    ));
                }
                cascade_rms(&sections, &ref_data, sr)
            };

            if current_rms < 1.0 {
                current_passes += 1;
            }
            if new_rms < 1.0 {
                new_passes += 1;
            }
            current_sum += current_rms;
            new_sum += new_rms;
            total += 1;
        }

        println!("{slope_tag}: current={current_passes}/{total} (avg={:.4}), new={new_passes}/{total} (avg={:.4})",
                 current_sum / total as f64, new_sum / total as f64);
    }
}

#[test]
fn analyze_single_biquad_q_error() {
    // Compare single biquad (s1 = order 2) at different Q values.
    // This isolates whether the error is in the biquad design vs cascade.
    use eq_dsp::coeff;
    use eq_dsp::filter_type::FilterType;
    let sr = 48000.0;

    let scenarios = [
        ("high_shelf_10000hz_+6db_q0.5_s1", 10000.0, 6.0, 0.5),
        ("high_shelf_10000hz_+6db_q1_s1", 10000.0, 6.0, 1.0),
        ("high_shelf_10000hz_+6db_q4_s1", 10000.0, 6.0, 4.0),
        ("high_shelf_10000hz_+6db_q10_s1", 10000.0, 6.0, 10.0),
        ("high_shelf_10000hz_-12db_q1_s1", 10000.0, -12.0, 1.0),
        ("high_shelf_1000hz_+6db_q1_s1", 1000.0, 6.0, 1.0),
        ("high_shelf_5000hz_+6db_q1_s1", 5000.0, 6.0, 1.0),
    ];

    println!(
        "\n{:<45} {:>10} {:>10} {:>10}",
        "Scenario", "Matched", "RBJ", "Ratio"
    );
    for (name, freq, gain_db, q_display) in scenarios {
        let csv_path = format!("/tmp/eq-test2/details/{name}.csv");
        if !std::path::Path::new(&csv_path).exists() {
            continue;
        }
        let ref_data = read_ref_response(&csv_path);

        // Our matched biquad (internal Q = q_display / √2)
        let internal_q = q_display / std::f64::consts::SQRT_2;
        let matched = coeff::calculate(FilterType::HighShelf, freq, internal_q, gain_db, sr);
        let matched_rms = cascade_rms(&[matched], &ref_data, sr);

        // RBJ biquad
        let rbj = rbj_high_shelf(freq, internal_q, gain_db, sr);
        let rbj_rms = cascade_rms(&[rbj], &ref_data, sr);

        println!(
            "{:<45} {:>10.4} {:>10.4} {:>10.2}",
            name,
            matched_rms,
            rbj_rms,
            matched_rms / rbj_rms
        );
    }

    // For Q=4 single biquad, sweep Q to find optimal match
    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q4_s1.csv";
    if std::path::Path::new(csv_path).exists() {
        let ref_data = read_ref_response(csv_path);
        println!("\n=== Q sweep for single biquad, nominal Q=4 ===");
        let mut best_q = 0.0;
        let mut best_rms = f64::MAX;
        for q_pct in 10..=500 {
            let q = q_pct as f64 / 100.0;
            let c = coeff::calculate(FilterType::HighShelf, 10000.0, q, 6.0, sr);
            let rms = cascade_rms(&[c], &ref_data, sr);
            if rms < best_rms {
                best_rms = rms;
                best_q = q;
            }
        }
        println!("Nominal internal Q: {:.4}", 4.0 / std::f64::consts::SQRT_2);
        println!("Best Q: {best_q:.2} → RMS={best_rms:.4} dB");
        println!("Ratio: {:.4}", best_q / (4.0 / std::f64::consts::SQRT_2));
    }

    // For Q=0.5 single biquad, sweep Q to find optimal match
    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q0.5_s1.csv";
    if std::path::Path::new(csv_path).exists() {
        let ref_data = read_ref_response(csv_path);
        println!("\n=== Q sweep for single biquad, nominal Q=0.5 ===");
        let mut best_q = 0.0;
        let mut best_rms = f64::MAX;
        for q_pct in 5..=300 {
            let q = q_pct as f64 / 100.0;
            let c = coeff::calculate(FilterType::HighShelf, 10000.0, q, 6.0, sr);
            let rms = cascade_rms(&[c], &ref_data, sr);
            if rms < best_rms {
                best_rms = rms;
                best_q = q;
            }
        }
        println!("Nominal internal Q: {:.4}", 0.5 / std::f64::consts::SQRT_2);
        println!("Best Q: {best_q:.2} → RMS={best_rms:.4} dB");
        println!("Ratio: {:.4}", best_q / (0.5 / std::f64::consts::SQRT_2));
    }

    // For Q=1 single biquad, sweep Q
    let csv_path = "/tmp/eq-test2/details/high_shelf_10000hz_+6db_q1_s1.csv";
    if std::path::Path::new(csv_path).exists() {
        let ref_data = read_ref_response(csv_path);
        println!("\n=== Q sweep for single biquad, nominal Q=1 ===");
        let mut best_q = 0.0;
        let mut best_rms = f64::MAX;
        for q_pct in 10..=300 {
            let q = q_pct as f64 / 100.0;
            let c = coeff::calculate(FilterType::HighShelf, 10000.0, q, 6.0, sr);
            let rms = cascade_rms(&[c], &ref_data, sr);
            if rms < best_rms {
                best_rms = rms;
                best_q = q;
            }
        }
        println!("Nominal internal Q: {:.4}", 1.0 / std::f64::consts::SQRT_2);
        println!("Best Q: {best_q:.2} → RMS={best_rms:.4} dB");
        println!("Ratio: {:.4}", best_q / (1.0 / std::f64::consts::SQRT_2));
    }
}

/// RBJ cookbook high shelf (standard BLT design).
fn rbj_high_shelf(freq_hz: f64, q: f64, gain_db: f64, sr: f64) -> [f64; 6] {
    let a_val = 10.0_f64.powf(gain_db / 40.0); // A = 10^(dBgain/40)
    let w0 = 2.0 * PI * freq_hz / sr;
    let cw = w0.cos();
    let sw = w0.sin();
    let alpha = sw / (2.0 * q);
    let two_sqrt_a_alpha = 2.0 * a_val.sqrt() * alpha;

    let b0 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cw + two_sqrt_a_alpha);
    let b1 = -2.0 * a_val * ((a_val - 1.0) + (a_val + 1.0) * cw);
    let b2 = a_val * ((a_val + 1.0) + (a_val - 1.0) * cw - two_sqrt_a_alpha);
    let a0 = (a_val + 1.0) - (a_val - 1.0) * cw + two_sqrt_a_alpha;
    let a1 = 2.0 * ((a_val - 1.0) - (a_val + 1.0) * cw);
    let a2 = (a_val + 1.0) - (a_val - 1.0) * cw - two_sqrt_a_alpha;

    // Normalize by a0
    [1.0, a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0]
}
