//! Analyze Pro-Q 4 impulse responses: extract biquad coefficients and compare to RBJ/matched.
//!
//! Usage: cargo run --release --package eq-dsp --example analyze_proq -- /tmp/proq4-impulse/48k

use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::{env, fs};

const SR: f64 = 48000.0;

fn main() {
    let ir_dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/proq4-impulse/48k".into());
    let ir_dir = PathBuf::from(ir_dir);

    println!("{:=<100}", "");
    println!("PRO-Q 4 COEFFICIENT ANALYSIS (vs RBJ)");
    println!("{:=<100}", "");

    let q_audio = 1.0 / 2.0_f64.sqrt(); // Q_display=1 → q=1/√2

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n{:=<100}", "");
        println!("  {} (Q=1, slope=2)", shelf.to_uppercase());
        println!("{:=<100}", "");

        for &freq in &[
            100, 500, 1000, 2000, 5000, 8000, 10000, 12000, 15000, 17000, 20000, 22000,
        ] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, resid) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);
                let rbj = match *shelf {
                    "low_shelf" => rbj_low_shelf(w0, q_audio, gain_lin),
                    "high_shelf" => rbj_high_shelf(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };

                let max_diff = (0..6)
                    .map(|i| (proq[i] - rbj[i]).abs())
                    .fold(0.0_f64, f64::max);

                // Recover Pro-Q's pole frequency and radius
                let r = proq[2].abs().sqrt();
                let cos_theta = if r > 1e-6 {
                    (-proq[1] / (2.0 * r)).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                let theta = cos_theta.acos();
                let pole_freq = theta * SR / (2.0 * PI);

                // Recover what Q the RBJ formula would need to produce Pro-Q's poles
                // For low shelf: poles come from denominator with A = sqrt(gain)
                // For high shelf: poles come from denominator with A = sqrt(gain)
                let effective_q = recover_rbj_q(&proq, w0, gain_lin, shelf);

                print!(
                    "  {:5}Hz {:+3}dB | max_diff={:.6} | pole_f={:7.1}Hz r={:.6}",
                    freq, gain_db, max_diff, pole_freq, r
                );
                if let Some(eq) = effective_q {
                    print!(" | eff_q={:.6}", eq);
                }
                println!(" | resid={:.1e}", resid);

                if max_diff > 0.01 {
                    println!(
                        "      ProQ: a1={:+.8} a2={:+.8} b0={:+.8} b1={:+.8} b2={:+.8}",
                        proq[1], proq[2], proq[3], proq[4], proq[5]
                    );
                    println!(
                        "      RBJ:  a1={:+.8} a2={:+.8} b0={:+.8} b1={:+.8} b2={:+.8}",
                        rbj[1], rbj[2], rbj[3], rbj[4], rbj[5]
                    );
                }
            }
        }
    }

    // Compare Pro-Q's poles against Vicanek impulse-invariance poles
    println!("\n{:=<100}", "");
    println!("  PRO-Q POLES vs VICANEK IMPULSE-INVARIANCE POLES");
    println!("  (Are Pro-Q's poles from impulse invariance?)");
    println!("{:=<100}", "");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n  --- {} ---", shelf.to_uppercase());
        for &freq in &[
            1000, 2000, 5000, 8000, 10000, 12000, 15000, 17000, 20000, 22000,
        ] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                // Vicanek impulse-invariance poles
                // For high shelf: poles ALWAYS at g*w0 where g = G^(1/4)
                // For low shelf: poles ALWAYS at w0/g where g = G^(1/4)
                // This is the DIRECT design (not via inversion)
                let g = gain_lin.sqrt().sqrt(); // G^(1/4)
                let damp = 1.0 / (2.0 * q_audio); // 1/(2Q)

                let pole_w0 = match *shelf {
                    "high_shelf" => (g * w0).min(PI - 1e-6),
                    "low_shelf" => (w0 / g).min(PI - 1e-6),
                    _ => unreachable!(),
                };

                let (vic_a1, vic_a2) = vicanek_poles(pole_w0, damp);

                let a1_diff = (proq[1] - vic_a1).abs();
                let a2_diff = (proq[2] - vic_a2).abs();
                let pole_diff = a1_diff.max(a2_diff);

                // Also compare against RBJ poles
                let rbj = match *shelf {
                    "low_shelf" => rbj_low_shelf(w0, q_audio, gain_lin),
                    "high_shelf" => rbj_high_shelf(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };
                let rbj_pole_diff = (proq[1] - rbj[1]).abs().max((proq[2] - rbj[2]).abs());

                println!(
                    "    {:5}Hz {:+3}dB | vic_pole_diff={:.6} | rbj_pole_diff={:.6} | winner={}",
                    freq,
                    gain_db,
                    pole_diff,
                    rbj_pole_diff,
                    if pole_diff < rbj_pole_diff {
                        "VICANEK"
                    } else {
                        "RBJ"
                    }
                );
            }
        }
    }

    // Compare FULL coefficients: use Vicanek impulse-invariance poles + 3-point matching zeros
    // against Pro-Q's extracted coefficients
    println!("\n{:=<100}", "");
    println!("  FULL VICANEK MATCHED vs PRO-Q (poles + zeros)");
    println!("  (Our current matched design applied DIRECTLY to cuts too)");
    println!("{:=<100}", "");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n  --- {} ---", shelf.to_uppercase());
        for &freq in &[1000, 5000, 8000, 10000, 12000, 15000, 17000, 20000, 22000] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                // Compute our matched design DIRECTLY (no inversion for cuts)
                let matched = match *shelf {
                    "high_shelf" => vicanek_high_shelf_direct(w0, q_audio, gain_lin),
                    "low_shelf" => vicanek_low_shelf_direct(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };

                let max_diff = (0..6)
                    .map(|i| (proq[i] - matched[i]).abs())
                    .fold(0.0_f64, f64::max);

                // Pole-only diff
                let pole_diff = (proq[1] - matched[1])
                    .abs()
                    .max((proq[2] - matched[2]).abs());
                // Zero-only diff
                let zero_diff = (proq[3] - matched[3])
                    .abs()
                    .max((proq[4] - matched[4]).abs())
                    .max((proq[5] - matched[5]).abs());

                {
                    println!(
                        "    {:5}Hz {:+3}dB | pole_diff={:.6} zero_diff={:.6} max={:.6}",
                        freq, gain_db, pole_diff, zero_diff, max_diff
                    );
                    if max_diff > 0.02 {
                        println!(
                            "      ProQ:    a1={:+.8} a2={:+.8} b0={:+.8} b1={:+.8} b2={:+.8}",
                            proq[1], proq[2], proq[3], proq[4], proq[5]
                        );
                        println!(
                            "      Matched: a1={:+.8} a2={:+.8} b0={:+.8} b1={:+.8} b2={:+.8}",
                            matched[1], matched[2], matched[3], matched[4], matched[5]
                        );
                    }
                }
            }
        }
    }

    // HYPOTHESIS: Pro-Q uses impulse invariance for BOTH poles AND zeros.
    // Poles at g*w0, zeros at w0/g (high shelf) or vice versa.
    // Numerator = impulse-invariance zeros scaled for correct DC gain.
    println!("\n{:=<100}", "");
    println!("  FULL IMPULSE INVARIANCE (poles AND zeros)");
    println!("{:=<100}", "");

    for shelf in &["high_shelf", "low_shelf"] {
        println!("\n  --- {} (Q=1, slope=2) ---", shelf.to_uppercase());
        for &freq in &[
            1000, 2000, 5000, 8000, 10000, 12000, 15000, 17000, 20000, 22000,
        ] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);
                let g = gain_lin.sqrt().sqrt();
                let damp = 1.0 / (2.0 * q_audio);

                // Full impulse invariance:
                // High shelf: poles at g*w0, zeros at w0/g, DC=1
                // Low shelf:  poles at w0/g, zeros at g*w0, DC=gain
                let (pole_w0, zero_w0, dc_target) = match *shelf {
                    "high_shelf" => (g * w0, w0 / g, 1.0),
                    "low_shelf" => (w0 / g, g * w0, gain_lin),
                    _ => unreachable!(),
                };

                let pole_w0_c = pole_w0.min(PI - 1e-6);
                let zero_w0_c = zero_w0.min(PI - 1e-6);

                let (a1, a2) = vicanek_poles(pole_w0_c, damp);
                let (c1, c2) = vicanek_poles(zero_w0_c, damp);

                // Scale numerator for correct DC gain
                let den_dc = 1.0 + a1 + a2;
                let num_dc = 1.0 + c1 + c2;
                let scale = dc_target * den_dc / num_dc;
                let b0 = scale;
                let b1 = scale * c1;
                let b2 = scale * c2;

                let max_diff = (proq[1] - a1)
                    .abs()
                    .max((proq[2] - a2).abs())
                    .max((proq[3] - b0).abs())
                    .max((proq[4] - b1).abs())
                    .max((proq[5] - b2).abs());

                println!(
                    "    {:5}Hz {:+3}dB | max_diff={:.6}",
                    freq, gain_db, max_diff
                );
                if max_diff > 0.05 {
                    println!(
                        "      ProQ: a1={:+.8} a2={:+.8} b0={:+.8} b1={:+.8} b2={:+.8}",
                        proq[1], proq[2], proq[3], proq[4], proq[5]
                    );
                    println!(
                        "      II:   a1={:+.8} a2={:+.8} b0={:+.8} b1={:+.8} b2={:+.8}",
                        a1, a2, b0, b1, b2
                    );
                }
            }
        }
    }

    // Check II vs BLT blend hypothesis: does Pro-Q crossfade between II and BLT near Nyquist?
    println!("\n{:=<100}", "");
    println!("  II vs BLT BLEND ANALYSIS (near-Nyquist shelves)");
    println!("  Checking if Pro-Q blends between impulse invariance and bilinear transform");
    println!("{:=<100}", "");

    for shelf in &["high_shelf", "low_shelf"] {
        println!("\n  --- {} (Q=1, slope=2) ---", shelf.to_uppercase());
        for &freq in &[
            5000, 8000, 10000, 12000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000,
        ] {
            for &gain_db in &[12, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);
                let g = gain_lin.sqrt().sqrt();
                let damp = 1.0 / (2.0 * q_audio);

                // Impulse invariance coefficients (unclamped — let w0 exceed π)
                let (pole_w0_ii, zero_w0_ii, dc_target) = match *shelf {
                    "high_shelf" => (g * w0, w0 / g, 1.0),
                    "low_shelf" => (w0 / g, g * w0, gain_lin),
                    _ => unreachable!(),
                };
                let (ii_a1, ii_a2) = vicanek_poles(pole_w0_ii, damp); // no clamping!
                let (ii_c1, ii_c2) = vicanek_poles(zero_w0_ii, damp);
                let ii_den_dc = 1.0 + ii_a1 + ii_a2;
                let ii_num_dc = 1.0 + ii_c1 + ii_c2;
                let ii_scale = dc_target * ii_den_dc / ii_num_dc;

                // BLT (RBJ) coefficients
                let rbj = match *shelf {
                    "low_shelf" => rbj_low_shelf(w0, q_audio, gain_lin),
                    "high_shelf" => rbj_high_shelf(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };

                // Compute blend factor: find alpha where ProQ = alpha*II + (1-alpha)*BLT
                // Use a1 coefficient for the blend estimation
                let alpha_a1 = if (ii_a1 - rbj[1]).abs() > 1e-10 {
                    (proq[1] - rbj[1]) / (ii_a1 - rbj[1])
                } else {
                    1.0
                };
                let alpha_a2 = if (ii_a2 - rbj[2]).abs() > 1e-10 {
                    (proq[2] - rbj[2]) / (ii_a2 - rbj[2])
                } else {
                    1.0
                };

                // Check how well a blended version matches
                // Try best alpha from a1
                let try_blend = |alpha: f64| -> f64 {
                    let ba1 = alpha * ii_a1 + (1.0 - alpha) * rbj[1];
                    let ba2 = alpha * ii_a2 + (1.0 - alpha) * rbj[2];
                    let bb0 = alpha * ii_scale + (1.0 - alpha) * rbj[3];
                    let bb1 = alpha * ii_scale * ii_c1 + (1.0 - alpha) * rbj[4];
                    let bb2 = alpha * ii_scale * ii_c2 + (1.0 - alpha) * rbj[5];
                    (proq[1] - ba1)
                        .abs()
                        .max((proq[2] - ba2).abs())
                        .max((proq[3] - bb0).abs())
                        .max((proq[4] - bb1).abs())
                        .max((proq[5] - bb2).abs())
                };

                // Brute-force best blend alpha
                let mut best_alpha = 1.0;
                let mut best_err = try_blend(1.0);
                for i in 0..=100 {
                    let a = i as f64 / 100.0;
                    let err = try_blend(a);
                    if err < best_err {
                        best_err = err;
                        best_alpha = a;
                    }
                }

                let ii_err = (0..6)
                    .map(|i| {
                        let ii_val = [
                            1.0,
                            ii_a1,
                            ii_a2,
                            ii_scale,
                            ii_scale * ii_c1,
                            ii_scale * ii_c2,
                        ][i];
                        (proq[i] - ii_val).abs()
                    })
                    .fold(0.0_f64, f64::max);
                let rbj_err = (0..6)
                    .map(|i| (proq[i] - rbj[i]).abs())
                    .fold(0.0_f64, f64::max);

                let w0_over_pi = w0 / PI;
                println!(
                    "    {:5}Hz {:+3}dB | w0/π={:.3} | ii_err={:.4} rbj_err={:.4} | blend_α={:.2} blend_err={:.4} | α_a1={:.2} α_a2={:.2}",
                    freq, gain_db, w0_over_pi, ii_err, rbj_err, best_alpha, best_err, alpha_a1, alpha_a2
                );
            }
        }
    }

    // Z-plane pole/zero analysis: extract actual pole/zero locations from Pro-Q
    // and compare to II predictions
    println!("\n{:=<100}", "");
    println!("  Z-PLANE POLE/ZERO LOCATIONS: Pro-Q vs II");
    println!("  (Showing actual pole radius/angle vs II prediction)");
    println!("{:=<100}", "");

    for shelf in &["high_shelf"] {
        println!("\n  --- {} +12dB (Q=1, slope=2) ---", shelf.to_uppercase());
        println!(
            "  {:>5} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>8} {:>8}",
            "freq", "PQ_|z|", "PQ_θ", "PQ_real?", "II_|z|", "II_θ", "II_pw0", "|z|_rat", "θ_diff"
        );

        for &freq in &[
            1000, 2000, 3000, 5000, 8000, 10000, 12000, 14000, 15000, 16000, 17000, 18000, 19000,
            20000, 21000, 22000,
        ] {
            let gain_db = 12;
            let sign = "+";
            let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
            let path = ir_dir.join(&fname);
            if !path.exists() {
                continue;
            }

            let ir = load_ir(&path);
            let (proq, _) = extract_biquad(&ir);

            let w0 = 2.0 * PI * freq as f64 / SR;
            let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);
            let g = gain_lin.sqrt().sqrt();
            let damp = 1.0 / (2.0 * q_audio);

            // Pro-Q pole locations from a1, a2
            let disc_pq = proq[1] * proq[1] - 4.0 * proq[2];
            let (pq_mag, pq_angle, pq_real) = if disc_pq < 0.0 {
                // Complex conjugate pair
                let mag = proq[2].abs().sqrt();
                let angle = (-proq[1] / (2.0 * mag)).clamp(-1.0, 1.0).acos();
                (mag, angle, false)
            } else {
                // Real poles
                let z1 = (-proq[1] + disc_pq.sqrt()) / 2.0;
                let z2 = (-proq[1] - disc_pq.sqrt()) / 2.0;
                (z1.abs().max(z2.abs()), 0.0, true)
            };

            // II pole locations (unclamped)
            let pole_w0 = g * w0; // high shelf: poles at g*w0
            let ii_mag = (-damp * pole_w0).exp();
            let ii_angle = (1.0 - damp * damp).sqrt() * pole_w0;
            // Wrap angle to [0, π]
            let ii_angle_wrapped = ii_angle % (2.0 * PI);
            let ii_angle_wrapped = if ii_angle_wrapped > PI {
                2.0 * PI - ii_angle_wrapped
            } else {
                ii_angle_wrapped
            };

            let mag_ratio = if ii_mag > 1e-10 {
                pq_mag / ii_mag
            } else {
                f64::NAN
            };
            let angle_diff = pq_angle - ii_angle_wrapped;

            println!(
                "  {:5} | {:8.5} {:8.4} {:>8} | {:8.5} {:8.4} {:8.3} | {:8.4} {:8.4}",
                freq,
                pq_mag,
                pq_angle,
                if pq_real { "REAL" } else { "conj" },
                ii_mag,
                ii_angle_wrapped,
                pole_w0,
                mag_ratio,
                angle_diff
            );
        }

        // Also show zeros
        println!("\n  --- {} +12dB ZEROS ---", shelf.to_uppercase());
        println!(
            "  {:>5} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
            "freq", "PQ_|z|", "PQ_θ", "PQ_real?", "II_|z|", "II_θ", "II_zw0"
        );

        for &freq in &[
            1000, 2000, 5000, 8000, 10000, 12000, 14000, 17000, 20000, 22000,
        ] {
            let gain_db = 12;
            let sign = "+";
            let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
            let path = ir_dir.join(&fname);
            if !path.exists() {
                continue;
            }

            let ir = load_ir(&path);
            let (proq, _) = extract_biquad(&ir);

            let w0 = 2.0 * PI * freq as f64 / SR;
            let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);
            let g = gain_lin.sqrt().sqrt();
            let damp = 1.0 / (2.0 * q_audio);

            // Pro-Q zero locations from b0, b1, b2
            // Zeros of b0 + b1*z^-1 + b2*z^-2 = 0
            // → b2*z² + b1*z + b0 = 0... no, H(z) = (b0 + b1*z^-1 + b2*z^-2) / den
            // Zeros: b0*z² + b1*z + b2 = 0 → z = (-b1 ± sqrt(b1²-4*b0*b2)) / (2*b0)
            let disc_z = proq[4] * proq[4] - 4.0 * proq[3] * proq[5];
            let (zz_mag, zz_angle, zz_real) = if disc_z < 0.0 {
                let mag = (proq[5] / proq[3]).abs().sqrt();
                let angle = (-proq[4] / (2.0 * proq[3] * mag)).clamp(-1.0, 1.0).acos();
                (mag, angle, false)
            } else {
                let z1 = (-proq[4] + disc_z.sqrt()) / (2.0 * proq[3]);
                let z2 = (-proq[4] - disc_z.sqrt()) / (2.0 * proq[3]);
                (z1.abs().max(z2.abs()), 0.0, true)
            };

            // II zero locations (high shelf: zeros at w0/g)
            let zero_w0 = w0 / g;
            let ii_z_mag = (-damp * zero_w0).exp();
            let ii_z_angle = (1.0 - damp * damp).sqrt() * zero_w0;

            println!(
                "  {:5} | {:8.5} {:8.4} {:>8} | {:8.5} {:8.4} {:8.3}",
                freq,
                zz_mag,
                zz_angle,
                if zz_real { "REAL" } else { "conj" },
                ii_z_mag,
                ii_z_angle,
                zero_w0
            );
        }
    }

    // Now do a focused analysis: for each scenario, find what w0 the RBJ
    // would need to exactly match Pro-Q's denominator (poles).
    println!("\n{:=<100}", "");
    println!("  EFFECTIVE FREQUENCY WARP ANALYSIS");
    println!("  (What w0 would RBJ need to produce Pro-Q's poles?)");
    println!("{:=<100}", "");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n  --- {} ---", shelf.to_uppercase());
        for &freq in &[
            1000, 2000, 5000, 8000, 10000, 12000, 15000, 17000, 20000, 22000,
        ] {
            for &gain_db in &[6, -6] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                // Search for the w0 that makes RBJ match Pro-Q's a1, a2
                if let Some((eff_w0, eff_q)) = find_rbj_warp(&proq, w0, q_audio, gain_lin, shelf) {
                    let eff_freq = eff_w0 * SR / (2.0 * PI);
                    let warp_ratio = eff_w0 / w0;
                    let blt_warp = 2.0 * (w0 / 2.0).tan() / w0; // standard BLT warp
                    println!(
                        "    {:5}Hz {:+3}dB | eff_freq={:8.1}Hz | warp={:.6} | blt_warp={:.6} | q_ratio={:.6}",
                        freq, gain_db, eff_freq, warp_ratio, blt_warp, eff_q / q_audio
                    );
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // FULL 6-COEFFICIENT BRUTE FORCE SEARCH
    // Search over (w0_eff, q_eff, gain_eff) to find what RBJ parameters
    // reproduce ALL of Pro-Q's coefficients, not just poles.
    // ═══════════════════════════════════════════════════════════════════
    println!("\n{:=<100}", "");
    println!("  FULL 6-COEFF RBJ PARAMETER SEARCH");
    println!("  (What w0, Q, gain make RBJ match ALL of Pro-Q's coefficients?)");
    println!("{:=<100}", "");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n  --- {} ---", shelf.to_uppercase());
        println!(
            "  {:>5} {:>4} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8} | {:>9}",
            "freq",
            "gain",
            "w0_ratio",
            "q_ratio",
            "g_ratio",
            "pole_err",
            "zero_err",
            "max_err",
            "method"
        );

        for &freq in &[
            100, 500, 1000, 2000, 5000, 8000, 10000, 12000, 14000, 15000, 16000, 17000, 18000,
            19000, 20000, 21000, 22000,
        ] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                // Method 1: Search (w0_eff, q_eff) with fixed gain
                let (best_w, best_q, best_g, best_err, method) =
                    find_rbj_full_match(&proq, w0, q_audio, gain_lin, shelf);

                let pole_err = (proq[1] - best_coeffs_rbj(best_w, best_q, best_g, shelf)[1])
                    .abs()
                    .max((proq[2] - best_coeffs_rbj(best_w, best_q, best_g, shelf)[2]).abs());
                let zero_err = (proq[3] - best_coeffs_rbj(best_w, best_q, best_g, shelf)[3])
                    .abs()
                    .max((proq[4] - best_coeffs_rbj(best_w, best_q, best_g, shelf)[4]).abs())
                    .max((proq[5] - best_coeffs_rbj(best_w, best_q, best_g, shelf)[5]).abs());

                println!(
                    "  {:5}Hz {:+3}dB | {:8.5} {:8.5} {:8.5} | {:8.2e} {:8.2e} {:8.2e} | {}",
                    freq,
                    gain_db,
                    best_w / w0,
                    best_q / q_audio,
                    best_g / gain_lin,
                    pole_err,
                    zero_err,
                    best_err,
                    method
                );

                // If high-frequency with significant error, print actual vs best-match coefficients
                if best_err > 0.001 && freq >= 15000 {
                    let best = best_coeffs_rbj(best_w, best_q, best_g, shelf);
                    println!(
                        "      ProQ: a=[{:+.6},{:+.6}] b=[{:+.6},{:+.6},{:+.6}]",
                        proq[1], proq[2], proq[3], proq[4], proq[5]
                    );
                    println!(
                        "      Best: a=[{:+.6},{:+.6}] b=[{:+.6},{:+.6},{:+.6}]",
                        best[1], best[2], best[3], best[4], best[5]
                    );
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // FULL 6-COEFF II PARAMETER SEARCH
    // Same brute force, but searching over II (impulse invariance) params
    // instead of RBJ. Search (w0_eff, q_eff) for II poles+zeros.
    // ═══════════════════════════════════════════════════════════════════
    println!("\n{:=<100}", "");
    println!("  FULL 6-COEFF II PARAMETER SEARCH");
    println!("  (What w0, Q make impulse invariance match ALL of Pro-Q's coefficients?)");
    println!("{:=<100}", "");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n  --- {} ---", shelf.to_uppercase());
        println!(
            "  {:>5} {:>4} | {:>8} {:>8} | {:>8} {:>8} {:>8}",
            "freq", "gain", "w0_ratio", "q_ratio", "pole_err", "zero_err", "max_err"
        );

        for &freq in &[
            1000, 5000, 10000, 12000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000,
        ] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                let (best_w, best_q, best_err) =
                    find_ii_full_match(&proq, w0, q_audio, gain_lin, shelf);

                let best = ii_shelf_coeffs(best_w, best_q, gain_lin, shelf);
                let pole_err = (proq[1] - best[1]).abs().max((proq[2] - best[2]).abs());
                let zero_err = (proq[3] - best[3])
                    .abs()
                    .max((proq[4] - best[4]).abs())
                    .max((proq[5] - best[5]).abs());

                println!(
                    "  {:5}Hz {:+3}dB | {:8.5} {:8.5} | {:8.2e} {:8.2e} {:8.2e}",
                    freq,
                    gain_db,
                    best_w / w0,
                    best_q / q_audio,
                    pole_err,
                    zero_err,
                    best_err
                );

                if best_err > 0.001 && freq >= 15000 {
                    println!(
                        "      ProQ: a=[{:+.6},{:+.6}] b=[{:+.6},{:+.6},{:+.6}]",
                        proq[1], proq[2], proq[3], proq[4], proq[5]
                    );
                    println!(
                        "      Best: a=[{:+.6},{:+.6}] b=[{:+.6},{:+.6},{:+.6}]",
                        best[1], best[2], best[3], best[4], best[5]
                    );
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // COMPENSATING SHELF HYPOTHESIS
    // ═══════════════════════════════════════════════════════════════════
    // Hypothesis: Pro-Q corrects Nyquist cramping by designing a SINGLE
    // biquad whose coefficients approximate the response of:
    //   standard_shelf(w0, q, gain) * compensating_high_shelf(comp_w0, comp_q, comp_gain)
    // This is NOT two cascaded biquads — it's one biquad designed to match
    // the combined response at specific frequency points.
    //
    // Two approaches:
    //   (A) FORWARD: search comp params to match ProQ response
    //   (B) INVERSE: divide ProQ response by RBJ shelf → see if residual is a shelf
    // ═══════════════════════════════════════════════════════════════════

    println!("\n{:=<100}", "");
    println!("  COMPENSATING SHELF HYPOTHESIS");
    println!("  Testing if ProQ ≈ RBJ_shelf * compensating_high_shelf (single-biquad fit)");
    println!("{:=<100}", "");

    // Evaluation frequencies: 50 log-spaced points from 20Hz to 22kHz
    let n_eval = 50;
    let eval_freqs: Vec<f64> = (0..n_eval)
        .map(|i| {
            let t = i as f64 / (n_eval - 1) as f64;
            20.0 * (22000.0 / 20.0_f64).powf(t)
        })
        .collect();
    let eval_w: Vec<f64> = eval_freqs.iter().map(|&f| 2.0 * PI * f / SR).collect();

    // ── PART A: Forward search ──────────────────────────────────────
    println!("\n  ── PART A: Forward search for compensating shelf parameters ──");
    println!("  For each ProQ shelf, search (comp_freq, comp_gain_dB, comp_q) such that");
    println!("  |RBJ_shelf * RBJ_high_shelf_comp| best matches |ProQ| at eval points.\n");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("  --- {} ---", shelf.to_uppercase());
        println!(
            "  {:>5} {:>4} | {:>8} {:>8} {:>6} | {:>10} {:>10} | {:>7}",
            "freq", "gain", "comp_Hz", "comp_dB", "comp_Q", "rms_err_dB", "max_err_dB", "verdict"
        );

        for &freq in &[1000, 5000, 10000, 12000, 15000, 17000, 20000, 22000] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                // ProQ magnitude at eval points
                let proq_mag: Vec<f64> = eval_w.iter().map(|&w| biquad_mag_at(&proq, w)).collect();

                // RBJ shelf magnitude at eval points
                let rbj = match *shelf {
                    "low_shelf" => rbj_low_shelf(w0, q_audio, gain_lin),
                    "high_shelf" => rbj_high_shelf(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };
                let rbj_mag: Vec<f64> = eval_w.iter().map(|&w| biquad_mag_at(&rbj, w)).collect();

                // Search over compensating high shelf parameters
                // comp_freq: 5kHz to 23kHz, comp_gain_dB: -12 to +12, comp_q: 0.3 to 3.0
                let mut best_comp_freq = 0.0_f64;
                let mut best_comp_gain_db = 0.0_f64;
                let mut best_comp_q = 0.7_f64;
                let mut best_rms = f64::MAX;

                // Coarse grid
                for cfi in 0..30 {
                    let cf = 5000.0 + (23000.0 - 5000.0) * cfi as f64 / 29.0;
                    let cw0 = 2.0 * PI * cf / SR;
                    for cgi in 0..25 {
                        let cg_db = -12.0 + 24.0 * cgi as f64 / 24.0;
                        let cg_lin = 10.0_f64.powf(cg_db / 20.0);
                        for cqi in 0..10 {
                            let cq = 0.3 + 2.7 * cqi as f64 / 9.0;
                            let comp = rbj_high_shelf(cw0, cq, cg_lin);
                            let comp_mag: Vec<f64> =
                                eval_w.iter().map(|&w| biquad_mag_at(&comp, w)).collect();

                            // Combined = rbj * comp (magnitudes multiply)
                            let rms: f64 = rbj_mag
                                .iter()
                                .zip(comp_mag.iter())
                                .zip(proq_mag.iter())
                                .map(|((&r, &c), &p)| {
                                    let combined_db = 20.0 * (r * c).log10();
                                    let proq_db = 20.0 * p.log10();
                                    let err = combined_db - proq_db;
                                    err * err
                                })
                                .sum::<f64>()
                                / n_eval as f64;
                            let rms = rms.sqrt();

                            if rms < best_rms {
                                best_rms = rms;
                                best_comp_freq = cf;
                                best_comp_gain_db = cg_db;
                                best_comp_q = cq;
                            }
                        }
                    }
                }

                // Refine around best
                let mut step_f = 600.0;
                let mut step_g = 1.0;
                let mut step_q = 0.15;
                for _ in 0..200 {
                    let mut improved = false;
                    for &df in &[-step_f, 0.0, step_f] {
                        for &dg in &[-step_g, 0.0, step_g] {
                            for &dq in &[-step_q, 0.0, step_q] {
                                let cf = (best_comp_freq + df).clamp(1000.0, 23500.0);
                                let cg_db = (best_comp_gain_db + dg).clamp(-18.0, 18.0);
                                let cq = (best_comp_q + dq).max(0.1);
                                let cw0 = 2.0 * PI * cf / SR;
                                let cg_lin = 10.0_f64.powf(cg_db / 20.0);
                                let comp = rbj_high_shelf(cw0, cq, cg_lin);
                                let rms: f64 = eval_w
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &w)| {
                                        let combined_db =
                                            20.0 * (rbj_mag[i] * biquad_mag_at(&comp, w)).log10();
                                        let proq_db = 20.0 * proq_mag[i].log10();
                                        let err = combined_db - proq_db;
                                        err * err
                                    })
                                    .sum::<f64>()
                                    / n_eval as f64;
                                let rms = rms.sqrt();
                                if rms < best_rms {
                                    best_rms = rms;
                                    best_comp_freq = cf;
                                    best_comp_gain_db = cg_db;
                                    best_comp_q = cq;
                                    improved = true;
                                }
                            }
                        }
                    }
                    if !improved {
                        step_f *= 0.5;
                        step_g *= 0.5;
                        step_q *= 0.5;
                        if step_f < 0.1 && step_g < 0.001 && step_q < 0.001 {
                            break;
                        }
                    }
                }

                // Compute max error too
                let cw0 = 2.0 * PI * best_comp_freq / SR;
                let cg_lin = 10.0_f64.powf(best_comp_gain_db / 20.0);
                let comp = rbj_high_shelf(cw0, best_comp_q, cg_lin);
                let max_err_db: f64 = eval_w
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| {
                        let combined_db = 20.0 * (rbj_mag[i] * biquad_mag_at(&comp, w)).log10();
                        let proq_db = 20.0 * proq_mag[i].log10();
                        (combined_db - proq_db).abs()
                    })
                    .fold(0.0_f64, f64::max);

                let verdict = if best_rms < 0.1 {
                    "MATCH"
                } else if best_rms < 0.5 {
                    "CLOSE"
                } else if best_rms < 1.0 {
                    "WEAK"
                } else {
                    "MISS"
                };

                println!(
                    "  {:5}Hz {:+3}dB | {:8.0} {:+8.2} {:6.2} | {:10.3} {:10.3} | {:>7}",
                    freq,
                    gain_db,
                    best_comp_freq,
                    best_comp_gain_db,
                    best_comp_q,
                    best_rms,
                    max_err_db,
                    verdict
                );
            }
        }
    }

    // ── PART B: Inverse approach ────────────────────────────────────
    // Divide ProQ response by RBJ shelf response to get "residual".
    // If the hypothesis is correct, the residual should look like a
    // smooth high shelf (monotonic, gentle slope).
    println!("\n  ── PART B: Inverse approach — residual = ProQ / RBJ_shelf ──");
    println!("  If hypothesis holds, residual should be a smooth, monotonic shelf shape.\n");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("  --- {} ---", shelf.to_uppercase());

        for &freq in &[5000, 10000, 15000, 20000] {
            for &gain_db in &[12, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                let rbj = match *shelf {
                    "low_shelf" => rbj_low_shelf(w0, q_audio, gain_lin),
                    "high_shelf" => rbj_high_shelf(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };

                // Compute residual magnitude (dB) at eval points
                let residual_db: Vec<f64> = eval_w
                    .iter()
                    .map(|&w| {
                        let proq_m = biquad_mag_at(&proq, w);
                        let rbj_m = biquad_mag_at(&rbj, w);
                        if rbj_m > 1e-20 {
                            20.0 * (proq_m / rbj_m).log10()
                        } else {
                            0.0
                        }
                    })
                    .collect();

                // Check monotonicity: count direction changes
                let mut direction_changes = 0;
                let mut prev_dir: Option<bool> = None; // true = increasing
                for i in 1..residual_db.len() {
                    let diff = residual_db[i] - residual_db[i - 1];
                    if diff.abs() > 0.01 {
                        let going_up = diff > 0.0;
                        if let Some(prev) = prev_dir {
                            if going_up != prev {
                                direction_changes += 1;
                            }
                        }
                        prev_dir = Some(going_up);
                    }
                }

                let dc_residual = residual_db[0];
                let nyquist_residual = *residual_db.last().unwrap();
                let range = residual_db.iter().cloned().fold(f64::MAX, f64::min)
                    ..=residual_db.iter().cloned().fold(f64::MIN, f64::max);
                let total_range = range.end() - range.start();

                let shape = if direction_changes == 0 {
                    "MONOTONIC"
                } else if direction_changes <= 2 {
                    "NEAR-MONO"
                } else {
                    "NON-MONO"
                };

                println!(
                    "  {:5}Hz {:+3}dB | DC={:+6.2}dB Nyq={:+6.2}dB range={:.2}dB | dirs={} {}",
                    freq,
                    gain_db,
                    dc_residual,
                    nyquist_residual,
                    total_range,
                    direction_changes,
                    shape
                );

                // Print the residual curve at a few key frequencies
                let spot_freqs = [100.0, 1000.0, 5000.0, 10000.0, 15000.0, 20000.0, 22000.0];
                let mut spots = String::from("    residual: ");
                for &sf in &spot_freqs {
                    let sw = 2.0 * PI * sf / SR;
                    let proq_m = biquad_mag_at(&proq, sw);
                    let rbj_m = biquad_mag_at(&rbj, sw);
                    let r_db = if rbj_m > 1e-20 {
                        20.0 * (proq_m / rbj_m).log10()
                    } else {
                        0.0
                    };
                    spots.push_str(&format!("{:.0}Hz={:+.2}dB ", sf, r_db));
                }
                println!("{}", spots);
            }
        }
    }

    // ── PART C: Summary statistics ──────────────────────────────────
    println!("\n  ── PART C: Overall assessment ──");
    println!("  Testing how well (RBJ * comp_shelf) can approximate ProQ vs RBJ alone.\n");

    println!(
        "  {:>10} {:>5} {:>4} | {:>10} {:>10} | {:>10}",
        "shelf", "freq", "gain", "rbj_rms_dB", "comp_rms_dB", "improvement"
    );

    for shelf in &["low_shelf", "high_shelf"] {
        for &freq in &[1000, 5000, 10000, 15000, 20000, 22000] {
            for &gain_db in &[12, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                let rbj = match *shelf {
                    "low_shelf" => rbj_low_shelf(w0, q_audio, gain_lin),
                    "high_shelf" => rbj_high_shelf(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };

                // RBJ-only error
                let rbj_rms: f64 = eval_w
                    .iter()
                    .map(|&w| {
                        let p_db = 20.0 * biquad_mag_at(&proq, w).log10();
                        let r_db = 20.0 * biquad_mag_at(&rbj, w).log10();
                        (p_db - r_db).powi(2)
                    })
                    .sum::<f64>()
                    / n_eval as f64;
                let rbj_rms = rbj_rms.sqrt();

                // Quick search for best compensating shelf
                let rbj_mag: Vec<f64> = eval_w.iter().map(|&w| biquad_mag_at(&rbj, w)).collect();
                let proq_mag: Vec<f64> = eval_w.iter().map(|&w| biquad_mag_at(&proq, w)).collect();

                let mut best_comp_rms = f64::MAX;
                for cfi in 0..20 {
                    let cf = 5000.0 + 18000.0 * cfi as f64 / 19.0;
                    let cw0 = 2.0 * PI * cf / SR;
                    for cgi in 0..20 {
                        let cg_db = -12.0 + 24.0 * cgi as f64 / 19.0;
                        let cg_lin = 10.0_f64.powf(cg_db / 20.0);
                        for cqi in 0..8 {
                            let cq = 0.3 + 2.7 * cqi as f64 / 7.0;
                            let comp = rbj_high_shelf(cw0, cq, cg_lin);
                            let rms: f64 = eval_w
                                .iter()
                                .enumerate()
                                .map(|(i, &w)| {
                                    let combined_db =
                                        20.0 * (rbj_mag[i] * biquad_mag_at(&comp, w)).log10();
                                    let proq_db = 20.0 * proq_mag[i].log10();
                                    (combined_db - proq_db).powi(2)
                                })
                                .sum::<f64>()
                                / n_eval as f64;
                            let rms = rms.sqrt();
                            if rms < best_comp_rms {
                                best_comp_rms = rms;
                            }
                        }
                    }
                }

                let improvement = if rbj_rms > 0.01 {
                    format!("{:.1}x", rbj_rms / best_comp_rms)
                } else {
                    "N/A (RBJ≈0)".to_string()
                };

                println!(
                    "  {:>10} {:5}Hz {:+3}dB | {:10.3} {:10.3} | {:>10}",
                    shelf, freq, gain_db, rbj_rms, best_comp_rms, improvement
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // ORFANIDIS-INSPIRED: II POLES + PRESCRIBED DC/NYQUIST GAIN ZEROS
    // ═══════════════════════════════════════════════════════════════════
    // Instead of using II for zeros (which diverges at high freq), use
    // II poles + solve for zeros from 3 constraints:
    //   1. H(DC) = target DC gain
    //   2. H(Nyquist) = target Nyquist gain (from analog prototype!)
    //   3. |H(w0)|² = target corner magnitude (from analog prototype)
    println!("\n{:=<100}", "");
    println!("  II POLES + PRESCRIBED DC/NYQUIST ZEROS (Orfanidis-inspired)");
    println!("{:=<100}", "");

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n  --- {} ---", shelf.to_uppercase());
        println!(
            "  {:>5} {:>4} | {:>8} {:>8} {:>8} | {:>10}",
            "freq", "gain", "pole_err", "zero_err", "max_err", "vs_pure_II"
        );

        for &freq in &[
            1000, 2000, 5000, 8000, 10000, 12000, 14000, 15000, 16000, 17000, 18000, 19000, 20000,
            21000, 22000,
        ] {
            for &gain_db in &[6, 12, -6, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                let orf = orfanidis_shelf(w0, q_audio, gain_lin, shelf);
                let ii = ii_shelf_coeffs(w0, q_audio, gain_lin, shelf);

                let orf_err = (0..6)
                    .map(|i| (proq[i] - orf[i]).abs())
                    .fold(0.0_f64, f64::max);
                let ii_err = (0..6)
                    .map(|i| (proq[i] - ii[i]).abs())
                    .fold(0.0_f64, f64::max);

                let pole_err = (proq[1] - orf[1]).abs().max((proq[2] - orf[2]).abs());
                let zero_err = (proq[3] - orf[3])
                    .abs()
                    .max((proq[4] - orf[4]).abs())
                    .max((proq[5] - orf[5]).abs());

                let comparison = if orf_err < ii_err * 0.9 {
                    "BETTER"
                } else if orf_err > ii_err * 1.1 {
                    "WORSE"
                } else {
                    "SAME"
                };

                println!(
                    "  {:5}Hz {:+3}dB | {:8.2e} {:8.2e} {:8.2e} | {:>10} (ii={:.2e})",
                    freq, gain_db, pole_err, zero_err, orf_err, comparison, ii_err
                );

                if orf_err > 0.01 && freq >= 16000 {
                    println!(
                        "      ProQ: a=[{:+.6},{:+.6}] b=[{:+.6},{:+.6},{:+.6}]",
                        proq[1], proq[2], proq[3], proq[4], proq[5]
                    );
                    println!(
                        "      Orf:  a=[{:+.6},{:+.6}] b=[{:+.6},{:+.6},{:+.6}]",
                        orf[1], orf[2], orf[3], orf[4], orf[5]
                    );
                    println!(
                        "      II:   a=[{:+.6},{:+.6}] b=[{:+.6},{:+.6},{:+.6}]",
                        ii[1], ii[2], ii[3], ii[4], ii[5]
                    );
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // VICANEK FULL MATCHED (4-point) vs II vs Orfanidis comparison
    // ═══════════════════════════════════════════════════════════════════
    println!("\n{:=<100}", "");
    println!("  COMPARISON: Vicanek-4pt vs II vs Orfanidis vs RBJ (RMS dB error)");
    println!("{:=<100}", "");

    println!(
        "  {:>5} {:>4} | {:>8} {:>8} {:>8} {:>8} | {:>6}",
        "freq", "gain", "RBJ", "II", "Orfanidis", "Vic4pt", "winner"
    );

    for shelf in &["low_shelf", "high_shelf"] {
        println!("\n  --- {} ---", shelf.to_uppercase());

        for &freq in &[1000, 5000, 10000, 14000, 16000, 18000, 20000, 22000] {
            for &gain_db in &[12, -12] {
                let sign = if gain_db >= 0 { "+" } else { "" };
                let fname = format!("{shelf}_{freq}hz_{sign}{gain_db}db_q1_s2.ir.bin");
                let path = ir_dir.join(&fname);
                if !path.exists() {
                    continue;
                }

                let ir = load_ir(&path);
                let (proq, _) = extract_biquad(&ir);

                let w0 = 2.0 * PI * freq as f64 / SR;
                let gain_lin = 10.0_f64.powf(gain_db as f64 / 20.0);

                // Compute all approaches
                let rbj = match *shelf {
                    "low_shelf" => rbj_low_shelf(w0, q_audio, gain_lin),
                    "high_shelf" => rbj_high_shelf(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };
                let ii = ii_shelf_coeffs(w0, q_audio, gain_lin, shelf);
                let orf = orfanidis_shelf(w0, q_audio, gain_lin, shelf);
                let vic = match *shelf {
                    "low_shelf" => vicanek_low_shelf_direct(w0, q_audio, gain_lin),
                    "high_shelf" => vicanek_high_shelf_direct(w0, q_audio, gain_lin),
                    _ => unreachable!(),
                };

                // Compute RMS dB error over frequency range
                let rms_db_err = |c: &[f64; 6]| -> f64 {
                    let n = 50;
                    let sum: f64 = (0..n)
                        .map(|i| {
                            let f = 20.0 * (22000.0_f64 / 20.0).powf(i as f64 / (n - 1) as f64);
                            let w = 2.0 * PI * f / SR;
                            let p_db = 20.0 * biquad_mag_at(&proq, w).max(1e-10).log10();
                            let c_db = 20.0 * biquad_mag_at(c, w).max(1e-10).log10();
                            (p_db - c_db).powi(2)
                        })
                        .sum();
                    (sum / n as f64).sqrt()
                };

                let rbj_e = rms_db_err(&rbj);
                let ii_e = rms_db_err(&ii);
                let orf_e = rms_db_err(&orf);
                let vic_e = rms_db_err(&vic);

                let min_e = rbj_e.min(ii_e).min(orf_e).min(vic_e);
                let winner = if (rbj_e - min_e).abs() < 1e-6 {
                    "RBJ"
                } else if (ii_e - min_e).abs() < 1e-6 {
                    "II"
                } else if (orf_e - min_e).abs() < 1e-6 {
                    "Orf"
                } else {
                    "Vic4"
                };

                println!(
                    "  {:5}Hz {:+3}dB | {:8.3} {:8.3} {:8.3} {:8.3} | {:>6}",
                    freq, gain_db, rbj_e, ii_e, orf_e, vic_e, winner
                );
            }
        }
    }
}

/// Evaluate biquad frequency response H(e^{jw}) at a given digital frequency w.
/// coeffs = [1.0, a1, a2, b0, b1, b2] (a0 normalized to 1).
/// Returns (magnitude, phase) as complex number (re, im).
fn biquad_response_at(coeffs: &[f64; 6], w: f64) -> (f64, f64) {
    // H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
    // z = e^{jw}, z^-1 = e^{-jw}
    let cos1 = w.cos();
    let sin1 = w.sin();
    let cos2 = (2.0 * w).cos();
    let sin2 = (2.0 * w).sin();

    let (b0, b1, b2) = (coeffs[3], coeffs[4], coeffs[5]);
    let (a1, a2) = (coeffs[1], coeffs[2]);

    // Numerator: b0 + b1*e^{-jw} + b2*e^{-2jw}
    let num_re = b0 + b1 * cos1 + b2 * cos2;
    let num_im = -b1 * sin1 - b2 * sin2;

    // Denominator: 1 + a1*e^{-jw} + a2*e^{-2jw}
    let den_re = 1.0 + a1 * cos1 + a2 * cos2;
    let den_im = -a1 * sin1 - a2 * sin2;

    // Complex division: num / den
    let den_mag_sq = den_re * den_re + den_im * den_im;
    if den_mag_sq < 1e-30 {
        return (1e10, 0.0);
    }
    let re = (num_re * den_re + num_im * den_im) / den_mag_sq;
    let im = (num_im * den_re - num_re * den_im) / den_mag_sq;
    (re, im)
}

/// Evaluate biquad magnitude response |H(e^{jw})| at a given digital frequency w.
fn biquad_mag_at(coeffs: &[f64; 6], w: f64) -> f64 {
    let (re, im) = biquad_response_at(coeffs, w);
    (re * re + im * im).sqrt()
}

/// Direct Vicanek matched high shelf (impulse-invariance poles + 3-point magnitude matching).
/// Works for both boosts AND cuts — no inversion needed.
fn vicanek_high_shelf_direct(w0: f64, q: f64, gain: f64) -> [f64; 6] {
    let fc = w0 / PI;
    let fc = fc.clamp(1e-6, 1.0 - 1e-6);
    let g = gain.sqrt().sqrt(); // G^(1/4)
    let q_clamped = q.max(0.01);
    let damp = 0.5 / q_clamped;

    // Analog magnitude-squared
    let analog_mag_sq = |f: f64| -> f64 {
        let ffc = f / fc;
        let ffc2 = ffc * ffc;
        let g2 = g * g;
        let num = (1.0 - g2 * ffc2).powi(2) + (g * ffc / q_clamped).powi(2);
        let den = (1.0 - ffc2 / g2).powi(2) + (ffc / (g * q_clamped)).powi(2);
        if den.abs() > 1e-30 {
            num / den
        } else {
            gain * gain
        }
    };

    // Impulse-invariance poles at g*w0
    let pole_w0 = (g * w0).min(PI - 1e-6);
    let (a1, a2) = vicanek_poles(pole_w0, damp);

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;

    let p0 = 0.5 + 0.5 * w0.cos();
    let p1 = 0.5 - 0.5 * w0.cos();

    let h_ny = analog_mag_sq(1.0);
    let h_corner = analog_mag_sq(fc);

    let den_at_corner = a0_big * p0 + a1_big * p1 + 4.0 * a2_big * p0 * p1;

    let b0_big = a0_big; // DC = 1
    let b1_big = a1_big * h_ny;
    let target_corner = h_corner * den_at_corner;
    let b2_big = (target_corner - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
}

/// Direct Vicanek matched low shelf (impulse-invariance poles + 3-point magnitude matching).
fn vicanek_low_shelf_direct(w0: f64, q: f64, gain: f64) -> [f64; 6] {
    let fc = w0 / PI;
    let fc = fc.clamp(1e-6, 1.0 - 1e-6);
    let g = gain.sqrt().sqrt(); // G^(1/4)
    let q_clamped = q.max(0.01);
    let damp = 0.5 / q_clamped;

    // Low shelf analog magnitude-squared
    let analog_mag_sq = |f: f64| -> f64 {
        let ffc = f / fc;
        let ffc2 = ffc * ffc;
        let g2 = g * g;
        let hs_num = (1.0 - g2 * ffc2).powi(2) + (g * ffc / q_clamped).powi(2);
        let hs_den = (1.0 - ffc2 / g2).powi(2) + (ffc / (g * q_clamped)).powi(2);
        if hs_num.abs() > 1e-30 {
            gain * gain * hs_den / hs_num
        } else {
            gain * gain
        }
    };

    // Impulse-invariance poles at w0/g
    let pole_w0 = (w0 / g).min(PI - 1e-6);
    let (a1, a2) = vicanek_poles(pole_w0, damp);

    let a0_big = (1.0 + a1 + a2).powi(2);
    let a1_big = (1.0 - a1 + a2).powi(2);
    let a2_big = -4.0 * a2;

    let p0 = 0.5 + 0.5 * w0.cos();
    let p1 = 0.5 - 0.5 * w0.cos();

    let h_dc = gain * gain;
    let h_ny = analog_mag_sq(1.0);
    let h_corner = analog_mag_sq(fc);

    let den_at_corner = a0_big * p0 + a1_big * p1 + 4.0 * a2_big * p0 * p1;

    let b0_big = a0_big * h_dc;
    let b1_big = a1_big * h_ny;
    let target_corner = h_corner * den_at_corner;
    let b2_big = (target_corner - b0_big * p0 - b1_big * p1) / (4.0 * p0 * p1);

    let (b0, b1, b2) = mag_sq_to_b([b0_big.max(0.0), b1_big.max(0.0), b2_big]);
    [1.0, a1, a2, b0, b1, b2]
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

/// Vicanek impulse-invariance pole mapping.
/// Maps analog poles at (damp ± j*sqrt(1-damp²)) * w0 to digital domain.
fn vicanek_poles(w0: f64, damp: f64) -> (f64, f64) {
    let t = (-damp * w0).exp();
    let a1 = if damp <= 1.0 {
        -2.0 * t * ((1.0 - damp * damp).sqrt() * w0).cos()
    } else {
        -2.0 * t * ((damp * damp - 1.0).sqrt() * w0).cosh()
    };
    let a2 = t * t;
    (a1, a2)
}

fn load_ir(path: &Path) -> Vec<f64> {
    let data = fs::read(path).expect("read IR file");
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

fn extract_biquad(ir: &[f64]) -> ([f64; 6], f64) {
    // Pro-Q has 1-sample latency
    let y: Vec<f64> = ir[1..].to_vec();
    let n = y.len().min(200);

    // Least squares: y[n] = -a1*y[n-1] - a2*y[n-2] for n >= 3
    let (mut s11, mut s12, mut s22, mut r1, mut r2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 3..n {
        let v1 = -y[i - 1];
        let v2 = -y[i - 2];
        s11 += v1 * v1;
        s12 += v1 * v2;
        s22 += v2 * v2;
        r1 += v1 * y[i];
        r2 += v2 * y[i];
    }

    let det = s11 * s22 - s12 * s12;
    let a1 = (s22 * r1 - s12 * r2) / det;
    let a2 = (s11 * r2 - s12 * r1) / det;

    let b0 = y[0];
    let b1 = y[1] + a1 * y[0];
    let b2 = y[2] + a1 * y[1] + a2 * y[0];

    let mut resid = 0.0;
    for i in 3..n {
        let pred = -a1 * y[i - 1] - a2 * y[i - 2];
        resid += (y[i] - pred).powi(2);
    }
    resid = (resid / (n - 3) as f64).sqrt();

    ([1.0, a1, a2, b0, b1, b2], resid)
}

fn rbj_low_shelf(w0: f64, q: f64, gain: f64) -> [f64; 6] {
    let a = gain.sqrt();
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let tsa = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + tsa);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - tsa);
    let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + tsa;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) + (a - 1.0) * cos_w0 - tsa;

    [1.0, a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0]
}

fn rbj_high_shelf(w0: f64, q: f64, gain: f64) -> [f64; 6] {
    let a = gain.sqrt();
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();
    let tsa = 2.0 * a.sqrt() * alpha;

    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + tsa);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - tsa);
    let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + tsa;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
    let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - tsa;

    [1.0, a1 / a0, a2 / a0, b0 / a0, b1 / a0, b2 / a0]
}

/// Try to recover what Q would make RBJ produce Pro-Q's a1/a2 at the given w0.
fn recover_rbj_q(proq: &[f64; 6], w0: f64, gain: f64, shelf: &str) -> Option<f64> {
    // Binary search for Q that minimizes |a1_rbj - a1_proq|^2 + |a2_rbj - a2_proq|^2
    let mut lo = 0.01_f64;
    let mut hi = 20.0_f64;

    let cost = |q: f64| -> f64 {
        let rbj = match shelf {
            "low_shelf" => rbj_low_shelf(w0, q, gain),
            "high_shelf" => rbj_high_shelf(w0, q, gain),
            _ => return 1e30,
        };
        (rbj[1] - proq[1]).powi(2) + (rbj[2] - proq[2]).powi(2)
    };

    for _ in 0..100 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if cost(m1) < cost(m2) {
            hi = m2;
        } else {
            lo = m1;
        }
    }
    let best_q = (lo + hi) / 2.0;
    let c = cost(best_q);
    if c < 1e-6 {
        Some(best_q)
    } else {
        None // No Q can make RBJ match these poles
    }
}

/// Orfanidis-inspired shelf: II poles + prescribed DC/Nyquist/corner zeros.
///
/// Uses impulse invariance for pole placement (good high-freq behavior),
/// then solves for numerator b0,b1,b2 from 3 linear constraints:
///   H(DC) = dc_target       (exact DC gain)
///   H(Nyq) = nyq_target     (Nyquist gain matches analog prototype)
///   |H(w0)|² = corner_sq    (corner magnitude matches analog)
fn orfanidis_shelf(w0: f64, q: f64, gain: f64, shelf: &str) -> [f64; 6] {
    let g = gain.sqrt().sqrt(); // G^(1/4)
    let q_clamped = q.max(0.01);
    let damp = 0.5 / q_clamped;

    // Analog prototype magnitude at normalized frequencies
    // High shelf analog: H_analog(s) has poles at s = (-damp ± j*sqrt(1-damp²)) * g*w0
    //                                    zeros at s = (-damp ± j*sqrt(1-damp²)) * w0/g
    // Low shelf: swap pole/zero placement, DC gain = G
    let fc = w0 / PI; // normalized corner (0..1)

    // Analog magnitude at any normalized frequency f (0=DC, 1=Nyquist)
    let analog_mag = |f: f64| -> f64 {
        let ffc = f / fc;
        let ffc2 = ffc * ffc;
        let g2 = g * g;
        match shelf {
            "high_shelf" => {
                let num2 = (1.0 - g2 * ffc2).powi(2) + (g * ffc / q_clamped).powi(2);
                let den2 = (1.0 - ffc2 / g2).powi(2) + (ffc / (g * q_clamped)).powi(2);
                if den2 > 1e-30 {
                    (num2 / den2).sqrt()
                } else {
                    gain
                }
            }
            "low_shelf" => {
                // Low shelf = gain / high_shelf
                let hs_num2 = (1.0 - g2 * ffc2).powi(2) + (g * ffc / q_clamped).powi(2);
                let hs_den2 = (1.0 - ffc2 / g2).powi(2) + (ffc / (g * q_clamped)).powi(2);
                if hs_num2 > 1e-30 {
                    gain * (hs_den2 / hs_num2).sqrt()
                } else {
                    gain
                }
            }
            _ => 1.0,
        }
    };

    // Poles from impulse invariance
    let (pole_w0, dc_target) = match shelf {
        "high_shelf" => (g * w0, 1.0),
        "low_shelf" => (w0 / g, gain),
        _ => return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };
    let pole_w0_c = pole_w0.min(PI - 1e-6);
    let (a1, a2) = vicanek_poles(pole_w0_c, damp);

    // Target gains
    let nyq_target = analog_mag(1.0); // analog mag at Nyquist

    // Solve for b0, b1, b2 from DC + Nyquist linear constraints:
    //   DC:  (b0 + b1 + b2) / (1 + a1 + a2) = dc_target
    //   Nyq: (b0 - b1 + b2) / (1 - a1 + a2) = nyq_target
    let den_dc = 1.0 + a1 + a2;
    let den_ny = 1.0 - a1 + a2;
    let sum_b = dc_target * den_dc; // b0 + b1 + b2
    let diff_b = nyq_target * den_ny; // b0 - b1 + b2

    let b1 = (sum_b - diff_b) / 2.0;
    let s = (sum_b + diff_b) / 2.0; // s = b0 + b2

    // Third constraint: corner magnitude matching
    // |H(w0)|² = target → solve quadratic for b0
    let corner_mag = analog_mag(fc);
    let corner_mag_sq = corner_mag * corner_mag;

    let cw = w0.cos();
    let sw = w0.sin();
    let c2w = 2.0 * cw * cw - 1.0;
    let s2w = 2.0 * sw * cw;

    // |D(w0)|²
    let d_re = 1.0 + a1 * cw + a2 * c2w;
    let d_im = -(a1 * sw + a2 * s2w);
    let den_mag_sq = d_re * d_re + d_im * d_im;
    let target_num_mag_sq = corner_mag_sq * den_mag_sq;

    // N(w0) with b2 = s - b0, b1 known:
    //   N_re = b0*(1 - c2w) + b1*cw + (s-b0)*c2w = b0*(1-2*c2w) + b1*cw + s*c2w
    //   Wait, let's be more careful:
    //   N(z) = b0 + b1*z^-1 + b2*z^-2 where z = e^{jw0}
    //   N_re = b0 + b1*cos(w0) + b2*cos(2w0)
    //   N_im = -(b1*sin(w0) + b2*sin(2w0))
    // With b2 = s - b0:
    //   N_re = b0 + b1*cw + (s - b0)*c2w = b0*(1 - c2w) + b1*cw + s*c2w
    //   N_im = -(b1*sw + (s - b0)*s2w) = b0*s2w - b1*sw - s*s2w

    let alpha = 1.0 - c2w;
    let delta = s2w;
    let k_re = b1 * cw + s * c2w;
    let k_im = -b1 * sw - s * s2w;

    // |N|² = (α² + δ²)*b0² + 2*(α*k_re + δ*k_im)*b0 + (k_re² + k_im²) = target
    let qa = alpha * alpha + delta * delta;
    let qb = 2.0 * (alpha * k_re + delta * k_im);
    let qc = k_re * k_re + k_im * k_im - target_num_mag_sq;

    let disc = qb * qb - 4.0 * qa * qc;
    let b0 = if disc >= 0.0 && qa.abs() > 1e-30 {
        let sqrt_disc = disc.sqrt();
        let r1 = (-qb + sqrt_disc) / (2.0 * qa);
        let r2 = (-qb - sqrt_disc) / (2.0 * qa);
        // Pick root that gives minimum-phase (b0 > 0, balanced b0/b2)
        if r1 > 0.0 && r2 > 0.0 {
            // Pick the one closer to s/2 (balanced)
            if (r1 - s / 2.0).abs() < (r2 - s / 2.0).abs() {
                r1
            } else {
                r2
            }
        } else if r1 > 0.0 {
            r1
        } else if r2 > 0.0 {
            r2
        } else {
            s / 2.0 // fallback
        }
    } else {
        s / 2.0 // fallback
    };

    let b2 = s - b0;

    [1.0, a1, a2, b0, b1, b2]
}

/// Helper: compute RBJ shelf coefficients for given params.
fn best_coeffs_rbj(w0: f64, q: f64, gain: f64, shelf: &str) -> [f64; 6] {
    match shelf {
        "low_shelf" => rbj_low_shelf(w0, q, gain),
        "high_shelf" => rbj_high_shelf(w0, q, gain),
        _ => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }
}

/// Compute impulse-invariance shelf coefficients for given (w0, q, gain).
fn ii_shelf_coeffs(w0: f64, q: f64, gain: f64, shelf: &str) -> [f64; 6] {
    let g = gain.sqrt().sqrt(); // G^(1/4)
    let damp = 0.5 / q.max(0.01);

    let (pole_w0, zero_w0, dc_target) = match shelf {
        "high_shelf" => (g * w0, w0 / g, 1.0),
        "low_shelf" => (w0 / g, g * w0, gain),
        _ => return [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };

    // Allow w0 to exceed π — vicanek_poles handles it via aliasing
    let (a1, a2) = vicanek_poles(pole_w0, damp);
    let (c1, c2) = vicanek_poles(zero_w0, damp);

    let den_dc = 1.0 + a1 + a2;
    let num_dc = 1.0 + c1 + c2;
    let scale = if num_dc.abs() > 1e-30 {
        dc_target * den_dc / num_dc
    } else {
        dc_target
    };

    [1.0, a1, a2, scale, scale * c1, scale * c2]
}

/// Full 6-coefficient RBJ parameter search.
/// Searches (w0, q, gain) to minimize max |proq[i] - rbj[i]| across all 6 coefficients.
/// Returns (best_w0, best_q, best_gain, best_error, method_label).
fn find_rbj_full_match(
    proq: &[f64; 6],
    w0_nom: f64,
    q_nom: f64,
    gain_nom: f64,
    shelf: &str,
) -> (f64, f64, f64, f64, &'static str) {
    let rbj_fn = match shelf {
        "low_shelf" => rbj_low_shelf as fn(f64, f64, f64) -> [f64; 6],
        "high_shelf" => rbj_high_shelf as fn(f64, f64, f64) -> [f64; 6],
        _ => return (w0_nom, q_nom, gain_nom, 1e30, "ERR"),
    };

    let cost = |w: f64, q: f64, g: f64| -> f64 {
        let c = rbj_fn(w, q, g);
        (0..6)
            .map(|i| (proq[i] - c[i]).abs())
            .fold(0.0_f64, f64::max)
    };

    // Phase 1: coarse 3D grid
    let mut best_w = w0_nom;
    let mut best_q = q_nom;
    let mut best_g = gain_nom;
    let mut best_c = cost(w0_nom, q_nom, gain_nom);

    let w_lo = (w0_nom * 0.3).max(0.001);
    let w_hi = (w0_nom * 1.5).min(PI - 0.001);
    let q_lo = (q_nom * 0.2).max(0.001);
    let q_hi = q_nom * 3.0;
    let g_lo = if gain_nom > 1.0 {
        gain_nom * 0.7
    } else {
        gain_nom * 0.7
    };
    let g_hi = if gain_nom > 1.0 {
        gain_nom * 1.3
    } else {
        gain_nom * 1.3
    };

    for wi in 0..100 {
        let w = w_lo + (w_hi - w_lo) * wi as f64 / 99.0;
        for qi in 0..50 {
            let q = q_lo + (q_hi - q_lo) * qi as f64 / 49.0;
            for gi in 0..20 {
                let g = g_lo + (g_hi - g_lo) * gi as f64 / 19.0;
                let c = cost(w, q, g);
                if c < best_c {
                    best_c = c;
                    best_w = w;
                    best_q = q;
                    best_g = g;
                }
            }
        }
    }

    // Phase 2: local refinement
    let mut step_w = (w_hi - w_lo) / 100.0;
    let mut step_q = (q_hi - q_lo) / 50.0;
    let mut step_g = (g_hi - g_lo) / 20.0;

    for _ in 0..500 {
        let mut improved = false;
        for &dw in &[-step_w, 0.0, step_w] {
            for &dq in &[-step_q, 0.0, step_q] {
                for &dg in &[-step_g, 0.0, step_g] {
                    let w = (best_w + dw).clamp(0.001, PI - 0.001);
                    let q = (best_q + dq).max(0.001);
                    let g = (best_g + dg).max(0.01);
                    let c = cost(w, q, g);
                    if c < best_c {
                        best_c = c;
                        best_w = w;
                        best_q = q;
                        best_g = g;
                        improved = true;
                    }
                }
            }
        }
        if !improved {
            step_w *= 0.5;
            step_q *= 0.5;
            step_g *= 0.5;
            if step_w < 1e-14 && step_q < 1e-14 && step_g < 1e-14 {
                break;
            }
        }
    }

    let method = if best_c < 1e-8 {
        "EXACT"
    } else if best_c < 1e-4 {
        "CLOSE"
    } else if best_c < 0.01 {
        "APPROX"
    } else {
        "MISS"
    };

    (best_w, best_q, best_g, best_c, method)
}

/// Full 6-coefficient II (impulse invariance) parameter search.
/// Searches (w0, q) to minimize max |proq[i] - ii[i]| across all 6 coefficients.
/// Gain is kept fixed (II determines gain from DC/Nyquist constraint).
fn find_ii_full_match(
    proq: &[f64; 6],
    w0_nom: f64,
    q_nom: f64,
    gain_nom: f64,
    shelf: &str,
) -> (f64, f64, f64) {
    let cost = |w: f64, q: f64| -> f64 {
        let c = ii_shelf_coeffs(w, q, gain_nom, shelf);
        (0..6)
            .map(|i| (proq[i] - c[i]).abs())
            .fold(0.0_f64, f64::max)
    };

    // Phase 1: coarse 2D grid
    let mut best_w = w0_nom;
    let mut best_q = q_nom;
    let mut best_c = cost(w0_nom, q_nom);

    let w_lo = (w0_nom * 0.3).max(0.001);
    let w_hi = (w0_nom * 2.0).min(PI * 2.0); // allow past π for II
    let q_lo = (q_nom * 0.1).max(0.001);
    let q_hi = q_nom * 5.0;

    for wi in 0..200 {
        let w = w_lo + (w_hi - w_lo) * wi as f64 / 199.0;
        for qi in 0..100 {
            let q = q_lo + (q_hi - q_lo) * qi as f64 / 99.0;
            let c = cost(w, q);
            if c < best_c {
                best_c = c;
                best_w = w;
                best_q = q;
            }
        }
    }

    // Phase 2: local refinement
    let mut step_w = (w_hi - w_lo) / 200.0;
    let mut step_q = (q_hi - q_lo) / 100.0;

    for _ in 0..500 {
        let mut improved = false;
        for &dw in &[-step_w, 0.0, step_w] {
            for &dq in &[-step_q, 0.0, step_q] {
                let w = (best_w + dw).max(0.001);
                let q = (best_q + dq).max(0.001);
                let c = cost(w, q);
                if c < best_c {
                    best_c = c;
                    best_w = w;
                    best_q = q;
                    improved = true;
                }
            }
        }
        if !improved {
            step_w *= 0.5;
            step_q *= 0.5;
            if step_w < 1e-14 && step_q < 1e-14 {
                break;
            }
        }
    }

    (best_w, best_q, best_c)
}

/// Find (w0_eff, q_eff) that makes RBJ produce Pro-Q's a1/a2.
/// Searches over both w0 and Q simultaneously.
fn find_rbj_warp(
    proq: &[f64; 6],
    w0_nom: f64,
    q_nom: f64,
    gain: f64,
    shelf: &str,
) -> Option<(f64, f64)> {
    // 2D search: w0 in [w0_nom*0.1, pi-eps], q in [0.01, 20]
    let rbj_fn = match shelf {
        "low_shelf" => rbj_low_shelf as fn(f64, f64, f64) -> [f64; 6],
        "high_shelf" => rbj_high_shelf as fn(f64, f64, f64) -> [f64; 6],
        _ => return None,
    };

    let cost = |w: f64, q: f64| -> f64 {
        let rbj = rbj_fn(w, q, gain);
        (rbj[1] - proq[1]).powi(2) + (rbj[2] - proq[2]).powi(2)
    };

    // Grid search first
    let mut best_w = w0_nom;
    let mut best_q = q_nom;
    let mut best_c = cost(w0_nom, q_nom);

    let w_lo = 0.001_f64;
    let w_hi = PI - 0.001;
    for wi in 0..200 {
        let w = w_lo + (w_hi - w_lo) * wi as f64 / 199.0;
        for qi in 0..100 {
            let q = 0.01 + 19.99 * qi as f64 / 99.0;
            let c = cost(w, q);
            if c < best_c {
                best_c = c;
                best_w = w;
                best_q = q;
            }
        }
    }

    // Refine with Nelder-Mead-like local search
    let mut step_w = 0.05;
    let mut step_q = 0.1;
    for _ in 0..200 {
        let mut improved = false;
        for &dw in &[-step_w, 0.0, step_w] {
            for &dq in &[-step_q, 0.0, step_q] {
                let w = (best_w + dw).clamp(0.001, PI - 0.001);
                let q = (best_q + dq).max(0.001);
                let c = cost(w, q);
                if c < best_c {
                    best_c = c;
                    best_w = w;
                    best_q = q;
                    improved = true;
                }
            }
        }
        if !improved {
            step_w *= 0.5;
            step_q *= 0.5;
            if step_w < 1e-12 && step_q < 1e-12 {
                break;
            }
        }
    }

    if best_c < 1e-10 {
        Some((best_w, best_q))
    } else {
        None
    }
}
