//! Shared test infrastructure for eq-dsp and downstream crates.
//!
//! Provides deterministic test signal generation, FFT-based spectral analysis,
//! and comparison utilities.

// r[impl test.infra.test-signals]
// r[impl test.infra.comparison]

use std::f64::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::band::Band;
use crate::chain::EqChain;
use fts_dsp::{AudioConfig, Processor};

// ── Signal Generation ──────────────────────────────────────────────────

/// Unit impulse: 1.0 at sample 0, 0.0 elsewhere.
pub fn impulse(len: usize) -> Vec<f64> {
    let mut buf = vec![0.0; len];
    if !buf.is_empty() {
        buf[0] = 1.0;
    }
    buf
}

/// Deterministic white noise via xorshift64.
pub fn white_noise(len: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1, 1]
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        })
        .collect()
}

/// Sine wave at a single frequency.
pub fn sine(len: usize, freq_hz: f64, sample_rate: f64) -> Vec<f64> {
    (0..len)
        .map(|i| (2.0 * PI * freq_hz * i as f64 / sample_rate).sin())
        .collect()
}

/// Linear sine sweep from `start_hz` to `end_hz`.
pub fn sine_sweep(len: usize, start_hz: f64, end_hz: f64, sample_rate: f64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let t = i as f64 / len as f64;
            let freq = start_hz + (end_hz - start_hz) * t;
            (2.0 * PI * freq * i as f64 / sample_rate).sin()
        })
        .collect()
}

/// Silence (all zeros).
pub fn silence(len: usize) -> Vec<f64> {
    vec![0.0; len]
}

/// DC offset (constant value).
pub fn dc_offset(len: usize, level: f64) -> Vec<f64> {
    vec![level; len]
}

/// Full-scale square wave at given frequency.
pub fn square_wave(len: usize, freq_hz: f64, sample_rate: f64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let phase = (freq_hz * i as f64 / sample_rate).fract();
            if phase < 0.5 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

// ── FFT Analysis ───────────────────────────────────────────────────────

/// Compute the magnitude spectrum in dB from a time-domain signal.
///
/// Returns `(num_bins / 2 + 1)` entries of `(freq_hz, magnitude_db)`,
/// covering DC through Nyquist.
pub fn fft_magnitude_db(signal: &[f64], sample_rate: f64) -> Vec<(f64, f64)> {
    let n = signal.len();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = signal.iter().map(|&s| Complex::new(s, 0.0)).collect();

    fft.process(&mut buffer);

    let num_bins = n / 2 + 1;
    let bin_hz = sample_rate / n as f64;

    (0..num_bins)
        .map(|i| {
            let freq = i as f64 * bin_hz;
            let mag = buffer[i].norm();
            let db = if mag > 1e-30 {
                20.0 * mag.log10()
            } else {
                -300.0
            };
            (freq, db)
        })
        .collect()
}

/// Look up the magnitude in dB at the nearest FFT bin to `target_hz`.
pub fn magnitude_at_freq(spectrum: &[(f64, f64)], target_hz: f64) -> f64 {
    spectrum
        .iter()
        .min_by(|a, b| {
            (a.0 - target_hz)
                .abs()
                .partial_cmp(&(b.0 - target_hz).abs())
                .unwrap()
        })
        .map(|&(_, db)| db)
        .unwrap_or(-300.0)
}

// ── Comparison ─────────────────────────────────────────────────────────

/// Maximum absolute difference between two signals.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// RMS difference between two signals.
pub fn rms_diff(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (sum_sq / n as f64).sqrt()
}

/// Assert that all samples are finite (not NaN or infinity).
pub fn assert_all_finite(signal: &[f64], label: &str) {
    for (i, &s) in signal.iter().enumerate() {
        assert!(s.is_finite(), "{label}: sample[{i}] = {s} is not finite");
    }
}

// ── Processing Helpers ─────────────────────────────────────────────────

/// Create a standard AudioConfig for testing.
pub fn test_config(sample_rate: f64) -> AudioConfig {
    AudioConfig {
        sample_rate,
        max_buffer_size: 512,
    }
}

/// Process a mono signal through a single Band (channel 0).
pub fn process_band_mono(band: &mut Band, input: &[f64]) -> Vec<f64> {
    input.iter().map(|&s| band.tick(s, 0)).collect()
}

/// Process a mono signal through an EqChain (left channel).
pub fn process_chain_mono(chain: &mut EqChain, input: &[f64], block_size: usize) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());
    for chunk in input.chunks(block_size) {
        let mut left: Vec<f64> = chunk.to_vec();
        let mut right = vec![0.0; chunk.len()];
        chain.process(&mut left, &mut right);
        output.extend_from_slice(&left);
    }
    output
}
