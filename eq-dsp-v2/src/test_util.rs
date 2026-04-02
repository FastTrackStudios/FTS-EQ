//! Test utilities for eq-dsp-v2.

use crate::biquad::Coeffs;

/// Process a buffer of samples through a cascade of biquad sections.
/// Returns the output buffer.
pub fn process_impulse(sections: &[Coeffs], num_samples: usize) -> Vec<f64> {
    let mut state: Vec<[f64; 2]> = vec![[0.0; 2]; sections.len()];
    let mut output = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let mut sample = if i == 0 { 1.0 } else { 0.0 };
        for (j, c) in sections.iter().enumerate() {
            let a0_inv = 1.0 / c[0];
            let out = sample * c[3] * a0_inv + state[j][0];
            state[j][0] = sample * c[4] * a0_inv - out * c[1] * a0_inv + state[j][1];
            state[j][1] = sample * c[5] * a0_inv - out * c[2] * a0_inv;
            sample = out;
        }
        output.push(sample);
    }

    output
}

/// Compute magnitude response in dB from impulse response via FFT.
pub fn impulse_to_mag_db(impulse: &[f64], num_bins: usize) -> Vec<f64> {
    use rustfft::num_complex::Complex as FftComplex;
    use rustfft::FftPlanner;

    let n = num_bins * 2;
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<FftComplex<f64>> = impulse
        .iter()
        .copied()
        .chain(std::iter::repeat(0.0))
        .take(n)
        .map(|x| FftComplex::new(x, 0.0))
        .collect();

    fft.process(&mut buffer);

    buffer[..num_bins]
        .iter()
        .map(|c| 20.0 * (c.re * c.re + c.im * c.im).sqrt().max(1e-30).log10())
        .collect()
}
