//! TDF2 biquad processing section with Pro-Q 4's double-precision history.

use crate::biquad::Coeffs;

const MAX_CH: usize = 2;

/// Transposed Direct Form II biquad section.
///
/// Double-precision state matching Pro-Q 4's internal processing path.
#[derive(Clone)]
pub struct Tdf2Section {
    c0: f64, // b0/a0
    c1: f64, // b1/a0
    c2: f64, // b2/a0
    c3: f64, // a1/a0
    c4: f64, // a2/a0
    s1: [f64; MAX_CH],
    s2: [f64; MAX_CH],
}

impl Tdf2Section {
    pub fn new() -> Self {
        Self {
            c0: 1.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
            c4: 0.0,
            s1: [0.0; MAX_CH],
            s2: [0.0; MAX_CH],
        }
    }

    /// Load biquad coefficients from [a0, a1, a2, b0, b1, b2] format.
    pub fn set_coeffs(&mut self, coeffs: Coeffs) {
        let a0_inv = 1.0 / coeffs[0];
        self.c0 = coeffs[3] * a0_inv; // b0/a0
        self.c1 = coeffs[4] * a0_inv; // b1/a0
        self.c2 = coeffs[5] * a0_inv; // b2/a0
        self.c3 = coeffs[1] * a0_inv; // a1/a0
        self.c4 = coeffs[2] * a0_inv; // a2/a0
    }

    /// Process one sample through the biquad (TDF2).
    #[inline]
    pub fn tick(&mut self, input: f64, ch: usize) -> f64 {
        let output = input * self.c0 + self.s1[ch];
        self.s1[ch] = input * self.c1 - output * self.c3 + self.s2[ch];
        self.s2[ch] = input * self.c2 - output * self.c4;
        output
    }

    /// Reset all state to zero.
    pub fn reset(&mut self) {
        self.s1 = [0.0; MAX_CH];
        self.s2 = [0.0; MAX_CH];
    }
}

impl Default for Tdf2Section {
    fn default() -> Self {
        Self::new()
    }
}

/// Direct Form I biquad — numerically stable alternative to TDF2.
///
/// From Pro-Q 4 binary (process_biquad_cascade_single_channel @ 0x18011ffc0).
/// Processes standard biquad H(z) = (b0+b1z^-1+b2z^-2)/(1+a1z^-1+a2z^-2)
/// using Direct Form I which is more stable than TDF2 for high-Q near Nyquist.
///
/// Direct Form I difference equation:
///   y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
#[derive(Clone)]
pub struct Df1Section {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: [f64; MAX_CH],
    x2: [f64; MAX_CH],
    y1: [f64; MAX_CH],
    y2: [f64; MAX_CH],
}

impl Df1Section {
    pub fn new() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: [0.0; MAX_CH],
            x2: [0.0; MAX_CH],
            y1: [0.0; MAX_CH],
            y2: [0.0; MAX_CH],
        }
    }

    /// Load biquad coefficients from [a0, a1, a2, b0, b1, b2] format.
    pub fn set_coeffs(&mut self, coeffs: Coeffs) {
        let a0_inv = 1.0 / coeffs[0];
        self.b0 = coeffs[3] * a0_inv;
        self.b1 = coeffs[4] * a0_inv;
        self.b2 = coeffs[5] * a0_inv;
        self.a1 = coeffs[1] * a0_inv;
        self.a2 = coeffs[2] * a0_inv;
    }

    /// Process one sample through the biquad (Direct Form I).
    #[inline]
    pub fn tick(&mut self, input: f64, ch: usize) -> f64 {
        let output = self.b0 * input + self.b1 * self.x1[ch] + self.b2 * self.x2[ch]
            - self.a1 * self.y1[ch]
            - self.a2 * self.y2[ch];
        self.x2[ch] = self.x1[ch];
        self.x1[ch] = input;
        self.y2[ch] = self.y1[ch];
        self.y1[ch] = output;
        output
    }

    /// Reset all state to zero.
    pub fn reset(&mut self) {
        self.x1 = [0.0; MAX_CH];
        self.x2 = [0.0; MAX_CH];
        self.y1 = [0.0; MAX_CH];
        self.y2 = [0.0; MAX_CH];
    }
}

impl Default for Df1Section {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::biquad::PASSTHROUGH;

    #[test]
    fn passthrough_returns_input() {
        let mut sec = Tdf2Section::new();
        sec.set_coeffs(PASSTHROUGH);

        for i in 0..100 {
            let input = (i as f64) * 0.01 - 0.5;
            let output = sec.tick(input, 0);
            assert!(
                (output - input).abs() < 1e-14,
                "Passthrough failed at sample {}: input={}, output={}",
                i,
                input,
                output
            );
        }
    }

    #[test]
    fn default_is_passthrough() {
        let mut sec = Tdf2Section::default();

        let vals = [1.0, -0.5, 0.25, 0.0, -1.0];
        for &v in &vals {
            let out = sec.tick(v, 0);
            assert!(
                (out - v).abs() < 1e-14,
                "Default section not passthrough: {} != {}",
                out,
                v
            );
        }
    }

    #[test]
    fn channels_are_independent() {
        let mut sec = Tdf2Section::new();
        // Simple low-pass-ish coefficients to produce state.
        sec.set_coeffs([1.0, -0.5, 0.0, 0.5, 0.5, 0.0]);

        // Feed different signals to channel 0 and channel 1.
        let out_ch0 = sec.tick(1.0, 0);
        let out_ch1 = sec.tick(0.0, 1);

        assert!(
            (out_ch0 - out_ch1).abs() > 1e-10,
            "Channels should produce different outputs for different inputs"
        );

        // Second sample: channel states should be independent.
        let out2_ch0 = sec.tick(0.0, 0);
        let out2_ch1 = sec.tick(1.0, 1);
        assert!(
            (out2_ch0 - out2_ch1).abs() > 1e-10,
            "Channel states should be independent"
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut sec = Tdf2Section::new();
        sec.set_coeffs([1.0, -0.9, 0.0, 0.1, 0.1, 0.0]);

        // Build up state.
        for _ in 0..10 {
            sec.tick(1.0, 0);
        }

        sec.reset();

        // After reset, first sample should match a fresh section.
        let mut fresh = Tdf2Section::new();
        fresh.set_coeffs([1.0, -0.9, 0.0, 0.1, 0.1, 0.0]);

        let out_reset = sec.tick(0.5, 0);
        let out_fresh = fresh.tick(0.5, 0);
        assert!(
            (out_reset - out_fresh).abs() < 1e-14,
            "Reset state should match fresh: {} != {}",
            out_reset,
            out_fresh
        );
    }

    #[test]
    fn df1_passthrough() {
        let mut sec = Df1Section::new();
        sec.set_coeffs(PASSTHROUGH);

        for i in 0..100 {
            let input = (i as f64) * 0.01 - 0.5;
            let output = sec.tick(input, 0);
            assert!(
                (output - input).abs() < 1e-14,
                "DF1 passthrough failed at sample {}: input={}, output={}",
                i,
                input,
                output
            );
        }
    }

    #[test]
    fn df1_matches_tdf2() {
        // Well-conditioned low-Q filter: both forms should agree closely.
        let coeffs: Coeffs = [1.0, -1.2, 0.5, 0.8, -0.6, 0.3];

        let mut df1 = Df1Section::new();
        df1.set_coeffs(coeffs);

        let mut tdf2 = Tdf2Section::new();
        tdf2.set_coeffs(coeffs);

        // Feed identical impulse + noise-like signal
        let signal: Vec<f64> = (0..200)
            .map(|i| if i == 0 { 1.0 } else { ((i as f64) * 0.7).sin() * 0.5 })
            .collect();

        let mut max_diff = 0.0_f64;
        for &x in &signal {
            let y_df1 = df1.tick(x, 0);
            let y_tdf2 = tdf2.tick(x, 0);
            max_diff = max_diff.max((y_df1 - y_tdf2).abs());
        }

        assert!(
            max_diff < 1e-10,
            "DF1 and TDF2 should match for well-conditioned filter, max diff = {max_diff}"
        );
    }

    #[test]
    fn df1_handles_high_q() {
        use std::f64::consts::PI;

        // High-Q filter near Nyquist (Q=50, f=0.45*fs)
        let w0 = 2.0 * PI * 0.45; // near Nyquist
        let q = 50.0;
        let gain_db = 12.0;
        let a = 10.0_f64.powf(gain_db / 40.0);
        let sin_w0 = w0.sin();
        let cos_w0 = w0.cos();
        let alpha = sin_w0 / (2.0 * q);

        let coeffs: Coeffs = [
            1.0 + alpha / a,
            -2.0 * cos_w0,
            1.0 - alpha / a,
            1.0 + alpha * a,
            -2.0 * cos_w0,
            1.0 - alpha * a,
        ];

        let mut sec = Df1Section::new();
        sec.set_coeffs(coeffs);

        // Process 1000 samples without NaN
        let mut any_nan = false;
        let mut out = 0.0;
        for i in 0..1000 {
            let input = if i == 0 { 1.0 } else { 0.0 };
            out = sec.tick(input, 0);
            if out.is_nan() || out.is_infinite() {
                any_nan = true;
                break;
            }
        }

        assert!(
            !any_nan,
            "DF1 should handle high-Q near Nyquist without NaN/Inf"
        );
        assert!(
            out.abs() < 1e6,
            "DF1 output should remain bounded, got {out}"
        );
    }
}
