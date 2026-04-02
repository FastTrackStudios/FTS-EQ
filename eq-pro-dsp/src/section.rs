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
}
