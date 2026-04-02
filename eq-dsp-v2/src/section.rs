//! Biquad processing section — TDF2 implementation.

use crate::biquad::Coeffs;

const MAX_CH: usize = 2;

/// Transposed Direct Form II biquad section.
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

    pub fn set_coeffs(&mut self, coeffs: Coeffs) {
        let a0_inv = 1.0 / coeffs[0];
        self.c0 = coeffs[3] * a0_inv;
        self.c1 = coeffs[4] * a0_inv;
        self.c2 = coeffs[5] * a0_inv;
        self.c3 = coeffs[1] * a0_inv;
        self.c4 = coeffs[2] * a0_inv;
    }

    #[inline]
    pub fn tick(&mut self, input: f64, ch: usize) -> f64 {
        let output = input * self.c0 + self.s1[ch];
        self.s1[ch] = input * self.c1 - output * self.c3 + self.s2[ch];
        self.s2[ch] = input * self.c2 - output * self.c4;
        output
    }

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
