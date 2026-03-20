//! Filter section — a single biquad stage with TDF2 or SVF processing.
//!
//! This is the atomic processing unit. Bands cascade multiple sections
//! for higher filter orders.

use crate::coeff::Coeffs;

/// Maximum number of stereo channels.
const MAX_CH: usize = 2;

// r[impl dsp.biquad.tdf2]
/// Transposed Direct Form II processing state.
#[derive(Clone)]
pub struct Tdf2State {
    s1: [f64; MAX_CH],
    s2: [f64; MAX_CH],
}

// r[impl dsp.biquad.tdf2]
/// TDF2 section with normalized coefficients.
#[derive(Clone)]
pub struct Tdf2Section {
    c0: f64, // b0/a0
    c1: f64, // b1/a0
    c2: f64, // b2/a0
    c3: f64, // a1/a0
    c4: f64, // a2/a0
    state: Tdf2State,
}

impl Tdf2Section {
    pub fn new() -> Self {
        Self {
            c0: 1.0,
            c1: 0.0,
            c2: 0.0,
            c3: 0.0,
            c4: 0.0,
            state: Tdf2State {
                s1: [0.0; MAX_CH],
                s2: [0.0; MAX_CH],
            },
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
        let s = &mut self.state;
        let output = input * self.c0 + s.s1[ch];
        s.s1[ch] = input * self.c1 - output * self.c3 + s.s2[ch];
        s.s2[ch] = input * self.c2 - output * self.c4;
        output
    }

    // r[impl dsp.biquad.reset]
    pub fn reset(&mut self) {
        self.state.s1 = [0.0; MAX_CH];
        self.state.s2 = [0.0; MAX_CH];
    }
}

impl Default for Tdf2Section {
    fn default() -> Self {
        Self::new()
    }
}

/// SVF section — converted from biquad coefficients.
///
/// Processes as a universal SVF that mixes HP/BP/LP outputs to replicate
/// any biquad transfer function. Better for parameter modulation since
/// it stays stable under fast coefficient changes.
#[derive(Clone)]
pub struct SvfSection {
    g: f64,
    r2: f64,
    h: f64,
    chp: f64,
    cbp: f64,
    clp: f64,
    s1: [f64; MAX_CH],
    s2: [f64; MAX_CH],
}

impl SvfSection {
    pub fn new() -> Self {
        Self {
            g: 0.0,
            r2: 0.0,
            h: 1.0,
            chp: 1.0,
            cbp: 0.0,
            clp: 0.0,
            s1: [0.0; MAX_CH],
            s2: [0.0; MAX_CH],
        }
    }

    /// Convert biquad coefficients to SVF parameters.
    pub fn set_coeffs(&mut self, c: Coeffs) {
        let a0 = c[0];
        let a1 = c[1];
        let a2 = c[2];
        let b0 = c[3];
        let b1 = c[4];
        let b2 = c[5];

        let temp1 = (-a0 - a1 - a2).abs().sqrt();
        let temp2 = (-a0 + a1 - a2).abs().sqrt();

        if temp2.abs() < 1e-30 {
            // Degenerate case — fall back to passthrough
            self.g = 0.0;
            self.r2 = 0.0;
            self.h = 1.0;
            self.chp = 1.0;
            self.cbp = 0.0;
            self.clp = 0.0;
            return;
        }

        self.g = temp1 / temp2;
        self.r2 = 2.0 * (a0 - a2) / (temp1 * temp2);
        self.h = 1.0 / (self.g * (self.r2 + self.g) + 1.0);

        let den_dc = a0 + a1 + a2;
        let den_ny = a0 - a1 + a2;

        self.chp = if den_ny.abs() > 1e-30 {
            (b0 - b1 + b2) / den_ny
        } else {
            0.0
        };
        self.cbp = if (temp1 * temp2).abs() > 1e-30 {
            2.0 * (b2 - b0) / (temp1 * temp2)
        } else {
            0.0
        };
        self.clp = if den_dc.abs() > 1e-30 {
            (b0 + b1 + b2) / den_dc
        } else {
            0.0
        };
    }

    #[inline]
    pub fn tick(&mut self, input: f64, ch: usize) -> f64 {
        let y_hp = self.h * (input - self.s1[ch] * (self.g + self.r2) - self.s2[ch]);
        let y_bp = y_hp * self.g + self.s1[ch];
        self.s1[ch] = y_hp * self.g + y_bp;
        let y_lp = y_bp * self.g + self.s2[ch];
        self.s2[ch] = y_bp * self.g + y_lp;

        self.chp * y_hp + self.cbp * y_bp + self.clp * y_lp
    }

    pub fn reset(&mut self) {
        self.s1 = [0.0; MAX_CH];
        self.s2 = [0.0; MAX_CH];
    }
}

impl Default for SvfSection {
    fn default() -> Self {
        Self::new()
    }
}
