//! Biquad processing sections: TDF2 and Gold-Rader lattice forms.
//!
//! TDF2 is the standard transposed direct-form II used for most filter types.
//! The Gold-Rader lattice form is numerically stable even with poles ON the
//! unit circle (a2 = 1.0), which is required for the binary-exact Butterworth
//! peak algorithm extracted from Pro-Q 4.

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

/// Lattice-form biquad section — numerically stable for unit-circle poles
/// (a2 = 1.0).
///
/// Inspired by Pro-Q 4 binary (process_iir_filter_with_history @ 0x1801200e0).
///
/// Architecture: IIR all-pole lattice filter + transversal FIR numerator.
///
/// The denominator `1/(1 + a1*z^-1 + a2*z^-2)` is implemented as a
/// 2nd-order all-pole lattice using Levinson-Durbin reflection coefficients:
///   k2 = a2                      (order-2 reflection coefficient)
///   k1 = a1 / (1 + a2)          (order-1 reflection coefficient)
///
/// For the Butterworth peak case where a2 = 1.0 (poles ON the unit circle),
/// k2 = 1 and k1 = a1/2 = cos(pole_angle). The lattice recursion remains
/// marginally stable without the runaway divergence seen in TDF2.
///
/// The numerator `b0 + b1*z^-1 + b2*z^-2` is applied as a 3-tap FIR on
/// the all-pole output signal w[n].
///
/// All-pole lattice recursion (Oppenheim & Schafer, Eq. 6.65):
///   f2[n] = x[n]                           (input at highest stage)
///   f1[n] = f2[n] - k2 * g1[n-1]          (stage 2, backward)
///   g2[n] = k2 * f1[n] + g1[n-1]
///   f0[n] = f1[n] - k1 * g0[n-1]          (stage 1, backward)
///   g1[n] = k1 * f0[n] + g0[n-1]
///   g0[n] = f0[n]                          (= w[n])
///
/// Output:
///   y[n] = b0*w[n] + b1*w[n-1] + b2*w[n-2]
#[derive(Clone)]
pub struct LatticeSection {
    // Reflection coefficients (Levinson-Durbin)
    k1: f64, // a1/(1+a2)  — order-1 reflection coefficient
    k2: f64, // a2         — order-2 reflection coefficient
    // Numerator FIR coefficients
    b0: f64,
    b1: f64,
    b2: f64,
    // Lattice backward state per channel: g0[n-1] (=w[n-1]), g1[n-1]
    g0: [f64; MAX_CH],
    g1: [f64; MAX_CH],
    // FIR delay line: w[n-2] per channel (w[n-1] = g0)
    w2: [f64; MAX_CH],
}

impl LatticeSection {
    pub fn new() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            g0: [0.0; MAX_CH],
            g1: [0.0; MAX_CH],
            w2: [0.0; MAX_CH],
        }
    }

    /// Load biquad coefficients from [a0, a1, a2, b0, b1, b2] format.
    ///
    /// Converts standard biquad to lattice + FIR. Numerically stable for
    /// a2/a0 = 1.0 (poles on the unit circle).
    pub fn set_coeffs(&mut self, coeffs: Coeffs) {
        let a0_inv = 1.0 / coeffs[0];
        let a1n = coeffs[1] * a0_inv;
        let a2n = coeffs[2] * a0_inv;

        // Levinson-Durbin reflection coefficients for A(z) = 1 + a1*z^-1 + a2*z^-2
        //   k2 = a2 (order-2 reflection coefficient)
        //   Step-down to order 1: A_1(z) = 1 + k1*z^-1 where k1 = a1/(1+a2)
        self.k2 = a2n;
        let denom = 1.0 + a2n;
        self.k1 = if denom.abs() > 1e-15 { a1n / denom } else { 0.0 };

        // Numerator FIR coefficients
        self.b0 = coeffs[3] * a0_inv;
        self.b1 = coeffs[4] * a0_inv;
        self.b2 = coeffs[5] * a0_inv;
    }

    /// Process one sample through the lattice biquad.
    ///
    /// Numerically stable for k2 = 1.0 (a2 = 1, poles on unit circle).
    #[inline]
    pub fn tick(&mut self, input: f64, ch: usize) -> f64 {
        // === All-pole lattice (from stage M=2 down to 1) ===
        // Implements 1/A(z) where A(z) = 1 + a1*z^-1 + a2*z^-2

        // Save old backward state
        let g0_prev = self.g0[ch]; // = w[n-1]
        let g1_prev = self.g1[ch];

        // Stage 2: f1 = x - k2 * g1[n-1]
        let f1 = input - self.k2 * g1_prev;

        // Stage 1: w = f1 - k1 * g0[n-1]
        let w = f1 - self.k1 * g0_prev;

        // Update backward state (bottom-up)
        self.g1[ch] = self.k1 * w + g0_prev;   // g1[n] = k1*f0[n] + g0[n-1]
        self.g0[ch] = w;                        // g0[n] = f0[n] = w[n]

        // === Numerator FIR: y = B(z) * w ===
        let output = self.b0 * w + self.b1 * g0_prev + self.b2 * self.w2[ch];

        // Update FIR delay (w2 = w[n-2]; w[n-1] is stored in g0)
        self.w2[ch] = g0_prev;

        output
    }

    /// Reset all state to zero.
    pub fn reset(&mut self) {
        self.g0 = [0.0; MAX_CH];
        self.g1 = [0.0; MAX_CH];
        self.w2 = [0.0; MAX_CH];
    }
}

impl Default for LatticeSection {
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

    // ─── LatticeSection tests ────────────────────────────────────────────

    #[test]
    fn lattice_passthrough_returns_input() {
        let mut sec = LatticeSection::new();
        sec.set_coeffs(PASSTHROUGH);

        for i in 0..100 {
            let input = (i as f64) * 0.01 - 0.5;
            let output = sec.tick(input, 0);
            assert!(
                (output - input).abs() < 1e-12,
                "Lattice passthrough failed at sample {}: input={}, output={}",
                i, input, output
            );
        }
    }

    #[test]
    fn lattice_matches_tdf2_for_stable_filter() {
        // Use a well-conditioned lowpass-ish biquad: a2 << 1
        let coeffs: Coeffs = [1.0, -0.5, 0.1, 0.15, 0.30, 0.15];

        let mut tdf2 = Tdf2Section::new();
        tdf2.set_coeffs(coeffs);

        let mut lattice = LatticeSection::new();
        lattice.set_coeffs(coeffs);

        // Feed an impulse then zeros
        let impulse = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (i, &x) in impulse.iter().enumerate() {
            let y_tdf2 = tdf2.tick(x, 0);
            let y_lattice = lattice.tick(x, 0);
            assert!(
                (y_tdf2 - y_lattice).abs() < 1e-10,
                "Lattice/TDF2 mismatch at sample {}: tdf2={}, lattice={}",
                i, y_tdf2, y_lattice
            );
        }
    }

    #[test]
    fn lattice_handles_unit_circle_poles() {
        // a2 = 1.0 (poles exactly on the unit circle) — this is the case
        // that makes TDF2 go unstable but should be fine for the lattice.
        // H(z) = z^-2 / (1 + 0.5*z^-1 + z^-2)
        let coeffs: Coeffs = [1.0, 0.5, 1.0, 0.0, 0.0, 1.0];

        let mut sec = LatticeSection::new();
        sec.set_coeffs(coeffs);

        // Process 1000 samples of a simple signal
        let mut all_finite = true;
        let mut max_abs = 0.0_f64;
        for i in 0..1000 {
            let input = if i == 0 { 1.0 } else { 0.0 };
            let output = sec.tick(input, 0);
            if !output.is_finite() {
                all_finite = false;
                break;
            }
            max_abs = max_abs.max(output.abs());
        }
        assert!(all_finite, "Lattice produced non-finite output with a2=1.0");
        assert!(
            max_abs < 100.0,
            "Lattice output grew too large with a2=1.0: max={}",
            max_abs
        );
    }

    #[test]
    fn lattice_reset_clears_state() {
        let coeffs: Coeffs = [1.0, -0.9, 0.5, 0.1, 0.1, 0.0];
        let mut sec = LatticeSection::new();
        sec.set_coeffs(coeffs);

        for _ in 0..10 {
            sec.tick(1.0, 0);
        }
        sec.reset();

        let mut fresh = LatticeSection::new();
        fresh.set_coeffs(coeffs);

        let out_reset = sec.tick(0.5, 0);
        let out_fresh = fresh.tick(0.5, 0);
        assert!(
            (out_reset - out_fresh).abs() < 1e-12,
            "Reset lattice should match fresh: {} != {}",
            out_reset, out_fresh
        );
    }

    #[test]
    fn lattice_channels_are_independent() {
        let coeffs: Coeffs = [1.0, -0.5, 0.3, 0.5, 0.5, 0.0];
        let mut sec = LatticeSection::new();
        sec.set_coeffs(coeffs);

        let out_ch0 = sec.tick(1.0, 0);
        let out_ch1 = sec.tick(0.0, 1);
        assert!(
            (out_ch0 - out_ch1).abs() > 1e-10,
            "Lattice channels should produce different outputs"
        );
    }
}
