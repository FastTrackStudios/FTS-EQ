//! Single EQ band using the pro design pipeline.
//!
//! Replaces eq-dsp v2's band.rs. Uses the Pro-Q 4 architecture:
//! analog prototype -> transform -> ZPK -> biquad sections, with
//! support for all 13 filter types and variable order up to 16.
//!
//! Peak filters use the Gold-Rader lattice form (`LatticeSection`) for
//! numerical stability with unit-circle poles. All other types use the
//! standard TDF2 form (`Tdf2Section`).

use crate::biquad::PASSTHROUGH;
use crate::cascade;
use crate::design::{self, FilterType};
use crate::section::{LatticeSection, Tdf2Section};

/// Maximum filter order (number of poles).
pub const MAX_ORDER: usize = 16;

/// Maximum number of cascaded 2nd-order sections.
const MAX_SECTIONS: usize = MAX_ORDER / 2 + 1; // +1 for odd-order 1st-order section

/// A single EQ band with variable order, using the pro ZPK design pipeline.
pub struct Band {
    pub filter_type: FilterType,
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
    pub order: usize,
    pub enabled: bool,
    /// Gain-Q interaction amount (0.0 = off, 1.0 = full). Only affects Peak.
    /// From Pro-Q 4 binary: offset 0x8c in band parameter object.
    pub gain_q_interaction: f64,

    // TDF2 sections (used for all filter types except Peak lattice)
    sections: [Tdf2Section; MAX_SECTIONS],
    num_sections: usize,

    // Lattice sections (used for Peak when lattice mode is active)
    lattice_sections: [LatticeSection; MAX_SECTIONS],
    num_lattice_sections: usize,
    /// Wet gain for the Peak lattice dry/wet blend.
    /// Output = input + peak_wet_gain * lattice_cascade(input)
    peak_wet_gain: f64,
    /// Whether this band is currently using lattice processing.
    use_lattice: bool,

    output_gain: f64,
    sample_rate: f64,
}

impl Band {
    pub fn new() -> Self {
        Self {
            filter_type: FilterType::Peak,
            freq_hz: 1000.0,
            gain_db: 0.0,
            q: 0.707,
            order: 2,
            enabled: true,
            gain_q_interaction: 0.0,
            sections: std::array::from_fn(|_| Tdf2Section::new()),
            num_sections: 1,
            lattice_sections: std::array::from_fn(|_| LatticeSection::new()),
            num_lattice_sections: 0,
            peak_wet_gain: 0.0,
            use_lattice: false,
            output_gain: 1.0,
            sample_rate: 48000.0,
        }
    }

    /// Recalculate coefficients using the pro ZPK design pipeline.
    pub fn update(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.use_lattice = false;

        if !self.enabled {
            return;
        }

        let order = self.order.clamp(0, MAX_ORDER);

        self.output_gain = 1.0;

        // Pro-Q 4 applies gain as flat output for Notch and Bandpass
        if matches!(self.filter_type, FilterType::Notch | FilterType::Bandpass)
            && self.gain_db.abs() > 0.001
        {
            self.num_sections = 0;
            self.num_lattice_sections = 0;
            self.output_gain = 10.0_f64.powf(self.gain_db / 20.0);
            return;
        }

        if order == 0 {
            self.num_sections = 0;
            self.num_lattice_sections = 0;
            return;
        }

        // Apply gain-Q interaction (only affects Peak, off by default)
        let q = if self.filter_type == FilterType::Peak && self.gain_q_interaction > 0.001 {
            design::apply_gain_q_interaction(self.q, self.gain_db, self.gain_q_interaction)
        } else {
            self.q
        };

        // Apply type-specific parameter adjustments from Pro-Q 4 binary
        let (effective_q, effective_gain) = match self.filter_type {
            FilterType::FlatTilt => {
                // Type 6: clip Q to 1.885 (from binary constant TYPE_6_CLIP)
                (q.min(1.884955592153876), self.gain_db)
            }
            FilterType::ShelfAlt => {
                // Type 12: same Q clipping as FlatTilt
                (q.min(1.884955592153876), self.gain_db)
            }
            _ => {
                // All other types: pass through raw Q and gain
                (q, self.gain_db)
            }
        };

        // Peak filters: use the lattice path for binary-exact Butterworth cascade
        if self.filter_type == FilterType::Peak {
            let result = cascade::compute_cascade_peak_lattice(
                self.freq_hz,
                effective_q,
                effective_gain,
                sample_rate,
                order,
            );

            self.num_lattice_sections = result.sections.len().min(MAX_SECTIONS);
            self.peak_wet_gain = result.wet_gain;
            self.num_sections = 0; // TDF2 not used for Peak

            for (i, coeffs) in result.sections.iter().enumerate().take(self.num_lattice_sections) {
                let stable = coeffs.iter().all(|c| c.is_finite() && c.abs() < 1e12);
                let coeffs = if stable { *coeffs } else { PASSTHROUGH };
                self.lattice_sections[i].set_coeffs(coeffs);
            }

            self.use_lattice = true;
            return;
        }

        // All other filter types: use the standard TDF2 design pipeline
        self.num_lattice_sections = 0;

        let sos = design::design_filter(
            self.filter_type,
            self.freq_hz,
            effective_q,
            effective_gain,
            sample_rate,
            order,
        );

        self.num_sections = sos.len().min(MAX_SECTIONS);
        for (i, coeffs) in sos.iter().enumerate().take(self.num_sections) {
            // Stability check
            let stable = coeffs.iter().all(|c| c.is_finite() && c.abs() < 1e12);
            let coeffs = if stable { *coeffs } else { PASSTHROUGH };
            self.sections[i].set_coeffs(coeffs);
        }
    }

    /// Process a single sample through all cascaded sections.
    #[inline]
    pub fn tick(&mut self, sample: f64, ch: usize) -> f64 {
        if !self.enabled {
            return sample;
        }

        if self.use_lattice {
            // Peak lattice path: y = input + wet_gain * cascade(input)
            let mut cascade_out = sample;
            for i in 0..self.num_lattice_sections {
                cascade_out = self.lattice_sections[i].tick(cascade_out, ch);
            }
            return sample + self.peak_wet_gain * cascade_out;
        }

        // Standard TDF2 path for all other filter types
        let mut out = sample;
        for i in 0..self.num_sections {
            out = self.sections[i].tick(out, ch);
        }
        out * self.output_gain
    }

    /// Reset all section state to zero.
    pub fn reset(&mut self) {
        for s in &mut self.sections {
            s.reset();
        }
        for s in &mut self.lattice_sections {
            s.reset();
        }
    }
}

impl Default for Band {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_band_passes_through() {
        let mut band = Band::new();
        band.gain_db = 0.0;
        band.update(48000.0);

        // With 0 dB peak, output should approximate input
        let out = band.tick(1.0, 0);
        assert!(
            (out - 1.0).abs() < 0.1,
            "Default band should pass through, got {out}"
        );
    }

    #[test]
    fn disabled_band_passes_through() {
        let mut band = Band::new();
        band.enabled = false;
        band.update(48000.0);

        let out = band.tick(0.5, 0);
        assert!(
            (out - 0.5).abs() < 1e-14,
            "Disabled band should pass through exactly"
        );
    }

    #[test]
    fn band_reset_clears_state() {
        let mut band = Band::new();
        band.filter_type = FilterType::Lowpass;
        band.freq_hz = 1000.0;
        band.order = 4;
        band.update(48000.0);

        // Process some samples to build state
        for _ in 0..100 {
            band.tick(1.0, 0);
        }

        band.reset();

        // After reset, first sample should match a fresh band
        let mut fresh = Band::new();
        fresh.filter_type = FilterType::Lowpass;
        fresh.freq_hz = 1000.0;
        fresh.order = 4;
        fresh.update(48000.0);

        let out_reset = band.tick(0.5, 0);
        let out_fresh = fresh.tick(0.5, 0);
        assert!(
            (out_reset - out_fresh).abs() < 1e-12,
            "Reset band should match fresh: {out_reset} vs {out_fresh}"
        );
    }

    #[test]
    fn bandpass_gain_applied_as_output() {
        let mut band = Band::new();
        band.filter_type = FilterType::Bandpass;
        band.gain_db = 6.0;
        band.update(48000.0);

        // With gain on bandpass, num_sections should be 0 and output_gain set
        assert_eq!(band.num_sections, 0);
        let expected_gain = 10.0_f64.powf(6.0 / 20.0);
        assert!(
            (band.output_gain - expected_gain).abs() < 1e-10,
            "Output gain should be ~{expected_gain}, got {}",
            band.output_gain
        );
    }
}
