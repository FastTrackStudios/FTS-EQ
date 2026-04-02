//! EQ band — a complete filter band using the v2 ZPK design pipeline.
//!
//! Replaces eq-dsp v1's band.rs. Instead of hand-tuned coefficient functions,
//! uses the Pro-Q 4 architecture: analog prototype → transform → ZPK → biquad.

use crate::biquad::PASSTHROUGH;
use crate::design::{self, FilterType};
use crate::section::Tdf2Section;

/// Maximum filter order (number of poles).
pub const MAX_ORDER: usize = 16;
/// Maximum number of cascaded 2nd-order sections.
const MAX_SECTIONS: usize = MAX_ORDER / 2 + 1; // +1 for odd-order 1st-order section

/// A single EQ band with variable order, using v2 ZPK design.
pub struct Band {
    pub filter_type: FilterType,
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
    pub order: usize,
    pub enabled: bool,

    sections: [Tdf2Section; MAX_SECTIONS],
    num_sections: usize,
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
            sections: std::array::from_fn(|_| Tdf2Section::new()),
            num_sections: 1,
            output_gain: 1.0,
            sample_rate: 48000.0,
        }
    }

    /// Recalculate coefficients using the v2 ZPK design pipeline.
    pub fn update(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;

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
            self.output_gain = 10.0_f64.powf(self.gain_db / 20.0);
            return;
        }

        if order == 0 {
            self.num_sections = 0;
            return;
        }

        // Use v2 design pipeline: analog prototype → ZPK → biquad sections
        let sos = design::design_filter(
            self.filter_type,
            self.freq_hz,
            self.q,
            self.gain_db,
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

        let mut out = sample;
        for i in 0..self.num_sections {
            out = self.sections[i].tick(out, ch);
        }
        out * self.output_gain
    }

    pub fn reset(&mut self) {
        for s in &mut self.sections {
            s.reset();
        }
    }
}

impl Default for Band {
    fn default() -> Self {
        Self::new()
    }
}
