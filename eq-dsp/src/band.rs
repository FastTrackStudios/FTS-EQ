//! EQ band — a complete filter band with variable order.
//!
//! Cascades multiple 2nd-order sections with Butterworth-distributed Q
//! values for higher-order slopes. Supports all 9 filter types.

use std::f64::consts::PI;

use fts_dsp::AudioConfig;

use crate::coeff;
use crate::filter_type::{FilterStructure, FilterType};
use crate::section::{SvfSection, Tdf2Section};

/// Maximum filter order (number of poles). Must be even for 2nd-order cascading.
pub const MAX_ORDER: usize = 12;
/// Maximum number of cascaded 2nd-order sections.
const MAX_SECTIONS: usize = MAX_ORDER / 2;

// r[impl eq.band.standard]
/// A single EQ band with variable order and filter structure.
pub struct Band {
    // Parameters
    pub filter_type: FilterType,
    pub structure: FilterStructure,
    pub freq_hz: f64,
    pub gain_db: f64,
    pub q: f64,
    pub order: usize, // 1, 2, 4, 6, 8, 10, 12
    pub enabled: bool,

    // Processing state — one of these is active
    tdf2: [Tdf2Section; MAX_SECTIONS],
    svf: [SvfSection; MAX_SECTIONS],
    num_sections: usize,
}

impl Band {
    pub fn new() -> Self {
        Self {
            filter_type: FilterType::Peak,
            structure: FilterStructure::Tdf2,
            freq_hz: 1000.0,
            gain_db: 0.0,
            q: 0.707,
            order: 2,
            enabled: true,
            tdf2: std::array::from_fn(|_| Tdf2Section::new()),
            svf: std::array::from_fn(|_| SvfSection::new()),
            num_sections: 1,
        }
    }

    /// Recalculate coefficients for all sections.
    pub fn update(&mut self, config: AudioConfig) {
        if !self.enabled {
            return;
        }

        let order = self.order.clamp(1, MAX_ORDER);

        match self.filter_type {
            FilterType::Lowpass | FilterType::Highpass => {
                self.update_pass_filter(order, config);
            }
            FilterType::LowShelf | FilterType::HighShelf | FilterType::TiltShelf => {
                self.update_shelf_filter(order, config);
            }
            FilterType::Peak => {
                self.update_peak_filter(order, config);
            }
            FilterType::BandShelf => {
                self.update_band_shelf(order, config);
            }
            FilterType::Bandpass | FilterType::Notch => {
                // Always 2nd order
                self.num_sections = 1;
                let c = coeff::calculate(
                    self.filter_type,
                    self.freq_hz,
                    self.q,
                    self.gain_db,
                    config.sample_rate,
                );
                self.set_section_coeffs(0, c);
            }
        }
    }

    fn update_pass_filter(&mut self, order: usize, config: AudioConfig) {
        if order == 1 {
            self.num_sections = 1;
            // 1st-order: use a 2nd-order section with low Q for gentle slope
            let c = coeff::calculate(self.filter_type, self.freq_hz, 0.5, 0.0, config.sample_rate);
            self.set_section_coeffs(0, c);
            return;
        }

        let num = order / 2;
        self.num_sections = num.min(MAX_SECTIONS);

        // Butterworth Q distribution for each section
        let theta0 = PI / (num as f64) / 4.0;
        let scale = (std::f64::consts::SQRT_2 * self.q).powf(1.0 / num as f64);

        for i in 0..self.num_sections {
            let theta = theta0 * (2 * i + 1) as f64;
            let q_section = 0.5 / theta.cos() * scale;
            let c = coeff::calculate(
                self.filter_type,
                self.freq_hz,
                q_section,
                0.0,
                config.sample_rate,
            );
            self.set_section_coeffs(i, c);
        }
    }

    fn update_shelf_filter(&mut self, order: usize, config: AudioConfig) {
        if order <= 2 {
            self.num_sections = 1;
            let c = coeff::calculate(
                self.filter_type,
                self.freq_hz,
                self.q,
                self.gain_db,
                config.sample_rate,
            );
            self.set_section_coeffs(0, c);
            return;
        }

        let num = order / 2;
        self.num_sections = num.min(MAX_SECTIONS);

        // Distribute gain across sections
        let g = 10.0_f64.powf(self.gain_db / 20.0);
        let g_per_section = g.powf(1.0 / num as f64);
        let gain_per_section = 20.0 * g_per_section.log10();

        let theta0 = PI / (num as f64) / 4.0;
        let scale = (std::f64::consts::SQRT_2 * self.q).powf(1.0 / num as f64);

        for i in 0..self.num_sections {
            let theta = theta0 * (2 * i + 1) as f64;
            let q_section = 0.5 / theta.cos() * scale;
            let c = coeff::calculate(
                self.filter_type,
                self.freq_hz,
                q_section,
                gain_per_section,
                config.sample_rate,
            );
            self.set_section_coeffs(i, c);
        }
    }

    fn update_peak_filter(&mut self, order: usize, config: AudioConfig) {
        if order <= 2 {
            self.num_sections = 1;
            let c = coeff::calculate(
                FilterType::Peak,
                self.freq_hz,
                self.q,
                self.gain_db,
                config.sample_rate,
            );
            self.set_section_coeffs(0, c);
            return;
        }

        // Higher-order peak: use band shelf (opposing shelf pair)
        self.update_band_shelf(order, config);
    }

    /// Band shelf: opposing low shelf pair at bandwidth edges.
    fn update_band_shelf(&mut self, order: usize, config: AudioConfig) {
        let halfbw = (0.5 / self.q).asinh() / 2.0_f64.ln();
        let scale = 2.0_f64.powf(halfbw);
        let w1 = self.freq_hz / scale; // lower edge
        let w2 = self.freq_hz * scale; // upper edge

        let num = (order / 2).max(1);
        // Need 2 shelves per section pair
        self.num_sections = (num * 2).min(MAX_SECTIONS);

        let g = 10.0_f64.powf(self.gain_db / 20.0);
        let g_per = g.powf(1.0 / num as f64);
        let gain_per = 20.0 * g_per.log10();
        let inv_gain_per = -gain_per;

        let q_shelf = std::f64::consts::SQRT_2 / 2.0;

        for i in 0..num.min(MAX_SECTIONS / 2) {
            // Low shelf cut at w1
            let c1 = coeff::calculate(
                FilterType::LowShelf,
                w1,
                q_shelf,
                inv_gain_per,
                config.sample_rate,
            );
            self.set_section_coeffs(i * 2, c1);

            // Low shelf boost at w2
            let c2 = coeff::calculate(
                FilterType::LowShelf,
                w2,
                q_shelf,
                gain_per,
                config.sample_rate,
            );
            self.set_section_coeffs(i * 2 + 1, c2);
        }
    }

    fn set_section_coeffs(&mut self, idx: usize, coeffs: coeff::Coeffs) {
        self.tdf2[idx].set_coeffs(coeffs);
        self.svf[idx].set_coeffs(coeffs);
    }

    /// Process a single sample through all cascaded sections.
    #[inline]
    pub fn tick(&mut self, mut sample: f64, ch: usize) -> f64 {
        if !self.enabled {
            return sample;
        }

        match self.structure {
            FilterStructure::Tdf2 => {
                for i in 0..self.num_sections {
                    sample = self.tdf2[i].tick(sample, ch);
                }
            }
            FilterStructure::Svf => {
                for i in 0..self.num_sections {
                    sample = self.svf[i].tick(sample, ch);
                }
            }
        }
        sample
    }

    // r[impl dsp.biquad.reset]
    pub fn reset(&mut self) {
        for s in &mut self.tdf2 {
            s.reset();
        }
        for s in &mut self.svf {
            s.reset();
        }
    }
}

impl Default for Band {
    fn default() -> Self {
        Self::new()
    }
}
