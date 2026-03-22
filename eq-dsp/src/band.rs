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
pub const MAX_ORDER: usize = 16;
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
    /// Output gain multiplier for filter types that use gain as output level
    /// (lowpass, highpass, bandpass, notch, allpass).
    output_gain: f64,
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
            output_gain: 1.0,
        }
    }

    /// Recalculate coefficients for all sections.
    pub fn update(&mut self, config: AudioConfig) {
        if !self.enabled {
            return;
        }

        let order = self.order.clamp(0, MAX_ORDER);

        // Reset output gain (filter types that need it will set it)
        self.output_gain = 1.0;

        // Pro-Q 4 applies gain as flat output for Notch and Bandpass
        // (filter is bypassed, only the gain level is applied).
        if matches!(self.filter_type, FilterType::Notch | FilterType::Bandpass)
            && self.gain_db.abs() > 0.001
        {
            self.num_sections = 0;
            self.output_gain = 10.0_f64.powf(self.gain_db / 20.0);
            return;
        }

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
            FilterType::Bandpass => {
                self.update_bandpass(order, config);
            }
            FilterType::Notch => {
                self.update_notch(order, config);
            }
            FilterType::Allpass => {
                self.update_allpass(order, config);
            }
            FilterType::FlatTilt => {
                self.update_flat_tilt(order, config);
            }
        }
    }

    fn update_pass_filter(&mut self, order: usize, config: AudioConfig) {
        // Pro-Q 4 bypasses LP/HP at slope 0 (order 0 from lp_hp_slope_to_order).
        if order == 0 {
            self.num_sections = 0;
            return;
        }

        // Order 1: single 1st-order section (6 dB/oct).
        if order == 1 {
            self.num_sections = 1;
            let c = match self.filter_type {
                FilterType::Lowpass => coeff::lowpass_1(self.freq_hz, config.sample_rate),
                FilterType::Highpass => coeff::highpass_1(self.freq_hz, config.sample_rate),
                _ => unreachable!(),
            };
            self.set_section_coeffs(0, c);
            return;
        }

        // For odd orders, use a 1st-order section + N 2nd-order sections
        let has_first_order = order % 2 == 1;
        let num_2nd = order / 2;
        let total_sections = if has_first_order {
            num_2nd + 1
        } else {
            num_2nd
        };
        self.num_sections = total_sections.min(MAX_SECTIONS);

        let mut section_idx = 0;

        if has_first_order && section_idx < self.num_sections {
            let c = match self.filter_type {
                FilterType::Lowpass => coeff::lowpass_1(self.freq_hz, config.sample_rate),
                FilterType::Highpass => coeff::highpass_1(self.freq_hz, config.sample_rate),
                _ => unreachable!(),
            };
            self.set_section_coeffs(section_idx, c);
            section_idx += 1;
        }

        // Butterworth Q cascade with user Q on the last section (resonance control).
        // Pro-Q 4 applies user Q to one biquad section for resonance at cutoff,
        // while other sections maintain Butterworth Q for proper rolloff shape.
        let num_biquads = num_2nd.min(self.num_sections - section_idx);
        for i in 0..num_biquads {
            let q_section = if i == num_biquads - 1 {
                // Last biquad: use user Q for resonance control
                self.q
            } else {
                // Other biquads: Butterworth Q for clean rolloff
                butterworth_q(num_2nd, i)
            };
            let c = coeff::calculate(
                self.filter_type,
                self.freq_hz,
                q_section,
                0.0,
                config.sample_rate,
            );
            self.set_section_coeffs(section_idx, c);
            section_idx += 1;
        }
    }

    fn update_shelf_filter(&mut self, order: usize, config: AudioConfig) {
        // Pro-Q 4 tilt shelf: +6dB gain means DC=-6dB, Nyquist=+6dB.
        // Our tilt_shelf_2 produces DC=√g, Nyquist=1/√g.
        // To get DC=1/g, Nyquist=g, pass 1/g² = 10^(-2*gain_db/20).
        let effective_gain_db = if self.filter_type == FilterType::TiltShelf {
            -2.0 * self.gain_db
        } else {
            self.gain_db
        };

        if order == 1 {
            self.num_sections = 1;
            let c = match self.filter_type {
                FilterType::LowShelf => {
                    coeff::low_shelf_1(self.freq_hz, self.gain_db, config.sample_rate)
                }
                FilterType::HighShelf => {
                    coeff::high_shelf_1(self.freq_hz, self.gain_db, config.sample_rate)
                }
                FilterType::TiltShelf => {
                    coeff::tilt_shelf_1(self.freq_hz, effective_gain_db, config.sample_rate)
                }
                _ => unreachable!(),
            };
            self.set_section_coeffs(0, c);
            return;
        }

        if order <= 2 {
            self.num_sections = 1;
            let c = coeff::calculate(
                self.filter_type,
                self.freq_hz,
                self.q,
                effective_gain_db,
                config.sample_rate,
            );
            self.set_section_coeffs(0, c);
            return;
        }

        // For odd orders, use a 1st-order shelf + N 2nd-order sections.
        let has_first_order = order % 2 == 1;
        let num_2nd = order / 2;
        let total = if has_first_order {
            num_2nd + 1
        } else {
            num_2nd
        };
        self.num_sections = total.min(MAX_SECTIONS);

        // Check if shelf resonance is needed (Q > 1 for low/high shelf).
        let q_user = self.q * std::f64::consts::SQRT_2;
        let use_resonance = q_user > 1.01
            && matches!(
                self.filter_type,
                FilterType::LowShelf | FilterType::HighShelf
            );

        // Distribute gain evenly across all sections (in dB domain).
        let g = 10.0_f64.powf(effective_gain_db / 20.0);
        let gain_per_section = 20.0 * g.powf(1.0 / total as f64).log10();

        let mut section_idx = 0;

        if has_first_order && section_idx < self.num_sections {
            let c = match self.filter_type {
                FilterType::LowShelf => {
                    coeff::low_shelf_1(self.freq_hz, gain_per_section, config.sample_rate)
                }
                FilterType::HighShelf => {
                    coeff::high_shelf_1(self.freq_hz, gain_per_section, config.sample_rate)
                }
                FilterType::TiltShelf => {
                    coeff::tilt_shelf_1(self.freq_hz, gain_per_section, config.sample_rate)
                }
                _ => unreachable!(),
            };
            self.set_section_coeffs(section_idx, c);
            section_idx += 1;
        }

        let num_biquads = num_2nd.min(self.num_sections - section_idx);
        for i in 0..num_biquads {
            let is_last = i == num_biquads - 1;
            if is_last && use_resonance {
                let q_eff = q_user.powf(0.75);
                let c = match self.filter_type {
                    FilterType::LowShelf => coeff::low_shelf_resonant(
                        self.freq_hz,
                        q_eff,
                        gain_per_section,
                        config.sample_rate,
                    ),
                    FilterType::HighShelf => coeff::high_shelf_resonant(
                        self.freq_hz,
                        q_eff,
                        gain_per_section,
                        config.sample_rate,
                    ),
                    _ => unreachable!(),
                };
                self.set_section_coeffs(section_idx, c);
            } else {
                let q_section = butterworth_q(num_2nd, i);
                let c = coeff::calculate(
                    self.filter_type,
                    self.freq_hz,
                    q_section,
                    gain_per_section,
                    config.sample_rate,
                );
                self.set_section_coeffs(section_idx, c);
            }
            section_idx += 1;
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

        // Higher-order peak: cascade multiple 2nd-order peak sections
        // with gain distributed across sections.
        let num = order / 2;
        self.num_sections = num.min(MAX_SECTIONS);

        let g = 10.0_f64.powf(self.gain_db / 20.0);
        let g_per = g.powf(1.0 / num as f64);
        let gain_per = 20.0 * g_per.log10();

        for i in 0..self.num_sections {
            let c = coeff::calculate(
                FilterType::Peak,
                self.freq_hz,
                self.q,
                gain_per,
                config.sample_rate,
            );
            self.set_section_coeffs(i, c);
        }
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

    /// Bandpass via cascaded 2nd-order matched bandpass sections.
    ///
    /// Pro-Q 4's bandpass uses resonant bandpass filters centered at the
    /// specified frequency. Higher slopes cascade more sections.
    fn update_bandpass(&mut self, order: usize, config: AudioConfig) {
        if order == 0 {
            self.num_sections = 0;
            return;
        }

        // Each 2nd-order bandpass section provides ~6 dB/oct rolloff per side.
        // Slope maps to order, and we need ceil(order/2) biquad sections
        // (since each biquad is 2nd order).
        // For odd orders, we still use a 2nd-order section — there's no
        // meaningful 1st-order bandpass.
        let num_sections = ((order + 1) / 2).max(1).min(MAX_SECTIONS);
        self.num_sections = num_sections;

        for i in 0..num_sections {
            let q_section = if i == num_sections - 1 {
                self.q // Last section: user Q for resonance control
            } else {
                butterworth_q(num_sections, i) // Others: Butterworth Q
            };
            let c = coeff::calculate(
                FilterType::Bandpass,
                self.freq_hz,
                q_section,
                0.0,
                config.sample_rate,
            );
            self.set_section_coeffs(i, c);
        }
    }

    /// Higher-order notch: cascade multiple 2nd-order notch sections.
    fn update_notch(&mut self, order: usize, config: AudioConfig) {
        if order <= 2 {
            self.num_sections = 1;
            let c = coeff::calculate(
                FilterType::Notch,
                self.freq_hz,
                self.q,
                0.0,
                config.sample_rate,
            );
            self.set_section_coeffs(0, c);
            return;
        }

        let num = (order / 2).max(1);
        self.num_sections = num.min(MAX_SECTIONS);

        for i in 0..self.num_sections {
            let c = coeff::calculate(
                FilterType::Notch,
                self.freq_hz,
                self.q,
                0.0,
                config.sample_rate,
            );
            self.set_section_coeffs(i, c);
        }
    }

    /// Allpass: phase rotation + peak for gain.
    ///
    /// Pro-Q 4's allpass applies gain as a bell/peak shape (not flat output
    /// level). At center frequency the magnitude equals gain_db; away from
    /// center it returns to 0 dB.
    fn update_allpass(&mut self, order: usize, config: AudioConfig) {
        if order == 1 {
            // 1st-order allpass + peak for gain
            self.num_sections = if self.gain_db.abs() > 0.001 { 2 } else { 1 };
            let c = coeff::allpass_1(self.freq_hz, config.sample_rate);
            self.set_section_coeffs(0, c);
            if self.num_sections == 2 {
                let c_peak = coeff::calculate(
                    FilterType::Peak,
                    self.freq_hz,
                    self.q,
                    self.gain_db,
                    config.sample_rate,
                );
                self.set_section_coeffs(1, c_peak);
            }
            return;
        }

        let has_first_order = order % 2 == 1;
        let num_2nd = order / 2;
        let allpass_sections = if has_first_order {
            num_2nd + 1
        } else {
            num_2nd
        };
        // Add one peak section for gain if needed
        let has_peak = self.gain_db.abs() > 0.001;
        let total = allpass_sections + if has_peak { 1 } else { 0 };
        self.num_sections = total.min(MAX_SECTIONS);

        let mut section_idx = 0;

        if has_first_order && section_idx < self.num_sections {
            let c = coeff::allpass_1(self.freq_hz, config.sample_rate);
            self.set_section_coeffs(section_idx, c);
            section_idx += 1;
        }

        let allpass_2nd_count =
            num_2nd.min(self.num_sections - section_idx - if has_peak { 1 } else { 0 });
        for i in 0..allpass_2nd_count {
            let q_section = butterworth_q(num_2nd, i);
            let c = coeff::calculate(
                FilterType::Allpass,
                self.freq_hz,
                q_section,
                0.0,
                config.sample_rate,
            );
            self.set_section_coeffs(section_idx, c);
            section_idx += 1;
        }

        // Add peak section for the gain component
        if has_peak && section_idx < self.num_sections {
            let c_peak = coeff::calculate(
                FilterType::Peak,
                self.freq_hz,
                self.q,
                self.gain_db,
                config.sample_rate,
            );
            self.set_section_coeffs(section_idx, c_peak);
        }
    }

    /// Flat tilt: constant dB/octave slope via Julius Smith's spectral tilt.
    ///
    /// Cascades N first-order pole-zero pairs at geometrically spaced frequencies.
    /// The slope parameter (alpha) maps from gain_db: alpha = gain_db / (6 * octaves).
    fn update_flat_tilt(&mut self, _order: usize, config: AudioConfig) {
        let sr = config.sample_rate;

        // Number of first-order sections to use (more = smoother approximation)
        let n = MAX_SECTIONS;
        self.num_sections = n;

        // Frequency range covering the audible spectrum
        let f_lo = 10.0; // Hz
        let f_hi = sr / 2.0 - 100.0; // Just below Nyquist

        // Total octaves from pivot to edges of the audible range
        let total_octaves = (20000.0_f64 / 20.0).log2(); // ~10 octaves

        // Alpha controls the slope: alpha=1 → +6 dB/octave, alpha=-1 → -6 dB/octave
        let alpha = self.gain_db / (6.0 * total_octaves);

        // Geometric spacing ratio for poles
        let r = (f_hi / f_lo).powf(1.0 / (n - 1) as f64);

        for i in 0..n {
            // Pole frequency
            let fp = f_lo * r.powi(i as i32);
            // Zero frequency: shifted by alpha
            let fz = f_lo * r.powf(-alpha + i as f64);

            let c = coeff::flat_tilt_section(fp, fz, self.freq_hz, sr);
            self.set_section_coeffs(i, c);
        }
    }

    fn set_section_coeffs(&mut self, idx: usize, coeffs: coeff::Coeffs) {
        // Stability check: fall back to passthrough if coefficients blow up
        let stable = coeffs.iter().all(|c| c.is_finite() && c.abs() < 1e12);
        let coeffs = if stable { coeffs } else { coeff::PASSTHROUGH };
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
        sample * self.output_gain
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

/// Butterworth Q value for the i-th 2nd-order section of an n-section cascade.
///
/// For an n-section (2n-pole) Butterworth filter, each section's Q is:
///   Q_i = 1 / (2 * cos(π * (2i + 1) / (4n)))
fn butterworth_q(n: usize, i: usize) -> f64 {
    let angle = PI * (2 * i + 1) as f64 / (4 * n) as f64;
    0.5 / angle.cos()
}
