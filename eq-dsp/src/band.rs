//! EQ band — a complete filter band with variable order.
//!
//! Cascades multiple 2nd-order sections with Butterworth-distributed Q
//! values for higher-order slopes. Supports all 9 filter types.

use std::f64::consts::{PI, SQRT_2};

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
    /// Tuning: peak cascade Q compensation fraction (0.0–1.0).
    /// Fraction of theoretical bandwidth narrowing correction applied.
    pub peak_q_comp: f64,

    // Processing state — one of these is active
    tdf2: [Tdf2Section; MAX_SECTIONS],
    svf: [SvfSection; MAX_SECTIONS],
    num_sections: usize,
    /// Output gain multiplier for filter types that use gain as output level
    /// (lowpass, highpass, bandpass, notch, allpass).
    output_gain: f64,
    /// Parallel bell gain scale: non-zero enables the parallel bell topology.
    ///
    /// For higher-order bells: y[n] = x[n] + parallel_bell_gain * bp_cascade[n]
    /// where bp_cascade is `num_sections` BP biquads applied in series.
    /// This implements H_bell = 1 + (G_linear - 1) * H_BP, matching Pro-Q 4's
    /// flat-top bell shape (vs cascaded peak biquads which over-estimate skirt gain).
    parallel_bell_gain: f64,
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
            peak_q_comp: 0.105,
            tdf2: std::array::from_fn(|_| Tdf2Section::new()),
            svf: std::array::from_fn(|_| SvfSection::new()),
            num_sections: 1,
            output_gain: 1.0,
            parallel_bell_gain: 0.0,
        }
    }

    /// Recalculate coefficients for all sections.
    pub fn update(&mut self, config: AudioConfig) {
        if !self.enabled {
            return;
        }

        let order = self.order.clamp(0, MAX_ORDER);

        // Reset per-update state
        self.output_gain = 1.0;
        self.parallel_bell_gain = 0.0;

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
            // Use matched 1st-order to avoid the bilinear LP zero at Nyquist /
            // HP zero at DC that causes 40-60 dB excess attenuation in odd-order cascades.
            let c = match self.filter_type {
                FilterType::Lowpass => coeff::lowpass_1_matched(self.freq_hz, config.sample_rate),
                FilterType::Highpass => coeff::highpass_1_matched(self.freq_hz, config.sample_rate),
                _ => unreachable!(),
            };
            self.set_section_coeffs(section_idx, c);
            section_idx += 1;
        }

        // Butterworth Q cascade with user-scaled Q on the resonant section.
        // butterworth_q_for_order gives the highest Q at i=0 (pole pair farthest from
        // the real axis, nearest the imaginary axis). This is the section responsible
        // for the resonance peak near fc, so user Q must scale section i=0.
        // At display Q=1.0 (self.q = 1/√2), the filter is true Butterworth.
        let num_biquads = num_2nd.min(self.num_sections - section_idx);
        for i in 0..num_biquads {
            let bw_q = butterworth_q_for_order(order, i);
            let q_section = if i == 0 { bw_q * self.q * SQRT_2 } else { bw_q };
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
            // Shelf Q compression: Pro-Q 4's display Q maps to internal Q
            // via √(q_display) scaling. self.q = q_display / √2, so
            // q_compressed = √(self.q * √2) / √2 = √(q_display) / √2.
            let q_shelf = (self.q * SQRT_2).sqrt() * std::f64::consts::FRAC_1_SQRT_2;
            let c = coeff::calculate(
                self.filter_type,
                self.freq_hz,
                q_shelf,
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

        let q_user = self.q * std::f64::consts::SQRT_2;

        // Gain distribution between 1st-order and biquad sections.
        // For order=3 (1st-order + single biquad), minimize the 1st-order
        // contribution. Its Vicanek matched design has a fundamental ~0.06 dB
        // error floor, while the biquad's impulse-invariance poles +
        // magnitude matching handles the full frequency range much better.
        let (gain_1st, gain_2nd) = if order == 3 {
            (effective_gain_db * 0.05, effective_gain_db * 0.95)
        } else {
            let gain_per_pole = effective_gain_db / order as f64;
            (gain_per_pole, gain_per_pole * 2.0)
        };

        let mut section_idx = 0;

        if has_first_order && section_idx < self.num_sections {
            let c = match self.filter_type {
                FilterType::LowShelf => {
                    coeff::low_shelf_1(self.freq_hz, gain_1st, config.sample_rate)
                }
                FilterType::HighShelf => {
                    coeff::high_shelf_1(self.freq_hz, gain_1st, config.sample_rate)
                }
                FilterType::TiltShelf => {
                    coeff::tilt_shelf_1(self.freq_hz, gain_1st, config.sample_rate)
                }
                _ => unreachable!(),
            };
            self.set_section_coeffs(section_idx, c);
            section_idx += 1;
        }

        let num_biquads = num_2nd.min(self.num_sections - section_idx);
        for i in 0..num_biquads {
            let is_last = i == num_biquads - 1;
            let is_second_last = i == num_biquads.saturating_sub(2) && num_biquads > 1;
            let bw_q = butterworth_q(num_2nd, i);
            let q_section = if num_biquads == 1 {
                // Single biquad (order=3): apply shelf Q compression.
                // Shelf resonance Q follows √(q_display) scaling, giving
                // q_internal = √(q_display) / √2 = √(self.q * √2) / √2.
                (self.q * std::f64::consts::SQRT_2).sqrt() * std::f64::consts::FRAC_1_SQRT_2
            } else if is_last {
                let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                let scale = if q_user > 1.0 {
                    1.0 + (q_user.ln() * 1.03 * blend)
                } else {
                    q_user.powf(blend * 0.75)
                };
                bw_q * scale
            } else if is_second_last && order >= 6 {
                // For high-order cascades, spread some Q influence to the
                // second-to-last section to reduce extreme resonance
                // concentration on the last section.
                let blend = (1.0 - (order as f64 - 3.0) / 10.0).clamp(0.5, 1.0);
                let scale = if q_user > 1.0 {
                    1.0 + (q_user.ln() * 0.3 * blend)
                } else {
                    1.0
                };
                bw_q * scale
            } else {
                bw_q
            };
            let c = coeff::calculate(
                self.filter_type,
                self.freq_hz,
                q_section,
                gain_2nd,
                config.sample_rate,
            );
            self.set_section_coeffs(section_idx, c);
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

        // Higher-order bell: cascade of identical peak biquads with Q compensation.
        // Each biquad shares the gain equally (gain_db / n_sections per section).
        let n = order / 2;
        self.num_sections = n.min(MAX_SECTIONS);

        let gain_per_section = self.gain_db / self.num_sections as f64;

        for i in 0..self.num_sections {
            let c = coeff::calculate(
                FilterType::Peak,
                self.freq_hz,
                self.q,
                gain_per_section,
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

    /// Bandpass via cascaded 2nd-order BPF sections.
    ///
    /// Uses Butterworth pole placement: each section gets a different Q
    /// from the Butterworth distribution, with the last section scaled by
    /// the user's Q for resonance control.
    fn update_bandpass(&mut self, order: usize, config: AudioConfig) {
        if order == 0 {
            self.num_sections = 0;
            return;
        }

        let num_sections = (order / 2).max(1).min(MAX_SECTIONS);
        self.num_sections = num_sections;

        for i in 0..num_sections {
            let q_section = if num_sections == 1 {
                self.q
            } else if i == num_sections - 1 {
                // Last section: user Q controls resonance
                self.q
            } else {
                // Inner sections: Butterworth Q distribution
                butterworth_q(num_sections, i)
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

        // Increase Q per section to compensate for cascade bandwidth expansion.
        // Cascading N identical notch sections widens the -3dB bandwidth;
        // scaling Q by N^0.1 approximately preserves the apparent bandwidth.
        let q_compensated = self.q * (self.num_sections as f64).powf(0.1);

        for i in 0..self.num_sections {
            let c = coeff::calculate(
                FilterType::Notch,
                self.freq_hz,
                q_compensated,
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

    /// Flat tilt: constant dB/octave slope via cascaded first-order shelves.
    ///
    /// Each section is a matched 1st-order shelf at a geometrically spaced frequency.
    /// The gain per section is chosen so the cascade produces a linear (in log-freq)
    /// tilt centered at the pivot frequency.
    fn update_flat_tilt(&mut self, _order: usize, config: AudioConfig) {
        use std::f64::consts::PI;
        let sr = config.sample_rate;

        let n = MAX_SECTIONS;
        self.num_sections = n;

        if self.gain_db.abs() < 1e-6 {
            for i in 0..n {
                self.set_section_coeffs(i, coeff::PASSTHROUGH);
            }
            return;
        }

        // Frequency range for the pole/zero placement
        let f_lo = 14.0;
        let f_hi = (sr * 0.43).min(20000.0);

        // Geometric spacing ratio
        let r = (f_hi / f_lo).powf(1.0 / (n - 1) as f64);

        // Desired slope calibrated to Pro-Q 4: slope ≈ gain_db / 5.0 dB/oct
        let desired_slope = self.gain_db / 5.0;

        // Helper: measure cascade gain at a frequency
        let measure_cascade = |secs: &[coeff::Coeffs], freq: f64| -> f64 {
            let w = 2.0 * PI * freq / sr;
            let cw = w.cos();
            let sw = w.sin();
            let mut ms = 1.0;
            for c in secs {
                let nr = c[3] + c[4] * cw;
                let ni = -c[4] * sw;
                let dr = c[0] + c[1] * cw;
                let di = -c[1] * sw;
                let nm = nr * nr + ni * ni;
                let dm = dr * dr + di * di;
                if dm > 1e-30 {
                    ms *= nm / dm;
                }
            }
            10.0 * ms.log10()
        };

        // Build with trial gain, measure actual slope, rescale to match desired
        let trial_gain = self.gain_db / n as f64;
        let trial_sections: Vec<coeff::Coeffs> = (0..n)
            .map(|i| coeff::high_shelf_1(f_lo * r.powi(i as i32), trial_gain, sr))
            .collect();
        let g_lo = measure_cascade(&trial_sections, 20.0);
        let g_hi = measure_cascade(&trial_sections, 20000.0);
        let trial_slope = (g_hi - g_lo) / (20000.0_f64 / 20.0).log2();
        let scale = if trial_slope.abs() > 1e-10 {
            desired_slope / trial_slope
        } else {
            1.0
        };
        let gain_per_section = trial_gain * scale;

        // Build final sections with calibrated gain
        let sections: Vec<coeff::Coeffs> = (0..n)
            .map(|i| coeff::high_shelf_1(f_lo * r.powi(i as i32), gain_per_section, sr))
            .collect();

        // Normalize at pivot frequency (0 dB at pivot)
        let pivot_gain_db = measure_cascade(&sections, self.freq_hz);
        let norm_linear = 10.0_f64.powf(-pivot_gain_db / 20.0);
        for (i, c) in sections.into_iter().enumerate() {
            let c = if i == 0 {
                [
                    c[0],
                    c[1],
                    c[2],
                    c[3] * norm_linear,
                    c[4] * norm_linear,
                    c[5],
                ]
            } else {
                c
            };
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
    pub fn tick(&mut self, sample: f64, ch: usize) -> f64 {
        if !self.enabled {
            return sample;
        }

        if self.parallel_bell_gain != 0.0 {
            // Parallel bell: y = x + (G - 1) * BP_cascade(x)
            // The BP biquads are stored in the sections array.
            let mut bp = sample;
            match self.structure {
                FilterStructure::Tdf2 => {
                    for i in 0..self.num_sections {
                        bp = self.tdf2[i].tick(bp, ch);
                    }
                }
                FilterStructure::Svf => {
                    for i in 0..self.num_sections {
                        bp = self.svf[i].tick(bp, ch);
                    }
                }
            }
            return sample + self.parallel_bell_gain * bp;
        }

        let mut out = sample;
        match self.structure {
            FilterStructure::Tdf2 => {
                for i in 0..self.num_sections {
                    out = self.tdf2[i].tick(out, ch);
                }
            }
            FilterStructure::Svf => {
                for i in 0..self.num_sections {
                    out = self.svf[i].tick(out, ch);
                }
            }
        }
        out * self.output_gain
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

/// Butterworth Q for the i-th biquad section based on total filter order.
///
/// For odd-order filters with a 1st-order + biquad cascade, the Q values
/// must account for all poles (including the real pole). The correct formula
/// uses the total order, not the number of biquad sections:
///   Q_i = 1 / (2 * sin(π * (2i + 1) / (2 * order)))
fn butterworth_q_for_order(order: usize, i: usize) -> f64 {
    let angle = PI * (2 * i + 1) as f64 / (2 * order) as f64;
    0.5 / angle.sin()
}
