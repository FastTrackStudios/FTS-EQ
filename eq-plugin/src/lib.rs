//! FTS EQ — nih-plug entry point with 24-band parametric EQ and Dioxus GUI.
//!
//! Pro-Q 4 style parametric equalizer with draggable band nodes,
//! frequency response visualization, and per-band filter type selection.

use atomic_float::AtomicF32;
use fts_plugin_core::prelude::*;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use eq_dsp::filter_type::{FilterStructure, FilterType};
use eq_dsp::EqChain;
use fts_dsp::{AudioConfig, Processor};

mod editor;

// ── Constants ───────────────────────────────────────────────────────

/// Number of parametric bands (Pro-Q style).
pub const NUM_BANDS: usize = 24;

// ── Shared UI State ─────────────────────────────────────────────────

/// Audio-thread → UI metering data.
pub struct EqUiState {
    pub params: Arc<FtsEqParams>,
    /// Peak input level in dB.
    pub input_peak_db: AtomicF32,
    /// Peak output level in dB.
    pub output_peak_db: AtomicF32,
}

impl EqUiState {
    fn new(params: Arc<FtsEqParams>) -> Self {
        Self {
            params,
            input_peak_db: AtomicF32::new(-100.0),
            output_peak_db: AtomicF32::new(-100.0),
        }
    }
}

// ── Per-Band Parameters ──────────────────────────────────────────────

#[derive(Params)]
pub struct BandParams {
    #[id = "on"]
    pub enabled: FloatParam,

    #[id = "type"]
    pub filter_type: IntParam,

    #[id = "freq"]
    pub freq_hz: FloatParam,

    #[id = "gain"]
    pub gain_db: FloatParam,

    #[id = "q"]
    pub q: FloatParam,
}

impl BandParams {
    fn new(idx: usize) -> Self {
        Self {
            enabled: FloatParam::new(
                &format!("B{} On", idx + 1),
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_value_to_string(Arc::new(|v| {
                if v > 0.5 {
                    "On".to_string()
                } else {
                    "Off".to_string()
                }
            }))
            .with_string_to_value(Arc::new(|s| {
                match s.trim().to_lowercase().as_str() {
                    "on" | "1" | "true" => Some(1.0),
                    "off" | "0" | "false" => Some(0.0),
                    _ => s.parse().ok(),
                }
            })),

            filter_type: IntParam::new(
                &format!("B{} Type", idx + 1),
                0, // Bell
                IntRange::Linear { min: 0, max: 9 },
            )
            .with_value_to_string(Arc::new(|v| match v {
                0 => "Bell".to_string(),
                1 => "Low Shelf".to_string(),
                2 => "High Shelf".to_string(),
                3 => "Low Cut".to_string(),
                4 => "High Cut".to_string(),
                5 => "Notch".to_string(),
                6 => "Bandpass".to_string(),
                7 => "Tilt Shelf".to_string(),
                8 => "Flat Tilt".to_string(),
                9 => "All Pass".to_string(),
                _ => "Bell".to_string(),
            })),

            freq_hz: FloatParam::new(
                &format!("B{} Freq", idx + 1),
                1000.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_unit(" Hz")
            .with_value_to_string(Arc::new(|v| {
                if v >= 1000.0 {
                    format!("{:.1}k", v / 1000.0)
                } else {
                    format!("{:.0}", v)
                }
            })),

            gain_db: FloatParam::new(
                &format!("B{} Gain", idx + 1),
                0.0,
                FloatRange::Linear {
                    min: -30.0,
                    max: 30.0,
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),

            q: FloatParam::new(
                &format!("B{} Q", idx + 1),
                1.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 18.0,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_value_to_string(formatters::v2s_f32_rounded(2)),
        }
    }
}

// ── Plugin Parameters ────────────────────────────────────────────────

#[derive(Params)]
pub struct FtsEqParams {
    #[id = "output_gain"]
    pub output_gain_db: FloatParam,

    #[nested(array, group = "Band {}")]
    pub bands: [BandParams; NUM_BANDS],
}

impl Default for FtsEqParams {
    fn default() -> Self {
        Self {
            output_gain_db: FloatParam::new(
                "Output",
                0.0,
                FloatRange::Linear {
                    min: -24.0,
                    max: 24.0,
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_rounded(1)),

            bands: std::array::from_fn(|i| BandParams::new(i)),
        }
    }
}

// ── Plugin ──────────────────────────────────────────────────────────

struct FtsEq {
    params: Arc<FtsEqParams>,
    ui_state: Arc<EqUiState>,
    editor_state: Arc<DioxusState>,
    chain: EqChain,
    sample_rate: f64,
}

impl Default for FtsEq {
    fn default() -> Self {
        let params = Arc::new(FtsEqParams::default());
        let ui_state = Arc::new(EqUiState::new(params.clone()));
        let mut chain = EqChain::new();
        // Pre-allocate all 24 bands
        for _ in 0..NUM_BANDS {
            chain.add_band();
        }
        Self {
            params,
            ui_state,
            editor_state: DioxusState::new(|| (1000, 600)),
            chain,
            sample_rate: 48000.0,
        }
    }
}

/// Map EqBandShape integer to eq-dsp FilterType.
fn shape_to_filter_type(shape: i32) -> FilterType {
    match shape {
        0 => FilterType::Peak,      // Bell
        1 => FilterType::LowShelf,  // Low Shelf
        2 => FilterType::HighShelf, // High Shelf
        3 => FilterType::Highpass,  // Low Cut (cuts lows = highpass)
        4 => FilterType::Lowpass,   // High Cut (cuts highs = lowpass)
        5 => FilterType::Notch,     // Notch
        6 => FilterType::Bandpass,  // Bandpass
        7 => FilterType::TiltShelf, // Tilt Shelf
        8 => FilterType::TiltShelf, // Flat Tilt (use tilt shelf)
        9 => FilterType::Peak,      // All Pass (placeholder)
        _ => FilterType::Peak,
    }
}

impl FtsEq {
    /// Sync nih-plug params → eq-dsp bands.
    fn sync_params(&mut self) {
        for i in 0..NUM_BANDS {
            let bp = &self.params.bands[i];
            if let Some(band) = self.chain.band_mut(i) {
                let enabled = bp.enabled.value() > 0.5;
                let ft = shape_to_filter_type(bp.filter_type.value());
                let freq = bp.freq_hz.value() as f64;
                let gain = bp.gain_db.value() as f64;
                let q = bp.q.value() as f64;

                if band.enabled != enabled
                    || band.filter_type != ft
                    || (band.freq_hz - freq).abs() > 0.01
                    || (band.gain_db - gain).abs() > 0.01
                    || (band.q - q).abs() > 0.001
                {
                    band.enabled = enabled;
                    band.filter_type = ft;
                    band.freq_hz = freq;
                    band.gain_db = gain;
                    band.q = q;
                    band.structure = FilterStructure::Tdf2;
                    self.chain.update_band(i);
                }
            }
        }
    }
}

impl Plugin for FtsEq {
    const NAME: &'static str = "FTS EQ";
    const VENDOR: &'static str = "FastTrackStudio";
    const URL: &'static str = "";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        create_dioxus_editor_with_state(
            self.editor_state.clone(),
            self.ui_state.clone(),
            editor::App,
        )
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = buffer_config.sample_rate as f64;
        self.chain.update(AudioConfig {
            sample_rate: self.sample_rate,
            max_buffer_size: buffer_config.max_buffer_size as usize,
        });
        true
    }

    fn reset(&mut self) {
        self.chain.reset();
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        self.sync_params();

        let output_gain = fts_dsp::db::db_to_linear(self.params.output_gain_db.value() as f64);

        for mut frame in buffer.iter_samples() {
            let mut channels = frame.iter_mut();
            let left_ref = channels.next().unwrap();
            let right_ref = channels.next().unwrap();

            let mut left = *left_ref as f64;
            let mut right = *right_ref as f64;

            // Track input peak
            let input_peak = left.abs().max(right.abs()) as f32;

            // Process through all EQ bands
            for i in 0..NUM_BANDS {
                if let Some(band) = self.chain.band_mut(i) {
                    left = band.tick(left, 0);
                    right = band.tick(right, 1);
                }
            }

            // Apply output gain
            left *= output_gain;
            right *= output_gain;

            *left_ref = left as f32;
            *right_ref = right as f32;

            // Track output peak
            let output_peak = (left.abs().max(right.abs())) as f32;

            // Update metering (exponential peak decay)
            let prev_in = self.ui_state.input_peak_db.load(Ordering::Relaxed);
            let in_db = if input_peak > 0.0 {
                20.0 * input_peak.log10()
            } else {
                -100.0
            };
            let new_in = if in_db > prev_in {
                in_db
            } else {
                prev_in - 0.3
            };
            self.ui_state.input_peak_db.store(new_in, Ordering::Relaxed);

            let prev_out = self.ui_state.output_peak_db.load(Ordering::Relaxed);
            let out_db = if output_peak > 0.0 {
                20.0 * output_peak.log10()
            } else {
                -100.0
            };
            let new_out = if out_db > prev_out {
                out_db
            } else {
                prev_out - 0.3
            };
            self.ui_state
                .output_peak_db
                .store(new_out, Ordering::Relaxed);
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for FtsEq {
    const CLAP_ID: &'static str = "com.fasttrackstudio.eq";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("24-band parametric EQ");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Equalizer,
        ClapFeature::Stereo,
    ];
}

impl Vst3Plugin for FtsEq {
    const VST3_CLASS_ID: [u8; 16] = *b"FtsEqPlugin00001";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Eq];
}

nih_export_clap!(FtsEq);
nih_export_vst3!(FtsEq);
