//! EQ editor — Dioxus GUI root component.
//!
//! Uses the Pro-Q style EqGraph component from audio-gui for the main
//! frequency response visualization with draggable band nodes.

use std::sync::atomic::Ordering;

use audio_gui::prelude::{theme, DragProvider, LevelMeterDb};
use audio_gui::viz::eq_graph::{EqBand, EqBandShape, EqGraph, get_band_color};
use fts_plugin_core::prelude::*;

use crate::{EqUiState, NUM_BANDS, SPECTRUM_BINS};

/// Map EqBandShape to the integer filter type parameter value.
fn shape_to_int(shape: EqBandShape) -> i32 {
    match shape {
        EqBandShape::Bell => 0,
        EqBandShape::LowShelf => 1,
        EqBandShape::HighShelf => 2,
        EqBandShape::LowCut => 3,
        EqBandShape::HighCut => 4,
        EqBandShape::Notch => 5,
        EqBandShape::BandPass => 6,
        EqBandShape::TiltShelf => 7,
        EqBandShape::FlatTilt => 8,
        EqBandShape::AllPass => 9,
    }
}

/// Map integer filter type parameter value to EqBandShape.
fn int_to_shape(v: i32) -> EqBandShape {
    match v {
        0 => EqBandShape::Bell,
        1 => EqBandShape::LowShelf,
        2 => EqBandShape::HighShelf,
        3 => EqBandShape::LowCut,
        4 => EqBandShape::HighCut,
        5 => EqBandShape::Notch,
        6 => EqBandShape::BandPass,
        7 => EqBandShape::TiltShelf,
        8 => EqBandShape::FlatTilt,
        9 => EqBandShape::AllPass,
        _ => EqBandShape::Bell,
    }
}

/// Root editor component.
#[component]
pub fn App() -> Element {
    let shared = use_context::<SharedState>();
    let ui = shared.get::<EqUiState>().expect("EqUiState missing");
    let ctx = use_param_context();
    let params = &ui.params;

    // Read metering values
    let input_db = ui.input_peak_db.load(Ordering::Relaxed);
    let output_db = ui.output_peak_db.load(Ordering::Relaxed);
    let sample_rate = ui.sample_rate.load(Ordering::Relaxed) as f64;

    // Read spectrum bins
    let spectrum: Vec<f32> = (0..SPECTRUM_BINS)
        .map(|i| ui.spectrum_bins[i].load(Ordering::Relaxed))
        .collect();

    // Focused band for detail panel
    let mut focused_band: Signal<Option<usize>> = use_signal(|| None);

    // Build EqBand vec from current parameter state
    let mut bands_vec: Vec<EqBand> = Vec::with_capacity(NUM_BANDS);
    for i in 0..NUM_BANDS {
        let bp = &params.bands[i];
        bands_vec.push(EqBand {
            index: i,
            used: bp.enabled.value() > 0.5,
            enabled: bp.enabled.value() > 0.5,
            frequency: bp.freq_hz.value(),
            gain: bp.gain_db.value(),
            q: bp.q.value(),
            shape: int_to_shape(bp.filter_type.value()),
            solo: false,
            stereo_mode: Default::default(),
        });
    }

    let mut bands_signal = use_signal(|| bands_vec.clone());
    // Update from params each render
    *bands_signal.write() = bands_vec;

    // Count active bands
    let active_count = bands_signal.read().iter().filter(|b| b.used).count();

    rsx! {
        document::Style { {theme::BASE_CSS} }

        DragProvider {
        div {
            style: format!(
                "width:100vw; height:100vh; \
                 background:{BG}; color:{TEXT}; \
                 font-family:system-ui,sans-serif; font-size:13px; user-select:none; \
                 display:flex; flex-direction:column; overflow:hidden;",
                BG = theme::BG, TEXT = theme::TEXT,
            ),

            // ── Header ───────────────────────────────────────────
            div {
                style: format!(
                    "display:flex; justify-content:space-between; align-items:center; \
                     padding:8px 14px; border-bottom:1px solid {BORDER};",
                    BORDER = theme::BORDER,
                ),
                div {
                    style: "display:flex; align-items:baseline; gap:12px;",
                    div {
                        style: "font-size:16px; font-weight:700; letter-spacing:0.5px;",
                        "FTS EQ"
                    }
                    div {
                        style: format!(
                            "font-size:11px; color:{};",
                            theme::TEXT_DIM,
                        ),
                        "{active_count} bands active"
                    }
                }
                div {
                    style: format!("font-size:11px; color:{};", theme::TEXT_DIM),
                    "FastTrackStudio"
                }
            }

            // ── Main EQ graph ────────────────────────────────────
            div {
                style: "flex:1; min-height:0; position:relative;",
                EqGraph {
                    bands: bands_signal,
                    db_range: 30.0,
                    sample_rate: sample_rate,
                    spectrum_db: spectrum,

                    on_focus_change: move |band_idx: Option<usize>| {
                        focused_band.set(band_idx);
                    },

                    on_band_change: {
                        let ctx = ctx.clone();
                        let params = ui.params.clone();
                        move |(idx, band): (usize, EqBand)| {
                            if idx < NUM_BANDS {
                                let bp = &params.bands[idx];

                                ctx.begin_set_raw(bp.freq_hz.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.freq_hz.as_ptr(),
                                    bp.freq_hz.preview_normalized(band.frequency),
                                );
                                ctx.end_set_raw(bp.freq_hz.as_ptr());

                                ctx.begin_set_raw(bp.gain_db.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.gain_db.as_ptr(),
                                    bp.gain_db.preview_normalized(band.gain),
                                );
                                ctx.end_set_raw(bp.gain_db.as_ptr());

                                ctx.begin_set_raw(bp.q.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.q.as_ptr(),
                                    bp.q.preview_normalized(band.q),
                                );
                                ctx.end_set_raw(bp.q.as_ptr());

                                // Update filter type
                                let shape_int = shape_to_int(band.shape);
                                ctx.begin_set_raw(bp.filter_type.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.filter_type.as_ptr(),
                                    bp.filter_type.preview_normalized(shape_int),
                                );
                                ctx.end_set_raw(bp.filter_type.as_ptr());

                                // Update enabled state
                                let enabled_val = if band.enabled { 1.0_f32 } else { 0.0 };
                                ctx.begin_set_raw(bp.enabled.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.enabled.as_ptr(),
                                    bp.enabled.preview_normalized(enabled_val),
                                );
                                ctx.end_set_raw(bp.enabled.as_ptr());
                            }
                        }
                    },

                    on_band_add: {
                        let ctx = ctx.clone();
                        let params = ui.params.clone();
                        move |band: EqBand| {
                            let idx = band.index;
                            if idx < NUM_BANDS {
                                let bp = &params.bands[idx];

                                // Enable the band
                                ctx.begin_set_raw(bp.enabled.as_ptr());
                                ctx.set_normalized_raw(bp.enabled.as_ptr(), 1.0);
                                ctx.end_set_raw(bp.enabled.as_ptr());

                                // Set frequency
                                ctx.begin_set_raw(bp.freq_hz.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.freq_hz.as_ptr(),
                                    bp.freq_hz.preview_normalized(band.frequency),
                                );
                                ctx.end_set_raw(bp.freq_hz.as_ptr());

                                // Set gain
                                ctx.begin_set_raw(bp.gain_db.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.gain_db.as_ptr(),
                                    bp.gain_db.preview_normalized(band.gain),
                                );
                                ctx.end_set_raw(bp.gain_db.as_ptr());

                                // Set Q
                                ctx.begin_set_raw(bp.q.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.q.as_ptr(),
                                    bp.q.preview_normalized(band.q),
                                );
                                ctx.end_set_raw(bp.q.as_ptr());

                                // Set filter type
                                let shape_int = shape_to_int(band.shape);
                                ctx.begin_set_raw(bp.filter_type.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.filter_type.as_ptr(),
                                    bp.filter_type.preview_normalized(shape_int),
                                );
                                ctx.end_set_raw(bp.filter_type.as_ptr());
                            }
                        }
                    },

                    on_band_remove: {
                        let ctx = ctx.clone();
                        let params = ui.params.clone();
                        move |idx: usize| {
                            if idx < NUM_BANDS {
                                let bp = &params.bands[idx];
                                // Disable the band
                                ctx.begin_set_raw(bp.enabled.as_ptr());
                                ctx.set_normalized_raw(bp.enabled.as_ptr(), 0.0);
                                ctx.end_set_raw(bp.enabled.as_ptr());

                                // Reset gain to 0
                                ctx.begin_set_raw(bp.gain_db.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.gain_db.as_ptr(),
                                    bp.gain_db.preview_normalized(0.0),
                                );
                                ctx.end_set_raw(bp.gain_db.as_ptr());
                            }
                        }
                    },
                }
            }

            // ── Bottom bar: meters + output gain ──────────────────
            div {
                style: format!(
                    "display:flex; align-items:center; gap:12px; \
                     padding:6px 14px; border-top:1px solid {BORDER}; \
                     background:{CARD_BG};",
                    BORDER = theme::BORDER, CARD_BG = theme::CARD_BG,
                ),
                LevelMeterDb { level_db: input_db, label: "IN".to_string(), height: 40.0 }
                LevelMeterDb { level_db: output_db, label: "OUT".to_string(), height: 40.0 }

                // Band detail panel (shown when a band is focused)
                if let Some(idx) = *focused_band.read() {
                    {
                        let bp = &params.bands[idx];
                        let freq = bp.freq_hz.value();
                        let gain = bp.gain_db.value();
                        let q = bp.q.value();
                        let shape = int_to_shape(bp.filter_type.value());
                        let enabled = bp.enabled.value() > 0.5;
                        let freq_str = if freq >= 1000.0 {
                            format!("{:.1}k Hz", freq / 1000.0)
                        } else {
                            format!("{:.0} Hz", freq)
                        };
                        let color = get_band_color(idx);

                        rsx! {
                            div {
                                style: format!(
                                    "display:flex; align-items:center; gap:10px; \
                                     padding:2px 10px; margin-left:8px; \
                                     border-left:2px solid {color}; \
                                     opacity:{op};",
                                    op = if enabled { "1.0" } else { "0.5" },
                                ),
                                span {
                                    style: format!(
                                        "font-size:11px; font-weight:600; color:{color};",
                                    ),
                                    "B{idx + 1}"
                                }
                                span {
                                    style: format!("font-size:10px; color:{};", theme::TEXT),
                                    "{shape.label()}"
                                }
                                span {
                                    style: format!(
                                        "font-size:10px; color:{}; font-variant-numeric:tabular-nums;",
                                        theme::TEXT_DIM,
                                    ),
                                    "{freq_str}  {gain:+.1} dB  Q {q:.2}"
                                }
                            }
                        }
                    }
                }

                div { style: "flex:1;" }

                // Output gain knob
                div {
                    style: "display:flex; align-items:center; gap:8px;",
                    span {
                        style: format!(
                            "font-size:10px; color:{}; text-transform:uppercase;",
                            theme::TEXT_DIM,
                        ),
                        "Output"
                    }
                    audio_gui::controls::knob::Knob {
                        param_ptr: params.output_gain_db.as_ptr(),
                        size: audio_gui::controls::knob::KnobSize::Small,
                    }
                }
            }
        }
        } // DragProvider
    }
}
