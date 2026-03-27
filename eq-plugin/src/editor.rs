//! EQ editor — Dioxus GUI root component.
//!
//! Uses the Pro-Q style EqGraph component from audio-gui for the main
//! frequency response visualization with draggable band nodes.

use std::sync::atomic::Ordering;

use audio_gui::prelude::{use_init_theme, DragProvider, LevelMeterDb};
use audio_gui::viz::eq_graph::{get_band_color, EqBand, EqBandShape, EqGraph};
use fts_plugin_core::prelude::*;

use crate::{EqUiState, NUM_BANDS, SPECTRUM_BINS};

/// Map EqBandShape to the integer filter type parameter value.
fn shape_to_int(shape: EqBandShape) -> i32 {
    match shape {
        EqBandShape::Bell => 0,
        EqBandShape::LowShelf => 1,
        EqBandShape::LowCut => 2,
        EqBandShape::HighShelf => 3,
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
        2 => EqBandShape::LowCut,
        3 => EqBandShape::HighShelf,
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
    let t = use_init_theme();
    let t = *t.read();

    let shared = use_context::<SharedState>();
    let ui = shared.get::<EqUiState>().expect("EqUiState missing");
    let ctx = use_param_context();
    let params = &ui.params;

    // Read metering values
    let input_db = ui.input_peak_db.load(Ordering::Relaxed);
    let output_db = ui.output_peak_db.load(Ordering::Relaxed);
    let sample_rate = ui.sample_rate.load(Ordering::Relaxed) as f64;

    // Read dB range from parameter (0=6, 1=12, 2=18, 3=24, 4=30)
    let db_range_idx = params.db_range.value();
    let db_range: f64 = match db_range_idx {
        0 => 6.0,
        1 => 12.0,
        2 => 18.0,
        3 => 24.0,
        _ => 30.0,
    };

    // Read spectrum bins
    let spectrum: Vec<f32> = (0..SPECTRUM_BINS)
        .map(|i| ui.spectrum_bins[i].load(Ordering::Relaxed))
        .collect();

    // Focused band for detail panel
    let focused_band: Signal<Option<usize>> = use_signal(|| None);

    // Read gain scale (0-200%)
    let gain_scale = params.gain_scale.value() / 100.0;

    // Build EqBand vec from current parameter state (gains scaled for display)
    let mut bands_vec: Vec<EqBand> = Vec::with_capacity(NUM_BANDS);
    for i in 0..NUM_BANDS {
        let bp = &params.bands[i];
        bands_vec.push(EqBand {
            index: i,
            used: bp.enabled.value() > 0.5,
            enabled: bp.enabled.value() > 0.5,
            frequency: bp.freq_hz.value(),
            gain: bp.gain_db.value() * gain_scale,
            q: bp.q.value(),
            shape: int_to_shape(bp.filter_type.value()),
            solo: bp.solo.value() > 0.5,
            stereo_mode: Default::default(),
        });
    }

    let mut bands_signal = use_signal(|| bands_vec.clone());
    // Update from params each render
    *bands_signal.write() = bands_vec;

    // Count active bands
    let active_count = bands_signal.read().iter().filter(|b| b.used).count();

    let base_css = t.base_css();
    let root_style = t.root_style();
    let spacing_root = t.spacing_root;
    let spacing_section = t.spacing_section;
    let spacing_tight = t.spacing_tight;
    let border = t.border;
    let shadow_subtle = t.shadow_subtle;
    let font_size_title = t.font_size_title;
    let text_bright = t.text_bright;
    let _text_dim = t.text_dim;
    let style_label = t.style_label();
    let style_inset = t.style_inset();
    let style_card = t.style_card();
    let _style_value = t.style_value();

    rsx! {
        document::Style { {base_css} }

        DragProvider {
        div {
            style: format!(
                "{root_style} display:flex; flex-direction:column; overflow:hidden;",
            ),

            // ── Header ───────────────────────────────────────────
            div {
                style: format!(
                    "display:flex; justify-content:space-between; align-items:center; \
                     padding:{spacing_root}; border-bottom:1px solid {border}; \
                     box-shadow:{shadow_subtle};",
                ),
                div {
                    style: "display:flex; align-items:baseline; gap:12px;",
                    div {
                        style: format!(
                            "font-size:{font_size_title}; font-weight:700; letter-spacing:0.5px; color:{text_bright};",
                        ),
                        "FTS EQ"
                    }
                    div {
                        style: format!("{style_label}"),
                        "{active_count} bands active"
                    }
                }
                div {
                    style: format!("{style_label}"),
                    "FastTrackStudio"
                }
            }

            // ── Main EQ graph ────────────────────────────────────
            div {
                style: format!(
                    "{style_inset} flex:1; min-height:0; position:relative; margin:4px 6px;",
                ),
                EqGraph {
                    bands: bands_signal,
                    db_range: db_range,
                    sample_rate: sample_rate,
                    spectrum_db: spectrum,
                    focused_band_out: focused_band,
                    // Blitz element_coordinates() returns element-relative
                    // coords (confirmed by debug: elem==client for the SVG).
                    // rendered_width/height = actual pixel size of the SVG element.
                    rendered_width: 1000.0,
                    rendered_height: 511.0,
                    offset_x: 0.0,
                    offset_y: 0.0,

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

                                // Update solo state
                                let solo_val = if band.solo { 1.0_f32 } else { 0.0 };
                                ctx.begin_set_raw(bp.solo.as_ptr());
                                ctx.set_normalized_raw(
                                    bp.solo.as_ptr(),
                                    bp.solo.preview_normalized(solo_val),
                                );
                                ctx.end_set_raw(bp.solo.as_ptr());
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
                    "{style_card} display:flex; align-items:center; gap:{spacing_section}; \
                     padding:{spacing_root}; border-top:1px solid {border}; \
                     border-radius:0;",
                ),
                LevelMeterDb { level_db: input_db, label: "IN".to_string(), height: 40.0 }
                LevelMeterDb { level_db: output_db, label: "OUT".to_string(), height: 40.0 }

                // Band detail panel with inline controls
                {
                    let focus_idx = *focused_band.read();
                    if let Some(idx) = focus_idx {
                        let bp = &params.bands[idx];
                        let freq = bp.freq_hz.value();
                        let gain = bp.gain_db.value();
                        let q = bp.q.value();
                        let shape = int_to_shape(bp.filter_type.value());
                        let enabled = bp.enabled.value() > 0.5;
                        let color = get_band_color(idx);
                        let opacity = if enabled { "1.0" } else { "0.5" };

                        let freq_str = if freq >= 1000.0 {
                            format!("{:.1}k", freq / 1000.0)
                        } else {
                            format!("{:.0}", freq)
                        };

                        rsx! {
                            div {
                                style: format!(
                                    "display:flex; align-items:center; gap:12px; \
                                     padding:{spacing_tight} 10px; margin-left:8px; \
                                     border-left:3px solid {color}; opacity:{opacity};",
                                ),

                                // Band number
                                div {
                                    style: format!(
                                        "font-weight:700; font-size:13px; color:{color}; \
                                         min-width:24px; text-align:center;",
                                    ),
                                    "{idx + 1}"
                                }

                                // Bypass toggle
                                div {
                                    style: format!(
                                        "{style_label} cursor:pointer; padding:2px 6px; \
                                         border:1px solid {border}; border-radius:3px; \
                                         background:{bg};",
                                        bg = if enabled { "transparent" } else { "rgba(255,80,80,0.2)" },
                                    ),
                                    onclick: {
                                        let ctx = ctx.clone();
                                        let enabled_ptr = bp.enabled.as_ptr();
                                        move |_| {
                                            let new_val = if enabled { 0.0_f32 } else { 1.0 };
                                            ctx.begin_set_raw(enabled_ptr);
                                            ctx.set_normalized_raw(enabled_ptr, new_val);
                                            ctx.end_set_raw(enabled_ptr);
                                        }
                                    },
                                    if enabled { "ON" } else { "OFF" }
                                }

                                // Filter type (clickable to cycle)
                                div {
                                    style: format!(
                                        "{style_label} cursor:pointer; padding:2px 6px; \
                                         border:1px solid {border}; border-radius:3px;",
                                    ),
                                    title: "Click to cycle filter type",
                                    onclick: {
                                        let ctx = ctx.clone();
                                        let ft_ptr = bp.filter_type.as_ptr();
                                        let current_shape = shape_to_int(shape);
                                        move |_| {
                                            let next = (current_shape + 1) % 10;
                                            ctx.begin_set_raw(ft_ptr);
                                            ctx.set_normalized_raw(ft_ptr, next as f32 / 9.0);
                                            ctx.end_set_raw(ft_ptr);
                                        }
                                    },
                                    "{shape.label()}"
                                }

                                // Frequency knob
                                div {
                                    style: "display:flex; align-items:center; gap:4px;",
                                    span { style: format!("{style_label}"), "{freq_str} Hz" }
                                    audio_gui::controls::knob::Knob {
                                        param_ptr: bp.freq_hz.as_ptr(),
                                        size: audio_gui::controls::knob::KnobSize::Small,
                                    }
                                }

                                // Gain knob
                                div {
                                    style: "display:flex; align-items:center; gap:4px;",
                                    span { style: format!("{style_label}"), "{gain:+.1} dB" }
                                    audio_gui::controls::knob::Knob {
                                        param_ptr: bp.gain_db.as_ptr(),
                                        size: audio_gui::controls::knob::KnobSize::Small,
                                    }
                                }

                                // Q knob
                                div {
                                    style: "display:flex; align-items:center; gap:4px;",
                                    span { style: format!("{style_label}"), "Q {q:.2}" }
                                    audio_gui::controls::knob::Knob {
                                        param_ptr: bp.q.as_ptr(),
                                        size: audio_gui::controls::knob::KnobSize::Small,
                                    }
                                }

                                // Delete button
                                div {
                                    style: format!(
                                        "{style_label} cursor:pointer; padding:2px 6px; \
                                         border:1px solid {border}; border-radius:3px; \
                                         color:rgba(255,80,80,0.8);",
                                    ),
                                    onclick: {
                                        let ctx = ctx.clone();
                                        let enabled_ptr = bp.enabled.as_ptr();
                                        let gain_ptr = bp.gain_db.as_ptr();
                                        move |_| {
                                            ctx.begin_set_raw(enabled_ptr);
                                            ctx.set_normalized_raw(enabled_ptr, 0.0);
                                            ctx.end_set_raw(enabled_ptr);
                                            ctx.begin_set_raw(gain_ptr);
                                            ctx.set_normalized_raw(gain_ptr, 0.5);
                                            ctx.end_set_raw(gain_ptr);
                                        }
                                    },
                                    "DEL"
                                }
                            }
                        }
                    } else {
                        rsx! {
                            div {
                                style: format!(
                                    "{style_label} padding:{spacing_tight} 10px; margin-left:8px; \
                                     opacity:0.3;",
                                ),
                                "Click a band to edit"
                            }
                        }
                    }
                }

                div { style: "flex:1;" }

                // dB range selector
                {
                    let db_range_label = match db_range_idx {
                        0 => "6 dB",
                        1 => "12 dB",
                        2 => "18 dB",
                        3 => "24 dB",
                        _ => "30 dB",
                    };
                    rsx! {
                        div {
                            style: format!(
                                "{style_label} cursor:pointer; padding:4px 8px; \
                                 border:1px solid {border}; border-radius:4px; \
                                 user-select:none;",
                            ),
                            title: "Click to cycle dB range",
                            onclick: {
                                let ctx = ctx.clone();
                                let db_range_param = params.db_range.as_ptr();
                                let current = db_range_idx;
                                move |_| {
                                    let next = (current + 1) % 5;
                                    ctx.begin_set_raw(db_range_param);
                                    ctx.set_normalized_raw(
                                        db_range_param,
                                        next as f32 / 4.0,
                                    );
                                    ctx.end_set_raw(db_range_param);
                                }
                            },
                            "{db_range_label}"
                        }
                    }
                }

                // Gain scale knob
                div {
                    style: "display:flex; align-items:center; gap:8px;",
                    span {
                        style: format!("{style_label}"),
                        "Scale"
                    }
                    audio_gui::controls::knob::Knob {
                        param_ptr: params.gain_scale.as_ptr(),
                        size: audio_gui::controls::knob::KnobSize::Small,
                    }
                }

                // Output gain knob
                div {
                    style: "display:flex; align-items:center; gap:8px;",
                    span {
                        style: format!("{style_label}"),
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
