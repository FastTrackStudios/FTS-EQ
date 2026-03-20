//! NaN-free tests — edge-case inputs must never produce NaN or infinity.
//!
//! r[verify test.dsp.nan-free]

use eq_dsp::band::Band;
use eq_dsp::filter_type::{FilterStructure, FilterType};
use eq_dsp::test_util::*;

const SAMPLE_RATE: f64 = 48000.0;

fn test_band_with_signal(label: &str, signal: &[f64]) {
    let types = [
        (FilterType::Peak, 6.0),
        (FilterType::LowShelf, 6.0),
        (FilterType::HighShelf, 6.0),
        (FilterType::TiltShelf, 6.0),
        (FilterType::Lowpass, 0.0),
        (FilterType::Highpass, 0.0),
        (FilterType::Bandpass, 0.0),
        (FilterType::Notch, 0.0),
    ];

    for (ft, gain) in types {
        for structure in [FilterStructure::Tdf2, FilterStructure::Svf] {
            let mut band = Band::new();
            band.filter_type = ft;
            band.structure = structure;
            band.freq_hz = 1000.0;
            band.q = 0.707;
            band.gain_db = gain;
            band.order = 2;
            band.update(test_config(SAMPLE_RATE));

            let output = process_band_mono(&mut band, signal);
            assert_all_finite(&output, &format!("{label}/{ft:?}/{structure:?}"));
        }
    }
}

#[test]
fn silence_input() {
    test_band_with_signal("silence", &silence(4096));
}

#[test]
fn dc_offset_positive() {
    test_band_with_signal("dc+1", &dc_offset(4096, 1.0));
}

#[test]
fn dc_offset_negative() {
    test_band_with_signal("dc-1", &dc_offset(4096, -1.0));
}

#[test]
fn full_scale_input() {
    // Very large but finite values
    test_band_with_signal("large", &dc_offset(4096, 1e6));
}

#[test]
fn impulse_input() {
    test_band_with_signal("impulse", &impulse(4096));
}

#[test]
fn white_noise_input() {
    test_band_with_signal("noise", &white_noise(4096, 12345));
}

#[test]
fn square_wave_input() {
    test_band_with_signal("square", &square_wave(4096, 440.0, SAMPLE_RATE));
}

#[test]
fn alternating_polarity() {
    let signal: Vec<f64> = (0..4096)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    test_band_with_signal("alternating", &signal);
}

// Edge case: extreme Q values
#[test]
fn extreme_q_values() {
    for q in [0.01, 0.1, 10.0, 100.0] {
        let mut band = Band::new();
        band.filter_type = FilterType::Peak;
        band.structure = FilterStructure::Tdf2;
        band.freq_hz = 1000.0;
        band.q = q;
        band.gain_db = 12.0;
        band.order = 2;
        band.update(test_config(SAMPLE_RATE));

        let input = white_noise(4096, 42);
        let output = process_band_mono(&mut band, &input);
        assert_all_finite(&output, &format!("extreme_q={q}"));
    }
}

// Edge case: extreme frequencies
#[test]
fn extreme_frequencies() {
    for freq in [20.0, 50.0, 10000.0, 20000.0, 23000.0] {
        let mut band = Band::new();
        band.filter_type = FilterType::Peak;
        band.structure = FilterStructure::Tdf2;
        band.freq_hz = freq;
        band.q = 0.707;
        band.gain_db = 6.0;
        band.order = 2;
        band.update(test_config(SAMPLE_RATE));

        let input = white_noise(4096, 42);
        let output = process_band_mono(&mut band, &input);
        assert_all_finite(&output, &format!("extreme_freq={freq}"));
    }
}

// Edge case: higher orders
#[test]
fn higher_orders_stable() {
    for order in [4, 6, 8, 10, 12] {
        let mut band = Band::new();
        band.filter_type = FilterType::Lowpass;
        band.structure = FilterStructure::Tdf2;
        band.freq_hz = 1000.0;
        band.q = 0.707;
        band.order = order;
        band.update(test_config(SAMPLE_RATE));

        let input = white_noise(4096, 42);
        let output = process_band_mono(&mut band, &input);
        assert_all_finite(&output, &format!("order={order}"));
    }
}
