//! Reset state tests — verify reset() returns processor to initial state.
//!
//! r[verify test.dsp.reset-state]

use eq_dsp::band::Band;
use eq_dsp::chain::EqChain;
use eq_dsp::filter_type::{FilterStructure, FilterType};
use eq_dsp::test_util::*;
use fts_dsp::Processor;

const SAMPLE_RATE: f64 = 48000.0;

#[test]
fn band_reset_produces_silence() {
    let types = [
        FilterType::Peak,
        FilterType::LowShelf,
        FilterType::HighShelf,
        FilterType::Lowpass,
        FilterType::Highpass,
        FilterType::Bandpass,
        FilterType::Notch,
    ];

    for ft in types {
        for structure in [FilterStructure::Tdf2, FilterStructure::Svf] {
            let mut band = Band::new();
            band.filter_type = ft;
            band.structure = structure;
            band.freq_hz = 1000.0;
            band.q = 0.707;
            band.gain_db = 6.0;
            band.order = 2;
            band.update(test_config(SAMPLE_RATE));

            // Process noise to fill state registers
            let noise = white_noise(1024, 42);
            let _ = process_band_mono(&mut band, &noise);

            // Reset
            band.reset();

            // Process silence — output must be exactly zero
            let silence_input = silence(1024);
            let output = process_band_mono(&mut band, &silence_input);

            for (i, &s) in output.iter().enumerate() {
                assert!(
                    s == 0.0,
                    "{ft:?}/{structure:?}: after reset, silence output[{i}] = {s} (expected 0.0)"
                );
            }
        }
    }
}

#[test]
fn chain_reset_produces_silence() {
    let mut chain = EqChain::new();
    chain.update(test_config(SAMPLE_RATE));

    // Add diverse bands
    let idx = chain.add_band();
    if let Some(b) = chain.band_mut(idx) {
        b.filter_type = FilterType::Peak;
        b.gain_db = 6.0;
        b.freq_hz = 500.0;
    }
    chain.update_band(idx);

    let idx = chain.add_band();
    if let Some(b) = chain.band_mut(idx) {
        b.filter_type = FilterType::Lowpass;
        b.freq_hz = 5000.0;
    }
    chain.update_band(idx);

    let idx = chain.add_band();
    if let Some(b) = chain.band_mut(idx) {
        b.filter_type = FilterType::HighShelf;
        b.gain_db = -3.0;
        b.freq_hz = 8000.0;
    }
    chain.update_band(idx);

    // Process noise
    let noise = white_noise(1024, 42);
    let _ = process_chain_mono(&mut chain, &noise, 512);

    // Reset
    chain.reset();

    // Process silence
    let silence_input = silence(1024);
    let output = process_chain_mono(&mut chain, &silence_input, 512);

    for (i, &s) in output.iter().enumerate() {
        assert!(
            s == 0.0,
            "Chain: after reset, silence output[{i}] = {s} (expected 0.0)"
        );
    }
}
