//! Sample rate independence tests — filters behave correctly across sample rates.
//!
//! r[verify test.dsp.sample-rate-independence]

use eq_dsp::band::Band;
use eq_dsp::filter_type::{FilterStructure, FilterType};
use eq_dsp::test_util::*;

const IMPULSE_LEN: usize = 8192;

const SAMPLE_RATES: [f64; 5] = [44100.0, 48000.0, 88200.0, 96000.0, 192000.0];

#[test]
fn lowpass_dc_gain_across_rates() {
    for &sr in &SAMPLE_RATES {
        let mut band = Band::new();
        band.filter_type = FilterType::Lowpass;
        band.structure = FilterStructure::Tdf2;
        band.freq_hz = 1000.0;
        band.q = 0.707;
        band.order = 2;
        band.update(test_config(sr));

        let input = impulse(IMPULSE_LEN);
        let output = process_band_mono(&mut band, &input);
        let spectrum = fft_magnitude_db(&output, sr);

        let dc = magnitude_at_freq(&spectrum, 0.0);
        assert!(
            dc.abs() < 1.0,
            "Lowpass DC at {sr}Hz sample rate: {dc:.2}dB (expected ~0dB)"
        );
    }
}

#[test]
fn lowpass_stopband_across_rates() {
    for &sr in &SAMPLE_RATES {
        let mut band = Band::new();
        band.filter_type = FilterType::Lowpass;
        band.structure = FilterStructure::Tdf2;
        band.freq_hz = 1000.0;
        band.q = 0.707;
        band.order = 2;
        band.update(test_config(sr));

        let input = impulse(IMPULSE_LEN);
        let output = process_band_mono(&mut band, &input);
        let spectrum = fft_magnitude_db(&output, sr);

        // 10kHz is well into stopband at all sample rates
        let at_10k = magnitude_at_freq(&spectrum, 10000.0);
        assert!(
            at_10k < -15.0,
            "Lowpass at 10kHz ({sr}Hz rate): {at_10k:.2}dB (expected < -15dB)"
        );
    }
}

#[test]
fn peak_gain_at_center_across_rates() {
    for &sr in &SAMPLE_RATES {
        let mut band = Band::new();
        band.filter_type = FilterType::Peak;
        band.structure = FilterStructure::Tdf2;
        band.freq_hz = 1000.0;
        band.q = 1.0;
        band.gain_db = 6.0;
        band.order = 2;
        band.update(test_config(sr));

        let input = impulse(IMPULSE_LEN);
        let output = process_band_mono(&mut band, &input);
        let spectrum = fft_magnitude_db(&output, sr);

        let at_center = magnitude_at_freq(&spectrum, 1000.0);
        assert!(
            (at_center - 6.0).abs() < 2.0,
            "Peak +6dB at 1kHz ({sr}Hz rate): {at_center:.2}dB (expected ~+6dB)"
        );
    }
}

#[test]
fn shelf_gain_consistent_across_rates() {
    for &sr in &SAMPLE_RATES {
        let mut band = Band::new();
        band.filter_type = FilterType::LowShelf;
        band.structure = FilterStructure::Tdf2;
        band.freq_hz = 500.0;
        band.q = 0.707;
        band.gain_db = 6.0;
        band.order = 2;
        band.update(test_config(sr));

        let input = impulse(IMPULSE_LEN);
        let output = process_band_mono(&mut band, &input);
        let spectrum = fft_magnitude_db(&output, sr);

        // Low frequencies should have the shelf gain
        let dc = magnitude_at_freq(&spectrum, 30.0);
        assert!(
            (dc - 6.0).abs() < 3.0,
            "LowShelf DC at {sr}Hz rate: {dc:.2}dB (expected ~+6dB)"
        );

        // High frequencies should be near unity
        let high = magnitude_at_freq(&spectrum, 10000.0);
        assert!(
            high.abs() < 2.0,
            "LowShelf 10kHz at {sr}Hz rate: {high:.2}dB (expected ~0dB)"
        );
    }
}
