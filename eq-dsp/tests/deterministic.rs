//! Deterministic tests — same input produces bit-identical output.
//!
//! r[verify test.dsp.deterministic]

use eq_dsp::band::Band;
use eq_dsp::filter_type::{FilterStructure, FilterType};
use eq_dsp::test_util::*;

const SAMPLE_RATE: f64 = 48000.0;

#[test]
fn two_identical_bands_produce_identical_output() {
    let types = [
        (FilterType::Peak, 6.0),
        (FilterType::LowShelf, 6.0),
        (FilterType::Lowpass, 0.0),
        (FilterType::Highpass, 0.0),
        (FilterType::Notch, 0.0),
    ];

    let input = white_noise(4096, 42);

    for (ft, gain) in types {
        for structure in [FilterStructure::Tdf2, FilterStructure::Svf] {
            let make_band = || {
                let mut band = Band::new();
                band.filter_type = ft;
                band.structure = structure;
                band.freq_hz = 1000.0;
                band.q = 0.707;
                band.gain_db = gain;
                band.order = 2;
                band.update(test_config(SAMPLE_RATE));
                band
            };

            let mut band_a = make_band();
            let mut band_b = make_band();

            let out_a = process_band_mono(&mut band_a, &input);
            let out_b = process_band_mono(&mut band_b, &input);

            for (i, (a, b)) in out_a.iter().zip(out_b.iter()).enumerate() {
                assert!(
                    (*a).to_bits() == (*b).to_bits(),
                    "{ft:?}/{structure:?}: sample[{i}] differs: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn reset_then_reprocess_is_identical() {
    let input = white_noise(2048, 99);

    let mut band = Band::new();
    band.filter_type = FilterType::Peak;
    band.structure = FilterStructure::Tdf2;
    band.freq_hz = 2000.0;
    band.q = 1.5;
    band.gain_db = 9.0;
    band.order = 2;
    band.update(test_config(SAMPLE_RATE));

    // First pass
    let out_1 = process_band_mono(&mut band, &input);

    // Reset and second pass
    band.reset();
    let out_2 = process_band_mono(&mut band, &input);

    for (i, (a, b)) in out_1.iter().zip(out_2.iter()).enumerate() {
        assert!(
            (*a).to_bits() == (*b).to_bits(),
            "Reset determinism: sample[{i}] differs: {a} vs {b}"
        );
    }
}
