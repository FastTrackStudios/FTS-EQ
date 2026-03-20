//! Impulse response tests — verify frequency response via FFT analysis.
//!
//! r[verify test.dsp.impulse-response]

use eq_dsp::band::Band;
use eq_dsp::filter_type::{FilterStructure, FilterType};
use eq_dsp::response;
use eq_dsp::test_util::*;

const IMPULSE_LEN: usize = 8192;
const SAMPLE_RATE: f64 = 48000.0;

/// Helper: create a band, process an impulse, return the FFT spectrum.
fn impulse_spectrum(
    filter_type: FilterType,
    structure: FilterStructure,
    freq_hz: f64,
    q: f64,
    gain_db: f64,
    order: usize,
) -> (Band, Vec<(f64, f64)>) {
    let mut band = Band::new();
    band.filter_type = filter_type;
    band.structure = structure;
    band.freq_hz = freq_hz;
    band.q = q;
    band.gain_db = gain_db;
    band.order = order;
    band.update(test_config(SAMPLE_RATE));

    let input = impulse(IMPULSE_LEN);
    let output = process_band_mono(&mut band, &input);
    let spectrum = fft_magnitude_db(&output, SAMPLE_RATE);
    (band, spectrum)
}

/// Compare impulse response FFT against analytical response at key frequencies.
fn check_against_analytical(
    band: &Band,
    spectrum: &[(f64, f64)],
    test_freqs: &[f64],
    tolerance_db: f64,
) {
    for &freq in test_freqs {
        let measured = magnitude_at_freq(spectrum, freq);
        let analytical = response::band_magnitude_db(band, freq, SAMPLE_RATE);

        let error = (measured - analytical).abs();
        assert!(
            error < tolerance_db,
            "At {freq}Hz: measured={measured:.2}dB, analytical={analytical:.2}dB, error={error:.2}dB (tol={tolerance_db}dB)"
        );
    }
}

// ── Lowpass ─────────────────────────────────────────────────────────────

#[test]
fn lowpass_tdf2_order2() {
    let (band, spectrum) = impulse_spectrum(
        FilterType::Lowpass,
        FilterStructure::Tdf2,
        1000.0,
        0.707,
        0.0,
        2,
    );
    // DC should be near 0dB, well above cutoff should be attenuated
    let dc = magnitude_at_freq(&spectrum, 0.0);
    assert!(dc.abs() < 1.0, "DC gain should be ~0dB, got {dc:.2}dB");

    let high = magnitude_at_freq(&spectrum, 10000.0);
    assert!(
        high < -15.0,
        "10kHz should be well attenuated, got {high:.2}dB"
    );

    check_against_analytical(&band, &spectrum, &[100.0, 500.0, 2000.0, 5000.0], 1.0);
}

#[test]
fn lowpass_svf_order2() {
    let (band, spectrum) = impulse_spectrum(
        FilterType::Lowpass,
        FilterStructure::Svf,
        1000.0,
        0.707,
        0.0,
        2,
    );
    let dc = magnitude_at_freq(&spectrum, 0.0);
    assert!(dc.abs() < 1.0, "DC gain should be ~0dB, got {dc:.2}dB");

    check_against_analytical(&band, &spectrum, &[100.0, 500.0, 2000.0, 5000.0], 1.0);
}

#[test]
fn lowpass_higher_order_steeper() {
    // Order 4 should attenuate more than order 2 at the same stopband frequency
    let (_, spec_2) = impulse_spectrum(
        FilterType::Lowpass,
        FilterStructure::Tdf2,
        1000.0,
        0.707,
        0.0,
        2,
    );
    let (_, spec_4) = impulse_spectrum(
        FilterType::Lowpass,
        FilterStructure::Tdf2,
        1000.0,
        0.707,
        0.0,
        4,
    );

    let atten_2 = magnitude_at_freq(&spec_2, 10000.0);
    let atten_4 = magnitude_at_freq(&spec_4, 10000.0);
    assert!(
        atten_4 < atten_2 - 10.0,
        "Order 4 should attenuate >10dB more than order 2 at 10kHz: order2={atten_2:.1}dB, order4={atten_4:.1}dB"
    );
}

// ── Highpass ────────────────────────────────────────────────────────────

#[test]
fn highpass_tdf2_order2() {
    let (band, spectrum) = impulse_spectrum(
        FilterType::Highpass,
        FilterStructure::Tdf2,
        1000.0,
        0.707,
        0.0,
        2,
    );
    let dc = magnitude_at_freq(&spectrum, 0.0);
    assert!(dc < -30.0, "DC should be heavily attenuated, got {dc:.2}dB");

    let high = magnitude_at_freq(&spectrum, 10000.0);
    assert!(
        (high).abs() < 2.0,
        "10kHz should be near 0dB, got {high:.2}dB"
    );

    check_against_analytical(&band, &spectrum, &[100.0, 500.0, 2000.0, 5000.0], 1.0);
}

// ── Peak ────────────────────────────────────────────────────────────────

#[test]
fn peak_boost_at_center() {
    let (band, spectrum) = impulse_spectrum(
        FilterType::Peak,
        FilterStructure::Tdf2,
        1000.0,
        1.0,
        12.0,
        2,
    );
    let at_center = magnitude_at_freq(&spectrum, 1000.0);
    assert!(
        (at_center - 12.0).abs() < 1.5,
        "Peak at 1kHz should be ~+12dB, got {at_center:.2}dB"
    );

    // DC and Nyquist should be near 0dB
    let dc = magnitude_at_freq(&spectrum, 50.0);
    assert!(dc.abs() < 1.0, "Below peak should be ~0dB, got {dc:.2}dB");

    check_against_analytical(&band, &spectrum, &[100.0, 500.0, 1000.0, 5000.0], 1.5);
}

#[test]
fn peak_cut_at_center() {
    let (band, spectrum) = impulse_spectrum(
        FilterType::Peak,
        FilterStructure::Tdf2,
        2000.0,
        1.0,
        -9.0,
        2,
    );
    let at_center = magnitude_at_freq(&spectrum, 2000.0);
    assert!(
        (at_center - (-9.0)).abs() < 1.5,
        "Cut at 2kHz should be ~-9dB, got {at_center:.2}dB"
    );

    check_against_analytical(&band, &spectrum, &[200.0, 1000.0, 2000.0, 8000.0], 1.5);
}

#[test]
fn peak_svf_matches_tdf2() {
    let (_, spec_tdf2) =
        impulse_spectrum(FilterType::Peak, FilterStructure::Tdf2, 1000.0, 1.0, 6.0, 2);
    let (_, spec_svf) =
        impulse_spectrum(FilterType::Peak, FilterStructure::Svf, 1000.0, 1.0, 6.0, 2);

    // Both structures should produce similar response
    for freq in [100.0, 500.0, 1000.0, 2000.0, 5000.0] {
        let tdf2_db = magnitude_at_freq(&spec_tdf2, freq);
        let svf_db = magnitude_at_freq(&spec_svf, freq);
        let diff = (tdf2_db - svf_db).abs();
        assert!(
            diff < 1.0,
            "TDF2 vs SVF mismatch at {freq}Hz: tdf2={tdf2_db:.2}dB, svf={svf_db:.2}dB"
        );
    }
}

// ── Shelves ─────────────────────────────────────────────────────────────

#[test]
fn low_shelf_boost() {
    let (band, spectrum) = impulse_spectrum(
        FilterType::LowShelf,
        FilterStructure::Tdf2,
        500.0,
        0.707,
        6.0,
        2,
    );
    let dc = magnitude_at_freq(&spectrum, 30.0);
    assert!(
        (dc - 6.0).abs() < 2.0,
        "Low shelf DC should be ~+6dB, got {dc:.2}dB"
    );

    let high = magnitude_at_freq(&spectrum, 10000.0);
    assert!(
        high.abs() < 2.0,
        "Above shelf should be ~0dB, got {high:.2}dB"
    );

    check_against_analytical(&band, &spectrum, &[50.0, 200.0, 1000.0, 5000.0], 1.5);
}

#[test]
fn high_shelf_cut() {
    let (band, spectrum) = impulse_spectrum(
        FilterType::HighShelf,
        FilterStructure::Tdf2,
        2000.0,
        0.707,
        -6.0,
        2,
    );
    let high = magnitude_at_freq(&spectrum, 15000.0);
    assert!(
        (high - (-6.0)).abs() < 2.0,
        "High shelf top should be ~-6dB, got {high:.2}dB"
    );

    let dc = magnitude_at_freq(&spectrum, 50.0);
    assert!(dc.abs() < 2.0, "Below shelf should be ~0dB, got {dc:.2}dB");

    check_against_analytical(&band, &spectrum, &[100.0, 1000.0, 5000.0, 15000.0], 1.5);
}

// ── Bandpass & Notch ────────────────────────────────────────────────────

#[test]
fn bandpass_peaks_at_center() {
    let (_, spectrum) = impulse_spectrum(
        FilterType::Bandpass,
        FilterStructure::Tdf2,
        2000.0,
        2.0,
        0.0,
        2,
    );
    let dc = magnitude_at_freq(&spectrum, 30.0);
    let center = magnitude_at_freq(&spectrum, 2000.0);
    // Bandpass should be louder at center than at DC
    assert!(
        center > dc + 10.0,
        "Bandpass center should be much louder than DC: center={center:.2}dB, dc={dc:.2}dB"
    );
}

#[test]
fn notch_dips_at_center() {
    let (_, spectrum) = impulse_spectrum(
        FilterType::Notch,
        FilterStructure::Tdf2,
        2000.0,
        2.0,
        0.0,
        2,
    );
    let center = magnitude_at_freq(&spectrum, 2000.0);
    let offcenter = magnitude_at_freq(&spectrum, 500.0);
    assert!(
        center < offcenter - 15.0,
        "Notch center should be deeply cut: center={center:.2}dB, offcenter={offcenter:.2}dB"
    );
}

// ── All filter types produce non-trivial response ──────────────────────

#[test]
fn all_types_produce_output() {
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
            let (_, spectrum) = impulse_spectrum(ft, structure, 1000.0, 0.707, gain, 2);
            let has_energy = spectrum.iter().any(|&(_, db)| db > -200.0);
            assert!(
                has_energy,
                "{ft:?}/{structure:?} produced no output from impulse"
            );
        }
    }
}
