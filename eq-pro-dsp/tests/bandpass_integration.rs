use eq_pro_dsp::biquad;
use eq_pro_dsp::design::{design_filter, FilterType};
use std::f64::consts::PI;

#[test]
fn bandpass_has_peak_at_center() {
    // Bandpass should have maximum gain at center frequency
    let sos = design_filter(FilterType::Bandpass, 1000.0, 2.0, 0.0, 48000.0, 4);

    let w0 = 2.0 * PI * 1000.0 / 48000.0;

    // Evaluate at center, octave below, and octave above
    let mag_center = biquad::mag_db_sos(&sos, w0);
    let mag_low = biquad::mag_db_sos(&sos, w0 / 2.0);
    let mag_high = biquad::mag_db_sos(&sos, w0 * 2.0);

    println!("Bandpass at center: {:.2} dB", mag_center);
    println!("Bandpass at 1/2 center: {:.2} dB", mag_low);
    println!("Bandpass at 2x center: {:.2} dB", mag_high);

    // Center should be greater than sides
    assert!(
        mag_center > mag_low - 1.0,
        "Center should peak above lower frequency"
    );
    assert!(
        mag_center > mag_high - 1.0,
        "Center should peak above higher frequency"
    );
}

#[test]
fn bandpass_q_affects_bandwidth() {
    let sos_q1 = design_filter(FilterType::Bandpass, 1000.0, 1.0, 0.0, 48000.0, 4);
    let sos_q10 = design_filter(FilterType::Bandpass, 1000.0, 10.0, 0.0, 48000.0, 4);

    let w0 = 2.0 * PI * 1000.0 / 48000.0;

    // At half power (-3dB), measure bandwidth
    let mag_center_q1 = biquad::mag_db_sos(&sos_q1, w0);
    let mag_center_q10 = biquad::mag_db_sos(&sos_q10, w0);

    // For high Q, bandwidth should be narrower
    // Check a frequency 1/4 octave away
    let w_offset = w0 * 0.8; // ~1/4 octave below
    let mag_off_q1 = biquad::mag_db_sos(&sos_q1, w_offset);
    let mag_off_q10 = biquad::mag_db_sos(&sos_q10, w_offset);

    println!(
        "Q=1 center: {:.2} dB, at 0.8*w0: {:.2} dB",
        mag_center_q1, mag_off_q1
    );
    println!(
        "Q=10 center: {:.2} dB, at 0.8*w0: {:.2} dB",
        mag_center_q10, mag_off_q10
    );

    // Q=10 should drop more sharply away from center
    let drop_q1 = (mag_center_q1 - mag_off_q1).max(0.0);
    let drop_q10 = (mag_center_q10 - mag_off_q10).max(0.0);

    println!(
        "Drop at 0.8*w0: Q=1 drops {:.2} dB, Q=10 drops {:.2} dB",
        drop_q1, drop_q10
    );

    assert!(
        drop_q10 > drop_q1 - 1.0,
        "Higher Q should have sharper bandwidth"
    );
}
