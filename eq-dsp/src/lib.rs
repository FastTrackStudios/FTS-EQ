//! EQ DSP engine — the filter implementation for the entire plugin suite.
//!
//! This crate is used directly by `eq-plugin` for the EQ product, and also
//! consumed by `comp-dsp` (sidechain EQ), `delay-dsp` (feedback filters),
//! `reverb-dsp` (input/output EQ), etc.
//!
//! # Filter Design
//!
//! Uses Martin Vicanek's "Matched Second Order Digital Filters" for coefficient
//! calculation instead of the standard bilinear transform. This gives better
//! frequency response matching near Nyquist with no cramping.
//!
//! # Filter Types
//!
//! 9 types matching ZLEqualizer: Peak, Low Shelf, High Shelf, Tilt Shelf,
//! Lowpass, Highpass, Bandpass, Notch, Band Shelf.
//!
//! # Filter Structures
//!
//! - TDF2 (Transposed Direct Form II) — minimum phase, lowest latency
//! - SVF (State Variable Filter) — simultaneous HP/BP/LP, better modulation
//!
//! # Variable Order
//!
//! Higher orders are built by cascading 2nd-order sections with
//! Butterworth-distributed Q values.

pub mod band;
pub mod chain;
pub mod coeff;
pub mod filter_type;
pub mod response;
pub mod section;

pub use band::Band;
pub use chain::EqChain;
pub use filter_type::FilterType;

#[cfg(any(test, feature = "test-util"))]
pub mod test_util;
