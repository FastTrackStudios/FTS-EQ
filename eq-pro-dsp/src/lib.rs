//! EQ Pro DSP — faithful Pro-Q 4 extraction.
//!
//! Complete filter design pipeline reverse-engineered from FabFilter Pro-Q 4:
//!
//!   1. Analog prototype (Butterworth poles via `maybe_butterworth_pole_next`)
//!   2. Frequency transformation (LP→BP via elliptic functions, LP→BS, bilinear)
//!   3. Cascade coefficient computation (peak/shelf via `compute_cascade_coefficients`)
//!   4. ZPK → biquad conversion (via `zpk_to_biquad_coefficients`)
//!   5. Anti-cramping delay cascade (3-level group delay compensation)
//!
//! Filter types (13 total, matching Pro-Q 4 type codes):
//!   0 = Peak/Bell        — own ZPK via cascade coefficients
//!   1 = Highpass          — Butterworth direct
//!   2 = Lowpass           — Butterworth direct
//!   3 = Bandpass          — Butterworth LP + elliptic LP→BP
//!   4 = Notch             — Butterworth LP + LP→BS
//!   5 = Multimode A       — elliptic LP→BP variant
//!   6 = Multimode B       — elliptic LP→BP variant
//!   7 = Low Shelf         — Butterworth + bilinear + shelf gain
//!   8 = High Shelf        — Butterworth + bilinear + shelf gain
//!   9 = Tilt Shelf        — Butterworth + bilinear + shelf gain
//!   10 = Band Shelf       — LP→BP + bilinear
//!   11 = Allpass           — negate zeros (transform type 4)
//!   12 = Shelf (alt)       — own ZPK via cascade coefficients

pub mod band;
pub mod biquad;
pub mod cascade;
pub mod chain;
pub mod constants;
pub mod delay;
pub mod design;
pub mod elliptic;
pub mod prototype;
pub mod response;
pub mod section;
pub mod shelf;
pub mod transform;
pub mod zpk;

pub use band::Band;
pub use chain::EqChain;
pub use design::FilterType;
