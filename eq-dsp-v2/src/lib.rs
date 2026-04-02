//! EQ DSP v2 — Pro-Q 4 architecture.
//!
//! Filter design pipeline:
//!   1. Analog prototype (Butterworth poles in s-domain)
//!   2. Frequency transformation (LP→BP, LP→BS, bilinear, etc.)
//!   3. ZPK representation (zeros, poles, gain)
//!   4. ZPK → biquad coefficient conversion
//!
//! This matches FabFilter Pro-Q 4's verified DSP architecture:
//!   setup_eq_band_filter (0x1800fdf10)
//!     → design_filter_zpk_and_transform (0x1800ff6f0)
//!         → filter_type_dispatcher (0x1800fe2a0)  — analog prototype
//!         → frequency transform based on type
//!         → zpk_to_biquad_coefficients (0x1800fe040)

pub mod zpk;
pub mod prototype;
pub mod transform;
pub mod biquad;
pub mod design;

#[cfg(any(test, feature = "test-util"))]
pub mod test_util;
