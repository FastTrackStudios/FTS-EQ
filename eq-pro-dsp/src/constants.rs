//! Magic constants extracted from Pro-Q 4 binary.
//!
//! All values decoded from IEEE double-precision floats at 0x180231a00+.

use std::f64::consts::PI;

// ─── Mathematical Constants ─────────────────────────────────────────────────

/// π (same as std but extracted from binary for reference)
pub const PRO_Q_PI: f64 = PI;
/// 2π
pub const TWO_PI: f64 = 2.0 * PI;
/// π/2
pub const HALF_PI: f64 = PI / 2.0;
/// π/4
pub const QUARTER_PI: f64 = PI / 4.0;
/// 1/π ≈ 0.636619772367581
pub const INV_PI: f64 = 1.0 / PI;
/// 3π/10 ≈ 0.942477796076938
pub const THREE_PI_TENTHS: f64 = 3.0 * PI / 10.0;
/// √2 ≈ 1.414213562373095
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;
/// 4√2 ≈ 5.656854249492381
pub const FOUR_SQRT_2: f64 = 4.0 * std::f64::consts::SQRT_2;
/// ln(2) ≈ 0.693147180559945
pub const LN_2: f64 = std::f64::consts::LN_2;
/// log2(e) ≈ 1.442695040888963
pub const LOG2_E: f64 = std::f64::consts::LOG2_E;

// ─── Scaling Factors (0x180231a00 - 0x180231ab0) ───────────────────────────

pub const SCALE_0_5: f64 = 0.5;
pub const SCALE_0_54: f64 = 0.54;
pub const SCALE_0_628: f64 = 0.628318530717959; // π/5
pub const SCALE_0_637: f64 = 0.636619772367581; // 1/π
pub const SCALE_0_65: f64 = 0.65;
pub const SCALE_0_667: f64 = 0.666666666666667; // 2/3
pub const SCALE_0_693: f64 = 0.693147180559945; // ln(2)
pub const SCALE_0_7: f64 = 0.7;
pub const SCALE_0_766: f64 = 0.76609;
pub const SCALE_0_784: f64 = 0.783594;
pub const SCALE_0_785: f64 = 0.785; // ≈ π/4
pub const SCALE_0_8: f64 = 0.8;
pub const SCALE_0_9: f64 = 0.9;
pub const SCALE_0_942: f64 = 0.942477796076938; // 3π/10
pub const SCALE_0_95: f64 = 0.95;
pub const SCALE_0_96: f64 = 0.96;
pub const SCALE_0_99: f64 = 0.99;
pub const SCALE_0_995: f64 = 0.995;
pub const SCALE_0_999: f64 = 0.999;
pub const SCALE_0_9995: f64 = 0.9995;
pub const SCALE_0_9998: f64 = 0.9998;

// ─── Multipliers (0x180231ab8 - 0x180231bf8) ───────────────────────────────

pub const MULT_1_0: f64 = 1.0;
pub const MULT_1_01: f64 = 1.01;
pub const MULT_1_2: f64 = 1.2;
pub const MULT_1_25: f64 = 1.25;
pub const MULT_1_29: f64 = 1.290372985;
pub const MULT_1_414: f64 = 1.414213538169861; // √2 (single precision)
pub const MULT_1_443: f64 = 1.442695040888963; // log2(e)
pub const MULT_1_5: f64 = 1.5;
pub const MULT_1_548: f64 = 1.547508449804351;
pub const MULT_HALF_PI: f64 = 1.570796326794897; // π/2
pub const MULT_1_8: f64 = 1.8;
pub const MULT_1_885: f64 = 1.884955592153876; // 6/π
pub const MULT_2_0: f64 = 2.0;
pub const MULT_2_199: f64 = 2.199114857512855;
pub const MULT_2_2: f64 = 2.2;
pub const MULT_2_419: f64 = 2.419026343264141;
pub const MULT_2_513: f64 = 2.513274122871834; // 4π/5
pub const MULT_2_608: f64 = 2.607521902479528;
pub const MULT_2_670: f64 = 2.670353755551324;
pub const MULT_2_827: f64 = 2.827433388230814; // 9π/10
pub const MULT_2_922: f64 = 2.921681167838508;
pub const MULT_2_985: f64 = 2.984513020910303;
pub const MULT_3_0: f64 = 3.0;
pub const MULT_3_079: f64 = 3.078760800517997;
pub const MULT_3_110: f64 = 3.110176727053895;
pub const MULT_3_135: f64 = 3.135309468282613;
pub const MULT_3_141_LO: f64 = 3.141278494324434; // π low precision
pub const MULT_3_141_MED: f64 = 3.141592534380504; // π medium precision
pub const MULT_PI: f64 = 3.141592653589793; // π full precision
pub const MULT_3_3: f64 = 3.3;
pub const MULT_4_0: f64 = 4.0;
pub const MULT_5_657: f64 = 5.656854249492381; // 4√2
pub const MULT_6_0: f64 = 6.0;
pub const MULT_TWO_PI: f64 = 6.283185307179586; // 2π
pub const MULT_5_HALF_PI: f64 = 7.853981633974483; // 5π/2

// ─── Adaptive Frequency Streaming State 2 Multipliers ──────────────────────
// From 0x1802119d0 - 0x1802119ec

/// 8-point adaptive multiplier array for state 2 frequency streaming.
/// Progressive scaling from 1/π to 1.2 for smooth resolution transitions.
pub const STATE2_MULTIPLIERS: [f64; 8] = [
    0.636619772367581, // 1/π — transition in from previous state
    0.785398163397448, // π/4 — quarter pi
    0.942477796076938, // 3π/10 — three-tenths pi
    0.990000000000000, // approaching 1.0
    0.999000000000000, // very close to 1.0
    1.000000000000000, // identity
    1.010000000000000, // slightly above 1.0
    1.200000000000000, // elevated multiplier for emphasis
];

// ─── Filter Design Constants ───────────────────────────────────────────────

/// Maximum gain in dB (from binary: ±500dB limit)
pub const MAX_GAIN_DB: f64 = 500.0;
/// Minimum meaningful float value for comparisons
pub const FLOAT_EPSILON: f32 = 1.0e-30;
/// Minimum Q value
pub const MIN_Q: f64 = 0.025;
/// Maximum Q value
pub const MAX_Q: f64 = 40.0;
/// Minimum frequency in Hz
pub const MIN_FREQ_HZ: f64 = 10.0;
/// Maximum frequency in Hz
pub const MAX_FREQ_HZ: f64 = 30000.0;

// ─── Anti-Cramping Constants ───────────────────────────────────────────────

/// Maximum delay samples per cascade level (≈10ms at 44.1kHz)
pub const MAX_DELAY_SAMPLES_PER_LEVEL: usize = 441;
/// Number of delay cascade levels
pub const NUM_DELAY_LEVELS: usize = 3;
/// Maximum total delay compensation (≈30ms)
pub const MAX_TOTAL_DELAY_SAMPLES: usize = MAX_DELAY_SAMPLES_PER_LEVEL * NUM_DELAY_LEVELS;

// ─── Analyzer Constants ────────────────────────────────────────────────────

/// Analyzer accumulation points per frame
pub const ANALYZER_POINTS: usize = 512;
/// Analyzer unroll: 8 outer × 64 inner
pub const ANALYZER_OUTER_ITERS: usize = 8;
pub const ANALYZER_INNER_ITERS: usize = 64;

// ─── FFT Constants ─────────────────────────────────────────────────────────

/// Default FFT timeout in seconds (≈π)
pub const FFT_TIMEOUT_SECONDS: f64 = 3.14;
/// Mode 1 fixed latency in samples
pub const MODE1_FIXED_LATENCY: usize = 320; // 0x140
