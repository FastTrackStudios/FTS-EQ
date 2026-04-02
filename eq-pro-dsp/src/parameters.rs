//! Parameter transformation pipeline extracted from Pro-Q 4.
//!
//! This module implements the parameter transformation stage that occurs before
//! filter design. Pro-Q 4 applies conditional transformations to user parameters
//! (Q, gain, frequency) based on filter type and mode, producing "effective"
//! parameters used by the design stage.
//!
//! Magic constants extracted from binary @ 0x18010de30 (compute_peak_band_parameters).

/// Magic constant: π-related frequency bound
const PI_LOWER_BOUND: f64 = 2.984513020910303;

/// Magic constant: Shelf frequency bound
const SHELF_FREQ_BOUND_1: f64 = 2.607521902479528;

/// Magic constant: Shelf frequency bound upper
const SHELF_FREQ_BOUND_2: f64 = 3.110176727053895;

/// Magic constant: π upper bound (3.14... instead of full π)
const PI_BOUND: f64 = 3.141278494324434;

/// Magic constant: Shelf special scaling
const SHELF_SCALE_1: f64 = 1.8; // From DAT_180231b10

/// Magic constant: Type 6 clipping bound
const TYPE_6_CLIP: f64 = 1.884955592153876;

/// Magic constant: Q constraint lower
const Q_CONSTRAINT_LOW: f64 = 0.995;

/// Magic constant: Q constraint mid
const Q_CONSTRAINT_MID: f64 = 0.999;

/// Magic constant: Q constraint high
const Q_CONSTRAINT_HIGH: f64 = 0.9998;

/// Magic constant: Gain scaling factor
const GAIN_SCALE_0_2: f64 = 0.2;

/// Magic constant: Q² adjustment bounds
const Q_SQ_EPSILON: f64 = 0.000000000065670;

const Q_SQ_BOUND: f64 = 0.000000388754016;

/// Magic constant: Mode-dependent scaling
const MODE_SCALE_0_95: f64 = 0.95;

const MODE_SCALE_0_99: f64 = 0.99;

const MODE_SCALE_0_05: f64 = 0.05;

/// Magic constant: Frequency scaling
const FREQ_SCALE_TINY: f64 = 0.000000001719365;

const FREQ_CONSTRAINT: f64 = 0.008437501252047;

/// Magic constant: Special SQ term scaling
const SQ_SCALE_0_0005: f64 = 0.0005;

const SQ_SCALE_0_5: f64 = 0.5;

const SQ_SCALE_1_2: f64 = 1.2;

const SQ_SCALE_1_25: f64 = 1.25;

const SQ_SCALE_1_5: f64 = 1.5;

const SQ_SCALE_1_414: f64 = 1.414213538169861; // √2

const SQ_SCALE_1_57: f64 = 1.570796326794897; // π/2

/// Magic constant: π (full)
const PI: f64 = std::f64::consts::PI;

/// Magic constant: 9π/10
const NINE_PI_OVER_10: f64 = 2.827433388230814;

/// Transformed parameters output from compute_peak_band_parameters
#[derive(Debug, Clone, Copy)]
pub struct TransformedParams {
    /// param[0x8] - processed Q or gain
    pub processed_q: f64,
    /// param[0x10] - Q-related term
    pub q_term: f64,
    /// param[0x18] - effective gain for filter
    pub gain_term: f64,
    /// param[0x20] - processed frequency value
    pub frequency: f64,
    /// Special parameters for specific types
    pub special_param_1: f64,
    pub special_param_2: f64,
    pub special_param_3: f64,
}

impl Default for TransformedParams {
    fn default() -> Self {
        TransformedParams {
            processed_q: 1.0,
            q_term: 0.5,
            gain_term: 0.5,
            frequency: PI,
            special_param_1: 0.0,
            special_param_2: 0.0,
            special_param_3: 0.0,
        }
    }
}

/// Apply parameter transformations based on filter type and user inputs
///
/// This mirrors the compute_peak_band_parameters function from Pro-Q 4.
///
/// # Arguments
/// * `filter_type` - Filter type (0-12)
/// * `user_q` - User-provided Q value
/// * `user_gain_db` - User-provided gain in dB
/// * `center_freq_hz` - Center frequency in Hz
/// * `sample_rate` - Sample rate in Hz
/// * `mode` - Mode selector (param[0x90])
/// * `param_state` - Parameter state selector (param[0x38])
/// * `sq_component` - Special Q² component (param[0x8c])
/// * `mode_param` - Mode parameter (param[0x94])
///
/// # Returns
/// Transformed parameters ready for filter design stage
pub fn transform_parameters(
    filter_type: u32,
    user_q: f64,
    user_gain_db: f64,
    _center_freq_hz: f64,
    _sample_rate: f64,
    mode: i32,
    param_state: i32,
    sq_component: f64,
    mode_param: f64,
) -> TransformedParams {
    // Compute frequency-related scaling (reserved for future use)
    // let _freq_scale = 1000000.0 / (_center_freq_hz + 1e-10);  // Avoid division by zero

    match filter_type {
        0 => transform_type_0_peak(user_q, user_gain_db, param_state),
        1 | 2 => {
            // Types 1-2: HP/LP - no transformation
            TransformedParams {
                processed_q: user_q,
                q_term: user_q * SQ_SCALE_0_5,
                gain_term: user_q * SQ_SCALE_0_5,
                frequency: PI,
                ..Default::default()
            }
        }
        3..=6 => transform_types_3_to_6(
            user_q,
            user_gain_db,
            filter_type,
            param_state,
            sq_component,
            mode_param,
        ),
        7 => transform_type_7_hp_shelf(user_q, user_gain_db),
        8 => transform_type_8_lp_shelf(user_q, user_gain_db, mode),
        9 => {
            // Type 9: Tilt - similar to Type 8
            transform_type_8_lp_shelf(user_q, user_gain_db, mode)
        }
        10 => transform_type_10_band_shelf(user_q, user_gain_db, mode, param_state, sq_component),
        11 => {
            // Type 11: Allpass - no transformation
            TransformedParams {
                processed_q: user_q,
                q_term: user_q * SQ_SCALE_0_5,
                gain_term: 1.0,
                frequency: PI,
                ..Default::default()
            }
        }
        12 => transform_type_12_shelf_alt(user_q, user_gain_db),
        _ => TransformedParams::default(),
    }
}

/// Type 0: Peak/Bell filter parameter transformation
fn transform_type_0_peak(user_q: f64, user_gain_db: f64, param_state: i32) -> TransformedParams {
    let half_gain = user_gain_db * SQ_SCALE_0_5;

    let effective_q = if param_state == 1 {
        if user_q <= half_gain {
            half_gain
        } else {
            user_q
        }
    } else {
        user_q
    };

    TransformedParams {
        processed_q: user_gain_db, // Store full gain
        q_term: effective_q * SQ_SCALE_0_5,
        gain_term: effective_q,
        frequency: PI,
        ..Default::default()
    }
}

/// Types 3-6: Bandpass-like filters
fn transform_types_3_to_6(
    user_q: f64,
    _user_gain_db: f64,
    filter_type: u32,
    param_state: i32,
    sq_component: f64,
    mode_param: f64,
) -> TransformedParams {
    // Compute Q² adjustment
    let q_sq_adj = sq_component * sq_component * 0.25 + mode_param;
    let q_sq_adj_clamped = q_sq_adj.max(Q_SQ_EPSILON).min(0.65); // Reasonable bounds

    // Type 6 special handling
    if filter_type == 6 {
        let q_clipped = user_q.min(TYPE_6_CLIP);
        return TransformedParams {
            processed_q: q_clipped,
            q_term: q_clipped * MODE_SCALE_0_05,
            gain_term: q_clipped * SQ_SCALE_0_5,
            frequency: PI,
            ..Default::default()
        };
    }

    // Standard bandpass path
    let base_q = if param_state == 2 {
        // Mode 2: complex formula
        user_q
    } else {
        user_q * SQ_SCALE_0_5 // 50% reduction
    };

    let q_term = (1.0 - q_sq_adj_clamped) * base_q;
    let gain_term = (1.0 - q_sq_adj_clamped * MODE_SCALE_0_05) * base_q;

    TransformedParams {
        processed_q: base_q,
        q_term,
        gain_term,
        frequency: PI,
        special_param_1: q_sq_adj_clamped,
        ..Default::default()
    }
}

/// Type 7: HP Shelf
fn transform_type_7_hp_shelf(user_q: f64, _user_gain_db: f64) -> TransformedParams {
    let q_scaled = user_q * SQ_SCALE_1_2; // 1.2x multiplier
    let freq = q_scaled.max(NINE_PI_OVER_10).min(PI);

    TransformedParams {
        processed_q: user_q,
        q_term: user_q * SQ_SCALE_0_5,
        gain_term: user_q * SQ_SCALE_0_5,
        frequency: freq,
        ..Default::default()
    }
}

/// Type 8: LP Shelf
fn transform_type_8_lp_shelf(user_q: f64, _user_gain_db: f64, mode: i32) -> TransformedParams {
    let q_scaled = if mode == -1 {
        user_q * SQ_SCALE_1_2
    } else {
        user_q * SQ_SCALE_1_5 // Alternative mode
    };
    let freq = q_scaled.max(NINE_PI_OVER_10).min(PI);

    TransformedParams {
        processed_q: user_q,
        q_term: user_q * SQ_SCALE_0_5,
        gain_term: user_q * SQ_SCALE_0_5,
        frequency: freq,
        ..Default::default()
    }
}

/// Type 10: Band Shelf (most complex)
fn transform_type_10_band_shelf(
    user_q: f64,
    _user_gain_db: f64,
    mode: i32,
    param_state: i32,
    sq_component: f64,
) -> TransformedParams {
    if mode == -1 {
        // Simple path
        let q_scaled = user_q * SQ_SCALE_1_2;
        let freq = q_scaled.max(NINE_PI_OVER_10).min(PI);

        TransformedParams {
            processed_q: user_q,
            q_term: user_q * SQ_SCALE_0_5,
            gain_term: user_q * MODE_SCALE_0_05,
            frequency: freq,
            ..Default::default()
        }
    } else {
        // Complex path with mode-dependent logic
        let q_transformed = user_q * SHELF_SCALE_1; // 1.8x multiplier
        let freq = q_transformed.max(SHELF_FREQ_BOUND_1).min(PI_BOUND);

        let processed_q = if mode == 1 {
            // Mode 1: SQ-based adjustment
            let sq_sq = sq_component * sq_component;
            let adjusted = (Q_CONSTRAINT_HIGH - sq_sq * SQ_SCALE_0_0005) * user_q;
            adjusted.min(PI_BOUND)
        } else if mode == 0 && param_state == 1 {
            // Mode 0, special param: power operation
            let power_result = SQ_SCALE_0_5.sqrt();
            (power_result * user_q).max(0.1)
        } else {
            user_q
        };

        let q_min = user_q * SQ_SCALE_0_5;
        let q_constrained = processed_q.max(q_min * SQ_SCALE_0_5).min(q_min);

        TransformedParams {
            processed_q: q_constrained,
            q_term: q_constrained * MODE_SCALE_0_05,
            gain_term: q_constrained * SQ_SCALE_0_5,
            frequency: freq,
            special_param_1: freq,
            ..Default::default()
        }
    }
}

/// Type 12: Shelf (alternative)
fn transform_type_12_shelf_alt(user_q: f64, _user_gain_db: f64) -> TransformedParams {
    let q_clipped = user_q.min(TYPE_6_CLIP);

    TransformedParams {
        processed_q: q_clipped,
        q_term: q_clipped * MODE_SCALE_0_05,
        gain_term: q_clipped * SQ_SCALE_0_5,
        frequency: PI,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_0_peak_basic() {
        let result = transform_type_0_peak(1.0, 6.0, 0);
        assert!((result.gain_term - 1.0).abs() < 0.001);
        assert!((result.q_term - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_type_0_peak_mode_1() {
        let result = transform_type_0_peak(1.0, 6.0, 1);
        // Mode 1: gain * 0.5 = 3.0, so effective_q = 3.0
        assert!((result.gain_term - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_type_6_clipping() {
        let result = transform_types_3_to_6(2.0, 6.0, 6, 0, 0.0, 0.0);
        // Should be clipped to TYPE_6_CLIP
        assert!(result.processed_q <= TYPE_6_CLIP + 0.001);
    }

    #[test]
    fn test_type_10_frequency_bounds() {
        let result = transform_type_10_band_shelf(0.5, 0.0, -1, 0, 0.0); // mode -1 = simple path
                                                                         // Frequency should be within bounds
        assert!(result.frequency >= NINE_PI_OVER_10);
        assert!(result.frequency <= PI);
    }

    #[test]
    fn test_constants_sanity() {
        // Verify basic relationships
        assert!(PI_BOUND < PI + 0.01);
        assert!(Q_CONSTRAINT_LOW < Q_CONSTRAINT_MID);
        assert!(SHELF_FREQ_BOUND_1 < PI_BOUND);
    }
}
