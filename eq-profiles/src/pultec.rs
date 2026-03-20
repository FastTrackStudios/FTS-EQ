//! Pultec EQP-1A profile.
//!
//! Maps the classic Pultec controls to the EQ DSP chain:
//! - Low Boost / Low Atten at stepped frequencies
//! - High Boost / High Atten at stepped frequencies
//! - Bandwidth (maps to Q)

use crate::core::{Constraint, ParamMapping, Profile, ProfileControl};

pub struct PultecProfile;

impl Profile for PultecProfile {
    fn id(&self) -> &'static str {
        "eq_pultec_eqp1a"
    }

    fn name(&self) -> &'static str {
        "Pultec EQP-1A"
    }

    fn controls(&self) -> &[ProfileControl] {
        &PULTEC_CONTROLS
    }

    fn constraints(&self) -> &[Constraint] {
        &PULTEC_CONSTRAINTS
    }
}

static PULTEC_CONTROLS: [ProfileControl; 6] = [
    ProfileControl {
        id: "low_freq",
        label: "Low Frequency",
        mapping: ParamMapping::Stepped {
            param: "band_0_freq",
            values: &[20.0, 30.0, 60.0, 100.0],
            labels: &["20", "30", "60", "100"],
        },
    },
    ProfileControl {
        id: "low_boost",
        label: "Low Boost",
        mapping: ParamMapping::Direct {
            param: "band_0_gain",
            range: 0.0..=16.0,
        },
    },
    ProfileControl {
        id: "low_atten",
        label: "Low Atten",
        mapping: ParamMapping::Direct {
            param: "low_shelf_gain",
            range: -16.0..=0.0,
        },
    },
    ProfileControl {
        id: "high_freq",
        label: "High Frequency",
        mapping: ParamMapping::Stepped {
            param: "band_1_freq",
            values: &[3000.0, 4000.0, 5000.0, 8000.0, 10000.0, 12000.0, 16000.0],
            labels: &["3k", "4k", "5k", "8k", "10k", "12k", "16k"],
        },
    },
    ProfileControl {
        id: "high_boost",
        label: "High Boost",
        mapping: ParamMapping::Direct {
            param: "band_1_gain",
            range: 0.0..=16.0,
        },
    },
    ProfileControl {
        id: "high_bandwidth",
        label: "Bandwidth",
        mapping: ParamMapping::Direct {
            param: "band_1_q",
            range: 0.3..=3.0,
        },
    },
];

static PULTEC_CONSTRAINTS: [Constraint; 2] = [
    // Pultec uses exactly 2 bands — lock band count
    Constraint::Fixed {
        param: "band_count",
        value: 2.0,
    },
    // Low band is always a low shelf type
    Constraint::Fixed {
        param: "band_0_type",
        value: 0.0, // 0 = low shelf
    },
];
