//! Profile trait and types — shared by all plugin categories.
//!
//! This module defines the core abstraction that all profiles implement.

use std::ops::RangeInclusive;

/// A hardware profile — defines what controls exist and how they map to DSP params.
pub trait Profile {
    /// Unique identifier for this profile.
    fn id(&self) -> &'static str;

    /// Display name.
    fn name(&self) -> &'static str;

    /// The controls this profile exposes.
    fn controls(&self) -> &[ProfileControl];

    /// Constraints that lock or limit core params when this profile is active.
    fn constraints(&self) -> &[Constraint];
}

/// A single control (knob, switch, selector) exposed by a profile.
pub struct ProfileControl {
    pub id: &'static str,
    pub label: &'static str,
    pub mapping: ParamMapping,
}

/// How a profile control maps to DSP parameters.
pub enum ParamMapping {
    /// One knob → one DSP param, continuous range.
    Direct {
        param: &'static str,
        range: RangeInclusive<f64>,
    },

    /// One knob → one DSP param, stepped/detented values.
    Stepped {
        param: &'static str,
        values: &'static [f64],
        labels: &'static [&'static str],
    },

    /// One knob → multiple DSP params on linked curves.
    /// (e.g., LA-2A "Peak Reduction" drives threshold + ratio + knee)
    Compound {
        mappings: &'static [(&'static str, fn(f64) -> f64)],
        range: RangeInclusive<f64>,
    },
}

/// A constraint that locks a DSP param when a profile is active.
pub enum Constraint {
    /// Lock a param to a fixed value.
    Fixed { param: &'static str, value: f64 },

    /// Limit a param to a range (narrower than the DSP core supports).
    Clamped {
        param: &'static str,
        range: RangeInclusive<f64>,
    },

    /// Lock a param to stepped values only.
    SteppedOnly {
        param: &'static str,
        values: &'static [f64],
    },
}
