//! Control profile — full parametric access (Pro-Q style).
//!
//! No constraints, all parameters exposed. This is the "advanced view."

use crate::core::{Constraint, ParamMapping, Profile, ProfileControl};

pub struct ControlProfile;

impl Profile for ControlProfile {
    fn id(&self) -> &'static str {
        "eq_control"
    }

    fn name(&self) -> &'static str {
        "Control"
    }

    fn controls(&self) -> &[ProfileControl] {
        // Control view builds its UI dynamically from the EqChain state
        // (variable band count, drag-to-add, etc.) — no fixed control list.
        &[]
    }

    fn constraints(&self) -> &[Constraint] {
        // No constraints — full access to all DSP params.
        &[]
    }
}
