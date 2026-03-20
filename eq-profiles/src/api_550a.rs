//! API 550A EQ profile — 3 bands with proportional Q.

use crate::core::{Profile, ProfileControl};

pub struct Api550aProfile;

impl Profile for Api550aProfile {
    fn id(&self) -> &'static str {
        "eq_api_550a"
    }
    fn name(&self) -> &'static str {
        "API 550A"
    }
    fn controls(&self) -> &[ProfileControl] {
        &[]
    } // TODO
    fn constraints(&self) -> &[crate::core::Constraint] {
        &[]
    } // TODO
}
