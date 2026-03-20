//! Neve 1073 EQ profile — 3 bands + HPF with fixed frequency selections.

use crate::core::{Profile, ProfileControl};

pub struct Neve1073Profile;

impl Profile for Neve1073Profile {
    fn id(&self) -> &'static str {
        "eq_neve_1073"
    }
    fn name(&self) -> &'static str {
        "Neve 1073"
    }
    fn controls(&self) -> &[ProfileControl] {
        &[]
    } // TODO
    fn constraints(&self) -> &[crate::core::Constraint] {
        &[]
    } // TODO
}
