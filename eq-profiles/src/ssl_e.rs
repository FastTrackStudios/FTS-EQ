//! SSL E-Series EQ profile — 4 bands (LF/LMF/HMF/HF) + HPF/LPF.

use crate::core::{Profile, ProfileControl};

pub struct SslEProfile;

impl Profile for SslEProfile {
    fn id(&self) -> &'static str {
        "eq_ssl_e"
    }
    fn name(&self) -> &'static str {
        "SSL E-Series"
    }
    fn controls(&self) -> &[ProfileControl] {
        &[]
    } // TODO
    fn constraints(&self) -> &[crate::core::Constraint] {
        &[]
    } // TODO
}
