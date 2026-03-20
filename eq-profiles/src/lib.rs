//! EQ hardware profiles — parameter mappings and constraints.
//!
//! A profile defines:
//! - Which controls appear (knobs, switches, stepped selectors)
//! - How each control maps to [`eq_dsp::EqChain`] parameters
//! - Constraints (locked frequencies, stepped values, linked params)
//!
//! Profiles are pure data + mapping functions. No GUI, no framework deps.

pub mod api_550a;
pub mod control;
pub mod core;
pub mod neve_1073;
pub mod pultec;
pub mod ssl_e;

pub use self::core::{Constraint, ParamMapping, Profile, ProfileControl};
