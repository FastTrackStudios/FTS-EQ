//! Filter type enum shared across the EQ engine.

// r[impl eq.band.filter-types]
/// All supported EQ filter types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    Peak,
    LowShelf,
    HighShelf,
    TiltShelf,
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    /// Band shelf: peak-like shape built from opposing shelf pair.
    /// Used for higher-order peak filters (order > 2).
    BandShelf,
}

/// Filter structure selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterStructure {
    /// Transposed Direct Form II — minimum phase, lowest latency.
    Tdf2,
    /// State Variable Filter — better for modulation, simultaneous outputs.
    Svf,
}

impl FilterType {
    /// Whether this filter type uses a gain parameter.
    pub fn has_gain(self) -> bool {
        matches!(
            self,
            FilterType::Peak
                | FilterType::LowShelf
                | FilterType::HighShelf
                | FilterType::TiltShelf
                | FilterType::BandShelf
        )
    }

    /// Whether this filter type uses a Q parameter.
    pub fn has_q(self) -> bool {
        // All types use Q except notch (fixed by design)
        !matches!(self, FilterType::Notch)
    }
}
