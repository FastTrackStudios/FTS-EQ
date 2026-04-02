//! EQ chain managing up to 24 bands using the pro design pipeline.
//!
//! Equivalent to Pro-Q 4's per-channel processing chain that cascades
//! all enabled bands in series.

use crate::band::Band;

/// Maximum number of bands in a chain (matches Pro-Q 4's 24-band limit).
pub const MAX_BANDS: usize = 24;

/// Complete EQ processing chain.
pub struct EqChain {
    bands: Vec<Band>,
    sample_rate: f64,
}

impl EqChain {
    pub fn new() -> Self {
        Self {
            bands: Vec::new(),
            sample_rate: 48000.0,
        }
    }

    /// Add a new band and return its index.
    ///
    /// Returns the last valid index if the chain is already at MAX_BANDS.
    pub fn add_band(&mut self) -> usize {
        if self.bands.len() >= MAX_BANDS {
            return self.bands.len() - 1;
        }
        let idx = self.bands.len();
        let mut band = Band::new();
        band.update(self.sample_rate);
        self.bands.push(band);
        idx
    }

    /// Get a mutable reference to a band by index.
    pub fn band_mut(&mut self, index: usize) -> Option<&mut Band> {
        self.bands.get_mut(index)
    }

    /// Get an immutable reference to a band by index.
    pub fn band(&self, index: usize) -> Option<&Band> {
        self.bands.get(index)
    }

    /// Return the number of bands in the chain.
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Recalculate coefficients for a single band.
    pub fn update_band(&mut self, index: usize) {
        if let Some(band) = self.bands.get_mut(index) {
            band.update(self.sample_rate);
        }
    }

    /// Set the sample rate and recalculate all band coefficients.
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for band in &mut self.bands {
            band.update(sample_rate);
        }
    }

    /// Reset all band processing state to zero.
    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    /// Process interleaved stereo buffers through all bands.
    ///
    /// Each sample passes through all bands in series (left then right).
    pub fn process(&mut self, left: &mut [f64], right: &mut [f64]) {
        for i in 0..left.len().min(right.len()) {
            for band in &mut self.bands {
                left[i] = band.tick(left[i], 0);
                right[i] = band.tick(right[i], 1);
            }
        }
    }
}

impl Default for EqChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::FilterType;

    #[test]
    fn empty_chain_passes_through() {
        let mut chain = EqChain::new();
        let mut left = vec![1.0; 4];
        let mut right = vec![0.5; 4];
        chain.process(&mut left, &mut right);
        assert_eq!(left, vec![1.0; 4]);
        assert_eq!(right, vec![0.5; 4]);
    }

    #[test]
    fn add_band_returns_index() {
        let mut chain = EqChain::new();
        assert_eq!(chain.add_band(), 0);
        assert_eq!(chain.add_band(), 1);
        assert_eq!(chain.num_bands(), 2);
    }

    #[test]
    fn max_bands_limit() {
        let mut chain = EqChain::new();
        for _ in 0..MAX_BANDS {
            chain.add_band();
        }
        assert_eq!(chain.num_bands(), MAX_BANDS);
        // Adding one more should return last valid index
        let idx = chain.add_band();
        assert_eq!(idx, MAX_BANDS - 1);
        assert_eq!(chain.num_bands(), MAX_BANDS);
    }

    #[test]
    fn band_mut_configures_band() {
        let mut chain = EqChain::new();
        let idx = chain.add_band();
        if let Some(band) = chain.band_mut(idx) {
            band.filter_type = FilterType::Lowpass;
            band.freq_hz = 2000.0;
            band.order = 4;
        }
        chain.update_band(idx);

        let band = chain.band(idx).unwrap();
        assert_eq!(band.filter_type, FilterType::Lowpass);
        assert_eq!(band.freq_hz, 2000.0);
    }

    #[test]
    fn set_sample_rate_updates_all() {
        let mut chain = EqChain::new();
        chain.add_band();
        chain.add_band();
        // Should not panic
        chain.set_sample_rate(96000.0);
    }

    #[test]
    fn reset_does_not_panic() {
        let mut chain = EqChain::new();
        chain.add_band();
        chain.reset();
    }
}
