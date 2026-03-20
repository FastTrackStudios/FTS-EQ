//! Complete EQ chain — manages up to 24 bands.

use crate::band::Band;
use fts_dsp::{AudioConfig, Processor};

// r[impl eq.chain.max-bands]
/// Maximum number of simultaneous EQ bands.
pub const MAX_BANDS: usize = 24;

// r[impl eq.chain.signal-flow]
/// Full EQ processing chain.
///
/// Processes audio through all enabled bands in series.
/// Bands are processed in order of their index.
pub struct EqChain {
    bands: Vec<Band>,
    config: AudioConfig,
}

impl EqChain {
    pub fn new() -> Self {
        Self {
            bands: Vec::new(),
            config: AudioConfig {
                sample_rate: 44100.0,
                max_buffer_size: 512,
            },
        }
    }

    // r[impl eq.chain.dynamic-bands]
    /// Add a new band with default parameters. Returns the band index.
    pub fn add_band(&mut self) -> usize {
        if self.bands.len() >= MAX_BANDS {
            return self.bands.len() - 1;
        }
        let idx = self.bands.len();
        let mut band = Band::new();
        band.update(self.config);
        self.bands.push(band);
        idx
    }

    // r[impl eq.chain.dynamic-bands]
    /// Remove a band by index.
    pub fn remove_band(&mut self, index: usize) {
        if index < self.bands.len() {
            self.bands.remove(index);
        }
    }

    /// Get a mutable reference to a band for parameter changes.
    pub fn band_mut(&mut self, index: usize) -> Option<&mut Band> {
        self.bands.get_mut(index)
    }

    /// Get an immutable reference to a band.
    pub fn band(&self, index: usize) -> Option<&Band> {
        self.bands.get(index)
    }

    /// Number of active bands.
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Update a single band's coefficients after parameter change.
    pub fn update_band(&mut self, index: usize) {
        if let Some(band) = self.bands.get_mut(index) {
            band.update(self.config);
        }
    }
}

impl Processor for EqChain {
    fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    fn update(&mut self, config: AudioConfig) {
        self.config = config;
        for band in &mut self.bands {
            band.update(config);
        }
    }

    fn process(&mut self, left: &mut [f64], right: &mut [f64]) {
        for i in 0..left.len() {
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
