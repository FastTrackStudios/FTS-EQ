//! Complete EQ chain — manages up to 24 bands using v2 design pipeline.

use crate::band::Band;

pub const MAX_BANDS: usize = 24;

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

    pub fn band_mut(&mut self, index: usize) -> Option<&mut Band> {
        self.bands.get_mut(index)
    }

    pub fn band(&self, index: usize) -> Option<&Band> {
        self.bands.get(index)
    }

    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    pub fn update_band(&mut self, index: usize) {
        if let Some(band) = self.bands.get_mut(index) {
            band.update(self.sample_rate);
        }
    }

    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate;
        for band in &mut self.bands {
            band.update(sample_rate);
        }
    }

    pub fn reset(&mut self) {
        for band in &mut self.bands {
            band.reset();
        }
    }

    pub fn process(&mut self, left: &mut [f64], right: &mut [f64]) {
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
