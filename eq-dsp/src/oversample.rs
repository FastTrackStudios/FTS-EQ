//! High-transparency 4× oversampling for EQ processing.
//!
//! Uses the `halfband` crate's IIR allpass polyphase filters (HIIR port)
//! for sample rate conversion. Two cascaded 2× stages give 4× total.
//!
//! Architecture:
//! 1. Upsample 1× → 2× → 4× (two halfband interpolation stages)
//! 2. Process at 4× rate (in f64)
//! 3. Downsample 4× → 2× → 1× (two halfband decimation stages)
//!
//! Uses 10-coefficient filters (~125 dB stopband rejection, 4 samples total latency).

use halfband::iir;

/// Maximum block size at 4× rate.
const MAX_OS_BLOCK: usize = 8192 * 4;

/// Stereo 4× oversampler using cascaded HIIR halfband stages.
pub struct EqOversampler {
    // Stage 1: 1× ↔ 2×
    up1_l: iir::Upsampler10,
    up1_r: iir::Upsampler10,
    down1_l: iir::Downsampler10,
    down1_r: iir::Downsampler10,
    // Stage 2: 2× ↔ 4×
    up2_l: iir::Upsampler10,
    up2_r: iir::Upsampler10,
    down2_l: iir::Downsampler10,
    down2_r: iir::Downsampler10,
    // Intermediate buffers (f32 for halfband, f64 for processing)
    buf_2x_l: Vec<f32>,
    buf_2x_r: Vec<f32>,
    buf_4x_l: Vec<f64>,
    buf_4x_r: Vec<f64>,
}

impl EqOversampler {
    pub fn new() -> Self {
        Self {
            up1_l: iir::Upsampler10::default(),
            up1_r: iir::Upsampler10::default(),
            down1_l: iir::Downsampler10::default(),
            down1_r: iir::Downsampler10::default(),
            up2_l: iir::Upsampler10::default(),
            up2_r: iir::Upsampler10::default(),
            down2_l: iir::Downsampler10::default(),
            down2_r: iir::Downsampler10::default(),
            buf_2x_l: vec![0.0; MAX_OS_BLOCK / 2],
            buf_2x_r: vec![0.0; MAX_OS_BLOCK / 2],
            buf_4x_l: vec![0.0; MAX_OS_BLOCK],
            buf_4x_r: vec![0.0; MAX_OS_BLOCK],
        }
    }

    /// Process stereo audio through 4× oversampling.
    ///
    /// The callback receives mutable f64 slices at 4× the input rate.
    pub fn process_stereo<F>(&mut self, left: &mut [f64], right: &mut [f64], mut callback: F)
    where
        F: FnMut(&mut [f64], &mut [f64]),
    {
        let n = left.len();
        let n2 = n * 2;
        let n4 = n * 4;

        // Stage 1: upsample 1× → 2×
        for i in 0..n {
            let [a, b] = self.up1_l.process(left[i] as f32);
            self.buf_2x_l[i * 2] = a;
            self.buf_2x_l[i * 2 + 1] = b;
            let [a, b] = self.up1_r.process(right[i] as f32);
            self.buf_2x_r[i * 2] = a;
            self.buf_2x_r[i * 2 + 1] = b;
        }

        // Stage 2: upsample 2× → 4×, converting to f64 for processing
        for i in 0..n2 {
            let [a, b] = self.up2_l.process(self.buf_2x_l[i]);
            self.buf_4x_l[i * 2] = a as f64;
            self.buf_4x_l[i * 2 + 1] = b as f64;
            let [a, b] = self.up2_r.process(self.buf_2x_r[i]);
            self.buf_4x_r[i * 2] = a as f64;
            self.buf_4x_r[i * 2 + 1] = b as f64;
        }

        // Process at 4× rate in f64
        callback(&mut self.buf_4x_l[..n4], &mut self.buf_4x_r[..n4]);

        // Stage 1: downsample 4× → 2× (f64 → f32 at boundary)
        for i in 0..n2 {
            self.buf_2x_l[i] = self
                .down1_l
                .process(self.buf_4x_l[i * 2] as f32, self.buf_4x_l[i * 2 + 1] as f32);
            self.buf_2x_r[i] = self
                .down1_r
                .process(self.buf_4x_r[i * 2] as f32, self.buf_4x_r[i * 2 + 1] as f32);
        }

        // Stage 2: downsample 2× → 1× (f32 → f64 output)
        for i in 0..n {
            left[i] =
                self.down2_l
                    .process(self.buf_2x_l[i * 2], self.buf_2x_l[i * 2 + 1]) as f64;
            right[i] =
                self.down2_r
                    .process(self.buf_2x_r[i * 2], self.buf_2x_r[i * 2 + 1]) as f64;
        }
    }

    pub fn reset(&mut self) {
        self.up1_l.clear();
        self.up1_r.clear();
        self.up2_l.clear();
        self.up2_r.clear();
        self.down1_l.clear();
        self.down1_r.clear();
        self.down2_l.clear();
        self.down2_r.clear();
    }
}

impl Default for EqOversampler {
    fn default() -> Self {
        Self::new()
    }
}
