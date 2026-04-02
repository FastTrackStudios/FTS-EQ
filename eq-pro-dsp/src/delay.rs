//! Anti-cramping delay filter cascade (3-level).
//!
//! Pro-Q 4 uses a 3-level delay cascade for group delay compensation
//! in frequency-dependent phase modes. Each level provides up to
//! MAX_DELAY_SAMPLES_PER_LEVEL samples of delay with a circular
//! power-of-2 buffer for efficient modular indexing.

use crate::constants::{MAX_DELAY_SAMPLES_PER_LEVEL, NUM_DELAY_LEVELS};

/// Single delay line with circular power-of-2 buffer.
///
/// Uses a bitmask for wrap-around instead of modulo, matching Pro-Q 4's
/// efficient delay implementation.
pub struct DelayFilter {
    buffer: Vec<f64>,
    mask: usize,      // power-of-2 - 1
    write_pos: usize,
    delay_samples: usize,
}

impl DelayFilter {
    /// Create a new delay filter with the given maximum delay in samples.
    ///
    /// The buffer is allocated as the next power of 2 >= max_delay + 1.
    pub fn new(max_delay: usize) -> Self {
        let size = (max_delay + 1).next_power_of_two();
        Self {
            buffer: vec![0.0; size],
            mask: size - 1,
            write_pos: 0,
            delay_samples: 0,
        }
    }

    /// Set the delay in samples. Clamped to the buffer capacity.
    pub fn set_delay(&mut self, samples: usize) {
        // mask + 1 is the buffer size; max delay is mask
        self.delay_samples = samples.min(self.mask);
    }

    /// Process one sample: write to buffer, read from delayed position.
    #[inline]
    pub fn process(&mut self, input: f64) -> f64 {
        self.buffer[self.write_pos] = input;
        let read_pos = (self.write_pos + self.buffer.len() - self.delay_samples) & self.mask;
        let output = self.buffer[read_pos];
        self.write_pos = (self.write_pos + 1) & self.mask;
        output
    }

    /// Reset the buffer to zero and rewind the write position.
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.write_pos = 0;
    }
}

/// 3-level delay cascade for anti-cramping group delay compensation.
///
/// Each level can independently compensate for the group delay of
/// different frequency bands, providing up to NUM_DELAY_LEVELS *
/// MAX_DELAY_SAMPLES_PER_LEVEL total delay.
pub struct DelayFilterCascade {
    delays: [DelayFilter; NUM_DELAY_LEVELS],
}

impl DelayFilterCascade {
    /// Create a new cascade with buffers sized for the given sample rate.
    ///
    /// Each level uses MAX_DELAY_SAMPLES_PER_LEVEL as its maximum.
    pub fn new(_sample_rate: f64) -> Self {
        Self {
            delays: [
                DelayFilter::new(MAX_DELAY_SAMPLES_PER_LEVEL),
                DelayFilter::new(MAX_DELAY_SAMPLES_PER_LEVEL),
                DelayFilter::new(MAX_DELAY_SAMPLES_PER_LEVEL),
            ],
        }
    }

    /// Set the group delay for a specific cascade level.
    ///
    /// `level` is 0..NUM_DELAY_LEVELS, `delay_samples` is the delay
    /// in samples for that level.
    pub fn set_group_delay(&mut self, level: usize, delay_samples: usize) {
        if level < NUM_DELAY_LEVELS {
            self.delays[level].set_delay(delay_samples);
        }
    }

    /// Process one sample through all cascade levels in series.
    pub fn process(&mut self, input: f64) -> f64 {
        let mut out = input;
        for delay in &mut self.delays {
            out = delay.process(out);
        }
        out
    }

    /// Reset all cascade levels.
    pub fn reset(&mut self) {
        for delay in &mut self.delays {
            delay.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_delay_passes_through() {
        let mut d = DelayFilter::new(64);
        d.set_delay(0);
        for i in 0..10 {
            let input = i as f64 * 0.1;
            let output = d.process(input);
            assert!(
                (output - input).abs() < 1e-14,
                "Zero delay should pass through: {output} != {input}"
            );
        }
    }

    #[test]
    fn one_sample_delay() {
        let mut d = DelayFilter::new(64);
        d.set_delay(1);

        // First output should be 0.0 (buffer initialized to zero)
        let out0 = d.process(1.0);
        assert!(
            out0.abs() < 1e-14,
            "First sample with delay=1 should be 0, got {out0}"
        );

        // Second output should be the first input
        let out1 = d.process(2.0);
        assert!(
            (out1 - 1.0).abs() < 1e-14,
            "Second sample should be first input (1.0), got {out1}"
        );

        let out2 = d.process(3.0);
        assert!(
            (out2 - 2.0).abs() < 1e-14,
            "Third sample should be second input (2.0), got {out2}"
        );
    }

    #[test]
    fn delay_wraps_around() {
        let mut d = DelayFilter::new(4);
        d.set_delay(3);

        // Feed 10 samples and verify delay behavior
        let mut outputs = Vec::new();
        for i in 0..10 {
            outputs.push(d.process((i + 1) as f64));
        }

        // First 3 outputs should be 0 (initial buffer)
        for (i, &out) in outputs.iter().take(3).enumerate() {
            assert!(
                out.abs() < 1e-14,
                "Sample {i} should be 0.0, got {out}"
            );
        }

        // After that, output should be input delayed by 3
        for i in 3..10 {
            let expected = (i - 2) as f64; // input was i+1, delayed by 3
            assert!(
                (outputs[i] - expected).abs() < 1e-14,
                "Sample {i}: expected {expected}, got {}",
                outputs[i]
            );
        }
    }

    #[test]
    fn reset_clears_buffer() {
        let mut d = DelayFilter::new(64);
        d.set_delay(2);

        // Build up state
        d.process(1.0);
        d.process(2.0);
        d.process(3.0);

        d.reset();

        // After reset, should get zeros
        let out = d.process(5.0);
        assert!(
            out.abs() < 1e-14,
            "After reset with delay=2, first output should be 0, got {out}"
        );
    }

    #[test]
    fn power_of_2_buffer_size() {
        let d = DelayFilter::new(100);
        // Buffer should be next power of 2 >= 101 = 128
        assert_eq!(d.buffer.len(), 128);
        assert_eq!(d.mask, 127);
    }

    #[test]
    fn cascade_zero_delay() {
        let mut cascade = DelayFilterCascade::new(48000.0);
        // All delays at 0 = passthrough
        for i in 0..10 {
            let input = i as f64;
            let output = cascade.process(input);
            assert!(
                (output - input).abs() < 1e-14,
                "Cascade with zero delay should pass through"
            );
        }
    }

    #[test]
    fn cascade_additive_delay() {
        let mut cascade = DelayFilterCascade::new(48000.0);
        cascade.set_group_delay(0, 1);
        cascade.set_group_delay(1, 1);
        cascade.set_group_delay(2, 1);

        // Total delay = 3 samples
        let mut outputs = Vec::new();
        for i in 0..8 {
            outputs.push(cascade.process((i + 1) as f64));
        }

        // First 3 should be ~0
        for (i, &out) in outputs.iter().take(3).enumerate() {
            assert!(
                out.abs() < 1e-14,
                "Sample {i} should be 0.0 with total delay=3, got {out}"
            );
        }

        // Sample 3 should be input 1.0
        assert!(
            (outputs[3] - 1.0).abs() < 1e-14,
            "Sample 3 should be 1.0, got {}",
            outputs[3]
        );
    }

    #[test]
    fn cascade_reset() {
        let mut cascade = DelayFilterCascade::new(48000.0);
        cascade.set_group_delay(0, 2);

        cascade.process(1.0);
        cascade.process(2.0);
        cascade.process(3.0);

        cascade.reset();

        let out = cascade.process(5.0);
        assert!(
            out.abs() < 1e-14,
            "After cascade reset, first output should be 0, got {out}"
        );
    }

    #[test]
    fn set_group_delay_out_of_range_ignored() {
        let mut cascade = DelayFilterCascade::new(48000.0);
        // Should not panic
        cascade.set_group_delay(5, 10);
    }
}
