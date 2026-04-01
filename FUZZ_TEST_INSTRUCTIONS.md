# FTS-EQ Fuzz Test: Iteration Guide

This document explains how to build, run, and interpret the fuzz test that compares FTS-EQ against FabFilter Pro-Q 4. Use this to iterate on DSP fixes until the two plugins match.

## Quick Start

All commands must run inside `nix develop` from the **fts-analyzer** directory:

```bash
cd /home/cody/Development/FastTrackStudio/fts-analyzer
```

### 1. Build FTS-EQ

```bash
nix develop --command bash -c \
  "cd /home/cody/Development/FastTrackStudio/FTS-Plugins && cargo build --release --package eq-plugin"
```

Then copy the shared library into the CLAP bundle location:

```bash
cp /home/cody/Development/FastTrackStudio/FTS-Plugins/target/release/libeq_plugin.so \
   /home/cody/Development/FastTrackStudio/FTS-Plugins/target/bundled/eq-plugin.clap
```

### 2. Build the Analyzer CLI

```bash
nix develop --command bash -c "cargo build --release --package fts-analyzer-cli"
```

### 3. Run the Fuzz Test

```bash
nix develop --command bash -c \
  "./target/release/fts-analyzer-cli fuzz-eq \
    '/home/cody/.clap/yabridge/FabFilter Pro-Q 4.clap' \
    /home/cody/Development/FastTrackStudio/FTS-Plugins/target/bundled/eq-plugin.clap \
    --iterations 20 --bands 1 --seed 99 --tolerance-db 1.0"
```

Key flags:
- `--iterations N` — number of randomized parameter sets to test
- `--bands N` — number of EQ bands to randomize (max 24)
- `--seed N` — deterministic PRNG seed (same seed = same test sequence)
- `--tolerance-db N` — RMS frequency response difference threshold for pass/fail
- `--duration N` — test signal length in seconds (default 2.0)
- `--sample-rate N` — default 44100
- `--block-size N` — default 512

### Full Rebuild + Test (one-liner)

```bash
nix develop --command bash -c \
  "cd /home/cody/Development/FastTrackStudio/FTS-Plugins && cargo build --release --package eq-plugin && \
   cp target/release/libeq_plugin.so target/bundled/eq-plugin.clap && \
   cd /home/cody/Development/FastTrackStudio/fts-analyzer && cargo build --release --package fts-analyzer-cli && \
   ./target/release/fts-analyzer-cli fuzz-eq \
     '/home/cody/.clap/yabridge/FabFilter Pro-Q 4.clap' \
     /home/cody/Development/FastTrackStudio/FTS-Plugins/target/bundled/eq-plugin.clap \
     --iterations 20 --bands 1 --seed 99 --tolerance-db 1.0"
```

## How the Fuzz Test Works

1. Loads both plugins as CLAP instances via clack-host
2. For each iteration:
   - Generates white noise input (deterministic from seed)
   - Randomizes shared parameters (Freq, Gain, Q, Shape, Slope) using display-text values
   - Processes the noise through both plugins
   - Computes frequency response H(f) via Welch's method (Hann-windowed 4096-point FFT)
   - Compares per-bin transfer functions: H_a(f) vs H_b(f) across 20 Hz–20 kHz
3. Reports RMS and max dB difference per iteration; prints worst bins on failure

## Parameter Mapping

| Parameter | Pro-Q 4 Name       | FTS-EQ Name  | Values                          |
|-----------|--------------------|--------------|---------------------------------|
| Frequency | `Band N Frequency` | `BN Freq`    | 20 Hz – 20000 Hz (display text) |
| Gain      | `Band N Gain`      | `BN Gain`    | -30.0 dB – 30.0 dB             |
| Q         | `Band N Q`         | `BN Q`       | 0.10 – 18.00                    |
| Shape     | `Band N Shape`     | `BN Type`    | 0–9 (integer, see below)        |
| Slope     | `Band N Slope`     | `BN Slope`   | 0–10 (integer, see below)       |
| Enabled   | `Band N Enabled`   | `BN On`      | 0/1                             |
| Used      | `Band N Used`      | (N/A)        | Pro-Q 4 only, set to 1          |

### Shape Values (0–9)

| Value | Pro-Q 4 Filter  | FTS-EQ FilterType |
|-------|-----------------|-------------------|
| 0     | Bell            | Peak              |
| 1     | Low Shelf       | LowShelf          |
| 2     | Low Cut (HPF)   | Highpass           |
| 3     | High Shelf      | HighShelf          |
| 4     | High Cut (LPF)  | Lowpass            |
| 5     | Notch           | Notch              |
| 6     | Band Pass       | Bandpass           |
| 7     | Tilt Shelf      | TiltShelf          |
| 8     | Flat Tilt       | **Not implemented** (mapped to Peak) |
| 9     | AllPass          | **Not implemented** (mapped to Peak) |

### Slope Values (0–10)

| Value | dB/oct  | Filter Order |
|-------|---------|-------------|
| 0     | 6       | 1           |
| 1     | 12      | 2           |
| 2     | 18      | 3 (default) |
| 3     | 24      | 4           |
| 4     | 30      | 5           |
| 5     | 36      | 6           |
| 6     | 48      | 8           |
| 7     | 60      | 10          |
| 8     | 72      | 12          |
| 9     | 96      | 16          |
| 10    | Brickwall | 16        |

Note: FTS-EQ's MAX_ORDER is 12 (MAX_SECTIONS=6). Orders above 12 require increasing this limit.

## Reading Fuzz Test Output

```
[   1/20] PASS rms_diff=0.31 dB  max_diff=1.42 dB @ 19624 Hz  (1849 bins)
[   2/20] FAIL rms_diff=5.23 dB  max_diff=24.81 dB @ 10000 Hz  (1849 bins)
         Band 1 Freq=5000 Hz
         Band 1 Gain=12.0 dB
         Band 1 Q=2.00
         Band 1 Shape=6
         Band 1 Slope=8
         10000 Hz: A=-3.21 dB  B=21.60 dB  delta=+24.81 dB
         ...
```

- **rms_diff**: RMS of per-bin dB differences across 20–20k Hz. This is the primary metric.
- **max_diff**: Worst single-bin difference. Shows where the response diverges most.
- **A**: Pro-Q 4's transfer function at that bin.
- **B**: FTS-EQ's transfer function at that bin.
- **delta**: B - A (positive = FTS-EQ is louder at that frequency).

On failure, the top 5 worst frequency bins are printed along with the parameter set that caused it.

## Known DSP Issues (Priority Order)

### 1. AllPass (Shape 9) — Not Implemented
**Status**: Mapped to Peak as placeholder. Outputs wrong response.
**Fix**: Implement a proper AllPass filter in eq-dsp. An AllPass has unity magnitude response but phase shift. The biquad coefficients for a 2nd-order AllPass are well-documented.
**Files**: `eq-dsp/src/band.rs` (dispatch), `eq-dsp/src/coeff.rs` (coefficients)

### 2. Flat Tilt (Shape 8) — Not Implemented
**Status**: Mapped to Peak as placeholder. Pro-Q 4's Flat Tilt applies a broadband tilt centered at the frequency parameter.
**Fix**: Research Pro-Q 4's Flat Tilt behavior and implement. It likely uses a series of shelf filters or an analog-matched tilt network.
**Files**: `eq-dsp/src/band.rs`, `eq-dsp/src/filter_type.rs`

### 3. High Slope Divergence (Slopes 8–10)
**Status**: Slopes 8 (72 dB/oct, order 12) and above show significant divergence, especially near Nyquist. This is likely due to cascaded biquad numerical precision at high orders.
**Fix**: Consider using SVF (state-variable filter) topology for high orders, or investigate if the Vicanek matched coefficients need adjustment at high orders. MAX_ORDER=12 also caps the achievable slope.
**Files**: `eq-dsp/src/band.rs` (MAX_ORDER, MAX_SECTIONS), `eq-dsp/src/coeff.rs`, `eq-dsp/src/tdf2.rs`, `eq-dsp/src/svf.rs`

### 4. Bandpass (Shape 6) with High Slopes
**Status**: Currently forced to 2nd order regardless of slope setting. Pro-Q 4 supports higher-order bandpass.
**Fix**: Remove the 2nd-order restriction for Bandpass and implement higher-order bandpass via cascaded sections.
**Files**: `eq-dsp/src/band.rs` (see the order clamping logic)

### 5. Tilt Shelf (Shape 7) Differences
**Status**: Minor differences in response shape compared to Pro-Q 4, especially at extreme Q values and slopes.
**Fix**: Compare coefficient calculation against Pro-Q 4's behavior across parameter ranges.
**Files**: `eq-dsp/src/coeff.rs`, `eq-dsp/src/band.rs`

## DSP Code Structure

All EQ DSP code lives in:
```
/home/cody/Development/FastTrackStudio/FTS-Plugins/crates/fts-eq/eq-dsp/src/
```

| File              | Purpose                                             |
|-------------------|-----------------------------------------------------|
| `band.rs`         | `Band` struct: holds filter state, dispatches update/tick |
| `filter_type.rs`  | `FilterType` enum (Peak, LowShelf, Highpass, etc.)  |
| `coeff.rs`        | Vicanek matched biquad coefficient calculation       |
| `tdf2.rs`         | Transposed Direct Form II biquad section             |
| `svf.rs`          | State Variable Filter biquad section                 |
| `lib.rs`          | Top-level EQ processor, per-band processing          |

The plugin wrapper (parameter handling, CLAP interface) lives in:
```
/home/cody/Development/FastTrackStudio/FTS-Plugins/crates/fts-eq/eq-plugin/src/lib.rs
```

Key constants in `band.rs`:
- `MAX_ORDER = 12` — maximum filter order (6 cascaded 2nd-order sections)
- `MAX_SECTIONS = 6` — maximum number of biquad sections per band

### How Band Processing Works

1. `Band::update(sample_rate)` is called when parameters change
2. Based on `filter_type`, it dispatches to `update_pass_filter`, `update_shelf_filter`, or `update_peak_filter`
3. Each update function computes biquad coefficients via `coeff.rs` (Vicanek matched design)
4. For higher orders, multiple 2nd-order sections are cascaded with Butterworth pole placement
5. `Band::tick(sample)` processes audio through all active sections in series

## Iteration Strategy

1. Pick the highest-priority issue from the list above
2. Modify the DSP code in `eq-dsp/src/`
3. Rebuild and run the fuzz test (use the one-liner above)
4. Start with `--bands 1 --iterations 20` for quick feedback
5. Once basic shapes pass, increase to `--bands 3 --iterations 100` for thorough testing
6. Lower `--tolerance-db` as fixes improve (target: 0.5 dB or lower for matching shapes)

To isolate a specific shape, use a fixed seed and low iteration count, then check which iterations use that shape in the output.
