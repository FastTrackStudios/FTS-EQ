# FTS-EQ — Reference Data & Comparison Guide

## Overview

We have a comprehensive reference capture of FabFilter Pro-Q 4's frequency response (magnitude + phase) across all 10 filter types, multiple slopes, Q values, gains, and 19 frequencies including dense coverage near Nyquist. Data is captured at both 48 kHz and 96 kHz.

Use this data to verify FTS-EQ matches Pro-Q 4's behavior — or to implement new filter types from scratch.

## Reference Data

**Location**: `/home/cody/Development/FastTrackStudio/fts-analyzer/reference/pro-q4/`

```
pro-q4/
├── 48k/              # 48 kHz sample rate
│   ├── metadata.json
│   └── *.bin         # 5,567 binary transfer functions
└── 96k/              # 96 kHz sample rate
    ├── metadata.json
    └── *.bin         # 5,567 binary transfer functions
```

### What's captured

**5,567 scenarios** at each sample rate:

#### Single-band (5,548 scenarios)

| Dimension   | Values | Count |
|-------------|--------|-------|
| Filter type | Bell(0), LowShelf(1), LowCut(2), HighShelf(3), HighCut(4), Notch(5), Bandpass(6), TiltShelf(7), FlatTilt(8), Allpass(9) | 10 |
| Frequency   | 20, 50, 100, 200, 500, 1k, 2k, 5k, 8k, 10k, 12k, 14k, 16k, 17k, 18k, 19k, 20k, 21k, 22k Hz | 19 |
| Gain        | -12, -6, +6, +12 dB (types with gain) or N/A | 4 or 1 |
| Q           | 0.5, 1.0, 4.0, 10.0 (types with Q) or N/A | 4 or 1 |
| Slope       | 6dB/oct(0), 18dB/oct(2), 36dB/oct(5), 72dB/oct(8) | 4 or 1 |

#### Multi-band interaction (19 scenarios)

| Category         | Scenarios |
|------------------|-----------|
| Shelf combos     | Low shelf 200 Hz + high shelf 8 kHz (4 gain combos) |
| Close bells      | Two bells near 1 kHz separated by 100/500/2k Hz (boost & cut) |
| Smiley/frown     | 3-band: low shelf + bell mid + high shelf |
| Cut combos       | Low cut 80 Hz + high cut 16 kHz at slopes 2, 5, 8 |
| Surgical         | Notch 500 Hz + bell 500 Hz rescue |
| Nyquist stacking | Two bells at 16k, 18k, 20k Hz |

### Measurement method

- **Signal**: 2 s white noise, deterministic seed 12345
- **FFT**: Welch's method, 4096-point Hann window, non-overlapping
- **Transfer function**: H(k) = avg[Y·conj(X)] / avg[|X|²] — gives magnitude (dB) and phase (radians)
- **Frequency resolution**: ~11.7 Hz at 48 kHz, ~23.4 Hz at 96 kHz

### Why two sample rates

At 48 kHz Nyquist is 24 kHz — filters near 20 kHz show cramping. At 96 kHz Nyquist is 48 kHz — the same filters behave more ideally. Comparing both shows how much cramping Pro-Q 4 has and sets the target for FTS-EQ.

---

## File Formats

### Binary (`.bin`)

One file per scenario, ~24 KB each at 48 kHz:

```
[num_bins: u32 LE]
[(freq_hz: f32 LE, mag_db: f32 LE, phase_rad: f32 LE) × num_bins]
```

### Metadata (`metadata.json`)

Contains signal parameters, FFT config, and all scenario definitions with per-band configs:

```json
{
  "sample_rate": 48000.0,
  "duration": 2.0,
  "fft_size": 4096,
  "block_size": 512,
  "includes_phase": true,
  "measurement": "white_noise_welch_cross_spectrum",
  "scenarios": [
    {
      "name": "bell_1000hz_+6db_q1_s2",
      "bands": [{ "shape": 0, "freq_hz": 1000.0, "gain_db": 6.0, "q": 1.0, "slope": 2 }]
    }
  ]
}
```

### Scenario naming

Single-band: `{filter}_{freq}hz_{gain}db_q{q}_s{slope}` or `{filter}_{freq}hz_q{q}_s{slope}` (no gain for types without gain)

```
bell_1000hz_+6db_q1_s2
low_cut_100hz_q1_s8
allpass_5000hz_q4_s0
flat_tilt_1000hz_+6db_s0
```

Multi-band: `multi_{description}`

```
multi_loshelf200_+6db_hishelf8k_+6db
multi_2xbell20000hz_+6db_q4
multi_locut80_hicut16k_s5
```

---

## Reading the Data

### Rust

```rust
use std::io::Read;

struct ResponseBin { freq_hz: f32, mag_db: f32, phase_rad: f32 }

fn read_eq_bin(path: &str) -> Vec<ResponseBin> {
    let data = std::fs::read(path).unwrap();
    let num_bins = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    (0..num_bins).map(|i| {
        let off = 4 + i * 12;
        ResponseBin {
            freq_hz:   f32::from_le_bytes(data[off..off+4].try_into().unwrap()),
            mag_db:    f32::from_le_bytes(data[off+4..off+8].try_into().unwrap()),
            phase_rad: f32::from_le_bytes(data[off+8..off+12].try_into().unwrap()),
        }
    }).collect()
}
```

### Python

```python
import struct, numpy as np

def read_eq_bin(path):
    with open(path, 'rb') as f:
        num_bins = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(num_bins * 12), dtype='<f4').reshape(-1, 3)
        return { 'freq_hz': data[:, 0], 'mag_db': data[:, 1], 'phase_rad': data[:, 2] }

# Example: read a bell at 1 kHz
ref = read_eq_bin('reference/pro-q4/48k/bell_1000hz_+6db_q1_s2.bin')
```

---

## Parameter Mapping: Pro-Q 4 ↔ FTS-EQ

| Parameter   | Pro-Q 4              | FTS-EQ       | Notes |
|-------------|----------------------|--------------|-------|
| Enabled     | `Band N Enabled`     | `BN On`      | Pro-Q 4 also needs `Band N Used` = 1 |
| Filter type | `Band N Shape`       | `BN Type`    | Same 0–9 integer mapping |
| Frequency   | `Band N Frequency`   | `BN Freq`    | Both accept display text like "1000 Hz" |
| Gain        | `Band N Gain`        | `BN Gain`    | Display text like "6.0 dB" |
| Q           | `Band N Q`           | `BN Q`       | Display text like "1.00" |
| Slope       | `Band N Slope`       | `BN Slope`   | Same 0–10 integer mapping |

### Filter types (Shape 0–9)

| Value | Name           | Has Gain | Has Q | DSP type in eq-dsp |
|-------|----------------|----------|-------|--------------------|
| 0     | Bell (Peak)    | Yes      | Yes   | `Peak`             |
| 1     | Low Shelf      | Yes      | Yes   | `LowShelf`         |
| 2     | Low Cut (HPF)  | No       | Yes   | `Highpass`          |
| 3     | High Shelf     | Yes      | Yes   | `HighShelf`         |
| 4     | High Cut (LPF) | No       | Yes   | `Lowpass`           |
| 5     | Notch          | No       | Yes   | `Notch`             |
| 6     | Bandpass       | No       | Yes   | `Bandpass`          |
| 7     | Tilt Shelf     | Yes      | No    | `TiltShelf`         |
| 8     | Flat Tilt      | Yes      | No    | `FlatTilt`          |
| 9     | Allpass        | No       | Yes   | `Allpass`           |

### Slopes (0–10)

| Value | Rate        | Filter Order |
|-------|-------------|-------------|
| 0     | 6 dB/oct    | 1           |
| 1     | 12 dB/oct   | 2           |
| 2     | 18 dB/oct   | 3 (default) |
| 3     | 24 dB/oct   | 4           |
| 4     | 30 dB/oct   | 5           |
| 5     | 36 dB/oct   | 6           |
| 6     | 48 dB/oct   | 8           |
| 7     | 60 dB/oct   | 10          |
| 8     | 72 dB/oct   | 12          |
| 9     | 96 dB/oct   | 16          |
| 10    | Brickwall   | 16          |

---

## Build & Compare Workflow

All commands run from the fts-analyzer directory:

```bash
cd /home/cody/Development/FastTrackStudio/fts-analyzer
```

### 1. Build FTS-EQ plugin

```bash
cd /home/cody/Development/FastTrackStudio/FTS-Plugins && \
  cargo build --release --package eq-plugin && \
  cp target/release/libeq_plugin.so target/bundled/eq-plugin.clap
```

### 2. Build the analyzer CLI

```bash
cd /home/cody/Development/FastTrackStudio/fts-analyzer && \
  cargo build --release --package fts-analyzer-cli
```

### 3. Run deterministic comparison against reference data

Compare FTS-EQ's output against the pre-captured Pro-Q 4 reference:

```bash
# 48 kHz
./target/release/fts-analyzer-cli compare-eq \
  /home/cody/Development/FastTrackStudio/FTS-Plugins/target/bundled/eq-plugin.clap \
  --reference ./reference/pro-q4 \
  --out ./results/fts-eq-48k \
  --sample-rate 48000 \
  --tolerance-db 1.0

# 96 kHz
./target/release/fts-analyzer-cli compare-eq \
  /home/cody/Development/FastTrackStudio/FTS-Plugins/target/bundled/eq-plugin.clap \
  --reference ./reference/pro-q4 \
  --out ./results/fts-eq-96k \
  --sample-rate 96000 \
  --tolerance-db 1.0
```

Output:

```
[    1/5567] PASS rms=0.12 dB max=0.45 dB  bell_20hz_-12db_q0.5_s0
[  200/5567] FAIL rms=2.34 dB max=5.67 dB  bell_20000hz_+12db_q10_s8
...
  Total: 5200/5567 passed
  Worst RMS diff: 2.340 dB
  Tolerance: 1.00 dB
```

### 4. Run fuzz testing (randomized multi-band)

For broader coverage with random parameter combinations:

```bash
./target/release/fts-analyzer-cli fuzz-eq \
  '/home/cody/.clap/yabridge/FabFilter Pro-Q 4.clap' \
  /home/cody/Development/FastTrackStudio/FTS-Plugins/target/bundled/eq-plugin.clap \
  --iterations 100 --bands 3 --tolerance-db 1.0
```

Flags:
- `--iterations N` — number of randomized tests
- `--bands N` — EQ bands per test (max 24)
- `--seed N` — deterministic PRNG seed
- `--tolerance-db N` — RMS dB threshold for pass/fail
- `--sample-rate N` — default 44100

### One-liner: rebuild + compare

```bash
cd /home/cody/Development/FastTrackStudio/FTS-Plugins && \
  cargo build --release --package eq-plugin && \
  cp target/release/libeq_plugin.so target/bundled/eq-plugin.clap && \
  cd /home/cody/Development/FastTrackStudio/fts-analyzer && \
  cargo build --release --package fts-analyzer-cli && \
  ./target/release/fts-analyzer-cli compare-eq \
    /home/cody/Development/FastTrackStudio/FTS-Plugins/target/bundled/eq-plugin.clap \
    --reference ./reference/pro-q4 \
    --out ./results/fts-eq-96k \
    --sample-rate 96000 \
    --tolerance-db 1.0
```

---

## DSP Code Structure

All EQ DSP code:

```
FTS-Plugins/crates/fts-eq/eq-dsp/src/
├── band.rs          # Band struct: holds filter state, dispatches update/tick
├── chain.rs         # EqChain: multi-band processing
├── coeff.rs         # Vicanek matched biquad coefficient calculation
├── filter_type.rs   # FilterType enum (Peak, LowShelf, Highpass, etc.)
├── response.rs      # Frequency response computation
├── section.rs       # Generic filter section trait
└── test_util.rs     # Test helpers
```

Plugin wrapper (CLAP interface, parameters): `eq-plugin/src/lib.rs`

### How band processing works

1. `Band::update(sample_rate)` — called when parameters change
2. Dispatches to the appropriate update function based on `FilterType`
3. Computes biquad coefficients via Vicanek matched design (`coeff.rs`)
4. For higher orders, cascades multiple 2nd-order sections with Butterworth pole placement
5. `Band::tick(sample)` — processes audio through all active sections in series

---

## Implementation Strategy

### Phase 1: Basic filters at mid frequencies

Focus on Bell, LowShelf, HighShelf, LowCut, HighCut at 200–5000 Hz where Nyquist effects are minimal:

1. Run `compare-eq --sample-rate 96000` first (cleaner reference, less cramping)
2. Fix gain curve shape for Bell and shelves
3. Fix Q/bandwidth mapping
4. Fix slope/order cascading for all supported slopes

### Phase 2: High-frequency behavior (Nyquist cramping)

Compare 48 kHz vs 96 kHz reference data to see Pro-Q 4's cramping:

1. Look at bell at 16k–22k Hz in 48k data — observe magnitude rolloff near Nyquist
2. Pro-Q 4 likely uses bilinear transform pre-warping or SVF topology
3. Target: **match** Pro-Q 4's cramping behavior, not eliminate it

### Phase 3: Multi-band interactions

Use the 19 multi-band scenarios to verify band stacking:

- Two bells at same frequency should sum gains
- Opposing shelves should approximate tilt
- Cut combos should create clean bandpass shapes

### Phase 4: Phase response

Every scenario includes phase data. Key areas:

- **Allpass**: should produce phase rotation with flat magnitude
- **Minimum phase**: all other filter types should be minimum-phase
- **Group delay**: consistency across filter types

---

## Re-capturing Reference Data

Only needed if the test matrix changes or Pro-Q 4 updates:

```bash
cd /home/cody/Development/FastTrackStudio/fts-analyzer && \
  cargo build --release --package fts-analyzer-cli && \
  ./target/release/fts-analyzer-cli capture-eq \
    '/home/cody/.clap/yabridge/FabFilter Pro-Q 4.clap' \
    --out ./reference/pro-q4 \
    --duration 2
```

Automatically captures at both 48 kHz and 96 kHz. ~5,567 scenarios × 2 = ~11,134 measurements. Takes about 2 minutes with 8 parallel Wine instances.
