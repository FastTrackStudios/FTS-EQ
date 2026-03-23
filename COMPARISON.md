# FTS-EQ vs Pro-Q 4 Comparison Report

- **Sample Rate:** 48000 Hz
- **Tolerance:** 0.50 dB RMS
- **Total:** 3549/5567 passed (63.8%)

## By Filter Type

| Filter | Pass | Fail | Total | Rate |
|--------|------|------|-------|------|
| bell | 724 | 492 | 1216 | 59.5% |
| low_shelf | 889 | 327 | 1216 | 73.1% |
| high_shelf | 889 | 327 | 1216 | 73.1% |
| low_cut | 139 | 165 | 304 | 45.7% |
| high_cut | 140 | 164 | 304 | 46.1% |
| notch | 139 | 165 | 304 | 45.7% |
| bandpass | 7 | 297 | 304 | 2.3% |
| tilt_shelf | 215 | 89 | 304 | 70.7% |
| flat_tilt | 75 | 1 | 76 | 98.7% |
| allpass | 303 | 1 | 304 | 99.7% |

## Changes from Previous (3511 → 3549, +38)

- **Per-pole shelf gain distribution**: Distribute gain proportional to pole count
  instead of evenly per section. 2nd-order sections get 2x the gain of 1st-order
  sections. This allows proper resonance at high Q by concentrating gain in the
  resonant biquad sections. Low shelf: 877→889 (+12), high shelf: 877→889 (+12),
  tilt shelf: 209→215 (+6)
