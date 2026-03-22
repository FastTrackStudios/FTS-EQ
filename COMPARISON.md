# FTS-EQ vs Pro-Q 4 Comparison Report

- **Sample Rate:** 48000 Hz
- **Tolerance:** 0.50 dB RMS
- **Total:** 3478/5567 passed (62.5%)

## By Filter Type

| Filter | Pass | Fail | Total | Rate |
|--------|------|------|-------|------|
| bell | 712 | 504 | 1216 | 58.6% |
| low_shelf | 870 | 346 | 1216 | 71.5% |
| high_shelf | 870 | 346 | 1216 | 71.5% |
| low_cut | 139 | 165 | 304 | 45.7% |
| high_cut | 140 | 164 | 304 | 46.1% |
| notch | 139 | 165 | 304 | 45.7% |
| bandpass | 7 | 297 | 304 | 2.3% |
| tilt_shelf | 205 | 99 | 304 | 67.4% |
| flat_tilt | 76 | 0 | 76 | 100.0% |
| allpass | 304 | 0 | 304 | 100.0% |

## Changes from Previous (3326 → 3478, +152)

- **Shelf resonance**: Replaced Zoelzer BLT resonant shelf with Vicanek matched
  shelf + logarithmic Q compression (k=1.03). Low/high shelf: 420→346 each (-148 total)
- **Notch Q compensation**: Adjusted from N^0.5 to N^0.3 for cascade bandwidth
  compensation. Notch: 169→165 (-4)
- **Dead code**: Removed unused Zoelzer resonant shelf functions
