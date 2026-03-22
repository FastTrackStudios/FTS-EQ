# FTS-EQ vs Pro-Q 4 Comparison Report

- **Sample Rate:** 48000 Hz
- **Tolerance:** 0.50 dB RMS
- **Total:** 3511/5567 passed (63.1%)

## By Filter Type

| Filter | Pass | Fail | Total | Rate |
|--------|------|------|-------|------|
| bell | 730 | 486 | 1216 | 60.0% |
| low_shelf | 877 | 339 | 1216 | 72.1% |
| high_shelf | 877 | 339 | 1216 | 72.1% |
| low_cut | 139 | 165 | 304 | 45.7% |
| high_cut | 140 | 164 | 304 | 46.1% |
| notch | 139 | 165 | 304 | 45.7% |
| bandpass | 7 | 297 | 304 | 2.3% |
| tilt_shelf | 209 | 95 | 304 | 68.8% |
| flat_tilt | 76 | 0 | 76 | 100.0% |
| allpass | 304 | 0 | 304 | 100.0% |

## Changes from Previous (3496 → 3511, +15)

- **Bell cascade Q compensation**: Apply 15% of theoretical bandwidth compensation
  √(2^(1/N)−1) for higher-order bell cascades. Widens each peak section slightly
  to counteract cascade bandwidth narrowing. Bell s5: 204→194 fails (-10),
  bell s8: 224→216 fails (-8)
