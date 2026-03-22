# FTS-EQ vs Pro-Q 4 Comparison Report

- **Sample Rate:** 48000 Hz
- **Tolerance:** 0.50 dB RMS
- **Total:** 3496/5567 passed (62.8%)

## By Filter Type

| Filter | Pass | Fail | Total | Rate |
|--------|------|------|-------|------|
| bell | 712 | 504 | 1216 | 58.6% |
| low_shelf | 877 | 339 | 1216 | 72.1% |
| high_shelf | 877 | 339 | 1216 | 72.1% |
| low_cut | 139 | 165 | 304 | 45.7% |
| high_cut | 140 | 164 | 304 | 46.1% |
| notch | 139 | 165 | 304 | 45.7% |
| bandpass | 7 | 297 | 304 | 2.3% |
| tilt_shelf | 209 | 95 | 304 | 68.8% |
| flat_tilt | 76 | 0 | 76 | 100.0% |
| allpass | 304 | 0 | 304 | 100.0% |

## Changes from Previous (3478 → 3496, +18)

- **Shelf low-Q scaling**: Reduced low-Q shelf blend exponent from 1.0 to 0.75,
  less aggressive Q reduction for Q<1 shelves. Low/high shelf: 346→339 each (-14 total)
- **1st-order shelf matching**: Moved matching point from fm=0.9 to fm=0.95,
  improving accuracy near Nyquist. Tilt shelf: 99→95 (-4)
