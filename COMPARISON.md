# FTS-EQ vs Pro-Q 4 Comparison Report

- **Sample Rate:** 48000 Hz
- **Tolerance:** 0.50 dB RMS
- **Total:** 3108/5567 passed (55.8%)
- **Avg RMS Error:** 3.033 dB
- **Worst RMS Error:** 122.888 dB

## By Filter Type

| Filter | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|--------|------|------|-------|------|---------|-----------|
| bell | 712 | 504 | 1216 | 58.6% | 0.555 | 3.933 |
| low_shelf | 756 | 460 | 1216 | 62.2% | 0.720 | 4.502 |
| high_shelf | 756 | 460 | 1216 | 62.2% | 0.720 | 4.502 |
| low_cut | 122 | 182 | 304 | 40.1% | 14.900 | 122.888 |
| high_cut | 137 | 167 | 304 | 45.1% | 3.643 | 10.899 |
| notch | 126 | 178 | 304 | 41.4% | 4.233 | 36.182 |
| bandpass | 7 | 297 | 304 | 2.3% | 22.596 | 80.146 |
| tilt_shelf | 172 | 132 | 304 | 56.6% | 0.681 | 4.121 |
| flat_tilt | 0 | 76 | 76 | 0.0% | 5.659 | 21.424 |
| allpass | 304 | 0 | 304 | 100.0% | 0.026 | 0.492 |
| multi | 16 | 3 | 19 | 84.2% | 1.101 | 8.748 |

## By Filter Type x Slope

| Filter | Slope | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|--------|-------|------|------|-------|------|---------|-----------|
| bell | 6 dB/oct | 259 | 45 | 304 | 85.2% | 0.234 | 2.538 |
| bell | 18 dB/oct | 259 | 45 | 304 | 85.2% | 0.234 | 2.538 |
| bell | 36 dB/oct | 106 | 198 | 304 | 34.9% | 0.717 | 3.521 |
| bell | 72 dB/oct | 88 | 216 | 304 | 28.9% | 1.036 | 3.933 |
| low_shelf | 6 dB/oct | 296 | 8 | 304 | 97.4% | 0.134 | 0.568 |
| low_shelf | 18 dB/oct | 151 | 153 | 304 | 49.7% | 1.046 | 4.502 |
| low_shelf | 36 dB/oct | 145 | 159 | 304 | 47.7% | 0.878 | 3.873 |
| low_shelf | 72 dB/oct | 164 | 140 | 304 | 53.9% | 0.820 | 4.130 |
| high_shelf | 6 dB/oct | 296 | 8 | 304 | 97.4% | 0.134 | 0.568 |
| high_shelf | 18 dB/oct | 151 | 153 | 304 | 49.7% | 1.046 | 4.502 |
| high_shelf | 36 dB/oct | 145 | 159 | 304 | 47.7% | 0.878 | 3.873 |
| high_shelf | 72 dB/oct | 164 | 140 | 304 | 53.9% | 0.820 | 4.130 |
| low_cut | 6 dB/oct | 76 | 0 | 76 | 100.0% | 0.000 | 0.000 |
| low_cut | 18 dB/oct | 24 | 52 | 76 | 31.6% | 9.532 | 36.931 |
| low_cut | 36 dB/oct | 14 | 62 | 76 | 18.4% | 17.423 | 58.024 |
| low_cut | 72 dB/oct | 8 | 68 | 76 | 10.5% | 32.646 | 122.888 |
| high_cut | 6 dB/oct | 76 | 0 | 76 | 100.0% | 0.000 | 0.000 |
| high_cut | 18 dB/oct | 61 | 15 | 76 | 80.3% | 0.273 | 3.024 |
| high_cut | 36 dB/oct | 0 | 76 | 76 | 0.0% | 7.658 | 10.899 |
| high_cut | 72 dB/oct | 0 | 76 | 76 | 0.0% | 6.640 | 10.542 |
| notch | 6 dB/oct | 54 | 22 | 76 | 71.1% | 0.959 | 8.144 |
| notch | 18 dB/oct | 54 | 22 | 76 | 71.1% | 0.959 | 8.144 |
| notch | 36 dB/oct | 12 | 64 | 76 | 15.8% | 7.406 | 27.731 |
| notch | 72 dB/oct | 6 | 70 | 76 | 7.9% | 7.608 | 36.182 |
| bandpass | 6 dB/oct | 7 | 69 | 76 | 9.2% | 0.538 | 0.644 |
| bandpass | 18 dB/oct | 0 | 76 | 76 | 0.0% | 9.694 | 28.452 |
| bandpass | 36 dB/oct | 0 | 76 | 76 | 0.0% | 33.800 | 80.146 |
| bandpass | 72 dB/oct | 0 | 76 | 76 | 0.0% | 46.354 | 79.333 |
| tilt_shelf | 6 dB/oct | 70 | 6 | 76 | 92.1% | 0.227 | 0.902 |
| tilt_shelf | 18 dB/oct | 42 | 34 | 76 | 55.3% | 0.637 | 1.952 |
| tilt_shelf | 36 dB/oct | 38 | 38 | 76 | 50.0% | 0.985 | 3.828 |
| tilt_shelf | 72 dB/oct | 22 | 54 | 76 | 28.9% | 0.876 | 4.121 |
| flat_tilt | 18 dB/oct | 0 | 76 | 76 | 0.0% | 5.659 | 21.424 |
| allpass | 6 dB/oct | 76 | 0 | 76 | 100.0% | 0.000 | 0.002 |
| allpass | 18 dB/oct | 76 | 0 | 76 | 100.0% | 0.023 | 0.222 |
| allpass | 36 dB/oct | 76 | 0 | 76 | 100.0% | 0.044 | 0.419 |
| allpass | 72 dB/oct | 76 | 0 | 76 | 100.0% | 0.037 | 0.492 |
| multi | 18 dB/oct | 16 | 1 | 17 | 94.1% | 0.261 | 0.958 |
| multi | 36 dB/oct | 0 | 1 | 1 | 0.0% | 8.748 | 8.748 |
| multi | 72 dB/oct | 0 | 1 | 1 | 0.0% | 7.742 | 7.742 |

## By Filter Type x Q

| Filter | Q | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|--------|---|------|------|-------|------|---------|-----------|
| bell | 0.5 | 168 | 136 | 304 | 55.3% | 0.572 | 3.162 |
| bell | 1.0 | 156 | 148 | 304 | 51.3% | 0.647 | 3.100 |
| bell | 4.0 | 182 | 122 | 304 | 59.9% | 0.544 | 2.759 |
| bell | 10.0 | 206 | 98 | 304 | 67.8% | 0.458 | 3.933 |
| low_shelf | 0.5 | 218 | 86 | 304 | 71.7% | 0.421 | 3.118 |
| low_shelf | 1.0 | 256 | 48 | 304 | 84.2% | 0.289 | 2.372 |
| low_shelf | 4.0 | 145 | 159 | 304 | 47.7% | 0.882 | 3.214 |
| low_shelf | 10.0 | 137 | 167 | 304 | 45.1% | 1.288 | 4.502 |
| high_shelf | 0.5 | 218 | 86 | 304 | 71.7% | 0.421 | 3.118 |
| high_shelf | 1.0 | 256 | 48 | 304 | 84.2% | 0.289 | 2.372 |
| high_shelf | 4.0 | 145 | 159 | 304 | 47.7% | 0.882 | 3.214 |
| high_shelf | 10.0 | 137 | 167 | 304 | 45.1% | 1.288 | 4.502 |
| low_cut | 0.5 | 27 | 49 | 76 | 35.5% | 18.976 | 122.888 |
| low_cut | 1.0 | 31 | 45 | 76 | 40.8% | 14.575 | 111.023 |
| low_cut | 4.0 | 30 | 46 | 76 | 39.5% | 13.071 | 100.843 |
| low_cut | 10.0 | 34 | 42 | 76 | 44.7% | 12.979 | 99.927 |
| high_cut | 0.5 | 33 | 43 | 76 | 43.4% | 4.345 | 10.899 |
| high_cut | 1.0 | 35 | 41 | 76 | 46.1% | 3.895 | 9.871 |
| high_cut | 4.0 | 35 | 41 | 76 | 46.1% | 3.278 | 8.516 |
| high_cut | 10.0 | 34 | 42 | 76 | 44.7% | 3.053 | 8.315 |
| notch | 0.5 | 19 | 57 | 76 | 25.0% | 6.155 | 27.731 |
| notch | 1.0 | 25 | 51 | 76 | 32.9% | 5.084 | 26.088 |
| notch | 4.0 | 40 | 36 | 76 | 52.6% | 3.237 | 36.182 |
| notch | 10.0 | 42 | 34 | 76 | 55.3% | 2.456 | 33.041 |
| bandpass | 0.5 | 1 | 75 | 76 | 1.3% | 26.938 | 80.146 |
| bandpass | 1.0 | 5 | 71 | 76 | 6.6% | 25.039 | 78.049 |
| bandpass | 4.0 | 1 | 75 | 76 | 1.3% | 19.936 | 75.056 |
| bandpass | 10.0 | 0 | 76 | 76 | 0.0% | 18.473 | 73.135 |
| tilt_shelf | 1.0 | 172 | 132 | 304 | 56.6% | 0.681 | 4.121 |
| flat_tilt | 1.0 | 0 | 76 | 76 | 0.0% | 5.659 | 21.424 |
| allpass | 0.5 | 76 | 0 | 76 | 100.0% | 0.018 | 0.274 |
| allpass | 1.0 | 76 | 0 | 76 | 100.0% | 0.009 | 0.156 |
| allpass | 4.0 | 76 | 0 | 76 | 100.0% | 0.031 | 0.384 |
| allpass | 10.0 | 76 | 0 | 76 | 100.0% | 0.047 | 0.492 |

## By Frequency

| Freq (Hz) | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|-----------|------|------|-------|------|---------|-----------|
| 20 | 264 | 28 | 292 | 90.4% | 2.283 | 80.146 |
| 50 | 259 | 33 | 292 | 88.7% | 2.264 | 72.032 |
| 100 | 255 | 37 | 292 | 87.3% | 2.224 | 77.407 |
| 200 | 249 | 43 | 292 | 85.3% | 2.154 | 79.333 |
| 500 | 230 | 62 | 292 | 78.8% | 2.020 | 75.103 |
| 1000 | 201 | 91 | 292 | 68.8% | 1.971 | 68.130 |
| 2000 | 193 | 99 | 292 | 66.1% | 1.957 | 62.495 |
| 5000 | 159 | 133 | 292 | 54.5% | 2.122 | 62.843 |
| 8000 | 146 | 146 | 292 | 50.0% | 2.276 | 57.850 |
| 10000 | 140 | 152 | 292 | 47.9% | 2.395 | 52.773 |
| 12000 | 143 | 149 | 292 | 49.0% | 2.585 | 49.397 |
| 14000 | 141 | 151 | 292 | 48.3% | 2.861 | 48.034 |
| 16000 | 136 | 156 | 292 | 46.6% | 3.176 | 49.290 |
| 17000 | 130 | 162 | 292 | 44.5% | 3.437 | 59.233 |
| 18000 | 122 | 170 | 292 | 41.8% | 3.773 | 74.961 |
| 19000 | 112 | 180 | 292 | 38.4% | 4.209 | 93.648 |
| 20000 | 90 | 202 | 292 | 30.8% | 4.781 | 115.804 |
| 21000 | 70 | 222 | 292 | 24.0% | 5.363 | 119.466 |
| 22000 | 52 | 240 | 292 | 17.8% | 5.901 | 122.888 |

## Failures (2459 total)

### Top 100 Worst Failures

| Scenario | RMS (dB) | Max (dB) | Filter | Freq | Q | Slope |
|----------|----------|----------|--------|------|---|-------|
| low_cut_22000hz_q0.5_s8 | 122.888 | 213.555 | low_cut | 22000 | 0.5 | 72 dB/oct |
| low_cut_21000hz_q0.5_s8 | 119.466 | 218.217 | low_cut | 21000 | 0.5 | 72 dB/oct |
| low_cut_20000hz_q0.5_s8 | 115.804 | 215.192 | low_cut | 20000 | 0.5 | 72 dB/oct |
| low_cut_22000hz_q1_s8 | 111.023 | 193.422 | low_cut | 22000 | 1 | 72 dB/oct |
| low_cut_21000hz_q1_s8 | 107.402 | 190.699 | low_cut | 21000 | 1 | 72 dB/oct |
| low_cut_20000hz_q1_s8 | 104.046 | 204.546 | low_cut | 20000 | 1 | 72 dB/oct |
| low_cut_22000hz_q4_s8 | 100.843 | 172.448 | low_cut | 22000 | 4 | 72 dB/oct |
| low_cut_22000hz_q10_s8 | 99.927 | 182.220 | low_cut | 22000 | 10 | 72 dB/oct |
| low_cut_21000hz_q4_s8 | 97.696 | 171.441 | low_cut | 21000 | 4 | 72 dB/oct |
| low_cut_21000hz_q10_s8 | 97.226 | 177.452 | low_cut | 21000 | 10 | 72 dB/oct |
| low_cut_20000hz_q4_s8 | 94.594 | 181.584 | low_cut | 20000 | 4 | 72 dB/oct |
| low_cut_20000hz_q10_s8 | 94.217 | 169.428 | low_cut | 20000 | 10 | 72 dB/oct |
| low_cut_19000hz_q0.5_s8 | 93.648 | 178.245 | low_cut | 19000 | 0.5 | 72 dB/oct |
| low_cut_19000hz_q1_s8 | 81.762 | 176.034 | low_cut | 19000 | 1 | 72 dB/oct |
| bandpass_20hz_q0.5_s5 | 80.146 | 132.113 | bandpass | 20 | 0.5 | 36 dB/oct |
| bandpass_200hz_q0.5_s8 | 79.333 | 127.155 | bandpass | 200 | 0.5 | 72 dB/oct |
| bandpass_20hz_q1_s5 | 78.049 | 123.398 | bandpass | 20 | 1 | 36 dB/oct |
| bandpass_100hz_q0.5_s8 | 77.407 | 124.955 | bandpass | 100 | 0.5 | 72 dB/oct |
| low_cut_19000hz_q4_s8 | 75.853 | 164.621 | low_cut | 19000 | 4 | 72 dB/oct |
| low_cut_19000hz_q10_s8 | 75.610 | 147.129 | low_cut | 19000 | 10 | 72 dB/oct |
| bandpass_500hz_q0.5_s8 | 75.103 | 122.075 | bandpass | 500 | 0.5 | 72 dB/oct |
| bandpass_20hz_q4_s5 | 75.056 | 116.225 | bandpass | 20 | 4 | 36 dB/oct |
| low_cut_18000hz_q0.5_s8 | 74.961 | 159.524 | low_cut | 18000 | 0.5 | 72 dB/oct |
| bandpass_20hz_q10_s5 | 73.135 | 107.008 | bandpass | 20 | 10 | 36 dB/oct |
| bandpass_50hz_q0.5_s8 | 72.032 | 111.804 | bandpass | 50 | 0.5 | 72 dB/oct |
| bandpass_50hz_q0.5_s5 | 70.170 | 133.655 | bandpass | 50 | 0.5 | 36 dB/oct |
| bandpass_200hz_q1_s8 | 69.337 | 117.213 | bandpass | 200 | 1 | 72 dB/oct |
| bandpass_50hz_q1_s5 | 68.369 | 128.370 | bandpass | 50 | 1 | 36 dB/oct |
| bandpass_1000hz_q0.5_s8 | 68.130 | 125.430 | bandpass | 1000 | 0.5 | 72 dB/oct |
| bandpass_100hz_q1_s8 | 66.503 | 104.244 | bandpass | 100 | 1 | 72 dB/oct |
| bandpass_500hz_q1_s8 | 66.398 | 112.320 | bandpass | 500 | 1 | 72 dB/oct |
| bandpass_50hz_q4_s5 | 65.858 | 120.479 | bandpass | 50 | 4 | 36 dB/oct |
| bandpass_50hz_q10_s5 | 64.588 | 113.414 | bandpass | 50 | 10 | 36 dB/oct |
| low_cut_18000hz_q1_s8 | 63.488 | 141.007 | low_cut | 18000 | 1 | 72 dB/oct |
| bandpass_5000hz_q0.5_s8 | 62.843 | 169.863 | bandpass | 5000 | 0.5 | 72 dB/oct |
| bandpass_2000hz_q0.5_s8 | 62.495 | 136.952 | bandpass | 2000 | 0.5 | 72 dB/oct |
| bandpass_100hz_q0.5_s5 | 62.205 | 134.802 | bandpass | 100 | 0.5 | 36 dB/oct |
| bandpass_20000hz_q0.5_s8 | 62.135 | 163.349 | bandpass | 20000 | 0.5 | 72 dB/oct |
| bandpass_21000hz_q0.5_s8 | 61.404 | 163.348 | bandpass | 21000 | 0.5 | 72 dB/oct |
| bandpass_20hz_q0.5_s8 | 61.142 | 97.778 | bandpass | 20 | 0.5 | 72 dB/oct |
| bandpass_50hz_q1_s8 | 60.854 | 94.927 | bandpass | 50 | 1 | 72 dB/oct |
| bandpass_22000hz_q0.5_s8 | 60.699 | 163.346 | bandpass | 22000 | 0.5 | 72 dB/oct |
| low_cut_18000hz_q10_s8 | 60.472 | 130.006 | low_cut | 18000 | 10 | 72 dB/oct |
| low_cut_18000hz_q4_s8 | 60.447 | 130.920 | low_cut | 18000 | 4 | 72 dB/oct |
| bandpass_100hz_q1_s5 | 60.255 | 131.352 | bandpass | 100 | 1 | 36 dB/oct |
| bandpass_1000hz_q1_s8 | 60.049 | 118.999 | bandpass | 1000 | 1 | 72 dB/oct |
| low_cut_17000hz_q0.5_s8 | 59.233 | 128.175 | low_cut | 17000 | 0.5 | 72 dB/oct |
| low_cut_22000hz_q0.5_s5 | 58.024 | 162.195 | low_cut | 22000 | 0.5 | 36 dB/oct |
| bandpass_100hz_q4_s5 | 57.866 | 123.124 | bandpass | 100 | 4 | 36 dB/oct |
| bandpass_8000hz_q0.5_s8 | 57.850 | 179.102 | bandpass | 8000 | 0.5 | 72 dB/oct |
| bandpass_100hz_q10_s5 | 56.999 | 115.092 | bandpass | 100 | 10 | 36 dB/oct |
| bandpass_19000hz_q0.5_s8 | 56.571 | 174.201 | bandpass | 19000 | 0.5 | 72 dB/oct |
| low_cut_21000hz_q0.5_s5 | 56.553 | 160.841 | low_cut | 21000 | 0.5 | 36 dB/oct |
| low_cut_20000hz_q0.5_s5 | 55.129 | 166.833 | low_cut | 20000 | 0.5 | 36 dB/oct |
| bandpass_20000hz_q1_s8 | 54.691 | 164.399 | bandpass | 20000 | 1 | 72 dB/oct |
| bandpass_200hz_q0.5_s5 | 54.612 | 141.605 | bandpass | 200 | 0.5 | 36 dB/oct |
| bandpass_21000hz_q1_s8 | 53.596 | 164.468 | bandpass | 21000 | 1 | 72 dB/oct |
| bandpass_18000hz_q0.5_s8 | 52.897 | 179.743 | bandpass | 18000 | 0.5 | 72 dB/oct |
| bandpass_10000hz_q0.5_s8 | 52.773 | 175.484 | bandpass | 10000 | 0.5 | 72 dB/oct |
| bandpass_22000hz_q1_s8 | 52.607 | 164.530 | bandpass | 22000 | 1 | 72 dB/oct |
| bandpass_200hz_q1_s5 | 52.310 | 129.793 | bandpass | 200 | 1 | 36 dB/oct |
| bandpass_2000hz_q1_s8 | 51.767 | 135.455 | bandpass | 2000 | 1 | 72 dB/oct |
| bandpass_200hz_q4_s8 | 50.748 | 93.594 | bandpass | 200 | 4 | 72 dB/oct |
| bandpass_17000hz_q0.5_s8 | 50.579 | 173.607 | bandpass | 17000 | 0.5 | 72 dB/oct |
| bandpass_500hz_q4_s8 | 50.546 | 90.637 | bandpass | 500 | 4 | 72 dB/oct |
| bandpass_19000hz_q1_s8 | 50.075 | 174.807 | bandpass | 19000 | 1 | 72 dB/oct |
| bandpass_20hz_q1_s8 | 49.925 | 93.593 | bandpass | 20 | 1 | 72 dB/oct |
| bandpass_200hz_q4_s5 | 49.657 | 120.166 | bandpass | 200 | 4 | 36 dB/oct |
| bandpass_12000hz_q0.5_s8 | 49.397 | 189.239 | bandpass | 12000 | 0.5 | 72 dB/oct |
| bandpass_16000hz_q0.5_s8 | 49.290 | 191.884 | bandpass | 16000 | 0.5 | 72 dB/oct |
| bandpass_200hz_q10_s5 | 48.785 | 115.108 | bandpass | 200 | 10 | 36 dB/oct |
| bandpass_10000hz_q1_s8 | 48.683 | 166.272 | bandpass | 10000 | 1 | 72 dB/oct |
| low_cut_19000hz_q0.5_s5 | 48.563 | 151.851 | low_cut | 19000 | 0.5 | 36 dB/oct |
| bandpass_8000hz_q1_s8 | 48.440 | 155.148 | bandpass | 8000 | 1 | 72 dB/oct |
| bandpass_14000hz_q0.5_s8 | 48.034 | 177.959 | bandpass | 14000 | 0.5 | 72 dB/oct |
| bandpass_12000hz_q1_s8 | 47.869 | 187.015 | bandpass | 12000 | 1 | 72 dB/oct |
| low_cut_17000hz_q1_s8 | 47.676 | 112.496 | low_cut | 17000 | 1 | 72 dB/oct |
| low_cut_22000hz_q1_s5 | 47.600 | 149.723 | low_cut | 22000 | 1 | 36 dB/oct |
| low_cut_17000hz_q4_s8 | 47.394 | 104.528 | low_cut | 17000 | 4 | 72 dB/oct |
| low_cut_17000hz_q10_s8 | 47.265 | 107.469 | low_cut | 17000 | 10 | 72 dB/oct |
| bandpass_18000hz_q1_s8 | 47.090 | 170.395 | bandpass | 18000 | 1 | 72 dB/oct |
| bandpass_1000hz_q4_s8 | 47.029 | 100.790 | bandpass | 1000 | 4 | 72 dB/oct |
| bandpass_100hz_q4_s8 | 46.846 | 86.631 | bandpass | 100 | 4 | 72 dB/oct |
| bandpass_5000hz_q1_s8 | 46.403 | 143.558 | bandpass | 5000 | 1 | 72 dB/oct |
| low_cut_16000hz_q0.5_s8 | 46.311 | 118.917 | low_cut | 16000 | 0.5 | 72 dB/oct |
| bandpass_500hz_q0.5_s5 | 46.133 | 140.134 | bandpass | 500 | 0.5 | 36 dB/oct |
| low_cut_21000hz_q1_s5 | 45.775 | 139.327 | low_cut | 21000 | 1 | 36 dB/oct |
| bandpass_17000hz_q1_s8 | 45.550 | 173.878 | bandpass | 17000 | 1 | 72 dB/oct |
| bandpass_14000hz_q1_s8 | 45.185 | 169.891 | bandpass | 14000 | 1 | 72 dB/oct |
| bandpass_16000hz_q1_s8 | 44.866 | 183.063 | bandpass | 16000 | 1 | 72 dB/oct |
| low_cut_20000hz_q1_s5 | 44.276 | 137.951 | low_cut | 20000 | 1 | 36 dB/oct |
| low_cut_18000hz_q0.5_s5 | 43.037 | 141.440 | low_cut | 18000 | 0.5 | 36 dB/oct |
| bandpass_500hz_q1_s5 | 42.913 | 127.333 | bandpass | 500 | 1 | 36 dB/oct |
| bandpass_2000hz_q4_s8 | 42.400 | 113.530 | bandpass | 2000 | 4 | 72 dB/oct |
| bandpass_500hz_q10_s8 | 41.630 | 78.614 | bandpass | 500 | 10 | 72 dB/oct |
| bandpass_1000hz_q0.5_s5 | 40.658 | 145.332 | bandpass | 1000 | 0.5 | 36 dB/oct |
| bandpass_1000hz_q10_s8 | 40.389 | 85.458 | bandpass | 1000 | 10 | 72 dB/oct |
| bandpass_20000hz_q4_s8 | 39.931 | 167.842 | bandpass | 20000 | 4 | 72 dB/oct |
| bandpass_200hz_q10_s8 | 39.843 | 74.095 | bandpass | 200 | 10 | 72 dB/oct |
| bandpass_50hz_q4_s8 | 39.754 | 79.215 | bandpass | 50 | 4 | 72 dB/oct |

(2359 more failures not shown)

## Passing (3108 total)

### Closest to Threshold (top 50)

| Scenario | RMS (dB) | Max (dB) | Filter | Freq | Q | Slope |
|----------|----------|----------|--------|------|---|-------|
| bandpass_8000hz_q4_s0 | 0.497 | 4.689 | bandpass | 8000 | 4 | 6 dB/oct |
| low_shelf_19000hz_+6db_q0.5_s2 | 0.497 | 0.771 | low_shelf | 19000 | 0.5 | 18 dB/oct |
| high_shelf_19000hz_-6db_q0.5_s2 | 0.497 | 0.771 | high_shelf | 19000 | 0.5 | 18 dB/oct |
| low_cut_500hz_q10_s5 | 0.496 | 12.258 | low_cut | 500 | 10 | 36 dB/oct |
| low_shelf_18000hz_+12db_q1_s5 | 0.496 | 0.817 | low_shelf | 18000 | 1 | 36 dB/oct |
| high_shelf_18000hz_-12db_q1_s5 | 0.496 | 0.817 | high_shelf | 18000 | 1 | 36 dB/oct |
| low_cut_200hz_q10_s8 | 0.495 | 10.593 | low_cut | 200 | 10 | 72 dB/oct |
| high_shelf_200hz_-12db_q10_s2 | 0.495 | 9.055 | high_shelf | 200 | 10 | 18 dB/oct |
| low_shelf_200hz_+12db_q10_s2 | 0.495 | 9.055 | low_shelf | 200 | 10 | 18 dB/oct |
| high_shelf_5000hz_+12db_q0.5_s2 | 0.494 | 0.821 | high_shelf | 5000 | 0.5 | 18 dB/oct |
| low_shelf_5000hz_-12db_q0.5_s2 | 0.494 | 0.821 | low_shelf | 5000 | 0.5 | 18 dB/oct |
| low_shelf_5000hz_+12db_q0.5_s2 | 0.494 | 0.818 | low_shelf | 5000 | 0.5 | 18 dB/oct |
| high_shelf_5000hz_-12db_q0.5_s2 | 0.494 | 0.818 | high_shelf | 5000 | 0.5 | 18 dB/oct |
| low_shelf_21000hz_+6db_q0.5_s2 | 0.493 | 0.848 | low_shelf | 21000 | 0.5 | 18 dB/oct |
| high_shelf_21000hz_-6db_q0.5_s2 | 0.493 | 0.848 | high_shelf | 21000 | 0.5 | 18 dB/oct |
| bell_12000hz_-12db_q0.5_s0 | 0.492 | 1.127 | bell | 12000 | 0.5 | 6 dB/oct |
| bell_12000hz_-12db_q0.5_s2 | 0.492 | 1.127 | bell | 12000 | 0.5 | 18 dB/oct |
| allpass_200hz_q10_s8 | 0.492 | 22.020 | allpass | 200 | 10 | 72 dB/oct |
| bell_16000hz_-6db_q10_s5 | 0.492 | 1.232 | bell | 16000 | 10 | 36 dB/oct |
| bell_8000hz_-6db_q10_s8 | 0.491 | 1.994 | bell | 8000 | 10 | 72 dB/oct |
| bell_16000hz_+6db_q10_s5 | 0.491 | 1.235 | bell | 16000 | 10 | 36 dB/oct |
| bandpass_14000hz_q1_s0 | 0.491 | 5.471 | bandpass | 14000 | 1 | 6 dB/oct |
| bell_20000hz_-6db_q4_s0 | 0.491 | 1.641 | bell | 20000 | 4 | 6 dB/oct |
| bell_20000hz_-6db_q4_s2 | 0.491 | 1.641 | bell | 20000 | 4 | 18 dB/oct |
| bell_8000hz_+6db_q10_s8 | 0.490 | 1.992 | bell | 8000 | 10 | 72 dB/oct |
| low_shelf_14000hz_+6db_q0.5_s5 | 0.488 | 0.836 | low_shelf | 14000 | 0.5 | 36 dB/oct |
| high_shelf_14000hz_-6db_q0.5_s5 | 0.488 | 0.836 | high_shelf | 14000 | 0.5 | 36 dB/oct |
| low_cut_100hz_q1_s5 | 0.487 | 7.935 | low_cut | 100 | 1 | 36 dB/oct |
| high_shelf_500hz_-12db_q4_s2 | 0.484 | 4.781 | high_shelf | 500 | 4 | 18 dB/oct |
| low_shelf_500hz_+12db_q4_s2 | 0.484 | 4.781 | low_shelf | 500 | 4 | 18 dB/oct |
| tilt_shelf_17000hz_-6db_q1_s2 | 0.483 | 0.962 | tilt_shelf | 17000 | 1 | 18 dB/oct |
| tilt_shelf_17000hz_+6db_q1_s2 | 0.483 | 0.962 | tilt_shelf | 17000 | 1 | 18 dB/oct |
| high_shelf_20000hz_+6db_q1_s2 | 0.482 | 0.835 | high_shelf | 20000 | 1 | 18 dB/oct |
| low_shelf_20000hz_-6db_q1_s2 | 0.482 | 0.835 | low_shelf | 20000 | 1 | 18 dB/oct |
| high_shelf_19000hz_+6db_q1_s5 | 0.479 | 0.791 | high_shelf | 19000 | 1 | 36 dB/oct |
| low_shelf_19000hz_-6db_q1_s5 | 0.479 | 0.791 | low_shelf | 19000 | 1 | 36 dB/oct |
| bell_2000hz_-12db_q10_s8 | 0.477 | 3.973 | bell | 2000 | 10 | 72 dB/oct |
| bell_200hz_-12db_q1_s8 | 0.476 | 3.833 | bell | 200 | 1 | 72 dB/oct |
| bell_2000hz_+12db_q10_s8 | 0.476 | 3.868 | bell | 2000 | 10 | 72 dB/oct |
| bell_200hz_+12db_q1_s8 | 0.475 | 3.849 | bell | 200 | 1 | 72 dB/oct |
| notch_50hz_q1_s5 | 0.475 | 9.169 | notch | 50 | 1 | 36 dB/oct |
| low_cut_1000hz_q10_s2 | 0.475 | 15.407 | low_cut | 1000 | 10 | 18 dB/oct |
| low_cut_50hz_q1_s8 | 0.473 | 9.992 | low_cut | 50 | 1 | 72 dB/oct |
| bell_200hz_-12db_q0.5_s5 | 0.472 | 2.175 | bell | 200 | 0.5 | 36 dB/oct |
| bell_200hz_+12db_q0.5_s5 | 0.472 | 2.176 | bell | 200 | 0.5 | 36 dB/oct |
| bell_22000hz_+6db_q0.5_s0 | 0.472 | 0.774 | bell | 22000 | 0.5 | 6 dB/oct |
| bell_22000hz_+6db_q0.5_s2 | 0.472 | 0.774 | bell | 22000 | 0.5 | 18 dB/oct |
| notch_20hz_q1_s8 | 0.472 | 12.653 | notch | 20 | 1 | 72 dB/oct |
| notch_500hz_q10_s5 | 0.472 | 9.765 | notch | 500 | 10 | 36 dB/oct |
| bell_100hz_-12db_q0.5_s8 | 0.472 | 3.882 | bell | 100 | 0.5 | 72 dB/oct |

