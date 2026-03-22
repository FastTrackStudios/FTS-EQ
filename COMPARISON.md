# FTS-EQ vs Pro-Q 4 Comparison Report

- **Sample Rate:** 48000 Hz
- **Tolerance:** 0.50 dB RMS
- **Total:** 3326/5567 passed (59.7%)
- **Avg RMS Error:** 1.840 dB
- **Worst RMS Error:** 78.663 dB

## By Filter Type

| Filter | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|--------|------|------|-------|------|---------|-----------|
| bell | 712 | 504 | 1216 | 58.6% | 0.555 | 3.933 |
| low_shelf | 796 | 420 | 1216 | 65.5% | 0.682 | 4.502 |
| high_shelf | 796 | 420 | 1216 | 65.5% | 0.682 | 4.502 |
| low_cut | 139 | 165 | 304 | 45.7% | 3.208 | 24.463 |
| high_cut | 140 | 164 | 304 | 46.1% | 3.221 | 10.844 |
| notch | 135 | 169 | 304 | 44.4% | 4.675 | 39.455 |
| bandpass | 7 | 297 | 304 | 2.3% | 14.100 | 78.663 |
| tilt_shelf | 205 | 99 | 304 | 67.4% | 0.698 | 12.883 |
| flat_tilt | 76 | 0 | 76 | 100.0% | 0.203 | 0.494 |
| allpass | 304 | 0 | 304 | 100.0% | 0.026 | 0.492 |
| multi | 16 | 3 | 19 | 84.2% | 0.593 | 5.519 |

## By Filter Type x Slope

| Filter | Slope | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|--------|-------|------|------|-------|------|---------|-----------|
| bell | 6 dB/oct | 259 | 45 | 304 | 85.2% | 0.234 | 2.538 |
| bell | 18 dB/oct | 259 | 45 | 304 | 85.2% | 0.234 | 2.538 |
| bell | 36 dB/oct | 106 | 198 | 304 | 34.9% | 0.717 | 3.521 |
| bell | 72 dB/oct | 88 | 216 | 304 | 28.9% | 1.036 | 3.933 |
| low_shelf | 6 dB/oct | 296 | 8 | 304 | 97.4% | 0.134 | 0.568 |
| low_shelf | 18 dB/oct | 169 | 135 | 304 | 55.6% | 0.974 | 4.502 |
| low_shelf | 36 dB/oct | 166 | 138 | 304 | 54.6% | 0.817 | 3.873 |
| low_shelf | 72 dB/oct | 165 | 139 | 304 | 54.3% | 0.803 | 4.130 |
| high_shelf | 6 dB/oct | 296 | 8 | 304 | 97.4% | 0.134 | 0.568 |
| high_shelf | 18 dB/oct | 169 | 135 | 304 | 55.6% | 0.974 | 4.502 |
| high_shelf | 36 dB/oct | 166 | 138 | 304 | 54.6% | 0.817 | 3.873 |
| high_shelf | 72 dB/oct | 165 | 139 | 304 | 54.3% | 0.803 | 4.130 |
| low_cut | 6 dB/oct | 76 | 0 | 76 | 100.0% | 0.000 | 0.000 |
| low_cut | 18 dB/oct | 32 | 44 | 76 | 42.1% | 3.896 | 16.335 |
| low_cut | 36 dB/oct | 18 | 58 | 76 | 23.7% | 4.977 | 24.463 |
| low_cut | 72 dB/oct | 13 | 63 | 76 | 17.1% | 3.958 | 14.349 |
| high_cut | 6 dB/oct | 76 | 0 | 76 | 100.0% | 0.000 | 0.000 |
| high_cut | 18 dB/oct | 61 | 15 | 76 | 80.3% | 0.273 | 3.024 |
| high_cut | 36 dB/oct | 0 | 76 | 76 | 0.0% | 6.461 | 8.875 |
| high_cut | 72 dB/oct | 3 | 73 | 76 | 3.9% | 6.150 | 10.844 |
| notch | 6 dB/oct | 54 | 22 | 76 | 71.1% | 0.959 | 8.144 |
| notch | 18 dB/oct | 54 | 22 | 76 | 71.1% | 0.959 | 8.144 |
| notch | 36 dB/oct | 19 | 57 | 76 | 25.0% | 3.734 | 19.973 |
| notch | 72 dB/oct | 8 | 68 | 76 | 10.5% | 13.050 | 39.455 |
| bandpass | 6 dB/oct | 7 | 69 | 76 | 9.2% | 0.538 | 0.644 |
| bandpass | 18 dB/oct | 0 | 76 | 76 | 0.0% | 2.975 | 22.870 |
| bandpass | 36 dB/oct | 0 | 76 | 76 | 0.0% | 20.335 | 70.934 |
| bandpass | 72 dB/oct | 0 | 76 | 76 | 0.0% | 32.551 | 78.663 |
| tilt_shelf | 6 dB/oct | 70 | 6 | 76 | 92.1% | 0.227 | 0.902 |
| tilt_shelf | 18 dB/oct | 48 | 28 | 76 | 63.2% | 0.706 | 5.506 |
| tilt_shelf | 36 dB/oct | 45 | 31 | 76 | 59.2% | 1.131 | 12.883 |
| tilt_shelf | 72 dB/oct | 42 | 34 | 76 | 55.3% | 0.728 | 5.140 |
| flat_tilt | 18 dB/oct | 76 | 0 | 76 | 100.0% | 0.203 | 0.494 |
| allpass | 6 dB/oct | 76 | 0 | 76 | 100.0% | 0.000 | 0.002 |
| allpass | 18 dB/oct | 76 | 0 | 76 | 100.0% | 0.023 | 0.222 |
| allpass | 36 dB/oct | 76 | 0 | 76 | 100.0% | 0.044 | 0.419 |
| allpass | 72 dB/oct | 76 | 0 | 76 | 100.0% | 0.037 | 0.492 |
| multi | 18 dB/oct | 16 | 1 | 17 | 94.1% | 0.261 | 0.958 |
| multi | 36 dB/oct | 0 | 1 | 1 | 0.0% | 5.519 | 5.519 |
| multi | 72 dB/oct | 0 | 1 | 1 | 0.0% | 1.305 | 1.305 |

## By Filter Type x Q

| Filter | Q | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|--------|---|------|------|-------|------|---------|-----------|
| bell | 0.5 | 168 | 136 | 304 | 55.3% | 0.572 | 3.162 |
| bell | 1.0 | 156 | 148 | 304 | 51.3% | 0.647 | 3.100 |
| bell | 4.0 | 182 | 122 | 304 | 59.9% | 0.544 | 2.759 |
| bell | 10.0 | 206 | 98 | 304 | 67.8% | 0.458 | 3.933 |
| low_shelf | 0.5 | 258 | 46 | 304 | 84.9% | 0.270 | 2.973 |
| low_shelf | 1.0 | 256 | 48 | 304 | 84.2% | 0.289 | 2.372 |
| low_shelf | 4.0 | 145 | 159 | 304 | 47.7% | 0.882 | 3.214 |
| low_shelf | 10.0 | 137 | 167 | 304 | 45.1% | 1.288 | 4.502 |
| high_shelf | 0.5 | 258 | 46 | 304 | 84.9% | 0.270 | 2.973 |
| high_shelf | 1.0 | 256 | 48 | 304 | 84.2% | 0.289 | 2.372 |
| high_shelf | 4.0 | 145 | 159 | 304 | 47.7% | 0.882 | 3.214 |
| high_shelf | 10.0 | 137 | 167 | 304 | 45.1% | 1.288 | 4.502 |
| low_cut | 0.5 | 33 | 43 | 76 | 43.4% | 2.533 | 12.294 |
| low_cut | 1.0 | 44 | 32 | 76 | 57.9% | 1.523 | 9.017 |
| low_cut | 4.0 | 32 | 44 | 76 | 42.1% | 3.983 | 20.436 |
| low_cut | 10.0 | 30 | 46 | 76 | 39.5% | 4.792 | 24.463 |
| high_cut | 0.5 | 33 | 43 | 76 | 43.4% | 2.832 | 8.875 |
| high_cut | 1.0 | 38 | 38 | 76 | 50.0% | 2.518 | 8.243 |
| high_cut | 4.0 | 35 | 41 | 76 | 46.1% | 3.454 | 9.160 |
| high_cut | 10.0 | 34 | 42 | 76 | 44.7% | 4.080 | 10.844 |
| notch | 0.5 | 21 | 55 | 76 | 27.6% | 6.660 | 30.805 |
| notch | 1.0 | 27 | 49 | 76 | 35.5% | 5.711 | 31.179 |
| notch | 4.0 | 42 | 34 | 76 | 55.3% | 3.767 | 39.455 |
| notch | 10.0 | 45 | 31 | 76 | 59.2% | 2.563 | 30.450 |
| bandpass | 0.5 | 1 | 75 | 76 | 1.3% | 15.294 | 78.663 |
| bandpass | 1.0 | 5 | 71 | 76 | 6.6% | 13.134 | 70.371 |
| bandpass | 4.0 | 1 | 75 | 76 | 1.3% | 12.890 | 69.780 |
| bandpass | 10.0 | 0 | 76 | 76 | 0.0% | 15.082 | 69.498 |
| tilt_shelf | 1.0 | 205 | 99 | 304 | 67.4% | 0.698 | 12.883 |
| flat_tilt | 1.0 | 76 | 0 | 76 | 100.0% | 0.203 | 0.494 |
| allpass | 0.5 | 76 | 0 | 76 | 100.0% | 0.018 | 0.274 |
| allpass | 1.0 | 76 | 0 | 76 | 100.0% | 0.009 | 0.156 |
| allpass | 4.0 | 76 | 0 | 76 | 100.0% | 0.031 | 0.384 |
| allpass | 10.0 | 76 | 0 | 76 | 100.0% | 0.047 | 0.492 |

## By Frequency

| Freq (Hz) | Pass | Fail | Total | Rate | Avg RMS | Worst RMS |
|-----------|------|------|-------|------|---------|-----------|
| 20 | 268 | 24 | 292 | 91.8% | 1.888 | 70.934 |
| 50 | 265 | 27 | 292 | 90.8% | 1.818 | 71.921 |
| 100 | 264 | 28 | 292 | 90.4% | 1.764 | 77.208 |
| 200 | 256 | 36 | 292 | 87.7% | 1.679 | 78.663 |
| 500 | 236 | 56 | 292 | 80.8% | 1.494 | 70.188 |
| 1000 | 211 | 81 | 292 | 72.3% | 1.311 | 51.078 |
| 2000 | 207 | 85 | 292 | 70.9% | 1.185 | 37.062 |
| 5000 | 168 | 124 | 292 | 57.5% | 1.338 | 31.339 |
| 8000 | 157 | 135 | 292 | 53.8% | 1.489 | 34.399 |
| 10000 | 150 | 142 | 292 | 51.4% | 1.570 | 36.048 |
| 12000 | 155 | 137 | 292 | 53.1% | 1.637 | 35.702 |
| 14000 | 156 | 136 | 292 | 53.4% | 1.713 | 35.098 |
| 16000 | 156 | 136 | 292 | 53.4% | 1.817 | 35.993 |
| 17000 | 152 | 140 | 292 | 52.1% | 1.894 | 36.518 |
| 18000 | 142 | 150 | 292 | 48.6% | 1.989 | 36.847 |
| 19000 | 126 | 166 | 292 | 43.2% | 2.121 | 37.376 |
| 20000 | 102 | 190 | 292 | 34.9% | 2.343 | 38.328 |
| 21000 | 80 | 212 | 292 | 27.4% | 2.779 | 40.255 |
| 22000 | 59 | 233 | 292 | 20.2% | 3.210 | 42.664 |

## Failures (2241 total)

### Top 100 Worst Failures

| Scenario | RMS (dB) | Max (dB) | Filter | Freq | Q | Slope |
|----------|----------|----------|--------|------|---|-------|
| bandpass_200hz_q0.5_s8 | 78.663 | 117.431 | bandpass | 200 | 0.5 | 72 dB/oct |
| bandpass_100hz_q0.5_s8 | 77.208 | 112.456 | bandpass | 100 | 0.5 | 72 dB/oct |
| bandpass_50hz_q0.5_s8 | 71.921 | 112.357 | bandpass | 50 | 0.5 | 72 dB/oct |
| bandpass_20hz_q0.5_s5 | 70.934 | 102.812 | bandpass | 20 | 0.5 | 36 dB/oct |
| bandpass_20hz_q1_s5 | 70.371 | 110.575 | bandpass | 20 | 1 | 36 dB/oct |
| bandpass_500hz_q0.5_s8 | 70.188 | 125.768 | bandpass | 500 | 0.5 | 72 dB/oct |
| bandpass_20hz_q4_s5 | 69.780 | 105.762 | bandpass | 20 | 4 | 36 dB/oct |
| bandpass_20hz_q10_s5 | 69.498 | 104.794 | bandpass | 20 | 10 | 36 dB/oct |
| bandpass_200hz_q1_s8 | 68.698 | 110.409 | bandpass | 200 | 1 | 72 dB/oct |
| bandpass_100hz_q1_s8 | 66.557 | 106.613 | bandpass | 100 | 1 | 72 dB/oct |
| bandpass_500hz_q1_s8 | 62.306 | 105.457 | bandpass | 500 | 1 | 72 dB/oct |
| bandpass_20hz_q0.5_s8 | 61.209 | 100.039 | bandpass | 20 | 0.5 | 72 dB/oct |
| bandpass_50hz_q1_s8 | 60.710 | 100.306 | bandpass | 50 | 1 | 72 dB/oct |
| bandpass_50hz_q0.5_s5 | 56.959 | 86.099 | bandpass | 50 | 0.5 | 36 dB/oct |
| bandpass_50hz_q1_s5 | 55.895 | 85.908 | bandpass | 50 | 1 | 36 dB/oct |
| bandpass_50hz_q4_s5 | 55.011 | 86.018 | bandpass | 50 | 4 | 36 dB/oct |
| bandpass_50hz_q10_s5 | 54.853 | 85.506 | bandpass | 50 | 10 | 36 dB/oct |
| bandpass_1000hz_q0.5_s8 | 51.078 | 89.003 | bandpass | 1000 | 0.5 | 72 dB/oct |
| bandpass_200hz_q4_s8 | 50.651 | 91.040 | bandpass | 200 | 4 | 72 dB/oct |
| bandpass_20hz_q1_s8 | 49.999 | 95.379 | bandpass | 20 | 1 | 72 dB/oct |
| bandpass_500hz_q4_s8 | 48.296 | 92.269 | bandpass | 500 | 4 | 72 dB/oct |
| bandpass_100hz_q0.5_s5 | 47.125 | 74.042 | bandpass | 100 | 0.5 | 36 dB/oct |
| bandpass_100hz_q4_s8 | 46.728 | 81.944 | bandpass | 100 | 4 | 72 dB/oct |
| bandpass_100hz_q1_s5 | 45.517 | 73.923 | bandpass | 100 | 1 | 36 dB/oct |
| bandpass_1000hz_q1_s8 | 44.451 | 82.547 | bandpass | 1000 | 1 | 72 dB/oct |
| bandpass_100hz_q4_s5 | 44.104 | 73.767 | bandpass | 100 | 4 | 36 dB/oct |
| bandpass_100hz_q10_s5 | 43.839 | 73.865 | bandpass | 100 | 10 | 36 dB/oct |
| bandpass_22000hz_q10_s8 | 42.664 | 95.111 | bandpass | 22000 | 10 | 72 dB/oct |
| bandpass_21000hz_q10_s8 | 40.255 | 90.991 | bandpass | 21000 | 10 | 72 dB/oct |
| bandpass_500hz_q10_s8 | 39.957 | 74.232 | bandpass | 500 | 10 | 72 dB/oct |
| bandpass_50hz_q4_s8 | 39.820 | 74.881 | bandpass | 50 | 4 | 72 dB/oct |
| bandpass_200hz_q10_s8 | 39.467 | 81.974 | bandpass | 200 | 10 | 72 dB/oct |
| notch_22000hz_q4_s8 | 39.455 | 162.753 | notch | 22000 | 4 | 72 dB/oct |
| bandpass_20000hz_q10_s8 | 38.328 | 87.640 | bandpass | 20000 | 10 | 72 dB/oct |
| bandpass_200hz_q0.5_s5 | 38.115 | 62.257 | bandpass | 200 | 0.5 | 36 dB/oct |
| bandpass_19000hz_q10_s8 | 37.376 | 87.221 | bandpass | 19000 | 10 | 72 dB/oct |
| bandpass_2000hz_q0.5_s8 | 37.062 | 75.943 | bandpass | 2000 | 0.5 | 72 dB/oct |
| bandpass_18000hz_q10_s8 | 36.847 | 86.789 | bandpass | 18000 | 10 | 72 dB/oct |
| bandpass_17000hz_q10_s8 | 36.518 | 86.674 | bandpass | 17000 | 10 | 72 dB/oct |
| bandpass_10000hz_q10_s8 | 36.048 | 81.723 | bandpass | 10000 | 10 | 72 dB/oct |
| bandpass_16000hz_q10_s8 | 35.993 | 86.381 | bandpass | 16000 | 10 | 72 dB/oct |
| bandpass_200hz_q1_s5 | 35.816 | 62.001 | bandpass | 200 | 1 | 36 dB/oct |
| bandpass_12000hz_q10_s8 | 35.702 | 82.918 | bandpass | 12000 | 10 | 72 dB/oct |
| bandpass_1000hz_q4_s8 | 35.602 | 70.475 | bandpass | 1000 | 4 | 72 dB/oct |
| bandpass_100hz_q10_s8 | 35.330 | 76.847 | bandpass | 100 | 10 | 72 dB/oct |
| bandpass_14000hz_q10_s8 | 35.098 | 84.245 | bandpass | 14000 | 10 | 72 dB/oct |
| bandpass_8000hz_q10_s8 | 34.399 | 84.730 | bandpass | 8000 | 10 | 72 dB/oct |
| bandpass_200hz_q4_s5 | 33.719 | 61.792 | bandpass | 200 | 4 | 36 dB/oct |
| bandpass_200hz_q10_s5 | 33.359 | 61.742 | bandpass | 200 | 10 | 36 dB/oct |
| bandpass_1000hz_q10_s8 | 31.522 | 77.670 | bandpass | 1000 | 10 | 72 dB/oct |
| bandpass_5000hz_q0.5_s8 | 31.339 | 94.542 | bandpass | 5000 | 0.5 | 72 dB/oct |
| notch_22000hz_q1_s8 | 31.179 | 108.681 | notch | 22000 | 1 | 72 dB/oct |
| notch_14000hz_q0.5_s8 | 30.805 | 51.925 | notch | 14000 | 0.5 | 72 dB/oct |
| notch_21000hz_q4_s8 | 30.780 | 139.327 | notch | 21000 | 4 | 72 dB/oct |
| notch_12000hz_q0.5_s8 | 30.627 | 59.772 | notch | 12000 | 0.5 | 72 dB/oct |
| bandpass_22000hz_q4_s8 | 30.497 | 72.495 | bandpass | 22000 | 4 | 72 dB/oct |
| notch_22000hz_q10_s8 | 30.450 | 155.243 | notch | 22000 | 10 | 72 dB/oct |
| bandpass_20hz_q4_s8 | 30.313 | 67.984 | bandpass | 20 | 4 | 72 dB/oct |
| bandpass_5000hz_q10_s8 | 29.609 | 75.360 | bandpass | 5000 | 10 | 72 dB/oct |
| notch_16000hz_q0.5_s8 | 29.472 | 56.669 | notch | 16000 | 0.5 | 72 dB/oct |
| notch_10000hz_q0.5_s8 | 28.865 | 56.063 | notch | 10000 | 0.5 | 72 dB/oct |
| bandpass_50hz_q10_s8 | 28.406 | 61.970 | bandpass | 50 | 10 | 72 dB/oct |
| notch_17000hz_q0.5_s8 | 28.185 | 61.596 | notch | 17000 | 0.5 | 72 dB/oct |
| bandpass_21000hz_q4_s8 | 28.042 | 69.981 | bandpass | 21000 | 4 | 72 dB/oct |
| bandpass_500hz_q0.5_s5 | 27.727 | 47.075 | bandpass | 500 | 0.5 | 36 dB/oct |
| notch_21000hz_q1_s8 | 26.909 | 90.275 | notch | 21000 | 1 | 72 dB/oct |
| bandpass_20000hz_q4_s8 | 26.758 | 68.437 | bandpass | 20000 | 4 | 72 dB/oct |
| notch_17000hz_q1_s8 | 26.588 | 68.909 | notch | 17000 | 1 | 72 dB/oct |
| notch_18000hz_q0.5_s8 | 26.487 | 51.447 | notch | 18000 | 0.5 | 72 dB/oct |
| bandpass_2000hz_q1_s8 | 26.327 | 61.529 | bandpass | 2000 | 1 | 72 dB/oct |
| notch_18000hz_q1_s8 | 26.326 | 53.839 | notch | 18000 | 1 | 72 dB/oct |
| notch_16000hz_q1_s8 | 26.203 | 51.413 | notch | 16000 | 1 | 72 dB/oct |
| notch_8000hz_q0.5_s8 | 25.865 | 57.139 | notch | 8000 | 0.5 | 72 dB/oct |
| notch_19000hz_q1_s8 | 25.372 | 52.755 | notch | 19000 | 1 | 72 dB/oct |
| bandpass_19000hz_q4_s8 | 25.318 | 66.800 | bandpass | 19000 | 4 | 72 dB/oct |
| bandpass_8000hz_q0.5_s8 | 25.155 | 87.608 | bandpass | 8000 | 0.5 | 72 dB/oct |
| notch_14000hz_q1_s8 | 24.482 | 55.205 | notch | 14000 | 1 | 72 dB/oct |
| low_cut_20000hz_q10_s5 | 24.463 | 41.607 | low_cut | 20000 | 10 | 36 dB/oct |
| notch_19000hz_q0.5_s8 | 24.384 | 45.199 | notch | 19000 | 0.5 | 72 dB/oct |
| bandpass_500hz_q1_s5 | 24.350 | 46.472 | bandpass | 500 | 1 | 36 dB/oct |
| bandpass_18000hz_q4_s8 | 24.060 | 65.869 | bandpass | 18000 | 4 | 72 dB/oct |
| bandpass_8000hz_q4_s8 | 23.872 | 65.813 | bandpass | 8000 | 4 | 72 dB/oct |
| notch_20000hz_q1_s8 | 23.862 | 57.486 | notch | 20000 | 1 | 72 dB/oct |
| notch_21000hz_q10_s8 | 23.804 | 144.367 | notch | 21000 | 10 | 72 dB/oct |
| bandpass_2000hz_q10_s8 | 23.329 | 83.747 | bandpass | 2000 | 10 | 72 dB/oct |
| notch_12000hz_q1_s8 | 23.151 | 55.418 | notch | 12000 | 1 | 72 dB/oct |
| bandpass_17000hz_q4_s8 | 22.957 | 63.624 | bandpass | 17000 | 4 | 72 dB/oct |
| bandpass_10000hz_q4_s8 | 22.911 | 66.354 | bandpass | 10000 | 4 | 72 dB/oct |
| bandpass_20hz_q1_s2 | 22.870 | 44.583 | bandpass | 20 | 1 | 18 dB/oct |
| low_cut_21000hz_q10_s5 | 22.661 | 41.143 | low_cut | 21000 | 10 | 36 dB/oct |
| bandpass_16000hz_q4_s8 | 22.265 | 67.659 | bandpass | 16000 | 4 | 72 dB/oct |
| notch_20000hz_q0.5_s8 | 21.923 | 60.726 | notch | 20000 | 0.5 | 72 dB/oct |
| bandpass_10000hz_q0.5_s8 | 21.864 | 94.967 | bandpass | 10000 | 0.5 | 72 dB/oct |
| bandpass_14000hz_q4_s8 | 21.832 | 65.131 | bandpass | 14000 | 4 | 72 dB/oct |
| bandpass_12000hz_q4_s8 | 21.689 | 63.207 | bandpass | 12000 | 4 | 72 dB/oct |
| bandpass_2000hz_q4_s8 | 21.667 | 57.934 | bandpass | 2000 | 4 | 72 dB/oct |
| bandpass_22000hz_q10_s5 | 21.487 | 66.739 | bandpass | 22000 | 10 | 36 dB/oct |
| bandpass_1000hz_q0.5_s5 | 21.260 | 38.428 | bandpass | 1000 | 0.5 | 36 dB/oct |
| bandpass_500hz_q4_s5 | 21.235 | 45.994 | bandpass | 500 | 4 | 36 dB/oct |
| bandpass_500hz_q10_s5 | 20.956 | 45.896 | bandpass | 500 | 10 | 36 dB/oct |

(2141 more failures not shown)

## Passing (3326 total)

### Closest to Threshold (top 50)

| Scenario | RMS (dB) | Max (dB) | Filter | Freq | Q | Slope |
|----------|----------|----------|--------|------|---|-------|
| low_shelf_12000hz_-12db_q0.5_s2 | 0.500 | 0.908 | low_shelf | 12000 | 0.5 | 18 dB/oct |
| high_shelf_12000hz_+12db_q0.5_s2 | 0.500 | 0.908 | high_shelf | 12000 | 0.5 | 18 dB/oct |
| bandpass_8000hz_q4_s0 | 0.497 | 4.689 | bandpass | 8000 | 4 | 6 dB/oct |
| low_shelf_18000hz_+12db_q1_s5 | 0.496 | 0.817 | low_shelf | 18000 | 1 | 36 dB/oct |
| high_shelf_18000hz_-12db_q1_s5 | 0.496 | 0.817 | high_shelf | 18000 | 1 | 36 dB/oct |
| high_shelf_200hz_-12db_q10_s2 | 0.495 | 9.055 | high_shelf | 200 | 10 | 18 dB/oct |
| low_shelf_200hz_+12db_q10_s2 | 0.495 | 9.055 | low_shelf | 200 | 10 | 18 dB/oct |
| flat_tilt_22000hz_-12db_q1_s2 | 0.494 | 0.995 | flat_tilt | 22000 | 1 | 18 dB/oct |
| flat_tilt_22000hz_+12db_q1_s2 | 0.494 | 0.907 | flat_tilt | 22000 | 1 | 18 dB/oct |
| bell_12000hz_-12db_q0.5_s0 | 0.492 | 1.127 | bell | 12000 | 0.5 | 6 dB/oct |
| bell_12000hz_-12db_q0.5_s2 | 0.492 | 1.127 | bell | 12000 | 0.5 | 18 dB/oct |
| high_shelf_14000hz_-12db_q0.5_s2 | 0.492 | 0.863 | high_shelf | 14000 | 0.5 | 18 dB/oct |
| low_shelf_14000hz_+12db_q0.5_s2 | 0.492 | 0.863 | low_shelf | 14000 | 0.5 | 18 dB/oct |
| allpass_200hz_q10_s8 | 0.492 | 22.020 | allpass | 200 | 10 | 72 dB/oct |
| bell_16000hz_-6db_q10_s5 | 0.492 | 1.232 | bell | 16000 | 10 | 36 dB/oct |
| bell_8000hz_-6db_q10_s8 | 0.491 | 1.994 | bell | 8000 | 10 | 72 dB/oct |
| bell_16000hz_+6db_q10_s5 | 0.491 | 1.235 | bell | 16000 | 10 | 36 dB/oct |
| bandpass_14000hz_q1_s0 | 0.491 | 5.471 | bandpass | 14000 | 1 | 6 dB/oct |
| bell_20000hz_-6db_q4_s0 | 0.491 | 1.641 | bell | 20000 | 4 | 6 dB/oct |
| bell_20000hz_-6db_q4_s2 | 0.491 | 1.641 | bell | 20000 | 4 | 18 dB/oct |
| bell_8000hz_+6db_q10_s8 | 0.490 | 1.992 | bell | 8000 | 10 | 72 dB/oct |
| tilt_shelf_18000hz_-6db_q1_s8 | 0.484 | 0.929 | tilt_shelf | 18000 | 1 | 72 dB/oct |
| high_shelf_500hz_-12db_q4_s2 | 0.484 | 4.781 | high_shelf | 500 | 4 | 18 dB/oct |
| low_shelf_500hz_+12db_q4_s2 | 0.484 | 4.781 | low_shelf | 500 | 4 | 18 dB/oct |
| high_shelf_20000hz_+6db_q1_s2 | 0.482 | 0.835 | high_shelf | 20000 | 1 | 18 dB/oct |
| low_shelf_20000hz_-6db_q1_s2 | 0.482 | 0.835 | low_shelf | 20000 | 1 | 18 dB/oct |
| tilt_shelf_18000hz_-6db_q1_s5 | 0.481 | 0.764 | tilt_shelf | 18000 | 1 | 36 dB/oct |
| high_shelf_19000hz_+6db_q1_s5 | 0.479 | 0.791 | high_shelf | 19000 | 1 | 36 dB/oct |
| low_shelf_19000hz_-6db_q1_s5 | 0.479 | 0.791 | low_shelf | 19000 | 1 | 36 dB/oct |
| bell_2000hz_-12db_q10_s8 | 0.477 | 3.973 | bell | 2000 | 10 | 72 dB/oct |
| notch_200hz_q10_s8 | 0.477 | 20.472 | notch | 200 | 10 | 72 dB/oct |
| bell_200hz_-12db_q1_s8 | 0.476 | 3.833 | bell | 200 | 1 | 72 dB/oct |
| bell_2000hz_+12db_q10_s8 | 0.476 | 3.868 | bell | 2000 | 10 | 72 dB/oct |
| bell_200hz_+12db_q1_s8 | 0.475 | 3.849 | bell | 200 | 1 | 72 dB/oct |
| bell_200hz_-12db_q0.5_s5 | 0.472 | 2.175 | bell | 200 | 0.5 | 36 dB/oct |
| bell_200hz_+12db_q0.5_s5 | 0.472 | 2.176 | bell | 200 | 0.5 | 36 dB/oct |
| bell_22000hz_+6db_q0.5_s0 | 0.472 | 0.774 | bell | 22000 | 0.5 | 6 dB/oct |
| bell_22000hz_+6db_q0.5_s2 | 0.472 | 0.774 | bell | 22000 | 0.5 | 18 dB/oct |
| bell_100hz_-12db_q0.5_s8 | 0.472 | 3.882 | bell | 100 | 0.5 | 72 dB/oct |
| bell_100hz_+12db_q0.5_s8 | 0.472 | 3.846 | bell | 100 | 0.5 | 72 dB/oct |
| high_shelf_16000hz_-12db_q1_s8 | 0.471 | 0.953 | high_shelf | 16000 | 1 | 72 dB/oct |
| low_shelf_16000hz_+12db_q1_s8 | 0.471 | 0.953 | low_shelf | 16000 | 1 | 72 dB/oct |
| tilt_shelf_20000hz_-12db_q1_s2 | 0.470 | 0.770 | tilt_shelf | 20000 | 1 | 18 dB/oct |
| low_cut_2000hz_q1_s8 | 0.469 | 11.422 | low_cut | 2000 | 1 | 72 dB/oct |
| low_cut_200hz_q4_s5 | 0.469 | 4.761 | low_cut | 200 | 4 | 36 dB/oct |
| high_shelf_5000hz_-12db_q1_s2 | 0.468 | 0.732 | high_shelf | 5000 | 1 | 18 dB/oct |
| low_shelf_5000hz_+12db_q1_s2 | 0.468 | 0.732 | low_shelf | 5000 | 1 | 18 dB/oct |
| low_shelf_5000hz_-12db_q1_s2 | 0.467 | 0.729 | low_shelf | 5000 | 1 | 18 dB/oct |
| high_shelf_5000hz_+12db_q1_s2 | 0.467 | 0.729 | high_shelf | 5000 | 1 | 18 dB/oct |
| high_shelf_2000hz_-6db_q10_s8 | 0.467 | 4.727 | high_shelf | 2000 | 10 | 72 dB/oct |

