# LP/HP Filter Matching: Approaches & Status

## Current Score: 4891/5567 (87.9%) — LP/HP: 70 remaining failures

### Breakdown of LP/HP failures

| Filter  | s2 | s5 | s8 | Total |
|---------|----|----|-----|-------|
| LP (high_cut) | 0 | 3 | 28 | 31 |
| HP (low_cut)  | 2 | 7 | 30 | 39 |
| **Total**     | 2 | 10 | 58 | **70** |

s8 cascades (order 8 = 4 biquads) account for 58/70 (83%) of remaining LP/HP failures.

---

## Architecture

**Pole placement**: Impulse Invariance (II) — `z = exp(sT)`, maps analog poles exactly in time domain but compresses frequency axis near Nyquist.

**Numerator**: Vicanek 3-point magnitude matching — pins DC, corner (|H(w0)|=Q²), and Nyquist gains. Uses squared-magnitude parametrization `[B0, B1, B2]` with `mag_sq_to_b` recovery.

**HP numerator constraint**: `(scale, -2*scale, scale)` — only one free parameter (scale) after enforcing DC=0. Testing showed free-form numerator always converges to b0≈b2, confirming the constraint is optimal.

**Cascade structure** (from `band.rs`):
- s2 = order 2 = 1 biquad
- s5 = order 5 = 1 matched 1st-order + 2 biquads
- s8 = order 8 = 4 biquads
- Section 0 gets user Q scaling (`bw_q * display_q`), others get pure Butterworth Q

---

## Approaches Tried

### 1. Sigma correction (CURRENT — working for s2)

Modify damping factor σ = 0.5/Q before II pole computation to compensate for frequency warping.

**LP underdamped** (σ < 1): Reduce σ → steeper transition band
```
correction = 0.982 * σ^0.529 * w_norm^4.069
scaled = correction / √num_biquads
σ_eff = σ * (1 - min(scaled, 0.49))
```

**LP overdamped** (σ ≥ 1, w_norm > 0.5): Increase σ + Nyquist scale
```
σ_eff = σ * (1 + 0.17 * w_norm), nyq_scale = 1.17
```

**HP high-Q** (Q > 1, w_norm > 0.3): Increase σ via power law
```
correction = 0.9785 * w_norm^6.1534 * ln(Q)^0.7695
scaled = correction / num_biquads
```

**HP overdamped** (σ ≥ 1, w_norm > 0.75): Reduce σ + scale adjustment
```
σ_corr = 0.3 * (w_norm - 0.15)
scale_ratio = clamp(1.293 - 0.44*w_norm, 0.7, 1.0)
```

**HP low-Q near Nyquist** (w0 > 0.8): Gentle BLT pole blend
```
q_factor = 0.024 / (Q² + 0.024)
w_factor = clamp((w0 - 0.8) * 0.5, 0, 1)
blt_weight = q_factor * w_factor
```

**Results**: Fixes all s2 LP (was 31 failures, now 0). Fixes most s2 HP (was 39, now 2 borderline). Insufficient for s5/s8 cascades.

**Limitation**: σ correction capped at 0.49 (can't halve σ further without instability). Even at max correction, s8 Q=4/Q=10 at 14-22kHz still 3-7 dB off. The per-section error compounds across 4 biquads.

### 2. Cascade division factor (`/√num_biquads` vs `/num_biquads`)

Tested dividing the sigma correction by different functions of `num_biquads`:

| Division | LP s5 | LP s8 | HP s5 | HP s8 |
|----------|-------|-------|-------|-------|
| None | 1 fail | 32 fail | 22 fail | — |
| `/num_biquads` | 6 fail | 28 fail | 7 fail | 30 fail |
| `/√num_biquads` | 3 fail | 28 fail | — | — |

**Outcome**: `√num_biquads` is best for LP, `num_biquads` is best for HP. Neither fixes s8.

### 3. LP overdamped: a2=0 approach (ABANDONED)

For overdamped LP sections, tried setting a2=0 (first-order-like response). Catastrophically broke low-frequency filters (20Hz Q=0.5 → 134 dB error). Even with frequency guard (`w_norm > 0.6`), the approach was fragile. Replaced by σ+nyq_scale correction.

### 4. BLT pole blend for HP low-Q (CURRENT — limited use)

For HP with Q ≤ 1 near Nyquist (w0 > 0.8), blend II poles toward BLT poles using a small weight. Works well for the specific case of underdamped low-Q sections but the weight formula doesn't generalize to cascades.

### 5. Free HP numerator (TESTED — no benefit)

Tested removing the `(scale, -2*scale, scale)` constraint for HP, allowing independent b0, b1, b2. Optimization always converged to b0 ≈ b2, confirming the constraint is correct. The improvement comes from the SCALE, not numerator asymmetry.

### 6. Matched 1st-order sections for odd-order cascades (CURRENT)

`lowpass_1_matched` / `highpass_1_matched` use matched-Z pole + 2-point magnitude matching (DC + Nyquist) instead of bilinear. The bilinear 1st-order LP has an exact zero at Nyquist (z=-1), causing 40-60 dB excess attenuation in odd-order cascades. Matched version preserves the analog Nyquist gain. Fixed ~48 scenarios.

---

## Approaches Still To Explore

### A. BLT poles for cascades (PROMISING — needs testing)

Use BLT (bilinear transform) poles instead of II poles for multi-section cascades. BLT naturally handles high frequencies correctly (pre-warping maps cutoff exactly) but introduces mid-frequency warping.

**Hypothesis**: For cascades, BLT's exact cutoff matching outweighs its mid-frequency distortion, since per-section errors compound multiplicatively.

**Testing needed**:
- Pure BLT poles + Vicanek numerator for s5/s8 at all frequencies
- II-BLT pole interpolation with frequency-dependent blend (more BLT near Nyquist)
- Check for regressions at low/mid frequencies where II is currently accurate

**Initial Python analysis** (per-section optimization) shows optimal corrections max out at 0.49 for Q≥4 s8, suggesting pole placement is the fundamental bottleneck, not correction magnitude.

### B. Per-section adaptive correction

Instead of uniform correction across all sections, use section-specific correction based on each section's Q:
- High-Q sections (section 0 with user Q) need more correction
- Low-Q Butterworth sections may need different correction shape

### C. Frequency-warped II poles

Compute II poles at a pre-warped frequency: `w0_eff = 2*fs*tan(w0/2)` rather than `w0` directly. This is equivalent to applying BLT frequency warping to the II mapping, potentially getting the best of both approaches.

### D. Vicanek pole matching (3-point denominator)

Instead of using II for poles, solve for a1, a2 by matching the denominator magnitude at 2-3 frequency points (similar to how the numerator already matches 3 points). This decouples pole angle and radius.

### E. Hybrid cascade topology

Use II poles for section 0 (resonant, user Q) and BLT poles for remaining Butterworth sections. Or vice versa. The error profiles may complement each other.

### F. Higher sigma correction ceiling

Currently capped at 0.49. Testing with cap at 0.7 or 0.8 may help some s8 cases, though instability risk increases as σ_eff → 0.

---

## Key Insights

1. **Per-section errors compound**: 0.4 dB per section × 4 sections = 1.6 dB total. Need <0.25 dB per section for s8 to pass at 1.0 dB tolerance.

2. **II fundamentally wrong near Nyquist**: The II mapping compresses frequencies near Nyquist, making transition bands too gradual. This is the ROOT CAUSE of all LP/HP cascade failures.

3. **BLT fundamentally wrong away from cutoff**: BLT warps the entire frequency axis, only matching at DC and the pre-warped cutoff. This is fine for single biquads but may cause issues in cascades where mid-band accuracy matters.

4. **σ correction has hard ceiling**: Even optimal per-section corrections (unconstrained optimization) leave s8 Q=4/Q=10 at 3-7 dB error when hitting the σ cap. The approach is fundamentally insufficient for these cases.

5. **HP s2 borderline cases**: 18kHz Q=0.5 (1.007 dB) and 22kHz Q=4 (1.004 dB) are at the fundamental limit of the (scale, -2*scale, scale) HP numerator structure. May require accepting these as irreducible.

## Reference Data

- Compare command: `fts-analyzer-cli compare-eq --reference reference/pro-q4 --out /tmp/eq-compare-48k <plugin.clap>`
- Reference path: `/home/cody/Development/FastTrackStudio/fts-analyzer/reference/pro-q4/48k`
- Build: `cargo build --release --package eq-plugin && cp target/release/libeq_plugin.so target/bundled/eq-plugin.clap`
- Python env: `nix-shell -p python313 python313Packages.numpy python313Packages.scipy`
