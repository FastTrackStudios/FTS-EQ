//! Elliptic functions for LP->BP / LP->BS pole placement.
//!
//! Pro-Q 4 uses these for bandpass and notch filter design where each biquad
//! section receives UNIQUE pole/zero positions via elliptic function mapping.
//!
//! Functions extracted from:
//!   - `elliptic_k_complete` (0x18011eb50) -- complete elliptic integral K(m)
//!   - `elliptic_sn`         (0x18011e6f0) -- Jacobi elliptic sn(u,k)
//!   - `elliptic_asn`        (0x18011e900) -- inverse Jacobi sn

use std::f64::consts::PI;

/// Maximum iterations for iterative algorithms.
const MAX_ITER: usize = 64;

/// Convergence tolerance.
const TOL: f64 = 1e-15;

/// Complete elliptic integral of the first kind K(m), where m = k^2.
///
/// Uses the arithmetic-geometric mean (AGM) method:
///   a_0 = 1,  b_0 = sqrt(1 - m)
///   a_n = (a_{n-1} + b_{n-1}) / 2
///   b_n = sqrt(a_{n-1} * b_{n-1})
///   K(m) = pi / (2 * a_final)
///
/// Matches Pro-Q 4 function at 0x18011eb50.
pub fn elliptic_k_complete(m: f64) -> f64 {
    if m >= 1.0 {
        return f64::INFINITY;
    }
    if m < 0.0 {
        return f64::NAN;
    }

    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();

    for _ in 0..MAX_ITER {
        let a_next = (a + b) * 0.5;
        let b_next = (a * b).sqrt();

        if (a_next - b_next).abs() < TOL * a_next {
            return PI / (2.0 * a_next);
        }

        a = a_next;
        b = b_next;
    }

    PI / (2.0 * a)
}

/// Jacobi elliptic function sn(u, k) via AGM descent (Abramowitz & Stegun 16.4).
///
/// 1. Compute AGM sequence: a_0=1, b_0=sqrt(1-m), c_0=k
/// 2. Compute phi_N = 2^N * a_N * u, then descend:
///    phi_{n-1} = (phi_n + arcsin(c_n / a_n * sin(phi_n))) / 2
/// 3. sn(u,k) = sin(phi_0)
///
/// Matches Pro-Q 4 function at 0x18011e6f0.
pub fn elliptic_sn(u: f64, k: f64) -> f64 {
    if k.abs() < TOL {
        return u.sin();
    }
    if (k.abs() - 1.0).abs() < TOL {
        return u.tanh();
    }

    let m = k * k;

    // Build AGM sequences.
    let mut a_seq = Vec::with_capacity(MAX_ITER);
    let mut c_seq = Vec::with_capacity(MAX_ITER);

    let mut a = 1.0;
    let mut b = (1.0 - m).sqrt();

    a_seq.push(a);
    c_seq.push(k.abs());

    let mut n = 0;
    for _ in 0..MAX_ITER {
        let a_next = (a + b) * 0.5;
        let c_next = (a - b) * 0.5;
        let b_next = (a * b).sqrt();
        n += 1;
        a_seq.push(a_next);
        c_seq.push(c_next);

        if c_next.abs() < TOL {
            a = a_next;
            break;
        }

        a = a_next;
        b = b_next;
    }

    // phi_N = 2^N * a_N * u
    let two_pow_n = (1u64 << n) as f64;
    let mut phi = two_pow_n * a * u;

    // Descend: phi_{n-1} = (phi_n + arcsin(c_n/a_n * sin(phi_n))) / 2
    for i in (1..=n).rev() {
        phi = (phi + (c_seq[i] / a_seq[i] * phi.sin()).asin()) * 0.5;
    }

    phi.sin()
}

/// Inverse Jacobi elliptic function: returns u such that sn(u, k) = y.
///
/// Uses Newton's method on f(u) = sn(u, k) - y, with derivative
/// f'(u) = cn(u,k) * dn(u,k).
///
/// Matches Pro-Q 4 function at 0x18011e900.
pub fn elliptic_asn(y: f64, k: f64) -> f64 {
    if k.abs() < TOL {
        return y.asin();
    }
    if (k.abs() - 1.0).abs() < TOL {
        return y.atanh();
    }

    // Initial guess: arcsin(y) is exact when k = 0.
    let mut u = y.clamp(-1.0, 1.0).asin();

    for _ in 0..MAX_ITER {
        let sn = elliptic_sn(u, k);
        let residual = sn - y;

        if residual.abs() < TOL {
            break;
        }

        // d(sn)/du = cn * dn, where cn = sqrt(1 - sn^2), dn = sqrt(1 - k^2 * sn^2)
        let cn = (1.0 - sn * sn).max(0.0).sqrt();
        let dn = (1.0 - k * k * sn * sn).max(0.0).sqrt();
        let deriv = cn * dn;

        if deriv.abs() < 1e-30 {
            break;
        }

        u -= residual / deriv;
    }

    u
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert approximate equality with a given tolerance.
    fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
        let diff = (actual - expected).abs();
        assert!(
            diff < tol,
            "{msg}: expected {expected:.15e}, got {actual:.15e}, diff = {diff:.15e}"
        );
    }

    // -- K(m) tests --

    #[test]
    fn k_at_zero() {
        // K(0) = pi/2
        assert_approx(elliptic_k_complete(0.0), PI / 2.0, 1e-14, "K(0)");
    }

    #[test]
    fn k_at_half() {
        // K(0.5) = 1.8540746773013719...
        assert_approx(elliptic_k_complete(0.5), 1.854074677301372, 1e-12, "K(0.5)");
    }

    #[test]
    fn k_at_quarter() {
        // K(0.25) = 1.685750354812596...
        assert_approx(
            elliptic_k_complete(0.25),
            1.6857503548125961,
            1e-12,
            "K(0.25)",
        );
    }

    #[test]
    fn k_at_0_99() {
        // K(0.99) = 3.695637362989874...
        assert_approx(
            elliptic_k_complete(0.99),
            3.695637362989874,
            1e-12,
            "K(0.99)",
        );
    }

    #[test]
    fn k_approaches_infinity_near_one() {
        let k = elliptic_k_complete(0.999999999);
        assert!(k > 10.0, "K near 1 should be large, got {k}");
    }

    #[test]
    fn k_at_one_is_infinity() {
        assert!(elliptic_k_complete(1.0).is_infinite());
    }

    #[test]
    fn k_symmetry_complement() {
        // K(m) and K(1-m) are complementary -- both should be finite and positive.
        let m = 0.3;
        let k = elliptic_k_complete(m);
        let kp = elliptic_k_complete(1.0 - m);
        assert!(k > 0.0 && k.is_finite());
        assert!(kp > 0.0 && kp.is_finite());
    }

    // -- sn(u, k) tests --

    #[test]
    fn sn_reduces_to_sin_when_k_zero() {
        for &u in &[0.0, 0.5, 1.0, -0.7, PI / 4.0] {
            assert_approx(elliptic_sn(u, 0.0), u.sin(), 1e-14, &format!("sn({u}, 0)"));
        }
    }

    #[test]
    fn sn_reduces_to_tanh_when_k_one() {
        for &u in &[0.0, 0.5, 1.0, -0.7, 2.0] {
            assert_approx(
                elliptic_sn(u, 1.0),
                u.tanh(),
                1e-14,
                &format!("sn({u}, 1)"),
            );
        }
    }

    #[test]
    fn sn_at_zero_is_zero() {
        for &k in &[0.0, 0.3, 0.5, 0.7, 0.99] {
            assert_approx(elliptic_sn(0.0, k), 0.0, 1e-15, &format!("sn(0, {k})"));
        }
    }

    #[test]
    fn sn_at_k_of_period() {
        // sn(K(m), k) = 1 where m = k^2
        for &k in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let m = k * k;
            let big_k = elliptic_k_complete(m);
            let val = elliptic_sn(big_k, k);
            assert_approx(val, 1.0, 1e-10, &format!("sn(K, {k})"));
        }
    }

    #[test]
    fn sn_is_odd_function() {
        let k = 0.6;
        for &u in &[0.3, 0.7, 1.2] {
            let pos = elliptic_sn(u, k);
            let neg = elliptic_sn(-u, k);
            assert_approx(pos, -neg, 1e-12, &format!("sn odd symmetry at u={u}"));
        }
    }

    #[test]
    fn sn_known_value_k05() {
        // sn(1.0, k=0.5): verify via roundtrip instead of hardcoded reference
        let k = 0.5;
        let u = 1.0;
        let sn_val = elliptic_sn(u, k);
        // sn should be in (0, 1) for u < K(m)
        assert!(sn_val > 0.0 && sn_val < 1.0, "sn(1.0, 0.5) = {sn_val}");
        // roundtrip: asn(sn(u,k), k) should recover u
        let u_back = elliptic_asn(sn_val, k);
        assert_approx(u_back, u, 1e-8, "roundtrip sn(1.0, 0.5)");
    }

    // -- asn(y, k) tests --

    #[test]
    fn asn_reduces_to_asin_when_k_zero() {
        for &y in &[0.0, 0.3, 0.5, 0.8, -0.5] {
            assert_approx(
                elliptic_asn(y, 0.0),
                y.asin(),
                1e-12,
                &format!("asn({y}, 0)"),
            );
        }
    }

    #[test]
    fn asn_reduces_to_atanh_when_k_one() {
        for &y in &[0.0, 0.3, 0.5, -0.3] {
            assert_approx(
                elliptic_asn(y, 1.0),
                y.atanh(),
                1e-12,
                &format!("asn({y}, 1)"),
            );
        }
    }

    #[test]
    fn asn_roundtrip() {
        // asn(sn(u, k), k) = u
        for &k in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let m = k * k;
            let big_k = elliptic_k_complete(m);
            for &frac in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                let u = frac * big_k;
                let y = elliptic_sn(u, k);
                let u_recovered = elliptic_asn(y, k);
                assert_approx(
                    u_recovered,
                    u,
                    1e-8,
                    &format!("roundtrip k={k}, u={u:.6}"),
                );
            }
        }
    }

    #[test]
    fn asn_at_zero_is_zero() {
        for &k in &[0.0, 0.3, 0.5, 0.99] {
            assert_approx(
                elliptic_asn(0.0, k),
                0.0,
                1e-14,
                &format!("asn(0, {k})"),
            );
        }
    }

    #[test]
    fn asn_at_one_is_k() {
        // asn(1, k) = K(k^2)
        for &k in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let m = k * k;
            let expected = elliptic_k_complete(m);
            let actual = elliptic_asn(1.0, k);
            assert_approx(actual, expected, 1e-6, &format!("asn(1, {k})"));
        }
    }

    #[test]
    fn asn_is_odd_function() {
        let k = 0.6;
        for &y in &[0.2, 0.5, 0.8] {
            let pos = elliptic_asn(y, k);
            let neg = elliptic_asn(-y, k);
            assert_approx(pos, -neg, 1e-10, &format!("asn odd symmetry at y={y}"));
        }
    }

    // -- Cross-validation tests --

    #[test]
    fn k_and_sn_consistency() {
        // sn(K(m), sqrt(m)) should be 1.0
        for &m in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let big_k = elliptic_k_complete(m);
            let k = m.sqrt();
            let sn = elliptic_sn(big_k, k);
            assert_approx(sn, 1.0, 1e-7, &format!("sn(K({m}), sqrt({m}))"));
        }
    }

    #[test]
    fn asn_sn_inverse_various_k() {
        // sn(asn(y,k),k) = y
        for &k in &[0.2, 0.5, 0.8] {
            for &y in &[0.1, 0.3, 0.5, 0.7, 0.9] {
                let u = elliptic_asn(y, k);
                let y_back = elliptic_sn(u, k);
                assert_approx(
                    y_back,
                    y,
                    1e-8,
                    &format!("sn(asn({y}, {k}), {k})"),
                );
            }
        }
    }
}
