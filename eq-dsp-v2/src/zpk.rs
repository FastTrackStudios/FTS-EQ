//! Zero-Pole-Gain representation.
//!
//! Pro-Q 4 uses ZPK internally (20 doubles per section at output_buffer+0x48):
//!   - Complex pole pairs
//!   - Complex zero pairs
//!   - Scalar gain
//!   - Infinity sentinel (0x7FF0000000000000) for unused poles/zeros

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number for filter pole/zero calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn mag_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn mag(self) -> f64 {
        self.mag_sq().sqrt()
    }

    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn is_real(self) -> bool {
        self.im.abs() < 1e-15
    }

    pub fn inv(self) -> Self {
        let d = self.mag_sq();
        Self {
            re: self.re / d,
            im: -self.im / d,
        }
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.inv()
    }
}

impl Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

impl Mul<Complex> for f64 {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        Complex {
            re: self * rhs.re,
            im: self * rhs.im,
        }
    }
}

impl Div<f64> for Complex {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self {
            re: self.re / rhs,
            im: self.im / rhs,
        }
    }
}

impl Add<f64> for Complex {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            re: self.re + rhs,
            im: self.im,
        }
    }
}

impl Sub<f64> for Complex {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Self {
            re: self.re - rhs,
            im: self.im,
        }
    }
}

/// A single second-order section in ZPK form.
///
/// Represents H(s) = gain * (s - z0)(s - z1) / ((s - p0)(s - p1))
/// where z0/z1 are zeros and p0/p1 are poles (conjugate pairs for real coefficients).
#[derive(Debug, Clone)]
pub struct Zpk {
    pub zeros: Vec<Complex>,
    pub poles: Vec<Complex>,
    pub gain: f64,
}

impl Zpk {
    pub fn new(zeros: Vec<Complex>, poles: Vec<Complex>, gain: f64) -> Self {
        Self { zeros, poles, gain }
    }

    /// Number of second-order sections needed.
    pub fn num_sos(&self) -> usize {
        let n = self.poles.len().max(self.zeros.len());
        (n + 1) / 2
    }

    /// Evaluate H(s) at a complex frequency point.
    pub fn eval(&self, s: Complex) -> Complex {
        let mut num = Complex::new(self.gain, 0.0);
        for &z in &self.zeros {
            num = num * (s - z);
        }
        let mut den = Complex::ONE;
        for &p in &self.poles {
            den = den * (s - p);
        }
        num / den
    }

    /// Evaluate on the unit circle: H(e^{jw}).
    pub fn eval_z(&self, w: f64) -> Complex {
        let ejw = Complex::from_polar(1.0, w);
        self.eval(ejw)
    }

    /// Evaluate magnitude response in dB at digital frequency w.
    pub fn mag_db(&self, w: f64) -> f64 {
        20.0 * self.eval_z(w).mag().log10()
    }
}

/// Pair complex conjugate poles/zeros for second-order sections.
///
/// Returns pairs of (pole_pair, zero_pair) ready for biquad conversion.
/// Real poles/zeros are paired together; conjugate pairs stay together.
pub fn pair_conjugates(zpk: &Zpk) -> Vec<(Vec<Complex>, Vec<Complex>, f64)> {
    let mut poles = zpk.poles.clone();
    let mut zeros = zpk.zeros.clone();

    // Sort: real poles first, then by imaginary part magnitude
    poles.sort_by(|a, b| {
        let a_real = a.im.abs() < 1e-12;
        let b_real = b.im.abs() < 1e-12;
        if a_real != b_real {
            return if a_real {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        a.im.abs().partial_cmp(&b.im.abs()).unwrap()
    });

    zeros.sort_by(|a, b| {
        let a_real = a.im.abs() < 1e-12;
        let b_real = b.im.abs() < 1e-12;
        if a_real != b_real {
            return if a_real {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        a.im.abs().partial_cmp(&b.im.abs()).unwrap()
    });

    // Group into conjugate pairs
    let pole_pairs = group_conjugate_pairs(&poles);
    let zero_pairs = group_conjugate_pairs(&zeros);

    let n = pole_pairs.len().max(zero_pairs.len());
    let gain_per = zpk.gain.abs().powf(1.0 / n as f64) * zpk.gain.signum();

    let mut sections = Vec::with_capacity(n);
    for i in 0..n {
        let pp = if i < pole_pairs.len() {
            pole_pairs[i].clone()
        } else {
            vec![]
        };
        let zp = if i < zero_pairs.len() {
            zero_pairs[i].clone()
        } else {
            vec![]
        };
        let g = if i == 0 {
            zpk.gain / gain_per.powi((n - 1) as i32)
        } else {
            gain_per
        };
        sections.push((pp, zp, g));
    }
    sections
}

fn group_conjugate_pairs(roots: &[Complex]) -> Vec<Vec<Complex>> {
    let mut used = vec![false; roots.len()];
    let mut pairs = Vec::new();

    for i in 0..roots.len() {
        if used[i] {
            continue;
        }
        used[i] = true;

        if roots[i].im.abs() < 1e-12 {
            // Real root — try to find another real root to pair with
            let mut found = false;
            for j in (i + 1)..roots.len() {
                if !used[j] && roots[j].im.abs() < 1e-12 {
                    pairs.push(vec![roots[i], roots[j]]);
                    used[j] = true;
                    found = true;
                    break;
                }
            }
            if !found {
                pairs.push(vec![roots[i]]);
            }
        } else {
            // Complex root — find its conjugate
            let conj = roots[i].conj();
            for j in (i + 1)..roots.len() {
                if !used[j]
                    && (roots[j].re - conj.re).abs() < 1e-12
                    && (roots[j].im - conj.im).abs() < 1e-12
                {
                    used[j] = true;
                    break;
                }
            }
            pairs.push(vec![roots[i], roots[i].conj()]);
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn complex_basic_ops() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let c = a * b;
        assert!((c.re - (-5.0)).abs() < 1e-10);
        assert!((c.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn complex_from_polar() {
        let c = Complex::from_polar(1.0, PI / 4.0);
        assert!((c.re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((c.im - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn complex_inv() {
        let c = Complex::new(1.0, 1.0);
        let inv = c.inv();
        let prod = c * inv;
        assert!((prod.re - 1.0).abs() < 1e-10);
        assert!(prod.im.abs() < 1e-10);
    }
}
