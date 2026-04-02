//! Zero-Pole-Gain representation.
//!
//! Pro-Q 4 stores ZPK internally (20 doubles per section at output+0x48).
//! Infinity sentinel (0x7FF0000000000000) marks unused poles/zeros.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Complex number for filter pole/zero calculations.
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

    pub fn sqrt(self) -> Self {
        let r = self.mag();
        let theta = self.arg();
        Self::from_polar(r.sqrt(), theta / 2.0)
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

/// Zero-Pole-Gain representation of a filter.
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

    pub fn num_sos(&self) -> usize {
        let n = self.poles.len().max(self.zeros.len());
        (n + 1) / 2
    }

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

    pub fn eval_z(&self, w: f64) -> Complex {
        let ejw = Complex::from_polar(1.0, w);
        self.eval(ejw)
    }

    pub fn mag_db(&self, w: f64) -> f64 {
        20.0 * self.eval_z(w).mag().log10()
    }
}

/// Pair complex conjugate poles/zeros for second-order sections.
pub fn pair_conjugates(zpk: &Zpk) -> Vec<(Vec<Complex>, Vec<Complex>, f64)> {
    let mut poles = zpk.poles.clone();
    let mut zeros = zpk.zeros.clone();

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

    let pole_pairs = group_conjugate_pairs(&poles);
    let zero_pairs = group_conjugate_pairs(&zeros);

    let n = pole_pairs.len().max(zero_pairs.len());
    let gain_per = if n > 0 {
        zpk.gain.abs().powf(1.0 / n as f64) * zpk.gain.signum()
    } else {
        zpk.gain
    };

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
    fn complex_sqrt_positive() {
        let z = Complex::new(4.0, 0.0);
        let s = z.sqrt();
        assert!((s.re - 2.0).abs() < 1e-10);
        assert!(s.im.abs() < 1e-10);
    }

    #[test]
    fn complex_sqrt_negative() {
        let z = Complex::new(-1.0, 0.0);
        let s = z.sqrt();
        assert!(s.re.abs() < 1e-10);
        assert!((s.im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn complex_from_polar() {
        let c = Complex::from_polar(1.0, PI / 4.0);
        assert!((c.re - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((c.im - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }
}
