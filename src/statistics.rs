//! Sampling distributions, ported from hypothesis.internal.statistics.
//!
//! The integer generator samples from a PiecewiseDistribution (uniform core +
//! heavy Student-t tail in log2 space). We port the Student-t CDF/quantile
//! (`stdtr`/`stdtrit`) and the distribution composition so unbounded/large
//! integer draws match hypothesis's distribution. Pure f64 math (gamma/lgamma
//! via `libm`).

use pyo3::prelude::*;
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Student's t CDF for integer df >= 1 (Abramowitz & Stegun 26.7.7-8).
pub(crate) fn stdtr(df: i64, t: f64) -> f64 {
    if t == 0.0 {
        return 0.5;
    }
    let dff = df as f64;
    let abs_t = t.abs();
    let z = 1.0 + abs_t * abs_t / dff;
    let p;
    if df % 2 == 1 {
        let u = abs_t / dff.sqrt();
        let mut pp = u.atan();
        if df > 1 {
            let mut f = 1.0;
            let mut tz = 1.0;
            let mut j = 3i64;
            while j <= df - 2 {
                tz *= (j as f64 - 1.0) / (z * j as f64);
                f += tz;
                j += 2;
            }
            pp += f * u / z;
        }
        pp *= 2.0 / PI;
        p = pp;
    } else {
        let mut f = 1.0;
        let mut tz = 1.0;
        let mut j = 2i64;
        while j <= df - 2 {
            tz *= (j as f64 - 1.0) / (z * j as f64);
            f += tz;
            j += 2;
        }
        p = f * abs_t / (z * dff).sqrt();
    }
    if t < 0.0 {
        0.5 - 0.5 * p
    } else {
        0.5 + 0.5 * p
    }
}

/// Inverse Student's t CDF (quantile) for integer df >= 1.
pub(crate) fn stdtrit(df: i64, p: f64) -> f64 {
    const EPS: f64 = 1e-10;
    const MAX_ITER: usize = 50;
    if p == 0.5 {
        return 0.0;
    }
    if df == 1 {
        if p > 0.5 {
            return (PI * (1.0 - p)).cos() / (PI * (1.0 - p)).sin();
        }
        return -(PI * p).cos() / (PI * p).sin();
    }
    if df == 2 {
        return (2.0 * p - 1.0) / (2.0 * p * (1.0 - p)).sqrt();
    }
    let dff = df as f64;
    let sign = if p > 0.5 { 1.0 } else { -1.0 };
    let q = if p > 0.5 { p } else { 1.0 - p };

    let mut lo = 0.0;
    let mut hi = 1.0;
    while stdtr(df, hi) < q {
        hi *= 2.0;
    }
    let log_norm =
        libm::lgamma(0.5 * (dff + 1.0)) - 0.5 * (dff * PI).ln() - libm::lgamma(0.5 * dff);
    let mut t = 0.5 * (lo + hi);
    for _ in 0..MAX_ITER {
        let big_f = stdtr(df, t);
        if big_f < q {
            lo = t;
        } else {
            hi = t;
        }
        let log_f = log_norm - 0.5 * (dff + 1.0) * (t * t / dff).ln_1p();
        let f = log_f.exp();
        if f == 0.0 {
            t = 0.5 * (lo + hi);
        } else {
            let t_newton = t - (big_f - q) / f;
            t = if lo <= t_newton && t_newton <= hi {
                t_newton
            } else {
                0.5 * (lo + hi)
            };
        }
        if hi - lo < EPS * (1.0 + t.abs()) {
            break;
        }
    }
    sign * t
}

fn clamp(lower: f64, value: f64, upper: f64) -> f64 {
    if value < lower {
        lower
    } else if value > upper {
        upper
    } else {
        value
    }
}

trait Distribution {
    fn cdf(&self, x: f64) -> f64;
    fn inverse_cdf(&self, u: f64) -> f64;
    fn pdf(&self, x: f64) -> f64;
}

struct Uniform {
    half_width: f64,
}

impl Distribution for Uniform {
    fn cdf(&self, x: f64) -> f64 {
        if x < -self.half_width {
            return 0.0;
        }
        if x > self.half_width {
            return 1.0;
        }
        (x + self.half_width) / (2.0 * self.half_width)
    }
    fn inverse_cdf(&self, u: f64) -> f64 {
        -self.half_width + 2.0 * self.half_width * u
    }
    fn pdf(&self, x: f64) -> f64 {
        if -self.half_width <= x && x <= self.half_width {
            1.0 / (2.0 * self.half_width)
        } else {
            0.0
        }
    }
}

const LN2: f64 = std::f64::consts::LN_2;

struct LogStudentT {
    scale_bits: f64,
    df: i64,
    t_coef: f64,
}

impl LogStudentT {
    fn new(scale_bits: f64, df: i64) -> Self {
        let dff = df as f64;
        let t_coef =
            libm::tgamma((dff + 1.0) / 2.0) / ((dff * PI).sqrt() * libm::tgamma(dff / 2.0));
        LogStudentT {
            scale_bits,
            df,
            t_coef,
        }
    }
}

impl Distribution for LogStudentT {
    fn cdf(&self, x: f64) -> f64 {
        let y = (1.0 + x.abs()).log2().copysign(x) / self.scale_bits;
        stdtr(self.df, y)
    }
    fn inverse_cdf(&self, u: f64) -> f64 {
        let y = self.scale_bits * stdtrit(self.df, u);
        let y = clamp(-1023.0, y, 1023.0);
        (y.abs() * LN2).exp_m1().copysign(y)
    }
    fn pdf(&self, x: f64) -> f64 {
        let y = (1.0 + x.abs()).log2().copysign(x) / self.scale_bits;
        let f_t = self.t_coef * (1.0 + y * y / self.df as f64).powf(-(self.df as f64 + 1.0) / 2.0);
        f_t / (self.scale_bits * (1.0 + x.abs()) * LN2)
    }
}

pub(crate) struct Piecewise {
    inner: Uniform,
    outer: LogStudentT,
    switchover: f64,
    alpha: f64,
    beta: f64,
    inner_mass: f64,
    left_mass: f64,
    inner_g_neg: f64,
    outer_g_pos: f64,
}

impl Piecewise {
    fn new(inner: Uniform, outer: LogStudentT, switchover: f64) -> Self {
        let inner_g_neg = inner.cdf(-switchover);
        let inner_g_pos = inner.cdf(switchover);
        let outer_g_neg = outer.cdf(-switchover);
        let outer_g_pos = outer.cdf(switchover);
        let outer_outer_mass = 1.0 - (outer_g_pos - outer_g_neg);
        let inner_inner_mass = inner_g_pos - inner_g_neg;
        let inner_pdf = inner.pdf(switchover);
        let outer_pdf = outer.pdf(switchover);
        let alpha = 1.0 / (outer_pdf * inner_inner_mass / inner_pdf + outer_outer_mass);
        let beta = alpha * outer_pdf / inner_pdf;
        let inner_mass = beta * inner_inner_mass;
        let left_mass = alpha * outer_g_neg;
        Piecewise {
            inner,
            outer,
            switchover,
            alpha,
            beta,
            inner_mass,
            left_mass,
            inner_g_neg,
            outer_g_pos,
        }
    }

    pub(crate) fn cdf(&self, x: f64) -> f64 {
        if x <= -self.switchover {
            return self.alpha * self.outer.cdf(x);
        }
        if x < self.switchover {
            return self.left_mass + self.beta * (self.inner.cdf(x) - self.inner_g_neg);
        }
        self.left_mass + self.inner_mass + self.alpha * (self.outer.cdf(x) - self.outer_g_pos)
    }

    pub(crate) fn inverse_cdf(&self, u: f64) -> f64 {
        if u <= self.left_mass {
            return self.outer.inverse_cdf(u / self.alpha);
        }
        if u < self.left_mass + self.inner_mass {
            let target = self.inner_g_neg + (u - self.left_mass) / self.beta;
            return self.inner.inverse_cdf(target);
        }
        self.outer
            .inverse_cdf((u - self.left_mass - self.inner_mass) / self.alpha + self.outer_g_pos)
    }
}

/// The fixed integer-sampling distribution (see hypothesis PR #4728).
pub(crate) fn integers_distribution() -> &'static Piecewise {
    static D: OnceLock<Piecewise> = OnceLock::new();
    D.get_or_init(|| {
        Piecewise::new(
            Uniform { half_width: 256.0 },
            LogStudentT::new(13.0, 2),
            256.0,
        )
    })
}

#[pyfunction]
#[pyo3(name = "stdtr")]
fn py_stdtr(df: i64, t: f64) -> f64 {
    stdtr(df, t)
}

#[pyfunction]
#[pyo3(name = "stdtrit")]
fn py_stdtrit(df: i64, p: f64) -> f64 {
    stdtrit(df, p)
}

#[pyfunction]
#[pyo3(name = "integers_dist_cdf")]
fn py_integers_dist_cdf(x: f64) -> f64 {
    integers_distribution().cdf(x)
}

#[pyfunction]
#[pyo3(name = "integers_dist_inverse_cdf")]
fn py_integers_dist_inverse_cdf(u: f64) -> f64 {
    integers_distribution().inverse_cdf(u)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_stdtr, m)?)?;
    m.add_function(wrap_pyfunction!(py_stdtrit, m)?)?;
    m.add_function(wrap_pyfunction!(py_integers_dist_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(py_integers_dist_inverse_cdf, m)?)?;
    Ok(())
}
