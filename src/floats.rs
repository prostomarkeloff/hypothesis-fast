//! Float bit-twiddling kernels, ported from hypothesis.internal.conjecture.floats.
//!
//! These define a lexical encoding of non-negative floats (`float_to_lex` /
//! `lex_to_float`) with good shrinking properties: simpler-looking floats get
//! smaller lex integers. Width-64 only — the genuinely hot, bit-level path used
//! by float shrinking and `choice_to_index`. Width 16/32 helpers stay in Python
//! (struct-based) where they are only used for counting/clamping.

use pyo3::prelude::*;
use std::sync::OnceLock;

const MAX_EXPONENT: u64 = 0x7FF;
const BIAS: i64 = 1023;
const MANTISSA_MASK: u64 = (1 << 52) - 1;

fn exponent_key(e: u64) -> f64 {
    if e == MAX_EXPONENT {
        return f64::INFINITY;
    }
    let unbiased = e as i64 - BIAS;
    if unbiased < 0 {
        (10000 - unbiased) as f64
    } else {
        unbiased as f64
    }
}

struct Tables {
    encoding: Vec<u64>,
    decoding: Vec<u64>,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let mut encoding: Vec<u64> = (0..=MAX_EXPONENT).collect();
        encoding.sort_by(|&a, &b| exponent_key(a).partial_cmp(&exponent_key(b)).unwrap());
        let mut decoding = vec![0u64; encoding.len()];
        for (i, &b) in encoding.iter().enumerate() {
            decoding[b as usize] = i as u64;
        }
        Tables { encoding, decoding }
    })
}

fn decode_exponent(e: u64) -> u64 {
    tables().encoding[e as usize]
}

fn encode_exponent(e: u64) -> u64 {
    tables().decoding[e as usize]
}

fn reverse_bits(x: u64, n: u32) -> u64 {
    // reverse the full 64-bit word, then drop the high (64 - n) bits.
    x.reverse_bits() >> (64 - n)
}

fn update_mantissa(unbiased_exponent: i64, mut mantissa: u64) -> u64 {
    if unbiased_exponent <= 0 {
        mantissa = reverse_bits(mantissa, 52);
    } else if unbiased_exponent <= 51 {
        let n_fractional_bits = (52 - unbiased_exponent) as u32;
        let fractional_part = mantissa & ((1u64 << n_fractional_bits) - 1);
        mantissa ^= fractional_part;
        mantissa |= reverse_bits(fractional_part, n_fractional_bits);
    }
    mantissa
}

pub(crate) fn is_simple_rs(f: f64) -> bool {
    if !f.is_finite() {
        return false;
    }
    if f != f.trunc() {
        return false;
    }
    // bit_length(int(f)) <= 56  <=>  |f| < 2**56  (for integral f)
    f.abs() < 72_057_594_037_927_936.0
}

fn base_float_to_lex(f: f64) -> u64 {
    let i = f.to_bits() & ((1u64 << 63) - 1);
    let exponent = i >> 52;
    let mantissa = update_mantissa(exponent as i64 - BIAS, i & MANTISSA_MASK);
    let exponent = encode_exponent(exponent);
    (1u64 << 63) | (exponent << 52) | mantissa
}

pub(crate) fn float_to_lex_rs(f: f64) -> u64 {
    if is_simple_rs(f) {
        return f as u64;
    }
    base_float_to_lex(f)
}

pub(crate) fn lex_to_float_rs(i: u64) -> f64 {
    let has_fractional_part = i >> 63;
    if has_fractional_part != 0 {
        let exponent = decode_exponent((i >> 52) & ((1u64 << 11) - 1));
        let mantissa = update_mantissa(exponent as i64 - BIAS, i & MANTISSA_MASK);
        f64::from_bits((exponent << 52) | mantissa)
    } else {
        let integral_part = i & ((1u64 << 56) - 1);
        integral_part as f64
    }
}

/// sign-aware <=, strictly ordering -0.0 below +0.0 (matches internal.floats).
pub(crate) fn sign_aware_lte(x: f64, y: f64) -> bool {
    if x == 0.0 && y == 0.0 {
        x.is_sign_negative() || y.is_sign_positive()
    } else {
        x <= y
    }
}

fn clamp(lower: f64, value: f64, upper: f64) -> f64 {
    if !sign_aware_lte(lower, value) {
        lower
    } else if !sign_aware_lte(value, upper) {
        upper
    } else {
        value
    }
}

pub(crate) fn permitted_float(
    f: f64,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    smallest_nonzero_magnitude: f64,
) -> bool {
    if f.is_nan() {
        return allow_nan;
    }
    if 0.0 < f.abs() && f.abs() < smallest_nonzero_magnitude {
        return false;
    }
    sign_aware_lte(min_value, f) && sign_aware_lte(f, max_value)
}

/// make_float_clamper's closure body, ported from internal.floats.make_float_clamper.
pub(crate) fn float_clamp(
    f: f64,
    min_value: f64,
    max_value: f64,
    allow_nan: bool,
    smallest_nonzero_magnitude: f64,
) -> f64 {
    if permitted_float(f, min_value, max_value, allow_nan, smallest_nonzero_magnitude) {
        return f;
    }
    let range_size = (max_value - min_value).min(f64::MAX);
    let mant = f.abs().to_bits() & MANTISSA_MASK;
    let mut f = min_value + range_size * (mant as f64 / MANTISSA_MASK as f64);
    if 0.0 < f.abs() && f.abs() < smallest_nonzero_magnitude {
        f = smallest_nonzero_magnitude;
        if smallest_nonzero_magnitude > max_value {
            f = -f;
        }
    }
    clamp(min_value, f, max_value)
}

pub(crate) const SIGNALING_NAN_BITS: u64 = 0x7FF8_0000_0000_0001;

/// IEEE-754 nextUp for width 64 (matches internal.floats.next_up).
pub(crate) fn next_up_rs(v: f64) -> f64 {
    if v.is_nan() || (v.is_infinite() && v > 0.0) {
        return v;
    }
    if v == 0.0 && v.is_sign_negative() {
        return 0.0;
    }
    let n = v.to_bits() as i64;
    let n = if n >= 0 { n + 1 } else { n - 1 };
    f64::from_bits(n as u64)
}

pub(crate) fn next_down_rs(v: f64) -> f64 {
    -next_up_rs(-v)
}

/// One IEEE-754 nextUp step at the given float `width` (toward +inf), via bit increment in
/// the width's own representation. `0.0`/`-0.0` step to the smallest positive subnormal at
/// that width. NaN passes through; +inf stays +inf.
fn ieee_next_up_at_width(v: f64, width: u32) -> f64 {
    match width {
        16 => {
            let h = half::f16::from_f64(v);
            if h.is_nan() {
                return v;
            }
            if h.is_infinite() && h.to_f64() > 0.0 {
                return f64::INFINITY;
            }
            let bits: u16 = if h.to_f64() == 0.0 { 0 } else { h.to_bits() };
            let n = bits as i16;
            let n = if n >= 0 { n.wrapping_add(1) } else { n.wrapping_sub(1) };
            half::f16::from_bits(n as u16).to_f64()
        }
        32 => {
            let f = v as f32;
            if f.is_nan() {
                return v;
            }
            if f.is_infinite() && f > 0.0 {
                return f64::INFINITY;
            }
            let bits: u32 = if f == 0.0 { 0 } else { f.to_bits() };
            let n = bits as i32;
            let n = if n >= 0 { n.wrapping_add(1) } else { n.wrapping_sub(1) };
            f32::from_bits(n as u32) as f64
        }
        _ => next_up_rs(v),
    }
}

/// Smallest value representable at `width` (16/32/64) that is strictly greater than `arg`.
/// Used to make exclude_min width-aware: a single float64 next_up rounds back to `arg` when
/// narrowed (e.g. next_up(0.0) at width 16 is below the smallest float16 subnormal), so we
/// step in the width's *own* ULP instead of float64 ULP — a handful of steps, never the
/// ~10^300 float64 increments between 0 and the smallest narrow subnormal.
pub(crate) fn next_up_width(arg: f64, width: u32) -> f64 {
    if width == 64 {
        let mut v = next_up_rs(arg);
        if v == arg {
            // signed-zero double-step: next_down(0.0) == -0.0 is still numerically equal.
            v = next_up_rs(v);
        }
        return v;
    }
    let mut cur = narrow_to_width(arg, width);
    loop {
        cur = ieee_next_up_at_width(cur, width);
        if !cur.is_finite() || cur > arg {
            return cur;
        }
    }
}

/// Largest value representable at `width` strictly less than `arg` (mirror of next_up_width).
pub(crate) fn next_down_width(arg: f64, width: u32) -> f64 {
    -next_up_width(-arg, width)
}

/// Largest finite magnitude representable in the given IEEE float width.
pub(crate) fn max_finite_for_width(width: u32) -> f64 {
    match width {
        16 => 65504.0,             // float16 max finite
        32 => f32::MAX as f64,     // ~3.4028235e38
        _ => f64::MAX,
    }
}

/// Round an f64 to the nearest value representable in the target float width
/// (round-trip through f32/f16). Non-finite values pass through unchanged.
pub(crate) fn narrow_to_width(v: f64, width: u32) -> f64 {
    if !v.is_finite() {
        return v;
    }
    match width {
        16 => half::f16::from_f64(v).to_f64(),
        32 => (v as f32) as f64,
        _ => v,
    }
}

#[pyfunction]
#[pyo3(name = "float_to_lex")]
pub(crate) fn float_to_lex(f: f64) -> u64 {
    float_to_lex_rs(f)
}

#[pyfunction]
#[pyo3(name = "lex_to_float")]
pub(crate) fn lex_to_float(i: u64) -> f64 {
    lex_to_float_rs(i)
}

#[pyfunction]
#[pyo3(name = "is_simple")]
pub(crate) fn is_simple(f: f64) -> bool {
    is_simple_rs(f)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(float_to_lex, m)?)?;
    m.add_function(wrap_pyfunction!(lex_to_float, m)?)?;
    m.add_function(wrap_pyfunction!(is_simple, m)?)?;
    Ok(())
}
