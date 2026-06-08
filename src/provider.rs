//! HypothesisProvider sampling helpers, ported from
//! hypothesis.internal.conjecture.providers.HypothesisProvider.
//!
//! Pure sampling over a Rust RNG: these decide *what value* a draw produces.
//! `ConjectureData` (data.rs) calls these, then records the typed choice. The
//! constant-injection pool (`_maybe_draw_constant`) is intentionally deferred —
//! generation is otherwise faithful (heavy-tail integers, weird-float upweight,
//! geometric collection sizes).

use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use rand::rngs::StdRng;
use rand::Rng;

use crate::statistics::integers_distribution;

fn two_pow(n: u32) -> BigInt {
    BigInt::from(1u8) << (n as usize)
}

/// Uniform integer in [0, range] inclusive (range >= 0), via rejection sampling.
fn uniform_bigint(rng: &mut StdRng, range: &BigInt) -> BigInt {
    // range == 0 is a single point; range < 0 means an inverted/degenerate span (lo > hi)
    // reached us anyway — return 0 so `uniform_int` yields `lo` instead of rejection-looping
    // forever (acc >= 0 can never be <= a negative range). The engine must NEVER spin in
    // native code holding the GIL: a per-test timeout can't interrupt it, so it would wedge
    // the whole worker. Callers that can produce an inverted span validate up front; this is
    // the last-line guarantee that no draw path can hang.
    if !range.is_positive() {
        return BigInt::zero();
    }
    let bits = range.bits(); // number of bits in range
    let nwords = bits.div_ceil(64);
    let mask = (BigInt::from(1u8) << (bits as usize)) - BigInt::from(1u8);
    loop {
        let mut acc = BigInt::zero();
        for _ in 0..nwords {
            let w: u64 = rng.gen();
            acc = (acc << 64) | BigInt::from(w);
        }
        acc &= &mask;
        if &acc <= range {
            return acc;
        }
    }
}

/// Uniform integer in [lo, hi] inclusive.
pub(crate) fn uniform_int(rng: &mut StdRng, lo: &BigInt, hi: &BigInt) -> BigInt {
    let range = hi - lo;
    lo + uniform_bigint(rng, &range)
}

fn clamp_bigint(lo: &BigInt, v: BigInt, hi: &BigInt) -> BigInt {
    if &v < lo {
        lo.clone()
    } else if &v > hi {
        hi.clone()
    } else {
        v
    }
}

/// Port of HypothesisProvider._draw_integer_from_distribution.
pub(crate) fn draw_integer_from_distribution(
    rng: &mut StdRng,
    min_value: Option<&BigInt>,
    max_value: Option<&BigInt>,
) -> BigInt {
    // Resolve to a concrete bounded range, matching upstream's adjustments. `two_pow(128)` is a
    // heap BigInt; compute it only inside the unbounded arms that actually need it — the common
    // (Some, Some) bounded case (collection sizes, charset indices, integers(a, b)) must not pay
    // that allocation on every draw. Value is identical; the RNG stream is untouched.
    let (min_v, max_v): (BigInt, BigInt) = match (min_value, max_value) {
        (None, None) => {
            let p128 = two_pow(128);
            (-(&p128), p128)
        }
        (None, Some(mx)) => {
            let lo = -std::cmp::max(two_pow(128), BigInt::from(2u8) * mx.abs());
            (lo, mx.clone())
        }
        (Some(mn), None) => {
            let hi = std::cmp::max(two_pow(128), BigInt::from(2u8) * mn.abs());
            (mn.clone(), hi)
        }
        (Some(mn), Some(mx)) => (mn.clone(), mx.clone()),
    };

    let dist = integers_distribution();
    let min_f = min_v.to_f64();
    let max_f = max_v.to_f64();
    let mut safe_bounds = true;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);

    match (min_f, max_f) {
        (Some(mnf), Some(mxf)) if mnf.is_finite() && mxf.is_finite() => {
            lo = dist.cdf(mnf - 0.5);
            hi = dist.cdf(mxf + 0.5);
            if hi - lo < 1e-13 {
                safe_bounds = false;
            }
        }
        _ => safe_bounds = false,
    }

    if safe_bounds {
        let mut p;
        loop {
            p = lo + rng.gen::<f64>() * (hi - lo);
            if p != 0.0 && p != 1.0 {
                break;
            }
        }
        let n = dist.inverse_cdf(p).round();
        if let Some(n_big) = BigInt::from_f64_round(n) {
            return clamp_bigint(&min_v, n_big, &max_v);
        }
        // fall through to uniform if conversion failed
    }

    uniform_int(rng, &min_v, &max_v)
}

/// `lex_to_float(getrandbits(64)) * sign` — the base float draw.
pub(crate) fn draw_float_raw(rng: &mut StdRng) -> f64 {
    let bits: u64 = rng.gen();
    let f = crate::floats::lex_to_float_rs(bits);
    if rng.gen::<bool>() {
        f
    } else {
        -f
    }
}

/// p_continue for a geometric collection-size loop targeting `average_size`.
/// Port of conjecture.utils._calc_p_continue / _p_continue_to_avg.
pub(crate) fn calc_p_continue(desired_avg: f64, max_size: f64) -> f64 {
    if desired_avg >= max_size {
        return 1.0;
    }
    let mut p_continue = 1.0 - 1.0 / (1.0 + desired_avg);
    if p_continue == 0.0 || max_size.is_infinite() {
        return p_continue;
    }
    while p_continue_to_avg(p_continue, max_size) > desired_avg {
        p_continue -= 0.0001;
        if p_continue < f64::MIN_POSITIVE {
            return f64::MIN_POSITIVE;
        }
    }
    let mut hi = 1.0;
    while desired_avg - p_continue_to_avg(p_continue, max_size) > 0.01 {
        let mid = (p_continue + hi) / 2.0;
        if p_continue_to_avg(mid, max_size) <= desired_avg {
            p_continue = mid;
        } else {
            hi = mid;
        }
    }
    p_continue
}

fn p_continue_to_avg(p_continue: f64, max_size: f64) -> f64 {
    if p_continue >= 1.0 {
        return max_size;
    }
    (1.0 / (1.0 - p_continue) - 1.0) * (1.0 - p_continue.powf(max_size))
}

/// Draw a collection length in [min_size, max_size] via the geometric loop.
pub(crate) fn draw_collection_size(
    rng: &mut StdRng,
    min_size: usize,
    max_size: usize,
) -> usize {
    if min_size == max_size {
        return min_size;
    }
    let avg = ((min_size as f64 * 2.0).max(min_size as f64 + 5.0))
        .min(0.5 * (min_size as f64 + max_size as f64));
    let p = calc_p_continue(avg - min_size as f64, (max_size - min_size) as f64);
    let mut count = 0usize;
    loop {
        let should_continue = if count < min_size {
            true
        } else if count >= max_size {
            false
        } else {
            rng.gen::<f64>() < p
        };
        if should_continue {
            count += 1;
        } else {
            return count;
        }
    }
}

trait FromF64Round: Sized {
    fn from_f64_round(f: f64) -> Option<Self>;
}

impl FromF64Round for BigInt {
    fn from_f64_round(f: f64) -> Option<BigInt> {
        if !f.is_finite() {
            return None;
        }
        // f is already rounded by the caller; convert exactly.
        num_traits::cast::FromPrimitive::from_f64(f)
    }
}
