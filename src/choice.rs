//! Typed choice-sequence math, ported from hypothesis.internal.conjecture.choice.
//!
//! Everything is Rust (arbitrary precision via `num-bigint`): `choice_to_index` /
//! `choice_from_index` give each choice its complexity index (0 = simplest) for the
//! shrinker; `choice_permitted` validates; `choice_key`/`choice_equal` give
//! shrink-safe identity. String/bytes ordering goes through the Rust `IntervalSet`
//! (no per-character FFI), floats through the Rust lex kernels.

use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use pyo3::exceptions::{PyKeyError, PyNotImplementedError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::intervalset::IntervalSet;

const BUFFER_SIZE: u64 = 8 * 1024;

// ---- dict helpers -----------------------------------------------------------

fn dget<'py>(c: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    match c.get_item(key)? {
        Some(v) => Ok(v),
        None => Err(PyKeyError::new_err(key.to_string())),
    }
}

fn opt_bigint(c: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<BigInt>> {
    match c.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn req_bigint(c: &Bound<'_, PyDict>, key: &str) -> PyResult<BigInt> {
    dget(c, key)?.extract()
}

fn req_u64(c: &Bound<'_, PyDict>, key: &str) -> PyResult<u64> {
    dget(c, key)?.extract()
}

fn req_f64(c: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    dget(c, key)?.extract()
}

fn req_bool(c: &Bound<'_, PyDict>, key: &str) -> PyResult<bool> {
    dget(c, key)?.extract()
}

fn choice_too_large(py: Python<'_>) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py
            .import("hypothesis_fast.errors")?
            .getattr("ChoiceTooLarge")?;
        Ok(PyErr::from_value(cls.call0()?))
    };
    build().unwrap_or_else(|e| e)
}

// ---- zigzag -----------------------------------------------------------------

fn zigzag_index_bi(value: &BigInt, shrink_towards: &BigInt) -> BigInt {
    let mut index = (shrink_towards - value).abs() * BigInt::from(2u8);
    if value > shrink_towards {
        index -= BigInt::one();
    }
    index
}

fn zigzag_value_bi(index: &BigInt, shrink_towards: &BigInt) -> BigInt {
    let two = BigInt::from(2u8);
    let n = (index + BigInt::one()) / &two;
    let n = if (index % &two).is_zero() { -n } else { n };
    shrink_towards + n
}

// ---- collection index/value -------------------------------------------------

fn size_to_index(size: u64, alphabet_size: u64) -> BigInt {
    if alphabet_size == 0 {
        return BigInt::zero();
    }
    if alphabet_size == 1 {
        return BigInt::from(size);
    }
    let a = BigInt::from(alphabet_size);
    (a.pow(size as u32) - BigInt::one()) / (BigInt::from(alphabet_size) - BigInt::one())
}

fn index_to_size(index: &BigInt, alphabet_size: u64) -> u64 {
    if alphabet_size == 0 {
        return 0;
    }
    if alphabet_size == 1 {
        return index.to_u64().unwrap_or(u64::MAX);
    }
    let a = BigInt::from(alphabet_size);
    let mut total = index * BigInt::from(alphabet_size - 1) + BigInt::one();
    let mut s = 0u64;
    while total >= a {
        total /= &a;
        s += 1;
    }
    s
}

fn collection_index_from_orders(orders: &[BigInt], min_size: u64, alphabet_size: u64) -> BigInt {
    let len = orders.len() as u64;
    let mut index =
        size_to_index(len, alphabet_size) - size_to_index(min_size, alphabet_size);
    let a = BigInt::from(alphabet_size);
    let mut running_exp = BigInt::one();
    for c in orders.iter().rev() {
        index += &running_exp * c;
        running_exp *= &a;
    }
    index
}

fn collection_value_to_orders(
    py: Python<'_>,
    index: BigInt,
    min_size: u64,
    alphabet_size: u64,
) -> PyResult<Vec<BigInt>> {
    let mut index = index + size_to_index(min_size, alphabet_size);
    let size = index_to_size(&index, alphabet_size);
    if size >= BUFFER_SIZE {
        return Err(choice_too_large(py));
    }
    index -= size_to_index(size, alphabet_size);
    let a = BigInt::from(alphabet_size);
    let mut vals: Vec<BigInt> = Vec::with_capacity(size as usize);
    for i in (0..size).rev() {
        let n = if index.is_zero() {
            BigInt::zero()
        } else {
            let p = a.pow(i as u32);
            let n = &index / &p;
            index -= &n * &p;
            n
        };
        vals.push(n);
    }
    Ok(vals)
}

// ---- integer to/from index --------------------------------------------------

fn clamp_shrink_towards(
    shrink_towards: &BigInt,
    min_value: &Option<BigInt>,
    max_value: &Option<BigInt>,
) -> BigInt {
    let mut st = shrink_towards.clone();
    if let Some(mn) = min_value {
        if *mn > st {
            st = mn.clone();
        }
    }
    if let Some(mx) = max_value {
        if *mx < st {
            st = mx.clone();
        }
    }
    st
}

fn int_to_index(
    choice: &BigInt,
    min_value: &Option<BigInt>,
    max_value: &Option<BigInt>,
    shrink_towards: &BigInt,
) -> BigInt {
    let st = clamp_shrink_towards(shrink_towards, min_value, max_value);
    match (min_value, max_value) {
        (None, None) => zigzag_index_bi(choice, &st),
        (Some(mn), None) => {
            if (choice - &st).abs() <= (&st - mn) {
                zigzag_index_bi(choice, &st)
            } else {
                choice - mn
            }
        }
        (None, Some(mx)) => {
            if (choice - &st).abs() <= (mx - &st) {
                zigzag_index_bi(choice, &st)
            } else {
                mx - choice
            }
        }
        (Some(mn), Some(mx)) => {
            if (&st - mn) < (mx - &st) {
                if (choice - &st).abs() <= (&st - mn) {
                    zigzag_index_bi(choice, &st)
                } else {
                    choice - mn
                }
            } else if (choice - &st).abs() <= (mx - &st) {
                zigzag_index_bi(choice, &st)
            } else {
                mx - choice
            }
        }
    }
}

fn int_from_index(
    index: &BigInt,
    min_value: &Option<BigInt>,
    max_value: &Option<BigInt>,
    shrink_towards: &BigInt,
) -> BigInt {
    let st = clamp_shrink_towards(shrink_towards, min_value, max_value);
    match (min_value, max_value) {
        (None, None) => zigzag_value_bi(index, &st),
        (Some(mn), None) => {
            let zz = zigzag_index_bi(mn, &st);
            if index <= &zz {
                zigzag_value_bi(index, &st)
            } else {
                index + mn
            }
        }
        (None, Some(mx)) => {
            let zz = zigzag_index_bi(mx, &st);
            if index <= &zz {
                zigzag_value_bi(index, &st)
            } else {
                mx - index
            }
        }
        (Some(mn), Some(mx)) => {
            if (&st - mn) < (mx - &st) {
                let zz = zigzag_index_bi(mn, &st);
                if index <= &zz {
                    zigzag_value_bi(index, &st)
                } else {
                    index + mn
                }
            } else {
                let zz = zigzag_index_bi(mx, &st);
                if index <= &zz {
                    zigzag_value_bi(index, &st)
                } else {
                    mx - index
                }
            }
        }
    }
}

// ---- string helpers ---------------------------------------------------------

pub(crate) fn cps_to_pystr(py: Python<'_>, cps: &[i64]) -> PyResult<Py<PyAny>> {
    let mut s = String::with_capacity(cps.len());
    let mut ok = true;
    for &cp in cps {
        match u32::try_from(cp).ok().and_then(char::from_u32) {
            Some(c) => s.push(c),
            None => {
                ok = false;
                break;
            }
        }
    }
    if ok {
        return Ok(PyString::new(py, &s).into_any().unbind());
    }
    // surrogate path: "".join(chr(cp) ...) via Python (flexible str)
    let chr = py.import("builtins")?.getattr("chr")?;
    let parts = PyList::empty(py);
    for &cp in cps {
        parts.append(chr.call1((cp,))?)?;
    }
    Ok(PyString::new(py, "")
        .call_method1("join", (parts,))?
        .into_any()
        .unbind())
}

// ---- public pyfunctions -----------------------------------------------------

#[pyfunction]
#[pyo3(name = "zigzag_index", signature = (value, *, shrink_towards))]
pub(crate) fn zigzag_index(value: BigInt, shrink_towards: BigInt) -> BigInt {
    zigzag_index_bi(&value, &shrink_towards)
}

#[pyfunction]
#[pyo3(name = "zigzag_value", signature = (index, *, shrink_towards))]
pub(crate) fn zigzag_value(index: BigInt, shrink_towards: BigInt) -> BigInt {
    zigzag_value_bi(&index, &shrink_towards)
}

#[pyfunction]
#[pyo3(name = "choice_to_index")]
pub(crate) fn choice_to_index(
    choice: &Bound<'_, PyAny>,
    constraints: &Bound<'_, PyDict>,
) -> PyResult<BigInt> {
    if choice.is_instance_of::<PyBool>() {
        let b: bool = choice.extract()?;
        let p: f64 = req_f64(constraints, "p")?;
        if !(2f64.powi(-64) < p && p < 1.0 - 2f64.powi(-64)) {
            return Ok(BigInt::zero());
        }
        return Ok(BigInt::from(b as u8));
    }
    if choice.is_instance_of::<PyInt>() {
        let v: BigInt = choice.extract()?;
        let min_value = opt_bigint(constraints, "min_value")?;
        let max_value = opt_bigint(constraints, "max_value")?;
        let shrink_towards = req_bigint(constraints, "shrink_towards")?;
        return Ok(int_to_index(&v, &min_value, &max_value, &shrink_towards));
    }
    if choice.is_instance_of::<PyBytes>() {
        let b = choice.downcast::<PyBytes>()?.as_bytes().to_vec();
        let min_size = req_u64(constraints, "min_size")?;
        let orders: Vec<BigInt> = b.iter().map(|&x| BigInt::from(x)).collect();
        return Ok(collection_index_from_orders(&orders, min_size, 256));
    }
    if choice.is_instance_of::<PyString>() {
        let s: String = choice.extract()?;
        let min_size = req_u64(constraints, "min_size")?;
        let isb = dget(constraints, "intervals")?;
        let iset = isb.downcast::<IntervalSet>()?.borrow();
        let alpha = iset.alphabet_len() as u64;
        let mut orders: Vec<BigInt> = Vec::with_capacity(s.chars().count());
        for ch in s.chars() {
            orders.push(BigInt::from(iset.shrink_order_index_of_cp(ch as i64)?));
        }
        return Ok(collection_index_from_orders(&orders, min_size, alpha));
    }
    if choice.is_instance_of::<PyFloat>() {
        let f: f64 = choice.extract()?;
        let sign: u64 = if f.is_sign_negative() { 1 } else { 0 };
        let lex = crate::floats::float_to_lex_rs(f.abs());
        return Ok((BigInt::from(sign) << 64usize) | BigInt::from(lex));
    }
    Err(PyNotImplementedError::new_err("unhandled choice type"))
}

#[pyfunction]
#[pyo3(name = "choice_from_index")]
pub(crate) fn choice_from_index(
    py: Python<'_>,
    index: BigInt,
    choice_type: &str,
    constraints: &Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    match choice_type {
        "integer" => {
            let min_value = opt_bigint(constraints, "min_value")?;
            let max_value = opt_bigint(constraints, "max_value")?;
            let shrink_towards = req_bigint(constraints, "shrink_towards")?;
            let v = int_from_index(&index, &min_value, &max_value, &shrink_towards);
            Ok(v.into_pyobject(py)?.into_any().unbind())
        }
        "boolean" => {
            let p: f64 = req_f64(constraints, "p")?;
            let val = if p <= 2f64.powi(-64) {
                false
            } else if p >= 1.0 - 2f64.powi(-64) {
                true
            } else {
                !index.is_zero()
            };
            Ok(val.into_pyobject(py)?.to_owned().into_any().unbind())
        }
        "bytes" => {
            let min_size = req_u64(constraints, "min_size")?;
            let orders = collection_value_to_orders(py, index, min_size, 256)?;
            let bytes_vec: Vec<u8> = orders.iter().map(|n| n.to_u8().unwrap_or(0)).collect();
            Ok(PyBytes::new(py, &bytes_vec).into_any().unbind())
        }
        "string" => {
            let min_size = req_u64(constraints, "min_size")?;
            let isb = dget(constraints, "intervals")?;
            let iset = isb.downcast::<IntervalSet>()?.borrow();
            let alpha = iset.alphabet_len() as u64;
            let orders = collection_value_to_orders(py, index, min_size, alpha)?;
            let mut cps: Vec<i64> = Vec::with_capacity(orders.len());
            for n in &orders {
                cps.push(iset.cp_in_shrink_order(n.to_i64().unwrap_or(0))?);
            }
            cps_to_pystr(py, &cps)
        }
        "float" => {
            let min_value = req_f64(constraints, "min_value")?;
            let max_value = req_f64(constraints, "max_value")?;
            let allow_nan = req_bool(constraints, "allow_nan")?;
            let snm = req_f64(constraints, "smallest_nonzero_magnitude")?;
            let sign = if (&index >> 64usize) != BigInt::zero() {
                -1.0
            } else {
                1.0
            };
            let mask = (BigInt::one() << 64usize) - BigInt::one();
            let low = (&index & &mask).to_u64().unwrap_or(0);
            let result = sign * crate::floats::lex_to_float_rs(low);
            let clamped =
                crate::floats::float_clamp(result, min_value, max_value, allow_nan, snm);
            Ok(PyFloat::new(py, clamped).into_any().unbind())
        }
        other => Err(PyNotImplementedError::new_err(format!(
            "unhandled choice_type {other}"
        ))),
    }
}

#[pyfunction]
#[pyo3(name = "choice_permitted")]
pub(crate) fn choice_permitted(
    choice: &Bound<'_, PyAny>,
    constraints: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    if choice.is_instance_of::<PyBool>() {
        let b: bool = choice.extract()?;
        let p: f64 = req_f64(constraints, "p")?;
        if p <= 0.0 {
            return Ok(!b);
        }
        if p >= 1.0 {
            return Ok(b);
        }
        return Ok(true);
    }
    if choice.is_instance_of::<PyInt>() {
        let v: BigInt = choice.extract()?;
        if let Some(mn) = opt_bigint(constraints, "min_value")? {
            if v < mn {
                return Ok(false);
            }
        }
        if let Some(mx) = opt_bigint(constraints, "max_value")? {
            if v > mx {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if choice.is_instance_of::<PyFloat>() {
        let f: f64 = choice.extract()?;
        return Ok(crate::floats::permitted_float(
            f,
            req_f64(constraints, "min_value")?,
            req_f64(constraints, "max_value")?,
            req_bool(constraints, "allow_nan")?,
            req_f64(constraints, "smallest_nonzero_magnitude")?,
        ));
    }
    if choice.is_instance_of::<PyString>() {
        let s: String = choice.extract()?;
        let len = s.chars().count() as u64;
        if len < req_u64(constraints, "min_size")? || len > req_u64(constraints, "max_size")? {
            return Ok(false);
        }
        let isb = dget(constraints, "intervals")?;
        let iset = isb.downcast::<IntervalSet>()?.borrow();
        for ch in s.chars() {
            if !iset.contains_cp(ch as i64) {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    if choice.is_instance_of::<PyBytes>() {
        let len = choice.downcast::<PyBytes>()?.len()? as u64;
        return Ok(
            len >= req_u64(constraints, "min_size")? && len <= req_u64(constraints, "max_size")?,
        );
    }
    Err(PyNotImplementedError::new_err("unhandled choice type"))
}

#[pyfunction]
#[pyo3(name = "choice_key")]
pub(crate) fn choice_key(py: Python<'_>, choice: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if choice.is_instance_of::<PyBool>() {
        let b: bool = choice.extract()?;
        let elems = vec![
            PyString::new(py, "bool").into_any(),
            b.into_pyobject(py)?.to_owned().into_any(),
        ];
        return Ok(PyTuple::new(py, elems)?.into_any().unbind());
    }
    if choice.is_instance_of::<PyFloat>() {
        let f: f64 = choice.extract()?;
        let bits = BigInt::from(f.to_bits());
        let elems = vec![
            PyString::new(py, "float").into_any(),
            bits.into_pyobject(py)?.into_any(),
        ];
        return Ok(PyTuple::new(py, elems)?.into_any().unbind());
    }
    Ok(choice.clone().unbind())
}

#[pyfunction]
#[pyo3(name = "choices_key")]
pub(crate) fn choices_key(py: Python<'_>, choices: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut keys: Vec<Bound<'_, PyAny>> = Vec::new();
    for it in choices.try_iter()? {
        keys.push(choice_key(py, &it?)?.into_bound(py));
    }
    Ok(PyTuple::new(py, keys)?.into_any().unbind())
}

#[pyfunction]
#[pyo3(name = "choice_equal")]
pub(crate) fn choice_equal(
    py: Python<'_>,
    choice1: &Bound<'_, PyAny>,
    choice2: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let k1 = choice_key(py, choice1)?;
    let k2 = choice_key(py, choice2)?;
    k1.bind(py).eq(k2.bind(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(zigzag_index, m)?)?;
    m.add_function(wrap_pyfunction!(zigzag_value, m)?)?;
    m.add_function(wrap_pyfunction!(choice_to_index, m)?)?;
    m.add_function(wrap_pyfunction!(choice_from_index, m)?)?;
    m.add_function(wrap_pyfunction!(choice_permitted, m)?)?;
    m.add_function(wrap_pyfunction!(choice_key, m)?)?;
    m.add_function(wrap_pyfunction!(choices_key, m)?)?;
    m.add_function(wrap_pyfunction!(choice_equal, m)?)?;
    Ok(())
}
