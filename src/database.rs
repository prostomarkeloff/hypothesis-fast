//! Choice-sequence (de)serialization, ported from hypothesis.database.
//!
//! Custom flat format (see upstream): each choice = metadata byte `tag_ssss`
//! (+ uint ULEB128 size if size>=31) + payload. Booleans inline the payload.
//! Used by the database/reproduce subsystems and for exact choice-size accounting
//! in ConjectureData (replacing the earlier approximation).

use num_bigint::BigInt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyTuple};
use std::collections::{BTreeSet, HashMap};

fn pack_uleb128(mut value: usize, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn unpack_uleb128(buffer: &[u8]) -> (usize, usize) {
    let mut value: usize = 0;
    let mut i = 0usize;
    for (idx, &byte) in buffer.iter().enumerate() {
        let n = (byte & 0x7f) as usize;
        value |= n << (idx * 7);
        i = idx;
        if byte >> 7 == 0 {
            break;
        }
    }
    (i + 1, value)
}

fn encode_choice(py: Python<'_>, choice: &Bound<'_, PyAny>, out: &mut Vec<u8>) -> PyResult<()> {
    use pyo3::types::{PyBool, PyInt, PyString};

    if choice.is_instance_of::<PyBool>() {
        let b: bool = choice.extract()?;
        out.push(if b { 1 } else { 0 });
        return Ok(());
    }

    let (tag, payload): (u8, Vec<u8>) = if choice.is_instance_of::<PyFloat>() {
        let f: f64 = choice.extract()?;
        (1 << 5, f.to_be_bytes().to_vec())
    } else if choice.is_instance_of::<PyInt>() {
        let b: BigInt = choice.extract()?;
        let nbytes = 1 + (b.bits() / 8) as usize;
        let kwargs = PyDict::new(py);
        kwargs.set_item("signed", true)?;
        let payload_obj = choice.call_method("to_bytes", (nbytes, "big"), Some(&kwargs))?;
        (2 << 5, payload_obj.downcast::<PyBytes>()?.as_bytes().to_vec())
    } else if choice.is_instance_of::<PyBytes>() {
        (3 << 5, choice.downcast::<PyBytes>()?.as_bytes().to_vec())
    } else if choice.is_instance_of::<PyString>() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("errors", "surrogatepass")?;
        let enc = choice.call_method("encode", ("utf-8",), Some(&kwargs))?;
        (4 << 5, enc.downcast::<PyBytes>()?.as_bytes().to_vec())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("unhandled choice type"));
    };

    let size = payload.len();
    if size < 0b11111 {
        out.push(tag | size as u8);
    } else {
        out.push(tag | 0b11111);
        pack_uleb128(size, out);
    }
    out.extend_from_slice(&payload);
    Ok(())
}

/// Number of bytes in the ULEB128 encoding of `n`.
fn uleb128_len(n: usize) -> usize {
    let mut len = 1;
    let mut v = n >> 7;
    while v > 0 {
        len += 1;
        v >>= 7;
    }
    len
}

/// Byte length of a single choice's serialization (for buffer-size accounting on the
/// hot draw path). Computed DIRECTLY from the value — no full serialization, no Python
/// `int.to_bytes()` round-trip, no Vec alloc — mirroring `encode_choice`'s sizes exactly.
pub(crate) fn one_choice_size(py: Python<'_>, choice: &Bound<'_, PyAny>) -> PyResult<usize> {
    use pyo3::types::{PyBool, PyInt, PyString};

    if choice.is_instance_of::<PyBool>() {
        return Ok(1); // bool: a single inlined tag byte, no payload
    }
    let payload_len: usize = if choice.is_instance_of::<PyFloat>() {
        8 // always 8 big-endian bytes
    } else if choice.is_instance_of::<PyInt>() {
        // magnitude bit length; fast path via i64 avoids _PyLong_AsByteArray (the slow
        // Python-int -> BigInt conversion). BigInt only for values that don't fit i64.
        let bits: u64 = match choice.extract::<i64>() {
            Ok(v) => {
                let m = v.unsigned_abs();
                if m == 0 { 0 } else { 64 - m.leading_zeros() as u64 }
            }
            Err(_) => choice.extract::<BigInt>()?.bits(),
        };
        1 + (bits / 8) as usize // matches encode_choice's nbytes
    } else if choice.is_instance_of::<PyBytes>() {
        choice.downcast::<PyBytes>()?.len()?
    } else if choice.is_instance_of::<PyString>() {
        // UTF-8 (surrogatepass) length: must encode, but we only take the length (no copy).
        let kwargs = PyDict::new(py);
        kwargs.set_item("errors", "surrogatepass")?;
        choice
            .call_method("encode", ("utf-8",), Some(&kwargs))?
            .downcast::<PyBytes>()?
            .len()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("unhandled choice type"));
    };
    // 1 tag byte + (ULEB128 size suffix when the payload is >= 31) + payload.
    Ok(1 + if payload_len < 0b11111 { 0 } else { uleb128_len(payload_len) } + payload_len)
}

#[pyfunction]
#[pyo3(name = "choices_to_bytes")]
fn choices_to_bytes(py: Python<'_>, choices: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut out = Vec::new();
    for it in choices.try_iter()? {
        encode_choice(py, &it?, &mut out)?;
    }
    Ok(PyBytes::new(py, &out).into_any().unbind())
}

/// Serialize a choice sequence to the flat custom format (engine DB-save path).
pub(crate) fn encode_choices(py: Python<'_>, choices: &[Py<PyAny>]) -> PyResult<Vec<u8>> {
    let mut out = Vec::new();
    for c in choices {
        encode_choice(py, c.bind(py), &mut out)?;
    }
    Ok(out)
}

pub(crate) fn decode(py: Python<'_>, buffer: &[u8]) -> PyResult<Vec<Py<PyAny>>> {
    let mut parts: Vec<Py<PyAny>> = Vec::new();
    let mut idx = 0usize;
    while idx < buffer.len() {
        let tag = buffer[idx] >> 5;
        let mut size = (buffer[idx] & 0b11111) as usize;
        idx += 1;
        if tag == 0 {
            parts.push((size != 0).into_pyobject(py)?.to_owned().into_any().unbind());
            continue;
        }
        if size == 0b11111 {
            let (offset, real) = unpack_uleb128(&buffer[idx..]);
            idx += offset;
            size = real;
        }
        if idx + size > buffer.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("truncated"));
        }
        let chunk = &buffer[idx..idx + size];
        idx += size;
        match tag {
            1 => {
                if size != 8 {
                    return Err(pyo3::exceptions::PyValueError::new_err("expected float64"));
                }
                let mut b = [0u8; 8];
                b.copy_from_slice(chunk);
                parts.push(PyFloat::new(py, f64::from_be_bytes(b)).into_any().unbind());
            }
            2 => {
                let v = BigInt::from_signed_bytes_be(chunk);
                parts.push(v.into_pyobject(py)?.into_any().unbind());
            }
            3 => {
                parts.push(PyBytes::new(py, chunk).into_any().unbind());
            }
            4 => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("errors", "surrogatepass")?;
                let s = PyBytes::new(py, chunk).call_method("decode", ("utf-8",), Some(&kwargs))?;
                parts.push(s.unbind());
            }
            _ => return Err(pyo3::exceptions::PyValueError::new_err("bad tag")),
        }
    }
    Ok(parts)
}

#[pyfunction]
#[pyo3(name = "choices_from_bytes")]
fn choices_from_bytes(py: Python<'_>, buffer: &[u8]) -> PyResult<Py<PyAny>> {
    match decode(py, buffer) {
        Ok(parts) => Ok(PyTuple::new(py, parts)?.into_any().unbind()),
        Err(_) => Ok(py.None()),
    }
}

/// In-memory ExampleDatabase, ported from hypothesis.database.InMemoryExampleDatabase.
#[pyclass(module = "hypothesis_fast._engine")]
pub(crate) struct InMemoryExampleDatabase {
    data: HashMap<Vec<u8>, BTreeSet<Vec<u8>>>,
}

#[pymethods]
impl InMemoryExampleDatabase {
    #[new]
    fn new() -> Self {
        InMemoryExampleDatabase { data: HashMap::new() }
    }

    fn save(&mut self, key: &[u8], value: &[u8]) {
        self.data.entry(key.to_vec()).or_default().insert(value.to_vec());
    }

    fn delete(&mut self, key: &[u8], value: &[u8]) {
        if let Some(s) = self.data.get_mut(key) {
            s.remove(value);
        }
    }

    fn fetch<'py>(&self, py: Python<'py>, key: &[u8]) -> Vec<Bound<'py, PyBytes>> {
        self.data
            .get(key)
            .map(|s| s.iter().map(|v| PyBytes::new(py, v)).collect())
            .unwrap_or_default()
    }

    #[pyo3(name = "move")]
    fn move_value(&mut self, src: &[u8], dest: &[u8], value: &[u8]) {
        if src == dest {
            self.save(dest, value);
            return;
        }
        self.delete(src, value);
        self.save(dest, value);
    }

    fn __repr__(&self) -> String {
        "InMemoryExampleDatabase()".to_string()
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(choices_to_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(choices_from_bytes, m)?)?;
    m.add_class::<InMemoryExampleDatabase>()?;
    Ok(())
}
