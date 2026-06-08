//! IntervalSet, ported from hypothesis.internal.intervalsets.
//!
//! A compact set of inclusive `(a, b)` codepoint intervals, treated like a set of
//! integers with O(log n) indexing. Used on the string generation/shrinking hot
//! path (each character is an index into the alphabet's IntervalSet).

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const ASCII_ZERO: i64 = 0x30; // ord('0')
const ASCII_Z: i64 = 0x5A; // ord('Z')

fn index_above_in(intervals: &[(i64, i64)], offsets: &[i64], size: i64, value: i64) -> i64 {
    for (idx, &(u, v)) in intervals.iter().enumerate() {
        let offset = offsets[idx];
        if u >= value {
            return offset;
        }
        if value <= v {
            return offset + (value - u);
        }
    }
    size
}

#[pyclass(module = "hypothesis_fast._engine", frozen)]
pub(crate) struct IntervalSet {
    intervals: Vec<(i64, i64)>,
    offsets: Vec<i64>,
    #[pyo3(get)]
    size: i64,
    idx_of_zero: i64,
    idx_of_z: i64,
}

impl IntervalSet {
    fn build(intervals: Vec<(i64, i64)>) -> Self {
        let mut offsets = Vec::with_capacity(intervals.len());
        let mut acc: i64 = 0;
        for &(u, v) in &intervals {
            offsets.push(acc);
            acc += v - u + 1;
        }
        let size = acc;
        let idx_of_zero = index_above_in(&intervals, &offsets, size, ASCII_ZERO);
        let idx_of_z = index_above_in(&intervals, &offsets, size, ASCII_Z).min((size - 1).max(0));
        IntervalSet {
            intervals,
            offsets,
            size,
            idx_of_zero,
            idx_of_z,
        }
    }

    pub(crate) fn alphabet_len(&self) -> i64 {
        self.size
    }

    pub(crate) fn contains_cp(&self, cp: i64) -> bool {
        self.intervals.iter().any(|&(a, b)| a <= cp && cp <= b)
    }

    /// `index(value)` ported for Rust callers (codepoint -> ordinal position).
    pub(crate) fn ordinal_of(&self, value: i64) -> PyResult<i64> {
        for (idx, &(u, v)) in self.intervals.iter().enumerate() {
            let offset = self.offsets[idx];
            if u == value {
                return Ok(offset);
            } else if u > value {
                return Err(PyValueError::new_err(format!("{value} is not in list")));
            }
            if value <= v {
                return Ok(offset + (value - u));
            }
        }
        Err(PyValueError::new_err(format!("{value} is not in list")))
    }

    /// `index_from_char_in_shrink_order` on a codepoint (no Python str needed).
    pub(crate) fn shrink_order_index_of_cp(&self, cp: i64) -> PyResult<i64> {
        let mut i = self.ordinal_of(cp)?;
        if i <= self.idx_of_z {
            let n = self.idx_of_z - self.idx_of_zero;
            if self.idx_of_zero <= i && i <= self.idx_of_z {
                i -= self.idx_of_zero;
            } else {
                i = self.idx_of_zero - i + n;
            }
        }
        Ok(i)
    }

    /// `char_in_shrink_order` returning a codepoint (no Python str built).
    pub(crate) fn cp_in_shrink_order(&self, i: i64) -> PyResult<i64> {
        let mut i = i;
        if i <= self.idx_of_z {
            let n = self.idx_of_z - self.idx_of_zero;
            if i <= n {
                i += self.idx_of_zero;
            } else {
                i = self.idx_of_zero - (i - n);
            }
        }
        self.getitem_rs(i)
    }

    fn getitem_rs(&self, i: i64) -> PyResult<i64> {
        let mut i = i;
        if i < 0 {
            i += self.size;
        }
        if i < 0 || i >= self.size {
            return Err(PyIndexError::new_err(format!(
                "Invalid index {i} for [0, {})",
                self.size
            )));
        }
        let mut j = self.intervals.len() - 1;
        if self.offsets[j] > i {
            let mut hi = j;
            let mut lo = 0usize;
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if self.offsets[mid] <= i {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            j = lo;
        }
        let t = i - self.offsets[j];
        let (u, _v) = self.intervals[j];
        Ok(u + t)
    }
}

#[pymethods]
impl IntervalSet {
    #[new]
    #[pyo3(signature = (intervals=None))]
    fn new(intervals: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut out: Vec<(i64, i64)> = Vec::new();
        if let Some(obj) = intervals {
            for item in obj.try_iter()? {
                let item = item?;
                let a: i64 = item.get_item(0)?.extract()?;
                let b: i64 = item.get_item(1)?.extract()?;
                // mirror the upstream `assert all(len(v) == 2 ...)`
                if item.len()? != 2 {
                    return Err(PyValueError::new_err("intervals must be 2-tuples"));
                }
                out.push((a, b));
            }
        }
        Ok(IntervalSet::build(out))
    }

    #[classmethod]
    fn from_string(_cls: &Bound<'_, PyType>, s: &str) -> Self {
        let mut cps: Vec<i64> = s.chars().map(|c| c as i64).collect();
        cps.sort_unstable();
        let pairs: Vec<(i64, i64)> = cps.iter().map(|&c| (c, c)).collect();
        let x = IntervalSet::build(pairs);
        x.union_rs(&x)
    }

    #[getter]
    fn intervals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let elems: Vec<Bound<'py, PyTuple>> = self
            .intervals
            .iter()
            .map(|&(a, b)| PyTuple::new(py, [a, b]))
            .collect::<PyResult<_>>()?;
        PyTuple::new(py, elems)
    }

    fn __len__(&self) -> usize {
        self.size as usize
    }

    fn __getitem__(&self, i: i64) -> PyResult<i64> {
        self.getitem_rs(i)
    }

    fn __contains__(&self, elem: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.contains_cp(codepoint_of(elem)?))
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> Py<IntervalSetIter> {
        Py::new(
            py,
            IntervalSetIter {
                intervals: slf.intervals.clone(),
                seg: 0,
                cur: slf.intervals.first().map(|&(u, _)| u).unwrap_or(0),
            },
        )
        .unwrap()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let t = self.intervals(py)?;
        Ok(format!("IntervalSet({})", t.repr()?.to_str()?))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.cast::<IntervalSet>() {
            Ok(o) => o.borrow().intervals == self.intervals,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.intervals.hash(&mut h);
        h.finish()
    }

    fn index(&self, value: i64) -> PyResult<i64> {
        self.ordinal_of(value)
    }

    fn index_above(&self, value: i64) -> i64 {
        index_above_in(&self.intervals, &self.offsets, self.size, value)
    }

    fn __or__(&self, other: &Bound<'_, IntervalSet>) -> IntervalSet {
        self.union_rs(&other.borrow())
    }

    fn __sub__(&self, other: &Bound<'_, IntervalSet>) -> IntervalSet {
        self.difference_rs(&other.borrow())
    }

    fn __and__(&self, other: &Bound<'_, IntervalSet>) -> IntervalSet {
        self.intersection_rs(&other.borrow())
    }

    fn union(&self, other: &Bound<'_, IntervalSet>) -> IntervalSet {
        self.union_rs(&other.borrow())
    }

    fn difference(&self, other: &Bound<'_, IntervalSet>) -> IntervalSet {
        self.difference_rs(&other.borrow())
    }

    fn intersection(&self, other: &Bound<'_, IntervalSet>) -> IntervalSet {
        self.intersection_rs(&other.borrow())
    }

    fn char_in_shrink_order(&self, py: Python<'_>, i: i64) -> PyResult<Py<PyAny>> {
        let cp = self.cp_in_shrink_order(i)?;
        codepoint_to_str(py, cp)
    }

    fn index_from_char_in_shrink_order(&self, c: &Bound<'_, PyAny>) -> PyResult<i64> {
        self.shrink_order_index_of_cp(codepoint_of(c)?)
    }
}

impl IntervalSet {
    pub(crate) fn from_pairs(pairs: Vec<(i64, i64)>) -> IntervalSet {
        IntervalSet::build(pairs)
    }

    pub(crate) fn from_string_rs(s: &str) -> IntervalSet {
        let mut cps: Vec<i64> = s.chars().map(|c| c as i64).collect();
        cps.sort_unstable();
        let pairs: Vec<(i64, i64)> = cps.iter().map(|&c| (c, c)).collect();
        let x = IntervalSet::build(pairs);
        x.union_rs(&x)
    }

    pub(crate) fn union_rs(&self, other: &IntervalSet) -> IntervalSet {
        let x = &self.intervals;
        let y = &other.intervals;
        if x.is_empty() {
            return IntervalSet::build(y.clone());
        }
        if y.is_empty() {
            return IntervalSet::build(x.clone());
        }
        // sort all intervals ascending, then merge overlapping/adjacent.
        let mut all: Vec<(i64, i64)> = x.iter().chain(y.iter()).copied().collect();
        all.sort_unstable();
        let mut result: Vec<(i64, i64)> = vec![all[0]];
        for &(u, v) in &all[1..] {
            let (a, b) = *result.last().unwrap();
            if u <= b + 1 {
                *result.last_mut().unwrap() = (a, v.max(b));
            } else {
                result.push((u, v));
            }
        }
        IntervalSet::build(result)
    }

    pub(crate) fn difference_rs(&self, other: &IntervalSet) -> IntervalSet {
        let y = &other.intervals;
        if y.is_empty() {
            return IntervalSet::build(self.intervals.clone());
        }
        let mut x: Vec<(i64, i64)> = self.intervals.clone();
        let mut i = 0usize;
        let mut j = 0usize;
        let mut result: Vec<(i64, i64)> = Vec::new();
        while i < x.len() && j < y.len() {
            let (xl, xr) = x[i];
            let (yl, yr) = y[j];
            if yr < xl {
                j += 1;
            } else if yl > xr {
                result.push(x[i]);
                i += 1;
            } else if yl <= xl {
                if yr >= xr {
                    i += 1;
                } else {
                    x[i].0 = yr + 1;
                    j += 1;
                }
            } else {
                result.push((xl, yl - 1));
                if yr + 1 <= xr {
                    x[i].0 = yr + 1;
                    j += 1;
                } else {
                    i += 1;
                }
            }
        }
        result.extend_from_slice(&x[i..]);
        IntervalSet::build(result)
    }

    pub(crate) fn intersection_rs(&self, other: &IntervalSet) -> IntervalSet {
        let mut result: Vec<(i64, i64)> = Vec::new();
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.intervals.len() && j < other.intervals.len() {
            let (u, v) = self.intervals[i];
            let (uu, vv) = other.intervals[j];
            if u > vv {
                j += 1;
            } else if uu > v {
                i += 1;
            } else {
                result.push((u.max(uu), v.min(vv)));
                if v < vv {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }
        IntervalSet::build(result)
    }
}

fn codepoint_to_str(py: Python<'_>, cp: i64) -> PyResult<Py<PyAny>> {
    if let Some(c) = u32::try_from(cp).ok().and_then(char::from_u32) {
        return Ok(PyString::new(py, &c.to_string()).into_any().unbind());
    }
    // lone surrogate or out-of-char codepoint: defer to Python's chr (flexible str).
    let builtins = py.import("builtins")?;
    Ok(builtins.getattr("chr")?.call1((cp,))?.unbind())
}

/// Codepoint of `elem`, which is either an int or a 1-char Python str. Strings go
/// through Python `ord` so lone surrogates (which can't cross into a Rust &str)
/// are handled, matching upstream's `ord(elem)`.
fn codepoint_of(elem: &Bound<'_, PyAny>) -> PyResult<i64> {
    if elem.is_instance_of::<PyString>() {
        let builtins = elem.py().import("builtins")?;
        builtins.getattr("ord")?.call1((elem,))?.extract()
    } else {
        elem.extract()
    }
}

#[pyclass(module = "hypothesis_fast._engine")]
pub(crate) struct IntervalSetIter {
    intervals: Vec<(i64, i64)>,
    seg: usize,
    cur: i64,
}

#[pymethods]
impl IntervalSetIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<i64> {
        while self.seg < self.intervals.len() {
            let (_u, v) = self.intervals[self.seg];
            if self.cur <= v {
                let out = self.cur;
                self.cur += 1;
                return Some(out);
            }
            self.seg += 1;
            if self.seg < self.intervals.len() {
                self.cur = self.intervals[self.seg].0;
            }
        }
        None
    }
}

use pyo3::types::PyType;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IntervalSet>()?;
    m.add_class::<IntervalSetIter>()?;
    Ok(())
}
