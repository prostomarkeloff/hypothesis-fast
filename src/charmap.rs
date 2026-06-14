//! Unicode general-category charmap, derived from the RUNNING interpreter's `unicodedata`.
//!
//! Maps each 2-letter general-category code to the codepoint intervals in that category. The
//! map is built ONCE (lazily, cached) by calling the Python helper
//! `hypothesis_fast.native_strategies._unicode_charmap`, which scans `unicodedata.category`.
//! Deriving it from the live interpreter — rather than a fixed table compiled into the
//! extension — is what lets `characters(categories=...)` agree with `unicodedata.category` on
//! whatever Unicode version this CPython ships (it differs by release: 3.11→14.0, 3.12→15.0,
//! 3.14→16.0); no single baked-in table can match every version.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;

type CharMap = HashMap<String, Vec<(i64, i64)>>;

static CHARMAP: GILOnceCell<CharMap> = GILOnceCell::new();

fn build_charmap(py: Python<'_>) -> PyResult<CharMap> {
    py.import("hypothesis_fast.native_strategies")?
        .getattr("_unicode_charmap")?
        .call0()?
        .extract()
}

/// code -> sorted, merged intervals of codepoints in that category (cached process-wide).
fn charmap(py: Python<'_>) -> PyResult<&'static CharMap> {
    CHARMAP.get_or_try_init(py, || build_charmap(py))
}

pub(crate) const ALL_CATEGORIES: [&str; 30] = [
    "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe",
    "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
];

/// Merged intervals for the union of the given category codes.
pub(crate) fn intervals_for_categories(
    py: Python<'_>,
    allowed: &[String],
) -> PyResult<Vec<(i64, i64)>> {
    let map = charmap(py)?;
    let mut all: Vec<(i64, i64)> = Vec::new();
    for cat in allowed {
        if let Some(v) = map.get(cat.as_str()) {
            all.extend_from_slice(v);
        }
    }
    all.sort_unstable();
    // merge adjacent/overlapping
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (lo, hi) in all {
        if let Some(last) = merged.last_mut() {
            if lo <= last.1 + 1 {
                last.1 = last.1.max(hi);
                continue;
            }
        }
        merged.push((lo, hi));
    }
    Ok(merged)
}
