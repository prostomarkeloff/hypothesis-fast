//! Text/characters alphabets + IntervalSet building + codec/codepoint validation.
//! Split out of strategy/constructors.rs.
#![allow(clippy::wildcard_imports)]
use super::*;


#[pyfunction]
#[pyo3(name = "text", signature = (alphabet=None, *, min_size=None, max_size=None))]
pub(crate) fn text(
    py: Python<'_>,
    alphabet: Option<Bound<'_, PyAny>>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // codec-confusion warning: a bare "ascii"/"utf-8" alphabet is almost always a
    // mistaken attempt to use that codec — st.text("ascii") actually draws from the
    // literal characters {'a','s','c','i'}, not the ascii codec. Mirrors upstream text().
    if let Some(a) = &alphabet {
        if let Ok(s) = a.extract::<String>() {
            if s == "ascii" || s == "utf-8" {
                let chars: Vec<String> = s.chars().map(|c| c.to_string()).collect();
                let msg = format!(
                    "st.text('{s}'): it seems like you are trying to use the codec '{s}'. \
                     st.text('{s}') instead generates strings using the literal characters \
                     {chars:?}. To specify the {s} codec, use st.text(st.characters(codec='{s}')). \
                     If you intended to use character literals, you can silence this warning by \
                     reordering the characters."
                );
                let hw = py
                    .import("hypothesis_fast.errors")?
                    .getattr("HypothesisWarning")?;
                py.import("warnings")?
                    .getattr("warn")?
                    .call1((msg, hw))?;
            }
        }
    }
    let intervals = build_intervals(py, alphabet)?;
    // empty alphabet only supports the empty string (min_size must be 0).
    let empty = intervals.bind(py).len().unwrap_or(1) == 0;
    let (min, max) = match collection_sizes(py, min_size, max_size, "text", empty, "the empty alphabet") {
        Ok(v) => v,
        Err(m) => return deferred_invalid(py, m),
    };
    SearchStrategy::wrap(py, StrategyNode::Text { intervals, min, max })
}

/// A character in a text() alphabet must be a length-one unicode string. Raises InvalidArgument
/// (like upstream) for a non-string or multi-character element.
pub(crate) fn validate_alphabet_char(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<String> {
    match v.extract::<String>() {
        Ok(c) if c.chars().count() == 1 => Ok(c),
        Ok(_) => Err(invalid_argument(
            py,
            format!(
                "The following elements in alphabet are not of length one, which leads to \
                 violating the size contract: [{}]",
                v.repr()?.extract::<String>()?
            ),
        )),
        Err(_) => Err(invalid_argument(
            py,
            format!(
                "The following elements in alphabet are not unicode strings: [{}]",
                v.repr()?.extract::<String>()?
            ),
        )),
    }
}

pub(crate) fn build_intervals(py: Python<'_>, alphabet: Option<Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
    let default = || -> PyResult<Py<PyAny>> {
        // characters() default alphabet: all codepoints except surrogates.
        let iset = build_characters_intervals(py, None, None, None, None, None, None, None)?
            .expect("default characters() alphabet is never empty/invalid");
        Ok(Py::new(py, iset)?.into_any())
    };
    match alphabet {
        None => default(),
        Some(a) if a.is_none() => default(),
        Some(a) => {
            if let Ok(s) = a.extract::<String>() {
                Ok(Py::new(py, crate::intervalset::IntervalSet::from_string_rs(&s))?.into_any())
            } else if a.downcast::<crate::intervalset::IntervalSet>().is_ok() {
                Ok(a.unbind())
            } else if let Ok(ss) = a.downcast::<SearchStrategy>() {
                // A strategy alphabet: its values must be length-one unicode strings.
                match &ss.borrow().node {
                    // text(alphabet=characters(...)) — reuse the characters' IntervalSet.
                    StrategyNode::Characters { intervals, .. } => Ok(intervals.clone_ref(py)),
                    // alphabet=just('x') / sampled_from(['a','b']) — inspect the constant
                    // value(s) and validate each is a single character.
                    StrategyNode::Just(v) => {
                        let c = validate_alphabet_char(py, v.bind(py))?;
                        Ok(Py::new(py, crate::intervalset::IntervalSet::from_string_rs(&c))?.into_any())
                    }
                    StrategyNode::SampledFrom { elements, .. } => {
                        let mut s = String::new();
                        for e in elements {
                            s.push_str(&validate_alphabet_char(py, e.bind(py))?);
                        }
                        Ok(Py::new(py, crate::intervalset::IntervalSet::from_string_rs(&s))?.into_any())
                    }
                    // Other strategies (builds, map, ...) can't be inspected statically; fall
                    // back to the default alphabet rather than over-rejecting.
                    _ => default(),
                }
            } else if a.hasattr("do_draw").unwrap_or(false) {
                // A FOREIGN (real-hypothesis) strategy alphabet — e.g. hypothesis-jsonschema's
                // CharStrategy(OneCharStringStrategy) for allow_x00=False / codec=. Resolve its
                // allowed-codepoint IntervalSet (so \x00 / non-codec characters are excluded) the
                // same way from_regex's alphabet= does, instead of defaulting to the full alphabet.
                match resolve_alphabet_iset(py, &a)? {
                    Some(iset) => Ok(Py::new(py, iset)?.into_any()),
                    None => default(),
                }
            } else if let Ok(iter) = a.try_iter() {
                // text(alphabet=['a','b',...]) — every element must be a single-char unicode str.
                let mut s = String::new();
                let mut bad_type: Vec<String> = Vec::new();
                let mut bad_len: Vec<String> = Vec::new();
                for it in iter {
                    let item = it?;
                    match item.extract::<String>() {
                        Ok(c) if c.chars().count() == 1 => s.push_str(&c),
                        Ok(_) => bad_len.push(item.repr()?.extract::<String>()?),
                        Err(_) => bad_type.push(item.repr()?.extract::<String>()?),
                    }
                }
                if !bad_type.is_empty() {
                    return Err(invalid_argument(
                        py,
                        format!(
                            "The following elements in alphabet are not unicode strings: [{}]",
                            bad_type.join(", ")
                        ),
                    ));
                }
                if !bad_len.is_empty() {
                    return Err(invalid_argument(
                        py,
                        format!(
                            "The following elements in alphabet are not of length one, which \
                             leads to violating the size contract: [{}]",
                            bad_len.join(", ")
                        ),
                    ));
                }
                Ok(Py::new(py, crate::intervalset::IntervalSet::from_string_rs(&s))?.into_any())
            } else {
                default()
            }
        }
    }
}

#[pyfunction]
#[pyo3(name = "characters", signature = (*, codec=None, min_codepoint=None, max_codepoint=None, categories=None, exclude_categories=None, include_characters=None, exclude_characters=None, whitelist_categories=None, blacklist_categories=None, whitelist_characters=None, blacklist_characters=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn characters(
    py: Python<'_>,
    codec: Option<Bound<'_, PyAny>>,
    min_codepoint: Option<Bound<'_, PyAny>>,
    max_codepoint: Option<Bound<'_, PyAny>>,
    categories: Option<Bound<'_, PyAny>>,
    exclude_categories: Option<Bound<'_, PyAny>>,
    include_characters: Option<Bound<'_, PyAny>>,
    exclude_characters: Option<Bound<'_, PyAny>>,
    // Deprecated aliases (still accepted, mapped to the modern names below).
    whitelist_categories: Option<Bound<'_, PyAny>>,
    blacklist_categories: Option<Bound<'_, PyAny>>,
    whitelist_characters: Option<Bound<'_, PyAny>>,
    blacklist_characters: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let categories =
        merge_deprecated_arg(py, categories, whitelist_categories, "categories", "whitelist_categories")?;
    let exclude_categories = merge_deprecated_arg(
        py, exclude_categories, blacklist_categories, "exclude_categories", "blacklist_categories",
    )?;
    let include_characters = merge_deprecated_arg(
        py, include_characters, whitelist_characters, "include_characters", "whitelist_characters",
    )?;
    let exclude_characters = merge_deprecated_arg(
        py, exclude_characters, blacklist_characters, "exclude_characters", "blacklist_characters",
    )?;
    let min_codepoint = coerce_codepoint(py, min_codepoint, "min_codepoint")?;
    let max_codepoint = coerce_codepoint(py, max_codepoint, "max_codepoint")?;
    // codec must be a valid codec-name string (codec=100 / an unknown name are errors).
    let codec: Option<String> = match codec {
        None => None,
        Some(c) if c.is_none() => None,
        Some(c) => {
            let s = match c.extract::<String>() {
                Ok(s) => s,
                Err(_) => {
                    return Err(invalid_argument(
                        py,
                        format!("codec={} must be a string.", c.repr()?.extract::<String>()?),
                    ));
                }
            };
            if py.import("codecs")?.getattr("lookup")?.call1((s.as_str(),)).is_err() {
                return Err(invalid_argument(py, format!("codec={s:?} is not a valid codec name.")));
            }
            Some(s)
        }
    };
    let repr = characters_force_repr(
        py,
        min_codepoint,
        max_codepoint,
        &categories,
        &exclude_categories,
        &include_characters,
        &exclude_characters,
        &codec,
    )?;
    match build_characters_intervals(
        py,
        min_codepoint,
        max_codepoint,
        categories,
        exclude_categories,
        include_characters,
        exclude_characters,
        codec,
    )? {
        Ok(iset) => {
            let intervals = Py::new(py, iset)?.into_any();
            SearchStrategy::wrap(py, StrategyNode::Characters { intervals, repr })
        }
        Err(msg) => deferred_invalid(py, msg),
    }
}

/// Build the `force_repr` for `characters(...)` — `characters(arg=val, ...)` listing
/// only non-default args, mirroring upstream's `OneCharStringStrategy.from_characters_args`
/// `_arg_repr` AFTER the public `characters()` normalization (codec `ascii`→max_codepoint
/// 127 + dropped, `utf-8` kept with categories defaulting to all-minus-Cs; `exclude_categories`
/// folded into `categories`; `categories` expanded via `as_general_categories` to a tuple).
#[allow(clippy::too_many_arguments)]
pub(crate) fn characters_force_repr(
    py: Python<'_>,
    min_codepoint: Option<i64>,
    max_codepoint: Option<i64>,
    categories: &Option<Bound<'_, PyAny>>,
    exclude_categories: &Option<Bound<'_, PyAny>>,
    include_characters: &Option<Bound<'_, PyAny>>,
    exclude_characters: &Option<Bound<'_, PyAny>>,
    codec: &Option<String>,
) -> PyResult<String> {
    use crate::charmap::ALL_CATEGORIES;

    let cat_tuple_repr = |v: &[String]| -> String {
        let inner = v.iter().map(|s| format!("'{s}'")).collect::<Vec<_>>().join(", ");
        if v.len() == 1 { format!("({inner},)") } else { format!("({inner})") }
    };
    let all_minus_cs: Vec<String> =
        ALL_CATEGORIES.iter().filter(|c| **c != "Cs").map(|c| c.to_string()).collect();

    // Resolve `categories` like the public characters(): explicit categories expand via
    // as_general_categories; otherwise all-minus-exclude_categories (or None when neither).
    let cats_list = to_str_list(categories.clone())?;
    let excl_list = to_str_list(exclude_categories.clone())?;
    let mut categories_norm: Option<Vec<String>> = if let Some(c) = &cats_list {
        Some(as_general_categories(c, &cat_tuple_repr(c), "categories").unwrap_or_default())
    } else if let Some(c) = &excl_list {
        let exc = as_general_categories(c, &cat_tuple_repr(c), "exclude_categories")
            .unwrap_or_default();
        Some(
            ALL_CATEGORIES
                .iter()
                .filter(|k| !exc.iter().any(|e| e == *k))
                .map(|k| k.to_string())
                .collect(),
        )
    } else {
        None
    };

    // Normalize codec: ascii caps max_codepoint at 127 and is then dropped; utf-8 stays
    // but forces categories to all-minus-Cs (which the skip rule below then elides).
    let mut max_cp = max_codepoint;
    let mut codec_norm: Option<String> = None;
    if let Some(c) = codec {
        let name: String = py
            .import("codecs")?
            .getattr("lookup")?
            .call1((c.as_str(),))?
            .getattr("name")?
            .extract()?;
        if name == "ascii" {
            if max_cp.map_or(true, |m| m > 127) {
                max_cp = Some(127);
            }
        } else if name == "utf-8" {
            categories_norm = Some(match categories_norm {
                Some(v) => v.into_iter().filter(|c| c != "Cs").collect(),
                None => all_minus_cs.clone(),
            });
            codec_norm = Some(name);
        } else {
            codec_norm = Some(name);
        }
    }

    // categories are shown unless equal to the all-minus-Cs default (upstream skip rule).
    let show_categories = match &categories_norm {
        Some(v) => {
            use std::collections::BTreeSet;
            let a: BTreeSet<&str> = v.iter().map(String::as_str).collect();
            let b: BTreeSet<&str> = all_minus_cs.iter().map(String::as_str).collect();
            a != b
        }
        None => false,
    };

    // include/exclude_characters: repr of the value as passed, skipping None / empty string.
    let str_repr = |o: &Option<Bound<'_, PyAny>>| -> PyResult<Option<String>> {
        match o {
            Some(v) if !v.is_none() => {
                if let Ok(s) = v.extract::<String>() {
                    if s.is_empty() {
                        return Ok(None);
                    }
                }
                Ok(Some(v.repr()?.to_string()))
            }
            _ => Ok(None),
        }
    };
    let inc_r = str_repr(include_characters)?;
    let exc_r = str_repr(exclude_characters)?;

    let mut parts: Vec<String> = Vec::new();
    if let Some(c) = &codec_norm {
        parts.push(format!("codec='{c}'"));
    }
    if let Some(lo) = min_codepoint {
        parts.push(format!("min_codepoint={lo}"));
    }
    if let Some(hi) = max_cp {
        parts.push(format!("max_codepoint={hi}"));
    }
    if show_categories {
        if let Some(v) = &categories_norm {
            parts.push(format!("categories={}", cat_tuple_repr(v)));
        }
    }
    if let Some(r) = exc_r {
        parts.push(format!("exclude_characters={r}"));
    }
    if let Some(r) = inc_r {
        parts.push(format!("include_characters={r}"));
    }
    Ok(format!("characters({})", parts.join(", ")))
}

pub(crate) fn to_str_list(obj: Option<Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    match obj {
        None => Ok(None),
        Some(o) if o.is_none() => Ok(None),
        Some(o) => {
            let mut out = Vec::new();
            for it in o.try_iter()? {
                out.push(it?.extract::<String>()?);
            }
            Ok(Some(out))
        }
    }
}

/// Expand one-letter major categories to their subclasses and validate that every
/// element is a real Unicode category — mirrors `charmap.as_general_categories`.
/// Returns Err(message) on an unknown category so the caller can defer it.
pub(crate) fn as_general_categories(cats: &[String], orig_repr: &str, name: &str) -> Result<Vec<String>, String> {
    use crate::charmap::ALL_CATEGORIES;
    const MAJOR: [&str; 7] = ["L", "M", "N", "P", "S", "Z", "C"];
    let mut out: Vec<String> = Vec::new();
    for c in cats {
        if MAJOR.contains(&c.as_str()) {
            for sub in ALL_CATEGORIES.iter().filter(|x| x.starts_with(c.as_str())) {
                if !out.iter().any(|o| o == sub) {
                    out.push(sub.to_string());
                }
            }
        } else if ALL_CATEGORIES.contains(&c.as_str()) {
            if !out.iter().any(|o| o == c) {
                out.push(c.clone());
            }
        } else {
            return Err(format!(
                "In {name}={orig_repr}, {cr} is not a valid Unicode category.",
                cr = PyReprStr(c),
            ));
        }
    }
    Ok(out)
}

/// Render a Rust &str the way Python's repr() would for a str (single-quoted).
struct PyReprStr<'a>(&'a str);
impl std::fmt::Display for PyReprStr<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "'{}'", self.0)
    }
}

/// Build the character IntervalSet, performing the full upstream `characters()`
/// validation. Returns `Ok(Ok(iset))` on success, `Ok(Err(msg))` for a deferred
/// InvalidArgument, and propagates real Python errors (iteration/extract) as Err.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
/// Map a deprecated characters() alias (whitelist_*/blacklist_*) to its modern name: returns
/// whichever was provided, erroring if BOTH the modern arg and its deprecated alias are passed.
pub(crate) fn merge_deprecated_arg<'py>(
    py: Python<'py>,
    modern: Option<Bound<'py, PyAny>>,
    deprecated: Option<Bound<'py, PyAny>>,
    modern_name: &str,
    deprecated_name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let modern_set = modern.as_ref().is_some_and(|o| !o.is_none());
    let deprecated_set = deprecated.as_ref().is_some_and(|o| !o.is_none());
    if modern_set && deprecated_set {
        return Err(invalid_argument(
            py,
            format!("Cannot pass both {modern_name} and the deprecated {deprecated_name}."),
        ));
    }
    Ok(if modern_set { modern } else if deprecated_set { deprecated } else { None })
}

/// A characters() codepoint bound must be a non-negative int: '1' isn't an int, -1 is negative.
pub(crate) fn coerce_codepoint(
    py: Python<'_>,
    v: Option<Bound<'_, PyAny>>,
    name: &str,
) -> PyResult<Option<i64>> {
    match v {
        None => Ok(None),
        Some(c) if c.is_none() => Ok(None),
        Some(c) => {
            let n = match c.extract::<i64>() {
                Ok(n) if c.is_instance_of::<pyo3::types::PyInt>() => n,
                _ => {
                    return Err(invalid_argument(
                        py,
                        format!("{name}={} must be an integer.", c.repr()?.extract::<String>()?),
                    ));
                }
            };
            if n < 0 {
                return Err(invalid_argument(
                    py,
                    format!("{name}={n} must not be negative."),
                ));
            }
            Ok(Some(n))
        }
    }
}

pub(crate) fn build_characters_intervals(
    py: Python<'_>,
    min_codepoint: Option<i64>,
    max_codepoint: Option<i64>,
    categories: Option<Bound<'_, PyAny>>,
    exclude_categories: Option<Bound<'_, PyAny>>,
    include_characters: Option<Bound<'_, PyAny>>,
    exclude_characters: Option<Bound<'_, PyAny>>,
    codec: Option<String>,
) -> PyResult<Result<crate::intervalset::IntervalSet, String>> {
    use crate::charmap::{intervals_for_categories, ALL_CATEGORIES};
    use crate::intervalset::IntervalSet;

    // min_codepoint > max_codepoint is invalid.
    if let (Some(lo), Some(hi)) = (min_codepoint, max_codepoint) {
        if lo > hi {
            return Ok(Err(format!(
                "min_codepoint={lo} is greater than max_codepoint={hi}"
            )));
        }
    }

    let has_categories = categories.as_ref().is_some_and(|o| !o.is_none());
    let has_exclude_categories = exclude_categories.as_ref().is_some_and(|o| !o.is_none());
    if has_categories && has_exclude_categories {
        return Ok(Err(
            "Pass at most one of categories and exclude_categories - these arguments \
             both specify which categories are allowed, so it doesn't make sense to \
             use both in a single call."
                .to_string(),
        ));
    }

    // Reprs for error messages (Python repr of the original objects).
    let inc_repr = match &include_characters {
        Some(o) if !o.is_none() => o.repr()?.to_string(),
        _ => "''".to_string(),
    };
    let exc_repr = match &exclude_characters {
        Some(o) if !o.is_none() => o.repr()?.to_string(),
        _ => "''".to_string(),
    };

    // "Nothing is excluded by other arguments" — include_characters alone has no effect.
    let has_include = include_characters.as_ref().is_some_and(|o| !o.is_none());
    if min_codepoint.is_none()
        && max_codepoint.is_none()
        && !has_categories
        && !has_exclude_categories
        && has_include
        && codec.is_none()
    {
        return Ok(Err(format!(
            "Nothing is excluded by other arguments, so passing only \
             include_characters={inc_repr} would have no effect.  Also pass \
             categories=(), or use sampled_from({inc_repr}) instead."
        )));
    }

    // Elements of include/exclude characters must each be a single character.
    let inc_elems = to_str_list(include_characters)?.unwrap_or_default();
    let exc_elems = to_str_list(exclude_characters)?.unwrap_or_default();
    if let Some(bad) = inc_elems.iter().find(|c| c.chars().count() != 1) {
        let bad_list = format!("[{}]", PyReprStr(bad));
        return Ok(Err(format!(
            "Elements of include_characters are required to be a single character, \
             but {bad_list} passed in include_characters={inc_repr} was not."
        )));
    }
    if let Some(bad) = exc_elems.iter().find(|c| c.chars().count() != 1) {
        let bad_list = format!("[{}]", PyReprStr(bad));
        return Ok(Err(format!(
            "Elements of exclude_characters are required to be a single character, \
             but {bad_list} passed in exclude_characters={exc_repr} was not."
        )));
    }

    let inc: String = inc_elems.concat();
    let exc: String = exc_elems.concat();

    // include/exclude overlap is invalid.
    let mut overlap: Vec<char> = inc
        .chars()
        .filter(|c| exc.contains(*c))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    overlap.sort_unstable();
    if !overlap.is_empty() {
        let listed = overlap
            .iter()
            .map(|c| format!("'{c}'"))
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(Err(format!(
            "Characters [{listed}] are present in both \
             include_characters={inc_repr} and exclude_characters={exc_repr}"
        )));
    }

    // Every include_characters element must be encodable with `codec` (codec='ascii' rejects 'é').
    if let Some(codec_name) = &codec {
        for ch in inc.chars() {
            let pystr = pyo3::types::PyString::new(py, &ch.to_string());
            if pystr.call_method1("encode", (codec_name.as_str(),)).is_err() {
                return Ok(Err(format!(
                    "Character {ch:?} in include_characters={inc_repr} cannot be encoded \
                     with codec={codec_name:?}"
                )));
            }
        }
    }

    // Resolve + validate the allowed category set.
    let categories_list = to_str_list(categories)?;
    let exclude_categories_list = to_str_list(exclude_categories)?;
    let cat_repr = |v: &[String]| -> String {
        let inner = v
            .iter()
            .map(|s| format!("'{s}'"))
            .collect::<Vec<_>>()
            .join(", ");
        if v.len() == 1 {
            format!("({inner},)")
        } else {
            format!("({inner})")
        }
    };

    let allowed: Vec<String> = if let Some(cats) = categories_list {
        let expanded = match as_general_categories(&cats, &cat_repr(&cats), "categories") {
            Ok(v) => v,
            Err(m) => return Ok(Err(m)),
        };
        if expanded.is_empty() && inc.is_empty() {
            return Ok(Err(
                "When `categories` is an empty collection and there are no characters \
                 specified in include_characters, nothing can be generated by the \
                 characters() strategy."
                    .to_string(),
            ));
        }
        expanded
    } else {
        let exc_cats = match exclude_categories_list {
            Some(c) => match as_general_categories(&c, &cat_repr(&c), "exclude_categories") {
                Ok(v) => v,
                Err(m) => return Ok(Err(m)),
            },
            None => Vec::new(),
        };
        // default: every category except surrogates, then drop exclude_categories.
        ALL_CATEGORIES
            .iter()
            .filter(|c| **c != "Cs" && !exc_cats.iter().any(|e| e == *c))
            .map(|c| c.to_string())
            .collect()
    };

    let lo = min_codepoint.unwrap_or(0);
    let mut hi = max_codepoint.unwrap_or(0x10FFFF);
    // ascii/utf-8 are common enough to special-case: ascii caps codepoints at 127.
    if codec.as_deref() == Some("ascii") {
        hi = hi.min(127);
    }
    let mut base = IntervalSet::from_pairs(intervals_for_categories(&allowed));
    // restrict to [lo, hi]
    let window = IntervalSet::from_pairs(vec![(lo, hi)]);
    base = base.intersection_rs(&window);

    if !exc.is_empty() {
        base = base.difference_rs(&IntervalSet::from_string_rs(&exc));
    }
    if !inc.is_empty() {
        // include_characters are forced in regardless of the codepoint window.
        base = base.union_rs(&IntervalSet::from_string_rs(&inc));
    }

    if base.alphabet_len() == 0 {
        let mut parts: Vec<String> = Vec::new();
        if let Some(lo) = min_codepoint {
            parts.push(format!("min_codepoint={lo}"));
        }
        if let Some(hi) = max_codepoint {
            parts.push(format!("max_codepoint={hi}"));
        }
        if !exc.is_empty() {
            parts.push(format!("exclude_characters={exc_repr}"));
        }
        if has_include {
            parts.push(format!("include_characters={inc_repr}"));
        }
        let _ = py;
        return Ok(Err(format!(
            "No characters are allowed to be generated by this combination of \
             arguments: {}",
            parts.join(", ")
        )));
    }
    Ok(Ok(base))
}
