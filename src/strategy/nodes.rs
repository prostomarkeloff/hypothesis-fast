//! Node analysis: filter-rewriting, typed-collection rebuild, node validation/
//! emptiness/reusable-values, and node repr. Split out of strategy/mod.rs.
#![allow(clippy::wildcard_imports)]
use super::*;

pub(crate) fn child_repr(child: &Py<PyAny>, py: Python<'_>) -> PyResult<String> {
    Ok(child.bind(py).repr()?.to_string())
}

/// `unwrap_strategies(s)`: the typed wrapped_strategy of a native SearchStrategy (recursing
/// LazyStrategy/wrapper layers), else the object itself.
pub(crate) fn unwrap_wrapped(strategy: &Py<PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
    match strategy.bind(py).downcast::<SearchStrategy>() {
        Ok(ss) => SearchStrategy::wrapped_strategy(ss, py),
        Err(_) => Ok(strategy.clone_ref(py)),
    }
}

/// Fold an order/equality predicate into integer bounds via hypothesis's predicate
/// analysis (construction-time, pure). Returns the rewritten strategy, or None to leave
/// the caller's plain `.filter()` in place (hypothesis not importable, or unanalyzable).
pub(crate) fn rewrite_integer_filter(
    slf: &Bound<'_, SearchStrategy>,
    py: Python<'_>,
    min: &Option<BigInt>,
    max: &Option<BigInt>,
    condition: &Py<PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    // hypothesis IntegersStrategy.filter special cases: every int is finite; none is inf/nan.
    let math = py.import("math")?;
    let cb = condition.bind(py);
    if cb.is(&math.getattr("isfinite")?) {
        return Ok(Some(slf.clone().into_any().unbind()));
    }
    if cb.is(&math.getattr("isinf")?) || cb.is(&math.getattr("isnan")?) {
        return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Nothing)?));
    }
    let Ok(filtering) = py.import("hypothesis.internal.filtering") else {
        return Ok(None);
    };
    let Ok(cp) = filtering
        .getattr("get_integer_predicate_bounds")
        .and_then(|f| f.call1((condition.bind(py),)))
    else {
        return Ok(None);
    };
    let constraints = cp.getattr("constraints")?;
    let cdict = constraints.downcast::<PyDict>()?;
    let mut new_min = min.clone();
    let mut new_max = max.clone();
    if let Some(v) = cdict.get_item("min_value")? {
        let cv: BigInt = v.extract()?;
        new_min = Some(match new_min {
            Some(m) => std::cmp::max(m, cv),
            None => cv,
        });
    }
    if let Some(v) = cdict.get_item("max_value")? {
        let cv: BigInt = v.extract()?;
        new_max = Some(match new_max {
            Some(m) => std::cmp::min(m, cv),
            None => cv,
        });
    }
    // A bound-preserving rewrite returns the SAME object (hypothesis returns `self`),
    // so `s.filter(noop) is s` (test_applying_noop_filter_returns_self).
    let changed = new_min != *min || new_max != *max;
    let base = if changed {
        if let (Some(a), Some(c)) = (&new_min, &new_max) {
            if a > c {
                return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Nothing)?));
            }
        }
        SearchStrategy::wrap(py, StrategyNode::Integers { min: new_min, max: new_max })?
    } else {
        slf.clone().into_any().unbind()
    };
    let remainder = cp.getattr("predicate")?;
    if remainder.is_none() {
        return Ok(Some(base));
    }
    Ok(Some(SearchStrategy::wrap(
        py,
        StrategyNode::Filter { base, func: remainder.unbind() },
    )?))
}

/// Fold a predicate into float bounds — a faithful port of hypothesis FloatStrategy.filter
/// (isfinite/isinf/isnan special cases + get_float_predicate_bounds + subnormal clamping).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rewrite_float_filter(
    slf: &Bound<'_, SearchStrategy>,
    py: Python<'_>,
    min: f64,
    max: f64,
    allow_nan: bool,
    allow_inf: bool,
    snm: f64,
    width: u32,
    condition: &Py<PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let math = py.import("math")?;
    let cond = condition.bind(py);
    if cond.is(&math.getattr("isfinite")?) {
        let nmin = min.max(crate::floats::next_up_rs(f64::NEG_INFINITY));
        let nmax = max.min(crate::floats::next_down_rs(f64::INFINITY));
        return Ok(Some(SearchStrategy::wrap(
            py,
            StrategyNode::Floats { min: nmin, max: nmax, allow_nan: false, allow_inf: false, snm, width },
        )?));
    }
    if cond.is(&math.getattr("isinf")?) {
        let mut permitted: Vec<Py<PyAny>> = Vec::new();
        for x in [f64::NEG_INFINITY, f64::INFINITY] {
            // An infinity is drawable only if it's within bounds AND the strategy allows
            // infinities (allow_infinity=False keeps -inf/inf as nominal node bounds but
            // forbids the values).
            if allow_inf && min <= x && x <= max {
                permitted.push(PyFloat::new(py, x).into_any().unbind());
            }
        }
        if permitted.is_empty() {
            return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Nothing)?));
        }
        return Ok(Some(SearchStrategy::wrap(
            py,
            StrategyNode::SampledFrom { elements: permitted, is_tuple: false },
        )?));
    }
    if cond.is(&math.getattr("isnan")?) {
        if !allow_nan {
            return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Nothing)?));
        }
        let nan = PyFloat::new(py, f64::NAN).into_any().unbind();
        return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Just(nan))?));
    }
    let Ok(filtering) = py.import("hypothesis.internal.filtering") else {
        return Ok(None);
    };
    let Ok(cp) = filtering
        .getattr("get_float_predicate_bounds")
        .and_then(|f| f.call1((cond,)))
    else {
        return Ok(None);
    };
    let cdict = cp.getattr("constraints")?;
    let cdict = cdict.downcast::<PyDict>()?;
    if cdict.is_empty() {
        return Ok(None);
    }
    let cmin = match cdict.get_item("min_value")? {
        Some(v) => v.extract::<f64>()?,
        None => f64::NEG_INFINITY,
    };
    let cmax = match cdict.get_item("max_value")? {
        Some(v) => v.extract::<f64>()?,
        None => f64::INFINITY,
    };
    let mut min_bound = cmin.max(min);
    let mut max_bound = cmax.min(max);
    if -snm < min_bound && min_bound < 0.0 {
        min_bound = -0.0;
    } else if 0.0 < min_bound && min_bound < snm {
        min_bound = snm;
    }
    if -snm < max_bound && max_bound < 0.0 {
        max_bound = -snm;
    } else if 0.0 < max_bound && max_bound < snm {
        max_bound = 0.0;
    }
    if min_bound > max_bound {
        return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Nothing)?));
    }
    // hypothesis only rebuilds when the bounds genuinely tighten (or nan is dropped);
    // otherwise it returns `self`, so `s.filter(noop) is s`.
    let changed = min_bound > min
        || max > max_bound
        || (allow_nan && (f64::NEG_INFINITY < min_bound || max_bound < f64::INFINITY));
    let base = if changed {
        let new_inf = min_bound == f64::NEG_INFINITY || max_bound == f64::INFINITY;
        SearchStrategy::wrap(
            py,
            StrategyNode::Floats { min: min_bound, max: max_bound, allow_nan: false, allow_inf: new_inf, snm, width },
        )?
    } else {
        slf.clone().into_any().unbind()
    };
    let pred = cp.getattr("predicate")?;
    if pred.is_none() {
        return Ok(Some(base));
    }
    Ok(Some(SearchStrategy::wrap(py, StrategyNode::Filter { base, func: pred.unbind() })?))
}

/// Fold a `partial(op, date)` order/equality predicate into date bounds — a port of
/// hypothesis DateStrategy.filter (ordinal arithmetic; OverflowError past date.min/max →
/// nothing(); a bound-preserving filter returns self).
pub(crate) fn rewrite_date_filter(
    slf: &Bound<'_, SearchStrategy>,
    py: Python<'_>,
    min_ord: i64,
    max_ord: i64,
    condition: &Py<PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let c = condition.bind(py);
    if !c.is_instance(&py.import("functools")?.getattr("partial")?)? {
        return Ok(None);
    }
    let args = c.getattr("args")?;
    if args.len()? != 1 || c.getattr("keywords")?.is_truthy()? {
        return Ok(None);
    }
    let arg = args.get_item(0)?;
    let date_cls = py.import("datetime")?.getattr("date")?;
    if !arg.is_instance(&date_cls)? {
        return Ok(None);
    }
    let op = py.import("operator")?;
    let func = c.getattr("func")?;
    let which = if func.is(&op.getattr("lt")?) {
        "lt"
    } else if func.is(&op.getattr("le")?) {
        "le"
    } else if func.is(&op.getattr("eq")?) {
        "eq"
    } else if func.is(&op.getattr("ge")?) {
        "ge"
    } else if func.is(&op.getattr("gt")?) {
        "gt"
    } else {
        return Ok(None);
    };
    // We're talking about op(arg, x) — the reverse of the usual intuition.
    let arg_ord: i64 = arg.call_method0("toordinal")?.extract()?;
    let adj = arg_ord + match which { "lt" => 1, "gt" => -1, _ => 0 };
    // date.min/date.max ordinals are 1 and 3_652_059; stepping past them is OverflowError.
    if !(1..=3_652_059).contains(&adj) {
        return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Nothing)?));
    }
    let (lo, hi) = match which {
        "lt" | "le" => (adj, max_ord),
        "eq" => (adj, adj),
        _ => (min_ord, adj),
    };
    let lo = lo.max(min_ord);
    let hi = hi.min(max_ord);
    if hi < lo {
        return Ok(Some(SearchStrategy::wrap(py, StrategyNode::Nothing)?));
    }
    if lo <= min_ord && max_ord <= hi {
        return Ok(Some(slf.clone().into_any().unbind()));
    }
    let lo_date = date_cls.call_method1("fromordinal", (lo,))?;
    let hi_date = date_cls.call_method1("fromordinal", (hi,))?;
    Ok(Some(dates(py, Some(lo_date), Some(hi_date))?))
}

pub(crate) enum CollRw {
    SelfUnchanged,
    Empty,
    Resize { min: usize, max: usize, keep: Option<Py<PyAny>> },
    Plain,
}

/// Collection/string filter-rewriting (hypothesis ListStrategy.filter + the string len/
/// content path): nonempty predicates (bool/len/tuple/list/identity) bump min_size to 1;
/// `len`-based predicates fold into min_size/max_size; string content methods (isalnum…)
/// bump min_size and keep the filter. Reuses hypothesis.internal.filtering's analysis.
pub(crate) fn analyze_coll_filter(
    py: Python<'_>,
    min_size: usize,
    max_size: usize,
    is_str: bool,
    is_bytes: bool,
    condition: &Py<PyAny>,
) -> PyResult<CollRw> {
    let c = condition.bind(py);
    let cls = if is_bytes {
        "BytesStrategy"
    } else if is_str {
        "TextStrategy"
    } else {
        "ListStrategy"
    };
    // str.lower/.title/.upper as a filter just means "nonempty" — warn (the user likely
    // meant str.islower etc.) then fall through to the nonempty handling below.
    if is_str {
        warn_suspicious_string_method(py, c, is_bytes)?;
    }
    let is_id = py
        .import("hypothesis.internal.reflection")
        .and_then(|m| m.getattr("is_identity_function"))
        .and_then(|f| f.call1((c,)))
        .and_then(|r| r.is_truthy())
        .unwrap_or(false);
    let is_nonempty = is_id || coll_class_filter_set(py, cls, "_nonempty_filters", c)?;
    if is_nonempty {
        if max_size < 1 {
            return Ok(CollRw::Empty);
        }
        if min_size >= 1 {
            return Ok(CollRw::SelfUnchanged);
        }
        return Ok(CollRw::Resize { min: 1, max: max_size, keep: None });
    }
    // len-based: get_integer_predicate_bounds with a `len` constraint.
    if let Ok(filtering) = py.import("hypothesis.internal.filtering") {
        if let Ok(cp) = filtering
            .getattr("get_integer_predicate_bounds")
            .and_then(|f| f.call1((c,)))
        {
            let cd = cp.getattr("constraints")?;
            let cd = cd.downcast::<PyDict>()?;
            let has_len = cd
                .get_item("len")?
                .map(|v| v.is_truthy().unwrap_or(false))
                .unwrap_or(false);
            let has_bound = cd.contains("min_value")? || cd.contains("max_value")?;
            if has_len && has_bound {
                let nm = match cd.get_item("min_value")? {
                    Some(v) => std::cmp::max(min_size, v.extract::<usize>()?),
                    None => min_size,
                };
                let nx = match cd.get_item("max_value")? {
                    Some(v) => std::cmp::min(max_size, v.extract::<usize>()?),
                    None => max_size,
                };
                if nm > nx {
                    return Ok(CollRw::Plain);
                }
                let keep = if cp.getattr("predicate")?.is_none() {
                    None
                } else {
                    Some(condition.clone_ref(py))
                };
                if nm == min_size && nx == max_size && keep.is_none() {
                    return Ok(CollRw::SelfUnchanged);
                }
                return Ok(CollRw::Resize { min: nm, max: nx, keep });
            }
        }
    }
    // string content methods (str.isalnum, .isdigit, …) — bump min_size + keep the filter.
    if is_str && max_size >= 1 && coll_class_filter_set(py, cls, "_nonempty_and_content_filters", c)? {
        return Ok(CollRw::Resize {
            min: std::cmp::max(1, min_size),
            max: max_size,
            keep: Some(condition.clone_ref(py)),
        });
    }
    Ok(CollRw::Plain)
}

/// hypothesis's regex branch of `_string_filter_rewrite`: a text/binary strategy filtered
/// by `re.compile(p).search`/match/findall/fullmatch becomes a `regex_strategy` so matching
/// strings are generated directly (drawn through our real-compatible ConjectureData) rather
/// than rejection-sampled. `finditer`/`split` (which need a draw-time warning) are NOT
/// handled here — they fall through to a plain filter.
pub(crate) fn try_regex_rewrite(
    slf: &Bound<'_, SearchStrategy>,
    py: Python<'_>,
    min_size: usize,
    max_size: usize,
    is_bytes: bool,
    condition: &Py<PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let c = condition.bind(py);
    let Ok(pat_self) = c.getattr("__self__") else {
        return Ok(None);
    };
    let re_mod = py.import("re")?;
    if !pat_self.is_instance(&re_mod.getattr("Pattern")?)? {
        return Ok(None);
    }
    let old_pat = pat_self.getattr("pattern")?;
    let kind = py
        .import("builtins")?
        .getattr(if is_bytes { "bytes" } else { "str" })?;
    if !old_pat.is_instance(&kind)? {
        return Ok(None);
    }
    let name: String = c.getattr("__name__")?.extract()?;
    // `match` is a `search` anchored at the start.
    let (pattern, eff_name): (Bound<'_, PyAny>, String) = if name == "match" {
        let flags = pat_self.getattr("flags")?;
        let compile = re_mod.getattr("compile")?;
        let new_src = if is_bytes {
            PyBytes::new(py, b"^(?:")
                .call_method1("__add__", (&old_pat,))?
                .call_method1("__add__", (PyBytes::new(py, b")"),))?
        } else {
            let s: String = old_pat.extract()?;
            PyString::new(py, &format!("^(?:{s})")).into_any()
        };
        (compile.call1((new_src, flags))?, "search".to_string())
    } else {
        (pat_self.clone(), name.clone())
    };
    // finditer/scanner (any string matches) and split (any nonempty string) don't actually
    // constrain the value, so hypothesis doesn't rewrite — it warns and keeps the base
    // strategy. The warning must fire at DRAW time (real hypothesis defers filter-rewriting
    // through LazyStrategy), so we wrap the base in a warn-then-identity map.
    if matches!(name.as_str(), "finditer" | "scanner" | "split") {
        let pretty: String = py
            .import("hypothesis.vendor.pretty")?
            .getattr("pretty")?
            .call1((c,))?
            .extract()?;
        let (msg, base): (String, Py<PyAny>) = if name == "split" {
            let b = py.import("builtins")?.getattr("bool")?;
            (
                format!(
                    "You applied {pretty} as a filter, but this allows any nonempty string!  \
                     Did you mean .search ?"
                ),
                slf.call_method1("filter", (b,))?.unbind(),
            )
        } else {
            (
                format!(
                    "You applied {pretty} as a filter, but this allows any string at all!  \
                     Did you mean .findall ?"
                ),
                slf.clone().into_any().unbind(),
            )
        };
        let warner = py
            .import("hypothesis_fast.native_strategies")?
            .getattr("_warn_then_identity")?
            .call1((msg,))?
            .unbind();
        return Ok(Some(MappedStrategy::build(py, base, warner)?));
    }
    if !matches!(eff_name.as_str(), "search" | "findall" | "fullmatch") {
        return Ok(None);
    }
    let alphabet = if is_bytes {
        py.None()
    } else {
        let intervals = match &slf.borrow().node {
            StrategyNode::Text { intervals, .. } => intervals.clone_ref(py),
            _ => return Ok(None),
        };
        let real_iv = py
            .import("hypothesis.internal.intervalsets")?
            .getattr("IntervalSet")?
            .call1((intervals.bind(py).getattr("intervals")?,))?;
        py.import("hypothesis.strategies._internal.strings")?
            .getattr("OneCharStringStrategy")?
            .call1((real_iv,))?
            .unbind()
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("alphabet", alphabet)?;
    let mut s = py
        .import("hypothesis.strategies._internal.regex")?
        .getattr("regex_strategy")?
        .call((pattern, eff_name == "fullmatch"), Some(&kwargs))?;
    let functools = py.import("functools")?;
    let filtering = py.import("hypothesis.internal.filtering")?;
    if min_size > 0 {
        let p = functools
            .getattr("partial")?
            .call1((filtering.getattr("min_len")?, min_size))?;
        s = s.call_method1("filter", (p,))?;
    }
    if max_size < COLLECTION_DEFAULT_MAX_SIZE {
        let p = functools
            .getattr("partial")?
            .call1((filtering.getattr("max_len")?, max_size))?;
        s = s.call_method1("filter", (p,))?;
    }
    Ok(Some(s.unbind()))
}

/// True if `condition` is a member of `hypothesis`'s named filter tuple (`_nonempty_filters`
/// or `_nonempty_and_content_filters`) on the given strategy class.
pub(crate) fn coll_class_filter_set(
    py: Python<'_>,
    cls: &str,
    attr: &str,
    condition: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let module = if cls == "ListStrategy" {
        "hypothesis.strategies._internal.collections"
    } else {
        "hypothesis.strategies._internal.strings"
    };
    let Ok(m) = py.import(module) else {
        return Ok(false);
    };
    let Ok(filters) = m.getattr(cls).and_then(|c| c.getattr(attr)) else {
        return Ok(false);
    };
    for f in filters.try_iter()? {
        if f?.is(condition) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Emit hypothesis's HypothesisWarning when str/bytes `.lower`/`.title`/`.upper` is used as a
/// filter (it only ensures non-emptiness; the user probably meant `.islower`/etc.).
pub(crate) fn warn_suspicious_string_method(
    py: Python<'_>,
    condition: &Bound<'_, PyAny>,
    is_bytes: bool,
) -> PyResult<()> {
    let kname = if is_bytes { "bytes" } else { "str" };
    let kind = py.import("builtins")?.getattr(kname)?;
    for name in ["lower", "title", "upper"] {
        if condition.is(&kind.getattr(name)?) {
            let msg = format!(
                "You applied {kname}.{name} as a filter, but this allows all nonempty strings!  \
                 Did you mean {kname}.is{name}?"
            );
            let hyp_warn = py
                .import("hypothesis.errors")?
                .getattr("HypothesisWarning")?;
            py.import("warnings")?
                .call_method1("warn", (msg, hyp_warn))?;
            return Ok(());
        }
    }
    Ok(())
}

/// Build the typed pyclass for a collection node (TextStrategy/BytesStrategy/ListStrategy)
/// so that `.min_size`/`.max_size` getters are present. Mirrors `wrapped_strategy`'s
/// collection arms; non-collection nodes fall back to a generic SearchStrategy.
pub(crate) fn build_typed_collection(py: Python<'_>, node: StrategyNode) -> PyResult<Py<PyAny>> {
    match &node {
        StrategyNode::Text { min, max, .. } => {
            let (mn, mx) = (*min, *max);
            TextStrategy::build(py, node, mn, mx)
        }
        StrategyNode::Binary { min, max } => {
            let (mn, mx) = (*min, *max);
            BytesStrategy::build(py, node, mn, mx)
        }
        StrategyNode::Lists { min, max, .. }
        | StrategyNode::Sets { min, max, .. }
        | StrategyNode::Dictionaries { min, max, .. } => {
            let (mn, mx) = (*min, *max);
            ListStrategy::build(py, node, mn, mx)
        }
        _ => SearchStrategy::wrap(py, node),
    }
}

/// If `strat`'s node is a `Filter`, its (base, func); else None.
pub(crate) fn as_filter_node(py: Python<'_>, strat: &Py<PyAny>) -> Option<(Py<PyAny>, Py<PyAny>)> {
    let bound = strat.bind(py);
    let ss = bound.downcast::<SearchStrategy>().ok()?;
    let borrowed = ss.borrow();
    match &borrowed.node {
        StrategyNode::Filter { base, func } => Some((base.clone_ref(py), func.clone_ref(py))),
        _ => None,
    }
}

/// Build a left-nested Filter node applying each of `conds` (non-empty) over `base`.
pub(crate) fn chain_filter_node(
    py: Python<'_>,
    base: Py<PyAny>,
    conds: &[Py<PyAny>],
) -> PyResult<StrategyNode> {
    let mut acc = base;
    for c in &conds[..conds.len() - 1] {
        acc = SearchStrategy::wrap(py, StrategyNode::Filter { base: acc, func: c.clone_ref(py) })?;
    }
    Ok(StrategyNode::Filter { base: acc, func: conds[conds.len() - 1].clone_ref(py) })
}

/// Flatten a chain of `.filter()`s into a single FilteredStrategy: collect every condition,
/// then replay each against the running (rewritten) base so all rewritable predicates fold
/// into the base bounds and the unhandled ones accumulate as flat_conditions. Mirrors
/// hypothesis's FilteredStrategy.filter consolidation + do_validate replay.
pub(crate) fn flatten_filtered(slf: &Bound<'_, SearchStrategy>, py: Python<'_>) -> PyResult<Py<PyAny>> {
    let mut conds_rev: Vec<Py<PyAny>> = Vec::new();
    let mut cur: Py<PyAny> = slf.clone().into_any().unbind();
    while let Some((base, func)) = as_filter_node(py, &cur) {
        conds_rev.push(func);
        cur = base;
    }
    conds_rev.reverse();
    let mut current = cur;
    let mut leftover: Vec<Py<PyAny>> = Vec::new();
    for cond in conds_rev {
        let result = current
            .bind(py)
            .call_method1("filter", (cond.bind(py),))?
            .unbind();
        // A Filter-node result means the predicate (or its remainder) wasn't fully folded:
        // keep the rewritten base and record the leftover condition.
        match as_filter_node(py, &result) {
            Some((rb, rfunc)) => {
                current = rb;
                leftover.push(rfunc);
            }
            None => current = result,
        }
    }
    let inner = unwrap_wrapped(&current, py)?;
    if leftover.is_empty() {
        return Ok(inner);
    }
    let conds = PyTuple::new(py, leftover.iter().map(|c| c.bind(py)))?
        .into_any()
        .unbind();
    let node = chain_filter_node(py, current, &leftover)?;
    FilteredStrategy::build(py, node, inner, conds)
}

/// True if `strat` is (a wrapper around) a `lists()` strategy node.
pub(crate) fn base_is_list(py: Python<'_>, strat: &Py<PyAny>) -> PyResult<bool> {
    let bound = strat.bind(py);
    if let Ok(ss) = bound.downcast::<SearchStrategy>() {
        return Ok(matches!(ss.borrow().node, StrategyNode::Lists { .. }));
    }
    Ok(false)
}

/// True if `pack` is a collection constructor (a `collections.abc.Collection` subtype, e.g.
/// tuple/list/set/frozenset/dict) or a known collection-ish function (`sorted`) — the packs
/// for which `len(pack(xs)) == len(xs)`, so a length filter can push through the map.
pub(crate) fn pack_is_collection_ish(py: Python<'_>, pack: &Py<PyAny>) -> PyResult<bool> {
    let p = pack.bind(py);
    let builtins = py.import("builtins")?;
    if p.is_instance(&builtins.getattr("type")?)? {
        let collection = py.import("collections.abc")?.getattr("Collection")?;
        if builtins
            .getattr("issubclass")?
            .call1((&p, collection))?
            .is_truthy()?
        {
            return Ok(true);
        }
    }
    Ok(p.is(&builtins.getattr("sorted")?))
}

/// Rebuild a Text/Binary/Lists node with new min/max sizes (other fields preserved).
pub(crate) fn rebuild_collection_node(
    node: &StrategyNode,
    new_min: usize,
    new_max: usize,
    py: Python<'_>,
) -> PyResult<StrategyNode> {
    Ok(match node {
        StrategyNode::Text { intervals, .. } => {
            StrategyNode::Text { intervals: intervals.clone_ref(py), min: new_min, max: new_max }
        }
        StrategyNode::Binary { .. } => StrategyNode::Binary { min: new_min, max: new_max },
        StrategyNode::Lists { elem, unique_by, swap_domain, .. } => StrategyNode::Lists {
            elem: elem.clone_ref(py),
            min: new_min,
            max: new_max,
            unique_by: unique_by.as_ref().map(|u| u.clone_ref(py)),
            swap_domain: swap_domain
                .as_ref()
                .map(|d| d.iter().map(|v| v.clone_ref(py)).collect()),
        },
        StrategyNode::Sets { elem, frozen, .. } => StrategyNode::Sets {
            elem: elem.clone_ref(py),
            min: new_min,
            max: new_max,
            frozen: *frozen,
        },
        StrategyNode::Dictionaries { keys, values, .. } => StrategyNode::Dictionaries {
            keys: keys.clone_ref(py),
            values: values.clone_ref(py),
            min: new_min,
            max: new_max,
        },
        _ => unreachable!("rebuild_collection_node called on non-collection node"),
    })
}

pub(crate) fn validate_child(child: &Py<PyAny>, py: Python<'_>) -> PyResult<()> {
    match child.bind(py).downcast::<SearchStrategy>() {
        Ok(ss) => node_validate(&ss.borrow().node, py),
        Err(_) => Ok(()),
    }
}

/// A builds() arg/kwarg must be a SearchStrategy; a non-strategy raises InvalidArgument
/// (upstream check_strategy). `prefix` is "" for a positional arg or "name=" for a kwarg.
pub(crate) fn validate_builds_arg(py: Python<'_>, prefix: &str, child: &Py<PyAny>) -> PyResult<()> {
    let c = child.bind(py);
    match c.downcast::<SearchStrategy>() {
        Ok(ss) => node_validate(&ss.borrow().node, py),
        Err(_) => Err(invalid_argument(
            py,
            format!(
                "Expected a SearchStrategy but got {prefix}{} (type={})",
                c.repr()?,
                c.get_type().name()?
            ),
        )),
    }
}

/// Recursively validate a node tree, raising for any deferred Invalid node.
/// Callback nodes (composite/builds/flatmap-func) and deferred() are NOT recursed
/// (they wrap user code / self-reference), avoiding infinite recursion.
pub(crate) fn node_validate(node: &StrategyNode, py: Python<'_>) -> PyResult<()> {
    match node {
        StrategyNode::Invalid { msg, resolution_failed } => Err(if *resolution_failed {
            resolution_failed_err(py, msg.clone())
        } else {
            invalid_argument(py, msg.clone())
        }),
        StrategyNode::Map { base, .. }
        | StrategyNode::Filter { base, .. }
        | StrategyNode::Flatmap { base, .. } => validate_child(base, py),
        StrategyNode::Lists { elem, .. } | StrategyNode::Sets { elem, .. } => {
            validate_child(elem, py)
        }
        StrategyNode::Dictionaries { keys, values, .. } => {
            validate_child(keys, py)?;
            validate_child(values, py)
        }
        StrategyNode::Tuples(cs) | StrategyNode::OneOf(cs) => {
            for c in cs {
                validate_child(c, py)?;
            }
            Ok(())
        }
        StrategyNode::FixedDict { items } => {
            for (_, v) in items {
                validate_child(v, py)?;
            }
            Ok(())
        }
        // builds() args/kwargs must each be a SearchStrategy — a non-strategy (e.g. a
        // raw list passed where a strategy was meant) is an InvalidArgument, surfaced at
        // validate() like upstream's check_strategy (test_does_not_error_on_unhashable_kwarg).
        StrategyNode::Builds { args, kwargs, .. } => {
            for a in args {
                validate_builds_arg(py, "", a)?;
            }
            for (k, v) in kwargs {
                validate_builds_arg(py, &format!("{k}="), v)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Whether `base.filter(func)` is provably unsatisfiable for a numeric base — the
/// native side of hypothesis's filter rewriting. Bound extraction stays in Rust; the
/// predicate analysis (math.isinf/isnan, functools.partial(operator.OP, N)) is the
/// conservative Python helper `_filter_makes_empty` (returns False on any uncertainty).
pub(crate) fn filter_rewrite_empty(py: Python<'_>, base: &Py<PyAny>, func: &Py<PyAny>) -> PyResult<bool> {
    let ss = match base.bind(py).downcast::<SearchStrategy>() {
        Ok(s) => s,
        Err(_) => return Ok(false),
    };
    let b = ss.borrow();
    let helper = py
        .import("hypothesis_fast.native_strategies")?
        .getattr("_filter_makes_empty")?;
    let none = || py.None().into_bound(py);
    let res = match &b.node {
        StrategyNode::Integers { min, max } => {
            let lo = match min {
                Some(m) => m.clone().into_pyobject(py)?.into_any(),
                None => none(),
            };
            let hi = match max {
                Some(m) => m.clone().into_pyobject(py)?.into_any(),
                None => none(),
            };
            helper.call1((func.bind(py), "int", lo, hi, false, false))?
        }
        StrategyNode::Floats { min, max, allow_nan, allow_inf, .. } => {
            helper.call1((func.bind(py), "float", *min, *max, *allow_nan, *allow_inf))?
        }
        _ => return Ok(false),
    };
    res.is_truthy()
}

/// has_reusable_values: whether a drawn value is safe to reuse without re-drawing —
/// i.e. immutable. Ports upstream's recursive_property: immutable leaves → True;
/// mutable collections (list/set/dict) → False; any .map/.filter/.flatmap or other
/// transformation → False; tuples/one_of → all children reusable; deferred → recurse.
pub(crate) fn node_has_reusable_values(node: &StrategyNode, py: Python<'_>) -> PyResult<bool> {
    let child_reusable = |c: &Py<PyAny>| -> PyResult<bool> {
        match c.bind(py).downcast::<SearchStrategy>() {
            Ok(ss) => node_has_reusable_values(&ss.borrow().node, py),
            Err(_) => Ok(false),
        }
    };
    Ok(match node {
        // Immutable scalar / atomic values — reusable.
        StrategyNode::Integers { .. }
        | StrategyNode::Booleans
        | StrategyNode::Floats { .. }
        | StrategyNode::NoneVal
        | StrategyNode::Just(_)
        | StrategyNode::Nothing
        | StrategyNode::SampledFrom { .. }
        | StrategyNode::SampledFromRange { .. }
        | StrategyNode::Text { .. }
        | StrategyNode::Characters { .. }
        | StrategyNode::Binary { .. }
        | StrategyNode::Uuids { .. }
        | StrategyNode::Dates { .. }
        | StrategyNode::Times { .. }
        | StrategyNode::Datetimes { .. }
        | StrategyNode::Timedeltas { .. }
        | StrategyNode::ComplexNumbers { .. }
        | StrategyNode::IpAddresses { .. }
        | StrategyNode::Slices { .. }
        | StrategyNode::Fractions { .. }
        | StrategyNode::Decimals { .. }
        | StrategyNode::Invalid { .. } => true,
        // Tuples / one_of are reusable iff every component is.
        StrategyNode::Tuples(cs) | StrategyNode::OneOf(cs) => {
            let mut all = true;
            for c in cs {
                if !child_reusable(c)? {
                    all = false;
                    break;
                }
            }
            all
        }
        StrategyNode::Deferred { thunk } => deferred_has_reusable(thunk, py)?,
        StrategyNode::Shared { base, .. } => child_reusable(base)?,
        // Mutable collections, transformations, and unknown/user-output values.
        StrategyNode::Lists { .. }
        | StrategyNode::Sets { .. }
        | StrategyNode::Dictionaries { .. }
        | StrategyNode::FixedDict { .. }
        | StrategyNode::Permutations(_)
        | StrategyNode::Map { .. }
        | StrategyNode::Filter { .. }
        | StrategyNode::Flatmap { .. }
        | StrategyNode::Builds { .. }
        | StrategyNode::Composite { .. }
        | StrategyNode::Functions { .. }
        | StrategyNode::Randoms { .. }
        | StrategyNode::Data => false,
    })
}

pub(crate) fn node_is_empty(node: &StrategyNode, py: Python<'_>) -> PyResult<bool> {
    let child_empty = |c: &Py<PyAny>| -> PyResult<bool> {
        match c.bind(py).downcast::<SearchStrategy>() {
            Ok(ss) => node_is_empty(&ss.borrow().node, py),
            Err(_) => Ok(false),
        }
    };
    Ok(match node {
        StrategyNode::Nothing => true,
        StrategyNode::SampledFrom { elements, .. } => elements.is_empty(),
        StrategyNode::OneOf(children) => {
            let mut all = !children.is_empty();
            for c in children {
                if !child_empty(c)? {
                    all = false;
                    break;
                }
            }
            children.is_empty() || all
        }
        StrategyNode::Map { base, .. } | StrategyNode::Flatmap { base, .. } => child_empty(base)?,
        StrategyNode::Filter { base, func } => {
            child_empty(base)? || filter_rewrite_empty(py, base, func)?
        }
        StrategyNode::Deferred { thunk } => deferred_is_empty(thunk, py)?,
        StrategyNode::FixedDict { items } => {
            let mut any = false;
            for (_, v) in items {
                if child_empty(v)? {
                    any = true;
                    break;
                }
            }
            any
        }
        StrategyNode::Lists { elem, min, .. } | StrategyNode::Sets { elem, min, .. } => {
            *min > 0 && child_empty(elem)?
        }
        StrategyNode::Tuples(children) => {
            let mut any = false;
            for c in children {
                if child_empty(c)? {
                    any = true;
                    break;
                }
            }
            any
        }
        _ => false,
    })
}

/// Render a callable for a strategy repr: a clean `__name__` for named functions
/// (so `.map(str)` matches hypothesis), `...` for lambdas (whose full source-repr
/// would need the lambda-source extraction subsystem we don't have natively yet).
pub(crate) fn fn_repr(func: &Py<PyAny>, py: Python<'_>) -> String {
    // Mirror upstream reprs: get_pretty_function_description renders lambdas as their
    // source, functools.partial as `functools.partial(f, 2)`, methodcaller, named
    // functions/classes by name, etc. (test_can_map_nameless / test_can_flatmap_nameless).
    if let Ok(desc) = py
        .import("hypothesis.internal.reflection")
        .and_then(|m| m.getattr("get_pretty_function_description"))
        .and_then(|f| f.call1((func.bind(py),)))
        .and_then(|r| r.extract::<String>())
    {
        return desc;
    }
    if let Ok(name) = func.bind(py).getattr("__name__").and_then(|n| n.extract::<String>()) {
        if name != "<lambda>" && !name.is_empty() {
            return name;
        }
    }
    "...".to_string()
}

pub(crate) fn repr_node(node: &StrategyNode, py: Python<'_>) -> PyResult<String> {
    Ok(match node {
        StrategyNode::Integers { min, max } => match (min, max) {
            (None, None) => "integers()".to_string(),
            (Some(a), None) => format!("integers(min_value={a})"),
            (None, Some(b)) => format!("integers(max_value={b})"),
            (Some(a), Some(b)) => format!("integers({a}, {b})"),
        },
        StrategyNode::Booleans => "booleans()".to_string(),
        StrategyNode::Floats { .. } => "floats()".to_string(),
        StrategyNode::NoneVal => "none()".to_string(),
        StrategyNode::Nothing => "nothing()".to_string(),
        StrategyNode::Just(v) => format!("just({})", v.bind(py).repr()?.to_str()?),
        StrategyNode::Text { .. } => "text()".to_string(),
        StrategyNode::Characters { repr, .. } => repr.clone(),
        StrategyNode::Binary { .. } => "binary()".to_string(),
        StrategyNode::Lists { elem, .. } => format!("lists({})", child_repr(elem, py)?),
        StrategyNode::Sets { elem, frozen, .. } => {
            let name = if *frozen { "frozensets" } else { "sets" };
            format!("{name}({})", child_repr(elem, py)?)
        }
        StrategyNode::Dictionaries { keys, values, .. } => {
            format!(
                "dictionaries(keys={}, values={})",
                child_repr(keys, py)?,
                child_repr(values, py)?
            )
        }
        StrategyNode::Tuples(children) => {
            let parts: Vec<String> =
                children.iter().map(|c| child_repr(c, py)).collect::<PyResult<_>>()?;
            format!("tuples({})", parts.join(", "))
        }
        StrategyNode::OneOf(children) => {
            let parts: Vec<String> =
                children.iter().map(|c| child_repr(c, py)).collect::<PyResult<_>>()?;
            format!("one_of({})", parts.join(", "))
        }
        StrategyNode::SampledFrom { elements, is_tuple } => {
            // nicerepr renders types/functions by name (`int`, not `<class 'int'>`), matching
            // upstream's sampled_from repr — e.g. the TypeVar union `sampled_from([NoneType,
            // bool, int, float, str, bytes])`.
            let elems: Vec<String> =
                elements.iter().map(|e| nice_repr(e.bind(py), py)).collect();
            // Preserve the original container type in the repr: tuples render with
            // parens, everything else (list/range/...) with brackets, matching upstream
            // (test_preserves_sequence_type_of_argument). A 1-tuple needs a trailing comma.
            if *is_tuple {
                if elements.len() == 1 {
                    format!("sampled_from(({},))", elems[0])
                } else {
                    format!("sampled_from(({}))", elems.join(", "))
                }
            } else {
                format!("sampled_from([{}])", elems.join(", "))
            }
        }
        StrategyNode::SampledFromRange { range } => {
            format!("sampled_from({})", range.bind(py).repr()?)
        }
        StrategyNode::Map { base, func } => {
            format!("{}.map({})", child_repr(base, py)?, fn_repr(func, py))
        }
        StrategyNode::Filter { base, func } => {
            format!("{}.filter({})", child_repr(base, py)?, fn_repr(func, py))
        }
        StrategyNode::Flatmap { base, func } => {
            format!("{}.flatmap({})", child_repr(base, py)?, fn_repr(func, py))
        }
        StrategyNode::Builds { target, args, kwargs } => {
            let tname = fn_repr(target, py);
            let mut parts: Vec<String> = Vec::new();
            for a in args {
                parts.push(child_repr(a, py)?);
            }
            for (k, v) in kwargs {
                parts.push(format!("{k}={}", child_repr(v, py)?));
            }
            if parts.is_empty() {
                format!("builds({tname})")
            } else {
                format!("builds({tname}, {})", parts.join(", "))
            }
        }
        StrategyNode::Uuids { .. } => "uuids()".to_string(),
        StrategyNode::Dates { .. } => "dates()".to_string(),
        StrategyNode::Times { .. } => "times()".to_string(),
        StrategyNode::Datetimes { .. } => "datetimes()".to_string(),
        StrategyNode::Timedeltas { .. } => "timedeltas()".to_string(),
        StrategyNode::ComplexNumbers { .. } => "complex_numbers()".to_string(),
        StrategyNode::Fractions { .. } => "fractions()".to_string(),
        StrategyNode::Decimals { .. } => "decimals()".to_string(),
        StrategyNode::Data => "data()".to_string(),
        StrategyNode::Randoms { .. } => "randoms()".to_string(),
        StrategyNode::Functions { like, returns, pure } => {
            // upstream functions() repr: `functions(like=<src>, returns=<strat>, pure=<bool>)`.
            format!(
                "functions(like={}, returns={}, pure={})",
                fn_repr(like, py),
                child_repr(returns, py)?,
                if *pure { "True" } else { "False" }
            )
        }
        StrategyNode::Shared { base, key } => {
            // upstream: `shared(<base>, key=<key!r>)`.
            let key_repr = pyo3::types::PyString::new(py, key)
                .repr()
                .and_then(|r| r.extract::<String>())
                .unwrap_or_else(|_| format!("'{key}'"));
            format!("shared({}, key={})", child_repr(base, py)?, key_repr)
        }
        _ => "SearchStrategy(...)".to_string(),
    })
}

/// Element repr matching upstream `nicerepr`: types/functions by name (`int`, not
/// `<class 'int'>`), everything else by `repr`. Falls back to plain repr.
fn nice_repr(e: &Bound<'_, PyAny>, py: Python<'_>) -> String {
    if let Ok(s) = py
        .import("hypothesis.internal.reflection")
        .and_then(|m| m.getattr("nicerepr"))
        .and_then(|f| f.call1((e,)))
        .and_then(|r| r.extract::<String>())
    {
        return s;
    }
    e.repr().and_then(|r| r.extract::<String>()).unwrap_or_default()
}

// ---- constructor pyfunctions ------------------------------------------------
