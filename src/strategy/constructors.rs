//! Strategy constructor #[pyfunction]s (collections/text/characters/datetimes/
//! builds/ip/regex-alphabet) + their validation helpers. Split out of strategy/mod.rs.
#![allow(clippy::wildcard_imports)]
use super::*;



#[pyfunction]
#[pyo3(name = "data")]
pub(crate) fn data(py: Python<'_>) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Data)
}

/// Build a functions() strategy node. `returns` must already be a native
/// SearchStrategy and `pure` a bool — the Python `functions()` wrapper validates
/// args and infers `returns` before calling this.
#[pyfunction]
#[pyo3(name = "randoms", signature = (*, note_method_calls=None, use_true_random=None))]
pub(crate) fn randoms(
    py: Python<'_>,
    // Accept any so a non-bool (use_true_random='False') raises InvalidArgument via
    // check_bool_flag, not a bare PyO3 arg-parse TypeError.
    note_method_calls: Option<Bound<'_, PyAny>>,
    use_true_random: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let note_method_calls = match note_method_calls {
        Some(v) => check_bool_flag(py, &v, "note_method_calls")?,
        None => false,
    };
    let use_true_random = match use_true_random {
        Some(v) => check_bool_flag(py, &v, "use_true_random")?,
        None => false,
    };
    SearchStrategy::wrap(py, StrategyNode::Randoms { use_true_random, note_method_calls })
}

#[pyfunction]
#[pyo3(name = "_functions_strategy")]
pub(crate) fn functions_strategy(
    py: Python<'_>,
    like: Py<PyAny>,
    returns: Py<PyAny>,
    pure: bool,
) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Functions { like, returns, pure })
}

#[pyfunction]
#[pyo3(name = "iterables", signature = (elements, *, min_size=None, max_size=None, unique_by=None, unique=false))]
pub(crate) fn iterables(
    py: Python<'_>,
    elements: Py<PyAny>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
    unique_by: Option<Py<PyAny>>,
    unique: bool,
) -> PyResult<Py<PyAny>> {
    let lst = lists(py, elements, min_size, max_size, unique_by, unique)?;
    // Wrap in _PrettyIter so the drawn iterator has a useful `iter([...])` repr + `_values`.
    let iter_fn = py
        .import("hypothesis_fast.native_strategies")?
        .getattr("_PrettyIter")?
        .unbind();
    SearchStrategy::wrap(py, StrategyNode::Map { base: lst, func: iter_fn })
}

#[pyfunction]
#[pyo3(name = "emails")]
pub(crate) fn emails(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let alpha = pyo3::types::PyString::new(py, "abcdefghijklmnopqrstuvwxyz0123456789").into_any();
    let one = || -> PyResult<Option<Bound<'_, PyAny>>> { Ok(Some(1i64.into_pyobject(py)?.into_any())) };
    let ten = || -> PyResult<Option<Bound<'_, PyAny>>> { Ok(Some(10i64.into_pyobject(py)?.into_any())) };
    let local = text(py, Some(alpha.clone()), one()?, ten()?)?;
    let domain = text(py, Some(alpha), one()?, ten()?)?;
    let pair = tuples(py, PyTuple::new(py, [local.bind(py), domain.bind(py)])?)?;
    let join = py
        .eval(c"(lambda t: t[0] + '@' + t[1] + '.test')", None, None)?
        .unbind();
    SearchStrategy::wrap(py, StrategyNode::Map { base: pair, func: join })
}

#[pyfunction]
#[pyo3(name = "recursive", signature = (base, extend, *, max_leaves=50))]
pub(crate) fn recursive(
    py: Python<'_>,
    base: Py<PyAny>,
    extend: Py<PyAny>,
    max_leaves: usize,
) -> PyResult<Py<PyAny>> {
    // extend=lambda x: x is a no-op (the recursion never deepens) — warn, matching upstream.
    if py
        .import("hypothesis_fast.strategies")?
        .getattr("_is_identity_function")?
        .call1((extend.bind(py),))?
        .is_truthy()?
    {
        let hw = py.import("hypothesis_fast.errors")?.getattr("HypothesisWarning")?;
        py.import("warnings")?
            .getattr("warn")?
            .call1(("extend=lambda x: x is a no-op (the recursive() strategy will never extend)", hw))?;
    }
    // Compose a bounded one_of tree: base | extend(base) | extend(one_of(base, extend(base))) | ...
    let depth = ((max_leaves.max(2) as f64).log2().ceil() as usize).max(1);
    let mut strat: Py<PyAny> = base.clone_ref(py);
    for _ in 0..depth {
        let extended = extend.bind(py).call1((strat.bind(py),))?;
        strat = SearchStrategy::wrap(
            py,
            StrategyNode::OneOf(vec![base.clone_ref(py), extended.unbind()]),
        )?;
    }
    Ok(strat)
}

#[pyfunction]
#[pyo3(name = "sets", signature = (elements, *, min_size=None, max_size=None))]
pub(crate) fn sets(
    py: Python<'_>,
    elements: Py<PyAny>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let er = elements.bind(py).repr().map(|s| s.to_string()).unwrap_or_default();
    let (min, max) = match collection_sizes(py, min_size, max_size, "sets", strat_is_empty(py, &elements), &er) {
        Ok(v) => v,
        Err(m) => return deferred_invalid(py, m),
    };
    SearchStrategy::wrap(py, StrategyNode::Sets { elem: elements, min, max, frozen: false })
}

#[pyfunction]
#[pyo3(name = "frozensets", signature = (elements, *, min_size=None, max_size=None))]
pub(crate) fn frozensets(
    py: Python<'_>,
    elements: Py<PyAny>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let er = elements.bind(py).repr().map(|s| s.to_string()).unwrap_or_default();
    let (min, max) = match collection_sizes(py, min_size, max_size, "frozensets", strat_is_empty(py, &elements), &er) {
        Ok(v) => v,
        Err(m) => return deferred_invalid(py, m),
    };
    SearchStrategy::wrap(py, StrategyNode::Sets { elem: elements, min, max, frozen: true })
}

#[pyfunction]
#[pyo3(name = "dictionaries", signature = (keys, values, *, min_size=None, max_size=None))]
pub(crate) fn dictionaries(
    py: Python<'_>,
    keys: Py<PyAny>,
    values: Py<PyAny>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    check_strategy(py, &keys.bind(py), "dictionaries() keys")?;
    check_strategy(py, &values.bind(py), "dictionaries() values")?;
    let empty = strat_is_empty(py, &keys) || strat_is_empty(py, &values);
    let kr = keys.bind(py).repr().map(|s| s.to_string()).unwrap_or_default();
    let (min, max) = match collection_sizes(py, min_size, max_size, "dictionaries", empty, &kr) {
        Ok(v) => v,
        Err(m) => return deferred_invalid(py, m),
    };
    SearchStrategy::wrap(py, StrategyNode::Dictionaries { keys, values, min, max })
}

#[pyfunction]
#[pyo3(name = "fixed_dictionaries", signature = (mapping, *, optional=None))]
pub(crate) fn fixed_dictionaries(
    py: Python<'_>,
    mapping: Bound<'_, PyAny>,
    optional: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // mapping must be a dict (subclasses like OrderedDict are fine); a non-dict (mapping='fish')
    // raises InvalidArgument, not the bare TypeError the old `Bound<PyDict>` signature gave.
    let map_dict = mapping.downcast::<PyDict>().map_err(|_| {
        invalid_argument(
            py,
            format!(
                "mapping={} must be a dict mapping keys to strategies.",
                mapping.repr().and_then(|r| r.extract::<String>()).unwrap_or_default()
            ),
        )
    })?;
    if let Some(opt) = &optional {
        if !opt.is_none() {
            let opt_dict = opt.downcast::<PyDict>().map_err(|_| {
                invalid_argument(
                    py,
                    format!(
                        "optional={} must be a dict mapping keys to strategies.",
                        opt.repr().and_then(|r| r.extract::<String>()).unwrap_or_default()
                    ),
                )
            })?;
            // mapping and optional must be the SAME concrete type (dict vs OrderedDict mix is
            // rejected, since the result type is taken from the inputs).
            if !mapping.get_type().is(&opt.get_type()) {
                return Err(invalid_argument(
                    py,
                    format!(
                        "Got arguments of different types: mapping={}, optional={}",
                        mapping.get_type().name()?,
                        opt.get_type().name()?
                    ),
                ));
            }
            for (k, v) in opt_dict.iter() {
                if map_dict.contains(&k)? {
                    return Err(invalid_argument(
                        py,
                        format!(
                            "The following keys were in both mapping and optional, which is \
                             invalid: {{{}}}",
                            k.repr()?
                        ),
                    ));
                }
                check_strategy(py, &v, "fixed_dictionaries() optional value")?;
            }
        }
    }
    let mut items = Vec::new();
    for (k, v) in map_dict.iter() {
        check_strategy(py, &v, "fixed_dictionaries() value")?;
        items.push((k.unbind(), v.unbind()));
    }
    SearchStrategy::wrap(py, StrategyNode::FixedDict { items })
}

#[pyfunction]
#[pyo3(name = "binary", signature = (*, min_size=None, max_size=None))]
pub(crate) fn binary(
    py: Python<'_>,
    min_size: Option<Bound<'_, PyAny>>,
    max_size: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let (min, max) = match collection_sizes(py, min_size, max_size, "binary", false, "") {
        Ok(v) => v,
        Err(m) => return deferred_invalid(py, m),
    };
    SearchStrategy::wrap(py, StrategyNode::Binary { min, max })
}

#[pyfunction]
#[pyo3(name = "uuids", signature = (*, version=None, allow_nil=None))]
pub(crate) fn uuids(
    py: Python<'_>,
    version: Option<u8>,
    allow_nil: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let invalid = |msg: String| -> PyResult<PyErr> {
        Ok(PyErr::from_value(
            py.import("hypothesis_fast.errors")?
                .getattr("InvalidArgument")?
                .call1((msg,))?,
        ))
    };
    let allow_nil: bool = match allow_nil {
        None => false,
        Some(obj) => {
            if !obj.is_instance_of::<pyo3::types::PyBool>() {
                return Err(invalid(format!(
                    "allow_nil must be a boolean value, got allow_nil={}",
                    obj.repr()?
                ))?);
            }
            obj.extract()?
        }
    };
    if let Some(v) = version {
        if !(1..=5).contains(&v) {
            return Err(invalid(format!(
                "version={v} is not valid; use a version between 1 and 5, or None"
            ))?);
        }
    }
    if allow_nil && version.is_some() {
        return Err(invalid(
            "The nil UUID is not of any version, so you must use version=None".to_string(),
        )?);
    }
    SearchStrategy::wrap(py, StrategyNode::Uuids { version, allow_nil })
}

#[pyfunction]
#[pyo3(name = "permutations")]
pub(crate) fn permutations(py: Python<'_>, values: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    check_ordered_sample(py, &values, "permutations")?;
    let mut out = Vec::new();
    for it in values.try_iter()? {
        out.push(it?.unbind());
    }
    SearchStrategy::wrap(py, StrategyNode::Permutations(out))
}

#[pyfunction]
#[pyo3(name = "builds", signature = (*args, **kwargs))]
pub(crate) fn builds(
    py: Python<'_>,
    args: Bound<'_, PyTuple>,
    kwargs: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let kw = match kwargs {
        Some(d) => d,
        None => PyDict::new(py),
    };
    // The target is the FIRST positional arg. builds() takes it via *args (NOT a named
    // `target` param), so a kwarg field literally named `target` doesn't collide with it —
    // builds(NamedTupleWithTargetField, target=integers()) works (test_build_class_with_
    // target_kwarg). The remaining positionals are argument strategies.
    if args.is_empty() {
        return Err(invalid_argument(
            py,
            "builds() must be passed a callable as the first positional argument, but \
             no positional arguments were given."
                .to_string(),
        ));
    }
    let tb = args.get_item(0)?;
    let rest = args.get_slice(1, args.len());
    // Per-run cache of the constructed builds() strategy, keyed by target + arg/kwarg object
    // identities (computed BEFORE inference mutates `kw`). Code that rebuilds builds(T, **fixed)
    // every draw (hypothesmith) gets an identical strategy, so reuse it — skipping the callable
    // check, arg inference, and node allocation. Registry-invalidated via _clear_resolution_caches.
    let cache_key = {
        let mut k = format!("{:x}", tb.as_ptr() as usize);
        for a in rest.iter() {
            k.push('|');
            k.push_str(&format!("{:x}", a.as_ptr() as usize));
        }
        let mut kvs: Vec<(String, usize)> = Vec::with_capacity(kw.len());
        for (kk, vv) in kw.iter() {
            kvs.push((kk.extract::<String>()?, vv.as_ptr() as usize));
        }
        kvs.sort();
        for (kk, vp) in &kvs {
            k.push('#');
            k.push_str(kk);
            k.push('=');
            k.push_str(&format!("{:x}", vp));
        }
        k
    };
    if let Some(s) = crate::data::builds_strategy_get(py, &cache_key) {
        return Ok(s);
    }
    // The target must be callable. A non-callable (e.g. a PEP-604 / typing Union, no longer
    // callable on 3.14+) can't be constructed — point the user at from_type, like upstream.
    let builtins = py.import("builtins")?;
    if !builtins.getattr("callable")?.call1((&tb,))?.is_truthy()? {
        let typing = py.import("typing")?;
        let is_union = typing
            .getattr("get_origin")
            .and_then(|f| f.call1((&tb,)))
            .map(|o| o.is(&typing.getattr("Union").unwrap()))
            .unwrap_or(false)
            || py
                .import("types")
                .and_then(|m| m.getattr("UnionType"))
                .map(|u| tb.is_instance(&u).unwrap_or(false))
                .unwrap_or(false);
        let suggestion = if is_union {
            format!(" Try using from_type({}) instead?", tb.str()?)
        } else {
            String::new()
        };
        return Err(invalid_argument(
            py,
            format!(
                "The first positional argument to builds() must be a callable target to \
                 construct, but got non-callable {}.{}",
                tb.repr()?,
                suggestion
            ),
        ));
    }
    let target = tb.clone().unbind();
    // hypothesis builds() infers any un-provided REQUIRED parameter from its type
    // hint (from_type), so `builds(Target)` fills annotated required args. Best-effort:
    // skip inference entirely if the target has no introspectable signature.
    infer_builds_kwargs(py, &tb, &rest, &kw)?;
    let mut av = Vec::new();
    for a in rest.iter() {
        av.push(a.unbind());
    }
    let mut kwv = Vec::new();
    for (k, v) in kw.iter() {
        kwv.push((k.extract::<String>()?, v.unbind()));
    }
    let result = SearchStrategy::wrap(py, StrategyNode::Builds { target, args: av, kwargs: kwv })?;
    crate::data::builds_strategy_put(cache_key, result.clone_ref(py));
    Ok(result)
}

/// Whether an annotation contains an unresolved `typing.ForwardRef` (directly or nested
/// in its type args, e.g. `Optional["Tree"]`). Such annotations must be resolved via
/// get_type_hints before use, or from_type would hit the legacy/interop path.
pub(crate) fn ann_has_forwardref(py: Python<'_>, ann: &Bound<'_, PyAny>) -> bool {
    let typing = match py.import("typing") {
        Ok(t) => t,
        Err(_) => return false,
    };
    if let Ok(fref) = typing.getattr("ForwardRef") {
        if ann.is_instance(&fref).unwrap_or(false) {
            return true;
        }
    }
    if let Ok(args) = typing.getattr("get_args").and_then(|f| f.call1((ann,))) {
        if let Ok(t) = args.downcast::<PyTuple>() {
            for a in t.iter() {
                if ann_has_forwardref(py, &a) {
                    return true;
                }
            }
        }
    }
    false
}

/// Augment `kw` in place with inferred strategies for each REQUIRED parameter of
/// `target` that is not already covered by a positional `args` slot or a `kw` entry.
/// Mirrors hypothesis builds()'s infer-by-default. Unintrospectable targets, var-args,
/// positional-only-without-name, and unannotated optionals are left untouched.
pub(crate) fn infer_builds_kwargs(
    py: Python<'_>,
    target: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kw: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let inspect = py.import("inspect")?;
    let typing = py.import("typing")?;
    let builtins = py.import("builtins")?;
    let eng = py.import("hypothesis_fast._engine")?;
    let ellipsis = builtins.getattr("Ellipsis")?;
    // `...` (Ellipsis) is builds()'s "infer from annotation" sentinel — only valid as a
    // keyword argument, never positional.
    for a in args.iter() {
        if a.is(&ellipsis) {
            return Err(invalid_argument(
                py,
                "... was passed as a positional argument to builds(), but is only allowed \
                 as a keyword argument"
                    .to_string(),
            ));
        }
    }
    // Inference is cacheable when there are no positional args and no kwarg is the `...`
    // infer-sentinel: then the inferred required-arg additions depend only on the target +
    // explicit-key set (+ the run-stable registry). builds_filtering-style code rebuilds
    // builds(T) every draw, so this skips a per-draw signature + get_type_hints + from_type
    // pass (the dominant generation cost there). Saved/restored per run.
    let mut cacheable = args.is_empty();
    if cacheable {
        for (_k, v) in kw.iter() {
            if v.is(&ellipsis) {
                cacheable = false;
                break;
            }
        }
    }
    let cache_key: Option<String> = if cacheable {
        let mut keys: Vec<String> = Vec::with_capacity(kw.len());
        for k in kw.keys().iter() {
            keys.push(k.extract::<String>()?);
        }
        keys.sort();
        Some(format!("{:x}|{}", target.as_ptr() as usize, keys.join("\u{1}")))
    } else {
        None
    };
    if let Some(k) = &cache_key {
        if let Some(adds) = crate::data::builds_infer_get(py, k) {
            for (name, strat) in adds {
                kw.set_item(name, strat.bind(py))?;
            }
            return Ok(());
        }
    }
    // A @no_type_check target must NOT have its arguments inferred (the decorator's whole
    // point) — leave required args unfilled so calling it errors, and builds()'s draw-time
    // handler explains why. Mirrors upstream, where get_type_hints() yields nothing here.
    if target
        .getattr("__no_type_check__")
        .map(|v| v.is_truthy().unwrap_or(false))
        .unwrap_or(false)
    {
        return Ok(());
    }
    let sig = match inspect.getattr("signature")?.call1((target,)) {
        Ok(s) => s,
        Err(_) => {
            // No introspectable signature: we cannot resolve an `...` infer-kwarg.
            for (k, v) in kw.iter() {
                if v.is(&ellipsis) {
                    return Err(invalid_argument(
                        py,
                        format!("Cannot infer a strategy for {k}=... (no signature available)"),
                    ));
                }
            }
            return Ok(());
        }
    };
    // Resolved type hints (ForwardRefs evaluated) for the parameters. For a class we merge
    // the CLASS-level annotations (NamedTuple / dataclass fields live there, with an empty
    // __init__) with __init__'s own hints, letting __init__ win — so e.g. TreeForwardRefs's
    // `l: Optional["TreeForwardRefs"]` fields are still inferred. For a plain callable, just
    // its own hints.
    let is_class = builtins
        .getattr("isinstance")?
        .call1((target, builtins.getattr("type")?))?
        .is_truthy()?;
    let get_hints = typing.getattr("get_type_hints")?;
    let hints: Option<Bound<'_, PyAny>> = if is_class {
        let merged = PyDict::new(py);
        if let Ok(ch) = get_hints.call1((target,)) {
            if let Ok(d) = ch.downcast::<PyDict>() {
                merged.update(d.as_mapping())?;
            }
        }
        if let Ok(init) = target.getattr("__init__") {
            if let Ok(ih) = get_hints.call1((&init,)) {
                if let Ok(d) = ih.downcast::<PyDict>() {
                    merged.update(d.as_mapping())?;
                }
            }
        }
        Some(merged.into_any())
    } else {
        get_hints.call1((target,)).ok()
    };
    let param_cls = inspect.getattr("Parameter")?;
    let empty = param_cls.getattr("empty")?;
    let var_pos = param_cls.getattr("VAR_POSITIONAL")?;
    let var_kw = param_cls.getattr("VAR_KEYWORD")?;
    let kw_only = param_cls.getattr("KEYWORD_ONLY")?;
    let pos_only = param_cls.getattr("POSITIONAL_ONLY")?;
    // While inferring args, treat the TARGET as in-progress so a recursive arg annotation
    // (`next_node: Optional[SomeClass]` for `builds(SomeClass, ...)`) resolves to a
    // deferred — re-resolved at draw time, picking up a strategy registered for the target
    // AFTER this builds() is built (test_resolving_recursive_type_with_registered_constraint).
    let _target_guard = InProgressGuard::enter(target);
    let n_pos = args.len();
    let mut pos_slot = 0usize;
    // Required-arg inferences captured for the per-run memo (only when `cacheable`).
    let mut additions: Vec<(String, Py<PyAny>)> = Vec::new();
    for item in sig.getattr("parameters")?.call_method0("values")?.try_iter()? {
        let p = item?;
        let kind = p.getattr("kind")?;
        if kind.is(&var_pos) || kind.is(&var_kw) {
            continue;
        }
        let is_positional = !kind.is(&kw_only);
        let covered_by_pos = is_positional && pos_slot < n_pos;
        if is_positional {
            pos_slot += 1;
        }
        if covered_by_pos {
            continue;
        }
        let name: String = p.getattr("name")?.extract()?;
        // Prefer the signature parameter's own annotation (so an overriding
        // __signature__ wins, and __init__'s type beats a class-level annotation);
        // fall back to resolved type hints when it's absent, a string (PEP 563), or
        // contains an unresolved ForwardRef (e.g. Optional["Tree"]) — get_type_hints
        // resolves those, and a raw ForwardRef would force the legacy/interop path.
        let p_ann = p.getattr("annotation")?;
        let ann: Option<Bound<'_, PyAny>> = if !p_ann.is(&empty)
            && !p_ann.is_instance_of::<PyString>()
            && !ann_has_forwardref(py, &p_ann)
        {
            Some(p_ann)
        } else {
            hints.as_ref().and_then(|h| h.get_item(&name).ok())
        };
        if kw.contains(&name)? {
            // An explicit `...` kwarg means "infer this param from its annotation"; an
            // unannotated param can't be inferred (matches upstream builds()).
            if kw.get_item(&name)?.is_some_and(|v| v.is(&ellipsis)) {
                match ann {
                    Some(a) => {
                        let mut s = from_type(py, a)?;
                        // A forced-infer param that ALSO has a default should be able to
                        // produce that default too (e.g. `a: str = None` → just(None)|text()).
                        let default = p.getattr("default")?;
                        if !default.is(&empty) {
                            // Inferred strategy FIRST so the native shrinker (which lowers
                            // the one_of branch index toward 0) can still minimise to the
                            // inferred value's minimum (e.g. integers()->0), not the default.
                            let just_d = eng.getattr("just")?.call1((&default,))?;
                            s = eng.getattr("one_of")?.call1((s.bind(py), just_d))?.unbind();
                        }
                        kw.set_item(&name, s.bind(py))?;
                    }
                    None => {
                        return Err(invalid_argument(
                            py,
                            format!("passed ... for {name}, but {name} has no type annotation"),
                        ));
                    }
                }
            }
            continue;
        }
        // Only infer REQUIRED params (no default); optionals keep their default.
        if !p.getattr("default")?.is(&empty) {
            continue;
        }
        if kind.is(&pos_only) {
            continue; // can't supply a positional-only param by keyword
        }
        if let Some(ann) = ann {
            let s = from_type(py, ann)?;
            kw.set_item(&name, s.bind(py))?;
            if cacheable {
                additions.push((name.clone(), s));
            }
        }
    }
    if let Some(k) = cache_key {
        crate::data::builds_infer_put(k, additions);
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "_composite_strategy")]
pub(crate) fn composite_strategy(
    py: Python<'_>,
    func: Py<PyAny>,
    args: Py<PyAny>,
    kwargs: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    SearchStrategy::wrap(py, StrategyNode::Composite { func, args, kwargs })
}

pub(crate) fn ordinal_of(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    v.call_method0("toordinal")?.extract()
}

/// "Interesting" date ordinals to inject during dates()/datetimes() generation: the range
/// endpoints, the millennium, and (when the span is small enough to enumerate) Feb 29 of each
/// leap year plus Jan 1 / Dec 31 of each year. This makes rare targets like "the one leap day
/// in a 3-year range" reliably generated instead of a needle in a uniform-ordinal haystack.
pub(crate) fn date_inject_candidates(
    py: Python<'_>,
    min_ord: i64,
    max_ord: i64,
) -> PyResult<Vec<BigInt>> {
    let mut out: Vec<BigInt> = vec![BigInt::from(min_ord), BigInt::from(max_ord)];
    if MILLENNIUM_ORDINAL > min_ord && MILLENNIUM_ORDINAL < max_ord {
        out.push(BigInt::from(MILLENNIUM_ORDINAL));
    }
    let date_cls = py.import("datetime")?.getattr("date")?;
    let min_year: i64 =
        date_cls.call_method1("fromordinal", (min_ord,))?.getattr("year")?.extract()?;
    let max_year: i64 =
        date_cls.call_method1("fromordinal", (max_ord,))?.getattr("year")?.extract()?;
    if max_year - min_year <= 50 {
        for y in min_year..=max_year {
            for (m, d) in [(2i32, 29i32), (1, 1), (12, 31)] {
                // date(y, 2, 29) raises ValueError in non-leap years — skip those.
                if let Ok(dobj) = date_cls.call1((y, m, d)) {
                    if let Ok(o) = dobj.call_method0("toordinal").and_then(|x| x.extract::<i64>()) {
                        if o >= min_ord && o <= max_ord {
                            out.push(BigInt::from(o));
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn time_to_us(t: &Bound<'_, PyAny>) -> PyResult<i64> {
    let h: i64 = t.getattr("hour")?.extract()?;
    let m: i64 = t.getattr("minute")?.extract()?;
    let s: i64 = t.getattr("second")?.extract()?;
    let micro: i64 = t.getattr("microsecond")?.extract()?;
    Ok(h * 3_600_000_000 + m * 60_000_000 + s * 1_000_000 + micro)
}

pub(crate) fn td_to_us(td: &Bound<'_, PyAny>) -> PyResult<BigInt> {
    let days: i64 = td.getattr("days")?.extract()?;
    let seconds: i64 = td.getattr("seconds")?.extract()?;
    let micro: i64 = td.getattr("microseconds")?.extract()?;
    Ok((BigInt::from(days) * 86400 + seconds) * 1_000_000 + micro)
}

pub(crate) fn dt_attr<'py>(
    py: Python<'py>,
    cls: &str,
    which: &str,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("datetime")?.getattr(cls)?.getattr(which)
}

/// Validate that a datetime-family bound is an instance of `datetime.<cls>`, raising
/// InvalidArgument (not a bare AttributeError from `.toordinal()`/`.hour`) when it isn't —
/// e.g. dates(min_value="fish"). Mirrors upstream's check_type(datetime.date, ...).
pub(crate) fn check_dt_type(py: Python<'_>, v: &Bound<'_, PyAny>, cls: &str, arg_name: &str) -> PyResult<()> {
    let expected = py.import("datetime")?.getattr(cls)?;
    if !v.is_instance(&expected)? {
        return Err(invalid_argument(
            py,
            format!(
                "Expected {arg_name} to be a datetime.{cls}, but got {}",
                v.repr()?.extract::<String>()?,
            ),
        ));
    }
    Ok(())
}

/// Coerce a strictly-bool flag, raising InvalidArgument for a non-bool (e.g. allow_imaginary=0,
/// use_true_random='False'). `0`/`1` are int, not bool (bool subclasses int), so they're rejected
/// — matching upstream's check_type(bool, ...).
pub(crate) fn check_bool_flag(py: Python<'_>, v: &Bound<'_, PyAny>, arg_name: &str) -> PyResult<bool> {
    if v.is_instance_of::<pyo3::types::PyBool>() {
        return v.extract();
    }
    Err(invalid_argument(
        py,
        format!("Expected {arg_name} to be a bool, but got {}", v.repr()?.extract::<String>()?),
    ))
}

#[pyfunction]
#[pyo3(name = "dates", signature = (min_value=None, max_value=None))]
pub(crate) fn dates(
    py: Python<'_>,
    min_value: Option<Bound<'_, PyAny>>,
    max_value: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let minv = match min_value {
        Some(v) => v,
        None => dt_attr(py, "date", "min")?,
    };
    let maxv = match max_value {
        Some(v) => v,
        None => dt_attr(py, "date", "max")?,
    };
    check_dt_type(py, &minv, "date", "min_value")?;
    check_dt_type(py, &maxv, "date", "max_value")?;
    // An inverted range (min > max) has no valid value. Upstream raises InvalidArgument
    // here (check_valid_interval); without this the node would draw an integer in an empty
    // [min_ord..max_ord] range, and the provider's rejection loop spins forever holding the
    // GIL — the per-test timeout can't interrupt native code, so the whole worker wedges.
    if maxv.lt(&minv)? {
        return Err(invalid_argument(
            py,
            format!(
                "Cannot have max_value={} < min_value={}",
                maxv.repr()?.extract::<String>()?,
                minv.repr()?.extract::<String>()?,
            ),
        ));
    }
    if minv.eq(&maxv)? {
        return SearchStrategy::wrap(py, StrategyNode::Just(minv.unbind()));
    }
    SearchStrategy::wrap(
        py,
        StrategyNode::Dates { min_ord: ordinal_of(&minv)?, max_ord: ordinal_of(&maxv)? },
    )
}

#[pyfunction]
#[pyo3(name = "times", signature = (min_value=None, max_value=None, *, timezones=None))]
pub(crate) fn times(
    py: Python<'_>,
    min_value: Option<Bound<'_, PyAny>>,
    max_value: Option<Bound<'_, PyAny>>,
    timezones: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = timezones;
    let minv = match min_value {
        Some(v) => v,
        None => dt_attr(py, "time", "min")?,
    };
    let maxv = match max_value {
        Some(v) => v,
        None => dt_attr(py, "time", "max")?,
    };
    check_dt_type(py, &minv, "time", "min_value")?;
    check_dt_type(py, &maxv, "time", "max_value")?;
    // An inverted range (min > max) has no valid value. Upstream raises InvalidArgument
    // here (check_valid_interval); without this the node would draw an integer in an empty
    // [min_ord..max_ord] range, and the provider's rejection loop spins forever holding the
    // GIL — the per-test timeout can't interrupt native code, so the whole worker wedges.
    if maxv.lt(&minv)? {
        return Err(invalid_argument(
            py,
            format!(
                "Cannot have max_value={} < min_value={}",
                maxv.repr()?.extract::<String>()?,
                minv.repr()?.extract::<String>()?,
            ),
        ));
    }
    if minv.eq(&maxv)? {
        return SearchStrategy::wrap(py, StrategyNode::Just(minv.unbind()));
    }
    SearchStrategy::wrap(
        py,
        StrategyNode::Times { min_us: time_to_us(&minv)?, max_us: time_to_us(&maxv)? },
    )
}

#[pyfunction]
#[pyo3(name = "datetimes", signature = (min_value=None, max_value=None, *, timezones=None, allow_imaginary=None))]
pub(crate) fn datetimes(
    py: Python<'_>,
    min_value: Option<Bound<'_, PyAny>>,
    max_value: Option<Bound<'_, PyAny>>,
    timezones: Option<Bound<'_, PyAny>>,
    // Accept any value so a non-bool (allow_imaginary=0) raises InvalidArgument via
    // check_bool_flag, not a bare PyO3 arg-parse TypeError. None = the default (True).
    allow_imaginary: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _ = timezones;
    if let Some(ai) = allow_imaginary {
        check_bool_flag(py, &ai, "allow_imaginary")?;
    }
    let minv = match min_value {
        Some(v) => v,
        None => dt_attr(py, "datetime", "min")?,
    };
    let maxv = match max_value {
        Some(v) => v,
        None => dt_attr(py, "datetime", "max")?,
    };
    check_dt_type(py, &minv, "datetime", "min_value")?;
    check_dt_type(py, &maxv, "datetime", "max_value")?;
    // An inverted range (min > max) has no valid value. Upstream raises InvalidArgument
    // here (check_valid_interval); without this the node would draw an integer in an empty
    // [min_ord..max_ord] range, and the provider's rejection loop spins forever holding the
    // GIL — the per-test timeout can't interrupt native code, so the whole worker wedges.
    if maxv.lt(&minv)? {
        return Err(invalid_argument(
            py,
            format!(
                "Cannot have max_value={} < min_value={}",
                maxv.repr()?.extract::<String>()?,
                minv.repr()?.extract::<String>()?,
            ),
        ));
    }
    if minv.eq(&maxv)? {
        return SearchStrategy::wrap(py, StrategyNode::Just(minv.unbind()));
    }
    let min_ord = ordinal_of(&minv.call_method0("date")?)?;
    let max_ord = ordinal_of(&maxv.call_method0("date")?)?;
    SearchStrategy::wrap(py, StrategyNode::Datetimes { min_ord, max_ord })
}

#[pyfunction]
#[pyo3(name = "timedeltas", signature = (min_value=None, max_value=None))]
pub(crate) fn timedeltas(
    py: Python<'_>,
    min_value: Option<Bound<'_, PyAny>>,
    max_value: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let minv = match min_value {
        Some(v) => v,
        None => dt_attr(py, "timedelta", "min")?,
    };
    let maxv = match max_value {
        Some(v) => v,
        None => dt_attr(py, "timedelta", "max")?,
    };
    check_dt_type(py, &minv, "timedelta", "min_value")?;
    check_dt_type(py, &maxv, "timedelta", "max_value")?;
    // An inverted range (min > max) has no valid value. Upstream raises InvalidArgument
    // here (check_valid_interval); without this the node would draw an integer in an empty
    // [min_ord..max_ord] range, and the provider's rejection loop spins forever holding the
    // GIL — the per-test timeout can't interrupt native code, so the whole worker wedges.
    if maxv.lt(&minv)? {
        return Err(invalid_argument(
            py,
            format!(
                "Cannot have max_value={} < min_value={}",
                maxv.repr()?.extract::<String>()?,
                minv.repr()?.extract::<String>()?,
            ),
        ));
    }
    if minv.eq(&maxv)? {
        return SearchStrategy::wrap(py, StrategyNode::Just(minv.unbind()));
    }
    SearchStrategy::wrap(
        py,
        StrategyNode::Timedeltas { min_us: td_to_us(&minv)?, max_us: td_to_us(&maxv)? },
    )
}

#[pyfunction]
#[pyo3(name = "complex_numbers", signature = (*, min_magnitude=None, max_magnitude=None, allow_nan=None, allow_infinity=None, width=128))]
pub(crate) fn complex_numbers(
    py: Python<'_>,
    min_magnitude: Option<Bound<'_, PyAny>>,
    max_magnitude: Option<Bound<'_, PyAny>>,
    allow_nan: Option<bool>,
    allow_infinity: Option<bool>,
    width: u32,
) -> PyResult<Py<PyAny>> {
    let _ = (min_magnitude, max_magnitude, allow_infinity, width);
    SearchStrategy::wrap(
        py,
        StrategyNode::ComplexNumbers { allow_nan: allow_nan.unwrap_or(true) },
    )
}

#[pyfunction]
#[pyo3(name = "ip_addresses", signature = (*, v=None, network=None))]
pub(crate) fn ip_addresses(
    py: Python<'_>,
    v: Option<Bound<'_, PyAny>>,
    network: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    // v must be the int 4 or 6 (v='4'/4.0/5 are invalid).
    let vnum: Option<u8> = match &v {
        None => None,
        Some(vv) if vv.is_none() => None,
        Some(vv) => {
            let n = match vv.extract::<i64>() {
                Ok(n) if vv.is_instance_of::<pyo3::types::PyInt>() => n,
                _ => {
                    return Err(invalid_argument(
                        py,
                        format!("v={}, but only v=4 or v=6 are valid.", vv.repr()?.extract::<String>()?),
                    ));
                }
            };
            if n != 4 && n != 6 {
                return Err(invalid_argument(py, format!("v={n}, but only v=4 or v=6 are valid.")));
            }
            Some(n as u8)
        }
    };
    // network: parse via ipaddress.ip_network; an unparseable value (e.g. bytes) and a version
    // mismatch with `v` are both InvalidArgument. When valid, capture its address range so
    // generation draws an address WITHIN the network.
    let net_info: Option<(bool, BigInt, BigInt)> = match &network {
        None => None,
        Some(net) if net.is_none() => None,
        Some(net) => {
            let net_obj = match py.import("ipaddress")?.getattr("ip_network")?.call1((net,)) {
                Ok(o) => o,
                Err(_) => {
                    return Err(invalid_argument(
                        py,
                        format!(
                            "network={} is not a valid IP network.",
                            net.repr()?.extract::<String>()?
                        ),
                    ));
                }
            };
            let ver: u8 = net_obj.getattr("version")?.extract()?;
            if let Some(vn) = vnum {
                if vn != ver {
                    return Err(invalid_argument(
                        py,
                        format!(
                            "v={vn} is incompatible with network={}",
                            net_obj.repr()?.extract::<String>()?
                        ),
                    ));
                }
            }
            let to_int = |a: Bound<'_, PyAny>| -> PyResult<BigInt> {
                py.import("builtins")?.getattr("int")?.call1((a,))?.extract()
            };
            let lo = to_int(net_obj.getattr("network_address")?)?;
            let hi = to_int(net_obj.getattr("broadcast_address")?)?;
            Some((ver == 6, lo, hi))
        }
    };
    let v6 = net_info
        .as_ref()
        .map(|(v6, _, _)| *v6)
        .or_else(|| vnum.map(|x| x == 6));
    let net_range = net_info.map(|(_, lo, hi)| (lo, hi));
    SearchStrategy::wrap(py, StrategyNode::IpAddresses { v6, net_range })
}

/// `hash(obj)` succeeds. Used as a `.filter` predicate on set elements / dict
/// keys when resolving `set[X]`/`dict[X, V]` via `from_type`, mirroring upstream
/// `_from_hashable_type`: a type may be a `Hashable` subclass yet have
/// unhashable instances (e.g. `Decimal('snan')`), which must be filtered out of
/// hashed containers.
#[pyfunction]
pub(crate) fn can_hash(obj: &Bound<'_, PyAny>) -> bool {
    obj.hash().is_ok()
}

/// Wrap an element/key strategy with `.filter(can_hash)` so `set[X]`/`dict[X,V]`
/// only draw hashable members (upstream `_from_hashable_type`). Skipped for the
/// ALWAYS_HASHABLE_TYPES ({NoneType, bool, int, float, complex, str, bytes}) — their
/// instances are always hashable, so the filter is a no-op; skipping it matches
/// upstream for performance AND repr parity (`set[int]` reprs `sets(integers())`,
/// not `sets(integers().filter(can_hash))`).
pub(crate) fn filtered_hashable<'py>(
    eng: &Bound<'py, PyModule>,
    py: Python<'py>,
    type_arg: &Bound<'py, PyAny>,
    strat: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let builtins = py.import("builtins")?;
    let none_type = builtins.getattr("type")?.call1((py.None(),))?;
    let is_always = type_arg.is(&none_type)
        || ["bool", "int", "float", "complex", "str", "bytes"]
            .iter()
            .any(|n| builtins.getattr(*n).map(|t| type_arg.is(&t)).unwrap_or(false));
    if is_always {
        return Ok(strat.clone());
    }
    let pred = eng.getattr("can_hash")?;
    strat.call_method1("filter", (pred,))
}

/// Recursively resolve a from_regex `alphabet=` strategy to its allowed-codepoint IntervalSet,
/// unioning across one_of(...). characters()/sampled_from()-of-chars/just(char)/one_of(of those)
/// resolve; an opaque strategy (builds/map/filter/...) returns None, which the caller turns into
/// InvalidArgument — matching upstream (which accepts any char-producing strategy but not builds).
pub(crate) fn resolve_alphabet_iset(
    py: Python<'_>,
    alphabet: &Bound<'_, PyAny>,
) -> PyResult<Option<crate::intervalset::IntervalSet>> {
    use crate::intervalset::IntervalSet;
    let ss = match alphabet.downcast::<SearchStrategy>() {
        Ok(s) => s,
        Err(_) => {
            // A REAL-hypothesis characters/text strategy (possibly a LazyStrategy wrapper)
            // exposes an IntervalSet via `.intervals`, whose own `.intervals` is a tuple of
            // inclusive (start, end) pairs. Unwrap + read those so `from_regex(alphabet=<real
            // characters(...)>)` works (hypothesis-jsonschema passes a real characters alphabet).
            let unwrapped = py
                .import("hypothesis.strategies._internal.lazy")
                .and_then(|m| m.getattr("unwrap_strategies"))
                .and_then(|f| f.call1((alphabet,)))
                .unwrap_or_else(|_| alphabet.clone());
            if let Ok(iset) = unwrapped.getattr("intervals") {
                if let Ok(pairs_obj) = iset.getattr("intervals") {
                    if let Ok(it) = pairs_obj.try_iter() {
                        let mut pairs: Vec<(i64, i64)> = Vec::new();
                        for pair in it {
                            let pair = pair?;
                            pairs.push((
                                pair.get_item(0)?.extract()?,
                                pair.get_item(1)?.extract()?,
                            ));
                        }
                        return Ok(Some(crate::intervalset::IntervalSet::from_pairs(pairs)));
                    }
                }
            }
            return Ok(None);
        }
    };
    let ssb = ss.borrow();
    match &ssb.node {
        StrategyNode::Characters { intervals, .. } => {
            let b = intervals.bind(py);
            let iset = b.downcast::<IntervalSet>()?.borrow();
            Ok(Some(IntervalSet::from_string_rs("").union_rs(&iset)))
        }
        StrategyNode::SampledFrom { elements, .. } => {
            let mut s = String::new();
            for e in elements {
                match e.bind(py).extract::<String>() {
                    Ok(c) if c.chars().count() == 1 => s.push_str(&c),
                    _ => return Ok(None),
                }
            }
            Ok(Some(IntervalSet::from_string_rs(&s)))
        }
        StrategyNode::Just(v) => match v.bind(py).extract::<String>() {
            Ok(c) if c.chars().count() == 1 => Ok(Some(IntervalSet::from_string_rs(&c))),
            _ => Ok(None),
        },
        StrategyNode::OneOf(children) => {
            let mut acc: Option<IntervalSet> = None;
            for child in children {
                match resolve_alphabet_iset(py, &child.bind(py))? {
                    Some(iset) => {
                        acc = Some(match acc {
                            Some(a) => a.union_rs(&iset),
                            None => iset,
                        });
                    }
                    None => return Ok(None),
                }
            }
            Ok(acc)
        }
        _ => Ok(None),
    }
}

#[pyfunction]
#[pyo3(name = "_regex_alphabet_intervals")]
pub(crate) fn regex_alphabet_intervals(py: Python<'_>, alphabet: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    match resolve_alphabet_iset(py, &alphabet)? {
        Some(iset) => Ok(Py::new(py, iset)?.into_any()),
        None => Err(invalid_argument(
            py,
            format!(
                "alphabet={} must be a sampled_from() or characters() strategy.",
                alphabet.repr().and_then(|r| r.extract::<String>()).unwrap_or_default()
            ),
        )),
    }
}
