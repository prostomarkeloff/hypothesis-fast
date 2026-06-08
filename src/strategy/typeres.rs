//! from_type / register_type_strategy: native type-annotation resolution.
//! Split out of strategy/mod.rs.
#![allow(clippy::wildcard_imports)]
use super::*;

/// Resolve a type/typing-annotation to a NATIVE strategy. Handles the common
/// builtins, parametrized containers, and Union/Optional by composing native
/// constructors; defers to the legacy `from_type` for registered types and anything
/// Resolve a `_NATIVE_TYPE_REGISTRY` entry to a strategy: a stored SearchStrategy is
/// returned directly; a factory is called with the type and must return a SearchStrategy
/// — UNLESS it returns NotImplemented, which means "I decline" → `Ok(None)` so the caller
/// falls through to other resolution (matches hypothesis's conditional resolvers).
fn resolve_registry_entry(
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
    thing: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    // A registered (or factory-produced) strategy that resolves EMPTY is a resolution
    // failure, not a usable strategy — upstream's `as_strategy` raises ResolutionFailed.
    let empty_guard = |py: Python<'_>, s: Py<PyAny>| -> PyResult<Py<PyAny>> {
        if let Ok(ss) = s.bind(py).downcast::<SearchStrategy>() {
            if node_is_empty(&ss.borrow().node, py)? {
                return deferred_resolution_failed(
                    py,
                    format!("Error: {} resolved to an empty strategy", thing.repr()?),
                );
            }
        }
        Ok(s)
    };
    // Accept a native SearchStrategy OR any drawable real-hypothesis strategy (has `do_draw`):
    // a registered factory may return a REAL strategy (e.g. the returns plugin builds a real
    // `one_of(builds(...))`), which is drawable against our native cd via the reverse interop
    // bridge — so it's a valid resolution, not a "non-strategy".
    let is_strategy = |o: &Bound<'_, PyAny>| {
        o.is_instance_of::<SearchStrategy>() || o.hasattr("do_draw").unwrap_or(false)
    };
    if is_strategy(entry) {
        return Ok(Some(empty_guard(py, entry.clone().unbind())?));
    }
    let result = entry.call1((thing,))?;
    if result.is(&py.NotImplemented().into_bound(py)) {
        return Ok(None);
    }
    if is_strategy(&result) {
        return Ok(Some(empty_guard(py, result.unbind())?));
    }
    // A registered factory that returns a non-strategy (and non-NotImplemented) is a
    // resolution failure, surfaced lazily as ResolutionFailed (matches upstream as_strategy);
    // not an eager InvalidArgument, so `from_type(T)` returns a strategy that errors on draw.
    Ok(Some(deferred_resolution_failed(
        py,
        format!(
            "Error: {} was registered for {}, but returned non-strategy {}",
            thing.repr()?,
            entry.repr()?,
            result.repr()?
        ),
    )?))
}

/// Last-resort fallback: a type REAL hypothesis has a registered strategy for (in its
/// `_global_type_lookup`) that native resolution didn't handle — e.g. `os._Environ`.
/// Returns the REAL strategy, which is drawable via the reverse interop bridge
/// (native `ConjectureData.draw` → real `do_draw`). Only consulted AFTER native's own
/// type handling (so native int/str/container/datetime/etc. always win first), so this
/// just catches the long tail of exotic stdlib types real pre-registers.
fn resolve_via_real_lookup(py: Python<'_>, thing: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let lookup = match py
        .import("hypothesis.strategies._internal.types")
        .and_then(|m| m.getattr("_global_type_lookup"))
    {
        Ok(l) => l,
        Err(_) => return Ok(None),
    };
    if !lookup.contains(thing).unwrap_or(false) {
        return Ok(None);
    }
    let entry = lookup.get_item(thing)?;
    // entry is either a strategy (has do_draw) or a factory `(type) -> strategy`.
    if entry.hasattr("do_draw").unwrap_or(false) {
        return Ok(Some(entry.unbind()));
    }
    match entry.call1((thing,)) {
        Ok(r) if r.hasattr("do_draw").unwrap_or(false) => Ok(Some(r.unbind())),
        _ => Ok(None),
    }
}

/// from_type fallback for a plain class: resolve `thing` via registered SUBCLASSES
/// (most-specific by real-MRO membership — a virtual ABC superclass like Hashable must
/// not shadow a user type). Returns None when no registered subclass applies, so the
/// caller falls through to building `thing`'s own constructor.
fn resolve_via_subclasses(py: Python<'_>, thing: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    let reg = match py
        .import("hypothesis_fast.native_strategies")
        .and_then(|m| m.getattr("_NATIVE_TYPE_REGISTRY"))
    {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };
    let builtins = py.import("builtins")?;
    let issubclass = builtins.getattr("issubclass")?;
    let type_cls = builtins.getattr("type")?;
    let mut keys: Vec<Bound<'_, PyAny>> = Vec::new();
    for k in reg.call_method0("keys")?.try_iter()? {
        let k = k?;
        if k.is_instance(&type_cls)? {
            keys.push(k);
        }
    }
    let is_sub = |a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>| -> bool {
        issubclass.call1((a, b)).and_then(|r| r.is_truthy()).unwrap_or(false)
    };
    let mut matched: Vec<Bound<'_, PyAny>> = Vec::new();
    for k in &keys {
        if k.is(thing) || !is_sub(k, thing) {
            continue; // direct registration handled earlier; only proper subclasses here
        }
        // Most-specific: keep k only if no OTHER registered type is in its real MRO.
        let mro_reg_count = match k.getattr("__mro__") {
            Ok(mro) => {
                let mut n = 0usize;
                for base in mro.try_iter()? {
                    let base = base?;
                    if keys.iter().any(|t| t.is(&base)) {
                        n += 1;
                    }
                }
                n
            }
            Err(_) => 1,
        };
        if mro_reg_count == 1 {
            matched.push(k.clone());
        }
    }
    if matched.is_empty() {
        return Ok(None);
    }
    matched.sort_by_key(|k| {
        k.repr().ok().and_then(|r| r.extract::<String>().ok()).unwrap_or_default()
    });
    let mut strats: Vec<Py<PyAny>> = Vec::new();
    for k in &matched {
        let entry = reg.get_item(k)?;
        // Call the subclass's resolver with the ORIGINAL requested type (`thing`), not the
        // subclass `k` — matches upstream `as_strategy(v, thing)`. So a conditional resolver
        // like `if thing == B` declines for a from_type(A) request even though B is a
        // registered subclass of A.
        if let Some(s) = resolve_registry_entry(py, &entry, thing)? {
            strats.push(s);
        }
    }
    match strats.len() {
        0 => Ok(None),
        1 => Ok(Some(strats.into_iter().next().unwrap())),
        _ => {
            let eng = py.import("hypothesis_fast._engine")?;
            let pt = PyTuple::new(py, strats.iter().map(|s| s.bind(py)))?;
            Ok(Some(eng.getattr("one_of")?.call1(pt)?.unbind()))
        }
    }
}

/// A broad native union of the common scalar types — the strategy for `object` and the
/// element strategy for bare containers / `Any`-typed positions. (Top-level `from_type(Any)`
/// itself is an error; only INTERNAL uses get this union.)
fn any_scalar_union(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let eng = py.import("hypothesis_fast._engine")?;
    let parts = [
        eng.getattr("none")?.call0()?,
        eng.getattr("booleans")?.call0()?,
        eng.getattr("integers")?.call0()?,
        eng.getattr("floats")?.call0()?,
        eng.getattr("text")?.call0()?,
    ];
    let pt = PyTuple::new(py, parts.iter())?;
    Ok(eng.getattr("one_of")?.call1(pt)?.unbind())
}

/// Resolve an ABSTRACT type at DRAW time (called lazily via deferred): a registered
/// subtype wins, else a union of its concrete `__subclasses__` (skipping empty /
/// ResolutionFailed results), else ResolutionFailed. Re-reads the registry each call, so a
/// concrete subclass registered after the strategy was built is picked up.
#[pyfunction]
#[pyo3(name = "_resolve_abstract")]
pub(crate) fn resolve_abstract(py: Python<'_>, thing: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Memoise by the TYPE's pointer (stable for the run — unlike the transient `partial`
    // wrapping this call). from_type(abstract) resolves via deferred(_resolve_abstract),
    // which re-runs per draw; the subclass walk + per-subclass from_type below is the bulk
    // of generation cost for class-hierarchy-heavy from_type (e.g. libcst's CSTNode tree).
    // The cache is cleared per run by run_native's guard, so registry changes are seen.
    let key = thing.as_ptr() as usize;
    if let Some(cached) = crate::data::deferred_cache_get(py, key) {
        return Ok(cached);
    }
    let result = resolve_abstract_uncached(py, &thing)?;
    crate::data::deferred_cache_put(key, result.clone_ref(py));
    Ok(result)
}

fn resolve_abstract_uncached(py: Python<'_>, thing: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    if let Some(s) = resolve_via_subclasses(py, thing)? {
        return Ok(s);
    }
    let eng = py.import("hypothesis_fast._engine")?;
    let mut subs_strats: Vec<Py<PyAny>> = Vec::new();
    if let Ok(subs) = thing.call_method0("__subclasses__") {
        if let Ok(it) = subs.try_iter() {
            for sc in it {
                let sc = match sc {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if let Ok(s) = from_type(py, sc) {
                    let usable = match s.bind(py).downcast::<SearchStrategy>() {
                        Ok(ss) => {
                            let b = ss.borrow();
                            !matches!(b.node, StrategyNode::Invalid { .. })
                                && !node_is_empty(&b.node, py).unwrap_or(false)
                        }
                        Err(_) => false,
                    };
                    if usable {
                        subs_strats.push(s);
                    }
                }
            }
        }
    }
    match subs_strats.len() {
        0 => deferred_resolution_failed(
            py,
            format!(
                "Could not resolve {} to a strategy, because it is an abstract type without \
                 resolvable subclasses. Consider using register_type_strategy",
                thing.repr()?
            ),
        ),
        1 => Ok(subs_strats.into_iter().next().unwrap()),
        _ => {
            let pt = PyTuple::new(py, subs_strats.iter().map(|s| s.bind(py)))?;
            Ok(eng.getattr("one_of")?.call1(pt)?.unbind())
        }
    }
}

/// True if `obj` is a `typing.ForwardRef` instance (what a string TypeVar bound /
/// annotation becomes, e.g. `TypeVar('T', bound='CustomType')`).
fn is_forward_ref(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let fref = py.import("typing")?.getattr("ForwardRef")?;
    obj.is_instance(&fref)
}

/// Evaluate a ForwardRef's `__forward_arg__` against the globals of the module that
/// defined `owner` (the TypeVar), mirroring upstream `_try_import_forward_ref`. Returns
/// the resolved object, or None when the name can't be evaluated — a missing import, a
/// `TYPE_CHECKING`-only name, or a typo — so the caller can fall back to the registry
/// or ResolutionFailed. Dot-access bounds (`'utils.ExcInfo'`) resolve too, since `eval`
/// walks attribute access against the module namespace.
fn eval_forward_ref_in_module<'py>(
    py: Python<'py>,
    owner: &Bound<'py, PyAny>,
    fref: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let arg = match fref.getattr("__forward_arg__") {
        Ok(a) => a,
        Err(_) => return Ok(None),
    };
    let module_name = match owner.getattr("__module__") {
        Ok(m) if !m.is_none() => m,
        _ => return Ok(None),
    };
    let modules = py.import("sys")?.getattr("modules")?;
    let module = match modules.get_item(&module_name) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let globalns = match module.getattr("__dict__") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    match py
        .import("builtins")?
        .getattr("eval")?
        .call1((&arg, &globalns))
    {
        Ok(v) => Ok(Some(v)),
        Err(_) => Ok(None),
    }
}

/// it doesn't handle (so register_type_strategy and exotic types keep working).
#[pyfunction]
#[pyo3(name = "from_type")]
pub(crate) fn from_type(py: Python<'_>, thing: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Recursion guard: if `thing` is already being resolved on this thread, a
    // self-referential type is recursing (e.g. Tree -> Optional[Tree] -> Tree). Break it
    // with a deferred() that re-resolves lazily at draw time (the Optional's None branch
    // then terminates the recursion probabilistically), mirroring upstream.
    let id = thing.as_ptr() as usize;
    if FROM_TYPE_INPROGRESS.with(|s| s.borrow().contains(&id)) {
        let eng = py.import("hypothesis_fast._engine")?;
        let ft = eng.getattr("from_type")?;
        let thunk = py.import("functools")?.getattr("partial")?.call1((ft, &thing))?;
        return Ok(eng.getattr("deferred")?.call1((thunk,))?.unbind());
    }
    FROM_TYPE_INPROGRESS.with(|s| {
        s.borrow_mut().insert(id);
    });
    let result = from_type_impl(py, thing);
    FROM_TYPE_INPROGRESS.with(|s| {
        s.borrow_mut().remove(&id);
    });
    result
}

fn from_type_impl(py: Python<'_>, thing: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let eng = py.import("hypothesis_fast._engine")?;
    let builtins = py.import("builtins")?;
    let typing = py.import("typing")?;
    let call0 = |name: &str| -> PyResult<Py<PyAny>> {
        Ok(eng.getattr(name)?.call0()?.unbind())
    };
    // Types the native resolver can't handle yet still defer to the legacy frontend
    // (the safety net). Removing this is the FINAL proptest-deletion step — only safe
    // once native resolution covers everything (else it regresses ~24 test_lookup cases).
    let legacy_from_type = |py: Python<'_>, t: &Bound<'_, PyAny>| -> PyResult<Py<PyAny>> {
        Ok(py
            .import("hypothesis_fast.strategies")?
            .getattr("from_type")?
            .call1((t,))?
            .unbind())
    };

    // USER-registered types take precedence — consult the NATIVE registry
    // (`native_strategies._NATIVE_TYPE_REGISTRY`, populated by the native
    // register_type_strategy). An entry is either a native SearchStrategy (return it)
    // or a factory `(type) -> SearchStrategy` (call it with the type, or no-arg).
    if let Ok(reg) = py
        .import("hypothesis_fast.native_strategies")
        .and_then(|m| m.getattr("_NATIVE_TYPE_REGISTRY"))
    {
        if reg.contains(&thing).unwrap_or(false) {
            let entry = reg.get_item(&thing)?;
            if let Some(s) = resolve_registry_entry(py, &entry, &thing)? {
                return Ok(s);
            }
            // factory declined (NotImplemented) — fall through to other resolution
        }
    }

    // typing.NewType -> resolve its supertype
    if !thing.is_none() && thing.hasattr("__supertype__").unwrap_or(false) {
        if let Ok(sup) = thing.getattr("__supertype__") {
            return from_type(py, sup);
        }
    }

    // typing.ForwardRef('Name') -> evaluate against builtins+typing, then resolve
    if let Ok(fref) = typing.getattr("ForwardRef") {
        if builtins
            .getattr("isinstance")?
            .call1((&thing, &fref))
            .and_then(|r| r.is_truthy())
            .unwrap_or(false)
        {
            if let Ok(arg) = thing.getattr("__forward_arg__") {
                let ns = typing.dict().copy()?;
                if let Ok(evaluated) = builtins.getattr("eval")?.call1((&arg, &ns)) {
                    return from_type(py, evaluated);
                }
            }
            return legacy_from_type(py, &thing);
        }
    }

    // typing.Annotated[X, ...]: if the metadata contains a SearchStrategy, hypothesis
    // uses the last such strategy directly; otherwise resolve the underlying type X.
    if !thing.is_none() && thing.hasattr("__metadata__").unwrap_or(false) {
        if let Ok(meta) = thing.getattr("__metadata__") {
            if let Ok(mt) = meta.downcast::<PyTuple>() {
                for m in mt.iter().rev() {
                    if m.is_instance_of::<SearchStrategy>() {
                        return Ok(m.unbind());
                    }
                    // annotated_types "grouped metadata": an iterable of sub-metadata
                    // (`__is_annotated_types_grouped_metadata__`); if it yields a
                    // SearchStrategy, use that (upstream from_typing_type Annotated).
                    let grouped = m
                        .getattr("__is_annotated_types_grouped_metadata__")
                        .map(|v| v.is_truthy().unwrap_or(false))
                        .unwrap_or(false);
                    if grouped {
                        if let Ok(it) = m.try_iter() {
                            for sub in it {
                                let sub = sub?;
                                if sub.is_instance_of::<SearchStrategy>() {
                                    return Ok(sub.unbind());
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Ok(under) = thing.getattr("__origin__") {
            return from_type(py, under);
        }
    }

    // None / NoneType
    let none_type = builtins.getattr("type")?.call1((py.None(),))?;
    if thing.is_none() || thing.is(&none_type) {
        return call0("none");
    }

    // typing.Any / bare typing.Union don't denote a concrete type — upstream raises
    // InvalidArgument for a top-level from_type() of them (they're meaningful only as
    // annotations). Internal element resolution uses any_scalar_union() directly, so it
    // is unaffected by this.
    if typing.getattr("Any").map(|a| thing.is(&a)).unwrap_or(false) {
        return deferred_invalid(
            py,
            "Could not resolve typing.Any to a strategy, because there is no such thing as \
             a runtime instance of typing.Any. Consider using an explicit type, or \
             register_type_strategy."
                .to_string(),
        );
    }
    if typing.getattr("Union").map(|u| thing.is(&u)).unwrap_or(false) {
        return deferred_invalid(
            py,
            "Could not resolve typing.Union to a strategy; a bare Union is not a concrete \
             type. Use Union[A, B, ...] with concrete members."
                .to_string(),
        );
    }
    // object -> a broad native union of common scalar types
    if thing.is(&builtins.getattr("object")?) {
        return any_scalar_union(py);
    }

    // bare builtin containers (list/set/frozenset/dict, no type args) -> native
    // containers of a broad element (real from_type(list) generates lists of stuff).
    for (tyname, ctor, arity) in
        [("list", "lists", 1), ("set", "sets", 1), ("frozenset", "frozensets", 1), ("dict", "dictionaries", 2)]
    {
        if thing.is(&builtins.getattr(tyname)?) {
            let elem = any_scalar_union(py)?;
            if arity == 1 {
                return Ok(eng.getattr(ctor)?.call1((elem.bind(py),))?.unbind());
            } else {
                let k = any_scalar_union(py)?;
                return Ok(eng.getattr("dictionaries")?.call1((k.bind(py), elem.bind(py)))?.unbind());
            }
        }
    }

    // primitive types by identity (bool before int — bool is a subclass)
    for (tyname, ctor) in [
        ("bool", "booleans"),
        ("int", "integers"),
        ("float", "floats"),
        ("str", "text"),
        ("bytes", "binary"),
    ] {
        if thing.is(&builtins.getattr(tyname)?) {
            return call0(ctor);
        }
    }

    // common stdlib value types -> their NATIVE composed strategies (resolve here
    // rather than deferring to legacy, so they're drawable by the native engine).
    for (tymod, tyname, smod, ctor) in [
        ("builtins", "complex", "hypothesis_fast.native_strategies", "complex_numbers"),
        ("decimal", "Decimal", "hypothesis_fast.native_strategies", "decimals"),
        ("fractions", "Fraction", "hypothesis_fast._engine", "fractions"),
        ("datetime", "datetime", "hypothesis_fast._engine", "datetimes"),
        ("datetime", "date", "hypothesis_fast._engine", "dates"),
        ("datetime", "time", "hypothesis_fast._engine", "times"),
        ("datetime", "timedelta", "hypothesis_fast._engine", "timedeltas"),
        ("uuid", "UUID", "hypothesis_fast._engine", "uuids"),
    ] {
        if let Ok(ty) = py.import(tymod).and_then(|m| m.getattr(tyname)) {
            if thing.is(&ty) {
                return Ok(py.import(smod)?.getattr(ctor)?.call0()?.unbind());
            }
        }
    }
    // bytearray / memoryview -> binary().map(ctor)
    for tyname in ["bytearray", "memoryview"] {
        if thing.is(&builtins.getattr(tyname)?) {
            let b = call0("binary")?;
            return Ok(b
                .bind(py)
                .call_method1("map", (builtins.getattr(tyname)?,))?
                .unbind());
        }
    }
    // range / slice -> integers().map(ctor): any single int makes a valid range/slice
    // (range(n) is the [0,n) range; slice(n) is slice(None, n, None)).
    for tyname in ["range", "slice"] {
        if thing.is(&builtins.getattr(tyname)?) {
            let ints = call0("integers")?;
            return Ok(ints
                .bind(py)
                .call_method1("map", (builtins.getattr(tyname)?,))?
                .unbind());
        }
    }
    // ExceptionGroup / BaseExceptionGroup -> builds(cls, text(), lists(builds(Exception),
    // min_size=1)); a non-empty list of exceptions is required by the constructor.
    for tyname in ["ExceptionGroup", "BaseExceptionGroup"] {
        if let Ok(eg) = builtins.getattr(tyname) {
            if thing.is(&eg) {
                let msg = call0("text")?;
                let exc = eng
                    .getattr("builds")?
                    .call1((builtins.getattr("Exception")?,))?;
                let kw = PyDict::new(py);
                kw.set_item("min_size", 1)?;
                let excs = eng.getattr("lists")?.call((exc,), Some(&kw))?;
                return Ok(eng
                    .getattr("builds")?
                    .call1((eg, msg.bind(py), excs))?
                    .unbind());
            }
        }
    }

    // TypeVar -> a registered strategy for ALL typevars (register_type_strategy(TypeVar,
    // ...)) takes precedence; else its bound; else its constraints (one_of); else a broad
    // scalar union (so distinct unconstrained typevars can take distinct types).
    if let Ok(typevar) = typing.getattr("TypeVar") {
        if builtins
            .getattr("isinstance")?
            .call1((&thing, &typevar))?
            .is_truthy()?
        {
            if let Ok(reg) = py
                .import("hypothesis_fast.native_strategies")
                .and_then(|m| m.getattr("_NATIVE_TYPE_REGISTRY"))
            {
                if reg.contains(&typevar).unwrap_or(false) {
                    let entry = reg.get_item(&typevar)?;
                    if let Some(s) = resolve_registry_entry(py, &entry, &thing)? {
                        return Ok(s);
                    }
                    // The user replaced the default TypeVar resolver (only the user ever
                    // registers `typing.TypeVar`) and it declined (NotImplemented) for this
                    // TypeVar. There is no default fallback, so resolution fails — matching
                    // upstream test_typevars_can_be_resolved_conditionally.
                    return deferred_resolution_failed(
                        py,
                        format!(
                            "Could not resolve {thing} to a strategy; consider using \
                             register_type_strategy"
                        ),
                    );
                }
            }
            if let Ok(bound) = thing.getattr("__bound__") {
                if !bound.is_none() {
                    // A string bound (`TypeVar('T', bound='CustomType')`) is stored as a
                    // ForwardRef. Upstream's _try_import_forward_ref evaluates it against the
                    // TypeVar's defining-module globals: on success we resolve the real type;
                    // on failure we consult the registry for the ForwardRef itself (the
                    // `TYPE_CHECKING`-only case explicitly registers `ForwardRef('Name')`),
                    // and otherwise raise ResolutionFailed (missing/typo'd reference).
                    if is_forward_ref(py, &bound)? {
                        if let Some(real) = eval_forward_ref_in_module(py, &thing, &bound)? {
                            return from_type(py, real);
                        }
                        if let Ok(reg) = py
                            .import("hypothesis_fast.native_strategies")
                            .and_then(|m| m.getattr("_NATIVE_TYPE_REGISTRY"))
                        {
                            if reg.contains(&bound).unwrap_or(false) {
                                let entry = reg.get_item(&bound)?;
                                if let Some(s) = resolve_registry_entry(py, &entry, &bound)? {
                                    return Ok(s);
                                }
                            }
                        }
                        let arg = bound
                            .getattr("__forward_arg__")
                            .ok()
                            .and_then(|a| a.extract::<String>().ok())
                            .unwrap_or_default();
                        return deferred_resolution_failed(
                            py,
                            format!(
                                "Could not resolve ForwardRef({arg:?}) to a type. Consider \
                                 register_type_strategy to register a strategy for this \
                                 forward reference."
                            ),
                        );
                    }
                    return from_type(py, bound);
                }
            }
            if let Ok(cons) = thing.getattr("__constraints__") {
                if let Ok(ct) = cons.downcast::<PyTuple>() {
                    if !ct.is_empty() {
                        let mut children: Vec<Py<PyAny>> = Vec::new();
                        let mut all_native = true;
                        for c in ct.iter() {
                            let s = from_type(py, c)?;
                            if !s.bind(py).is_instance_of::<SearchStrategy>() {
                                all_native = false;
                                break;
                            }
                            children.push(s);
                        }
                        if all_native {
                            let pt = PyTuple::new(py, children.iter().map(|c| c.bind(py)))?;
                            return Ok(eng.getattr("one_of")?.call1(pt)?.unbind());
                        }
                    }
                }
            }
            // Unconstrained: pick one of the common scalar TYPES (SHARED per typevar — keyed by
            // the TypeVar's repr — so repeated uses of the same TypeVar in one example resolve to
            // the SAME type, test_same_typevars_same_type), then resolve it with from_type. Mirrors
            // upstream's exact shape: `shared(sampled_from([NoneType, bool, int, float, str,
            // bytes]), key='typevar=~T').flatmap(from_type)`.
            let builtins = py.import("builtins")?;
            let none_type = py
                .import("types")
                .and_then(|m| m.getattr("NoneType"))
                .or_else(|_| builtins.getattr("type")?.call1((py.None(),)))?;
            let types_list = pyo3::types::PyList::new(
                py,
                [
                    none_type,
                    builtins.getattr("bool")?,
                    builtins.getattr("int")?,
                    builtins.getattr("float")?,
                    builtins.getattr("str")?,
                    builtins.getattr("bytes")?,
                ],
            )?;
            let sampled = eng.getattr("sampled_from")?.call1((types_list,))?.unbind();
            let key = format!(
                "typevar={}",
                thing.repr().and_then(|r| r.extract::<String>()).unwrap_or_default()
            );
            let shared_strat = SearchStrategy::wrap(py, StrategyNode::Shared { base: sampled, key })?;
            let from_type_fn = eng.getattr("from_type")?.unbind();
            return SearchStrategy::wrap(
                py,
                StrategyNode::Flatmap { base: shared_strat, func: from_type_fn },
            );
        }
    }

    // Enum subclasses -> sampled_from(members)
    if let (Ok(issubclass), Ok(isinstance_), Ok(type_), Ok(enum_cls)) = (
        builtins.getattr("issubclass"),
        builtins.getattr("isinstance"),
        builtins.getattr("type"),
        py.import("enum").and_then(|m| m.getattr("Enum")),
    ) {
        let is_class = isinstance_
            .call1((&thing, &type_))
            .and_then(|r| r.is_truthy())
            .unwrap_or(false);
        if is_class
            && issubclass
                .call1((&thing, &enum_cls))
                .and_then(|r| r.is_truthy())
                .unwrap_or(false)
        {
            let members = builtins.getattr("list")?.call1((&thing,))?;
            return Ok(eng.getattr("sampled_from")?.call1((members,))?.unbind());
        }
    }

    // parametrized generics: list[int], dict[k,v], tuple[...], Optional/Union, ...
    let origin = typing.getattr("get_origin")?.call1((&thing,))?;
    if !origin.is_none() {
        // A USER-registered generic subtype of this request's origin (e.g. a registered
        // Sequence resolving Sequence[int]/Container[int]) takes precedence over built-in
        // container handling — upstream's from_typing_type subtype resolution. Returns None
        // (fast) unless some user registration applies, so element-aware handling is intact.
        if let Ok(ns) = py.import("hypothesis_fast.native_strategies") {
            if let Ok(f) = ns.getattr("_resolve_generic_subtypes") {
                let r = f.call1((&thing,))?;
                if !r.is_none() {
                    // A registered generic that resolves EMPTY (e.g. returns' law machinery
                    // registers an unconstructable container whose factory yields `one_of()`)
                    // is a resolution failure, not a usable strategy — upstream's `as_strategy`
                    // raises ResolutionFailed. Guard here as the other registry paths do.
                    if let Ok(ss) = r.downcast::<SearchStrategy>() {
                        if node_is_empty(&ss.borrow().node, py)? {
                            return deferred_resolution_failed(
                                py,
                                format!("Error: {} resolved to an empty strategy", thing.repr()?),
                            );
                        }
                    }
                    return Ok(r.unbind());
                }
            }
        }
        let args = typing.getattr("get_args")?.call1((&thing,))?;
        let args_tuple = args.downcast::<PyTuple>()?;
        // Bare generic alias with no args (typing.List, typing.Dict, typing.Set, ...):
        // resolve element/key/value types as Any -> a broad native strategy, so the
        // container is still drawable by the native engine (else it defers to legacy).
        if args_tuple.is_empty() {
            // `Tuple[()]` (the empty-tuple TYPE) has `__args__ == ()`, whereas bare `Tuple`
            // has no `__args__` and is variadic. get_args() can't tell them apart (both ()),
            // so distinguish on hasattr — only the explicit empty form resolves to just(()).
            if origin.is(&builtins.getattr("tuple")?) && thing.hasattr("__args__").unwrap_or(false)
            {
                return Ok(eng.getattr("just")?.call1((PyTuple::empty(py),))?.unbind());
            }
            let abc = py.import("collections.abc").ok();
            let any_el = || any_scalar_union(py);
            let origin_in = |bi: &[&str], ab: &[&str]| -> bool {
                bi.iter()
                    .any(|n| builtins.getattr(n).map(|o| origin.is(&o)).unwrap_or(false))
                    || ab.iter().any(|n| {
                        abc.as_ref()
                            .and_then(|a| a.getattr(n).ok())
                            .map(|o| origin.is(&o))
                            .unwrap_or(false)
                    })
            };
            if origin_in(&["list"], &["Sequence", "MutableSequence", "Iterable", "Collection"]) {
                return Ok(eng.getattr("lists")?.call1((any_el()?.bind(py),))?.unbind());
            }
            if origin_in(&["set"], &["Set", "MutableSet", "AbstractSet"]) {
                return Ok(eng.getattr("sets")?.call1((any_el()?.bind(py),))?.unbind());
            }
            if origin_in(&["frozenset"], &[]) {
                return Ok(eng.getattr("frozensets")?.call1((any_el()?.bind(py),))?.unbind());
            }
            if origin_in(&["dict"], &["Mapping", "MutableMapping"]) {
                return Ok(eng
                    .getattr("dictionaries")?
                    .call1((any_el()?.bind(py), any_el()?.bind(py)))?
                    .unbind());
            }
            if origin_in(&["tuple"], &[]) {
                let lst = eng.getattr("lists")?.call1((any_el()?.bind(py),))?;
                return Ok(lst.call_method1("map", (builtins.getattr("tuple")?,))?.unbind());
            }
            if origin_in(&[], &["Iterator", "Iterable"]) {
                let lst = eng.getattr("lists")?.call1((any_el()?.bind(py),))?;
                return Ok(lst.call_method1("map", (builtins.getattr("iter")?,))?.unbind());
            }
        }
        // Literal[...] -> sampled_from(the literal VALUES) (args are values, not types).
        if typing.getattr("Literal").map(|l| origin.is(&l)).unwrap_or(false) {
            return Ok(eng.getattr("sampled_from")?.call1((args_tuple,))?.unbind());
        }
        // Final[X] / ClassVar[X] -> from_type(X)
        let is_wrapper = ["Final", "ClassVar"]
            .iter()
            .any(|n| typing.getattr(n).map(|w| origin.is(&w)).unwrap_or(false));
        if is_wrapper && !args_tuple.is_empty() {
            return from_type(py, args_tuple.get_item(0)?);
        }
        // type[X] / typing.Type[X] -> the type object(s). X may be a bare string or
        // ForwardRef (e.g. `type["ArithmeticError"]`), which must be resolved to the
        // class before `just`. type[Union[A, B]] -> sampled_from([A, B]).
        if origin.is(&builtins.getattr("type")?) && !args_tuple.is_empty() {
            let inner = args_tuple.get_item(0)?;
            let inner_origin = typing.getattr("get_origin")?.call1((&inner,))?;
            let inner_is_union = !inner_origin.is_none()
                && (inner_origin.is(&typing.getattr("Union")?)
                    || py
                        .import("types")
                        .and_then(|m| m.getattr("UnionType"))
                        .map(|ut| inner_origin.is(&ut))
                        .unwrap_or(false));
            if inner_is_union {
                let uargs = typing.getattr("get_args")?.call1((&inner,))?;
                let ut = uargs.downcast::<PyTuple>()?;
                let mut members: Vec<Py<PyAny>> = Vec::new();
                for a in ut.iter() {
                    match resolve_type_arg(py, &builtins, &typing, &a)? {
                        Some(t) => members.push(t),
                        None => return legacy_from_type(py, &thing),
                    }
                }
                let lst = PyList::new(py, members.iter().map(|m| m.bind(py)))?;
                return Ok(eng.getattr("sampled_from")?.call1((lst,))?.unbind());
            }
            return match resolve_type_arg(py, &builtins, &typing, &inner)? {
                Some(t) => Ok(eng.getattr("just")?.call1((t.bind(py),))?.unbind()),
                None => legacy_from_type(py, &thing),
            };
        }
        // Callable[[..], R] -> functions(returns=from_type(R)). The first arg is the
        // parameter-type list (or Ellipsis), which is NOT a type, so resolve before
        // the generic child loop (which would choke on the list).
        let is_callable = py
            .import("collections.abc")
            .and_then(|m| m.getattr("Callable"))
            .map(|c| origin.is(&c))
            .unwrap_or(false);
        if is_callable {
            let nat = py.import("hypothesis_fast.native_strategies")?;
            let kw = PyDict::new(py);
            // `like` shapes the generated function's call signature. Callable[[a,b],R]
            // (an explicit arg-type LIST) -> a function of exactly that many positional
            // params (so `f(1)` on Callable[[],R] raises TypeError); Callable[...,R]
            // (Ellipsis) -> `_any_callable` (accepts any args); BARE Callable -> no-arg
            // default like.
            if !args_tuple.is_empty() {
                let first = args_tuple.get_item(0)?;
                if let Ok(arglist) = first.downcast::<PyList>() {
                    let like = nat.getattr("_arity_callable")?.call1((arglist.len(),))?;
                    kw.set_item("like", like)?;
                } else {
                    kw.set_item("like", nat.getattr("_any_callable")?)?;
                }
            }
            if let Some(ret) = (args_tuple.len() >= 2).then(|| args_tuple.get_item(args_tuple.len() - 1)) {
                let rs = from_type(py, ret?)?;
                if rs.bind(py).is_instance_of::<SearchStrategy>() {
                    kw.set_item("returns", rs)?;
                }
            }
            return Ok(nat.getattr("functions")?.call((), Some(&kw))?.unbind());
        }
        // tuple[X, ...] (variadic) and tuple[()] (empty) — handle before resolving
        // children, since the Ellipsis / empty-tuple args aren't types.
        if origin.is(&builtins.getattr("tuple")?) {
            let ellipsis = builtins.getattr("Ellipsis")?;
            let n = args_tuple.len();
            let is_empty_tuple = n == 0
                || (n == 1
                    && args_tuple
                        .get_item(0)
                        .ok()
                        .and_then(|a| a.downcast::<PyTuple>().ok().map(|t| t.is_empty()))
                        .unwrap_or(false));
            if is_empty_tuple {
                return Ok(eng.getattr("just")?.call1((PyTuple::empty(py),))?.unbind());
            }
            let variadic =
                (0..n).any(|i| args_tuple.get_item(i).map(|a| a.is(&ellipsis)).unwrap_or(false));
            if variadic {
                let arg0 = args_tuple.get_item(0)?;
                let elem = if typing.getattr("Any").map(|x| arg0.is(&x)).unwrap_or(false) {
                    any_scalar_union(py)?
                } else {
                    from_type(py, arg0)?
                };
                if elem.bind(py).is_instance_of::<SearchStrategy>() {
                    let lst = eng.getattr("lists")?.call1((elem.bind(py),))?;
                    return Ok(lst.call_method1("map", (builtins.getattr("tuple")?,))?.unbind());
                }
                return legacy_from_type(py, &thing);
            }
        }
        // Resolve all child types; a native container/one_of can only hold NATIVE
        // element strategies, so if ANY child resolves to a foreign (legacy/real)
        // strategy — e.g. a registered custom type — defer the whole thing to legacy.
        let any_ref = typing.getattr("Any").ok();
        let mut children: Vec<Py<PyAny>> = Vec::new();
        for a in args_tuple.iter() {
            // An `Any` type-arg (e.g. List[Any], Dict[str, Any]) means "any element" —
            // resolve to the broad union, NOT the top-level from_type(Any) error.
            if any_ref.as_ref().map(|x| a.is(x)).unwrap_or(false) {
                children.push(any_scalar_union(py)?);
            } else {
                children.push(from_type(py, a)?);
            }
        }
        let all_native = children
            .iter()
            .all(|c| c.bind(py).is_instance_of::<SearchStrategy>());
        if !all_native {
            return legacy_from_type(py, &thing);
        }
        let pt = || PyTuple::new(py, children.iter().map(|c| c.bind(py)));

        // Union / Optional (typing.Union and PEP 604 X | Y)
        let is_union = origin.is(&typing.getattr("Union")?)
            || py
                .import("types")
                .and_then(|m| m.getattr("UnionType"))
                .map(|ut| origin.is(&ut))
                .unwrap_or(false);
        if is_union {
            // Put NoneType branches first so Optional[X] minimises to None, matching
            // hypothesis (one_of prefers earlier branches when shrinking).
            let mut ordered: Vec<&Py<PyAny>> = Vec::new();
            for (i, a) in args_tuple.iter().enumerate() {
                if a.is(&none_type) {
                    ordered.push(&children[i]);
                }
            }
            for (i, a) in args_tuple.iter().enumerate() {
                if !a.is(&none_type) {
                    ordered.push(&children[i]);
                }
            }
            let opt = PyTuple::new(py, ordered.iter().map(|c| c.bind(py)))?;
            return Ok(eng.getattr("one_of")?.call1(opt)?.unbind());
        }
        if origin.is(&builtins.getattr("list")?) && children.len() == 1 {
            return Ok(eng.getattr("lists")?.call1((children[0].bind(py),))?.unbind());
        }
        for (tyname, ctor) in [("set", "sets"), ("frozenset", "frozensets")] {
            if origin.is(&builtins.getattr(tyname)?) && children.len() == 1 {
                let el = filtered_hashable(&eng, py, &args_tuple.get_item(0)?, &children[0].bind(py))?;
                return Ok(eng.getattr(ctor)?.call1((el,))?.unbind());
            }
        }
        if origin.is(&builtins.getattr("dict")?) && children.len() == 2 {
            let k = filtered_hashable(&eng, py, &args_tuple.get_item(0)?, &children[0].bind(py))?;
            return Ok(eng
                .getattr("dictionaries")?
                .call1((k, children[1].bind(py)))?
                .unbind());
        }
        if origin.is(&builtins.getattr("tuple")?) {
            let ellipsis = builtins.getattr("Ellipsis")?;
            let n = args_tuple.len();
            let variadic = n == 0 || (0..n).any(|i| {
                args_tuple.get_item(i).map(|a| a.is(&ellipsis)).unwrap_or(false)
            });
            if !variadic {
                return Ok(eng.getattr("tuples")?.call1(pt()?)?.unbind());
            }
        }
        // collections.abc container generics -> native containers
        if let Ok(abc) = py.import("collections.abc") {
            // Sequence-likes: bytes is also a Sequence of ints, so include binary()
            // when int is an admissible element type (matches hypothesis).
            // A list satisfies all of these (Sequence/Iterable/Collection/Container/
            // Reversible). bytes is also a Sequence of ints -> include binary() when int
            // is an admissible element type (matches hypothesis).
            for name in [
                "Sequence",
                "MutableSequence",
                "Iterable",
                "Collection",
                "Container",
                "Reversible",
            ] {
                if let Ok(o) = abc.getattr(name) {
                    if origin.is(&o) && children.len() == 1 {
                        let lst = eng.getattr("lists")?.call1((children[0].bind(py),))?;
                        if int_admissible(py, &args_tuple.get_item(0)?)? {
                            let bin = eng.getattr("binary")?.call0()?;
                            return Ok(eng.getattr("one_of")?.call1((lst, bin))?.unbind());
                        }
                        return Ok(lst.unbind());
                    }
                }
            }
            // Generator[Y, S, R] -> a generator that yields Y-values and returns an R
            // (via StopIteration.value). Draw a list of yields and a return value, then
            // assemble the generator object (native_strategies._make_generator).
            if let Ok(o) = abc.getattr("Generator") {
                if origin.is(&o) && children.len() == 3 {
                    let nat = py.import("hypothesis_fast.native_strategies")?;
                    let lst = eng.getattr("lists")?.call1((children[0].bind(py),))?;
                    let tup = eng.getattr("tuples")?.call1((lst, children[2].bind(py)))?;
                    return Ok(tup
                        .call_method1("map", (nat.getattr("_make_generator")?,))?
                        .unbind());
                }
            }
            // Iterator[X] must itself be an iterator, not a list.
            if let Ok(o) = abc.getattr("Iterator") {
                if origin.is(&o) && children.len() == 1 {
                    let lst = eng.getattr("lists")?.call1((children[0].bind(py),))?;
                    return Ok(lst.call_method1("map", (builtins.getattr("iter")?,))?.unbind());
                }
            }
            for name in ["Set", "MutableSet"] {
                if let Ok(o) = abc.getattr(name) {
                    if origin.is(&o) && children.len() == 1 {
                        let el = filtered_hashable(&eng, py, &args_tuple.get_item(0)?, &children[0].bind(py))?;
                        return Ok(eng.getattr("sets")?.call1((el,))?.unbind());
                    }
                }
            }
            for name in ["Mapping", "MutableMapping"] {
                if let Ok(o) = abc.getattr(name) {
                    if origin.is(&o) && children.len() == 2 {
                        let k = filtered_hashable(&eng, py, &args_tuple.get_item(0)?, &children[0].bind(py))?;
                        return Ok(eng
                            .getattr("dictionaries")?
                            .call1((k, children[1].bind(py)))?
                            .unbind());
                    }
                }
            }
            // dict views: build a dict with the right key/value types, then take the view.
            let mc = |m: &str| -> PyResult<Py<PyAny>> {
                Ok(py.import("operator")?.getattr("methodcaller")?.call1((m,))?.unbind())
            };
            if let Ok(o) = abc.getattr("KeysView") {
                if origin.is(&o) && children.len() == 1 {
                    let d = eng
                        .getattr("dictionaries")?
                        .call1((children[0].bind(py), eng.getattr("none")?.call0()?))?;
                    return Ok(d.call_method1("map", (mc("keys")?,))?.unbind());
                }
            }
            if let Ok(o) = abc.getattr("ValuesView") {
                if origin.is(&o) && children.len() == 1 {
                    let d = eng
                        .getattr("dictionaries")?
                        .call1((eng.getattr("integers")?.call0()?, children[0].bind(py)))?;
                    return Ok(d.call_method1("map", (mc("values")?,))?.unbind());
                }
            }
            if let Ok(o) = abc.getattr("ItemsView") {
                if origin.is(&o) && children.len() == 2 {
                    let d = eng
                        .getattr("dictionaries")?
                        .call1((children[0].bind(py), children[1].bind(py)))?;
                    return Ok(d.call_method1("map", (mc("items")?,))?.unbind());
                }
            }
        }
        // collections concrete generics -> native container .map(constructor)
        if let Ok(collections) = py.import("collections") {
            // deque[X] -> lists(from_type(X)).map(deque)
            if let Ok(deque) = collections.getattr("deque") {
                if origin.is(&deque) && children.len() == 1 {
                    let lst = eng.getattr("lists")?.call1((children[0].bind(py),))?;
                    return Ok(lst.call_method1("map", (&deque,))?.unbind());
                }
            }
            // OrderedDict[K, V] -> dictionaries(K, V).map(OrderedDict)
            if let Ok(od) = collections.getattr("OrderedDict") {
                if origin.is(&od) && children.len() == 2 {
                    let d = eng
                        .getattr("dictionaries")?
                        .call1((children[0].bind(py), children[1].bind(py)))?;
                    return Ok(d.call_method1("map", (&od,))?.unbind());
                }
            }
            // Counter[K] -> dictionaries(from_type(K), integers(min_value=0)).map(Counter)
            if let Ok(counter) = collections.getattr("Counter") {
                if origin.is(&counter) && children.len() == 1 {
                    let kw = PyDict::new(py);
                    kw.set_item("min_value", 0)?;
                    let counts = eng.getattr("integers")?.call((), Some(&kw))?;
                    let d = eng
                        .getattr("dictionaries")?
                        .call1((children[0].bind(py), counts))?;
                    return Ok(d.call_method1("map", (&counter,))?.unbind());
                }
            }
            // defaultdict[K, V] -> dictionaries(K, V).map(lambda d: defaultdict(None, d))
            if let Ok(dd) = collections.getattr("defaultdict") {
                if origin.is(&dd) && children.len() == 2 {
                    let d = eng
                        .getattr("dictionaries")?
                        .call1((children[0].bind(py), children[1].bind(py)))?;
                    let conv = py
                        .import("functools")?
                        .getattr("partial")?
                        .call1((&dd, py.None()))?;
                    return Ok(d.call_method1("map", (conv,))?.unbind());
                }
            }
            // ChainMap[K, V] -> dictionaries(K, V).map(ChainMap)
            if let Ok(cm) = collections.getattr("ChainMap") {
                if origin.is(&cm) && children.len() == 2 {
                    let d = eng
                        .getattr("dictionaries")?
                        .call1((children[0].bind(py), children[1].bind(py)))?;
                    return Ok(d.call_method1("map", (&cm,))?.unbind());
                }
            }
        }
        // A bare-generic user registration (register_type_strategy(MyGeneric, strat))
        // also resolves its parametrizations: MyGeneric[int] -> the strategy/factory
        // registered for the origin. Checked AFTER built-in container handling so it
        // only catches user generics (builtins/abc resolve element-aware above).
        if let Ok(reg) = py
            .import("hypothesis_fast.native_strategies")
            .and_then(|m| m.getattr("_NATIVE_TYPE_REGISTRY"))
        {
            if reg.contains(&origin).unwrap_or(false) {
                let entry = reg.get_item(&origin)?;
                // resolver gets the full parametrized type (e.g. MyGeneric[int]).
                if let Some(s) = resolve_registry_entry(py, &entry, &thing)? {
                    return Ok(s);
                }
                // declined (NotImplemented) — fall through to legacy
            }
        }
        // An UNregistered user-defined generic parametrized with a FREE TypeVar (e.g.
        // `MyGeneric[T]`, T unbound) is unresolvable — upstream raises ResolutionFailed.
        // (A registered generic is handled above; concrete args like `MyGeneric[int]`
        // have no free TypeVar so they fall through to building the class.)
        if origin.is_instance(&builtins.getattr("type")?).unwrap_or(false) {
            if let Ok(tv) = typing.getattr("TypeVar") {
                let has_free_typevar = args_tuple
                    .iter()
                    .any(|a| a.is_instance(&tv).unwrap_or(false));
                if has_free_typevar {
                    return deferred_resolution_failed(
                        py,
                        format!(
                            "Could not resolve {} to a strategy; consider using \
                             register_type_strategy",
                            thing.repr()?
                        ),
                    );
                }
            }
        }
        return legacy_from_type(py, &thing);
    }

    // Abstract classes (incl. abstract dataclasses — e.g. libcst's CSTNode bases, which
    // ARE dataclasses) must be resolved to their concrete subtypes, NOT built directly.
    // Check this BEFORE the TypedDict / dataclass / generic-class branches below, any of
    // which would otherwise `builds(abstract_cls)` and fail at draw with "Can't instantiate
    // abstract class". (User-registered strategies already took precedence above.)
    if builtins
        .getattr("isinstance")?
        .call1((&thing, builtins.getattr("type")?))?
        .is_truthy()?
    {
        let inspect = py.import("inspect")?;
        let is_abstract = inspect
            .getattr("isabstract")?
            .call1((&thing,))
            .and_then(|r| r.is_truthy())
            .unwrap_or(false);
        let is_protocol = thing
            .getattr("_is_protocol")
            .and_then(|v| v.is_truthy())
            .unwrap_or(false);
        if is_abstract && !is_protocol {
            // Lazy (deferred): a concrete subclass may be registered after this strategy is
            // built but before it's drawn. `_resolve_abstract` re-checks registry +
            // __subclasses__ at draw time and one_of's the concrete survivors.
            let f = eng.getattr("_resolve_abstract")?;
            let thunk = py.import("functools")?.getattr("partial")?.call1((f, &thing))?;
            return Ok(eng.getattr("deferred")?.call1((thunk,))?.unbind());
        }
    }

    // TypedDict -> fixed_dictionaries({key: from_type(value_type)}) (all-native guard)
    let is_typeddict = typing
        .getattr("is_typeddict")
        .and_then(|f| f.call1((&thing,)))
        .and_then(|r| r.is_truthy())
        .unwrap_or(false)
        || thing.hasattr("__required_keys__").unwrap_or(false);
    if is_typeddict {
        if let Ok(hints) = typing.getattr("get_type_hints")?.call1((&thing,)) {
            if let Ok(hd) = hints.downcast::<PyDict>() {
                // Keys in __optional_keys__ (total=False / NotRequired) go into the
                // `optional=` group so fixed_dictionaries omits them sometimes — matching
                // upstream (test_{simple,layered}_optional_key_is_optional).
                let optional_keys = thing.getattr("__optional_keys__").ok();
                let mapping = PyDict::new(py);
                let optional = PyDict::new(py);
                let mut all_native = true;
                for (k, v) in hd.iter() {
                    let strat = from_type(py, v)?;
                    if !strat.bind(py).is_instance_of::<SearchStrategy>() {
                        all_native = false;
                        break;
                    }
                    let is_opt = optional_keys
                        .as_ref()
                        .is_some_and(|ok| ok.contains(&k).unwrap_or(false));
                    if is_opt {
                        optional.set_item(k, strat)?;
                    } else {
                        mapping.set_item(k, strat)?;
                    }
                }
                if all_native {
                    if optional.is_empty() {
                        return Ok(eng.getattr("fixed_dictionaries")?.call1((mapping,))?.unbind());
                    }
                    let kw = PyDict::new(py);
                    kw.set_item("optional", optional)?;
                    return Ok(eng
                        .getattr("fixed_dictionaries")?
                        .call((mapping,), Some(&kw))?
                        .unbind());
                }
            }
        }
        return legacy_from_type(py, &thing);
    }

    // dataclasses / NamedTuple -> builds(cls, **{field: from_type(field_type)}) when
    // every field type resolves NATIVE (else defer the whole class to legacy, since a
    // native builds can't pass a foreign element to the constructor).
    if let Some(field_names) = dataclass_or_namedtuple_fields(py, &thing)? {
        if let Ok(hints) = typing.getattr("get_type_hints")?.call1((&thing,)) {
            if let Ok(hd) = hints.downcast::<PyDict>() {
                let kwargs = PyDict::new(py);
                let mut all_native = true;
                for name in &field_names {
                    if let Some(hint) = hd.get_item(name)? {
                        let strat = from_type(py, hint)?;
                        if !strat.bind(py).is_instance_of::<SearchStrategy>() {
                            all_native = false;
                            break;
                        }
                        kwargs.set_item(name, strat)?;
                    }
                }
                if all_native {
                    return Ok(eng
                        .getattr("builds")?
                        .call((&thing,), Some(&kwargs))?
                        .unbind());
                }
            }
        }
        return legacy_from_type(py, &thing);
    }

    // generic user class -> builds(cls, **{param: from_type(__init__ annotation)}).
    // Skip abstract/Protocol classes and any whose required params can't resolve native.
    let inspect = py.import("inspect")?;
    let is_class = builtins
        .getattr("isinstance")?
        .call1((&thing, builtins.getattr("type")?))?
        .is_truthy()?;
    if is_class {
        let is_abstract = inspect
            .getattr("isabstract")?
            .call1((&thing,))
            .and_then(|r| r.is_truthy())
            .unwrap_or(false);
        let is_protocol = thing
            .getattr("_is_protocol")
            .and_then(|v| v.is_truthy())
            .unwrap_or(false);
        if is_abstract && !is_protocol {
            // Abstract types resolve LAZILY: a concrete subclass may be registered AFTER
            // this strategy is built but BEFORE it's drawn (upstream from_type is lazy).
            // Defer to `_resolve_abstract`, which re-checks the registry + __subclasses__
            // at draw time. (Returns a concrete strategy, so no infinite deferral.)
            let f = eng.getattr("_resolve_abstract")?;
            let thunk = py.import("functools")?.getattr("partial")?.call1((f, &thing))?;
            return Ok(eng.getattr("deferred")?.call1((thunk,))?.unbind());
        }
        // A registered subclass resolves `thing` (incl. a concrete type registered for
        // its abstract base) before we try to build `thing` itself.
        if let Some(s) = resolve_via_subclasses(py, &thing)? {
            return Ok(s);
        }
        if !is_protocol {
            if let Some(s) = generic_class_strategy(py, &thing, &eng, &inspect, &typing)? {
                return Ok(s);
            }
        }
        return legacy_from_type(py, &thing);
    }

    // ForwardRef / TypeVar / type-alias / ... — leave to the legacy resolver
    // (then they can't be drawn natively: the interop tail).
    legacy_from_type(py, &thing)
}

/// Whether `int` is an admissible element type for `x` — i.e. a `bytes` value (a
/// sequence of ints) is a valid `Sequence[x]`. True when `issubclass(int, x)` (so
/// int/object/numbers.Real qualify) or, for a Union, any member qualifies.
fn int_admissible(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<bool> {
    let typing = py.import("typing")?;
    let builtins = py.import("builtins")?;
    let origin = typing.getattr("get_origin")?.call1((x,))?;
    if !origin.is_none() {
        let is_union = origin.is(&typing.getattr("Union")?)
            || py
                .import("types")
                .and_then(|m| m.getattr("UnionType"))
                .map(|u| origin.is(&u))
                .unwrap_or(false);
        if is_union {
            let args = typing.getattr("get_args")?.call1((x,))?;
            for a in args.downcast::<PyTuple>()?.iter() {
                if int_admissible(py, &a)? {
                    return Ok(true);
                }
            }
        }
        return Ok(false);
    }
    if builtins
        .getattr("isinstance")?
        .call1((x, builtins.getattr("type")?))?
        .is_truthy()?
    {
        if let Ok(r) = builtins.getattr("issubclass")?.call1((builtins.getattr("int")?, x)) {
            return r.is_truthy();
        }
    }
    Ok(false)
}

/// Resolve a `type[...]` argument to a concrete type object: a bare string or
/// `ForwardRef` is `eval`'d (builtins are auto-injected into the namespace, so
/// builtin/exception names resolve); an actual class is returned as-is. Returns None
/// if it cannot be resolved (caller defers to legacy).
fn resolve_type_arg(
    py: Python<'_>,
    builtins: &Bound<'_, PyModule>,
    typing: &Bound<'_, PyModule>,
    x: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    let type_ = builtins.getattr("type")?;
    if builtins.getattr("isinstance")?.call1((x, &type_))?.is_truthy()? {
        return Ok(Some(x.clone().unbind()));
    }
    if let Ok(s) = x.downcast::<PyString>() {
        let ns = PyDict::new(py);
        return Ok(builtins.getattr("eval")?.call1((s, &ns)).ok().map(|t| t.unbind()));
    }
    if let Ok(fref) = typing.getattr("ForwardRef") {
        if builtins.getattr("isinstance")?.call1((x, &fref))?.is_truthy()? {
            if let Ok(arg) = x.getattr("__forward_arg__") {
                let ns = PyDict::new(py);
                return Ok(builtins.getattr("eval")?.call1((&arg, &ns)).ok().map(|t| t.unbind()));
            }
        }
    }
    Ok(None)
}

/// builds(cls, **kwargs) inferring kwargs from cls.__init__ annotations. Returns None
/// (caller falls back to legacy) if any REQUIRED param can't resolve to a native
/// strategy (unannotated required param, or an annotation that resolves non-native).
fn generic_class_strategy(
    py: Python<'_>,
    thing: &Bound<'_, PyAny>,
    eng: &Bound<'_, PyModule>,
    inspect: &Bound<'_, PyModule>,
    typing: &Bound<'_, PyModule>,
) -> PyResult<Option<Py<PyAny>>> {
    let sig = match inspect.getattr("signature")?.call1((thing,)) {
        Ok(s) => s,
        // Unintrospectable builtin/exception (inspect.signature raises ValueError):
        // most construct with no args (e.g. every BaseException subclass). Build it
        // natively — `cls()` raising at draw is no worse than deferring to a legacy
        // strategy the native engine cannot draw at all.
        Err(_) => return Ok(Some(eng.getattr("builds")?.call1((thing,))?.unbind())),
    };
    let hints = thing
        .getattr("__init__")
        .ok()
        .and_then(|init| typing.getattr("get_type_hints").ok().and_then(|f| f.call1((init,)).ok()));
    let param_cls = inspect.getattr("Parameter")?;
    let empty = param_cls.getattr("empty")?;
    let var_pos = param_cls.getattr("VAR_POSITIONAL")?;
    let var_kw = param_cls.getattr("VAR_KEYWORD")?;
    let pos_only = param_cls.getattr("POSITIONAL_ONLY")?;
    let kwargs = PyDict::new(py);
    // Positional-only ctor params (`def __init__(self, value, /)`) must be passed
    // positionally to builds, not by keyword (`builds(cls, value=...)` raises). Collected
    // in signature order (posonly always precede the rest), passed before kwargs.
    let mut posargs: Vec<Py<PyAny>> = Vec::new();
    for item in sig.getattr("parameters")?.call_method0("values")?.try_iter()? {
        let p = item?;
        let kind = p.getattr("kind")?;
        if kind.is(&var_pos) || kind.is(&var_kw) {
            continue;
        }
        let name: String = p.getattr("name")?.extract()?;
        let ann = hints.as_ref().and_then(|h| h.get_item(&name).ok());
        match ann {
            Some(a) => {
                let strat = from_type(py, a)?;
                if !strat.bind(py).is_instance_of::<SearchStrategy>() {
                    return Ok(None);
                }
                if kind.is(&pos_only) {
                    posargs.push(strat);
                } else {
                    kwargs.set_item(&name, strat)?;
                }
            }
            None => {
                // required (no default) and unannotated -> native can't build it. If real
                // hypothesis pre-registered a strategy for this exact type (e.g. os._Environ),
                // use it (drawn via the reverse interop bridge) before giving up. Otherwise
                // surface ResolutionFailed (deferred), matching upstream.
                if p.getattr("default")?.is(&empty) {
                    if let Some(s) = resolve_via_real_lookup(py, thing)? {
                        return Ok(Some(s));
                    }
                    return Ok(Some(deferred_resolution_failed(
                        py,
                        format!(
                            "Could not resolve {} to a strategy; consider using register_type_strategy",
                            thing.repr()?
                        ),
                    )?));
                }
            }
        }
    }
    let mut call_args: Vec<Bound<'_, PyAny>> = vec![thing.clone()];
    for pa in &posargs {
        call_args.push(pa.bind(py).clone());
    }
    let call_tuple = PyTuple::new(py, call_args)?;
    Ok(Some(eng.getattr("builds")?.call(call_tuple, Some(&kwargs))?.unbind()))
}

/// If `thing` is a dataclass or a typing.NamedTuple, return its init field names (in
/// order); else None. Used to resolve them via native `builds`.
fn dataclass_or_namedtuple_fields(
    py: Python<'_>,
    thing: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<String>>> {
    let builtins = py.import("builtins")?;
    let is_class = builtins
        .getattr("isinstance")?
        .call1((thing, builtins.getattr("type")?))?
        .is_truthy()?;
    if !is_class {
        return Ok(None);
    }
    // dataclass: dataclasses.fields(thing) where f.init
    if let Ok(dc) = py.import("dataclasses") {
        if dc.getattr("is_dataclass")?.call1((thing,))?.is_truthy()? {
            let mut names = Vec::new();
            for f in dc.getattr("fields")?.call1((thing,))?.try_iter()? {
                let f = f?;
                if f.getattr("init")?.is_truthy()? {
                    names.push(f.getattr("name")?.extract::<String>()?);
                }
            }
            return Ok(Some(names));
        }
    }
    // NamedTuple: a tuple subclass carrying `_fields`
    let is_tuple_sub = builtins
        .getattr("issubclass")?
        .call1((thing, builtins.getattr("tuple")?))
        .and_then(|r| r.is_truthy())
        .unwrap_or(false);
    if is_tuple_sub && thing.hasattr("_fields")? {
        let mut names = Vec::new();
        for n in thing.getattr("_fields")?.try_iter()? {
            names.push(n?.extract::<String>()?);
        }
        return Ok(Some(names));
    }
    Ok(None)
}
