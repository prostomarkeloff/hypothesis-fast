//! Draw cluster: Many geometry, draw_child/foreign bridges, draw_node, idraw.
//! Split out of strategy/mod.rs.
#![allow(clippy::wildcard_imports)]
use super::*;
use pyo3::intern;

struct Many {
    min_size: usize,
    max_size: usize,
    p_continue: f64,
    count: usize,
    rejections: usize,
    force_stop: bool,
}

impl Many {
    fn new(min_size: usize, max_size: usize) -> Self {
        let avg = ((min_size as f64 * 2.0).max(min_size as f64 + 5.0))
            .min(0.5 * (min_size as f64 + max_size as f64));
        let p_continue = if min_size == max_size {
            0.0
        } else {
            calc_p_continue(avg - min_size as f64, (max_size - min_size) as f64)
        };
        Many {
            min_size,
            max_size,
            p_continue,
            count: 0,
            rejections: 0,
            force_stop: false,
        }
    }

    fn more(&mut self, py: Python<'_>, data: &Bound<'_, ConjectureData>) -> PyResult<bool> {
        if self.min_size == self.max_size {
            let cont = self.count < self.min_size;
            if cont {
                self.count += 1;
            }
            return Ok(cont);
        }
        let forced: Option<bool> = if self.force_stop {
            Some(false)
        } else if self.count < self.min_size {
            Some(true)
        } else if self.count >= self.max_size {
            Some(false)
        } else {
            None
        };
        let cont = data.borrow_mut().draw_boolean_rs(py, self.p_continue, forced)?;
        if cont {
            self.count += 1;
        }
        Ok(cont)
    }

    fn reject(&mut self, py: Python<'_>, data: &Bound<'_, ConjectureData>) -> PyResult<()> {
        self.count -= 1;
        self.rejections += 1;
        if self.rejections > std::cmp::max(3, 2 * self.count) {
            if self.count < self.min_size {
                return Err(data.borrow_mut().mark_invalid_err(py));
            } else {
                self.force_stop = true;
            }
        }
        Ok(())
    }

    /// As `more`, but against a FOREIGN (real-hypothesis) ConjectureData via its draw_boolean
    /// method — used when a native collection strategy is drawn against a real cd (interop).
    fn more_foreign(&mut self, py: Python<'_>, cd: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self.min_size == self.max_size {
            let cont = self.count < self.min_size;
            if cont {
                self.count += 1;
            }
            return Ok(cont);
        }
        let forced: Option<bool> = if self.force_stop {
            Some(false)
        } else if self.count < self.min_size {
            Some(true)
        } else if self.count >= self.max_size {
            Some(false)
        } else {
            None
        };
        let kw = PyDict::new(py);
        kw.set_item("p", self.p_continue)?;
        if let Some(f) = forced {
            kw.set_item("forced", f)?;
        }
        let cont: bool = cd.call_method("draw_boolean", (), Some(&kw))?.extract()?;
        if cont {
            self.count += 1;
        }
        Ok(cont)
    }

    fn reject_foreign(&mut self, _py: Python<'_>, cd: &Bound<'_, PyAny>) -> PyResult<()> {
        self.count -= 1;
        self.rejections += 1;
        if self.rejections > std::cmp::max(3, 2 * self.count) {
            if self.count < self.min_size {
                cd.call_method0("mark_invalid")?;
            } else {
                self.force_stop = true;
            }
        }
        Ok(())
    }
}

// ---- drawing ----------------------------------------------------------------

fn child_node<'a>(
    child: &Py<PyAny>,
    py: Python<'a>,
) -> PyResult<pyo3::PyRef<'a, SearchStrategy>> {
    let ss = child.bind(py).downcast::<SearchStrategy>()?;
    Ok(ss.borrow())
}

fn draw_child(
    child: &Py<PyAny>,
    data: &Bound<'_, ConjectureData>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    match child_node(child, py) {
        Ok(ssref) => {
            // A Python subclass of SearchStrategy that overrides do_draw (defined OUTSIDE
            // _engine — e.g. the hypothesis_fast.extra.* array strategies, or our LazyStrategy)
            // carries only a placeholder node, so its node MUST NOT be drawn directly: route it
            // back through cd.draw, which dispatches to its own do_draw (with the right span).
            // Exact / _engine-internal natives keep the fast draw_node path with no extra call.
            let cb = child.bind(py);
            if !cb.is_exact_instance_of::<SearchStrategy>()
                && cb
                    .get_type()
                    .getattr(intern!(py, "__module__"))
                    .ok()
                    .map(|m| !m.eq(intern!(py, "hypothesis_fast._engine")).unwrap_or(false))
                    .unwrap_or(false)
            {
                return Ok(data.call_method1("draw", (cb,))?.unbind());
            }
            draw_node(&ssref.node, data, py)
        }
        // A FOREIGN (real-hypothesis) strategy child — e.g. a registered real `set`
        // strategy spliced into a native one_of by from_type's generic resolution. Draw
        // it through the native cd (the reverse interop bridge handles it).
        Err(_) => Ok(data.call_method1("draw", (child.bind(py),))?.unbind()),
    }
}

fn draw_child_foreign(
    child: &Py<PyAny>,
    cd: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let cb = child.bind(py);
    match child_node(child, py) {
        Ok(ssref) => {
            // A Python subclass of SearchStrategy that overrides do_draw (defined OUTSIDE
            // _engine — e.g. the extra.* array strategies / our LazyStrategy) carries only a
            // placeholder node; route it back through the (real) cd's draw so its own do_draw
            // runs, exactly as the native draw_child does. Otherwise it would draw the empty
            // base node and yield None (e.g. real .example() of an extra.pandas series whose
            // elements are `from_dtype(...).map(...)` over our native LazyStrategy).
            if !cb.is_exact_instance_of::<SearchStrategy>()
                && cb
                    .get_type()
                    .getattr(intern!(py, "__module__"))
                    .ok()
                    .map(|m| !m.eq(intern!(py, "hypothesis_fast._engine")).unwrap_or(false))
                    .unwrap_or(false)
            {
                return Ok(cd.call_method1("draw", (cb,))?.unbind());
            }
            draw_node_foreign(&ssref.node, cd, py)
        }
        // A FOREIGN (real-hypothesis) strategy child — draw it through the real cd directly.
        Err(_) => Ok(cd.call_method1("draw", (cb,))?.unbind()),
    }
}

/// Draw a native StrategyNode against a FOREIGN (real-hypothesis) ConjectureData by
/// calling its own draw_*/mark_invalid methods. This keeps ONE cd and one choice
/// sequence (vs a wrapper) — the real engine drawing a native strategy (e.g. a real
/// FilteredStrategy wrapping `st.none()`) records/marks on its own cd. Native `draw_node`
/// (the fast path) is untouched. Covers leaf + common composite nodes; anything else
/// raises (that path keeps its prior failure rather than misbehaving).
pub(crate) fn draw_node_foreign(
    node: &StrategyNode,
    cd: &Bound<'_, PyAny>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    let draw_int = |lo: Option<&BigInt>, hi: Option<&BigInt>| -> PyResult<Py<PyAny>> {
        let kw = PyDict::new(py);
        if let Some(b) = lo {
            kw.set_item("min_value", b.clone().into_pyobject(py)?)?;
        }
        if let Some(b) = hi {
            kw.set_item("max_value", b.clone().into_pyobject(py)?)?;
        }
        Ok(cd.call_method("draw_integer", (), Some(&kw))?.unbind())
    };
    match node {
        StrategyNode::NoneVal => Ok(py.None()),
        StrategyNode::Just(v) => Ok(v.clone_ref(py)),
        StrategyNode::Nothing => {
            cd.call_method0("mark_invalid")?;
            Ok(py.None())
        }
        StrategyNode::Integers { min, max } => draw_int(min.as_ref(), max.as_ref()),
        StrategyNode::Booleans => {
            let kw = PyDict::new(py);
            kw.set_item("p", 0.5)?;
            Ok(cd.call_method("draw_boolean", (), Some(&kw))?.unbind())
        }
        StrategyNode::Floats { min, max, allow_nan, snm, .. } => {
            let kw = PyDict::new(py);
            kw.set_item("min_value", *min)?;
            kw.set_item("max_value", *max)?;
            kw.set_item("allow_nan", *allow_nan)?;
            kw.set_item("smallest_nonzero_magnitude", *snm)?;
            Ok(cd.call_method("draw_float", (), Some(&kw))?.unbind())
        }
        StrategyNode::SampledFrom { elements, .. } => {
            if elements.is_empty() {
                cd.call_method0("mark_invalid")?;
                return Ok(py.None());
            }
            let i = draw_int(Some(&BigInt::from(0)), Some(&BigInt::from(elements.len() - 1)))?;
            let idx: usize = i.bind(py).extract()?;
            Ok(elements[idx].clone_ref(py))
        }
        StrategyNode::SampledFromRange { range } => {
            // Lazy range: draw an index in [0, len) — len may be a big int (range(10**100)) —
            // and compute start + i*step directly (range[i] via get_item would overflow C
            // ssize_t for a big index).
            let r = range.bind(py);
            let len: BigInt = r.call_method0("__len__")?.extract()?;
            if len <= BigInt::from(0) {
                cd.call_method0("mark_invalid")?;
                return Ok(py.None());
            }
            let i = draw_int(Some(&BigInt::from(0)), Some(&(&len - 1)))?;
            let idx: BigInt = i.bind(py).extract()?;
            let start: BigInt = r.getattr("start")?.extract()?;
            let step: BigInt = r.getattr("step")?.extract()?;
            Ok((start + idx * step).into_pyobject(py)?.into_any().unbind())
        }
        StrategyNode::OneOf(children) => {
            if children.is_empty() {
                cd.call_method0("mark_invalid")?;
                return Ok(py.None());
            }
            let i = draw_int(Some(&BigInt::from(0)), Some(&BigInt::from(children.len() - 1)))?;
            let idx: usize = i.bind(py).extract()?;
            draw_child_foreign(&children[idx], cd, py)
        }
        StrategyNode::Tuples(children) => {
            let mut out: Vec<Bound<'_, PyAny>> = Vec::with_capacity(children.len());
            for c in children {
                out.push(draw_child_foreign(c, cd, py)?.into_bound(py));
            }
            Ok(PyTuple::new(py, out)?.into_any().unbind())
        }
        StrategyNode::Map { base, func } => {
            let v = draw_child_foreign(base, cd, py)?;
            Ok(func.bind(py).call1((v.bind(py),))?.unbind())
        }
        StrategyNode::Filter { base, func } => {
            for _ in 0..3 {
                let v = draw_child_foreign(base, cd, py)?;
                if func.bind(py).call1((v.bind(py),))?.is_truthy()? {
                    return Ok(v);
                }
            }
            cd.call_method0("mark_invalid")?;
            Ok(py.None())
        }
        StrategyNode::Shared { base, .. } => draw_child_foreign(base, cd, py),
        StrategyNode::Binary { min, max } => {
            let kw = PyDict::new(py);
            kw.set_item("min_size", *min)?;
            kw.set_item("max_size", *max)?;
            Ok(cd.call_method("draw_bytes", (), Some(&kw))?.unbind())
        }
        StrategyNode::Text { intervals, min, max } => {
            let real_iv = py
                .import("hypothesis.internal.intervalsets")?
                .getattr("IntervalSet")?
                .call1((intervals.bind(py).getattr("intervals")?,))?;
            let kw = PyDict::new(py);
            kw.set_item("min_size", *min)?;
            kw.set_item("max_size", *max)?;
            Ok(cd.call_method("draw_string", (real_iv,), Some(&kw))?.unbind())
        }
        // Collections / composites against a real cd: replicate the native draw using the
        // foreign Many geometry (draw_boolean on the real cd) and draw_child_foreign for
        // elements (test_failure_sequence_inducing — a real-hypothesis outer @given drives a
        // native one_of(lists(...), <composite>)).
        StrategyNode::Lists { elem, min, max, unique_by, swap_domain } => {
            if let Some(domain) = swap_domain {
                // Sample WITHOUT replacement against a real (foreign) cd; dedupe by value
                // since the domain may contain duplicates (e.g. sampled_from([0]*100)).
                let result = PyList::empty(py);
                let seen = PySet::empty(py)?;
                let mut remaining: Vec<Py<PyAny>> = domain.iter().map(|v| v.clone_ref(py)).collect();
                let mut many = Many::new(*min, *max);
                while !remaining.is_empty() && many.more_foreign(py, cd)? {
                    let hi = remaining.len() - 1;
                    let j: usize = cd.call_method1("draw_integer", (0usize, hi))?.extract()?;
                    let v = remaining.swap_remove(j);
                    let vb = v.bind(py);
                    if seen.contains(vb)? {
                        many.reject_foreign(py, cd)?;
                    } else {
                        seen.add(vb)?;
                        result.append(vb)?;
                    }
                }
                return Ok(result.into_any().unbind());
            }
            let result = PyList::empty(py);
            let mut many = Many::new(*min, *max);
            match unique_by {
                None => {
                    while many.more_foreign(py, cd)? {
                        let v = draw_child_foreign(elem, cd, py)?;
                        result.append(v.bind(py))?;
                    }
                }
                Some(keyfn) => {
                    let kf = keyfn.bind(py);
                    let key_funcs: Vec<Bound<'_, PyAny>> = if kf.is_callable() {
                        vec![kf.clone()]
                    } else {
                        let mut fs = Vec::new();
                        for f in kf.try_iter()? {
                            fs.push(f?);
                        }
                        fs
                    };
                    let seen: Vec<Bound<'_, PySet>> = (0..key_funcs.len())
                        .map(|_| PySet::empty(py))
                        .collect::<PyResult<_>>()?;
                    while many.more_foreign(py, cd)? {
                        let v = draw_child_foreign(elem, cd, py)?;
                        let mut keys: Vec<Bound<'_, PyAny>> = Vec::with_capacity(key_funcs.len());
                        let mut collision = false;
                        for (f, s) in key_funcs.iter().zip(seen.iter()) {
                            let key = f.call1((v.bind(py),))?;
                            if s.contains(&key)? {
                                collision = true;
                                break;
                            }
                            keys.push(key);
                        }
                        if collision {
                            many.reject_foreign(py, cd)?;
                        } else {
                            for (s, key) in seen.iter().zip(keys.into_iter()) {
                                s.add(&key)?;
                            }
                            result.append(v.bind(py))?;
                        }
                    }
                }
            }
            Ok(result.into_any().unbind())
        }
        StrategyNode::Sets { elem, min, max, frozen } => {
            let result = PyList::empty(py);
            let seen = PySet::empty(py)?;
            let mut many = Many::new(*min, *max);
            while many.more_foreign(py, cd)? {
                let v = draw_child_foreign(elem, cd, py)?;
                let vb = v.bind(py);
                if seen.contains(vb)? {
                    many.reject_foreign(py, cd)?;
                } else {
                    seen.add(vb)?;
                    result.append(vb)?;
                }
            }
            if *frozen {
                Ok(PyFrozenSet::new(py, result.iter())?.into_any().unbind())
            } else {
                Ok(PySet::new(py, result.iter())?.into_any().unbind())
            }
        }
        StrategyNode::Dictionaries { keys, values, min, max } => {
            let result = PyDict::new(py);
            let mut many = Many::new(*min, *max);
            while many.more_foreign(py, cd)? {
                let k = draw_child_foreign(keys, cd, py)?;
                let v = draw_child_foreign(values, cd, py)?;
                let kb = k.bind(py);
                if result.contains(kb)? {
                    many.reject_foreign(py, cd)?;
                } else {
                    result.set_item(kb, v.bind(py))?;
                }
            }
            Ok(result.into_any().unbind())
        }
        StrategyNode::Composite { func, args, kwargs } => {
            // The real cd's own `draw` routes native child strategies back through do_draw ->
            // draw_node_foreign, so a @composite just calls func(cd.draw, *args, **kwargs).
            let draw_fn = cd.getattr("draw")?;
            let extra = args.bind(py).downcast::<PyTuple>()?;
            let mut call_args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(extra.len() + 1);
            call_args.push(draw_fn);
            for a in extra.iter() {
                call_args.push(a);
            }
            let tup = PyTuple::new(py, call_args)?;
            let kw = kwargs.bind(py).downcast::<PyDict>()?;
            Ok(func.bind(py).call(tup, Some(kw))?.unbind())
        }
        _ => Err(invalid_argument(
            py,
            "this native strategy cannot yet be drawn against a real-hypothesis \
             ConjectureData (interop)"
                .to_string(),
        )),
    }
}

/// Record a "Retried draw from <repr> to satisfy filter" event (upstream writes this to
/// ConjectureData.events on the first filter retry). Buffered in a native_strategies
/// thread-local because argument draws run before the per-example build context exists;
/// the @given runner drains it into the per-test-case statistics. Best-effort / non-fatal.
fn record_filter_retry_event(py: Python<'_>, base: &Py<PyAny>, func: &Py<PyAny>) {
    let Ok(brepr) = child_repr(base, py) else {
        return;
    };
    let msg = format!(
        "Retried draw from {}.filter({}) to satisfy filter",
        brepr,
        fn_repr(func, py)
    );
    if let Ok(m) = py.import("hypothesis_fast.native_strategies") {
        if let Ok(rec) = m.getattr("_record_draw_event") {
            let _ = rec.call1((msg,));
        }
    }
}

/// Record how a mapped value was created (`func(base)`) into the active build context's
/// known-object printers, so failure reporting / RepresentationPrinter can show it "as
/// created" (e.g. `Foo(None)`) — test_pretty. Only on the final replay, where reprs are
/// reported, to keep the per-draw hot path cheap; a no-op outside a build context.
fn record_created_call(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    func: &Bound<'_, PyAny>,
    base_val: &Bound<'_, PyAny>,
) {
    let Ok(ctrl) = py.import("hypothesis.control") else {
        return;
    };
    let Ok(ctx) = ctrl.getattr("_current_build_context").and_then(|v| v.getattr("value")) else {
        return;
    };
    if ctx.is_none() {
        return;
    }
    if !ctx.getattr("is_final").and_then(|f| f.is_truthy()).unwrap_or(false) {
        return;
    }
    let kw = PyDict::new(py);
    let Ok(args) = PyList::new(py, [base_val]) else {
        return;
    };
    let _ = kw.set_item("args", args);
    let _ = kw.set_item("kwargs", PyDict::new(py));
    let _ = ctx.call_method("record_call", (result, func), Some(&kw));
}

/// If `strat`'s node is a non-empty `sampled_from`, a clone of its elements.
fn sampled_from_elements(strat: &Py<PyAny>, py: Python<'_>) -> Option<Vec<Py<PyAny>>> {
    let bound = strat.bind(py);
    let ss = bound.downcast::<SearchStrategy>().ok()?;
    match &ss.borrow().node {
        StrategyNode::SampledFrom { elements, .. } if !elements.is_empty() => {
            Some(elements.iter().map(|e| e.clone_ref(py)).collect())
        }
        // A lazily-stored range: materialise it for the filter rejection-stage optimisation
        // (compute allowed indices / Unsatisfiable early) — but ONLY when small. A huge range
        // (range(10**100)) stays lazy and uses ordinary rejection sampling instead.
        StrategyNode::SampledFromRange { range } => {
            let r = range.bind(py);
            let len: BigInt = r.call_method0("__len__").ok()?.extract().ok()?;
            if len <= BigInt::from(0) || len > BigInt::from(1_000_000u32) {
                return None;
            }
            let mut out = Vec::new();
            for it in r.try_iter().ok()? {
                out.push(it.ok()?.unbind());
            }
            Some(out)
        }
        _ => None,
    }
}

fn draw_index_rs(data: &Bound<'_, ConjectureData>, py: Python<'_>, n: usize) -> PyResult<usize> {
    let i = data.borrow_mut().draw_integer_rs(
        py,
        Some(&BigInt::from(0)),
        Some(&BigInt::from(n - 1)),
        None,
        0,
    )?;
    Ok(i.bind(py).extract()?)
}

/// hypothesis SampledFromStrategy.do_filtered_draw: 3 ordinary rejection tries, then (if the
/// finite set isn't fully ruled out) a speculative pick over the still-allowed indices. The
/// short, bounded choice sequence lets the DataTree see a fully-rejected small set as
/// exhausted (→ Unsatisfiable) while a large set stays unexhausted (→ filter_too_much).
fn sampled_from_filter_draw(
    elements: &[Py<PyAny>],
    func: &Py<PyAny>,
    data: &Bound<'_, ConjectureData>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    const MAX_FILTER_CALLS: usize = 10_000;
    let n = elements.len();
    let pred_ok = |el: &Py<PyAny>| -> PyResult<bool> {
        func.bind(py).call1((el.bind(py),))?.is_truthy()
    };
    let mut known_bad: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for _ in 0..3 {
        let i = draw_index_rs(data, py, n)?;
        if !known_bad.contains(&i) {
            if pred_ok(&elements[i])? {
                return Ok(elements[i].clone_ref(py));
            }
            data.borrow_mut().note_discard();
            known_bad.insert(i);
        }
    }
    let max_good = n - known_bad.len();
    if max_good == 0 {
        return Err(data.borrow_mut().mark_invalid_err(py));
    }
    let speculative = draw_index_rs(data, py, std::cmp::min(max_good, MAX_FILTER_CALLS - 3))?;
    let mut allowed: Vec<usize> = Vec::new();
    for i in 0..std::cmp::min(n, MAX_FILTER_CALLS - 3) {
        if known_bad.contains(&i) {
            continue;
        }
        if pred_ok(&elements[i])? {
            if allowed.len() == speculative {
                return Ok(elements[i].clone_ref(py));
            }
            allowed.push(i);
        }
    }
    if allowed.is_empty() {
        return Err(data.borrow_mut().mark_invalid_err(py));
    }
    let j = draw_index_rs(data, py, allowed.len())?;
    Ok(elements[allowed[j]].clone_ref(py))
}

pub(crate) fn draw_node(
    node: &StrategyNode,
    data: &Bound<'_, ConjectureData>,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    match node {
        StrategyNode::Integers { min, max } => {
            // bias toward bounds for wide bounded ranges (IntegersStrategy.do_draw)
            let weights = match (min, max) {
                (Some(lo), Some(hi)) if (hi - lo) > BigInt::from(127) => {
                    let one = BigInt::from(1);
                    let d = PyDict::new(py);
                    d.set_item(lo.clone(), 2.0 / 128.0)?;
                    d.set_item(lo + &one, 1.0 / 128.0)?;
                    d.set_item(hi - &one, 1.0 / 128.0)?;
                    d.set_item(hi.clone(), 2.0 / 128.0)?;
                    Some(d)
                }
                _ => None,
            };
            data.borrow_mut()
                .draw_integer_rs(py, min.as_ref(), max.as_ref(), weights.as_ref(), 0)
        }
        StrategyNode::Booleans => {
            let b = data.borrow_mut().draw_boolean_rs(py, 0.5, None)?;
            Ok(b.into_pyobject(py)?.to_owned().into_any().unbind())
        }
        StrategyNode::Floats { min, max, allow_nan, allow_inf, snm, width } => {
            use crate::floats::{max_finite_for_width, narrow_to_width};
            // When infinities are disallowed but a bound is open (±inf), clamp the
            // effective bound to the width's finite extreme so no infinity (and no
            // value too large for the target width) is generated.
            let wmax = max_finite_for_width(*width);
            let emin = if *min == f64::NEG_INFINITY {
                if !*allow_inf { -wmax } else { *min }
            } else {
                (*min).max(-wmax)
            };
            let emax = if *max == f64::INFINITY {
                if !*allow_inf { wmax } else { *max }
            } else {
                (*max).min(wmax)
            };
            let drawn = data.borrow_mut().draw_float_rs(py, emin, emax, *allow_nan, *snm)?;
            if *width == 64 {
                return Ok(drawn);
            }
            // Narrow the drawn value to the target float width, then clamp back into
            // the effective range so rounding can't push it past a finite bound.
            let v: f64 = drawn.bind(py).extract()?;
            if !v.is_finite() {
                return Ok(drawn);
            }
            let mut n = narrow_to_width(v, *width);
            if n < emin {
                n = emin;
            } else if n > emax {
                n = emax;
            }
            Ok(PyFloat::new(py, n).into_any().unbind())
        }
        StrategyNode::NoneVal => Ok(py.None()),
        StrategyNode::Just(v) => Ok(v.clone_ref(py)),
        StrategyNode::Nothing => Err(data.borrow_mut().mark_invalid_err(py)),
        StrategyNode::SampledFrom { elements, .. } => {
            if elements.is_empty() {
                return Err(data.borrow_mut().mark_invalid_err(py));
            }
            // #3819: a sampled_from whose elements are ALL strategies is a likely one_of
            // mistake; record it (thread-local) so a TypeError mentioning SearchStrategy
            // escaping the test gets the "Was one_of intended?" note. Fast pre-check on
            // the first element keeps the common non-strategy case at one isinstance.
            if elements
                .first()
                .map(|e| e.bind(py).is_instance_of::<SearchStrategy>())
                .unwrap_or(false)
                && elements
                    .iter()
                    .all(|e| e.bind(py).is_instance_of::<SearchStrategy>())
            {
                if let Ok(ctrl) = py.import("hypothesis_fast.control") {
                    let pt = PyTuple::new(py, elements.iter().map(|e| e.bind(py)))?;
                    let _ = ctrl.call_method1("_record_native_3819", (pt,));
                }
            }
            let i = data
                .borrow_mut()
                .draw_integer_rs(py, Some(&BigInt::from(0)), Some(&BigInt::from(elements.len() - 1)), None, 0)?;
            let idx: usize = i.bind(py).extract()?;
            Ok(elements[idx].clone_ref(py))
        }
        StrategyNode::SampledFromRange { range } => {
            // Lazy range: draw a (possibly big-int) index in [0, len) and compute
            // start + i*step (range[i] via get_item would overflow C ssize_t for a big index).
            let r = range.bind(py);
            let len: BigInt = r.call_method0("__len__")?.extract()?;
            if len <= BigInt::from(0) {
                return Err(data.borrow_mut().mark_invalid_err(py));
            }
            let i = data
                .borrow_mut()
                .draw_integer_rs(py, Some(&BigInt::from(0)), Some(&(&len - 1)), None, 0)?;
            let idx: BigInt = i.bind(py).extract()?;
            let start: BigInt = r.getattr("start")?.extract()?;
            let step: BigInt = r.getattr("step")?.extract()?;
            Ok((start + idx * step).into_pyobject(py)?.into_any().unbind())
        }
        StrategyNode::OneOf(children) => {
            if children.is_empty() {
                return Err(data.borrow_mut().mark_invalid_err(py));
            }
            let i = data.borrow_mut().draw_integer_rs(
                py,
                Some(&BigInt::from(0)),
                Some(&BigInt::from(children.len() - 1)),
                None,
                0,
            )?;
            let idx: usize = i.bind(py).extract()?;
            draw_child(&children[idx], data, py)
        }
        StrategyNode::Tuples(children) => {
            let mut out: Vec<Bound<'_, PyAny>> = Vec::with_capacity(children.len());
            for c in children {
                out.push(draw_child(c, data, py)?.into_bound(py));
            }
            Ok(PyTuple::new(py, out)?.into_any().unbind())
        }
        StrategyNode::Lists { elem, min, max, unique_by, swap_domain } => {
            if let Some(domain) = swap_domain {
                // Sample WITHOUT replacement: pop a uniformly-chosen index out of a shrinking
                // copy of the domain (UniqueSampledListStrategy). The domain itself may hold
                // duplicate *values* (e.g. sampled_from([0]*100)), so still dedupe by value.
                let result = PyList::empty(py);
                let seen = PySet::empty(py)?;
                let mut remaining: Vec<Py<PyAny>> = domain.iter().map(|v| v.clone_ref(py)).collect();
                let mut many = Many::new(*min, *max);
                while !remaining.is_empty() && many.more(py, data)? {
                    let hi = BigInt::from(remaining.len() - 1);
                    let j_obj = data
                        .borrow_mut()
                        .draw_integer_rs(py, Some(&BigInt::from(0)), Some(&hi), None, 0)?;
                    let j: usize = j_obj.bind(py).extract()?;
                    let v = remaining.swap_remove(j);
                    let vb = v.bind(py);
                    if seen.contains(vb)? {
                        many.reject(py, data)?;
                    } else {
                        seen.add(vb)?;
                        result.append(vb)?;
                    }
                }
                return Ok(result.into_any().unbind());
            }
            let result = PyList::empty(py);
            let mut many = Many::new(*min, *max);
            match unique_by {
                None => {
                    while many.more(py, data)? {
                        let v = draw_child(elem, data, py)?;
                        result.append(v.bind(py))?;
                    }
                }
                Some(keyfn) => {
                    let kf = keyfn.bind(py);
                    // unique_by may be a single key fn, or a TUPLE of key fns — each
                    // tracked independently (a value is rejected if ANY key collides).
                    let key_funcs: Vec<Bound<'_, PyAny>> = if kf.is_callable() {
                        vec![kf.clone()]
                    } else {
                        let mut fs = Vec::new();
                        for f in kf.try_iter()? {
                            fs.push(f?);
                        }
                        fs
                    };
                    let seen: Vec<Bound<'_, PySet>> = (0..key_funcs.len())
                        .map(|_| PySet::empty(py))
                        .collect::<PyResult<_>>()?;
                    while many.more(py, data)? {
                        let v = draw_child(elem, data, py)?;
                        let mut keys: Vec<Bound<'_, PyAny>> = Vec::with_capacity(key_funcs.len());
                        let mut collision = false;
                        for (f, s) in key_funcs.iter().zip(seen.iter()) {
                            let key = f.call1((v.bind(py),))?;
                            if s.contains(&key)? {
                                collision = true;
                                break;
                            }
                            keys.push(key);
                        }
                        if collision {
                            many.reject(py, data)?;
                        } else {
                            for (s, key) in seen.iter().zip(keys.into_iter()) {
                                s.add(&key)?;
                            }
                            result.append(v.bind(py))?;
                        }
                    }
                }
            }
            Ok(result.into_any().unbind())
        }
        StrategyNode::Deferred { thunk } => {
            // Lazy validation (matches hypothesis): the definition must be callable,
            // return a SearchStrategy, and not return the deferred strategy itself.
            let tb = thunk.bind(py);
            let mk_invalid = |msg: String| -> PyErr {
                match py
                    .import("hypothesis_fast.errors")
                    .and_then(|m| m.getattr("InvalidArgument"))
                    .and_then(|c| c.call1((msg,)))
                {
                    Ok(e) => PyErr::from_value(e),
                    Err(e) => e,
                }
            };
            if !tb.is_callable() {
                return Err(mk_invalid(format!(
                    "deferred() was passed a non-callable definition {}",
                    tb.repr()?
                )));
            }
            let strat = tb.call0()?;
            let ss = match strat.downcast::<SearchStrategy>() {
                Ok(s) => s,
                Err(_) => {
                    return Err(mk_invalid(format!(
                        "Expected a SearchStrategy but got {} (type={})",
                        strat.repr()?,
                        strat.get_type().name()?
                    )));
                }
            };
            // definition-as-self: the thunk returned the same deferred (same thunk).
            if let StrategyNode::Deferred { thunk: inner } = &ss.borrow().node {
                if inner.bind(py).is(tb) {
                    return Err(mk_invalid(
                        "Cannot define a deferred strategy to be itself".to_string(),
                    ));
                }
            }
            let cur = DEFERRED_DRAW_DEPTH.with(|d| d.get());
            if cur >= DEFERRED_DRAW_LIMIT {
                // Too deep — an unproductive self-recursion. Discard this example.
                return Err(data.borrow_mut().mark_invalid_err(py));
            }
            DEFERRED_DRAW_DEPTH.with(|d| d.set(cur + 1));
            let ssref = ss.borrow();
            let result = draw_node(&ssref.node, data, py);
            DEFERRED_DRAW_DEPTH.with(|d| d.set(cur));
            result
        }
        StrategyNode::Sets { elem, min, max, frozen } => {
            let result = PyList::empty(py);
            let seen = PySet::empty(py)?;
            let mut many = Many::new(*min, *max);
            while many.more(py, data)? {
                let v = draw_child(elem, data, py)?;
                let vb = v.bind(py);
                if seen.contains(vb)? {
                    many.reject(py, data)?;
                } else {
                    seen.add(vb)?;
                    result.append(vb)?;
                }
            }
            if *frozen {
                Ok(PyFrozenSet::new(py, result.iter())?.into_any().unbind())
            } else {
                Ok(PySet::new(py, result.iter())?.into_any().unbind())
            }
        }
        StrategyNode::Dictionaries { keys, values, min, max } => {
            let result = PyDict::new(py);
            let mut many = Many::new(*min, *max);
            while many.more(py, data)? {
                let k = draw_child(keys, data, py)?;
                let v = draw_child(values, data, py)?;
                let kb = k.bind(py);
                if result.contains(kb)? {
                    many.reject(py, data)?;
                } else {
                    result.set_item(kb, v.bind(py))?;
                }
            }
            Ok(result.into_any().unbind())
        }
        StrategyNode::FixedDict { items } => {
            let result = PyDict::new(py);
            for (k, vstrat) in items {
                let v = draw_child(vstrat, data, py)?;
                result.set_item(k.bind(py), v.bind(py))?;
            }
            Ok(result.into_any().unbind())
        }
        StrategyNode::Text { intervals, min, max } => {
            data.borrow_mut().draw_string_rs(py, intervals.bind(py), *min, *max)
        }
        StrategyNode::Characters { intervals, .. } => {
            data.borrow_mut().draw_string_rs(py, intervals.bind(py), 1, 1)
        }
        StrategyNode::Binary { min, max } => data.borrow_mut().draw_bytes_rs(py, *min, *max),
        StrategyNode::Map { base, func } => {
            let v = draw_child(base, data, py)?;
            let result = func.bind(py).call1((v.bind(py),))?;
            record_created_call(py, &result, func.bind(py), v.bind(py));
            Ok(result.unbind())
        }
        StrategyNode::Filter { base, func } => {
            // Filter over a finite sampled_from: use hypothesis's bounded
            // do_filtered_draw (3 rejection tries, then a speculative index over the
            // remaining), which keeps the choice sequence SHALLOW so the engine's
            // DataTree can detect an all-rejected small set as exhausted → Unsatisfiable.
            if let Some(elements) = sampled_from_elements(base, py) {
                return sampled_from_filter_draw(&elements, func, data, py);
            }
            // try a bounded number of times, rejecting via mark_invalid like upstream.
            let mut tries = 0;
            loop {
                let v = draw_child(base, data, py)?;
                let ok = func.bind(py).call1((v.bind(py),))?.is_truthy()?;
                if ok {
                    return Ok(v);
                }
                // a rejected draw is a discard (test_filter_iterations_are_marked_as_discarded)
                data.borrow_mut().note_discard();
                tries += 1;
                if tries == 1 {
                    // Record the filter-retry event the way upstream does (data.events):
                    // "Retried draw from <strategy repr> to satisfy filter". Drawing runs
                    // before the build context exists, so it goes through a thread-local the
                    // @given runner drains into per-case statistics (test_has_lambdas_in_output).
                    record_filter_retry_event(py, base, func);
                }
                if tries > 50 {
                    return Err(data.borrow_mut().mark_invalid_err(py));
                }
            }
        }
        StrategyNode::Flatmap { base, func } => {
            let v = draw_child(base, data, py)?;
            let strat = func.bind(py).call1((v,))?;
            // upstream FlatMapStrategy check_strategy's the expand() result before drawing
            // — a non-strategy (e.g. `flatmap(lambda n: "a")`) is InvalidArgument, not a
            // raw cast error.
            match strat.downcast::<SearchStrategy>() {
                // expand() may return a do_draw-overriding subclass (e.g. arrays(...) ->
                // LazyStrategy); draw_child routes those back through cd.draw so their do_draw
                // runs, while keeping exact/_engine natives on the fast draw_node path.
                Ok(_) => draw_child(&strat.clone().unbind(), data, py),
                Err(_) => Err(invalid_argument(
                    py,
                    format!(
                        "Expected a SearchStrategy but got {} (type={})",
                        strat.repr()?.to_str()?,
                        strat.get_type().name()?,
                    ),
                )),
            }
        }
        StrategyNode::Uuids { version, allow_nil } => {
            let hi = (BigInt::from(1) << 128) - 1;
            // Exclude the nil UUID (int=0) unless explicitly allowed, by drawing from
            // [1, hi]; with a version set the version bits already prevent nil.
            let lo = BigInt::from(if *allow_nil { 0 } else { 1 });
            let n = data.borrow_mut().draw_integer_rs(py, Some(&lo), Some(&hi), None, 0)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("int", n)?;
            if let Some(v) = version {
                kwargs.set_item("version", *v)?;
            }
            Ok(py.import("uuid")?.getattr("UUID")?.call((), Some(&kwargs))?.unbind())
        }
        StrategyNode::Permutations(values) => {
            let n = values.len();
            let mut result: Vec<Py<PyAny>> = values.iter().map(|p| p.clone_ref(py)).collect();
            for i in 0..n.saturating_sub(1) {
                let j = data.borrow_mut().draw_integer_rs(
                    py,
                    Some(&BigInt::from(i)),
                    Some(&BigInt::from(n - 1)),
                    None,
                    0,
                )?;
                let jj: usize = j.bind(py).extract()?;
                result.swap(i, jj);
            }
            Ok(PyList::new(py, result)?.into_any().unbind())
        }
        StrategyNode::Builds { target, args, kwargs } => {
            let mut av: Vec<Bound<'_, PyAny>> = Vec::with_capacity(args.len());
            for s in args {
                av.push(draw_child(s, data, py)?.into_bound(py));
            }
            let kw = PyDict::new(py);
            for (k, s) in kwargs {
                let v = draw_child(s, data, py)?;
                kw.set_item(k, v.bind(py))?;
            }
            let tup = PyTuple::new(py, av)?;
            let tb = target.bind(py);
            match tb.call(tup, Some(&kw)) {
                Ok(obj) => Ok(obj.unbind()),
                Err(err) if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {
                    // Better messages when constructing `target` fails (port of upstream
                    // BuildsStrategy.do_draw): a no-arg generic/NewType points at from_type;
                    // a @no_type_check target explains the missing inferred arguments.
                    if args.is_empty() && kwargs.is_empty() {
                        let typing = py.import("typing")?;
                        let is_newtype = tb.hasattr("__supertype__").unwrap_or(false);
                        let is_generic = typing
                            .getattr("get_origin")
                            .and_then(|f| f.call1((tb,)))
                            .map(|o| !o.is_none())
                            .unwrap_or(false);
                        if is_newtype || is_generic {
                            let r = tb.repr()?;
                            return Err(invalid_argument(
                                py,
                                format!(
                                    "Calling {r} with no arguments raised an error - try \
                                     using from_type({r}) instead of builds({r})"
                                ),
                            ));
                        }
                    }
                    if tb
                        .getattr("__no_type_check__")
                        .map(|v| v.is_truthy().unwrap_or(false))
                        .unwrap_or(false)
                    {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "This might be because the @no_type_check decorator prevented \
                             Hypothesis from inferring a strategy for some required arguments.",
                        ));
                    }
                    Err(err)
                }
                Err(err) => Err(err),
            }
        }
        StrategyNode::Composite { func, args, kwargs } => {
            // draw callable bound to this data: `data.draw`
            let draw_fn = data.getattr("draw")?;
            let extra = args.bind(py).downcast::<PyTuple>()?;
            let mut call_args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(extra.len() + 1);
            call_args.push(draw_fn);
            for a in extra.iter() {
                call_args.push(a);
            }
            let tup = PyTuple::new(py, call_args)?;
            let kw = kwargs.bind(py).downcast::<PyDict>()?;
            Ok(func.bind(py).call(tup, Some(kw))?.unbind())
        }
        StrategyNode::Dates { min_ord, max_ord } => {
            crate::data::set_inject_candidates(date_inject_candidates(py, *min_ord, *max_ord)?);
            let n = data.borrow_mut().draw_integer_rs(
                py,
                Some(&BigInt::from(*min_ord)),
                Some(&BigInt::from(*max_ord)),
                None,
                MILLENNIUM_ORDINAL,
            );
            crate::data::clear_inject_candidates();
            let n = n?;
            Ok(py
                .import("datetime")?
                .getattr("date")?
                .call_method1("fromordinal", (n,))?
                .unbind())
        }
        StrategyNode::Times { min_us, max_us } => {
            let us = data.borrow_mut().draw_integer_rs(
                py,
                Some(&BigInt::from(*min_us)),
                Some(&BigInt::from(*max_us)),
                None,
                0,
            )?;
            let us_i: i64 = us.bind(py).extract()?;
            let fold = data.borrow_mut().draw_boolean_rs(py, 0.5, None)?;
            let t = time_from_us(py, us_i)?;
            let kw = PyDict::new(py);
            kw.set_item("fold", fold as i64)?;
            Ok(t.call_method("replace", (), Some(&kw))?.unbind())
        }
        StrategyNode::Datetimes { min_ord, max_ord } => {
            crate::data::set_inject_candidates(date_inject_candidates(py, *min_ord, *max_ord)?);
            let ord = data.borrow_mut().draw_integer_rs(
                py,
                Some(&BigInt::from(*min_ord)),
                Some(&BigInt::from(*max_ord)),
                None,
                MILLENNIUM_ORDINAL,
            );
            crate::data::clear_inject_candidates();
            let ord = ord?;
            let us = data.borrow_mut().draw_integer_rs(
                py,
                Some(&BigInt::from(0)),
                Some(&BigInt::from(US_PER_DAY - 1)),
                None,
                0,
            )?;
            let fold = data.borrow_mut().draw_boolean_rs(py, 0.5, None)?;
            let dt = py.import("datetime")?;
            let date = dt.getattr("date")?.call_method1("fromordinal", (ord,))?;
            let us_i: i64 = us.bind(py).extract()?;
            let time = time_from_us(py, us_i)?;
            let combined = dt.getattr("datetime")?.call_method1("combine", (date, time))?;
            let kw = PyDict::new(py);
            kw.set_item("fold", fold as i64)?;
            Ok(combined.call_method("replace", (), Some(&kw))?.unbind())
        }
        StrategyNode::Timedeltas { min_us, max_us } => {
            // Draw (days, intra-day microseconds) separately — like hypothesis — so
            // shrinking prefers whole days (minimal negative delta = timedelta(-1)).
            use num_integer::Integer;
            let per_day = BigInt::from(US_PER_DAY);
            let min_days = min_us.div_floor(&per_day);
            let max_days = max_us.div_floor(&per_day);
            let min_within = min_us.mod_floor(&per_day); // in [0, US_PER_DAY)
            let max_within = max_us.mod_floor(&per_day);

            let days = data
                .borrow_mut()
                .draw_integer_rs(py, Some(&min_days), Some(&max_days), None, 0)?;
            let days_b: BigInt = days.bind(py).extract()?;

            let lo = if days_b == min_days { min_within.clone() } else { BigInt::from(0) };
            let hi = if days_b == max_days {
                max_within.clone()
            } else {
                BigInt::from(US_PER_DAY - 1)
            };
            let within = if lo > hi {
                BigInt::from(0)
            } else {
                let w = data.borrow_mut().draw_integer_rs(py, Some(&lo), Some(&hi), None, 0)?;
                w.bind(py).extract()?
            };

            let kwargs = PyDict::new(py);
            kwargs.set_item("days", days_b.into_pyobject(py)?)?;
            kwargs.set_item("microseconds", within.into_pyobject(py)?)?;
            Ok(py
                .import("datetime")?
                .getattr("timedelta")?
                .call((), Some(&kwargs))?
                .unbind())
        }
        StrategyNode::ComplexNumbers { allow_nan } => {
            let re = data
                .borrow_mut()
                .draw_float_rs(py, f64::NEG_INFINITY, f64::INFINITY, *allow_nan, SMALLEST_SUBNORMAL)?;
            let im = data
                .borrow_mut()
                .draw_float_rs(py, f64::NEG_INFINITY, f64::INFINITY, *allow_nan, SMALLEST_SUBNORMAL)?;
            let re_f: f64 = re.bind(py).extract()?;
            let im_f: f64 = im.bind(py).extract()?;
            Ok(pyo3::types::PyComplex::from_doubles(py, re_f, im_f).into_any().unbind())
        }
        StrategyNode::IpAddresses { v6, net_range } => {
            let is_v6 = match v6 {
                Some(b) => *b,
                None => data.borrow_mut().draw_boolean_rs(py, 0.5, None)?,
            };
            let (bits, cls) = if is_v6 { (128u32, "IPv6Address") } else { (32u32, "IPv4Address") };
            // Within `network` if given (draw in [network_address, broadcast_address]), else the
            // whole address space — so ip_addresses(network=X) always yields an address in X.
            let (lo, hi) = match net_range {
                Some((lo, hi)) => (lo.clone(), hi.clone()),
                None => (BigInt::from(0), (BigInt::from(1) << bits) - 1),
            };
            let n = data.borrow_mut().draw_integer_rs(py, Some(&lo), Some(&hi), None, 0)?;
            Ok(py.import("ipaddress")?.getattr(cls)?.call1((n,))?.unbind())
        }
        StrategyNode::Slices { size } => {
            // Faithful port of hypothesis's @composite slices(size).
            let size = *size as i64;
            let slice_fn = py.import("builtins")?.getattr("slice")?;
            let to_obj = |v: Option<i64>| -> Py<PyAny> {
                match v {
                    Some(n) => n.into_pyobject(py).unwrap().into_any().unbind(),
                    None => py.None(),
                }
            };
            let bdraw = |d: &Bound<'_, ConjectureData>| -> PyResult<bool> {
                d.borrow_mut().draw_boolean_rs(py, 0.5, None)
            };
            if size == 0 {
                // step = none() | integers().filter(bool)  (none branch is simplest)
                let step_obj = if bdraw(data)? {
                    let v = data.borrow_mut().draw_integer_rs(py, None, None, None, 0)?;
                    let vb: BigInt = v.bind(py).extract()?;
                    let vb = if vb == BigInt::from(0) { BigInt::from(1) } else { vb };
                    vb.into_pyobject(py)?.into_any().unbind()
                } else {
                    py.None()
                };
                return Ok(slice_fn.call1((py.None(), py.None(), step_obj))?.unbind());
            }
            // start = integers(0, size-1) | none(); stop = integers(0, size) | none()
            let mut start = if bdraw(data)? { None } else { Some(idraw(data, py, 0, size - 1)?) };
            let mut stop = if bdraw(data)? { None } else { Some(idraw(data, py, 0, size)?) };
            let max_step = match (start, stop) {
                (None, None) => size,
                (None, Some(s)) => s,
                (Some(s), None) => s,
                (Some(a), Some(b)) => (a - b).abs(),
            };
            let mut step = idraw(data, py, 1, max_step.max(1))?;
            if (bdraw(data)? && start == stop) || stop.unwrap_or(0) < start.unwrap_or(0) {
                step = -step;
            }
            if bdraw(data)? {
                if let Some(s) = start {
                    start = Some(s - size);
                }
            }
            if bdraw(data)? {
                if let Some(s) = stop {
                    stop = Some(s - size);
                }
            }
            let step_is_none = !bdraw(data)? && step == 1;
            let step_obj = if step_is_none { py.None() } else { to_obj(Some(step)) };
            Ok(slice_fn
                .call1((to_obj(start), to_obj(stop), step_obj))?
                .unbind())
        }
        StrategyNode::Fractions { min, max, max_denom } => {
            let frac = py.import("fractions")?.getattr("Fraction")?;
            let mathmod = py.import("math")?;
            let d = idraw(data, py, 1, *max_denom)?;
            let to_frac = |b: &Py<PyAny>| -> PyResult<Bound<'_, PyAny>> { frac.call1((b.bind(py),)) };
            let n: Py<PyAny> = match (min, max) {
                (Some(mn), Some(mx)) => {
                    let lo = mathmod.call_method1("ceil", (to_frac(mn)?.call_method1("__mul__", (d,))?,))?;
                    let hi = mathmod.call_method1("floor", (to_frac(mx)?.call_method1("__mul__", (d,))?,))?;
                    let lo_b: BigInt = lo.extract()?;
                    let hi_b: BigInt = hi.extract()?;
                    if lo_b > hi_b {
                        return Err(data.borrow_mut().mark_invalid_err(py));
                    }
                    data.borrow_mut().draw_integer_rs(py, Some(&lo_b), Some(&hi_b), None, 0)?
                }
                (Some(mn), None) => {
                    let lo = mathmod.call_method1("ceil", (to_frac(mn)?.call_method1("__mul__", (d,))?,))?;
                    let lo_b: BigInt = lo.extract()?;
                    data.borrow_mut().draw_integer_rs(py, Some(&lo_b), None, None, 0)?
                }
                (None, Some(mx)) => {
                    let hi = mathmod.call_method1("floor", (to_frac(mx)?.call_method1("__mul__", (d,))?,))?;
                    let hi_b: BigInt = hi.extract()?;
                    data.borrow_mut().draw_integer_rs(py, None, Some(&hi_b), None, 0)?
                }
                (None, None) => data.borrow_mut().draw_integer_rs(py, None, None, None, 0)?,
            };
            Ok(frac.call1((n, d))?.unbind())
        }
        StrategyNode::Decimals { min, max, places } => {
            let dec = py.import("decimal")?.getattr("Decimal")?;
            let frac = py.import("fractions")?.getattr("Fraction")?;
            let mathmod = py.import("math")?;
            // draw a rational n/d, then convert to Decimal(n)/Decimal(d), optionally quantized.
            let d = 10_i64.pow(places.unwrap_or(3).clamp(0, 12) as u32);
            let to_frac = |b: &Py<PyAny>| -> PyResult<Bound<'_, PyAny>> { frac.call1((b.bind(py),)) };
            let n: Py<PyAny> = match (min, max) {
                (Some(mn), Some(mx)) => {
                    let lo = mathmod.call_method1("ceil", (to_frac(mn)?.call_method1("__mul__", (d,))?,))?;
                    let hi = mathmod.call_method1("floor", (to_frac(mx)?.call_method1("__mul__", (d,))?,))?;
                    let lo_b: BigInt = lo.extract()?;
                    let hi_b: BigInt = hi.extract()?;
                    if lo_b > hi_b {
                        return Err(data.borrow_mut().mark_invalid_err(py));
                    }
                    data.borrow_mut().draw_integer_rs(py, Some(&lo_b), Some(&hi_b), None, 0)?
                }
                (Some(mn), None) => {
                    let lo = mathmod.call_method1("ceil", (to_frac(mn)?.call_method1("__mul__", (d,))?,))?;
                    let lo_b: BigInt = lo.extract()?;
                    data.borrow_mut().draw_integer_rs(py, Some(&lo_b), None, None, 0)?
                }
                (None, Some(mx)) => {
                    let hi = mathmod.call_method1("floor", (to_frac(mx)?.call_method1("__mul__", (d,))?,))?;
                    let hi_b: BigInt = hi.extract()?;
                    data.borrow_mut().draw_integer_rs(py, None, Some(&hi_b), None, 0)?
                }
                (None, None) => data.borrow_mut().draw_integer_rs(py, None, None, None, 0)?,
            };
            let num = dec.call1((n,))?;
            let den = dec.call1((d,))?;
            Ok(num.call_method1("__truediv__", (den,))?.unbind())
        }
        StrategyNode::Data => {
            Ok(Py::new(py, DataObject::user(data.clone().unbind()))?.into_any())
        }
        StrategyNode::Functions { like, returns, pure } => {
            // Hand the live ConjectureData (as a DataObject) to a Python builder that
            // wraps it in a `like`-signatured callable drawing `returns` at call time.
            let dataobj = Py::new(py, DataObject::internal(data.clone().unbind()))?.into_any();
            let builder = py
                .import("hypothesis_fast.native_strategies")?
                .getattr("_build_function")?;
            Ok(builder
                .call1((like.bind(py), returns.bind(py), *pure, dataobj))?
                .unbind())
        }
        StrategyNode::Randoms { use_true_random, note_method_calls } => {
            // Hand the live data (as a DataObject) to the Python builder, which returns
            // a random.Random whose entropy is drawn from it.
            let dataobj = Py::new(py, DataObject::internal(data.clone().unbind()))?.into_any();
            let builder = py
                .import("hypothesis_fast.native_strategies")?
                .getattr("_build_random")?;
            Ok(builder
                .call1((dataobj, *use_true_random, *note_method_calls))?
                .unbind())
        }
        StrategyNode::Invalid { msg, resolution_failed } => Err(if *resolution_failed {
            resolution_failed_err(py, msg.clone())
        } else {
            invalid_argument(py, msg.clone())
        }),
        StrategyNode::Shared { base, key } => {
            // Cache the drawn value on the cd's instance __dict__ keyed by `key`, so all
            // uses with the same key in one example return the SAME value (shared()).
            let dunder = data.getattr("__dict__")?;
            let dd = dunder.downcast::<PyDict>()?;
            let ck = format!("__shared__{key}");
            match dd.get_item(&ck)? {
                Some(v) => Ok(v.unbind()),
                None => {
                    let v = draw_child(base, data, py)?;
                    dd.set_item(&ck, v.bind(py))?;
                    Ok(v)
                }
            }
        }
    }
}

/// Draw a plain bounded i64 (helper for slices/fraction component draws).
fn idraw(data: &Bound<'_, ConjectureData>, py: Python<'_>, lo: i64, hi: i64) -> PyResult<i64> {
    let v = data
        .borrow_mut()
        .draw_integer_rs(py, Some(&BigInt::from(lo)), Some(&BigInt::from(hi)), None, 0)?;
    v.bind(py).extract()
}

// ---- SearchStrategy pyclass -------------------------------------------------

pub(crate) static STRATEGY_LABEL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
