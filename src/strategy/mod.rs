//! Strategy-node tree + Rust drawers, ported from hypothesis.strategies._internal.
//!
//! Every `st.*` constructor builds a `StrategyNode` wrapped in a `SearchStrategy`
//! pyclass. The engine draws by calling `draw_node(node, data)` which runs
//! ENTIRELY in Rust against the Rust `ConjectureData`, crossing to Python only for
//! user callables (map/filter/flatmap) and final object construction. Borrow
//! discipline: only short `borrow_mut()`s on ConjectureData, never held across a
//! recursive element draw or a Python callback (avoids pyclass double-borrow).

use std::cell::{Cell, RefCell};
use std::collections::HashSet;

use num_bigint::BigInt;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyFrozenSet, PyList, PySet, PyString, PyTuple};

use crate::data::ConjectureData;
use crate::provider::calc_p_continue;

thread_local! {
    // deferred() strategies currently having is_empty computed — a recursion guard so a
    // self-referential `deferred(lambda: tuples(x))` resolves to "empty" instead of
    // recursing forever. Pessimistic (assume empty on re-entry); a real base case in an
    // enclosing one_of/tuple overrides it (one_of is all-empty, tuple is any-empty).
    static DEFERRED_EMPTY_INPROGRESS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    // Nesting depth of deferred() draws — a self-recursive deferred (e.g.
    // deferred(lambda: tuples(x))) would otherwise recurse in Python until a stack
    // overflow; past the limit we mark the example invalid so it's discarded (matching
    // hypothesis, where such draws overrun the choice buffer and produce no example).
    static DEFERRED_DRAW_DEPTH: Cell<u32> = const { Cell::new(0) };
    // deferred() strategies currently resolving .branches (recursion guard).
    static DEFERRED_BRANCHES_INPROGRESS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    // deferred() strategies currently resolving has_reusable_values (recursion guard).
    // Optimistic on re-entry (assume True — the recursive_property default), so an
    // enclosing one_of/tuple's non-reusable branch (a collection/map) drives the result.
    static DEFERRED_REUSABLE_INPROGRESS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    // Types currently being resolved by from_type on this thread — re-entry for the same
    // type means a self-referential type (Tree -> Optional[Tree] -> Tree), which we break
    // by returning a deferred() that re-resolves lazily at draw time (matches upstream's
    // _recurse_guard + deferred). Keyed by id(thing).
    static FROM_TYPE_INPROGRESS: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
}

const DEFERRED_DRAW_LIMIT: u32 = 100;

/// RAII guard: marks a type id as "being resolved" in FROM_TYPE_INPROGRESS for its scope,
/// removing it on drop (even on early return / error). Used by builds-arg inference so a
/// recursive reference to the build TARGET resolves lazily (deferred) — picking up a
/// strategy registered for the target AFTER the builds() is constructed.
struct InProgressGuard {
    id: usize,
    added: bool,
}
impl InProgressGuard {
    fn enter(thing: &Bound<'_, PyAny>) -> Self {
        let id = thing.as_ptr() as usize;
        let added = FROM_TYPE_INPROGRESS.with(|s| s.borrow_mut().insert(id));
        InProgressGuard { id, added }
    }
}
impl Drop for InProgressGuard {
    fn drop(&mut self) {
        if self.added {
            FROM_TYPE_INPROGRESS.with(|s| {
                s.borrow_mut().remove(&self.id);
            });
        }
    }
}

/// is_empty for a deferred() with a self-reference recursion guard.
fn deferred_is_empty(thunk: &Py<PyAny>, py: Python<'_>) -> PyResult<bool> {
    let id = thunk.as_ptr() as usize;
    if DEFERRED_EMPTY_INPROGRESS.with(|s| s.borrow().contains(&id)) {
        return Ok(true);
    }
    DEFERRED_EMPTY_INPROGRESS.with(|s| {
        s.borrow_mut().insert(id);
    });
    let result = (|| -> PyResult<bool> {
        let strat = thunk.bind(py).call0()?;
        match strat.downcast::<SearchStrategy>() {
            Ok(ss) => node_is_empty(&ss.borrow().node, py),
            Err(_) => Ok(false),
        }
    })();
    DEFERRED_EMPTY_INPROGRESS.with(|s| {
        s.borrow_mut().remove(&id);
    });
    result
}

/// has_reusable_values for a deferred() with a self-reference recursion guard.
fn deferred_has_reusable(thunk: &Py<PyAny>, py: Python<'_>) -> PyResult<bool> {
    let id = thunk.as_ptr() as usize;
    if DEFERRED_REUSABLE_INPROGRESS.with(|s| s.borrow().contains(&id)) {
        return Ok(true);
    }
    DEFERRED_REUSABLE_INPROGRESS.with(|s| {
        s.borrow_mut().insert(id);
    });
    let result = (|| -> PyResult<bool> {
        let strat = thunk.bind(py).call0()?;
        match strat.downcast::<SearchStrategy>() {
            Ok(ss) => node_has_reusable_values(&ss.borrow().node, py),
            Err(_) => Ok(false),
        }
    })();
    DEFERRED_REUSABLE_INPROGRESS.with(|s| {
        s.borrow_mut().remove(&id);
    });
    result
}

const COLLECTION_DEFAULT_MAX_SIZE: usize = 10_000_000_000;
const SMALLEST_SUBNORMAL: f64 = 5e-324;

pub(crate) enum StrategyNode {
    Integers { min: Option<BigInt>, max: Option<BigInt> },
    Booleans,
    Floats { min: f64, max: f64, allow_nan: bool, allow_inf: bool, snm: f64, width: u32 },
    NoneVal,
    Just(Py<PyAny>),
    Nothing,
    SampledFrom { elements: Vec<Py<PyAny>>, is_tuple: bool },
    // sampled_from over a `range` — kept LAZY (the range object, not its elements) so that
    // `sampled_from(range(10**100))` neither materialises 10^100 ints (OOM) nor reprs them;
    // drawing indexes the range with a (possibly big-int) index. Matches upstream check_sample.
    SampledFromRange { range: Py<PyAny> },
    OneOf(Vec<Py<PyAny>>),
    Tuples(Vec<Py<PyAny>>),
    // `swap_domain`: when Some, this is a unique list over a small finite domain drawn by
    // sampling WITHOUT replacement (popping from a copy of these values) instead of rejecting
    // duplicates — mirrors hypothesis's UniqueSampledListStrategy and avoids the birthday
    // paradox. `elem` is kept (unchanged) only for the strategy's repr.
    Lists {
        elem: Py<PyAny>,
        min: usize,
        max: usize,
        unique_by: Option<Py<PyAny>>,
        swap_domain: Option<Vec<Py<PyAny>>>,
    },
    Deferred { thunk: Py<PyAny> },
    Sets { elem: Py<PyAny>, min: usize, max: usize, frozen: bool },
    Dictionaries { keys: Py<PyAny>, values: Py<PyAny>, min: usize, max: usize },
    FixedDict { items: Vec<(Py<PyAny>, Py<PyAny>)> },
    Text { intervals: Py<PyAny>, min: usize, max: usize },
    Characters { intervals: Py<PyAny>, repr: String },
    Binary { min: usize, max: usize },
    Map { base: Py<PyAny>, func: Py<PyAny> },
    Filter { base: Py<PyAny>, func: Py<PyAny> },
    Flatmap { base: Py<PyAny>, func: Py<PyAny> },
    Uuids { version: Option<u8>, allow_nil: bool },
    Permutations(Vec<Py<PyAny>>),
    Builds { target: Py<PyAny>, args: Vec<Py<PyAny>>, kwargs: Vec<(String, Py<PyAny>)> },
    Composite { func: Py<PyAny>, args: Py<PyAny>, kwargs: Py<PyAny> },
    Dates { min_ord: i64, max_ord: i64 },
    Times { min_us: i64, max_us: i64 },
    Datetimes { min_ord: i64, max_ord: i64 },
    Timedeltas { min_us: BigInt, max_us: BigInt },
    ComplexNumbers { allow_nan: bool },
    IpAddresses { v6: Option<bool>, net_range: Option<(BigInt, BigInt)> },
    Slices { size: usize },
    Fractions { min: Option<Py<PyAny>>, max: Option<Py<PyAny>>, max_denom: i64 },
    Decimals { min: Option<Py<PyAny>>, max: Option<Py<PyAny>>, places: Option<i32> },
    Data,
    /// functions(like=, returns=, pure=): draws a callable that mimics `like`'s
    /// signature and draws its return value from `returns` against the live data.
    Functions { like: Py<PyAny>, returns: Py<PyAny>, pure: bool },
    /// randoms(): draws a `random.Random`-like object backed by the live data.
    Randoms { use_true_random: bool, note_method_calls: bool },
    /// Deferred validation error — construction succeeds, but validate()/draw raise
    /// InvalidArgument (matches hypothesis's lazy/deferred validation).
    Invalid { msg: String, resolution_failed: bool },
    /// shared(base, key): draws `base` once per example and caches the result on the
    /// ConjectureData (keyed by `key`), so all uses with the same key in one example
    /// return the SAME value. Used for TypeVar resolution (repeated `A` params correlate).
    Shared { base: Py<PyAny>, key: String },
}

/// Interactive draw handle yielded by `st.data()`; `.draw(s)` routes back into
/// the live ConjectureData so the test body can draw on the fly.
#[pyclass(module = "hypothesis_fast._engine")]
pub(crate) struct DataObject {
    cd: Py<ConjectureData>,
    // True only for a user-facing st.data() object: its draws are logged ("Draw N: ...")
    // for failure reporting. The internal DataObjects handed to randoms()/functions()
    // builders set this false so their entropy draws don't pollute the report.
    record: bool,
}

impl DataObject {
    pub(crate) fn user(cd: Py<ConjectureData>) -> Self {
        DataObject { cd, record: true }
    }
    pub(crate) fn internal(cd: Py<ConjectureData>) -> Self {
        DataObject { cd, record: false }
    }
}

#[pymethods]
impl DataObject {
    #[pyo3(signature = (strategy, label=None))]
    fn draw(
        &self,
        py: Python<'_>,
        strategy: Bound<'_, PyAny>,
        label: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        // data.draw(5) / data.draw("not a strategy") must raise InvalidArgument, not pass a
        // non-strategy down to cd.draw (test_data_explicitly_rejects_non_strategies).
        check_strategy(py, &strategy, "Cannot draw from a non-strategy")?;
        let cd = self.cd.bind(py);
        if !self.record {
            let _ = label;
            let clock = crate::engine::draw_clock_start(py);
            let v = cd.call_method1("draw", (strategy,))?.unbind();
            crate::engine::add_draw_secs(crate::engine::draw_clock_elapsed(py, &clock));
            return Ok(v);
        }
        // The per-example draw log lives on the shared cd (not the DataObject), so the
        // "Draw N" counter is shared across every st.data() arg in one example, matching
        // hypothesis's single cached DataObject per ConjectureData.
        let draws = match cd.getattr("_data_draws") {
            Ok(d) if !d.is_none() => d,
            _ => {
                let l = PyList::empty(py);
                cd.setattr("_data_draws", &l)?;
                l.into_any()
            }
        };
        let count = draws.len()? + 1;
        let clock = crate::engine::draw_clock_start(py);
        let result = cd.call_method1("draw", (&strategy,))?;
        crate::engine::add_draw_secs(crate::engine::draw_clock_elapsed(py, &clock));
        let label_s = match label {
            Some(l) if !l.is_none() => format!(" ({})", l.str()?.extract::<String>()?),
            _ => String::new(),
        };
        let pretty: String = py
            .import("hypothesis.vendor.pretty")?
            .getattr("pretty")?
            .call1((&result,))?
            .extract()?;
        draws.call_method1("append", (format!("Draw {count}{label_s}: {pretty}"),))?;
        Ok(result.unbind())
    }
    /// Whether the underlying ConjectureData is frozen (example finished) — a
    /// functions()-generated callable raises InvalidState once this is True.
    fn is_frozen(&self, py: Python<'_>) -> PyResult<bool> {
        self.cd.bind(py).getattr("frozen")?.extract()
    }
    /// The shared underlying ConjectureData. All DataObjects created for the same draw
    /// wrap the SAME cd, so native randoms() can stash per-cd state (seeds_to_states /
    /// states_for_ids) here and have it shared across randoms drawn from one st.data().
    #[getter]
    fn cd(&self, py: Python<'_>) -> Py<ConjectureData> {
        self.cd.clone_ref(py)
    }
    /// Upstream public-ish name (`data.conjecture_data`): the underlying ConjectureData,
    /// used by tests that reach into `.choices` / `.freeze()` / `.testcounter`.
    #[getter]
    fn conjecture_data(&self, py: Python<'_>) -> Py<ConjectureData> {
        self.cd.clone_ref(py)
    }
    /// The "Draw N: ..." log accumulated this example, read by the @given runner to attach
    /// these as notes on a failing exception. Empty list when nothing was drawn.
    #[getter]
    fn _drawn_notes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.cd.bind(py).getattr("_data_draws") {
            Ok(d) if !d.is_none() => Ok(d.unbind()),
            _ => Ok(PyList::empty(py).into_any().unbind()),
        }
    }
    fn __repr__(&self) -> String {
        "data(...)".to_string()
    }
}

const US_PER_DAY: i64 = 86_400_000_000;
// date(2000, 1, 1).toordinal() — datetimes/dates shrink toward the millennium.
const MILLENNIUM_ORDINAL: i64 = 730_120;

fn time_from_us(py: Python<'_>, us: i64) -> PyResult<Bound<'_, PyAny>> {
    let h = us / 3_600_000_000;
    let rem = us % 3_600_000_000;
    let m = rem / 60_000_000;
    let rem = rem % 60_000_000;
    let s = rem / 1_000_000;
    let micro = rem % 1_000_000;
    py.import("datetime")?.getattr("time")?.call1((h, m, s, micro))
}

// ---- collection sizing (Rust `many`) ---------------------------------------

mod draw;
pub(crate) use draw::*;


// `dict` so real-hypothesis internal machinery (recursive_property's `cached_*`
// memoization, etc.) can stash attributes on a native strategy it's consuming.
#[pyclass(module = "hypothesis_fast._engine", subclass, dict)]
pub(crate) struct SearchStrategy {
    pub(crate) node: StrategyNode,
    // Stable per-instance span label, mirroring hypothesis SearchStrategy.label.
    label: u64,
}

impl SearchStrategy {
    fn new(node: StrategyNode) -> SearchStrategy {
        let label = STRATEGY_LABEL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SearchStrategy { node, label }
    }
    fn wrap(py: Python<'_>, node: StrategyNode) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, SearchStrategy::new(node))?.into_any())
    }
}

#[pymethods]
impl SearchStrategy {
    /// Allow Python subclassing — `class Foo(SearchStrategy): def do_draw(self, data): ...`
    /// (test_filtering_most_things_fails_a_health_check). The node is a non-empty placeholder
    /// (a subclass overrides do_draw, which the engine calls instead — see ConjectureData.draw
    /// — but it must not look like an empty/`nothing()` strategy to the @given driver).
    ///
    /// Accept (and ignore) *args/**kwargs so a subclass can define `__init__(self, ...)` and
    /// call `super().__init__()` while constructing as `Foo(a, b)` — the inherited `__new__`
    /// (this `#[new]`) receives the subclass args and must not reject them. Mirrors CPython's
    /// `object.__new__` tolerating extra args when `__init__` is overridden. Needed by the
    /// hypothesis_fast.extra.* custom strategies (ArrayStrategy, BasicIndexStrategy, ...).
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn py_new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> SearchStrategy {
        SearchStrategy::new(StrategyNode::NoneVal)
    }

    fn do_draw(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // Native fast path: a native ConjectureData drives draw_node directly (unchanged).
        if let Ok(native) = data.downcast::<ConjectureData>() {
            return draw_node(&self.node, native, py);
        }
        // The real engine handed us a real ConjectureData (it's drawing a native strategy,
        // e.g. a real FilteredStrategy wrapping st.none()). Draw against it directly so the
        // real cd's choice sequence / mark / spans stay consistent.
        draw_node_foreign(&self.node, data, py)
    }

    // ---- real-hypothesis SearchStrategy interface (so internal code that consumes
    // a native strategy — one_of flattening, datatree span labels, reusability
    // caching — works instead of AttributeError'ing) ----
    #[getter]
    fn label(&self) -> u64 {
        self.label
    }

    /// Real hypothesis's SharedStrategy.calc_label() delegates to `self.base.calc_label()`;
    /// when the base is a native strategy (e.g. `shared(integers())`), it must answer this
    /// rather than AttributeError (test_compatible_nested_shared_strategies_do_not_warn).
    fn calc_label(&self) -> u64 {
        self.label
    }

    #[getter]
    fn branches(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // deferred() passes .branches through to the strategy it resolves to (guarded
        // against self-reference); every other strategy is its own only branch.
        let thunk = match &slf.borrow().node {
            StrategyNode::Deferred { thunk } => Some(thunk.clone_ref(py)),
            _ => None,
        };
        if let Some(thunk) = thunk {
            let id = thunk.as_ptr() as usize;
            if !DEFERRED_BRANCHES_INPROGRESS.with(|s| s.borrow().contains(&id)) {
                DEFERRED_BRANCHES_INPROGRESS.with(|s| {
                    s.borrow_mut().insert(id);
                });
                let res = (|| -> PyResult<Py<PyAny>> {
                    Ok(thunk.bind(py).call0()?.getattr("branches")?.unbind())
                })();
                DEFERRED_BRANCHES_INPROGRESS.with(|s| {
                    s.borrow_mut().remove(&id);
                });
                return res;
            }
        }
        Ok(PyList::new(py, [slf])?.into_any().unbind())
    }

    #[getter]
    fn has_reusable_values(&self, py: Python<'_>) -> PyResult<bool> {
        node_has_reusable_values(&self.node, py)
    }

    /// The character set of a characters()/text() strategy, as a REAL
    /// hypothesis IntervalSet — real-hypothesis regex internals do
    /// `unwrap_strategies(alphabet).intervals & charmap.query(...)`, so it must be a real
    /// IntervalSet (test_regex group/leak/jsonschema). AttributeError on other strategies.
    #[getter]
    fn intervals(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let native = match &self.node {
            StrategyNode::Characters { intervals, .. } | StrategyNode::Text { intervals, .. } => {
                intervals
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>("intervals"));
            }
        };
        let tuples = native.bind(py).getattr("intervals")?;
        Ok(py
            .import("hypothesis.internal.intervalsets")?
            .getattr("IntervalSet")?
            .call1((tuples,))?
            .unbind())
    }

    // Make `SearchStrategy[X]` work (generic subscription) like hypothesis's
    // SearchStrategy[Ex] — cover files use it in annotations / parametrize ids at
    // import time, so without it those modules fail to even collect.
    #[classmethod]
    fn __class_getitem__(
        cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        item: Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))?
            .unbind())
    }

    fn calc_is_empty(&self, py: Python<'_>, recur: Bound<'_, PyAny>) -> PyResult<bool> {
        let _ = recur; // our node knows its own emptiness without recursion
        node_is_empty(&self.node, py)
    }

    fn calc_has_reusable_values(&self, py: Python<'_>, recur: Bound<'_, PyAny>) -> PyResult<bool> {
        let _ = recur; // our node knows its own reusability without external recursion
        node_has_reusable_values(&self.node, py)
    }

    fn map(slf: &Bound<'_, Self>, py: Python<'_>, pack: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if matches!(slf.borrow().node, StrategyNode::Data) {
            return Err(not_a_first_class_strategy(py, "map"));
        }
        // `s.map(identity)` is a no-op — return self (test_identity_map_is_noop), matching
        // upstream SearchStrategy.map's is_identity_function short-circuit.
        if py
            .import("hypothesis.internal.reflection")
            .and_then(|m| m.getattr("is_identity_function"))
            .and_then(|f| f.call1((pack.bind(py),)))
            .and_then(|r| r.is_truthy())
            .unwrap_or(false)
        {
            return Ok(slf.clone().into_any().unbind());
        }
        SearchStrategy::wrap(
            py,
            StrategyNode::Map {
                base: slf.clone().into_any().unbind(),
                func: pack,
            },
        )
    }

    fn filter(slf: &Bound<'_, Self>, py: Python<'_>, condition: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if matches!(slf.borrow().node, StrategyNode::Data) {
            return Err(not_a_first_class_strategy(py, "filter"));
        }
        // Filter-rewriting: an order/equality predicate on integers folds into the bounds
        // (hypothesis IntegersStrategy.filter). We reuse hypothesis.internal.filtering's
        // predicate analysis (a pure, construction-time call) — value generation stays
        // native. Other node types / non-rewritable predicates fall through to a Filter.
        let int_bounds = match &slf.borrow().node {
            StrategyNode::Integers { min, max } => Some((min.clone(), max.clone())),
            _ => None,
        };
        if let Some((mn, mx)) = int_bounds {
            if let Some(rewritten) = rewrite_integer_filter(slf, py, &mn, &mx, &condition)? {
                return Ok(rewritten);
            }
        }
        let float_fields = match &slf.borrow().node {
            StrategyNode::Floats { min, max, allow_nan, allow_inf, snm, width } => {
                Some((*min, *max, *allow_nan, *allow_inf, *snm, *width))
            }
            _ => None,
        };
        if let Some((mn, mx, an, ai, s, w)) = float_fields {
            if let Some(rewritten) = rewrite_float_filter(slf, py, mn, mx, an, ai, s, w, &condition)? {
                return Ok(rewritten);
            }
        }
        let date_bounds = match &slf.borrow().node {
            StrategyNode::Dates { min_ord, max_ord } => Some((*min_ord, *max_ord)),
            _ => None,
        };
        if let Some((mn, mx)) = date_bounds {
            if let Some(rewritten) = rewrite_date_filter(slf, py, mn, mx, &condition)? {
                return Ok(rewritten);
            }
        }
        // Collection/string filter-rewriting (Text/Binary/Lists): len + nonempty + content.
        let coll = match &slf.borrow().node {
            StrategyNode::Text { min, max, .. } => Some((true, false, *min, *max)),
            StrategyNode::Binary { min, max } => Some((true, true, *min, *max)),
            StrategyNode::Lists { min, max, .. } => Some((false, false, *min, *max)),
            StrategyNode::Sets { min, max, .. } => Some((false, false, *min, *max)),
            StrategyNode::Dictionaries { min, max, .. } => Some((false, false, *min, *max)),
            _ => None,
        };
        if let Some((is_str, is_bytes, mn, mx)) = coll {
            if is_str {
                if let Some(r) = try_regex_rewrite(slf, py, mn, mx, is_bytes, &condition)? {
                    return Ok(r);
                }
            }
            match analyze_coll_filter(py, mn, mx, is_str, is_bytes, &condition)? {
                CollRw::SelfUnchanged => return Ok(slf.clone().into_any().unbind()),
                CollRw::Empty => return SearchStrategy::wrap(py, StrategyNode::Nothing),
                CollRw::Resize { min: nm, max: nx, keep } => {
                    let new_node = rebuild_collection_node(&slf.borrow().node, nm, nx, py)?;
                    let typed = build_typed_collection(py, new_node)?;
                    return match keep {
                        None => Ok(typed),
                        Some(c) => {
                            let conds = PyTuple::new(py, [c.bind(py)])?.into_any().unbind();
                            let filter_node = StrategyNode::Filter {
                                base: typed.clone_ref(py),
                                func: c,
                            };
                            FilteredStrategy::build(py, filter_node, typed, conds)
                        }
                    };
                }
                CollRw::Plain => {}
            }
        }
        // `lists(...).map(tuple).filter(len_pred)`: hypothesis pushes a collection-length
        // filter through the map onto the inner list (since len(pack(xs)) == len(xs) for
        // collection packs), then re-applies the outer filter to cover packs that resize.
        let map_info = match &slf.borrow().node {
            StrategyNode::Map { base, func } => Some((base.clone_ref(py), func.clone_ref(py))),
            _ => None,
        };
        if let Some((base, pack)) = map_info {
            if base_is_list(py, &base)? && pack_is_collection_ish(py, &pack)? {
                let new = base
                    .bind(py)
                    .call_method1("filter", (condition.bind(py),))?
                    .unbind();
                // A bare SearchStrategy back means the inner list couldn't rewrite the
                // predicate (Plain/SelfUnchanged) — fall through to a plain outer filter.
                let rewrote = !new
                    .bind(py)
                    .get_type()
                    .is(&py.get_type::<SearchStrategy>());
                if rewrote {
                    let mapped = MappedStrategy::build(py, new, pack)?;
                    let conds = PyTuple::new(py, [condition.bind(py)])?.into_any().unbind();
                    let fnode = StrategyNode::Filter {
                        base: mapped.clone_ref(py),
                        func: condition.clone_ref(py),
                    };
                    return FilteredStrategy::build(py, fnode, mapped, conds);
                }
            }
        }
        SearchStrategy::wrap(
            py,
            StrategyNode::Filter {
                base: slf.clone().into_any().unbind(),
                func: condition,
            },
        )
    }

    /// hypothesis LazyStrategy.wrapped_strategy: the built inner strategy. Numeric nodes
    /// surface a typed IntegersStrategy/FloatStrategy (with the rewritten bounds); a
    /// Filter surfaces a FilteredStrategy; otherwise the strategy itself.
    #[getter]
    fn wrapped_strategy(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &slf.borrow().node {
            StrategyNode::Integers { min, max } => {
                IntegersStrategy::build(py, min.clone(), max.clone())
            }
            StrategyNode::Floats { min, max, allow_nan, allow_inf, snm, width } => {
                let node = StrategyNode::Floats {
                    min: *min,
                    max: *max,
                    allow_nan: *allow_nan,
                    allow_inf: *allow_inf,
                    snm: *snm,
                    width: *width,
                };
                FloatStrategy::build(py, node, *min, *max)
            }
            StrategyNode::Filter { base, func } => {
                // A chained filter (base is itself a Filter) is flattened + replayed so all
                // rewritable predicates fold and the rest become a single flat_conditions
                // tuple. A lone filter keeps the simple wrapping.
                if as_filter_node(py, base).is_some() {
                    flatten_filtered(slf, py)
                } else {
                    let inner = unwrap_wrapped(base, py)?;
                    let conds = PyTuple::new(py, [func.bind(py)])?.into_any().unbind();
                    let node = StrategyNode::Filter {
                        base: base.clone_ref(py),
                        func: func.clone_ref(py),
                    };
                    FilteredStrategy::build(py, node, inner, conds)
                }
            }
            StrategyNode::Text { intervals, min, max } => {
                let node = StrategyNode::Text { intervals: intervals.clone_ref(py), min: *min, max: *max };
                TextStrategy::build(py, node, *min, *max)
            }
            StrategyNode::Binary { min, max } => {
                BytesStrategy::build(py, StrategyNode::Binary { min: *min, max: *max }, *min, *max)
            }
            StrategyNode::Lists { elem, min, max, unique_by, swap_domain } => {
                let node = StrategyNode::Lists {
                    elem: elem.clone_ref(py),
                    min: *min,
                    max: *max,
                    unique_by: unique_by.as_ref().map(|u| u.clone_ref(py)),
                    swap_domain: swap_domain
                        .as_ref()
                        .map(|d| d.iter().map(|v| v.clone_ref(py)).collect()),
                };
                ListStrategy::build(py, node, *min, *max)
            }
            StrategyNode::Sets { elem, min, max, frozen } => {
                let node = StrategyNode::Sets {
                    elem: elem.clone_ref(py),
                    min: *min,
                    max: *max,
                    frozen: *frozen,
                };
                ListStrategy::build(py, node, *min, *max)
            }
            StrategyNode::Dictionaries { keys, values, min, max } => {
                let node = StrategyNode::Dictionaries {
                    keys: keys.clone_ref(py),
                    values: values.clone_ref(py),
                    min: *min,
                    max: *max,
                };
                ListStrategy::build(py, node, *min, *max)
            }
            StrategyNode::Map { base, func } => {
                let inner = unwrap_wrapped(base, py)?;
                MappedStrategy::build(py, inner, func.clone_ref(py))
            }
            _ => Ok(slf.clone().into_any().unbind()),
        }
    }

    fn flatmap(slf: &Bound<'_, Self>, py: Python<'_>, expand: Py<PyAny>) -> PyResult<Py<PyAny>> {
        if matches!(slf.borrow().node, StrategyNode::Data) {
            return Err(not_a_first_class_strategy(py, "flatmap"));
        }
        SearchStrategy::wrap(
            py,
            StrategyNode::Flatmap {
                base: slf.clone().into_any().unbind(),
                func: expand,
            },
        )
    }

    /// `s1 | s2` → one_of(s1, s2).
    fn __or__(slf: &Bound<'_, Self>, py: Python<'_>, other: Py<PyAny>) -> PyResult<Py<PyAny>> {
        // Flatten explicitly-or'd strategies so `(a | b) | c` reprs as
        // one_of(a, b, c), matching hypothesis. (one_of() itself does NOT flatten.)
        let mut children: Vec<Py<PyAny>> = Vec::new();
        if let StrategyNode::OneOf(inner) = &slf.borrow().node {
            for ic in inner {
                children.push(ic.clone_ref(py));
            }
        } else {
            children.push(slf.clone().into_any().unbind());
        }
        let ob = other.bind(py);
        match ob.downcast::<SearchStrategy>() {
            Ok(oss) => {
                if let StrategyNode::OneOf(inner) = &oss.borrow().node {
                    for ic in inner {
                        children.push(ic.clone_ref(py));
                    }
                } else {
                    children.push(other.clone_ref(py));
                }
            }
            Err(_) => {
                // A real-hypothesis strategy (has do_draw) can be a one_of child — it draws via
                // the reverse interop bridge (draw_child's foreign path / native cd.draw). This
                // lets `native_strategy | real_strategy` work, e.g. combining a native strategy
                // with a real from_regex/maybe_pad result (hypothesis-jsonschema). Only a genuine
                // non-strategy is an error.
                if ob.hasattr("do_draw").unwrap_or(false) {
                    children.push(other.clone_ref(py));
                } else {
                    return Err(PyErr::from_value(
                        py.import("builtins")?.getattr("ValueError")?.call1((format!(
                            "Cannot | a SearchStrategy with {}",
                            ob.repr()?
                        ),))?,
                    ));
                }
            }
        }
        SearchStrategy::wrap(py, StrategyNode::OneOf(children))
    }

    /// Draw a single example outside a running test (st.example()). Like hypothesis, this
    /// is for interactive use only: it warns (NonInteractiveExampleWarning) under pytest,
    /// and errors if called inside a running test / strategy definition.
    fn example(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Non-interactive warning (suppressed by the cover fixture / user filters).
        let interactive = py.import("sys").map(|s| s.hasattr("ps1").unwrap_or(false)).unwrap_or(false);
        let under_pytest = py
            .import("os")
            .and_then(|os| os.getattr("environ"))
            .and_then(|e| e.call_method1("get", ("PYTEST_CURRENT_TEST",)))
            .map(|v| !v.is_none())
            .unwrap_or(false);
        if !interactive && under_pytest {
            if let Ok(errs) = py.import("hypothesis.errors") {
                if let Ok(warn_cls) = errs.getattr("NonInteractiveExampleWarning") {
                    let repr = repr_node(&slf.borrow().node, py).unwrap_or_default();
                    let msg = format!(
                        "The `.example()` method is good for exploring strategies, but should \
                         only be used interactively.  We recommend using `@given` for tests - \
                         it performs better, saves and replays failures to avoid flakiness, \
                         and reports minimal examples. (strategy: {repr})"
                    );
                    let _ = py
                        .import("warnings")
                        .and_then(|w| w.call_method1("warn", (msg, warn_cls)));
                }
            }
        }
        // Calling .example() inside @given / find / a strategy definition is an error.
        if let Ok(ctrl) = py.import("hypothesis.control") {
            if let Ok(val) = ctrl
                .getattr("_current_build_context")
                .and_then(|v| v.getattr("value"))
            {
                if !val.is_none() {
                    let depth = val
                        .getattr("data")
                        .and_then(|d| d.getattr("depth"))
                        .and_then(|d| d.extract::<i64>())
                        .unwrap_or(0);
                    let msg = if depth > 0 {
                        "Using example() inside a strategy definition is a bad idea. Instead \
                         consider using hypothesis.strategies.builds() or \
                         @hypothesis.strategies.composite to define your strategy."
                    } else {
                        "Using example() inside a test function is a bad idea. Instead consider \
                         using hypothesis.strategies.data() to draw more examples during testing."
                    };
                    let errs = py.import("hypothesis.errors")?;
                    return Err(PyErr::from_value(
                        errs.getattr("HypothesisException")?.call1((msg,))?,
                    ));
                }
            }
        }
        // Also forbidden while OUR native engine is drawing (find_any / @given argument
        // generation), where no real BuildContext is pushed but example() is still nested
        // inside a running search (test_example_inside_strategy).
        if py
            .import("hypothesis_fast.control")
            .and_then(|c| c.getattr("currently_drawing"))
            .and_then(|f| f.call0())
            .and_then(|r| r.is_truthy())
            .unwrap_or(false)
        {
            let errs = py.import("hypothesis.errors")?;
            return Err(PyErr::from_value(errs.getattr("HypothesisException")?.call1((
                "Using example() inside a strategy definition is a bad idea. Instead \
                 consider using hypothesis.strategies.builds() or \
                 @hypothesis.strategies.composite to define your strategy.",
            ))?));
        }
        // data() only makes sense inside a running test (it draws from the live test data).
        if matches!(slf.borrow().node, StrategyNode::Data) {
            return Err(invalid_argument(
                py,
                "Cannot call example() on st.data(): it can only draw values inside a \
                 running test."
                    .to_string(),
            ));
        }
        for _ in 0..200 {
            let cd = Py::new(py, ConjectureData::new_generate(rand::random()))?;
            // Draw via the cd's polymorphic `draw` (not draw_node directly) so a Python
            // SUBCLASS of SearchStrategy that overrides do_draw — e.g. the extra.* ports'
            // LazyStrategy — runs its own do_draw instead of drawing the (empty) base node,
            // which would yield None. Exact native strategies still take the fast draw_node
            // path inside `draw`, so this is identical for them.
            match cd.bind(py).call_method1("draw", (slf,)) {
                Ok(v) => return Ok(v.unbind()),
                Err(e) => {
                    let n = e.get_type(py).name().map(|s| s.to_string()).unwrap_or_default();
                    if n == "StopTest" {
                        continue;
                    }
                    return Err(e);
                }
            }
        }
        let errors = py.import("hypothesis_fast.errors")?;
        Err(PyErr::from_value(
            errors.getattr("Unsatisfiable")?.call1(("could not find an example",))?,
        ))
    }

    /// Validate the strategy (recursively). Raises the deferred InvalidArgument for
    /// any Invalid node in the tree (matches hypothesis's recursive do_validate).
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        node_validate(&self.node, py)
    }

    #[getter]
    fn is_empty(&self, py: Python<'_>) -> PyResult<bool> {
        node_is_empty(&self.node, py)
    }

    fn __repr__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<String> {
        // A LazyStrategy-style strategy (e.g. functions()) whose repr should show only the
        // explicitly-passed args sets `_hf_repr_override` — honour it before the node default.
        if let Ok(ov) = slf.getattr("_hf_repr_override") {
            if let Ok(s) = ov.extract::<String>() {
                return Ok(s);
            }
        }
        repr_node(&slf.borrow().node, py)
    }

    /// `bool(strategy)` is always True, but it's almost always a mistake (the user
    /// forgot to draw a value, e.g. `if st.booleans():`), so warn — matching upstream
    /// SearchStrategy.__bool__.
    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        let msg = format!(
            "bool({}) is always True, did you mean to draw a value?",
            repr_node(&self.node, py)?
        );
        let hw = py
            .import("hypothesis_fast.errors")?
            .getattr("HypothesisWarning")?;
        py.import("warnings")?.getattr("warn")?.call1((msg, hw))?;
        Ok(true)
    }
}

// ---- Inner typed strategies exposed via SearchStrategy.wrapped_strategy ----
// These mirror hypothesis's internal strategy classes so that filter-rewriting cover
// tests' `isinstance(s.wrapped_strategy, IntegersStrategy)` + `.start`/`.end` /
// `.min_value`/`.max_value` checks pass. The conftest aliases the real internal class
// names (hypothesis.strategies._internal.numbers.IntegersStrategy, …) to these.

// All three EXTEND SearchStrategy so the objects returned by `wrapped_strategy` remain
// drawable (inherit do_draw/label/etc.) — real-hypothesis internal code unwraps our
// LazyStrategy (== SearchStrategy) via `.wrapped_strategy` and then draws the result.

#[pyclass(extends = SearchStrategy, module = "hypothesis_fast._engine")]
pub(crate) struct IntegersStrategy {
    start_v: Option<BigInt>,
    end_v: Option<BigInt>,
}

#[pymethods]
impl IntegersStrategy {
    #[getter]
    fn start(&self) -> Option<BigInt> {
        self.start_v.clone()
    }
    #[getter]
    fn end(&self) -> Option<BigInt> {
        self.end_v.clone()
    }
}

impl IntegersStrategy {
    fn build(py: Python<'_>, min: Option<BigInt>, max: Option<BigInt>) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(SearchStrategy::new(StrategyNode::Integers {
            min: min.clone(),
            max: max.clone(),
        }))
        .add_subclass(IntegersStrategy { start_v: min, end_v: max });
        Ok(Py::new(py, init)?.into_any())
    }
}

#[pyclass(extends = SearchStrategy, module = "hypothesis_fast._engine")]
pub(crate) struct FloatStrategy {
    min_v: f64,
    max_v: f64,
}

#[pymethods]
impl FloatStrategy {
    #[getter]
    fn min_value(&self) -> f64 {
        self.min_v
    }
    #[getter]
    fn max_value(&self) -> f64 {
        self.max_v
    }
}

impl FloatStrategy {
    fn build(py: Python<'_>, node: StrategyNode, min: f64, max: f64) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(SearchStrategy::new(node))
            .add_subclass(FloatStrategy { min_v: min, max_v: max });
        Ok(Py::new(py, init)?.into_any())
    }
}

/// FilteredStrategy: the wrapped_strategy of a non-rewritten `.filter()`. Drawable (extends
/// SearchStrategy with the same Filter node) + exposes the inner strategy + conditions.
#[pyclass(extends = SearchStrategy, module = "hypothesis_fast._engine")]
pub(crate) struct FilteredStrategy {
    filtered: Py<PyAny>,
    conditions: Py<PyAny>,
}

#[pymethods]
impl FilteredStrategy {
    #[getter]
    fn filtered_strategy(&self, py: Python<'_>) -> Py<PyAny> {
        self.filtered.clone_ref(py)
    }
    #[getter]
    fn flat_conditions(&self, py: Python<'_>) -> Py<PyAny> {
        self.conditions.clone_ref(py)
    }
    #[getter]
    fn condition(&self, py: Python<'_>) -> Py<PyAny> {
        self.conditions.clone_ref(py)
    }
}

impl FilteredStrategy {
    fn build(
        py: Python<'_>,
        node: StrategyNode,
        filtered: Py<PyAny>,
        conditions: Py<PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let init = PyClassInitializer::from(SearchStrategy::new(node))
            .add_subclass(FilteredStrategy { filtered, conditions });
        Ok(Py::new(py, init)?.into_any())
    }
}

/// MappedStrategy: the wrapped_strategy of a `.map()`. Drawable (extends SearchStrategy with
/// a Map node) + exposes the inner strategy and the pack/mapping function.
#[pyclass(extends = SearchStrategy, module = "hypothesis_fast._engine")]
pub(crate) struct MappedStrategy {
    mapped: Py<PyAny>,
    pack_fn: Py<PyAny>,
}

#[pymethods]
impl MappedStrategy {
    #[getter]
    fn mapped_strategy(&self, py: Python<'_>) -> Py<PyAny> {
        self.mapped.clone_ref(py)
    }
    #[getter]
    fn pack(&self, py: Python<'_>) -> Py<PyAny> {
        self.pack_fn.clone_ref(py)
    }
}

impl MappedStrategy {
    fn build(py: Python<'_>, inner: Py<PyAny>, pack: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let node = StrategyNode::Map {
            base: inner.clone_ref(py),
            func: pack.clone_ref(py),
        };
        let init = PyClassInitializer::from(SearchStrategy::new(node))
            .add_subclass(MappedStrategy { mapped: inner, pack_fn: pack });
        Ok(Py::new(py, init)?.into_any())
    }
}

// Collection-typed wrapped strategies (min_size/max_size), mirroring hypothesis's
// TextStrategy/BytesStrategy/ListStrategy for the collection filter-rewriting tests.
macro_rules! coll_strategy {
    ($name:ident, $inf_cap:expr) => {
        #[pyclass(extends = SearchStrategy, module = "hypothesis_fast._engine")]
        pub(crate) struct $name {
            min_size_v: usize,
            max_size_v: usize,
        }
        #[pymethods]
        impl $name {
            #[getter]
            fn min_size(&self) -> usize {
                self.min_size_v
            }
            // Unbounded text/list/set/dict report max_size as math.inf (hypothesis);
            // binary keeps the finite COLLECTION_DEFAULT_MAX_SIZE cap.
            #[getter]
            fn max_size(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                if $inf_cap && self.max_size_v == COLLECTION_DEFAULT_MAX_SIZE {
                    Ok(PyFloat::new(py, f64::INFINITY).into_any().unbind())
                } else {
                    Ok(self.max_size_v.into_pyobject(py)?.into_any().unbind())
                }
            }
        }
        impl $name {
            fn build(py: Python<'_>, node: StrategyNode, min: usize, max: usize) -> PyResult<Py<PyAny>> {
                let init = PyClassInitializer::from(SearchStrategy::new(node))
                    .add_subclass($name { min_size_v: min, max_size_v: max });
                Ok(Py::new(py, init)?.into_any())
            }
        }
    };
}
coll_strategy!(TextStrategy, true);
coll_strategy!(BytesStrategy, false);
coll_strategy!(ListStrategy, true);

mod nodes;
pub(crate) use nodes::*;


mod scalars;
pub(crate) use scalars::*;


mod typeres;
pub(crate) use typeres::*;
mod constructors;
pub(crate) use constructors::*;
mod charset;
pub(crate) use charset::*;


pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchStrategy>()?;
    m.add_class::<DataObject>()?;
    m.add_class::<IntegersStrategy>()?;
    m.add_class::<FloatStrategy>()?;
    m.add_class::<FilteredStrategy>()?;
    m.add_class::<TextStrategy>()?;
    m.add_class::<BytesStrategy>()?;
    m.add_class::<ListStrategy>()?;
    m.add_class::<MappedStrategy>()?;
    m.add_function(wrap_pyfunction!(integers, m)?)?;
    m.add_function(wrap_pyfunction!(booleans, m)?)?;
    m.add_function(wrap_pyfunction!(floats, m)?)?;
    m.add_function(wrap_pyfunction!(none, m)?)?;
    m.add_function(wrap_pyfunction!(just, m)?)?;
    m.add_function(wrap_pyfunction!(nothing, m)?)?;
    m.add_function(wrap_pyfunction!(sampled_from, m)?)?;
    m.add_function(wrap_pyfunction!(one_of, m)?)?;
    m.add_function(wrap_pyfunction!(tuples, m)?)?;
    m.add_function(wrap_pyfunction!(lists, m)?)?;
    m.add_function(wrap_pyfunction!(sets, m)?)?;
    m.add_function(wrap_pyfunction!(frozensets, m)?)?;
    m.add_function(wrap_pyfunction!(dictionaries, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_dictionaries, m)?)?;
    m.add_function(wrap_pyfunction!(text, m)?)?;
    m.add_function(wrap_pyfunction!(characters, m)?)?;
    m.add_function(wrap_pyfunction!(binary, m)?)?;
    m.add_function(wrap_pyfunction!(uuids, m)?)?;
    m.add_function(wrap_pyfunction!(permutations, m)?)?;
    m.add_function(wrap_pyfunction!(builds, m)?)?;
    m.add_function(wrap_pyfunction!(composite_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(dates, m)?)?;
    m.add_function(wrap_pyfunction!(times, m)?)?;
    m.add_function(wrap_pyfunction!(datetimes, m)?)?;
    m.add_function(wrap_pyfunction!(timedeltas, m)?)?;
    m.add_function(wrap_pyfunction!(complex_numbers, m)?)?;
    m.add_function(wrap_pyfunction!(ip_addresses, m)?)?;
    m.add_function(wrap_pyfunction!(deferred, m)?)?;
    m.add_function(wrap_pyfunction!(slices, m)?)?;
    m.add_function(wrap_pyfunction!(fractions, m)?)?;
    m.add_function(wrap_pyfunction!(decimals, m)?)?;
    m.add_function(wrap_pyfunction!(recursive, m)?)?;
    m.add_function(wrap_pyfunction!(data, m)?)?;
    m.add_function(wrap_pyfunction!(from_type, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_abstract, m)?)?;
    m.add_function(wrap_pyfunction!(randoms, m)?)?;
    m.add_function(wrap_pyfunction!(functions_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(iterables, m)?)?;
    m.add_function(wrap_pyfunction!(emails, m)?)?;
    m.add_function(wrap_pyfunction!(can_hash, m)?)?;
    m.add_function(wrap_pyfunction!(regex_alphabet_intervals, m)?)?;
    Ok(())
}
