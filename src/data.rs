//! ConjectureData, ported from hypothesis.internal.conjecture.data.
//!
//! The execution context for one test case. Strategies draw typed choices
//! (integer/boolean/float/string/bytes) through `draw_*`; each top-level
//! (`observe=True`) draw samples a value from the provider (HypothesisProvider,
//! folded in here) and records one typed node. In replay mode (`for_choices`) a
//! prefix is played back with misalignment handling, which is what the shrinker
//! and database reuse rely on. `mark_*` raise StopTest to unwind to the engine.

use num_bigint::BigInt;
use num_traits::ToPrimitive;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyString};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::intervalset::IntervalSet;
use crate::provider;

const BUFFER_SIZE: usize = 8 * 1024;
const COLLECTION_DEFAULT_MAX_SIZE: usize = 10_000_000_000;
const SMALLEST_SUBNORMAL: f64 = 5e-324;
const MAX_DEPTH: i64 = 100;

thread_local! {
    // Domain-specific "interesting" integer candidates for the NEXT integer draw, set by a
    // strategy (e.g. dates() supplies leap-day / boundary ordinals) so the generic integer
    // sampler can inject them. Empty for ordinary integer draws — no effect there. Consulted
    // only during fresh generation (replay consumes the recorded choice), so it can't
    // desync for_choices.
    static INJECT_CANDIDATES: std::cell::RefCell<Vec<num_bigint::BigInt>> =
        const { std::cell::RefCell::new(Vec::new()) };

    // Optional override for the per-example buffer-size limit (max_length). When set
    // (tests.conjecture.common.buffer_size_limit), an example whose recorded choice
    // sequence exceeds it overruns, so the engine can report "too large to finish
    // generating". None ⇒ the default BUFFER_SIZE.
    static BUFFER_LIMIT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };

    // Per-run resolution cache keyed by an object's pointer → resolved strategy. Used by
    // from_type's abstract-class resolution (_resolve_abstract), keyed by the TYPE pointer
    // (stable for the run). from_type(abstract) resolves via deferred(_resolve_abstract),
    // which re-runs on every draw; without this the subclass walk + per-subclass from_type
    // (e.g. over libcst's CSTNode hierarchy) reruns per draw, costing millions of
    // isinstance/abc checks. Saved/restored around a (possibly nested) run by run_native's
    // re-entrancy guard, so a registry change between runs re-resolves.
    static DEFERRED_RESOLVED: std::cell::RefCell<std::collections::HashMap<usize, Py<PyAny>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The memoised resolution for `key` (a stable object pointer), if resolved this run.
pub(crate) fn deferred_cache_get(py: Python<'_>, key: usize) -> Option<Py<PyAny>> {
    DEFERRED_RESOLVED.with(|m| m.borrow().get(&key).map(|v| v.clone_ref(py)))
}

/// Memoise a resolved strategy under `key` for the rest of the run.
pub(crate) fn deferred_cache_put(key: usize, val: Py<PyAny>) {
    DEFERRED_RESOLVED.with(|m| {
        m.borrow_mut().insert(key, val);
    });
}

/// Take (snapshot + empty) the resolution cache — run_native's guard saves the outer run's
/// cache and gives the (possibly nested) run a fresh one.
pub(crate) fn take_deferred_cache() -> std::collections::HashMap<usize, Py<PyAny>> {
    DEFERRED_RESOLVED.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

/// Restore a previously-taken resolution cache.
pub(crate) fn set_deferred_cache(c: std::collections::HashMap<usize, Py<PyAny>>) {
    DEFERRED_RESOLVED.with(|m| *m.borrow_mut() = c);
}

thread_local! {
    // Memoised result of builds()'s argument inference, keyed by "<target ptr>|<sorted
    // explicit kwarg keys>". `infer_builds_kwargs` runs inspect.signature + get_type_hints +
    // a from_type per required param on EVERY builds() construction; under code that rebuilds
    // builds(T) per draw (e.g. hypothesmith's builds_filtering) this was ~67% of generation
    // time. The inferred {param -> strategy} additions are a pure function of the target +
    // explicit-key set + (run-stable) registry, so cache them. Saved/restored per run.
    static BUILDS_INFER_CACHE:
        std::cell::RefCell<std::collections::HashMap<String, Vec<(String, Py<PyAny>)>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The memoised inferred-arg additions for a builds() target+keys, if resolved this run.
pub(crate) fn builds_infer_get(py: Python<'_>, key: &str) -> Option<Vec<(String, Py<PyAny>)>> {
    BUILDS_INFER_CACHE.with(|m| {
        m.borrow()
            .get(key)
            .map(|v| v.iter().map(|(n, s)| (n.clone(), s.clone_ref(py))).collect())
    })
}

/// Memoise builds()'s inferred-arg additions for the rest of the run.
pub(crate) fn builds_infer_put(key: String, val: Vec<(String, Py<PyAny>)>) {
    BUILDS_INFER_CACHE.with(|m| {
        m.borrow_mut().insert(key, val);
    });
}

/// Take (snapshot + empty) the builds-infer cache — run_native's guard saves the outer run's.
pub(crate) fn take_builds_infer_cache(
) -> std::collections::HashMap<String, Vec<(String, Py<PyAny>)>> {
    BUILDS_INFER_CACHE.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

/// Restore a previously-taken builds-infer cache.
pub(crate) fn set_builds_infer_cache(
    c: std::collections::HashMap<String, Vec<(String, Py<PyAny>)>>,
) {
    BUILDS_INFER_CACHE.with(|m| *m.borrow_mut() = c);
}

thread_local! {
    // Memoised builds() STRATEGY objects, keyed by target + arg/kwarg object identities. Code
    // that rebuilds builds(T, **fixed) per draw (hypothesmith) gets an identical strategy each
    // time, so reuse it — skipping the callable check, arg inference, and node allocation.
    // Saved/restored per run; registry-invalidated (the inferred args depend on the registry).
    static BUILDS_STRATEGY_CACHE:
        std::cell::RefCell<std::collections::HashMap<String, Py<PyAny>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The memoised builds() strategy for `key`, if constructed this run.
pub(crate) fn builds_strategy_get(py: Python<'_>, key: &str) -> Option<Py<PyAny>> {
    BUILDS_STRATEGY_CACHE.with(|m| m.borrow().get(key).map(|v| v.clone_ref(py)))
}

/// Memoise a constructed builds() strategy under `key` for the rest of the run.
pub(crate) fn builds_strategy_put(key: String, val: Py<PyAny>) {
    BUILDS_STRATEGY_CACHE.with(|m| {
        m.borrow_mut().insert(key, val);
    });
}

/// Take (snapshot + empty) the builds-strategy cache — run_native's guard saves the outer run's.
pub(crate) fn take_builds_strategy_cache() -> std::collections::HashMap<String, Py<PyAny>> {
    BUILDS_STRATEGY_CACHE.with(|m| std::mem::take(&mut *m.borrow_mut()))
}

/// Restore a previously-taken builds-strategy cache.
pub(crate) fn set_builds_strategy_cache(c: std::collections::HashMap<String, Py<PyAny>>) {
    BUILDS_STRATEGY_CACHE.with(|m| *m.borrow_mut() = c);
}

/// Arm the next integer draw with domain "interesting" candidates (e.g. date ordinals).
pub(crate) fn set_inject_candidates(c: Vec<num_bigint::BigInt>) {
    INJECT_CANDIDATES.with(|cell| *cell.borrow_mut() = c);
}

/// Clear the injected candidates (always call after the targeted draw).
pub(crate) fn clear_inject_candidates() {
    INJECT_CANDIDATES.with(|cell| cell.borrow_mut().clear());
}

/// Take (snapshot + empty) the injected candidates — used by run_native's re-entrancy guard
/// to save/restore this data.rs thread-local around a (possibly nested) run.
pub(crate) fn take_inject_candidates() -> Vec<num_bigint::BigInt> {
    INJECT_CANDIDATES.with(|cell| std::mem::take(&mut *cell.borrow_mut()))
}

/// Constrain the buffer-size limit (max_length) of subsequently-created examples, forcing
/// over-large draws to overrun. Mirrors hypothesis's `buffer_size_limit` test helper.
#[pyfunction]
fn set_buffer_limit(limit: usize) {
    BUFFER_LIMIT.with(|c| c.set(Some(limit)));
}

/// Restore the default buffer-size limit (BUFFER_SIZE).
#[pyfunction]
fn clear_buffer_limit() {
    BUFFER_LIMIT.with(|c| c.set(None));
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChoiceType {
    Integer,
    Float,
    String,
    Bytes,
    Boolean,
}

impl ChoiceType {
    fn name(self) -> &'static str {
        match self {
            ChoiceType::Integer => "integer",
            ChoiceType::Float => "float",
            ChoiceType::String => "string",
            ChoiceType::Bytes => "bytes",
            ChoiceType::Boolean => "boolean",
        }
    }
    fn of_value(v: &Bound<'_, PyAny>) -> Option<ChoiceType> {
        use pyo3::types::{PyBool, PyInt};
        if v.is_instance_of::<PyBool>() {
            Some(ChoiceType::Boolean)
        } else if v.is_instance_of::<PyInt>() {
            Some(ChoiceType::Integer)
        } else if v.is_instance_of::<PyFloat>() {
            Some(ChoiceType::Float)
        } else if v.is_instance_of::<PyString>() {
            Some(ChoiceType::String)
        } else if v.is_instance_of::<PyBytes>() {
            Some(ChoiceType::Bytes)
        } else {
            None
        }
    }
}

struct NodeData {
    ctype: ChoiceType,
    value: Py<PyAny>,
    constraints: Py<PyDict>,
    was_forced: bool,
}

enum PrefixEntry {
    Value(Py<PyAny>),
    Simplest { count: Option<i64> },
}

// Status codes (match hypothesis.internal.conjecture.data.Status)
const STATUS_OVERRUN: u8 = 0;
const STATUS_INVALID: u8 = 1;
const STATUS_VALID: u8 = 2;
const STATUS_INTERESTING: u8 = 3;

fn choice_size(py: Python<'_>, v: &Bound<'_, PyAny>) -> usize {
    crate::database::one_choice_size(py, v).unwrap_or(1)
}

fn stoptest_err(py: Python<'_>, counter: u64) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py.import("hypothesis_fast.errors")?.getattr("StopTest")?;
        Ok(PyErr::from_value(cls.call1((counter,))?))
    };
    build().unwrap_or_else(|e| e)
}

fn frozen_err(py: Python<'_>, name: &str) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py.import("hypothesis_fast.errors")?.getattr("Frozen")?;
        Ok(PyErr::from_value(
            cls.call1((format!("Cannot call {name} on frozen ConjectureData"),))?,
        ))
    };
    build().unwrap_or_else(|e| e)
}

#[pyclass(module = "hypothesis_fast._engine", subclass, dict)]
pub(crate) struct ConjectureData {
    rng: StdRng,
    #[pyo3(get)]
    max_length: usize,
    max_choices: Option<usize>,
    #[pyo3(get)]
    length: usize,
    index: usize,
    #[pyo3(get)]
    status: u8,
    #[pyo3(get)]
    frozen: bool,
    depth: i64,
    #[pyo3(get)]
    max_depth: i64,
    #[pyo3(get)]
    has_discards: bool,
    prefix: Option<Vec<PrefixEntry>>,
    nodes: Vec<NodeData>,
    // span recording (trail-based, like SpanRecord)
    span_trail: Vec<i64>,
    span_labels: Vec<u64>,
    span_label_index: std::collections::HashMap<u64, usize>,
    misaligned: bool,
    interesting_origin: Option<Py<PyAny>>,
    testcounter: u64,
    #[pyo3(get)]
    output: String,
}

const TRAIL_STOP_DISCARD: i64 = 1;
const TRAIL_STOP_NO_DISCARD: i64 = 2;
const TRAIL_CHOICE: i64 = 3;

impl ConjectureData {
    fn seed_from(random: Option<&Bound<'_, PyAny>>) -> u64 {
        if let Some(r) = random {
            if let Ok(v) = r.call_method1("getrandbits", (64,)) {
                if let Ok(seed) = v.extract::<u64>() {
                    return seed;
                }
            }
        }
        rand::random()
    }

    fn build(
        random: Option<&Bound<'_, PyAny>>,
        prefix: Option<Vec<PrefixEntry>>,
        max_choices: Option<usize>,
    ) -> Self {
        let seed = Self::seed_from(random);
        Self::build_with_seed(seed, prefix, max_choices)
    }

    /// Generation-mode ConjectureData with an explicit seed (used by the engine).
    pub(crate) fn new_generate(seed: u64) -> Self {
        Self::build_with_seed(seed, None, None)
    }

    /// Generation over the all-simplest template: every draw returns its index-0
    /// (simplest) value — int→0, bool→false, collections→empty — like hypothesis's
    /// `ChoiceTemplate("simplest", count=None)`. Used to try the simplest example
    /// first so edge-case-only bugs (e.g. fails iff x==0) are reliably found.
    pub(crate) fn new_simplest() -> Self {
        Self::build_with_seed(0, Some(vec![PrefixEntry::Simplest { count: None }]), None)
    }

    /// Replay-mode ConjectureData over a list of concrete choice values.
    pub(crate) fn new_for_choices(py: Python<'_>, choices: &[Py<PyAny>]) -> Self {
        let pfx: Vec<PrefixEntry> =
            choices.iter().map(|c| PrefixEntry::Value(c.clone_ref(py))).collect();
        let mc = pfx.len();
        Self::build_with_seed(rand::random(), Some(pfx), Some(mc))
    }

    fn build_with_seed(
        seed: u64,
        prefix: Option<Vec<PrefixEntry>>,
        max_choices: Option<usize>,
    ) -> Self {
        let mut cd = ConjectureData {
            rng: StdRng::seed_from_u64(seed),
            max_length: BUFFER_LIMIT.with(|c| c.get()).unwrap_or(BUFFER_SIZE),
            max_choices,
            length: 0,
            index: 0,
            status: STATUS_VALID,
            frozen: false,
            depth: -1,
            max_depth: 0,
            has_discards: false,
            prefix,
            nodes: Vec::new(),
            span_trail: Vec::new(),
            span_labels: Vec::new(),
            span_label_index: std::collections::HashMap::new(),
            misaligned: false,
            interesting_origin: None,
            testcounter: 0,
            output: String::new(),
        };
        // TOP_LABEL span; label value is opaque (use 0 sentinel for the top span).
        cd.start_span_rs(0);
        cd
    }

    fn start_span_rs(&mut self, label: u64) {
        self.depth += 1;
        if self.depth > self.max_depth {
            self.max_depth = self.depth;
        }
        let n = self.span_labels.len();
        let idx = *self.span_label_index.entry(label).or_insert(n);
        if idx == n {
            self.span_labels.push(label);
        }
        self.span_trail.push(TRAIL_CHOICE + 1 + idx as i64);
    }

    fn stop_span_rs(&mut self, discard: bool) {
        if self.frozen {
            return;
        }
        if discard {
            self.has_discards = true;
        }
        self.depth -= 1;
        self.span_trail
            .push(if discard { TRAIL_STOP_DISCARD } else { TRAIL_STOP_NO_DISCARD });
    }

    /// Build the constraints dict stored on a node.
    fn record_node(
        &mut self,
        py: Python<'_>,
        ctype: ChoiceType,
        value: &Bound<'_, PyAny>,
        constraints: &Bound<'_, PyDict>,
        was_forced: bool,
    ) {
        self.span_trail.push(TRAIL_CHOICE);
        self.length += choice_size(py, value);
        self.nodes.push(NodeData {
            ctype,
            value: value.clone().unbind(),
            constraints: constraints.clone().unbind(),
            was_forced,
        });
    }

    fn mark(&mut self, py: Python<'_>, status: u8, origin: Option<Py<PyAny>>) -> PyErr {
        self.interesting_origin = origin;
        self.status = status;
        self.freeze_rs();
        stoptest_err(py, self.testcounter)
    }

    fn freeze_rs(&mut self) {
        if self.frozen {
            return;
        }
        while self.depth >= 0 {
            self.stop_span_rs(false);
        }
        self.frozen = true;
    }

    /// True if this example finished with VALID status (the default; not marked
    /// invalid/overrun/interesting). A StopTest unwinding a still-valid example means
    /// the example simply completed, so the engine counts it as a valid run.
    pub(crate) fn status_is_valid(&self) -> bool {
        self.status == STATUS_VALID
    }

    /// True if this example finished OVERRUN (its choice sequence exceeded the buffer-size
    /// limit) — distinct from a filter/assume rejection, so the engine can report
    /// "too large to finish generating" rather than "failed a .filter()".
    pub(crate) fn status_is_overrun(&self) -> bool {
        self.status == STATUS_OVERRUN
    }

    /// The core draw: replay/sample/force, then record if observed.
    fn draw_inner<'py>(
        &mut self,
        py: Python<'py>,
        ctype: ChoiceType,
        constraints: &Bound<'py, PyDict>,
        observe: bool,
        forced: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if self.length >= self.max_length {
            return Err(self.mark(py, STATUS_OVERRUN, None));
        }
        if Some(self.nodes.len()) == self.max_choices {
            return Err(self.mark(py, STATUS_OVERRUN, None));
        }

        let in_prefix = observe
            && self.prefix.is_some()
            && self.index < self.prefix.as_ref().unwrap().len();

        let mut value: Bound<'py, PyAny>;
        if in_prefix {
            value = self.pop_choice(py, ctype, constraints, forced)?;
        } else if forced.is_none() {
            value = self.provider_sample(py, ctype, constraints)?;
        } else {
            value = forced.unwrap().clone();
        }
        if let Some(f) = forced {
            value = f.clone();
        }

        if observe {
            self.record_node(py, ctype, &value, constraints, forced.is_some());
            if self.length > self.max_length {
                return Err(self.mark(py, STATUS_OVERRUN, None));
            }
        }
        Ok(value.unbind())
    }

    fn pop_choice<'py>(
        &mut self,
        py: Python<'py>,
        ctype: ChoiceType,
        constraints: &Bound<'py, PyDict>,
        forced: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // index < prefix.len() guaranteed by caller
        let entry_is_template;
        let template_count;
        {
            let prefix = self.prefix.as_ref().unwrap();
            match &prefix[self.index] {
                PrefixEntry::Simplest { count } => {
                    entry_is_template = true;
                    template_count = *count;
                }
                PrefixEntry::Value(_) => {
                    entry_is_template = false;
                    template_count = None;
                }
            }
        }

        if entry_is_template {
            // simplest template: produce index-0 choice (or forced)
            let choice = if let Some(f) = forced {
                f.clone()
            } else {
                match self.simplest_choice(py, ctype, constraints) {
                    Ok(v) => v,
                    Err(_) => return Err(self.mark(py, STATUS_OVERRUN, None)),
                }
            };
            if let Some(c) = template_count {
                let c2 = c - 1;
                if let Some(PrefixEntry::Simplest { count }) =
                    self.prefix.as_mut().unwrap().get_mut(self.index)
                {
                    *count = Some(c2);
                }
                if c2 < 0 {
                    return Err(self.mark(py, STATUS_OVERRUN, None));
                }
            }
            return Ok(choice);
        }

        // concrete value entry
        let raw = match &self.prefix.as_ref().unwrap()[self.index] {
            PrefixEntry::Value(v) => v.clone_ref(py).into_bound(py),
            _ => unreachable!(),
        };
        let node_ct = ChoiceType::of_value(&raw);
        let permitted = node_ct == Some(ctype)
            && crate::choice::choice_permitted(&raw, constraints).unwrap_or(false);
        let result = if permitted {
            raw
        } else {
            if !self.misaligned {
                self.misaligned = true;
            }
            match self.simplest_choice(py, ctype, constraints) {
                Ok(v) => v,
                Err(_) => return Err(self.mark(py, STATUS_OVERRUN, None)),
            }
        };
        self.index += 1;
        Ok(result)
    }

    fn simplest_choice<'py>(
        &self,
        py: Python<'py>,
        ctype: ChoiceType,
        constraints: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let idx = BigInt::from(0u8);
        crate::choice::choice_from_index(py, idx, ctype.name(), constraints)
            .map(|p| p.into_bound(py))
    }

    fn provider_sample<'py>(
        &mut self,
        py: Python<'py>,
        ctype: ChoiceType,
        constraints: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match ctype {
            ChoiceType::Boolean => {
                let p: f64 = constraints.get_item(intern!(py, "p"))?.unwrap().extract()?;
                let b = if p <= 0.0 {
                    false
                } else if p >= 1.0 {
                    true
                } else {
                    self.rng.gen::<f64>() < p
                };
                Ok(b.into_pyobject(py)?.to_owned().into_any())
            }
            ChoiceType::Integer => {
                let min_value = opt_bigint(constraints, intern!(py, "min_value"))?;
                let max_value = opt_bigint(constraints, intern!(py, "max_value"))?;
                let weights = constraints.get_item(intern!(py, "weights"))?;
                let v = self.sample_integer(py, min_value, max_value, weights)?;
                // Fast path: build the Python int from i64 when it fits — avoids the
                // slower BigInt -> Python int byte-array conversion for the common case.
                Ok(match v.to_i64() {
                    Some(i) => i.into_pyobject(py)?.into_any(),
                    None => v.into_pyobject(py)?.into_any(),
                })
            }
            ChoiceType::Float => {
                let min_value: f64 = constraints.get_item(intern!(py, "min_value"))?.unwrap().extract()?;
                let max_value: f64 = constraints.get_item(intern!(py, "max_value"))?.unwrap().extract()?;
                let allow_nan: bool = constraints.get_item(intern!(py, "allow_nan"))?.unwrap().extract()?;
                let snm: f64 = constraints
                    .get_item(intern!(py, "smallest_nonzero_magnitude"))?
                    .unwrap()
                    .extract()?;
                let f = self.sample_float(min_value, max_value, allow_nan, snm);
                Ok(PyFloat::new(py, f).into_any())
            }
            ChoiceType::String => {
                let isb = constraints.get_item(intern!(py, "intervals"))?.unwrap();
                let iset = isb.downcast::<IntervalSet>()?.borrow();
                let min_size: usize = constraints.get_item(intern!(py, "min_size"))?.unwrap().extract()?;
                let max_size: usize = constraints.get_item(intern!(py, "max_size"))?.unwrap().extract()?;
                self.sample_string(py, &iset, min_size, max_size)
            }
            ChoiceType::Bytes => {
                let min_size: usize = constraints.get_item(intern!(py, "min_size"))?.unwrap().extract()?;
                let max_size: usize = constraints.get_item(intern!(py, "max_size"))?.unwrap().extract()?;
                let n = provider::draw_collection_size(&mut self.rng, min_size, max_size);
                let mut buf = vec![0u8; n];
                self.rng.fill(&mut buf[..]);
                Ok(PyBytes::new(py, &buf).into_any())
            }
        }
    }

    fn sample_integer(
        &mut self,
        _py: Python<'_>,
        min_value: Option<BigInt>,
        max_value: Option<BigInt>,
        weights: Option<Bound<'_, PyAny>>,
    ) -> PyResult<BigInt> {
        // Domain "interesting" injection (e.g. dates() leap-day / boundary ordinals): with a
        // modest probability return one of the supplied in-range candidates, so rare targets
        // (Feb 29 of a single leap year in a range) are reliably generated. No-op when no
        // candidates are armed (every ordinary integer draw).
        let cands: Vec<BigInt> = INJECT_CANDIDATES.with(|c| c.borrow().clone());
        if !cands.is_empty() && self.rng.gen::<f64>() < 0.15 {
            let valid: Vec<BigInt> = cands
                .into_iter()
                .filter(|v| {
                    min_value.as_ref().is_none_or(|m| v >= m)
                        && max_value.as_ref().is_none_or(|m| v <= m)
                })
                .collect();
            if !valid.is_empty() {
                let i = self.rng.gen_range(0..valid.len());
                return Ok(valid[i].clone());
            }
        }
        if let Some(w) = weights {
            if !w.is_none() {
                let wd = w.downcast::<PyDict>()?;
                let mut total = 0.0f64;
                let mut entries: Vec<(BigInt, f64)> = Vec::new();
                for (k, v) in wd.iter() {
                    let kb: BigInt = k.extract()?;
                    let pv: f64 = v.extract()?;
                    total += pv;
                    entries.push((kb, pv));
                }
                let r = self.rng.gen::<f64>();
                if r < (1.0 - total) {
                    return Ok(provider::draw_integer_from_distribution(
                        &mut self.rng,
                        min_value.as_ref(),
                        max_value.as_ref(),
                    ));
                }
                // pick weighted key
                let mut acc = 1.0 - total;
                for (k, pv) in &entries {
                    acc += pv;
                    if r < acc {
                        return Ok(k.clone());
                    }
                }
                return Ok(entries.last().map(|(k, _)| k.clone()).unwrap_or_default());
            }
        }
        Ok(provider::draw_integer_from_distribution(
            &mut self.rng,
            min_value.as_ref(),
            max_value.as_ref(),
        ))
    }

    fn sample_float(&mut self, min_value: f64, max_value: f64, allow_nan: bool, snm: f64) -> f64 {
        use crate::floats::{
            next_down_rs, next_up_rs, permitted_float, SIGNALING_NAN_BITS,
        };
        let snan = f64::from_bits(SIGNALING_NAN_BITS);
        let candidates = [
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -f64::NAN,
            snan,
            -snan,
            min_value,
            next_up_rs(min_value),
            min_value + 1.0,
            max_value - 1.0,
            next_down_rs(max_value),
            max_value,
            // subnormals (the "nasty" tiny floats) — permitted only when
            // smallest_nonzero_magnitude allows them (allow_subnormal=True).
            SMALLEST_SUBNORMAL,
            -SMALLEST_SUBNORMAL,
            f64::MIN_POSITIVE / 2.0,
            -f64::MIN_POSITIVE / 2.0,
        ];
        let weird: Vec<f64> = candidates
            .into_iter()
            .filter(|&f| permitted_float(f, min_value, max_value, allow_nan, snm))
            .collect();
        // Upstream injects special floats via TWO independent mechanisms: a constants
        // pool at p=0.15 (HypothesisProvider._maybe_draw_constant) AND this weird/boundary
        // list at p=0.05 — a combined ~0.2 special-value density. We only have the weird
        // list, so a flat 0.05 leaves nan/inf/boundary draws far rarer than upstream, which
        // is why rare-find tests (a NaN in an allow_nan one_of branch) flaked. Match the
        // density at 0.15 so the engine reliably surfaces these within the example budget.
        if !weird.is_empty() && self.rng.gen::<f64>() < 0.15 {
            let i = self.rng.gen_range(0..weird.len());
            return weird[i];
        }
        let result = provider::draw_float_raw(&mut self.rng);
        let clamped = if allow_nan && result.is_nan() {
            result
        } else {
            crate::floats::float_clamp(result, min_value, max_value, allow_nan, snm)
        };
        if clamped.to_bits() != result.to_bits() && !(result.is_nan() && allow_nan) {
            clamped
        } else {
            result
        }
    }

    fn sample_string<'py>(
        &mut self,
        py: Python<'py>,
        iset: &IntervalSet,
        min_size: usize,
        max_size: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let alpha = iset.alphabet_len();
        if alpha == 0 {
            return Ok(PyString::new(py, "").into_any());
        }
        let n = provider::draw_collection_size(&mut self.rng, min_size, max_size);
        let mut cps: Vec<i64> = Vec::with_capacity(n);
        for _ in 0..n {
            let i = if alpha > 256 {
                if self.rng.gen::<f64>() < 0.2 {
                    self.rng.gen_range(256..alpha)
                } else {
                    self.rng.gen_range(0..256)
                }
            } else {
                self.rng.gen_range(0..alpha)
            };
            cps.push(iset.cp_in_shrink_order(i)?);
        }
        crate::choice::cps_to_pystr(py, &cps).map(|p| p.into_bound(py))
    }
}

// Rust-native draw helpers used by the strategy layer (strategy.rs). Each takes a
// short &mut borrow; the strategy layer must NOT hold these across element draws or
// Python callbacks (re-entrancy would double-borrow the pyclass).
impl ConjectureData {
    pub(crate) fn draw_integer_rs(
        &mut self,
        py: Python<'_>,
        min: Option<&BigInt>,
        max: Option<&BigInt>,
        weights: Option<&Bound<'_, PyDict>>,
        shrink_towards: i64,
    ) -> PyResult<Py<PyAny>> {
        // Fast path: the overwhelmingly-common unbounded `integers()` draw reuses one
        // shared immutable constraints dict (no per-draw dict alloc / key hashing).
        if min.is_none() && max.is_none() && weights.is_none() && shrink_towards == 0 {
            let constraints = unbounded_int_constraints(py)?;
            return self.draw_inner(py, ChoiceType::Integer, &constraints, true, None);
        }
        let none = py.None().into_bound(py);
        let min_obj = match min {
            Some(b) => b.clone().into_pyobject(py)?.into_any(),
            None => none.clone(),
        };
        let max_obj = match max {
            Some(b) => b.clone().into_pyobject(py)?.into_any(),
            None => none.clone(),
        };
        let w_obj = match weights {
            Some(d) => d.clone().into_any(),
            None => none.clone(),
        };
        let constraints = make_constraints(
            py,
            &[
                ("min_value", min_obj),
                ("max_value", max_obj),
                ("weights", w_obj),
                ("shrink_towards", shrink_towards.into_pyobject(py)?.into_any()),
            ],
        )?;
        self.draw_inner(py, ChoiceType::Integer, &constraints, true, None)
    }

    pub(crate) fn draw_float_rs(
        &mut self,
        py: Python<'_>,
        min: f64,
        max: f64,
        allow_nan: bool,
        snm: f64,
    ) -> PyResult<Py<PyAny>> {
        let constraints = make_constraints(
            py,
            &[
                ("min_value", PyFloat::new(py, min).into_any()),
                ("max_value", PyFloat::new(py, max).into_any()),
                ("allow_nan", allow_nan.into_pyobject(py)?.to_owned().into_any()),
                ("smallest_nonzero_magnitude", PyFloat::new(py, snm).into_any()),
            ],
        )?;
        self.draw_inner(py, ChoiceType::Float, &constraints, true, None)
    }

    pub(crate) fn draw_string_rs(
        &mut self,
        py: Python<'_>,
        intervals: &Bound<'_, PyAny>,
        min_size: usize,
        max_size: usize,
    ) -> PyResult<Py<PyAny>> {
        let constraints = make_constraints(
            py,
            &[
                ("intervals", intervals.clone()),
                ("min_size", min_size.into_pyobject(py)?.into_any()),
                ("max_size", max_size.into_pyobject(py)?.into_any()),
            ],
        )?;
        self.draw_inner(py, ChoiceType::String, &constraints, true, None)
    }

    pub(crate) fn draw_bytes_rs(
        &mut self,
        py: Python<'_>,
        min_size: usize,
        max_size: usize,
    ) -> PyResult<Py<PyAny>> {
        let constraints = make_constraints(
            py,
            &[
                ("min_size", min_size.into_pyobject(py)?.into_any()),
                ("max_size", max_size.into_pyobject(py)?.into_any()),
            ],
        )?;
        self.draw_inner(py, ChoiceType::Bytes, &constraints, true, None)
    }

    pub(crate) fn draw_boolean_rs(
        &mut self,
        py: Python<'_>,
        p: f64,
        forced: Option<bool>,
    ) -> PyResult<bool> {
        let constraints = make_constraints(py, &[("p", PyFloat::new(py, p).into_any())])?;
        let forced_obj = match forced {
            Some(b) => Some(b.into_pyobject(py)?.to_owned().into_any()),
            None => None,
        };
        let v = self.draw_inner(py, ChoiceType::Boolean, &constraints, true, forced_obj.as_ref())?;
        v.bind(py).extract()
    }

    pub(crate) fn start_span_pub(&mut self, label: u64) {
        self.start_span_rs(label);
    }
    pub(crate) fn stop_span_pub(&mut self, discard: bool) {
        self.stop_span_rs(discard);
    }
    pub(crate) fn mark_invalid_err(&mut self, py: Python<'_>) -> PyErr {
        self.mark(py, STATUS_INVALID, None)
    }
    /// Record that a draw was discarded (a `.filter()` rejection). Upstream marks the
    /// span discarded; we only need the observable `has_discards` flag
    /// (test_filter_iterations_are_marked_as_discarded), which nothing internal reads.
    pub(crate) fn note_discard(&mut self) {
        self.has_discards = true;
    }
    pub(crate) fn depth_val(&self) -> i64 {
        self.depth
    }
}

fn opt_bigint(c: &Bound<'_, PyDict>, key: &Bound<'_, PyString>) -> PyResult<Option<BigInt>> {
    match c.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

fn make_constraints<'py>(
    py: Python<'py>,
    pairs: &[(&str, Bound<'py, PyAny>)],
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

// The constraints dict for an unbounded integer draw (min/max/weights=None,
// shrink_towards=0) is identical for every such draw — by far the most common case
// (e.g. every `integers()` / list element). Build it ONCE and share the immutable dict
// instead of allocating + hashing a fresh 4-entry dict per draw.
static UNBOUNDED_INT_CONSTRAINTS: GILOnceCell<Py<PyDict>> = GILOnceCell::new();

fn unbounded_int_constraints(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let cached = UNBOUNDED_INT_CONSTRAINTS.get_or_try_init(py, || -> PyResult<Py<PyDict>> {
        let none = py.None().into_bound(py);
        Ok(make_constraints(
            py,
            &[
                ("min_value", none.clone()),
                ("max_value", none.clone()),
                ("weights", none.clone()),
                ("shrink_towards", 0i64.into_pyobject(py)?.into_any()),
            ],
        )?
        .unbind())
    })?;
    Ok(cached.bind(py).clone())
}

/// Coerce a foreign IntervalSet (a real-hypothesis `hypothesis.internal.intervalsets.
/// IntervalSet`, passed by real `datatree.draw_choice` when ConjectureData is aliased to
/// native) into the native engine's IntervalSet, which the string provider requires.
/// A native IntervalSet passes through unchanged (cheap downcast). Any object exposing
/// `.intervals` (the (start, end) pair tuple) is rebuilt as native.
fn coerce_intervals<'py>(
    py: Python<'py>,
    iv: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if iv.downcast::<IntervalSet>().is_ok() {
        return Ok(iv);
    }
    if let Ok(intervals) = iv.getattr("intervals") {
        let cls = py
            .import("hypothesis_fast._engine")?
            .getattr("IntervalSet")?;
        return cls.call1((intervals,));
    }
    Ok(iv)
}

#[pymethods]
impl ConjectureData {
    #[new]
    #[pyo3(signature = (*, random=None, observer=None, provider=None, prefix=None, max_choices=None, provider_kw=None))]
    fn new(
        py: Python<'_>,
        random: Option<Bound<'_, PyAny>>,
        observer: Option<Bound<'_, PyAny>>,
        provider: Option<Bound<'_, PyAny>>,
        prefix: Option<Bound<'_, PyAny>>,
        max_choices: Option<usize>,
        provider_kw: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let _ = (observer, provider, provider_kw);
        let pfx = match prefix {
            Some(seq) => Some(parse_prefix(py, &seq)?),
            None => None,
        };
        Ok(ConjectureData::build(random.as_ref(), pfx, max_choices))
    }

    #[classmethod]
    #[pyo3(signature = (choices, *, observer=None, provider=None, random=None))]
    fn for_choices(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        choices: Bound<'_, PyAny>,
        observer: Option<Bound<'_, PyAny>>,
        provider: Option<Bound<'_, PyAny>>,
        random: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let _ = (observer, provider);
        let pfx = parse_prefix(py, &choices)?;
        let max_choices = prefix_choice_count(&pfx);
        Ok(ConjectureData::build(random.as_ref(), Some(pfx), max_choices))
    }

    #[pyo3(signature = (min_value=None, max_value=None, *, weights=None, shrink_towards=None, forced=None, observe=true))]
    fn draw_integer(
        &mut self,
        py: Python<'_>,
        min_value: Option<Bound<'_, PyAny>>,
        max_value: Option<Bound<'_, PyAny>>,
        weights: Option<Bound<'_, PyAny>>,
        shrink_towards: Option<Bound<'_, PyAny>>,
        forced: Option<Bound<'_, PyAny>>,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let none = py.None().into_bound(py);
        let st = match shrink_towards {
            Some(s) => s,
            None => 0i64.into_pyobject(py)?.into_any(),
        };
        let constraints = make_constraints(
            py,
            &[
                ("min_value", min_value.unwrap_or_else(|| none.clone())),
                ("max_value", max_value.unwrap_or_else(|| none.clone())),
                ("weights", weights.unwrap_or_else(|| none.clone())),
                ("shrink_towards", st),
            ],
        )?;
        self.draw_inner(py, ChoiceType::Integer, &constraints, observe, forced.as_ref())
    }

    #[pyo3(signature = (min_value=None, max_value=None, *, allow_nan=true, smallest_nonzero_magnitude=SMALLEST_SUBNORMAL, forced=None, observe=true))]
    fn draw_float(
        &mut self,
        py: Python<'_>,
        min_value: Option<f64>,
        max_value: Option<f64>,
        allow_nan: bool,
        smallest_nonzero_magnitude: f64,
        forced: Option<Bound<'_, PyAny>>,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let constraints = make_constraints(
            py,
            &[
                (
                    "min_value",
                    PyFloat::new(py, min_value.unwrap_or(f64::NEG_INFINITY)).into_any(),
                ),
                (
                    "max_value",
                    PyFloat::new(py, max_value.unwrap_or(f64::INFINITY)).into_any(),
                ),
                ("allow_nan", allow_nan.into_pyobject(py)?.to_owned().into_any()),
                (
                    "smallest_nonzero_magnitude",
                    PyFloat::new(py, smallest_nonzero_magnitude).into_any(),
                ),
            ],
        )?;
        self.draw_inner(py, ChoiceType::Float, &constraints, observe, forced.as_ref())
    }

    #[pyo3(signature = (intervals, *, min_size=0, max_size=COLLECTION_DEFAULT_MAX_SIZE, forced=None, observe=true))]
    fn draw_string(
        &mut self,
        py: Python<'_>,
        intervals: Bound<'_, PyAny>,
        min_size: usize,
        max_size: usize,
        forced: Option<Bound<'_, PyAny>>,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let intervals = coerce_intervals(py, intervals)?;
        let constraints = make_constraints(
            py,
            &[
                ("intervals", intervals),
                ("min_size", min_size.into_pyobject(py)?.into_any()),
                ("max_size", max_size.into_pyobject(py)?.into_any()),
            ],
        )?;
        self.draw_inner(py, ChoiceType::String, &constraints, observe, forced.as_ref())
    }

    #[pyo3(signature = (min_size=0, max_size=COLLECTION_DEFAULT_MAX_SIZE, *, forced=None, observe=true))]
    fn draw_bytes(
        &mut self,
        py: Python<'_>,
        min_size: usize,
        max_size: usize,
        forced: Option<Bound<'_, PyAny>>,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let constraints = make_constraints(
            py,
            &[
                ("min_size", min_size.into_pyobject(py)?.into_any()),
                ("max_size", max_size.into_pyobject(py)?.into_any()),
            ],
        )?;
        self.draw_inner(py, ChoiceType::Bytes, &constraints, observe, forced.as_ref())
    }

    #[pyo3(signature = (p=0.5, *, forced=None, observe=true))]
    fn draw_boolean(
        &mut self,
        py: Python<'_>,
        p: f64,
        forced: Option<Bound<'_, PyAny>>,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let constraints = make_constraints(py, &[("p", PyFloat::new(py, p).into_any())])?;
        self.draw_inner(py, ChoiceType::Boolean, &constraints, observe, forced.as_ref())
    }

    /// Real-hypothesis `ConjectureData.events` (a dict) — read/written by code that draws a
    /// native strategy through our cd, e.g. FilteredStrategy.do_draw records
    /// `data.events["Retried draw from ... to satisfy filter"] = ""`. Lazily created and
    /// cached in the instance __dict__ so repeated writes accumulate on one object.
    #[getter]
    fn events(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match slf.getattr("_events_dict") {
            Ok(d) if !d.is_none() => Ok(d.unbind()),
            _ => {
                let d = PyDict::new(py);
                slf.setattr("_events_dict", &d)?;
                Ok(d.into_any().unbind())
            }
        }
    }

    #[pyo3(signature = (values, *, forced=None, observe=true))]
    fn choice(
        &mut self,
        py: Python<'_>,
        values: Bound<'_, PyAny>,
        forced: Option<Bound<'_, PyAny>>,
        observe: bool,
    ) -> PyResult<Py<PyAny>> {
        let n = values.len()?;
        let forced_i = match &forced {
            Some(f) => {
                let idx = values.call_method1("index", (f,))?;
                Some(idx.extract::<BigInt>()?)
            }
            None => None,
        };
        let none = py.None().into_bound(py);
        let constraints = make_constraints(
            py,
            &[
                ("min_value", 0i64.into_pyobject(py)?.into_any()),
                ("max_value", (n as i64 - 1).into_pyobject(py)?.into_any()),
                ("weights", none.clone()),
                ("shrink_towards", 0i64.into_pyobject(py)?.into_any()),
            ],
        )?;
        let forced_obj = match forced_i {
            Some(b) => Some(b.into_pyobject(py)?.into_any()),
            None => None,
        };
        let i = self.draw_inner(
            py,
            ChoiceType::Integer,
            &constraints,
            observe,
            forced_obj.as_ref(),
        )?;
        let idx: usize = i.bind(py).extract()?;
        Ok(values.get_item(idx)?.unbind())
    }

    #[pyo3(signature = (strategy, label=None, observe_as=None))]
    fn draw(
        slf: &Bound<'_, ConjectureData>,
        py: Python<'_>,
        strategy: &Bound<'_, PyAny>,
        label: Option<Bound<'_, PyAny>>,
        observe_as: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let _ = observe_as;
        // `label` may be an int span-id OR a str (a real `@composite` does
        // `draw(strategy, label="fieldname")` — hypothesis-jsonschema). We only use it as a
        // span id, so extract an int when given one and otherwise fall back to the default 1.
        let label = label.and_then(|l| l.extract::<u64>().ok());
        match strategy.downcast::<crate::strategy::SearchStrategy>() {
            Ok(ss) => {
                // A Python subclass of SearchStrategy that overrides do_draw (defined OUTSIDE
                // our _engine module, e.g. a custom `class Foo(SearchStrategy)` in a test) has
                // no meaningful node — it must draw via its own do_draw. All native strategies
                // (incl. the extends-SearchStrategy wrapper types like IntegersStrategy) live
                // in _engine and take the fast draw_node path with no extra Python call. An
                // EXACT SearchStrategy (the common wrap-created case) skips the check entirely.
                let foreign = !strategy.is_exact_instance_of::<crate::strategy::SearchStrategy>()
                    && strategy
                        .get_type()
                        .getattr(intern!(py, "__module__"))
                        .ok()
                        .map(|m| !m.eq(intern!(py, "hypothesis_fast._engine")).unwrap_or(false))
                        .unwrap_or(false);
                if foreign {
                    slf.borrow_mut().start_span_pub(label.unwrap_or(1));
                    let r = strategy.call_method1("do_draw", (slf,));
                    slf.borrow_mut().stop_span_pub(false);
                    return r.map(pyo3::Bound::unbind);
                }
                slf.borrow_mut().start_span_pub(label.unwrap_or(1));
                let result = {
                    let node_ref = ss.borrow();
                    crate::strategy::draw_node(&node_ref.node, slf, py)
                };
                slf.borrow_mut().stop_span_pub(false);
                result
            }
            Err(_) => {
                // Reverse interop: a FOREIGN (real-hypothesis) strategy drawn against the
                // native cd, e.g. `st.data().draw(real FeatureStrategy)` or a registered real
                // strategy spliced into a native one_of. The native cd is real-draw-compatible
                // (draw_integer/boolean/float/string/bytes, .provider, start_span/stop_span,
                // mark_invalid), so let the real strategy draw through it via its own do_draw.
                // builds/mapped real strategies call current_build_context().record_call, so if
                // NO real BuildContext is active, push a transient one around the draw.
                // st.shared (used by regex_strategy etc.) reads/writes
                // data._shared_strategy_draws; our cd has a __dict__, so ensure it exists.
                if !slf.hasattr("_shared_strategy_draws").unwrap_or(true) {
                    let _ = slf.setattr("_shared_strategy_draws", PyDict::new(py));
                }
                let ctrl = py.import("hypothesis.control")?;
                if ctrl.call_method0("current_build_context").is_ok() {
                    return Ok(strategy.call_method1("do_draw", (slf,))?.unbind());
                }
                let noop = py.eval(c"(lambda *a, **k: None)", None, None)?;
                let kw = PyDict::new(py);
                kw.set_item("wrapped_test", noop)?;
                let ctx = ctrl.getattr("BuildContext")?.call((slf,), Some(&kw))?;
                ctx.call_method0("__enter__")?;
                let r = strategy.call_method1("do_draw", (slf,));
                let _ = ctx.call_method(
                    "__exit__",
                    (py.None(), py.None(), py.None()),
                    None,
                );
                Ok(r?.unbind())
            }
        }
    }

    fn start_span(&mut self, py: Python<'_>, label: u64) -> PyResult<()> {
        if self.frozen {
            return Err(frozen_err(py, "start_span"));
        }
        self.start_span_rs(label);
        Ok(())
    }

    #[pyo3(signature = (*, discard=false))]
    fn stop_span(&mut self, discard: bool) {
        self.stop_span_rs(discard);
    }

    fn freeze(&mut self) {
        self.freeze_rs();
    }

    /// Per-example id (matches real ConjectureData.testcounter). Used by code that raises
    /// a StopTest tagged with the data it belongs to (test_exceptiongroup).
    #[getter]
    fn testcounter(&self) -> u64 {
        self.testcounter
    }

    /// Real-hypothesis `ConjectureData.slice_comments` (a dict): RepresentationPrinter reads
    /// `context.data.slice_comments` to annotate drawn-argument slices. Lazily created and
    /// cached in the instance __dict__ (test_pretty / reprs-as-created).
    #[getter]
    fn slice_comments(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match slf.getattr("_slice_comments_dict") {
            Ok(d) if !d.is_none() => Ok(d.unbind()),
            _ => {
                let d = PyDict::new(py);
                slf.setattr("_slice_comments_dict", &d)?;
                Ok(d.into_any().unbind())
            }
        }
    }

    /// Real-hypothesis `ConjectureData.arg_slices` (a set of (start, end)): real
    /// `BuildContext.track_arg_label` does `self.data.arg_slices.add((start, end))` for the
    /// drawn-argument repr annotation when a real builds()/@composite/mapped strategy draws
    /// against our native cd (schemathesis). Lazily created + cached in the instance __dict__.
    #[getter]
    fn arg_slices(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match slf.getattr("_arg_slices_set") {
            Ok(s) if !s.is_none() => Ok(s.unbind()),
            _ => {
                let s = pyo3::types::PySet::empty(py)?;
                slf.setattr("_arg_slices_set", &s)?;
                Ok(s.into_any().unbind())
            }
        }
    }

    /// Real-hypothesis `ConjectureData._observability_predicates`
    /// (`defaultdict[str, PredicateCounts]`): real `assume()`/`reject()` (control.py) do
    /// `data._observability_predicates[where].update_count(...)`, e.g. a real numbers strategy
    /// rejecting an out-of-range value against our native cd (pandera). Lazily created as a
    /// real `defaultdict(PredicateCounts)` and cached so the counts accumulate.
    #[getter]
    fn _observability_predicates(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match slf.getattr("_obs_predicates_dd") {
            Ok(d) if !d.is_none() => Ok(d.unbind()),
            _ => {
                let pc = py
                    .import("hypothesis.internal.conjecture.data")?
                    .getattr("PredicateCounts")?;
                let dd = py
                    .import("collections")?
                    .getattr("defaultdict")?
                    .call1((pc,))?;
                slf.setattr("_obs_predicates_dd", &dd)?;
                Ok(dd.unbind())
            }
        }
    }

    fn note(&mut self, py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<()> {
        if self.frozen {
            return Err(frozen_err(py, "note"));
        }
        let s = if let Ok(st) = value.downcast::<PyString>() {
            st.to_string()
        } else {
            value.repr()?.to_string()
        };
        self.output.push_str(&s);
        Ok(())
    }

    #[pyo3(signature = (why=None))]
    fn mark_invalid(&mut self, py: Python<'_>, why: Option<String>) -> PyResult<()> {
        let _ = why;
        Err(self.mark(py, STATUS_INVALID, None))
    }

    fn mark_overrun(&mut self, py: Python<'_>) -> PyResult<()> {
        Err(self.mark(py, STATUS_OVERRUN, None))
    }

    fn mark_interesting(
        &mut self,
        py: Python<'_>,
        interesting_origin: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Err(self.mark(py, STATUS_INTERESTING, Some(interesting_origin.unbind())))
    }

    #[pyo3(signature = (status, interesting_origin=None))]
    fn conclude_test(
        &mut self,
        py: Python<'_>,
        status: u8,
        interesting_origin: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        Err(self.mark(py, status, interesting_origin.map(|o| o.unbind())))
    }

    /// The recorded typed choice values, as a Python tuple.
    #[getter]
    fn choices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let elems: Vec<Bound<'_, PyAny>> =
            self.nodes.iter().map(|n| n.value.clone_ref(py).into_bound(py)).collect();
        Ok(pyo3::types::PyTuple::new(py, elems)?.into_any().unbind())
    }

    /// The recorded nodes, as a tuple of ChoiceNode (from the Python choice module).
    #[getter]
    fn nodes(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let choice_mod = py.import("hypothesis_fast.internal.conjecture.choice")?;
        let node_cls = choice_mod.getattr("ChoiceNode")?;
        let mut out: Vec<Bound<'_, PyAny>> = Vec::with_capacity(self.nodes.len());
        for (i, n) in self.nodes.iter().enumerate() {
            let kwargs = PyDict::new(py);
            kwargs.set_item("type", n.ctype.name())?;
            kwargs.set_item("value", n.value.clone_ref(py))?;
            kwargs.set_item("constraints", n.constraints.clone_ref(py))?;
            kwargs.set_item("was_forced", n.was_forced)?;
            kwargs.set_item("index", i)?;
            out.push(node_cls.call((), Some(&kwargs))?);
        }
        Ok(pyo3::types::PyTuple::new(py, out)?.into_any().unbind())
    }

    /// The "primitive provider" is the object real hypothesis samples raw choices from
    /// (`cd.provider.draw_integer(**constraints)`, e.g. in datatree.draw_choice). Native
    /// ConjectureData IS its own provider — it exposes draw_integer/boolean/float/string/
    /// bytes with the upstream constraint-kwarg signatures — so return self. Lets real
    /// internal code that constructs a native CD (when ConjectureData is aliased to native)
    /// sample through the provider API.
    #[getter]
    fn provider(slf: PyRef<'_, Self>) -> Py<ConjectureData> {
        slf.into()
    }

    /// Provider flag read by real hypothesis (`context.data.provider.avoid_realization`,
    /// e.g. in `cached_strategy`). The native provider never uses symbolic/lazy
    /// realization, so it is always False.
    #[getter]
    fn avoid_realization(&self) -> bool {
        false
    }

    #[getter]
    fn interesting_origin(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.interesting_origin {
            Some(o) => o.clone_ref(py),
            None => py.None(),
        }
    }

    #[getter]
    fn misaligned_at(&self, py: Python<'_>) -> Py<PyAny> {
        // simplified: expose whether a misalignment occurred via a bool-ish None/True
        if self.misaligned {
            true.into_pyobject(py).unwrap().to_owned().into_any().unbind()
        } else {
            py.None()
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ConjectureData(Status={}, {} choices{})",
            self.status,
            self.nodes.len(),
            if self.frozen { ", frozen" } else { "" }
        )
    }
}

fn parse_prefix(py: Python<'_>, seq: &Bound<'_, PyAny>) -> PyResult<Vec<PrefixEntry>> {
    let mut out = Vec::new();
    for item in seq.try_iter()? {
        let item = item?;
        // ChoiceTemplate has attributes .type == "simplest" and .count
        if let Ok(ty) = item.getattr("type") {
            if let Ok(s) = ty.extract::<String>() {
                if s == "simplest" && item.hasattr("count")? {
                    let count: Option<i64> = item.getattr("count")?.extract().ok();
                    out.push(PrefixEntry::Simplest { count });
                    continue;
                }
            }
        }
        out.push(PrefixEntry::Value(item.unbind()));
    }
    let _ = py;
    Ok(out)
}

fn prefix_choice_count(pfx: &[PrefixEntry]) -> Option<usize> {
    let mut total = 0usize;
    for e in pfx {
        match e {
            PrefixEntry::Value(_) => total += 1,
            PrefixEntry::Simplest { count } => match count {
                Some(c) => total += (*c).max(0) as usize,
                None => return None,
            },
        }
    }
    Some(total)
}

/// Clear the per-run resolution caches (abstract-type + builds-arg inference). Called
/// NATIVELY when the type registry changes outside a run (register_type_strategy /
/// temp_registered), so from_type/builds resolutions can't reuse a result computed under a
/// different registry. (Within a run the registry is stable and run_native's guard handles
/// save/restore.)
#[pyfunction]
#[pyo3(name = "_clear_resolution_caches")]
fn clear_resolution_caches() {
    let _ = take_deferred_cache();
    let _ = take_builds_infer_cache();
    let _ = take_builds_strategy_cache();
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ConjectureData>()?;
    m.add_function(wrap_pyfunction!(set_buffer_limit, m)?)?;
    m.add_function(wrap_pyfunction!(clear_buffer_limit, m)?)?;
    m.add_function(wrap_pyfunction!(clear_resolution_caches, m)?)?;
    Ok(())
}
