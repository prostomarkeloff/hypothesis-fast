//! Minimal ConjectureRunner: generate + basic shrink, ported in spirit from
//! hypothesis.internal.conjecture.engine.
//!
//! Phase 5 scope: drive the Python test against the native stack — generate
//! examples (ConjectureData per attempt, draw args, run test), and on failure
//! shrink the choice sequence by lowering each node toward its index-0 value and
//! replaying via for_choices. The full lexicographic shrinker is Phase 6.

use num_bigint::BigInt;
use num_traits::{One, Zero};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};

use crate::data::ConjectureData;
use std::time::Instant;

thread_local! {
    // Seconds spent drawing the @given arguments in the most recent run_one_inner call.
    // run_native reads + accumulates it for the HealthCheck.too_slow check (draws happen
    // in Rust, so the Python runner can't measure them itself).
    static LAST_DRAW_SECS: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    // When true, draws are timed via Python's `time.perf_counter()` instead of a Rust
    // `Instant`, so the statistics tests' frozen clock is observed (test_draw_timing).
    // Off by default: timing draws via a Python call on every example/interactive draw is
    // needless GIL traffic on the hot shrink path. core.py turns it on only while it is
    // collecting per-example statistics.
    static USE_PY_CLOCK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Read Python's `time.perf_counter()`. Looked up fresh each call (NOT cached) so a
/// monkeypatched `time.perf_counter` (the statistics tests' frozen clock) is seen. Falls
/// back to 0.0 if the import/call fails — timing is best-effort and must never break a run.
fn perf_counter(py: Python<'_>) -> f64 {
    py.import("time")
        .and_then(|t| t.getattr("perf_counter"))
        .and_then(|f| f.call0())
        .and_then(|v| v.extract::<f64>())
        .unwrap_or(0.0)
}

/// A draw-timing stopwatch: a Rust `Instant` normally (no GIL), or a Python perf_counter
/// reading when USE_PY_CLOCK is on (so a frozen-clock stats run measures generation time).
pub(crate) enum DrawClock {
    Rust(Instant),
    Py(f64),
}

/// Start the draw stopwatch (see USE_PY_CLOCK).
pub(crate) fn draw_clock_start(py: Python<'_>) -> DrawClock {
    if USE_PY_CLOCK.with(|c| c.get()) {
        DrawClock::Py(perf_counter(py))
    } else {
        DrawClock::Rust(Instant::now())
    }
}

/// Seconds elapsed since `clock` started.
pub(crate) fn draw_clock_elapsed(py: Python<'_>, clock: &DrawClock) -> f64 {
    match clock {
        DrawClock::Rust(i) => i.elapsed().as_secs_f64(),
        DrawClock::Py(t0) => perf_counter(py) - t0,
    }
}

/// Toggle Python-clock draw timing (called by core.py around statistics collection).
#[pyfunction]
#[pyo3(name = "set_use_py_clock")]
fn set_use_py_clock_py(on: bool) {
    USE_PY_CLOCK.with(|c| c.set(on));
}

/// Add `secs` to the current example's draw-time tally (called by DataObject.draw for
/// interactive st.data() draws inside the body, which run after run_one_inner set the
/// argument-draw time).
pub(crate) fn add_draw_secs(secs: f64) {
    LAST_DRAW_SECS.with(|c| c.set(c.get() + secs));
}

// Shrink-pass profiling for the debug-verbosity report (test_reports_passes,
// test_debug_information) and the MAX_SHRINKS stop reason (test_stops_after_x_shrinks).
#[derive(Default)]
struct ShrinkReport {
    // (pass name, total trial calls, successful shrinks) accumulated across the fixpoint.
    passes: Vec<(String, u64, u64)>,
    total_shrinks: u64,
    capped: bool,
}

type SlippedBug = ((String, i64), Vec<Py<PyAny>>, Py<PyAny>, Py<PyAny>);

thread_local! {
    // Trial-replay counter, incremented by trial_fails; used to attribute calls to passes.
    static SHRINK_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static SHRINK_REPORT: std::cell::RefCell<ShrinkReport> =
        std::cell::RefCell::new(ShrinkReport::default());
    // "Slippage": while shrinking one bug, a trial sometimes reproduces a DIFFERENT
    // interesting origin. With report_multiple_bugs we collect those distinct failures here
    // (first occurrence per origin) and shrink+report them too (test_slippage multi-bug).
    static SLIP_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static SLIPPED: std::cell::RefCell<Vec<SlippedBug>> = const { std::cell::RefCell::new(Vec::new()) };
    // The primary failure's minimal choice sequence, so core.py can build the
    // @reproduce_failure(version, blob) note when print_blob is set (test_prints_reproduction).
    static MINIMAL_CHOICES: std::cell::RefCell<Vec<Py<PyAny>>> =
        const { std::cell::RefCell::new(Vec::new()) };
    // PUBLISHED copies of the two thread-locals above that core.py reads AFTER run_native
    // returns (shrink_report() / minimal_choices()). run_native's exit guard moves the working
    // values here and restores the working ones to their pre-run state — so a NESTED run_native
    // (re-entrancy) gets its own published result without clobbering the outer's.
    static SHRINK_REPORT_RESULT: std::cell::RefCell<ShrinkReport> =
        std::cell::RefCell::new(ShrinkReport::default());
    static MINIMAL_CHOICES_RESULT: std::cell::RefCell<Vec<Py<PyAny>>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Save/restore-and-publish guard for run_native's thread-locals, making re-entrancy safe: a
/// NESTED run_native (an inner @given/find/minimal inside an outer run's body) snapshots the
/// outer's run-scoped state at entry and, on Drop (every exit path, return OR `?`-error),
/// PUBLISHES its own SHRINK_REPORT/MINIMAL_CHOICES to the *_RESULT thread-locals (which the
/// post-run pyfunctions read) and RESTORES the working thread-locals to the outer's values.
/// The other run-scoped locals are never read post-run, so they're simply restored.
struct RunNativeGuard {
    slip_enabled: bool,
    slipped: Vec<SlippedBug>,
    last_draw_secs: f64,
    shrink_calls: u64,
    use_py_clock: bool,
    inject_candidates: Vec<num_bigint::BigInt>,
    shrink_report: ShrinkReport,
    minimal_choices: Vec<Py<PyAny>>,
    deferred_cache: std::collections::HashMap<usize, Py<PyAny>>,
    builds_infer_cache: std::collections::HashMap<String, Vec<(String, Py<PyAny>)>>,
    builds_strategy_cache: std::collections::HashMap<String, Py<PyAny>>,
}

impl RunNativeGuard {
    fn snapshot() -> Self {
        RunNativeGuard {
            slip_enabled: SLIP_ENABLED.with(|c| c.get()),
            slipped: SLIPPED.with(|s| std::mem::take(&mut *s.borrow_mut())),
            last_draw_secs: LAST_DRAW_SECS.with(|c| c.get()),
            shrink_calls: SHRINK_CALLS.with(|c| c.get()),
            use_py_clock: USE_PY_CLOCK.with(|c| c.get()),
            inject_candidates: crate::data::take_inject_candidates(),
            shrink_report: SHRINK_REPORT.with(|r| std::mem::take(&mut *r.borrow_mut())),
            minimal_choices: MINIMAL_CHOICES.with(|c| std::mem::take(&mut *c.borrow_mut())),
            deferred_cache: crate::data::take_deferred_cache(),
            builds_infer_cache: crate::data::take_builds_infer_cache(),
            builds_strategy_cache: crate::data::take_builds_strategy_cache(),
        }
    }
}

impl Drop for RunNativeGuard {
    fn drop(&mut self) {
        // Publish this run's result for core.py's post-run shrink_report()/minimal_choices().
        let report = SHRINK_REPORT.with(|r| std::mem::take(&mut *r.borrow_mut()));
        SHRINK_REPORT_RESULT.with(|r| *r.borrow_mut() = report);
        let choices = MINIMAL_CHOICES.with(|c| std::mem::take(&mut *c.borrow_mut()));
        MINIMAL_CHOICES_RESULT.with(|c| *c.borrow_mut() = choices);
        // Restore the working thread-locals to the (outer) caller's pre-run state.
        SHRINK_REPORT.with(|r| *r.borrow_mut() = std::mem::take(&mut self.shrink_report));
        MINIMAL_CHOICES.with(|c| *c.borrow_mut() = std::mem::take(&mut self.minimal_choices));
        SLIP_ENABLED.with(|c| c.set(self.slip_enabled));
        SLIPPED.with(|s| *s.borrow_mut() = std::mem::take(&mut self.slipped));
        LAST_DRAW_SECS.with(|c| c.set(self.last_draw_secs));
        SHRINK_CALLS.with(|c| c.set(self.shrink_calls));
        USE_PY_CLOCK.with(|c| c.set(self.use_py_clock));
        crate::data::set_inject_candidates(std::mem::take(&mut self.inject_candidates));
        crate::data::set_deferred_cache(std::mem::take(&mut self.deferred_cache));
        crate::data::set_builds_infer_cache(std::mem::take(&mut self.builds_infer_cache));
        crate::data::set_builds_strategy_cache(std::mem::take(&mut self.builds_strategy_cache));
    }
}

/// Expose the most recent run's primary minimal choice sequence (for @reproduce_failure).
#[pyfunction]
#[pyo3(name = "minimal_choices")]
fn minimal_choices_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // Read the PUBLISHED result (set by run_native's exit guard), so a nested run can't have
    // clobbered the value the outer run is about to read.
    MINIMAL_CHOICES_RESULT.with(|c| {
        let lst = PyList::new(py, c.borrow().iter().map(|p| p.bind(py)))?;
        Ok(lst.into_any().unbind())
    })
}

/// Record a slipped (different-origin) failure seen during shrinking, deduped by origin.
fn record_slip(
    py: Python<'_>,
    origin: (String, i64),
    choices: &[Py<PyAny>],
    args: Py<PyAny>,
    exc: Py<PyAny>,
) {
    SLIPPED.with(|s| {
        let mut s = s.borrow_mut();
        if !s.iter().any(|b| b.0 == origin) {
            s.push((origin, choices.iter().map(|p| p.clone_ref(py)).collect(), args, exc));
        }
    });
}

/// Attribute `calls` trial-replays (and a success flag) to a named shrink pass, summing
/// across the fixpoint loop. Bumps the global successful-shrink count on a win.
fn record_pass(name: &str, calls: u64, shrank: bool) {
    SHRINK_REPORT.with(|r| {
        let mut r = r.borrow_mut();
        if shrank {
            r.total_shrinks += 1;
        }
        if let Some(e) = r.passes.iter_mut().find(|e| e.0 == name) {
            e.1 += calls;
            if shrank {
                e.2 += 1;
            }
        } else {
            r.passes.push((name.to_string(), calls, u64::from(shrank)));
        }
    });
}

enum Outcome {
    Valid,
    Invalid,
    Interesting(Py<PyAny>, Py<PyAny>), // (args tuple, exception instance)
    // The drawn example duplicates a choice sequence already explored this run; the test
    // body was NOT executed (novel-prefix generation). Only produced during generation.
    Redundant,
}

/// A pruned DataTree used only to decide, when zero valid examples are found, whether the
/// choice space was fully EXHAUSTED (every reachable finite branch explored ⇒ Unsatisfiable)
/// versus merely heavily filtered (filter_too_much). It does NOT influence generation, so
/// its blast radius is just the 0-valid error type.
#[derive(Default)]
struct TreeNode {
    children: std::collections::HashMap<String, TreeNode>,
    // possible distinct values for the choice made AT this node (None ⇒ effectively
    // infinite — float/string/unbounded-int — so this node can never be exhausted).
    max_children: Option<u64>,
    terminal: bool,
}

impl TreeNode {
    fn add(&mut self, seq: &[(String, Option<u64>)]) {
        match seq.split_first() {
            None => self.terminal = true,
            Some(((key, maxc), rest)) => {
                if self.max_children.is_none() {
                    self.max_children = *maxc;
                }
                self.children.entry(key.clone()).or_default().add(rest);
            }
        }
    }

    /// True if `seq` is already a complete (terminal) path in this tree — i.e. an identical
    /// example was already generated, so re-running it would be redundant.
    fn has_terminal(&self, seq: &[(String, Option<u64>)]) -> bool {
        match seq.split_first() {
            None => self.terminal,
            Some(((key, _), rest)) => {
                self.children.get(key).is_some_and(|c| c.has_terminal(rest))
            }
        }
    }

    fn exhausted(&self) -> bool {
        if self.children.is_empty() {
            return self.terminal;
        }
        match self.max_children {
            Some(m) if self.children.len() as u64 >= m => {
                self.children.values().all(|c| c.exhausted())
            }
            _ => false,
        }
    }
}

/// Possible distinct values for a choice of the given type/constraints — only small finite
/// integer ranges and booleans count; everything else is treated as infinite (None).
fn choice_max_children(ctype: &str, cdict: &Bound<'_, PyDict>) -> Option<u64> {
    match ctype {
        "boolean" => Some(2),
        "integer" => {
            let mn = cdict.get_item("min_value").ok().flatten();
            let mx = cdict.get_item("max_value").ok().flatten();
            if let (Some(a), Some(b)) = (mn, mx) {
                if let (Ok(a), Ok(b)) = (a.extract::<i128>(), b.extract::<i128>()) {
                    let n = b - a + 1;
                    if (1..=256).contains(&n) {
                        return Some(n as u64);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// The choice sequence of `cd` as (value-repr, max-children) pairs for DataTree insertion.
fn node_seq(_py: Python<'_>, cd: &Bound<'_, ConjectureData>) -> PyResult<Vec<(String, Option<u64>)>> {
    let nodes = cd.getattr("nodes")?;
    let mut seq = Vec::with_capacity(nodes.len()?);
    for i in 0..nodes.len()? {
        let n = nodes.get_item(i)?;
        let key = n.getattr("value")?.repr()?.to_string();
        let ctype: String = n.getattr("type")?.extract()?;
        let cdict = n.getattr("constraints")?;
        let maxc = choice_max_children(&ctype, cdict.downcast::<PyDict>()?);
        seq.push((key, maxc));
    }
    Ok(seq)
}

/// Record `cd`'s choice sequence into the all-examples tree for novel-prefix exhaustion.
/// If the sequence contains an infinite-cardinality node the space can never be exhausted,
/// so tracking is switched off (callers then skip the per-example node_seq cost).
// Above this many choices an example is far too large for a finite space to be exhausted,
// and node_seq (O(nodes), repr per node) would be ruinously slow — e.g. a list of 100_000
// elements. Stop tracking such examples entirely (treat the space as effectively infinite).
const MAX_TRACKED_NODES: usize = 256;

// Total node budget for the all-examples tree across a whole run. A huge generation run
// over a small-but-not-tiny finite space (e.g. `find_any(dates(), …, max_examples=10**6)`
// hunting a rare condition) would otherwise accumulate millions of small examples into the
// tree → multiple GB → the worker is OOM-killed. Past this many nodes, stop tracking: the
// tree is only an optimisation (novel-prefix generation + exhaustion detection) and a space
// large enough to blow this budget won't be exhausted anyway. ~500k nodes ≈ well under
// 100 MB, while remaining ample for the small finite spaces novel-prefix actually helps.
const MAX_TOTAL_TRACKED_NODES: usize = 500_000;

fn track_example(
    py: Python<'_>,
    track: &mut bool,
    tree: &mut TreeNode,
    tracked_nodes: &mut usize,
    cd: &Bound<'_, ConjectureData>,
) {
    if !*track {
        return;
    }
    // Bail out BEFORE node_seq on huge examples — its per-node repr is too expensive and a
    // space that large can never be exhausted anyway (guards against generation hangs).
    if cd.getattr("nodes").and_then(|n| n.len()).map(|n| n > MAX_TRACKED_NODES).unwrap_or(true) {
        *track = false;
        return;
    }
    if let Ok(seq) = node_seq(py, cd) {
        if seq.iter().any(|(_, m)| m.is_none()) {
            *track = false;
        } else {
            tree.add(&seq);
            *tracked_nodes += seq.len();
            // Disable tracking once the cumulative tree size exceeds the budget (OOM guard).
            if *tracked_nodes > MAX_TOTAL_TRACKED_NODES {
                *track = false;
            }
        }
    }
}

fn err_name(py: Python<'_>, e: &PyErr) -> String {
    e.get_type(py)
        .name()
        .ok()
        .map(|n| n.to_string())
        .unwrap_or_default()
}

fn run_one(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    cd: &Bound<'_, ConjectureData>,
) -> PyResult<Outcome> {
    run_one_named(py, test, strategies, cd, None, None)
}

/// Run `f` (one example execution) wrapped in the test's executor protocol: call the
/// executor's `setup_example()` before and `teardown_example(token)` after (always, even
/// on error), matching hypothesis's new_style_executor. Plain tests (no executor) just run.
fn with_executor<F>(
    py: Python<'_>,
    executor: Option<&Bound<'_, PyAny>>,
    f: F,
) -> PyResult<Outcome>
where
    F: FnOnce() -> PyResult<Outcome>,
{
    let Some(ex) = executor else {
        return f();
    };
    let token = match ex.getattr("setup_example") {
        Ok(setup) => setup.call0()?.unbind(),
        Err(_) => py.None(),
    };
    let result = f();
    if let Ok(teardown) = ex.getattr("teardown_example") {
        let _ = teardown.call1((token,));
    }
    result
}

/// As `run_one`, but `names` (when present, one per strategy) lets a draw-time error
/// be re-raised with hypothesis's "while generating '{name}' from {strategy!r}" note —
/// used by the generation phase so an exception while building an argument is reported
/// against the parameter it was generating.
fn run_one_named(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    cd: &Bound<'_, ConjectureData>,
    names: Option<&[String]>,
    redundancy: Option<&TreeNode>,
) -> PyResult<Outcome> {
    let outcome = run_one_inner(py, test, strategies, cd, names, redundancy);
    // Freeze the example's data so any callable that captured it (e.g. a
    // functions()-generated function) raises InvalidState if called afterwards.
    let _ = cd.call_method0("freeze");
    outcome
}

fn run_one_inner(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    cd: &Bound<'_, ConjectureData>,
    names: Option<&[String]>,
    // When set (generation only), an already-explored choice sequence is skipped as
    // Redundant WITHOUT executing the test body — novel-prefix generation.
    redundancy: Option<&TreeNode>,
) -> PyResult<Outcome> {
    let mut args: Vec<Bound<'_, PyAny>> = Vec::with_capacity(strategies.len());
    let draw_clock = draw_clock_start(py);
    for (i, s) in strategies.iter().enumerate() {
        match cd.call_method1("draw", (s.bind(py),)) {
            Ok(v) => args.push(v),
            Err(e) => {
                LAST_DRAW_SECS.with(|c| c.set(draw_clock_elapsed(py, &draw_clock)));
                // assume() raised inside a draw-time callback (e.g. integers().map(
                // lambda x: assume(cond) and x)) rejects the example, exactly like a
                // filter — not a test error. StopTest is the engine's own overrun signal.
                let n = err_name(py, &e);
                if n == "StopTest" || n == "UnsatisfiedAssumption" {
                    return Ok(Outcome::Invalid);
                }
                // A genuine exception while drawing argument `i` — annotate it with the
                // strategy it was generated from (matches hypothesis's draw-error note).
                if let Some(names) = names {
                    if let Some(name) = names.get(i) {
                        let note = s
                            .bind(py)
                            .repr()
                            .map(|r| format!("while generating '{name}' from {r}"))
                            .unwrap_or_else(|_| format!("while generating '{name}'"));
                        let _ = e.value(py).call_method1("add_note", (note,));
                    }
                }
                return Err(e);
            }
        }
    }
    LAST_DRAW_SECS.with(|c| c.set(draw_clock_elapsed(py, &draw_clock)));
    // Novel-prefix generation: if this exact choice sequence was already explored, skip it
    // (don't run the test) — so a finite strategy like booleans() yields each value once.
    if let Some(tree) = redundancy {
        if let Ok(seq) = node_seq(py, cd) {
            if tree.has_terminal(&seq) {
                return Ok(Outcome::Redundant);
            }
        }
    }
    let tup = PyTuple::new(py, &args)?;
    match test.bind(py).call1(&tup) {
        Ok(_) => Ok(Outcome::Valid),
        Err(e) => {
            let n = err_name(py, &e);
            if n == "UnsatisfiedAssumption" {
                Ok(Outcome::Invalid)
            } else if n == "StopTest" {
                // A StopTest tagged with THIS example's testcounter is the engine's own
                // end-of-example signal -> the example simply ends. Its outcome follows the
                // data's status: a body that froze still-valid data ends VALID, while an
                // overrun / mark_invalid ends INVALID. A StopTest carrying a different
                // counter is a user-raised control exception (test_exceptiongroup
                // multiple_stoptest_2) that must propagate out unchanged.
                if stoptest_matches(py, &e, cd) {
                    if cd.borrow().status_is_valid() {
                        Ok(Outcome::Valid)
                    } else {
                        Ok(Outcome::Invalid)
                    }
                } else {
                    Err(e)
                }
            } else if is_skip_exception(py, &e) || is_fatal_base_exception(py, &e) {
                // pytest.skip()/unittest.SkipTest, KeyboardInterrupt and SystemExit
                // propagate immediately — they're not test failures to shrink
                // (test_pytest_skip_skips_shrinking, test_does_not_catch_interrupt_during_falsify).
                Err(e)
            } else {
                let exc = e.value(py).clone().into_any().unbind();
                Ok(Outcome::Interesting(tup.into_any().unbind(), exc))
            }
        }
    }
}

/// True if `e` is a StopTest whose `.testcounter` is this data's (the engine's own
/// end-of-example signal). Defaults to true when either counter can't be read, so the
/// common engine-internal StopTest (overrun/mark_*) is always treated as a discard.
fn stoptest_matches(py: Python<'_>, e: &PyErr, cd: &Bound<'_, ConjectureData>) -> bool {
    let etc = e.value(py).getattr("testcounter").and_then(|t| t.extract::<i64>()).ok();
    let ctc = cd.getattr("testcounter").and_then(|t| t.extract::<i64>()).ok();
    match (etc, ctc) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

/// True if `e` is a KeyboardInterrupt or SystemExit — a fatal control exception that must
/// stop the engine and propagate, never be treated as a (shrinkable) test failure.
fn is_fatal_base_exception(py: Python<'_>, e: &PyErr) -> bool {
    let v = e.value(py);
    v.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>()
        || v.is_instance_of::<pyo3::exceptions::PySystemExit>()
}

/// True if `e` is a skip signal (pytest.skip's Skipped or unittest.SkipTest).
fn is_skip_exception(py: Python<'_>, e: &PyErr) -> bool {
    if err_name(py, e) == "Skipped" {
        return true;
    }
    py.import("unittest")
        .and_then(|m| m.getattr("SkipTest"))
        .and_then(|st| e.value(py).is_instance(&st))
        .unwrap_or(false)
}

fn extract_choices(_py: Python<'_>, cd: &Bound<'_, ConjectureData>) -> PyResult<Vec<Py<PyAny>>> {
    let choices = cd.getattr("choices")?;
    let mut out = Vec::new();
    for it in choices.try_iter()? {
        out.push(it?.unbind());
    }
    Ok(out)
}

fn clone_choices(py: Python<'_>, choices: &[Py<PyAny>]) -> Vec<Py<PyAny>> {
    choices.iter().map(|p| p.clone_ref(py)).collect()
}

fn trial_fails(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    choices: &[Py<PyAny>],
    origin: &(String, i64),
) -> PyResult<bool> {
    SHRINK_CALLS.with(|c| c.set(c.get() + 1));
    let cd = Py::new(py, ConjectureData::new_for_choices(py, choices))?;
    match run_one(py, test, strategies, cd.bind(py))? {
        // Origin-preserving: a shrink only counts if it reproduces the SAME interesting
        // origin (type + location). Otherwise shrinking could "slip" one bug into a
        // different one (test_slippage.test_raises_multiple_failures_with_varying_type).
        Outcome::Interesting(args, exc) => {
            let o = origin_of(py, &exc)?;
            if &o == origin {
                Ok(true)
            } else {
                // Slipped to a DIFFERENT bug — record it (with report_multiple_bugs) so it's
                // shrunk and reported alongside the primary failure.
                if SLIP_ENABLED.with(|c| c.get()) {
                    record_slip(py, o, choices, args, exc);
                }
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

/// Build the choice list with node `i` replaced by the value at complexity `index`,
/// and report whether the test still fails. Returns the trial on success.
#[allow(clippy::too_many_arguments)]
fn try_node_index(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    canonical: &[Py<PyAny>],
    i: usize,
    index: &BigInt,
    ctype: &str,
    cdict: &Bound<'_, PyDict>,
    origin: &(String, i64),
) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let cand_val = crate::choice::choice_from_index(py, index.clone(), ctype, cdict)?;
    let mut trial = clone_choices(py, canonical);
    trial[i] = cand_val.into_bound(py).unbind();
    if trial_fails(py, test, strategies, &trial, origin)? {
        Ok(Some(trial))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn set_node_value(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    canonical: &[Py<PyAny>],
    i: usize,
    val: Bound<'_, PyAny>,
    origin: &(String, i64),
) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let mut trial = clone_choices(py, canonical);
    trial[i] = val.unbind();
    if trial_fails(py, test, strategies, &trial, origin)? {
        Ok(Some(trial))
    } else {
        Ok(None)
    }
}

/// Structural minimization of a string choice (atomic, but with a giant complexity
/// index — so we shrink char-level): delete a char, or lower a char toward the
/// alphabet's simplest (binary search in char_in_shrink_order space). Returns the
/// improved choice list on the first improvement.
#[allow(clippy::too_many_arguments)]
fn minimize_string_node(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    canonical: &[Py<PyAny>],
    i: usize,
    value: &Bound<'_, PyAny>,
    intervals: &Bound<'_, PyAny>,
    origin: &(String, i64),
) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let builtins = py.import("builtins")?;
    let chars = builtins.getattr("list")?.call1((value,))?;
    let n = chars.len()?;
    let empty = PyString::new(py, "");

    // delete pass: remove each char.
    for j in 0..n {
        let parts = PyList::empty(py);
        for k in 0..n {
            if k != j {
                parts.append(chars.get_item(k)?)?;
            }
        }
        let cand = empty.call_method1("join", (&parts,))?;
        if let Some(t) = set_node_value(py, test, strategies, canonical, i, cand, origin)? {
            return Ok(Some(t));
        }
    }

    // per-char minimization. The find-predicate is NOT monotonic in shrink-order index
    // when a cross-char constraint is ordinal (e.g. x[0] < x[1]: '0'.shrink_idx==0 fails
    // but '1'.shrink_idx==1 reproduces), so a pure binary search can overshoot to a far
    // larger char. Do a LINEAR scan over the first CAP shrink-order indices (cheap; finds
    // the true smallest reproducing char regardless of monotonicity), then fall back to a
    // binary search beyond the window for large alphabets.
    const LINEAR_CAP: i64 = 24;
    for j in 0..n {
        let ch = chars.get_item(j)?;
        let cur_idx: i64 = intervals
            .call_method1("index_from_char_in_shrink_order", (&ch,))?
            .extract()?;
        if cur_idx == 0 {
            continue;
        }
        let try_idx = |idx: i64| -> PyResult<Option<Vec<Py<PyAny>>>> {
            let newch = intervals.call_method1("char_in_shrink_order", (idx,))?;
            let parts = PyList::empty(py);
            for k in 0..n {
                if k == j {
                    parts.append(&newch)?;
                } else {
                    parts.append(chars.get_item(k)?)?;
                }
            }
            let cand = empty.call_method1("join", (&parts,))?;
            set_node_value(py, test, strategies, canonical, i, cand, origin)
        };
        // linear scan of the low window: the first reproducing index is the true minimum
        // (no monotonicity assumption).
        let window = cur_idx.min(LINEAR_CAP);
        for idx in 0..window {
            if let Some(t) = try_idx(idx)? {
                return Ok(Some(t));
            }
        }
        // beyond the window, binary-search for a smaller reproducing index (monotonic case).
        if cur_idx > LINEAR_CAP {
            let mut lo = LINEAR_CAP - 1; // last index known NOT to reproduce
            let mut hi = cur_idx;
            while hi - lo > 1 {
                let mid = lo + (hi - lo) / 2;
                if try_idx(mid)?.is_some() {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            if hi < cur_idx {
                if let Some(t) = try_idx(hi)? {
                    return Ok(Some(t));
                }
            }
        }
    }
    Ok(None)
}

/// Adaptive deletion: try removing blocks of choices (size 8,4,2,1), keeping any
/// removal that preserves the failure. Shrinks collection length.
fn delete_pass(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    choices: Vec<Py<PyAny>>,
    origin: &(String, i64),
) -> PyResult<(Vec<Py<PyAny>>, bool)> {
    let mut cur = choices;
    let mut improved = false;
    for &size in &[8usize, 4, 3, 2, 1] {
        let mut start = 0;
        while start + size <= cur.len() {
            let mut trial: Vec<Py<PyAny>> = Vec::with_capacity(cur.len() - size);
            for (j, p) in cur.iter().enumerate() {
                if j < start || j >= start + size {
                    trial.push(p.clone_ref(py));
                }
            }
            if trial_fails(py, test, strategies, &trial, origin)? {
                cur = trial;
                improved = true;
                // don't advance start; the next block shifted into place
            } else {
                start += 1;
            }
        }
    }
    Ok((cur, improved))
}

/// Lower every group of choices that currently share a value TOGETHER. Catches coupled
/// failures where lowering either member alone breaks the bug but lowering both to a
/// smaller common value preserves it (e.g. a test that fails only when `x == y`).
/// Intrinsically reducing: only sets the group to a strictly smaller complexity index.
fn minimize_duplicated_choices(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    canonical: &[Py<PyAny>],
    nodes: &Bound<'_, PyAny>,
    origin: &(String, i64),
) -> PyResult<Option<Vec<Py<PyAny>>>> {
    let nlen = nodes.len()?;
    // O(n^2) grouping — skip very long choice sequences (huge collections); the per-node
    // and deletion passes already handle those, and the win here is for small coupled sets.
    if nlen > 256 {
        return Ok(None);
    }
    let mut handled = vec![false; nlen];
    for a in 0..nlen {
        if a >= canonical.len() || handled[a] {
            continue;
        }
        let node_a = nodes.get_item(a)?;
        let ctype: String = node_a.getattr("type")?.extract()?;
        // strings carry a giant per-char complexity index — handled by minimize_string_node.
        if ctype == "string" {
            continue;
        }
        let val_a = node_a.getattr("value")?;
        let cobj = node_a.getattr("constraints")?;
        let cdict = cobj.downcast::<PyDict>()?;
        let idx_a = crate::choice::choice_to_index(&val_a, cdict)?;
        if idx_a.is_zero() {
            continue;
        }
        // gather later nodes with an equal value
        let mut group = vec![a];
        for b in (a + 1)..nlen {
            if b >= canonical.len() || handled[b] {
                continue;
            }
            let val_b = nodes.get_item(b)?.getattr("value")?;
            if crate::choice::choice_equal(py, &val_a, &val_b)? {
                group.push(b);
            }
        }
        if group.len() < 2 {
            continue;
        }
        for &g in &group {
            handled[g] = true;
        }
        let try_common = |cand: &BigInt| -> PyResult<Option<Vec<Py<PyAny>>>> {
            let cv = crate::choice::choice_from_index(py, cand.clone(), &ctype, cdict)?;
            let cvb = cv.into_bound(py);
            let mut trial = clone_choices(py, canonical);
            for &g in &group {
                trial[g] = cvb.clone().unbind();
            }
            if trial_fails(py, test, strategies, &trial, origin)? {
                Ok(Some(trial))
            } else {
                Ok(None)
            }
        };
        if let Some(t) = try_common(&BigInt::zero())? {
            return Ok(Some(t));
        }
        let mut lo = BigInt::zero();
        let mut hi = idx_a.clone();
        while &hi - &lo > BigInt::one() {
            let mid = (&lo + &hi) / 2;
            if try_common(&mid)?.is_some() {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        if hi < idx_a {
            if let Some(t) = try_common(&hi)? {
                return Ok(Some(t));
            }
        }
    }
    Ok(None)
}

/// For scalar (int/float/boolean) pairs within a sliding window, try lowering BOTH toward
/// their simplest value simultaneously. Catches coupled failures where neither member can
/// be lowered alone (per-node passes leave them stuck) but both can drop together. Bounded
/// to a small window so the cost stays roughly linear in the node count.
fn minimize_pairs_to_zero(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    canonical: &[Py<PyAny>],
    nodes: &Bound<'_, PyAny>,
    origin: &(String, i64),
) -> PyResult<Option<Vec<Py<PyAny>>>> {
    const WINDOW: usize = 8;
    let nlen = nodes.len()?;
    if nlen > 256 {
        return Ok(None);
    }
    let is_scalar = |ct: &str| matches!(ct, "integer" | "float" | "boolean");
    for i in 0..nlen {
        if i >= canonical.len() {
            continue;
        }
        let node_i = nodes.get_item(i)?;
        let cti: String = node_i.getattr("type")?.extract()?;
        if !is_scalar(&cti) {
            continue;
        }
        let ci = node_i.getattr("constraints")?;
        let cdi = ci.downcast::<PyDict>()?;
        let idx_i = crate::choice::choice_to_index(&node_i.getattr("value")?, cdi)?;
        if idx_i.is_zero() {
            continue;
        }
        let zero_i = crate::choice::choice_from_index(py, BigInt::zero(), &cti, cdi)?.into_bound(py);
        for j in (i + 1)..(i + 1 + WINDOW).min(nlen) {
            if j >= canonical.len() {
                continue;
            }
            let node_j = nodes.get_item(j)?;
            let ctj: String = node_j.getattr("type")?.extract()?;
            if !is_scalar(&ctj) {
                continue;
            }
            let cj = node_j.getattr("constraints")?;
            let cdj = cj.downcast::<PyDict>()?;
            let idx_j = crate::choice::choice_to_index(&node_j.getattr("value")?, cdj)?;
            if idx_j.is_zero() {
                continue;
            }
            let zero_j =
                crate::choice::choice_from_index(py, BigInt::zero(), &ctj, cdj)?.into_bound(py);
            let mut trial = clone_choices(py, canonical);
            trial[i] = zero_i.clone().unbind();
            trial[j] = zero_j.unbind();
            if trial_fails(py, test, strategies, &trial, origin)? {
                return Ok(Some(trial));
            }
        }
    }
    Ok(None)
}

/// Shrink a failure, then confirm the minimal example still reproduces the same origin. A
/// divergent replay means the failure was flaky (it failed once but not on re-execution),
/// so report it as a FlakyFailure wrapping the original. Returns (args, report_exc, schoices).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn shrink_confirmed(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    origin: &(String, i64),
    choices: Vec<Py<PyAny>>,
    args: Py<PyAny>,
    exc: Py<PyAny>,
    shrink_cap: u64,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Vec<Py<PyAny>>)> {
    let (sargs, sexc, schoices) = shrink(py, test, strategies, choices, args, exc, shrink_cap)?;
    let cd = Py::new(py, ConjectureData::new_for_choices(py, &schoices))?;
    let reproduces = match run_one(py, test, strategies, cd.bind(py))? {
        Outcome::Interesting(_, e2) => &origin_of(py, &e2)? == origin,
        _ => false,
    };
    let report_exc = if reproduces {
        sexc
    } else {
        flaky_failure(
            py,
            "Falsified on the first call but did not on a subsequent one",
            &[sexc],
        )
        .value(py)
        .clone()
        .into_any()
        .unbind()
    };
    Ok((sargs, report_exc, schoices))
}

/// Drain the slipped (different-origin) failures collected during shrinking, shrink each
/// (with slip-collection disabled so we don't recurse forever), persist them, and return
/// (args, exc) per NEW distinct origin not already in `existing`. Used to surface multi-bug
/// slippage discovered while minimizing the primary failure.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn shrink_slipped(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    existing: &[(String, i64)],
    shrink_cap: u64,
    database: Option<&Py<PyAny>>,
    db_key: Option<&Vec<u8>>,
) -> PyResult<Vec<(Py<PyAny>, Py<PyAny>, Vec<Py<PyAny>>)>> {
    let prev = SLIP_ENABLED.with(|c| c.replace(false));
    let drained: Vec<SlippedBug> = SLIPPED.with(|s| std::mem::take(&mut *s.borrow_mut()));
    let mut seen: Vec<(String, i64)> = existing.to_vec();
    let mut out = Vec::new();
    for (o, choices, args, exc) in drained {
        if seen.iter().any(|e| e == &o) {
            continue;
        }
        seen.push(o.clone());
        let (sargs, report_exc, schoices) =
            shrink_confirmed(py, test, strategies, &o, choices, args, exc, shrink_cap)?;
        if let (Some(db), Some(key)) = (database, db_key) {
            if let Ok(blob) = crate::database::encode_choices(py, &schoices) {
                let dbb = db.bind(py);
                let _ = dbb.call_method1(
                    "save",
                    (PyBytes::new(py, key), PyBytes::new(py, &blob)),
                );
            }
        }
        out.push((sargs, report_exc, schoices));
    }
    SLIP_ENABLED.with(|c| c.set(prev));
    Ok(out)
}

/// sort_key ordering on choice sequences: shorter first, then value-key lexicographic.
/// Used to pick the single bug to report when report_multiple_bugs is off (upstream
/// reports the globally-minimal failure across distinct origins).
fn choices_lt(py: Python<'_>, a: &[Py<PyAny>], b: &[Py<PyAny>]) -> PyResult<bool> {
    if a.len() != b.len() {
        return Ok(a.len() < b.len());
    }
    let la = PyList::new(py, a)?;
    let lb = PyList::new(py, b)?;
    let ka = crate::choice::choices_key(py, la.as_any())?;
    let kb = crate::choice::choices_key(py, lb.as_any())?;
    ka.bind(py).lt(kb.bind(py))
}

#[allow(clippy::type_complexity)]
fn shrink(
    py: Python<'_>,
    test: &Py<PyAny>,
    strategies: &[Py<PyAny>],
    mut choices: Vec<Py<PyAny>>,
    mut best_args: Py<PyAny>,
    mut best_exc: Py<PyAny>,
    shrink_cap: u64,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Vec<Py<PyAny>>)> {
    // Shrinks must preserve the bug's interesting origin (type + location).
    let target_origin = origin_of(py, &best_exc)?;
    SHRINK_CALLS.with(|c| c.set(0));
    SHRINK_REPORT.with(|r| *r.borrow_mut() = ShrinkReport::default());
    let mut passes = 0;
    // Deletion (PASS 1) is O(len) trials and dominates the cost on large value-heavy
    // examples (e.g. a 64-int matrix) where it never finds anything — only deletion changes
    // the length, per-node lowering doesn't. So run it once per distinct length, then skip it
    // while values are lowered, and re-run it ONCE more after the value passes reach a
    // fixpoint (a now-minimal value can occasionally unlock a deletion). `last_delete_len` is
    // the length at the last deletion attempt; `retry_delete_after_values` forces that final
    // recheck. (test_can_shrink_matrices_with_length_param: ~20k trials → ~700.)
    let mut last_delete_len: Option<usize> = None;
    let mut retry_delete_after_values = false;
    loop {
        passes += 1;
        if passes > 300 {
            break;
        }
        // Stop once MAX_SHRINKS successful shrinks have been made (upstream's cap; with
        // cap==0 we don't shrink at all). Records the "shrunk example N times" stop reason
        // — including cap==0, which test_stops_after_x_shrinks relies on (MAX_SHRINKS=0 ⇒
        // "shrunk example 0 times").
        if SHRINK_REPORT.with(|r| r.borrow().total_shrinks) >= shrink_cap {
            SHRINK_REPORT.with(|r| r.borrow_mut().capped = true);
            break;
        }
        // replay current choices to recapture args/exc/nodes
        let cd = Py::new(py, ConjectureData::new_for_choices(py, &choices))?;
        let cdb = cd.bind(py);
        match run_one(py, test, strategies, cdb)? {
            Outcome::Interesting(a, e) => {
                best_args = a;
                best_exc = e;
            }
            _ => break,
        }
        let canonical = extract_choices(py, cdb)?;
        let cur_len = canonical.len();

        // PASS 1: adaptive block deletion (collection length). Skip when we already deleted at
        // this length and no value pass has since settled into a fresh fixpoint to recheck.
        let run_delete = last_delete_len != Some(cur_len) || retry_delete_after_values;
        if run_delete {
            retry_delete_after_values = false;
            last_delete_len = Some(cur_len);
            let c0 = SHRINK_CALLS.with(|c| c.get());
            let (deleted, del_improved) =
                delete_pass(py, test, strategies, clone_choices(py, &canonical), &target_origin)?;
            record_pass(
                "adaptive_example_deletion",
                SHRINK_CALLS.with(|c| c.get()) - c0,
                del_improved,
            );
            if del_improved {
                choices = deleted;
                continue;
            }
        }

        // PASS 2: per-node value lowering ("minimize_individual_choices")
        let nodes = cdb.getattr("nodes")?;
        let nlen = nodes.len()?;
        let mut improved = false;
        let node_c0 = SHRINK_CALLS.with(|c| c.get());
        for i in 0..nlen {
            if i >= canonical.len() {
                continue;
            }
            let node = nodes.get_item(i)?;
            // A forced choice (e.g. the min_size `more` booleans of a collection) can't be
            // shrunk — the draw forces it back regardless of the choice value, so "lowering"
            // it is a no-op that `try` accepts, re-extracts forced, and loops on forever,
            // burning the whole shrink budget before reaching the real element choices (which
            // is exactly why uniqueness-constrained collections of min_size>=2 never minimised).
            if node.getattr("was_forced")?.extract::<bool>()? {
                continue;
            }
            let value = node.getattr("value")?;
            let ctype: String = node.getattr("type")?.extract()?;
            let constraints = node.getattr("constraints")?;
            let cdict = constraints.downcast::<PyDict>()?;

            // Strings have an astronomically large complexity index (alphabet^len),
            // so binary-searching it is impractical — minimize char-by-char instead.
            if ctype == "string" {
                if let Some(intervals) = cdict.get_item("intervals")? {
                    if let Some(t) =
                        minimize_string_node(py, test, strategies, &canonical, i, &value, &intervals, &target_origin)?
                    {
                        choices = t;
                        improved = true;
                        break;
                    }
                }
                continue;
            }

            let idx = crate::choice::choice_to_index(&value, cdict)?;
            // Floats: the find-predicate is NOT monotonic in the complexity index (lex
            // ordering + bound clamping), so the binary search below can't reliably reach
            // the simplest value — most notably -0.0 for a sign-constrained find. Try the
            // canonical simplest floats explicitly: a PERMITTED candidate (so it isn't
            // clamped away) with a strictly smaller index that is still a find wins.
            if ctype == "float" {
                for cand in [0.0f64, -0.0, 1.0, -1.0] {
                    let cv = pyo3::types::PyFloat::new(py, cand).into_any();
                    if !crate::choice::choice_permitted(&cv, cdict)? {
                        continue;
                    }
                    if crate::choice::choice_to_index(&cv, cdict)? >= idx {
                        continue;
                    }
                    if let Some(t) = set_node_value(py, test, strategies, &canonical, i, cv, &target_origin)? {
                        choices = t;
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
            if idx.is_zero() {
                continue;
            }
            // Try index 0 (simplest); if that still fails, take it.
            if let Some(t) =
                try_node_index(py, test, strategies, &canonical, i, &BigInt::zero(), &ctype, cdict, &target_origin)?
            {
                choices = t;
                improved = true;
                break;
            }
            // 0 passes — binary-search the smallest complexity index in (0, idx]
            // that still fails, giving the true per-node minimum in O(log idx) tries.
            let mut lo = BigInt::zero();
            let mut hi = idx.clone();
            while &hi - &lo > BigInt::one() {
                let mid = (&lo + &hi) / 2;
                if try_node_index(py, test, strategies, &canonical, i, &mid, &ctype, cdict, &target_origin)?
                    .is_some()
                {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            if hi < idx {
                if let Some(t) =
                    try_node_index(py, test, strategies, &canonical, i, &hi, &ctype, cdict, &target_origin)?
                {
                    choices = t;
                    improved = true;
                    break;
                }
            }
        }
        record_pass(
            "minimize_individual_choices",
            SHRINK_CALLS.with(|c| c.get()) - node_c0,
            improved,
        );
        if improved {
            continue;
        }

        // PASS 3: lower duplicated choices together (coupled-equal failures).
        let c3 = SHRINK_CALLS.with(|c| c.get());
        let dup = minimize_duplicated_choices(py, test, strategies, &canonical, &nodes, &target_origin)?;
        record_pass(
            "minimize_duplicated_choices",
            SHRINK_CALLS.with(|c| c.get()) - c3,
            dup.is_some(),
        );
        if let Some(t) = dup {
            choices = t;
            continue;
        }

        // PASS 4: lower scalar pairs to their simplest together (coupled non-equal failures
        // that per-node lowering can't move).
        let c4 = SHRINK_CALLS.with(|c| c.get());
        let pair = minimize_pairs_to_zero(py, test, strategies, &canonical, &nodes, &target_origin)?;
        record_pass(
            "redistribute_numeric_pairs",
            SHRINK_CALLS.with(|c| c.get()) - c4,
            pair.is_some(),
        );
        if let Some(t) = pair {
            choices = t;
            continue;
        }

        // The value passes are at a fixpoint. If we skipped deletion this iteration (it had
        // already come up empty at this length), give it one final shot now that values are
        // minimal — then stop. `retry_delete_after_values` makes the next iteration run
        // deletion; if it too finds nothing we fall through here with run_delete=true and break.
        if !run_delete {
            retry_delete_after_values = true;
            continue;
        }

        break;
    }
    Ok((best_args, best_exc, choices))
}

/// The shrink-pass profiling report from the most recent shrink(): a tuple
/// (total_calls, total_shrinks, capped, [(pass_name, calls, shrinks), ...]). core.py
/// formats it for the debug-verbosity output and derives the "shrunk example N times"
/// statistics stop reason.
#[pyfunction]
#[pyo3(name = "shrink_report")]
fn shrink_report_py(py: Python<'_>) -> PyResult<Py<PyAny>> {
    // Read the PUBLISHED result (run_native's exit guard), not the live working report.
    SHRINK_REPORT_RESULT.with(|r| {
        let r = r.borrow();
        let total_calls: u64 = r.passes.iter().map(|p| p.1).sum();
        let passes = PyList::empty(py);
        for (name, calls, shrinks) in &r.passes {
            passes.append((name.as_str(), *calls, *shrinks))?;
        }
        Ok(PyTuple::new(py, [
            total_calls.into_pyobject(py)?.into_any(),
            r.total_shrinks.into_pyobject(py)?.into_any(),
            r.capped.into_pyobject(py)?.to_owned().into_any(),
            passes.into_any(),
        ])?
        .into_any()
        .unbind())
    })
}

/// Run `test(*args)` over examples drawn from `strategies`. Returns None if all
/// pass, else `(shrunk_args_tuple, exception_instance)`. The exception comes from
/// replaying the minimal choice sequence into a FRESH ConjectureData, so it's the
/// correct shrunk exception even for interactive data() draws.
#[pyfunction]
#[pyo3(name = "run_native", signature = (test, strategies, max_examples=100, seed=0, database=None, db_key=None, report_multiple_bugs=false, names=None, executor=None, suppress_health_check=None, invalid_cap=0, test_name=None, deadline_ms=0.0, max_shrinks=500))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_native(
    py: Python<'_>,
    test: Py<PyAny>,
    strategies: Vec<Py<PyAny>>,
    max_examples: u32,
    seed: u64,
    database: Option<Py<PyAny>>,
    db_key: Option<Vec<u8>>,
    report_multiple_bugs: bool,
    names: Option<Vec<String>>,
    executor: Option<Py<PyAny>>,
    // suppressed HealthCheck names (e.g. "filter_too_much"); when filter_too_much is
    // suppressed a fully-rejected run reports Unsatisfiable-with-counts instead of the
    // health check. invalid_cap = INVALID_THRESHOLD_BASE+1 (the point at which a
    // zero-valid run gives up); 0 means "use the default generation cap". test_name is
    // the pretty test name for the Unsatisfiable message.
    suppress_health_check: Option<Vec<String>>,
    invalid_cap: u64,
    test_name: Option<String>,
    deadline_ms: f64,
    // upstream MAX_SHRINKS (monkeypatchable): stop after this many successful shrinks and
    // report the "shrunk example N times" stop reason (test_stops_after_x_shrinks).
    max_shrinks: u64,
) -> PyResult<Option<Vec<(Py<PyAny>, Py<PyAny>)>>> {
    // Make run_native re-entrancy-safe: snapshot every run-scoped thread-local now and, on any
    // exit, publish this run's SHRINK_REPORT/MINIMAL_CHOICES for the post-run pyfunctions and
    // restore the outer caller's working state. MUST be the first statement (before the resets
    // below). A nested run_native (inner @given / find() / minimal() inside an outer run's
    // body) then can't corrupt the outer run.
    let _tls_guard = RunNativeGuard::snapshot();
    // Clear stale shrink-profiling from a previous run on this thread (the from_db replay
    // path returns without calling shrink(), which is what resets it otherwise).
    SHRINK_REPORT.with(|r| *r.borrow_mut() = ShrinkReport::default());
    // Multi-bug slippage collection: always collect (even with report_multiple_bugs off,
    // so we can report the globally-minimal failure across origins); the result paths
    // decide whether to report all of them or just the single smallest-sort_key one.
    SLIPPED.with(|s| s.borrow_mut().clear());
    SLIP_ENABLED.with(|c| c.set(true));
    MINIMAL_CHOICES.with(|c| c.borrow_mut().clear());
    let suppress = suppress_health_check.unwrap_or_default();
    let filter_suppressed = suppress.iter().any(|s| s == "filter_too_much");
    let too_slow_suppressed = suppress.iter().any(|s| s == "too_slow");
    // Allow at least 1s, or 5x the deadline (default 6s when unset), of total argument-draw
    // time across the first 10 valid examples before HealthCheck.too_slow trips.
    let deadline_secs = if deadline_ms > 0.0 { deadline_ms / 1000.0 } else { 6.0 };
    let draw_time_limit = (5.0 * deadline_secs).max(1.0);
    let mut total_draw = 0.0f64;
    // When valid==0 we give up after invalid_cap rejected examples (matching hypothesis's
    // INVALID_THRESHOLD_BASE+1), so the reported "{n} of {n}" count is exact.
    let invalid_giveup = if invalid_cap > 0 { invalid_cap } else { u64::MAX };
    let executor_b = executor.as_ref().map(|e| e.bind(py));
    type Bug = ((String, i64), Vec<Py<PyAny>>, Py<PyAny>, Py<PyAny>);
    let names_ref = names.as_deref();
    let mut valid = 0u32;
    let mut invalid = 0u64;
    // Subset of `invalid` that overran the buffer-size limit (too large to finish
    // generating) rather than failing a filter/assume — so an all-rejected run reports the
    // right reason (test_notes_high_overrun_rates_in_unsatisfiable_error).
    let mut overrun = 0u64;
    let mut iters = 0u64;
    // DataTree of invalid choice-sequences (built only while no valid example exists). If
    // it ends up fully exhausted with zero valid examples, the space is unsatisfiable.
    let mut invalid_tree = TreeNode::default();
    // DataTree of ALL examples (any status), for novel-prefix exhaustion: when the whole
    // finite choice space has been generated there's nothing new to try, so stop early
    // (e.g. booleans() yields exactly 2 examples — test_given_usable_inline_on_lambdas).
    // Tracking is abandoned the moment an infinite-cardinality node appears (the common
    // case), so it costs ~one node_seq for infinite-space tests.
    let mut all_tree = TreeNode::default();
    let mut track_tree = true;
    // Cumulative node count in `all_tree`; tracking is disabled once it exceeds the budget
    // (OOM guard for huge generation runs — see MAX_TOTAL_TRACKED_NODES).
    let mut tracked_nodes: usize = 0;
    let cap = (max_examples as u64) * 10 + 100;
    // Distinct interesting bugs, keyed (and deduped) by interesting-origin. With
    // report_multiple_bugs we keep generating to surface ALL distinct origins; otherwise
    // we stop at the first.
    let mut bugs: Vec<Bug> = Vec::new();
    // True when the (single) failing example came straight from the database: it was
    // already shrunk on the run that saved it, so we replay it as-is and skip shrinking
    // (hypothesis's "does not shrink on replay" behavior). A saved example that no
    // longer reproduces is deleted and we fall through to normal generation+shrink.
    let mut from_db = false;
    // should_generate_more bookkeeping (upstream MIN_TEST_CALLS=10): once a bug is found we
    // keep generating only while call_count < 10 OR call_count < min(first_bug+1000,
    // last_bug*2) — so a lone bug stops the search near MIN_TEST_CALLS, but a stream of new
    // distinct bugs keeps it alive (test_stops_immediately_on_replay, test_finds_multiple).
    const MIN_TEST_CALLS: u64 = 10;
    let mut call_count = 0u64;
    let mut first_bug_at: Option<u64> = None;
    let mut last_bug_at = 0u64;

    // ---- Database reuse phase ----
    // Replay previously-saved minimal examples before generating anything. The DB is
    // duck-typed (real hypothesis ExampleDatabase or our native one): fetch(key) yields
    // serialized choice sequences; delete(key, value) prunes stale ones.
    if let (Some(db), Some(key)) = (database.as_ref(), db_key.as_ref()) {
        let dbb = db.bind(py);
        let key_obj = PyBytes::new(py, key);
        if let Ok(saved) = dbb.call_method1("fetch", (&key_obj,)) {
            if let Ok(it) = saved.try_iter() {
                for item in it {
                    let Ok(buf) = item else { continue };
                    let Ok(raw) = buf.extract::<Vec<u8>>() else { continue };
                    let Ok(dchoices) = crate::database::decode(py, &raw) else {
                        let _ = dbb.call_method1("delete", (&key_obj, &buf));
                        continue;
                    };
                    let cd = Py::new(py, ConjectureData::new_for_choices(py, &dchoices))?;
                    match with_executor(py, executor_b, || {
                        run_one(py, &test, &strategies, cd.bind(py))
                    })? {
                        Outcome::Interesting(args, exc) => {
                            // Collect EVERY distinct-origin saved failure (don't stop at the
                            // first): a multi-bug run saves each bug, and replay must surface
                            // them all as an ExceptionGroup (test_replays_both_failing_values).
                            let choices = extract_choices(py, cd.bind(py))?;
                            let o = origin_of(py, &exc)?;
                            call_count += 1;
                            if !bugs.iter().any(|b| b.0 == o) {
                                bugs.push((o, choices, args, exc));
                            }
                            from_db = true;
                        }
                        _ => {
                            // No longer reproduces (or filtered) — prune the stale entry.
                            let _ = dbb.call_method1("delete", (&key_obj, &buf));
                        }
                    }
                }
            }
        }
    }

    // Try the all-simplest example first, like hypothesis's "simplest" ChoiceTemplate:
    // every draw returns its simplest value (0 for ints, false for bools, empty
    // collections), reliably exercising edge cases (e.g. a test that only fails on
    // x==0) that random sampling rarely hits.
    if bugs.is_empty() {
        let cd = Py::new(py, ConjectureData::new_simplest())?;
        call_count += 1;
        match with_executor(py, executor_b, || {
            run_one_named(py, &test, &strategies, cd.bind(py), names_ref, None)
        })? {
            Outcome::Valid => valid += 1,
            Outcome::Invalid => {
                invalid += 1;
                if cd.bind(py).borrow().status_is_overrun() {
                    overrun += 1;
                }
                if let Ok(seq) = node_seq(py, cd.bind(py)) {
                    invalid_tree.add(&seq);
                }
            }
            Outcome::Interesting(args, exc) => {
                let choices = extract_choices(py, cd.bind(py))?;
                let o = origin_of(py, &exc)?;
                bugs.push((o, choices, args, exc));
                first_bug_at.get_or_insert(call_count);
                last_bug_at = call_count;
            }
            Outcome::Redundant => {} // simplest passes no redundancy tree; unreachable
        }
        track_example(py, &mut track_tree, &mut all_tree, &mut tracked_nodes, cd.bind(py));
        total_draw += LAST_DRAW_SECS.with(|c| c.get());
    }

    // Generation phase. Single-bug: stop at the first failure. Multi-bug: keep going to
    // collect every distinct origin, bounded by a generation budget (max_examples) once
    // at least one bug is in hand so all-failing tests don't spin to `cap`. A failure that
    // came straight from the database is already the known (shrunk) bug, so we stop after
    // replaying it instead of re-searching — replay runs exactly one example
    // (test_stops_immediately_on_replay).
    let mut keep_going = bugs.is_empty() || (report_multiple_bugs && !from_db);
    while keep_going && valid < max_examples && iters < cap {
        // A long generation run (e.g. find_any with max_examples=10**6 hunting a rare
        // condition) holds the GIL continuously across the whole loop, starving the
        // pytest-fast worker's heartbeat thread — the daemon then SIGKILLs the worker as
        // unresponsive (a -9 with no pytest timeout). Briefly drop the GIL every few thousand
        // iterations so other Python threads (heartbeat, signals) get a chance to run.
        if iters > 0 && iters % 2048 == 0 {
            py.allow_threads(std::thread::yield_now);
        }
        let cd = Py::new(py, ConjectureData::new_generate(seed.wrapping_add(iters)))?;
        // Pass the all-examples tree (when tracking a finite space) so an already-explored
        // choice sequence is skipped without running the test — novel-prefix generation.
        // The borrow is confined to the closure so track_example can mutate it afterwards.
        match with_executor(py, executor_b, || {
            run_one_named(
                py,
                &test,
                &strategies,
                cd.bind(py),
                names_ref,
                if track_tree { Some(&all_tree) } else { None },
            )
        })? {
            Outcome::Valid => valid += 1,
            // Already-seen example, test not run — retry (the loop's iters bound + the
            // exhaustion check below terminate it once the finite space is covered).
            Outcome::Redundant => {}
            Outcome::Invalid => {
                invalid += 1;
                if cd.bind(py).borrow().status_is_overrun() {
                    overrun += 1;
                }
                if valid == 0 {
                    if let Ok(seq) = node_seq(py, cd.bind(py)) {
                        invalid_tree.add(&seq);
                    }
                }
            }
            Outcome::Interesting(args, exc) => {
                let o = origin_of(py, &exc)?;
                if !bugs.iter().any(|b| b.0 == o) {
                    let choices = extract_choices(py, cd.bind(py))?;
                    bugs.push((o, choices, args, exc));
                    first_bug_at.get_or_insert(call_count + 1);
                    last_bug_at = call_count + 1;
                }
                if !report_multiple_bugs {
                    keep_going = false;
                }
            }
        }
        track_example(py, &mut track_tree, &mut all_tree, &mut tracked_nodes, cd.bind(py));
        iters += 1;
        call_count += 1;
        total_draw += LAST_DRAW_SECS.with(|c| c.get());
        // Novel-prefix exhaustion: the entire finite choice space has been generated, so
        // there's nothing new left to try — stop (test_given_usable_inline_on_lambdas:
        // booleans() has exactly 2 values).
        if track_tree && all_tree.exhausted() {
            keep_going = false;
        }
        // HealthCheck.too_slow: within the first 10 valid examples, if total argument-draw
        // time exceeds the allowance, generation is too slow to be useful.
        if !too_slow_suppressed && valid < 10 && bugs.is_empty() && total_draw > draw_time_limit {
            return Err(too_slow_err(py, valid, total_draw));
        }
        // should_generate_more: once a bug exists, stop unless we're still under
        // MIN_TEST_CALLS or actively finding new distinct bugs (the first_bug+1000 /
        // last_bug*2 window). Replaces the blunt "run to max_examples" stop.
        if report_multiple_bugs && !bugs.is_empty() {
            let fb = first_bug_at.unwrap_or(call_count);
            let window = (fb + 1000).min(last_bug_at.saturating_mul(2));
            if call_count >= MIN_TEST_CALLS && call_count >= window {
                keep_going = false;
            }
        }
        // Zero valid examples and we've hit the give-up threshold: stop so the reported
        // reject count is exactly INVALID_THRESHOLD_BASE+1 (test_notes_high_filter_rates).
        if valid == 0 && invalid >= invalid_giveup {
            keep_going = false;
        }
    }

    if bugs.is_empty() {
        // Found no failure. If we also found NO valid examples (everything was
        // filtered/rejected), that's a too-much-filtering health check failure,
        // matching hypothesis (FailedHealthCheck.filter_too_much / Unsatisfiable).
        if valid == 0 && invalid > 0 {
            // Every rejection was an overrun (the example was too large to finish
            // generating, not filtered): report that specific reason, regardless of
            // filter_too_much suppression — the data_too_large health check that would
            // otherwise fire is what the test suppresses
            // (test_notes_high_overrun_rates_in_unsatisfiable_error).
            if overrun == invalid {
                let name = test_name.as_deref().unwrap_or("test");
                let msg = format!(
                    "Unable to satisfy assumptions of {name}. {overrun} of {invalid} \
                     examples were too large to finish generating; try reducing the \
                     typical size of your inputs?"
                );
                return Err(unsatisfiable_msg_err(py, msg));
            }
            // Every reachable finite branch explored and all invalid ⇒ Unsatisfiable;
            // otherwise the strategy just filters out a lot ⇒ filter_too_much health check.
            if invalid_tree.exhausted() {
                return Err(unsatisfiable_err(py));
            }
            // When filter_too_much is suppressed, the run isn't a health-check failure —
            // it's genuinely unsatisfiable, reported with the reject counts (every example
            // here was rejected, so invalid == call_count).
            if filter_suppressed {
                let name = test_name.as_deref().unwrap_or("test");
                let msg = format!(
                    "Unable to satisfy assumptions of {name}. {invalid} of {invalid} \
                     examples failed a .filter() or assume() condition. Try making your \
                     filters or assumes less strict, or rewrite using strategy parameters: \
                     st.integers().filter(lambda x: x > 0) fails less often (that is, never) \
                     when rewritten as st.integers(min_value=1)."
                );
                return Err(unsatisfiable_msg_err(py, msg));
            }
            return Err(filter_too_much_err(py));
        }
        return Ok(None);
    }

    // ---- Single-bug path (the overwhelmingly common case) ----
    // Verify the failure reproduces with the SAME interesting-origin before shrinking.
    // Re-running the SAME choices and getting a DIFFERENT failure origin means the
    // test is non-deterministic — hypothesis surfaces this as a FlakyFailure rather
    // than shrinking a moving target. (A replay that PASSES or REJECTS is handled by
    // the @given wrapper's own replay-divergence check.) This re-run is also the
    // second test call on a DB replay (the "does not shrink on replay" count).
    if bugs.len() == 1 {
        let (origin0, choices, args, exc) = bugs.pop().expect("len==1");
        // A DB replay was already verified on the run that saved it, and the @given
        // wrapper does its own confirming `runner(*falsifying)` replay — so skipping
        // this re-run keeps the replay-call count at the upstream value (it would
        // otherwise be one too many: "does not shrink on replay").
        // Likewise skip when not shrinking (max_shrinks==0, i.e. phases=no_shrink):
        // there is no moving-target shrink to protect, and the wrapper's own replay is
        // the single confirming re-run, so the failing example runs exactly twice
        // (find + reproduce) — test_when_set_to_no_simplifies_runs_failing_example_twice.
        if !from_db && max_shrinks > 0 {
            let cd2 = Py::new(py, ConjectureData::new_for_choices(py, &choices))?;
            if let Outcome::Interesting(_, exc2) = run_one(py, &test, &strategies, cd2.bind(py))? {
                if origin_of(py, &exc2)? != origin0 {
                    return Err(flaky_failure(
                        py,
                        "Inconsistent results from replaying a test case!",
                        &[exc, exc2],
                    ));
                }
            }
        }
        let (sargs, sexc, schoices) = if from_db {
            (args, exc, choices)
        } else {
            shrink(py, &test, &strategies, choices, args, exc, max_shrinks)?
        };
        MINIMAL_CHOICES.with(|c| *c.borrow_mut() = schoices.iter().map(|p| p.clone_ref(py)).collect());
        // Persist the minimal failing choice sequence so a subsequent run replays it
        // directly (and does not re-shrink). Stale entries were pruned during reuse.
        if let (Some(db), Some(key)) = (database.as_ref(), db_key.as_ref()) {
            if let Ok(blob) = crate::database::encode_choices(py, &schoices) {
                let dbb = db.bind(py);
                let key_obj = PyBytes::new(py, key);
                let val_obj = PyBytes::new(py, &blob);
                let _ = dbb.call_method1("save", (&key_obj, &val_obj));
            }
        }
        // Shrinking may have slipped into other distinct bugs.
        let slipped = shrink_slipped(
            py,
            &test,
            &strategies,
            &[origin0],
            max_shrinks,
            database.as_ref(),
            db_key.as_ref(),
        )?;
        if slipped.is_empty() {
            return Ok(Some(vec![(sargs, sexc)]));
        }
        if report_multiple_bugs {
            let mut results = vec![(sargs, sexc)];
            for (a, e, _c) in slipped {
                results.push((a, e));
            }
            return Ok(Some(results));
        }
        // report_multiple_bugs off: report only the globally-minimal failure (smallest
        // sort_key across the slipped origins) — test_can_disable_multiple_error_reporting.
        let mut best = (sargs, sexc, schoices);
        for (a, e, c) in slipped {
            if choices_lt(py, &c, &best.2)? {
                best = (a, e, c);
            }
        }
        return Ok(Some(vec![(best.0, best.1)]));
    }

    // ---- Multi-bug path: shrink each distinct origin independently ----
    // The @given wrapper assembles these into an ExceptionGroup (report_multiple_bugs).
    // Bugs replayed from the database are already minimal, so we replay them as-is (no
    // re-shrink, no re-save) — matching "does not shrink on replay".
    let mut results: Vec<(Py<PyAny>, Py<PyAny>)> = Vec::new();
    let mut origins: Vec<(String, i64)> = Vec::new();
    for (origin, choices, args, exc) in bugs {
        if from_db {
            origins.push(origin);
            results.push((args, exc));
            continue;
        }
        // shrink + confirm: a bug that doesn't reproduce on replay is flaky -> FlakyFailure
        // inside the group (test_handles_flaky_tests_where_only_one_is_flaky).
        let (sargs, sexc, schoices) =
            shrink_confirmed(py, &test, &strategies, &origin, choices, args, exc, max_shrinks)?;
        origins.push(origin);
        if let (Some(db), Some(key)) = (database.as_ref(), db_key.as_ref()) {
            if let Ok(blob) = crate::database::encode_choices(py, &schoices) {
                let dbb = db.bind(py);
                let key_obj = PyBytes::new(py, key);
                let val_obj = PyBytes::new(py, &blob);
                let _ = dbb.call_method1("save", (&key_obj, &val_obj));
            }
        }
        results.push((sargs, sexc));
    }
    // Shrinking the known bugs may have slipped into further distinct bugs — surface them.
    if !from_db {
        for (a, e, _c) in shrink_slipped(
            py,
            &test,
            &strategies,
            &origins,
            max_shrinks,
            database.as_ref(),
            db_key.as_ref(),
        )? {
            results.push((a, e));
        }
    }
    Ok(Some(results))
}

/// The interesting-origin of an exception: (type name, deepest traceback line). Two
/// failures are "the same bug" iff these match — used to detect flaky divergence.
fn origin_of(py: Python<'_>, exc: &Py<PyAny>) -> PyResult<(String, i64)> {
    let e = exc.bind(py);
    let tname = e.get_type().name()?.to_string();
    let mut lineno = -1i64;
    let mut cur = e.getattr("__traceback__").ok();
    while let Some(tb) = cur {
        if tb.is_none() {
            break;
        }
        if let Ok(ln) = tb.getattr("tb_lineno").and_then(|n| n.extract::<i64>()) {
            lineno = ln;
        }
        cur = tb.getattr("tb_next").ok();
    }
    Ok((tname, lineno))
}

/// Build a `FlakyFailure(msg, [excs])` (a BaseExceptionGroup) as a raisable PyErr.
fn flaky_failure(py: Python<'_>, msg: &str, excs: &[Py<PyAny>]) -> PyErr {
    let list = PyList::new(py, excs.iter().map(|e| e.bind(py))).ok();
    for (module, cls) in [
        ("hypothesis.errors", "FlakyFailure"),
        ("hypothesis_fast.errors", "FlakyFailure"),
    ] {
        if let (Some(l), Ok(c)) = (&list, py.import(module).and_then(|m| m.getattr(cls))) {
            if let Ok(inst) = c.call1((msg, l)) {
                return PyErr::from_value(inst);
            }
        }
    }
    pyo3::exceptions::PyAssertionError::new_err(msg.to_string())
}

/// Unsatisfiable: the (finite) choice space was exhausted and no value satisfied the test.
fn unsatisfiable_err(py: Python<'_>) -> PyErr {
    unsatisfiable_msg_err(
        py,
        "Unable to satisfy assumptions of test: all possible values were \
         rejected (Unsatisfiable)."
            .to_string(),
    )
}

fn unsatisfiable_msg_err(py: Python<'_>, msg: String) -> PyErr {
    for (module, cls) in [
        ("hypothesis.errors", "Unsatisfiable"),
        ("hypothesis_fast.errors", "Unsatisfiable"),
    ] {
        if let Ok(c) = py.import(module).and_then(|m| m.getattr(cls)) {
            if let Ok(inst) = c.call1((&msg,)) {
                return PyErr::from_value(inst);
            }
        }
    }
    pyo3::exceptions::PyAssertionError::new_err(msg)
}

fn too_slow_err(py: Python<'_>, valid: u32, draw_secs: f64) -> PyErr {
    let msg = format!(
        "Data generation is extremely slow: Only produced {valid} valid examples in \
         {draw_secs:.2} seconds of generation (HealthCheck.too_slow). Try decreasing the \
         size of the data you're generating, or disable the health check."
    );
    for (module, cls) in [
        ("hypothesis.errors", "FailedHealthCheck"),
        ("hypothesis_fast.errors", "FailedHealthCheck"),
    ] {
        if let Ok(c) = py.import(module).and_then(|m| m.getattr(cls)) {
            if let Ok(inst) = c.call1((&msg,)) {
                return PyErr::from_value(inst);
            }
        }
    }
    pyo3::exceptions::PyRuntimeError::new_err(msg)
}

fn filter_too_much_err(py: Python<'_>) -> PyErr {
    let msg = "It looks like your strategy filters out a lot of data; the test \
               was unable to find any valid examples (filter_too_much).";
    // Prefer FailedHealthCheck; fall back to Unsatisfiable, then a generic error.
    for (module, cls) in [
        ("hypothesis.errors", "FailedHealthCheck"),
        ("hypothesis_fast.errors", "Unsatisfiable"),
    ] {
        if let Ok(c) = py.import(module).and_then(|m| m.getattr(cls)) {
            if let Ok(inst) = c.call1((msg,)) {
                return PyErr::from_value(inst);
            }
        }
    }
    pyo3::exceptions::PyRuntimeError::new_err(msg)
}

/// Replay the exact `choices` through the test ONCE (for `@reproduce_failure`).
/// Returns `(args_tuple, exception)` if the replay reproduces a failure, else None
/// (the test passed, rejected, or the choice shape no longer matches the strategy).
#[pyfunction]
#[pyo3(name = "reproduce_native", signature = (test, strategies, choices))]
pub(crate) fn reproduce_native(
    py: Python<'_>,
    test: Py<PyAny>,
    strategies: Vec<Py<PyAny>>,
    choices: Vec<Py<PyAny>>,
) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
    let cd = Py::new(py, ConjectureData::new_for_choices(py, &choices))?;
    match run_one(py, &test, &strategies, cd.bind(py))? {
        Outcome::Interesting(args, exc) => Ok(Some((args, exc))),
        _ => Ok(None),
    }
}

/// Current example's accumulated draw time in seconds (argument draws + interactive
/// st.data() draws). The @given runner reads it to exclude draw time from the per-example
/// deadline measurement (the deadline applies to test execution, not data generation).
#[pyfunction]
#[pyo3(name = "draw_secs")]
fn draw_secs_py() -> f64 {
    LAST_DRAW_SECS.with(|c| c.get())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_native, m)?)?;
    m.add_function(wrap_pyfunction!(reproduce_native, m)?)?;
    m.add_function(wrap_pyfunction!(draw_secs_py, m)?)?;
    m.add_function(wrap_pyfunction!(set_use_py_clock_py, m)?)?;
    m.add_function(wrap_pyfunction!(shrink_report_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimal_choices_py, m)?)?;
    Ok(())
}
