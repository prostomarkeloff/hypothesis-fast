//! StatefulRunner: the per-example step loop for `RuleBasedStateMachine`, in Rust.
//!
//! It drives the SAME Python per-step helpers as the Python reference loop
//! (`hypothesis_fast.stateful._must_stop` / `_select_rule` / `_run_step` / `_emit`), so the
//! observable behaviour is identical — only the loop control (the stop-boolean draw, step
//! counting, and the try/finally teardown + falsifying-trace attach) lives in Rust. This is
//! the "everything in Rust" deliverable: no Python hot-path `while` loop.
#![allow(clippy::wildcard_imports)]
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Built once per state-machine test from a Python descriptor and reused as the `@given`
/// body (wrapped by a thin Python function so `given` can introspect a normal signature).
#[pyclass(module = "hypothesis_fast._engine")]
pub(crate) struct StatefulRunner {
    factory: Py<PyAny>,
    settings: Py<PyAny>,
    min_steps: i64,
    flaky_state: Py<PyAny>,
}

#[pymethods]
impl StatefulRunner {
    #[new]
    fn new(factory: Py<PyAny>, settings: Py<PyAny>, min_steps: i64, flaky_state: Py<PyAny>) -> Self {
        StatefulRunner { factory, settings, min_steps, flaky_state }
    }

    /// core reads this on the @given body to suppress its default `run_state_machine(data=...)`
    /// note (we attach our own step trace instead) — mirror the Python loop's flag.
    #[getter]
    fn _hypothesis_internal_print_given_args(&self) -> bool {
        false
    }

    fn __call__(&self, py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let stateful = py.import("hypothesis_fast.stateful")?;
        let cd = data.getattr("conjecture_data")?;
        let machine = self.factory.bind(py).call0()?;

        let rbsm = stateful.getattr("RuleBasedStateMachine")?;
        if !machine.is_instance(&rbsm)? {
            let inv = py.import("hypothesis_fast.errors")?.getattr("InvalidArgument")?;
            return Err(PyErr::from_value(inv.call1((format!(
                "state_machine_factory() must return a RuleBasedStateMachine, got {}",
                machine.repr()?
            ),))?));
        }
        // Stash the machine on the cd so bundle draws (even nested inside native collections)
        // can reach it; per-cd, so a nested run can't see this machine.
        cd.setattr("_hf_stateful_machine", &machine)?;

        let is_final = stateful.getattr("_is_final")?.call0()?.is_truthy()?;
        let verbosity_debug = py
            .import("hypothesis_fast.settings")?
            .getattr("Verbosity")?
            .getattr("debug")?;
        let verbose = stateful
            .getattr("_verbosity_at_least")?
            .call1((verbosity_debug,))?
            .is_truthy()?;
        let print_steps = is_final || verbose;

        let printed = PyList::empty(py);
        // output = functools.partial(_emit, printed, print_steps) — the same callable shape the
        // Python loop builds as a closure, so _run_step / check_invariants are unchanged.
        let output = py
            .import("functools")?
            .getattr("partial")?
            .call1((stateful.getattr("_emit")?, &printed, print_steps))?;

        let must_stop_fn = stateful.getattr("_must_stop")?;
        let select_rule_fn = stateful.getattr("_select_rule")?;
        let run_step_fn = stateful.getattr("_run_step")?;
        let settings = self.settings.bind(py);
        let flaky_state = self.flaky_state.bind(py);

        let body = (|| -> PyResult<()> {
            output.call1((format!("state = {}()", machine.get_type().name()?),))?;
            machine.call_method1("check_invariants", (settings, &output))?;
            let max_steps: i64 = settings.getattr("stateful_step_count")?.extract()?;
            let p = 2f64.powi(-16);
            let mut steps_run: i64 = 0;
            loop {
                let cd_length: i64 = cd.getattr("length")?.extract()?;
                let must_stop =
                    must_stop_fn.call1((steps_run, self.min_steps, max_steps, cd_length))?;
                let kw = PyDict::new(py);
                kw.set_item("p", p)?;
                kw.set_item("forced", &must_stop)?;
                if cd.call_method("draw_boolean", (), Some(&kw))?.is_truthy()? {
                    break;
                }
                steps_run += 1;
                let rule = select_rule_fn.call1((&machine, &cd, flaky_state))?;
                run_step_fn.call1((&machine, &cd, &rule, settings, &output, print_steps))?;
                machine.call_method1("check_invariants", (settings, &output))?;
            }
            Ok(())
        })();

        // finally: teardown always runs; a raise here (like Python's finally) overrides the
        // body result and skips the trace attach.
        output.call1(("state.teardown()",))?;
        machine.call_method0("teardown")?;

        // On the final (minimal) replay, attach the collected program (prefixed with the
        // "Falsifying example:" header) to the in-flight failure as notes.
        if let Err(ref e) = body {
            if is_final {
                let add_note = stateful.getattr("_add_note")?;
                let inflight = e.value(py);
                add_note.call1((&inflight, "Falsifying example:"))?;
                for line in printed.iter() {
                    add_note.call1((&inflight, &line))?;
                }
            }
        }
        body
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StatefulRunner>()?;
    Ok(())
}
