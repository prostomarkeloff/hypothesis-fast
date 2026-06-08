//! hypothesis_fast._engine — native Rust Conjecture engine.
//!
//! A from-scratch port of hypothesis's Conjecture engine, entirely in Rust: a flat
//! typed choice sequence (`choice`/`floats`/`intervalset`/`charmap`), `ConjectureData`
//! + primitive provider (`data`/`provider`), a strategy-node tree drawn entirely in
//! Rust (`strategy`), the generate+shrink runner (`engine`), and the choice-sequence
//! database format (`database`). The Python layer (`hypothesis_fast.strategies`/
//! `.core`) is a thin frontend over the pyclasses/pyfunctions registered here.

mod charmap;
mod choice;
mod data;
mod database;
mod engine;
mod floats;
mod intervalset;
mod provider;
mod statistics;
mod stateful;
mod strategy;

use pyo3::prelude::*;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    floats::register(m)?;
    intervalset::register(m)?;
    choice::register(m)?;
    statistics::register(m)?;
    data::register(m)?;
    strategy::register(m)?;
    engine::register(m)?;
    database::register(m)?;
    stateful::register(m)?;
    Ok(())
}
