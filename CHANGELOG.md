# Changelog

## 0.0.3

- Performance: cut the engine's fixed per-example overhead ~2.7× (≈16.9 → 6.3 µs) by building the
  per-run `BuildContext` once and reusing it across examples, instead of reconstructing a throwaway
  `ConjectureData` + `BuildContext` on every example. The realised speedup scales with how cheap
  each example is: the simplest tests — a bare `integers()`/`text()` with a trivial body — run
  ~2.5–2.7× faster than before (≈33–38× upstream), while generation-heavy strategies, already
  dominated by the draw itself, are roughly unchanged. The realistic per-strategy generation battery
  is a geometric mean ~9.3× upstream. See BENCHMARKS.md.
- No public API or behaviour change; the unmodified upstream Hypothesis test suite still runs with
  0 failures.

## 0.0.2

- Fix: `HealthCheck` now shares object identity with the real `hypothesis` enum when that
  package is importable (mirroring the exception-class rebinding in `errors.py`). The upstream
  pytest plugin gates its function-scoped-fixture warning on
  `HealthCheck.function_scoped_fixture in settings.suppress_health_check`, compared by enum
  identity. With two distinct enums that membership test was always false, so a consumer suite
  that kept the upstream plugin enabled saw the health check fire even after suppressing it
  through `@settings(suppress_health_check=[...])`. Member names and integer values already
  matched upstream and the engine consumes suppression by name, so behaviour is otherwise
  unchanged.

## 0.0.1 — initial public alpha

First public release. A native-Rust reimplementation of [Hypothesis](https://hypothesis.readthedocs.io/),
a drop-in replacement for its public API with example generation and shrinking moved off the
Python interpreter and into a compiled engine.

- The whole Conjecture stack in Rust (via PyO3): the typed choice sequence, `ConjectureData` +
  primitive provider, a strategy node tree drawn entirely in native code, the generate-and-shrink
  runner, and the database choice-format.
- Drop-in public API: `@given`, `@settings`, `@example`, `@composite`, `data()`, `find`, `assume`,
  `target`, `note`, `from_type` / `register_type_strategy`, settings profiles,
  `@reproduce_failure` / `@seed`, `RuleBasedStateMachine` stateful testing, async test bodies and
  pytest fixtures.
- Compatibility validated by running the unmodified upstream Hypothesis test files against this
  engine: 0 failures.
- Transparent fallback to the real `hypothesis` package (optional `[fallback]` extra) for tests
  that pass Hypothesis-internal strategy objects directly to `@given`.

Known gaps (alpha, search-quality not correctness): `target()` is observe-only (no Pareto/optimise
phase); the shrinker uses fewer passes than upstream. See the README.
