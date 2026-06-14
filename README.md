# hypothesis-fast

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

A reimplementation of [Hypothesis](https://hypothesis.readthedocs.io/) whose engine is written
in Rust. The public API is unchanged — `@given`, `strategies`, `@composite`, `data()`,
`from_type`, stateful testing — but example generation and shrinking happen in native code
instead of the Python interpreter. For suites that spend their time generating data, that is the
difference between waiting and not.

```python
import hypothesis_fast as hypothesis            # drop-in alias
from hypothesis_fast import given, strategies as st

@given(st.lists(st.integers()))
def test_sorting_is_idempotent(xs):
    assert sorted(sorted(xs)) == sorted(xs)
```

One import. Everything below it is the Hypothesis you already write.

## Why it's faster

Hypothesis represents every test input as a sequence of choices, and upstream draws each of those
choices in Python: a method call per integer, per character, per list element, interpreted one at
a time — then shrinking replays that work hundreds of times to minimise a failure. The design is
sound. The interpreter is the tax.

hypothesis-fast compiles each strategy into a typed node tree and draws it entirely in Rust.
`st.lists(st.integers())` becomes a structure the engine walks natively, emitting the choice
sequence with no Python per element. The only Python that runs while an example is being built is
the code that has to: your `map` / `filter` / `@composite` callables, and the test body itself.

The entire Conjecture stack lives on the Rust side — the choice sequence, the provider that
samples values, the generate-and-shrink runner, the example database. The Python layer is a thin
frontend over it.

## How fast

Two numbers matter: how fast the engine *generates*, and how much that moves a real run.

**Generation.** On realistic data — domain records, API payloads, `from_regex` fields, nested
models, the input a real suite actually builds — example generation runs a geometric mean of
**~9×** upstream (measured per strategy, since you can't sum ex/s across strategies that produce
different things): **5–6×** on light fields (uuid, timestamp, name), climbing to **12–24×** on
regex-constrained fields, nested dataclasses and recursive JSON — exactly what serialization and
schema suites spend their time on.

**Per-example overhead.** Separately from the draw, the engine pays a fixed cost per `@given`
example — building the test context, pinning PRNGs. That cost is now **~6µs** (down ~2.7× after a
2026-06 reuse of the per-run build context), so the *simplest* tests — a bare
`st.integers()`/`st.text()` with a cheap assertion — run **~33×** upstream (166k vs 5k ex/s), where
upstream's ~200µs-per-example machinery dominates. That's the common shape of everyday property
tests, and where the engine helps most outside generation-heavy suites.

**End to end.** How much you feel depends on the share of wall time in the engine versus your test
body. Measured with `pytest-fast --bench` on real projects:

| suite | hypothesis | hypothesis-fast | speedup |
|---|--:|--:|--:|
| cattrs — 260 tests | 8.81s | 2.05s | **4.3×** |
| hyperlink — 14 | 1.05s | 0.21s | **5.0×** |
| attrs — 843 | 0.76s | 0.23s | **3.3×** |
| bidict — 114 (stateful) | 0.81s | 0.28s | **2.9×** |

Same interpreter, same tests, same settings; only the engine differs. The full per-strategy
generation table, the end-to-end methodology, and why the two differ are in
**[BENCHMARKS.md](BENCHMARKS.md)**.

## What works

The surface is the public Hypothesis API, and the bar for "works" is concrete: hypothesis-fast
runs the **unmodified upstream Hypothesis test files** against its own engine — **3,865 passing,
zero failures** (n=3956; the rest are deliberate xfails and intractable skips, documented under
[Limitations](#what-isnt-ported-yet)). `@settings`, `@example`, profiles, deadlines, phases,
`@reproduce_failure` / `@seed`, `target()`, `find`, `assume`, the full `strategies` module
including `from_type` / `register_type_strategy`, `@composite`, interactive `data()`, and
`RuleBasedStateMachine` stateful testing all run natively.

Failures are indistinguishable from upstream: the engine shrinks to a minimal counterexample,
re-runs it, and reports the original exception, traceback, and `Falsifying example:`. Async test
bodies and pytest fixtures work — `@given` keeps fixture params in the wrapped signature for
pytest to inject, supplies the strategy params itself, and runs `async def` bodies on an event
loop.

## The fallback

A few tests pass Hypothesis's *internal* strategy objects straight to `@given` —
`FeatureStrategy`, `random_module()`, raw provider strategies. Those are not part of the public
API and are not reimplemented; when `@given` is handed one, it delegates the whole test to the
real `hypothesis` package, installed via the optional `[fallback]` extra. Anything constructed
through `hypothesis_fast.strategies` runs entirely on the native engine.

## What isn't ported yet

It's an alpha, so here's the honest ledger. Alongside the 3,865 passing tests, **79 are deferred**
(37 xfail + 42 skip). **None is a behavioral correctness gap** — no test that passes upstream for
the *right answer* fails here. They group as:

**Functional gaps you could actually hit** — the two worth knowing about:

- **`target()` is observe-only.** Scores are collected and shown in the statistics, but there's no
  Pareto/optimise phase that steers generation toward them. Targeted tests pass; they just don't
  get the guided search.
- **The shrinker is leaner** — fewer passes than upstream. On ordinary failures you get the same
  minimal example; on a few adversarial shapes upstream shrinks a touch further.

**Different by design, never changes a verdict** (~17): the native generator's value
**distribution** and **shrink minimality** aren't bit-identical to upstream, so a handful of tests
that pin an exact rare-event hit-rate or absolute-minimal shrink are xfailed on purpose; and
interactive **`data()`** replay semantics differ slightly (kept native for speed).

**Internal assertions, not behavior** (~24): tests that check an exact internal `repr`, a
validation-message string, or a warning's wording — our strategy reprs read `hypothesis_fast…`,
our objects live at a different module path. The engine does the right thing; the string just
isn't byte-identical.

**Genuinely not built yet** (~6 + skips): some statistics/reporting *format* details; the
alternative non-random backends (the `crosshair` symbolic provider) aren't ported; and a batch of
skips are external test tooling (`syrupy` snapshots, `pexpect` subprocess drivers) or tests that
are slow on upstream too (`sleep` × examples).

Bottom line: if you build through `hypothesis_fast.strategies` and run ordinary property tests,
none of this is in your way — a failing test fails and shrinks to a counterexample, a passing test
passes.

## Install

Alpha (pre-release). From PyPI with uv (or `pip install`):

```sh
uv add hypothesis-fast
# with the internal-strategy fallback to real hypothesis:
uv add "hypothesis-fast[fallback]"
```

Prebuilt wheels are currently **macOS (Apple Silicon)** only; on other platforms the install
builds from the sdist, which needs a Rust toolchain ([`rustup`](https://rustup.rs)). Installing
from git, for the latest unreleased commit, has the same requirement:

```sh
uv add git+https://github.com/prostomarkeloff/hypothesis-fast.git
```

## Develop

```sh
git clone https://github.com/prostomarkeloff/hypothesis-fast
cd hypothesis-fast && uv sync

make build        # build the Rust engine (maturin develop --release)
make test         # the upstream parity suite
make lint-heavy   # clippy + ruff + pyright
```

The suite runs through [pytest-fast](https://github.com/prostomarkeloff/pytest-fast). After a
Rust change, run `make build` — the compiled extension lives outside the watched paths, so the
daemon won't pick it up on its own.

## License

[MPL-2.0](LICENSE), the same license as upstream Hypothesis.
