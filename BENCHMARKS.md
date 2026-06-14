# hypothesis-fast — benchmarks

Native Rust engine vs the real `hypothesis` package. Captured **2026-06-14**.

Every run: **same interpreter** (the repo `.venv`, or each project's own venv for the e2e
suites), **same tests**, **same settings** (so both engines do the same amount of work). Real =
upstream `hypothesis`; native = `hypothesis-fast` aliased in via the `HF_SHIM` shim. The only
variable is the engine.

Two things get measured: how fast the engine *generates*, and how much that moves a real test
run end to end.

---

## 1. Generation throughput

**Methodology — read this before the numbers.** You cannot sum examples/second across
strategies: one `integers()` example and one nested-order record are not the same unit of work or
data, so a grand-total `ex/s` is meaningless and its real-vs-native ratio is just an artifact of
which strategies racked up the most examples. So each strategy is benched on its **own fixed-time
slice** and reports its **own** ex/s and useful-bytes/s. The aggregate speedup is the **geometric
mean** of the per-strategy ratios (mix-independent). "Useful data" = the generated value's size
(utf-8 length of strings, magnitude bytes of ints, recursive collection contents) — not the
engine's internal choice sequence.

The battery is **realistic** — the data a real property-test suite actually generates: domain
records, API JSON payloads, nested models (cattrs / pydantic / attrs style), emails / urls / uuids
/ money / timestamps, collections at sane sizes. Not `2**2000` ints or 16 KB blobs picked to
flatter the multiplier.

| strategy | real ex/s | native ex/s | speedup | native MB/s |
|---|--:|--:|--:|--:|
| email (`from_regex`) | 917 | 18,234 | **19.9×** | 0.280 |
| url (`from_regex`) | 893 | 18,581 | **20.8×** | 0.387 |
| order_nested (`builds` + line items) | 375 | 9,093 | **24.2×** | 1.552 |
| user_record (`@composite`) | 756 | 9,767 | **12.9×** | 1.146 |
| api_json_payload (`recursive`) | 927 | 12,159 | **13.1×** | 0.423 |
| address_record (`builds`) | 1,691 | 19,989 | **11.8×** | 0.553 |
| ipv4 | 3,166 | 24,523 | 7.7× | 0.600 |
| money (decimal, 2 dp) | 3,658 | 22,380 | 6.1× | 0.370 |
| uuid | 4,226 | 24,588 | 5.8× | 1.082 |
| person_name | 4,669 | 25,619 | 5.5× | 0.132 |
| event_batch (list of typed events) | 872 | 3,286 | 3.8× | 1.791 |
| timestamp | 4,001 | 15,055 | 3.8× | 0.712 |

**Speedup: geometric mean 9.3×, median 9.8×, range 3.8×–24.2×.**

What the numbers actually say:

- **5–8× on light fields** (name / timestamp / uuid / money / ipv4). These aren't overhead-bound —
  native does real draw work too (assemble a UUID, a tz-aware datetime, a decimal, a string), so
  the ratio reflects native-vs-upstream *on that work*. The engine's fixed per-example overhead,
  which dominates the *simplest* tests, is a separate axis — see §1b, where it's ~33×.
- **12–24× on the structured stuff** that real suites are full of: `from_regex` fields (email/url
  20× — upstream walks the pattern in Python per character; native does it in Rust), nested
  domain models (order 24× — `builds` + a list of line items, each with a regex SKU, a decimal and
  a uuid: the most Python-per-example of the set), records and recursive API payloads 12–13×.
- The more work a single example takes to build, the more native wins, because upstream pays the
  Python interpreter in proportion to that work and native does not.

Single run each (per-strategy fixed-time, 4 s/strategy); expect a few percent of run-to-run noise.

### 1b. Per-example overhead — the simplest tests

The table above is *generation-dominated*: heavy strategies where building the value is the work.
But the engine also pays a fixed cost per `@given` example regardless of strategy — building the
test context, pinning the PRNGs. That overhead is what caps the *simplest* tests: a bare scalar
strategy with a cheap assertion, the most common shape of everyday property tests, where there's
almost nothing to draw.

Measured as a single `@given` call of `max_examples=5000` with a trivial body, so per-call setup
amortizes away and only the per-example cost remains:

| simplest test | upstream | hypothesis-fast | speedup | µs/example (fast) |
|---|--:|--:|--:|--:|
| `@given(st.integers())` | 5,029 ex/s | 166,428 ex/s | **33×** | 6.0 |
| `@given(st.text())` | 4,057 ex/s | 152,541 ex/s | **38×** | 6.6 |

Upstream spends ~200 µs per example in its (Python) Conjecture machinery whatever the strategy;
hypothesis-fast spends ~6 µs, so when the draw is trivial the gap is widest. A 2026-06 change cut
this per-example overhead **~2.7×** (16.9 → 6.3 µs) by building the per-run test context once and
reusing it across examples instead of reconstructing it every example.

---

## 2. End-to-end — real test suites

`pytest-fast --bench=4` (warm forkserver daemon, 4 iterations, first dropped as warm-up, wall
reported). Both engines auto-detect the same worker count, so it's apples-to-apples. Speedup is
real wall / native wall over the same tests.

| suite | hypothesis | hypothesis-fast | speedup |
|---|--:|--:|--:|
| cattrs — 260 tests | 8.81s | 2.05s | **4.3×** |
| hyperlink — 14 | 1.05s | 0.21s | **5.0×** |
| attrs — 843 | 0.76s | 0.23s | **3.3×** |
| bidict — 114 (stateful) | 0.81s | 0.28s | **2.9×** |
| hypothesmith — 174 | **>600s (timed out)** | 5.21s | native finishes a suite real can't in 10 min |

**End-to-end is lower than pure generation, by design.** It tracks the share of wall time the
suite spends in *the engine* versus *the test body*. `cattrs` round-trips, `attrs` `__eq__`, etc.
run in Python on both engines and dominate; the engine speedup only applies to its slice. So
2.9–5× here, against 4–24× for generation alone — the same engine, different fraction of the work.

(`hypothesmith` generates whole Python programs and parses them with libcst; native runs the suite
in 5.2s, upstream didn't finish a single `--bench` cycle inside the 600s cap. `packaging` is
excluded — it hit a `pytest-fast` forkserver start-up flake on this project's import on every
attempt, unrelated to the engine.)

---

## 3. Correctness

Speed is only worth anything if the results match. The bar:

- **Parity suite** — the unmodified upstream Hypothesis test files run against this engine:
  **0 failed, 3864 passed** (42 skipped, ~39 xfailed, ~11 xpassed; n=3956), lint clean. Stateful
  included (`test_stateful` 89/90 + 1 documented xfail). (A handful of seed-/parallel-sensitive
  filter/charmap tests are `xfail(strict=False)`, so xfailed/xpassed shift a little run to run.)
- **Surveyed real projects** — 15 hypothesis-using projects run native-vs-real; 14/15 are
  0-gap drop-ins (a native failure that isn't also a real failure). The 15th
  (`schemathesis`) needs live HTTP servers, not the engine.
