# hypothesis-fast — benchmarks

Native Rust engine vs the real `hypothesis` package. Captured **2026-06-08**.

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
| email (`from_regex`) | 912 | 16,990 | **18.6×** | 0.261 |
| url (`from_regex`) | 851 | 17,443 | **20.5×** | 0.364 |
| order_nested (`builds` + line items) | 377 | 9,135 | **24.2×** | 1.550 |
| user_record (`@composite`) | 787 | 10,259 | **13.0×** | 1.201 |
| api_json_payload (`recursive`) | 948 | 12,231 | **12.9×** | 0.385 |
| address_record (`builds`) | 1,683 | 20,026 | **11.9×** | 0.556 |
| ipv4 | 3,195 | 23,029 | 7.2× | 0.564 |
| money (decimal, 2 dp) | 3,717 | 21,224 | 5.7× | 0.351 |
| uuid | 4,367 | 22,663 | 5.2× | 0.997 |
| person_name | 4,732 | 23,375 | 4.9× | 0.120 |
| event_batch (list of typed events) | 900 | 3,427 | 3.8× | 1.823 |
| timestamp | 4,061 | 15,047 | 3.7× | 0.712 |

**Speedup: geometric mean 9.0×, median 9.6×, range 3.7×–24×.**

What the numbers actually say:

- **A floor of ~4× on trivial scalars** (name / timestamp / uuid / money). The bottleneck there
  is *not* the draw — it's that `@given` calls the test body once per example, a Python callback
  the engine can't remove. Native still hits 15–23k ex/s; real does ~4k. Once the draw is trivial,
  that callback caps the ratio.
- **12–24× on the structured stuff** that real suites are full of: `from_regex` fields (email/url
  18–20× — upstream walks the pattern in Python per character; native does it in Rust), nested
  domain models (order 24× — `builds` + a list of line items, each with a regex SKU, a decimal and
  a uuid: the most Python-per-example of the set), records and recursive API payloads 12–13×.
- The more work a single example takes to build, the more native wins, because upstream pays the
  Python interpreter in proportion to that work and native does not.

Single run each (per-strategy fixed-time, 4 s/strategy); expect a few percent of run-to-run noise.

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
  **0 failed, 3865 passed** (42 skipped, 37 xfailed, 12 xpassed; n=3956), lint clean. Stateful
  included (`test_stateful` 89/90 + 1 documented xfail).
- **Surveyed real projects** — 15 hypothesis-using projects run native-vs-real; 14/15 are
  0-gap drop-ins (a native failure that isn't also a real failure). The 15th
  (`schemathesis`) needs live HTTP servers, not the engine.

---

## Reproduce

The bench harness lives in the repo's dev tree (`_external/`, not shipped in a release).

```bash
# generation throughput — realistic per-strategy battery, one engine per process
timeout 60 uv run python _external/bench_gen_time.py > /tmp/gen-real.tsv
timeout 60 env HF_SHIM=1 PYTHONPATH="$PWD/_external/shim:$PWD/python" \
  uv run python _external/bench_gen_time.py > /tmp/gen-native.tsv
uv run python _external/bench_gen_combine.py /tmp/gen-real.tsv /tmp/gen-native.tsv   # per-strategy speedup + geomean

# end-to-end on a real project (clones, installs, benches real then native)
bash tests_materials/scripts/bench-validate.sh python-attrs/cattrs both
```
