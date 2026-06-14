"""Diff a `pytest-fast --dump` outcome map against the committed parity baseline.

`tests/parity_outcomes.json` is the explicit, committed expectation for EVERY test in the
upstream-Hypothesis parity suite. `xfailed` and `xpassed` are normalized to a single `xfail`
class: the strict=False flaky tests (see the conftest `_XFAIL` map) flip between those two
run-to-run, but the normalized map is stable. The check fails on any:

  - DROP      — a baseline test not run (e.g. a missing optional dep silently hid a whole file),
  - ADDITION  — a test run but absent from the baseline (a new/undeclared test),
  - REGRESSION/MARK-CHANGE — an outcome that differs (passed→failed, xfail→passed, …),

forcing a deliberate `make parity-baseline` whenever the suite legitimately changes, so silent
drift (like a 3956→3517 collection drop) turns CI red instead of green.

Usage:
    python tests/check_parity_outcomes.py <dump.json>            # compare; exit 1 on any diff
    python tests/check_parity_outcomes.py <dump.json> --update   # rewrite the baseline from <dump>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE = Path(__file__).parent / "parity_outcomes.json"
ABOUT = (
    "Expected per-test outcomes of the upstream-Hypothesis parity suite (pytest-fast --runs 1). "
    "xfailed and xpassed are normalized to 'xfail' (strict=False flaky tests flip between them "
    "run-to-run). Regenerate with `make parity-baseline`; CI diffs every run against this file "
    "via tests/check_parity_outcomes.py."
)


def normalize(dump: dict[str, str]) -> dict[str, str]:
    return {k: ("xfail" if v in ("xfailed", "xpassed") else v) for k, v in dump.items()}


def load_run(path: str) -> dict[str, str]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"dump {path!r} is empty or not a {{nodeid: outcome}} object")
    return normalize(data)


def write_baseline(outcomes: dict[str, str]) -> None:
    payload = {"_about": ABOUT, "outcomes": dict(sorted(outcomes.items()))}
    BASELINE.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {BASELINE.name}: {len(outcomes)} tests")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: check_parity_outcomes.py <dump.json> [--update]")
    run = load_run(args[0])

    if "--update" in args:
        write_baseline(run)
        return 0

    if not BASELINE.exists():
        raise SystemExit(f"{BASELINE.name} missing — generate it with `make parity-baseline`")
    base: dict[str, str] = json.loads(BASELINE.read_text())["outcomes"]

    run_keys, base_keys = set(run), set(base)
    dropped = sorted(base_keys - run_keys)
    added = sorted(run_keys - base_keys)
    changed = sorted((k, base[k], run[k]) for k in (base_keys & run_keys) if base[k] != run[k])

    print(f"parity: {len(run)} tests run vs {len(base)} expected")

    def section(title: str, rows: list[str]) -> None:
        print(f"\n{title} ({len(rows)}):")
        for r in rows[:50]:
            print(f"  {r}")
        if len(rows) > 50:
            print(f"  … +{len(rows) - 50} more")

    if dropped:
        section("DROPPED — expected but not run (a hidden file / missing dependency?)",
                [f"- {k}  (expected {base[k]})" for k in dropped])
    if added:
        section("ADDED — run but not declared in the baseline",
                [f"+ {k}  ({run[k]})" for k in added])
    if changed:
        section("CHANGED — outcome differs from the baseline",
                [f"! {k}: expected {b}, got {r}" for k, b, r in changed])

    if dropped or added or changed:
        print("\nPARITY MISMATCH — if this change is intended, run `make parity-baseline` and commit.")
        return 1
    print("\nPARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
