"""Diff a `pytest-fast --dump` outcome map against the committed PER-CONFIG parity baseline.

Parity outcomes legitimately differ by (OS, Python version) — different builtins/tests collect,
Union/Optional reprs differ, the Unicode DB version differs — so baselines are keyed by config:
`tests/parity_baselines/{os}-py{major}{minor}.json` (e.g. `linux-py312.json`, `macos-py314.json`).
Each is the explicit, committed expectation for that config. `xfailed` and `xpassed` are normalized
to one `xfail` class (the strict=False flaky tests flip between them run-to-run; normalized is
stable). A present baseline fails the check on any drop / addition / regression / mark-change.

If no baseline exists for the current config yet (bootstrap), the check still FAILS on any real
`failed`/`error` outcome (so a broken config can't pass green), but PASSES otherwise with a note to
commit a baseline (`make parity-baseline`). The run's dump is always uploaded as a CI artifact.

Usage:
    python tests/check_parity_outcomes.py <dump.json>            # compare; exit 1 on diff/failure
    python tests/check_parity_outcomes.py <dump.json> --update   # write this config's baseline
"""
from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from pathlib import Path

BASELINE_DIR = Path(__file__).parent / "parity_baselines"
_OS = {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(platform.system(), "other")
CONFIG = f"{_OS}-py{sys.version_info.major}{sys.version_info.minor}"
BASELINE = BASELINE_DIR / f"{CONFIG}.json"
ABOUT = (
    "Expected per-test outcomes of the upstream-Hypothesis parity suite (pytest-fast --runs 1) for "
    f"config {CONFIG}. xfailed/xpassed normalized to 'xfail'. Regenerate on this config with "
    "`make parity-baseline`; CI diffs every run against the matching {os}-py{ver} file."
)


def normalize(dump: dict[str, str]) -> dict[str, str]:
    return {k: ("xfail" if v in ("xfailed", "xpassed") else v) for k, v in dump.items()}


def load_run(path: str) -> dict[str, str]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"dump {path!r} is empty or not a {{nodeid: outcome}} object")
    return normalize(data)


def write_baseline(outcomes: dict[str, str]) -> None:
    BASELINE_DIR.mkdir(exist_ok=True)
    payload = {"_about": ABOUT, "outcomes": dict(sorted(outcomes.items()))}
    BASELINE.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {BASELINE.relative_to(Path(__file__).parent.parent)}: {len(outcomes)} tests")


def section(title: str, rows: list[str]) -> None:
    print(f"\n{title} ({len(rows)}):")
    for r in rows[:60]:
        print(f"  {r}")
    if len(rows) > 60:
        print(f"  … +{len(rows) - 60} more")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        raise SystemExit("usage: check_parity_outcomes.py <dump.json> [--update|--failures-only]")
    run = load_run(args[0])

    if "--update" in args:
        write_baseline(run)
        return 0

    # --failures-only: ignore the (pinned-version) baseline entirely and just fail on real
    # failures. Used by the scheduled job that fetches the LATEST upstream tests — "do we still
    # PASS the newest suite?" — where a full diff vs the pinned baseline would be all test churn.
    if "--failures-only" in args:
        bad = sorted(k for k, v in run.items() if v in ("failed", "error"))
        print(f"parity [{CONFIG}, failures-only]: {len(run)} tests run, {len(bad)} failed/error")
        if bad:
            section("FAILED/ERROR", [f"! {k}: {run[k]}" for k in bad])
            return 1
        print("\nNO FAILURES")
        return 0

    if not BASELINE.exists():
        # Bootstrap: no committed baseline for this config. Don't pass a config that has real
        # failures, but otherwise allow it (and ask for a baseline) so the matrix can be filled in.
        bad = sorted(k for k, v in run.items() if v in ("failed", "error"))
        print(f"parity: {len(run)} tests run on {CONFIG} — NO committed baseline yet")
        if bad:
            section("FAILED/ERROR with no baseline to bless them", [f"! {k}: {run[k]}" for k in bad])
            print(f"\nNO BASELINE for {CONFIG} and {len(bad)} real failures — fix them, then `make parity-baseline`.")
            return 1
        print(f"\nNO BASELINE for {CONFIG} (no failures) — commit one with `make parity-baseline`.")
        return 0

    base: dict[str, str] = json.loads(BASELINE.read_text())["outcomes"]
    run_keys, base_keys = set(run), set(base)
    dropped = sorted(base_keys - run_keys)
    added = sorted(run_keys - base_keys)
    changed = sorted((k, base[k], run[k]) for k in (base_keys & run_keys) if base[k] != run[k])

    print(f"parity [{CONFIG}]: {len(run)} tests run vs {len(base)} expected")
    if dropped:
        # Group dropped tests by file and statically flag whole-file drops: when every test a
        # file contributes to the baseline vanishes, the file failed to COLLECT (an upstream
        # symbol it imports was removed/renamed) — the root cause, not 19 separate lines.
        base_per_file = Counter(k.split("::", 1)[0] for k in base_keys)
        drop_per_file = Counter(k.split("::", 1)[0] for k in dropped)
        rows = []
        for f in sorted(drop_per_file):
            n, total = drop_per_file[f], base_per_file[f]
            rows.append(
                f"{f} — ALL {n} tests gone -> the file FAILED TO COLLECT (removed/renamed upstream symbol?)"
                if n == total
                else f"{f} — {n}/{total} tests gone"
            )
        section("DROPPED (grouped by file)", rows)
    if added:
        section("ADDED — run but not in the baseline", [f"+ {k}  ({run[k]})" for k in added])
    if changed:
        section("CHANGED — outcome differs from the baseline",
                [f"! {k}: expected {b}, got {r}" for k, b, r in changed])

    if dropped or added or changed:
        print(f"\nPARITY MISMATCH [{CONFIG}] — if intended, run `make parity-baseline` on this config and commit.")
        return 1
    print(f"\nPARITY OK [{CONFIG}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
