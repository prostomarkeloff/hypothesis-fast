# hypothesis-fast — work is organized around TWO commands:
#   make test        fast + reliable full parity suite (pytest-fast warm daemon)
#   make lint-heavy  clippy + ruff + pyright + find-dup-defs (all of them, aggregated)
# Helpers (build / probe / native / kill) support the dev loop.
# Never call plain `pytest` — always go through the pytest-fast daemon.

.DEFAULT_GOAL := help

UV  ?= uv
TTL ?= 600
# Worker count is AUTO-DETECTED by pytest-fast (>= v0.7.1): it pins to the performance-core
# count (e.g. `hw.perflevel0.physicalcpu` on Apple Silicon), because a worker scheduled onto
# a ~half-speed efficiency core becomes the straggler that bounds a work-stealing run. Don't
# hardcode it; override with `PYTEST_FAST_WORKERS=N make test` on the rare occasion you must.

# `--timeout=60` (pytest-timeout): any individual test taking >60s is killed with a traceback
# dump → no silent stalls. `--timeout-method=thread` is safer than `signal` under pytest-fast
# workers (no SIGALRM races inside Rust callbacks).
ADDOPTS = -p no:hypothesispytest -p no:cacheprovider -p pytester --timeout=60 --timeout-method=thread

# Per-worktree socket so two checkouts (git worktrees) don't fight over one daemon — and so
# `make kill` / `make build` can scope their pkill to THIS project's daemon (never a broad
# `pkill -f pytest-fast`, which would nuke other projects' resident daemons).
WT_PATH := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)
WT_HASH := $(shell printf '%s' "$(WT_PATH)" | shasum | cut -c1-6)
WT_SLUG := $(shell basename "$(WT_PATH)" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$$//' | cut -c1-40)
# SOCK_BASE is the shared prefix of both sockets. The v0.7.x daemon self-spawns as
# `python -m pytest_fast --serve --address <SOCK>` (underscore module name), so an old
# `pkill -f "pytest-fast.*<sock>"` pattern misses it — match the socket PATH instead, which
# every process (client, --serve daemon, forkserver, workers) carries in its argv.
SOCK_BASE  := /tmp/pytest-fast-$(WT_SLUG)-$(WT_HASH)
SOCK       := $(SOCK_BASE).sock
SOCK_PROBE := $(SOCK_BASE)-probe.sock

# Watch our Python source + tests; the daemon respawns when any *.py under them changes.
WATCH_ENV = PYTEST_FAST_WATCH_DIRS=python,tests

FAST = $(WATCH_ENV) $(UV) run pytest-fast --ttl $(TTL)

# find-dup-defs is a cargo-installed binary (auto-installed by the `$(DUP_DEFS)` rule).
DUP_DEFS := $(HOME)/.cargo/bin/find-dup-defs

.PHONY: help test lint-heavy native build kill probe parity-check parity-baseline fetch-tests

help:
	@echo "PRIMARY:"
	@echo "  make test        - full parity suite via pytest-fast (warm daemon, flaky tests xfailed)"
	@echo "  make lint-heavy  - clippy + ruff + pyright + find-dup-defs (runs all, aggregates failures)"
	@echo "HELPERS:"
	@echo "  make build       - maturin develop --release, then restart the daemon(s) (fresh .so)"
	@echo "  make probe FILE=test_x.py - one cover file via a separate daemon"
	@echo "  make native      - native package tests via pytest-fast"
	@echo "  make fetch-tests - (re)fetch the upstream parity tests at tests/hypothesis_compat/UPSTREAM_REF"
	@echo "  make kill        - stop the daemon/watcher and remove its sockets"
	@echo
	@echo "workers auto-detect to the perf-core count; override with PYTEST_FAST_WORKERS=N."

# The upstream parity tests are NOT vendored — they're fetched at the pinned UPSTREAM_REF (see
# tests/hypothesis_compat/fetch_upstream_tests.sh). The parity-running targets auto-fetch them
# once if absent (fresh clone); `make fetch-tests` forces a re-fetch (e.g. after bumping the ref).
PARITY_TESTS_SENTINEL := tests/hypothesis_compat/test_core.py

fetch-tests:
	bash tests/hypothesis_compat/fetch_upstream_tests.sh

$(PARITY_TESTS_SENTINEL):
	bash tests/hypothesis_compat/fetch_upstream_tests.sh

test probe parity-check parity-baseline: $(PARITY_TESTS_SENTINEL)

# ── PRIMARY: full parity suite ───────────────────────────────────────────────
# A warm pytest-fast daemon (forkserver + collect-once + work-stealing) that auto-respawns
# when python/ or tests/ change, so a clean one-shot run reuses warm workers. Known-flaky
# tests (RNG-state, unguided-find) are xfail(strict=False) in conftest, so order-dependent
# flakes never fail the run. NOTE: after a RUST edit run `make build` first — the .so is
# rebuilt into python/hypothesis_fast/, which the daemon's *.py-only watch doesn't track.
test:
	PYTEST_ADDOPTS="$(ADDOPTS) tests/hypothesis_compat" \
	  $(FAST) --address $(SOCK)

# ── PRIMARY: heavy lint ──────────────────────────────────────────────────────
# Runs every linter even if an earlier one fails (so you see ALL findings), then exits
# non-zero if any failed — suitable as a CI gate.
lint-heavy: $(DUP_DEFS)
	@fail=0; \
	printf '\n═══════════ clippy ═══════════\n'; \
	  cargo clippy --all-targets || fail=1; \
	printf '\n═══════════ ruff ═══════════\n'; \
	  $(UV) run ruff check python tests bin || fail=1; \
	printf '\n═══════════ pyright ═══════════\n'; \
	  $(UV) run pyright python || fail=1; \
	printf '\n═══════════ find-dup-defs ═══════════\n'; \
	  $(DUP_DEFS) python -D 'suppress:<constants>T=module-local TypeVar, not a real dup' || fail=1; \
	printf '\n'; \
	if [ $$fail -ne 0 ]; then echo "lint-heavy: FAILURES above"; else echo "lint-heavy: clean"; fi; \
	exit $$fail

# Auto-install find-dup-defs from crates.io if it isn't on disk yet.
$(DUP_DEFS):
	cargo install find-dup-defs --locked

# ── HELPERS ──────────────────────────────────────────────────────────────────
# Native package tests (engine-direct).
native:
	PYTEST_ADDOPTS="$(ADDOPTS) tests/test_strategies.py tests/test_given.py tests/test_fallback.py" \
	  $(FAST) --address $(SOCK)

# Rebuild the Rust engine and force fresh daemons: pytest-fast's staleness keys on *.py
# mtime, but the .so is rebuilt into python/hypothesis_fast/ (not a *.py), so the watch
# can't see it. Kill BOTH the main and probe daemons so neither serves a stale engine.
build:
	$(UV) run maturin develop --release
	-pkill -f "$(SOCK_BASE)" 2>/dev/null || true

# Triage a single cover file on a separate socket so the main daemon is untouched.
probe:
	PYTEST_ADDOPTS="$(ADDOPTS) tests/hypothesis_compat/$(FILE)" \
	  $(FAST) --address $(SOCK_PROBE)

kill:
	-pkill -f "$(SOCK_BASE)" 2>/dev/null || true
	-rm -f $(SOCK) $(SOCK).pid $(SOCK).watcher.lock $(SOCK).respawn.lock \
	       $(SOCK).staging $(SOCK).staging.pid \
	       $(SOCK_PROBE) $(SOCK_PROBE).pid $(SOCK_PROBE).watcher.lock \
	       $(SOCK_PROBE).respawn.lock $(SOCK_PROBE).staging $(SOCK_PROBE).staging.pid

# ── PARITY-OUTCOME BASELINE (what CI gates on) ───────────────────────────────
# Single-process pytest-fast (no resident daemon) writes a {nodeid: outcome} dump, which is
# diffed against the committed tests/parity_outcomes.json. Any drop / addition / regression /
# xfail<->pass change fails — the same check CI runs. xfailed/xpassed are normalized (they flip
# run-to-run), so the comparison is stable.
PARITY_DUMP    ?= /tmp/hf-parity-$(WT_HASH).json
PARITY_ADDOPTS  = $(ADDOPTS) --timeout=120 tests/hypothesis_compat

parity-check:
	-PYTEST_ADDOPTS="$(PARITY_ADDOPTS)" $(UV) run pytest-fast --runs 1 --dump $(PARITY_DUMP)
	$(UV) run python tests/check_parity_outcomes.py $(PARITY_DUMP)

# Regenerate the committed baseline from a fresh run; review the git diff before committing.
parity-baseline:
	-PYTEST_ADDOPTS="$(PARITY_ADDOPTS)" $(UV) run pytest-fast --runs 1 --dump $(PARITY_DUMP)
	$(UV) run python tests/check_parity_outcomes.py $(PARITY_DUMP) --update
