#!/usr/bin/env bash
# Fetch the upstream Hypothesis parity test files at the PINNED ref into this directory.
#
# The parity test files are NOT vendored in git — they are unmodified upstream files, fetched
# here at the version recorded in UPSTREAM_REF (a tag or commit SHA). This gives an explicit,
# declarative pin to a Hypothesis version + clear provenance; bumping = edit UPSTREAM_REF,
# re-fetch, regenerate the per-config baselines (`make parity-baseline`), commit. The exact set
# of files comes from UPSTREAM_FILES (one `subdir/name.py` per line; `numpy/*` -> extra_numpy/).
#
# Our own harness here (conftest.py, _compat_*.py, extra_numpy/{conftest,__init__}.py, the
# baselines) is tracked in git and untouched. Run via `make fetch-tests` (or directly).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default to the pinned ref; HF_UPSTREAM_REF overrides it (the scheduled "latest" jobs set it to
# a newer tag to fetch a different version's tests against the pinned baselines).
REF="${HF_UPSTREAM_REF:-$(tr -d '[:space:]' < "$HERE/UPSTREAM_REF")}"
REPO="${HF_UPSTREAM_REPO:-HypothesisWorks/hypothesis}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo ">>> fetching $REPO parity tests @ $REF"
curl -fsSL --retry 3 "https://github.com/$REPO/archive/$REF.tar.gz" -o "$TMP/src.tgz"
tar -xzf "$TMP/src.tgz" -C "$TMP"
SRC="$(find "$TMP" -maxdepth 3 -type d -path '*/hypothesis-python/tests' -print -quit)"
[ -n "$SRC" ] || { echo "ERROR: no hypothesis-python/tests/ in the $REF archive" >&2; exit 1; }

n=0
skipped=0
while IFS= read -r rel; do
  [ -n "$rel" ] || continue
  src="$SRC/$rel"
  if [ ! -f "$src" ]; then
    # HF_FETCH_TOLERANT (the scheduled latest-tests scenario fetches the PINNED file list at a
    # NEWER ref, where upstream may have renamed/removed files): warn + skip instead of failing,
    # so the scenario still runs the tests that DO exist. Default (pinned ref): hard error.
    if [ -n "${HF_FETCH_TOLERANT:-}" ]; then
      echo "SKIP (absent @ $REF): $rel" >&2
      skipped=$((skipped + 1))
      continue
    fi
    echo "ERROR: $rel absent in upstream @ $REF (bump/adjust UPSTREAM_FILES)" >&2
    exit 1
  fi
  case "$rel" in
    numpy/*) dest="$HERE/extra_numpy/$(basename "$rel")" ;;
    *) dest="$HERE/$(basename "$rel")" ;;
  esac
  cp "$src" "$dest"
  n=$((n + 1))
done < "$HERE/UPSTREAM_FILES"

echo ">>> fetched $n parity test files into tests/hypothesis_compat/ at $REF (skipped $skipped absent)"
