#!/usr/bin/env bash
# probe_files.sh <file1.py> [file2.py ...] — fast targeted run of a subset of cover
# files. Use this for tight feedback loops (5–10s) instead of the full suite.
#
#   probe_files.sh test_map.py test_searchstrategy.py
set -euo pipefail
ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
cd "$ROOT"
if [ $# -lt 1 ]; then
  echo "usage: probe_files.sh <test_file.py> [test_file.py ...]" >&2
  exit 1
fi

# Build the path list.
PATHS=""
for f in "$@"; do
  PATHS+="tests/hypothesis_compat/$f "
done

WORKERS="${WORKERS:-2}"
TIMEOUT="${TIMEOUT:-20}"
ADDR="${ADDR:-/tmp/hp-probe-files.sock}"

make kill > /dev/null 2>&1 || true
sleep 1
PYTEST_FAST_WATCH_DIRS=python,tests \
  PYTEST_ADDOPTS="-p no:hypothesispytest -p no:cacheprovider -p pytester --timeout=60 --timeout-method=thread $PATHS" \
  timeout "$TIMEOUT" uv run pytest-fast --workers "$WORKERS" --address "$ADDR" \
  > /tmp/hp-probe.log 2>&1 || true

awk '/results :|^    \?|^    ✗/' /tmp/hp-probe.log
