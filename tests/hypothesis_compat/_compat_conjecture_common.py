"""Reimplementation of upstream `tests.conjecture.common` helpers used by the
copied cover files. Only the pieces our compat suite needs."""

from __future__ import annotations

import contextlib
import math
from typing import Any

import hypothesis.strategies as _st
from hypothesis.internal.conjecture.choice import ChoiceNode
from hypothesis.internal.floats import SMALLEST_SUBNORMAL
from hypothesis.internal.intervalsets import IntervalSet

# full-unicode interval set, the default "any character" string constraint.
_FULL_INTERVALS = IntervalSet([(0, 0x10FFFF)])
_COLLECTION_MAX = 100_000


def _node_for(value: Any) -> ChoiceNode:
    """A conjecture ChoiceNode wrapping `value`, with default constraints for its
    choice type. Mirrors upstream tests.conjecture.common's node construction; only
    `.value` is exercised by the database round-trip test."""
    if isinstance(value, bool):
        ctype, constraints = "boolean", {"p": 0.5}
    elif isinstance(value, int):
        ctype, constraints = "integer", {
            "min_value": None,
            "max_value": None,
            "weights": None,
            "shrink_towards": 0,
        }
    elif isinstance(value, float):
        ctype, constraints = "float", {
            "min_value": -math.inf,
            "max_value": math.inf,
            "allow_nan": True,
            "smallest_nonzero_magnitude": SMALLEST_SUBNORMAL,
        }
    elif isinstance(value, bytes):
        ctype, constraints = "bytes", {"min_size": 0, "max_size": _COLLECTION_MAX}
    elif isinstance(value, str):
        ctype, constraints = "string", {
            "intervals": _FULL_INTERVALS,
            "min_size": 0,
            "max_size": _COLLECTION_MAX,
        }
    else:
        raise TypeError(f"no choice type for {value!r}")
    return ChoiceNode(type=ctype, value=value, constraints=constraints, was_forced=False)


def nodes_inline(*values: Any) -> tuple[ChoiceNode, ...]:
    """Build a tuple of ChoiceNodes directly from literal choice values."""
    return tuple(_node_for(v) for v in values)


def nodes() -> Any:
    """A strategy producing a single ChoiceNode of a random choice type (used inside
    `lists(nodes())`; delegates to real hypothesis generation via the compat fallback)."""
    return _st.one_of(
        _st.integers().map(_node_for),
        _st.floats(allow_nan=True, allow_infinity=True).map(_node_for),
        _st.booleans().map(_node_for),
        _st.binary().map(_node_for),
        _st.text().map(_node_for),
    )


@contextlib.contextmanager
def buffer_size_limit(limit: int):
    """Constrain the engine's per-example buffer-size limit (max_length), so examples whose
    choice sequence exceeds `limit` bytes overrun — letting the engine report 'too large to
    finish generating' (test_notes_high_overrun_rates_in_unsatisfiable_error)."""
    from hypothesis_fast import _engine

    _engine.set_buffer_limit(limit)
    try:
        yield
    finally:
        _engine.clear_buffer_limit()


def float_constr(
    min_value: float = -math.inf,
    max_value: float = math.inf,
    allow_nan: bool = True,
    smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL,
) -> dict[str, Any]:
    """Build a FloatConstraints dict (the shape hypothesis's float clamper /
    choice_permitted expect), matching internal.conjecture float-constraint specs."""
    return {
        "min_value": min_value,
        "max_value": max_value,
        "forced": None,
        "allow_nan": allow_nan,
        "smallest_nonzero_magnitude": smallest_nonzero_magnitude,
    }
