"""IntervalSet — re-exported from the Rust engine.

The implementation lives in Rust (`hypothesis_fast._engine`) because it is on
the string generation/shrinking hot path. This module exists so that
`hypothesis.internal.intervalsets` can be aliased to it.
"""

from __future__ import annotations

from typing import TypeAlias

from hypothesis_fast._engine import IntervalSet

IntervalsT: TypeAlias = tuple[tuple[int, int], ...]

__all__ = ["IntervalSet", "IntervalsT"]
