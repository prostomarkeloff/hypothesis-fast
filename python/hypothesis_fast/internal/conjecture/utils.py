"""Conjecture helper utilities (subset).

Grows as later phases need Sampler / many() / label helpers. For now only the
pieces `choice.py` requires.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def identity(v: T) -> T:
    return v
