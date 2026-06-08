"""hypothesis_fast — a native-Rust, drop-in-compatible reimplementation of the
Hypothesis property-based testing engine.

Drop-in usage::

    import hypothesis_fast as hypothesis
    from hypothesis_fast import given, settings
    from hypothesis_fast import strategies as st

    @given(st.integers(), st.integers())
    def test_commutative(a, b):
        assert a + b == b + a
"""

from __future__ import annotations

import importlib

from . import native_strategies
from .control import assume, event, note, reject, target
from .core import example, find, given, is_hypothesis_test, reproduce_failure, seed
from .errors import (
    DidNotReproduce,
    HypothesisException,
    InvalidArgument,
    NoSuchExample,
    Unsatisfiable,
    UnsatisfiedAssumption,
)
from .settings import HealthCheck, Phase, Verbosity, settings

# The native (all-Rust) frontend is the public default `strategies`. The legacy `_spec`
# frontend is kept loaded in sys.modules (not bound here) so native_strategies'
# `__getattr__` can delegate there for any name it doesn't yet implement natively.
importlib.import_module(f"{__name__}.strategies")
strategies = native_strategies

__version__ = "0.0.1"


def __getattr__(name: str) -> object:
    """Drop-in fallback: any public hypothesis name we don't (yet) define
    natively resolves to the real hypothesis package. Names we implement here
    shadow this. Without hypothesis installed, the name simply doesn't exist."""
    from .strategies import _real_hypothesis

    try:
        real = _real_hypothesis()
    except Exception:
        raise AttributeError(name) from None
    try:
        return getattr(real, name)
    except AttributeError:
        raise AttributeError(name) from None

__all__ = [
    "given",
    "find",
    "is_hypothesis_test",
    "example",
    "settings",
    "assume",
    "reject",
    "note",
    "event",
    "target",
    "seed",
    "reproduce_failure",
    "strategies",
    "HealthCheck",
    "Phase",
    "Verbosity",
    "HypothesisException",
    "InvalidArgument",
    "Unsatisfiable",
    "NoSuchExample",
    "UnsatisfiedAssumption",
    "DidNotReproduce",
    "__version__",
]
