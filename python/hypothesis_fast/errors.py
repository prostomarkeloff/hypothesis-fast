"""Exception types.

Core classes are always defined here. When the real `hypothesis` package is
importable we rebind them to *its* classes so exception identity matches (the
upstream test suite imports `from hypothesis.errors import ...`). Any other error
class is resolved lazily via `__getattr__`.
"""


class HypothesisException(Exception):
    """Base class for all library exceptions."""


class InvalidArgument(HypothesisException, TypeError):
    """A strategy or decorator was called with arguments that don't make sense."""


class Unsatisfiable(HypothesisException):
    """Could not find any valid example (every example was rejected)."""


class NoSuchExample(HypothesisException):
    """find() could not satisfy the predicate."""


class UnsatisfiedAssumption(BaseException):
    """Raised by assume()/reject() to discard the current example.

    Subclasses BaseException (not Exception) so a test body's `except Exception`
    never swallows it. The Rust engine recognises this by class name and treats
    it as a reject (try another example), not a failure.
    """


class DidNotReproduce(HypothesisException):
    """A reproduce_failure example failed to reproduce."""


class InvalidState(HypothesisException):
    """An operation was attempted that the current state doesn't allow (e.g.
    calling a functions()-generated callable outside its @given scope)."""


class FailedHealthCheck(HypothesisException):
    """A health check (filter_too_much, return_value, nested_given, ...) tripped."""


class DeadlineExceeded(HypothesisException):
    """A single example took longer than the configured deadline."""

    def __init__(self, runtime: object = None, deadline: object = None) -> None:
        super().__init__(f"Test took too long: {runtime!r} > deadline {deadline!r}")
        self.runtime = runtime
        self.deadline = deadline


class FlakyFailure(HypothesisException, BaseExceptionGroup):  # type: ignore[misc]
    """A test failed, but did not reproduce consistently on replay. Carries the
    divergent exceptions as a BaseExceptionGroup (members may be BaseException)."""

    def __new__(cls, msg: str, group: list[BaseException]):
        return super().__new__(cls, msg, group)


class StopTest(BaseException):
    """Raised to unwind the current example back to the engine, tagged with the
    testcounter of the data it belongs to. Subclasses BaseException so a test body's
    `except Exception` can't swallow it."""

    def __init__(self, testcounter: int) -> None:
        super().__init__(testcounter)
        self.testcounter = testcounter


class Frozen(HypothesisException):
    """Raised when an operation (draw/note/mark) is attempted on a frozen
    ConjectureData (its example has already finished)."""


class HypothesisWarning(HypothesisException, UserWarning):
    """Base class for hypothesis warnings."""


class HypothesisDeprecationWarning(HypothesisWarning, FutureWarning):
    """A deprecated hypothesis usage (e.g. @composite that never calls draw)."""


# Rebind to the real classes when available, for shared identity with the suite.
try:
    import hypothesis.errors as _real_errors

    HypothesisException = _real_errors.HypothesisException  # type: ignore[misc,assignment]
    InvalidArgument = _real_errors.InvalidArgument  # type: ignore[misc,assignment]
    HypothesisWarning = _real_errors.HypothesisWarning  # type: ignore[misc,assignment]
    HypothesisDeprecationWarning = _real_errors.HypothesisDeprecationWarning  # type: ignore[misc,assignment]
    Unsatisfiable = _real_errors.Unsatisfiable  # type: ignore[misc,assignment]
    NoSuchExample = _real_errors.NoSuchExample  # type: ignore[misc,assignment]
    UnsatisfiedAssumption = _real_errors.UnsatisfiedAssumption  # type: ignore[misc,assignment]
    DidNotReproduce = _real_errors.DidNotReproduce  # type: ignore[misc,assignment]
    InvalidState = _real_errors.InvalidState  # type: ignore[misc,assignment]
    FailedHealthCheck = _real_errors.FailedHealthCheck  # type: ignore[misc,assignment]
    DeadlineExceeded = _real_errors.DeadlineExceeded  # type: ignore[misc,assignment]
    FlakyFailure = _real_errors.FlakyFailure  # type: ignore[misc,assignment]
    StopTest = _real_errors.StopTest  # type: ignore[misc,assignment]
    Frozen = _real_errors.Frozen  # type: ignore[misc,assignment]
except Exception:  # noqa: BLE001 - hypothesis absent or import side effects
    _real_errors = None  # type: ignore[assignment]


__all__ = [
    "HypothesisException",
    "HypothesisWarning",
    "HypothesisDeprecationWarning",
    "InvalidArgument",
    "Unsatisfiable",
    "NoSuchExample",
    "UnsatisfiedAssumption",
    "DidNotReproduce",
    "InvalidState",
    "FailedHealthCheck",
    "DeadlineExceeded",
    "FlakyFailure",
    "StopTest",
    "Frozen",
]


def __getattr__(name: str) -> object:
    """Defer any other error class (DeadlineExceeded, Frozen, StopTest, ...) to
    real hypothesis.errors, captured at import (never re-imported, so no alias
    loop)."""
    import sys

    if _real_errors is None or _real_errors is sys.modules.get(__name__):
        raise AttributeError(name)
    try:
        return getattr(_real_errors, name)
    except AttributeError:
        raise AttributeError(name) from None
