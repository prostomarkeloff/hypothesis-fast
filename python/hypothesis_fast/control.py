"""Runtime control functions usable inside a @given test body."""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Sequence
from typing import Any, NoReturn
from weakref import WeakKeyDictionary

from .errors import InvalidArgument, UnsatisfiedAssumption

# Note matching hypothesis #3819: when sampled_from() is given a collection of
# strategies (a likely one_of() mistake) and a TypeError mentioning SearchStrategy
# escapes the test, this note is attached to the error.
_SAMPLED_FROM_3819_TEMPLATE = (
    "sampled_from was given a collection of strategies: {!r}. Was one_of intended?"
)


class _ExampleContext:
    """Per-example state pushed by the @given driver around each test body.

    Holds target() labels (for out-of-test / duplicate-label detection) and the
    pending #3819 note (set when a sampled_from-of-strategies yields a strategy).
    """

    __slots__ = ("labels", "sampled_from_message", "events", "notes")

    def __init__(self) -> None:
        self.labels: set[str] = set()
        self.sampled_from_message: str | None = None
        # event() recordings for this example: {value_str: payload}, like hypothesis's
        # ConjectureData.events. Drained into per-test-case statistics.
        self.events: dict[str, Any] = {}
        # note() recordings for this example, attached to a failing exception by the runner.
        self.notes: list[str] = []


# Per-thread context: two threads running @given concurrently (e.g.
# test_run_given_concurrently in upstream's test_threading) must each see their own
# stack — otherwise thread B's enter sees thread A's frame on a shared list and
# fires HealthCheck.nested_given. Reuse context objects across examples within a
# thread — the per-example hot path (one enter+exit per generated case) avoids a
# fresh _ExampleContext()+set() each time, matters for the x200-x300 perf target.
_TLS = threading.local()


def _tls_stack() -> list[_ExampleContext]:
    stack = getattr(_TLS, "stack", None)
    if stack is None:
        stack = []
        _TLS.stack = stack
    return stack


def _tls_pool() -> list[_ExampleContext]:
    pool = getattr(_TLS, "pool", None)
    if pool is None:
        pool = []
        _TLS.pool = pool
    return pool


def _enter_test_context() -> None:
    pool = _tls_pool()
    if pool:
        ctx = pool.pop()
        if ctx.labels:
            ctx.labels.clear()
        if ctx.events:
            ctx.events.clear()
        if ctx.notes:
            ctx.notes.clear()
        ctx.sampled_from_message = None
    else:
        ctx = _ExampleContext()
    _tls_stack().append(ctx)


def _exit_test_context() -> None:
    stack = _tls_stack()
    if stack:
        _tls_pool().append(stack.pop())


def _current_context() -> _ExampleContext | None:
    stack = getattr(_TLS, "stack", None)
    return stack[-1] if stack else None


def _context_stack_nonempty() -> bool:
    """True iff this thread is currently inside a @given example. Read by core.py
    for the HealthCheck.nested_given guard — must be per-thread (see above)."""
    stack = getattr(_TLS, "stack", None)
    return bool(stack)


def record_sampled_from_strategies(elements: Sequence[Any]) -> None:
    """Remember that a sampled_from-of-strategies was drawn this example (#3819)."""
    ctx = _current_context()
    if ctx is not None:
        ctx.sampled_from_message = _SAMPLED_FROM_3819_TEMPLATE.format(tuple(elements))


def maybe_add_sampled_from_note(exc: BaseException) -> None:
    """Attach the #3819 note if a strategy-collection was sampled and the error
    (a TypeError) mentions SearchStrategy — mirrors hypothesis core.py."""
    ctx = _current_context()
    if ctx is None or ctx.sampled_from_message is None:
        return
    if "SearchStrategy" not in str(exc):
        return
    msg = ctx.sampled_from_message
    if msg not in getattr(exc, "__notes__", []) and hasattr(exc, "add_note"):
        exc.add_note(msg)


def _record_native_3819(elements: Sequence[Any]) -> None:
    """Native #3819: remember that a sampled_from-of-strategies was drawn this example.
    Stored on a thread-local (NOT the per-example `_ExampleContext`) because the native
    engine draws the @given args in Rust BEFORE the test-body context is pushed — so the
    `indirect` case (a sampled_from nested in a list arg) records here and the runner
    reads it after the body raises."""
    _TLS.native_sf3819 = _SAMPLED_FROM_3819_TEMPLATE.format(tuple(elements))


def _native_3819_note(exc: BaseException) -> None:
    """Native #3819: attach the note if a strategy-collection was sampled this example and
    the error (a TypeError) mentions SearchStrategy."""
    msg = getattr(_TLS, "native_sf3819", None)
    if msg is None or "SearchStrategy" not in str(exc):
        return
    if msg not in getattr(exc, "__notes__", []) and hasattr(exc, "add_note"):
        exc.add_note(msg)


def _clear_native_3819() -> None:
    _TLS.native_sf3819 = None


def set_engine_active(active: bool) -> None:
    """Marked True by the native engine for the duration of a run, so draw-time assume()/
    reject() (called before the per-example context is pushed) aren't mistaken for
    out-of-test usage."""
    _TLS.engine_active = active


def currently_drawing() -> bool:
    """True while our native engine is generating values (between set_engine_active
    True/False, or inside a pushed example context). `.example()` is forbidden then."""
    return getattr(_TLS, "engine_active", False) or _current_context() is not None


def _out_of_any_test_context() -> bool:
    """True when neither our engine nor real hypothesis is running a test — used to warn
    that assume()/reject() outside a property-based test is deprecated."""
    if _current_context() is not None or getattr(_TLS, "engine_active", False):
        return False
    hc = sys.modules.get("hypothesis.control")
    if hc is None:
        return True
    try:
        return not hc.currently_in_test_context()
    except Exception:  # noqa: BLE001 - be conservative; only used to gate a warning
        return False


def _warn_outside_test(name: str) -> None:
    import warnings

    from .errors import HypothesisDeprecationWarning

    warnings.warn(
        f"Using `{name}` outside a property-based test is deprecated",
        HypothesisDeprecationWarning,
        stacklevel=3,
    )


def assume(condition: object) -> bool:
    """Discard the current example unless `condition` is truthy."""
    if _out_of_any_test_context():
        _warn_outside_test("assume")
    if not condition:
        raise UnsatisfiedAssumption
    return True


def reject() -> NoReturn:
    """Unconditionally discard the current example."""
    if _out_of_any_test_context():
        _warn_outside_test("reject")
    raise UnsatisfiedAssumption


def note(value: object) -> None:
    """Record a note for the current example; the @given runner attaches the current
    example's notes to a failing exception (so failures report them). When verbose, also
    report it immediately. No-op outside a running example."""
    ctx = _current_context()
    if ctx is None:
        return
    s = value if isinstance(value, str) else repr(value)
    ctx.notes.append(s)


def current_notes() -> list[str]:
    """The notes recorded for the current example (empty if none / no example running)."""
    ctx = _current_context()
    return ctx.notes if ctx is not None else []


_events_to_strings: "WeakKeyDictionary[Any, str]" = WeakKeyDictionary()


def _event_to_string(event: Any, allowed_types: type | tuple[type, ...] = str) -> Any:
    if isinstance(event, allowed_types):
        return event
    try:
        return _events_to_strings[event]
    except (KeyError, TypeError):
        pass
    result = str(event)
    try:
        _events_to_strings[event] = result
    except TypeError:
        pass
    return result


def event(value: str, payload: str | int | float = "") -> None:
    """Record an event for reporting in the statistics output. Values/payloads are
    stringified (with a cache keyed by hash/eq, so equal objects stringify once)."""
    ctx = _current_context()
    if ctx is None:
        # Our engine isn't driving this body — defer to real hypothesis (live driver),
        # which records into its own data (and raises if genuinely outside a test).
        from .strategies import _real_hypothesis

        _real_hypothesis().event(value, payload)
        return
    payload = _event_to_string(payload, (str, int, float))
    value = _event_to_string(value)
    ctx.events[value] = payload


def drain_events() -> list[str]:
    """The current example's events as sorted `value` / `value: payload` strings."""
    ctx = _current_context()
    if ctx is None or not ctx.events:
        return []
    return sorted(k if v == "" else f"{k}: {v}" for k, v in ctx.events.items())


def target(observation: int | float, *, label: str = "") -> int | float:
    """Register an optimisation target. Returns `observation` unchanged.

    Targeted *shrinking* is not yet implemented, but the validation contract is:
    target() must be called inside a running test, with a finite real number, and
    each label (default ``""``) at most once per example.
    """
    ctx = _current_context()
    if ctx is None and not getattr(_TLS, "engine_active", False):
        # Neither a pushed example context nor an active native run: the test fell back
        # to real hypothesis (live driver — its target() works), or we're genuinely
        # outside any test (its target() then raises InvalidArgument).
        from .strategies import _real_hypothesis

        return _real_hypothesis().target(observation, label=label)
    # `target()` is also valid mid-draw — a strategy/composite may call it during
    # generation (e.g. hypothesmith's auto_target). The native engine draws @given args
    # in Rust BEFORE the per-example body context is pushed, so ctx is None then; accept
    # it as long as the engine is running and record the score.
    if isinstance(observation, bool) or not isinstance(observation, (int, float)):
        raise InvalidArgument(f"observation={observation!r} must be an int or float")
    if math.isnan(observation) or math.isinf(observation):
        raise InvalidArgument(f"observation={observation!r} must be a finite number")
    if not isinstance(label, str):
        raise InvalidArgument(f"label={label!r} must be a string")
    # Label-dedup is per-example and only enforceable with a pushed context (mid-draw
    # there's no example context yet, hence nothing to dedup against).
    if ctx is not None:
        if label in ctx.labels:
            raise InvalidArgument(
                f"Tried to call target({observation!r}, label={label!r}), but "
                "this label has already been used for this test"
            )
        ctx.labels.add(label)
    # Accumulate the best (highest) score per label across the run, for the statistics
    # "targets" report. Only active while a collector is gathering statistics.
    targets = getattr(_TLS, "targets", None)
    if targets is not None:
        score = float(observation)
        prev = targets.get(label)
        targets[label] = score if prev is None else max(prev, score)
    return observation


def start_target_collection() -> None:
    """Begin accumulating target() scores for statistics (per run)."""
    _TLS.targets = {}


def drain_target_collection() -> dict[str, float]:
    """The best target() score per label since start_target_collection(); clears state."""
    targets = getattr(_TLS, "targets", None)
    _TLS.targets = None
    return targets or {}
