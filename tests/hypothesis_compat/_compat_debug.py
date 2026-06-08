"""Reimplementation of hypothesis's `tests.common.debug` helpers on top of the
hypothesis_fast public API + engine. Injected as `tests.common.debug` by the
conftest so copied upstream test files import it unchanged."""

from __future__ import annotations

from typing import Any

import secrets

from hypothesis_fast import _engine
from hypothesis_fast import given as _given, settings as _settings
from hypothesis_fast.errors import (
    NoSuchExample,
    UnsatisfiedAssumption,
    Unsatisfiable,
)


def _is_native(strategy) -> bool:
    return isinstance(strategy, _engine.SearchStrategy)


def _native_find(definition, condition, settings, default_max=500, shrink=True):
    """Run `definition` through the native engine, raising on a match so the engine
    shrinks to the minimal example satisfying `condition`. Returns (matched, value).

    `default_max` is the example budget when no settings are supplied. find_any uses
    a large budget because it stops at the first match (so the cost is only paid when
    the target is rare — e.g. a specific mixed-case string under IGNORECASE — not on
    the common case), whereas minimal()'s conditions are usually easy to hit.

    `shrink=False` (find_any) skips minimisation entirely — we only need ANY matching
    example, so shrinking a large find (e.g. a min_size=50 collection) is wasted work
    and can be very slow / memory-heavy. minimal() keeps shrink=True for a real minimum."""
    max_ex = int(settings.max_examples) if settings is not None else default_max

    # In verbose mode, report each trial ("Trying example: ...") just like the @given
    # runner does — the real engine emits these during generation and shrinking, and
    # tests.common.debug.minimal() inherits that (test_includes_progress_in_verbose_mode).
    from hypothesis_fast.settings import Verbosity

    _verbose = (
        settings is not None
        and getattr(settings, "verbosity", None) is not None
        and settings.verbosity >= Verbosity.verbose
    )
    if _verbose:
        from hypothesis_fast.core import _report

        def body(x):
            _report(f"Trying example: {x!r}")
            if _condition_holds(condition, x):
                raise AssertionError("__match__")
    else:

        def body(x):
            if _condition_holds(condition, x):
                raise AssertionError("__match__")

    from hypothesis_fast.control import set_engine_active

    # Mark the engine active for the duration so .example() called inside a drawn
    # strategy is recognised as nested (test_example_inside_strategy).
    set_engine_active(True)
    try:
        res = _engine.run_native(
            body, [definition], max_ex, secrets.randbits(64),
            max_shrinks=(500 if shrink else 0),
        )
    finally:
        set_engine_active(False)
    if not res:
        return (False, None)
    # run_native returns a list of (args_tuple, exc) per distinct bug; the minimal
    # value is the sole drawn arg of the first (and here only) failing example.
    return (True, res[0][0][0])


def _condition_holds(condition, value) -> bool:
    """Evaluate a find/minimal condition, treating assume()/reject() inside it as
    'not a match' (mirrors hypothesis, where a discarded example isn't a hit)."""
    try:
        return bool(condition(value))
    except UnsatisfiedAssumption:
        return False


class _Found(Exception):
    pass


def _build_context():
    """The current real-hypothesis build context, if we're inside a running test."""
    try:
        from hypothesis.control import _current_build_context

        return _current_build_context.value
    except Exception:
        return None


def _as_real(strategy):
    from hypothesis_fast.strategies import _real_strategies, _to_hyp

    return _to_hyp(strategy, _real_strategies())


def minimal(definition, condition=lambda x: True, settings=None):
    """Return the minimal example of `definition` satisfying `condition`.

    For engine-native strategies, run the strategy through the engine with a body
    that raises when the condition holds; the engine shrinks to the minimum and
    the @given driver reproduces it (recording the shrunk value). For fallback
    strategies, delegate to real hypothesis's find() for hypothesis-quality minima.
    """
    from hypothesis_fast.strategies import (
        _is_supported,
        _real_hypothesis,
        _real_strategies,
        _to_hyp,
    )

    definition.validate()
    if _is_native(definition):
        matched, value = _native_find(definition, condition, settings)
        if not matched:
            raise Unsatisfiable(
                f"Could not find any examples from {definition!r} that satisfied condition"
            )
        return value
    if not _is_supported(definition):
        real = _real_hypothesis()
        real_strat = _to_hyp(definition, _real_strategies())
        kwargs = {}
        if settings is not None:
            kwargs["settings"] = real.settings(max_examples=int(settings.max_examples))
        return real.find(real_strat, condition, **kwargs)

    found: list[Any] = []

    @_given(definition)
    def inner(x):
        if condition(x):
            found.append(x)
            raise _Found

    se = settings if settings is not None else _settings(max_examples=500)
    inner = se(inner)
    try:
        inner()
    except _Found:
        return found[-1]
    raise Unsatisfiable(
        f"Could not find any examples from {definition!r} that satisfied condition"
    )


def find_any(definition, condition=lambda _: True, settings=None):
    definition.validate()
    if _is_native(definition):
        matched, value = _native_find(
            definition, condition, settings, default_max=5000, shrink=False
        )
        if not matched:
            raise NoSuchExample(f"No example of {definition!r} satisfied condition")
        return value
    ctx = _build_context()
    if ctx is None:
        # outside a running test: use real hypothesis's GUIDED find — random .example()
        # sampling flaked for rare conditions (e.g. "is an ASCII char", a specific
        # mixed-case string). find() returns the minimal match, which is still "any".
        from hypothesis_fast.strategies import (
            _real_hypothesis,
            _real_strategies,
            _to_hyp,
        )

        real = _real_hypothesis()
        real_strat = _to_hyp(definition, _real_strategies())
        kwargs = {}
        if settings is not None:
            kwargs["settings"] = real.settings(max_examples=int(settings.max_examples))
        try:
            return real.find(real_strat, condition, **kwargs)
        except (Unsatisfiable, NoSuchExample) as exc:
            raise NoSuchExample(
                f"No example of {definition!r} satisfied condition"
            ) from exc
    # inside a running test: draw from the live data (guided find can't run nested)
    attempts = 1000 if settings is None else max(1000, int(settings.max_examples))
    real_strat = _as_real(definition)
    for _ in range(attempts):
        s = ctx.data.draw(real_strat)
        if _condition_holds(condition, s):
            return s
    raise NoSuchExample(f"No example of {definition!r} satisfied condition in {attempts} tries")


def assert_all_examples(strategy, predicate, settings=None):
    if _is_native(strategy):
        max_ex = int(settings.max_examples) if settings is not None else 100

        def body(s):
            assert predicate(s), f"Found {s!r} using strategy {strategy} which does not match"

        from hypothesis_fast.control import set_engine_active

        # Mark the engine active for the duration so assume()/reject()/.example() called
        # inside a drawn strategy's do_draw (e.g. basic_indices) are recognised as nested
        # rather than warning "outside a property-based test" (test_basic_indices_*_warn).
        set_engine_active(True)
        try:
            res = _engine.run_native(body, [strategy], max_ex, secrets.randbits(64))
        finally:
            set_engine_active(False)
        if res:
            raise AssertionError(res[0][1])
        return
    ctx = _build_context()
    if ctx is not None:
        real_strat = _as_real(strategy)
        for _ in range(20):
            s = ctx.data.draw(real_strat)
            assert predicate(s), f"Found {s!r} using strategy {strategy} which does not match"
        return

    @_given(strategy)
    def check(s):
        assert predicate(s), f"Found {s!r} using strategy {strategy} which does not match"

    if settings is not None:
        check = settings(check)
    check()


def assert_simple_property(strategy, predicate, settings=None):
    # map our FailedHealthCheck.filter_too_much → Unsatisfiable to mirror real
    # hypothesis (callers that catch Unsatisfiable expect this exception type).
    try:
        assert_all_examples(strategy, predicate, _settings(max_examples=15))
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ == "FailedHealthCheck":
            raise Unsatisfiable(str(exc)) from None
        raise


def check_can_generate_examples(strategy, settings=None):
    assert_simple_property(strategy, lambda _: True)


def assert_all_examples_then_raise_unsatisfiable_as_real_would(strategy, predicate, settings=None):
    """Like assert_all_examples but maps our FailedHealthCheck.filter_too_much →
    Unsatisfiable. Real hypothesis raises Unsatisfiable when a strategy rejects
    every example via assume(); we raise FailedHealthCheck. Callers of
    `assert_simple_property`/`assert_all_examples` that expect Unsatisfiable get
    the real-hypothesis-compatible exception type."""
    try:
        assert_all_examples(strategy, predicate, settings)
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ == "FailedHealthCheck":
            raise Unsatisfiable(str(exc)) from None
        raise


def assert_no_examples(strategy, condition=lambda _: True):
    try:
        assert_all_examples(strategy, lambda v: not condition(v))
    except (Unsatisfiable, NoSuchExample):
        pass
    except Exception as exc:  # noqa: BLE001
        # our engine raises FailedHealthCheck.filter_too_much rather than
        # Unsatisfiable when a strategy rejects every example (e.g. assume(False)
        # inside .map). Treat that as "no examples", matching real hypothesis's
        # observable behaviour from the caller's point of view.
        if exc.__class__.__name__ == "FailedHealthCheck":
            return
        raise
