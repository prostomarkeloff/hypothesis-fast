"""Reimplementation of the portable subset of hypothesis's `tests.common.utils`.
Injected as `tests.common.utils` by the conftest. Symbols that depend on
hypothesis internals (observability, database, reporting) are provided as skips
or no-ops so copied upstream files import cleanly; tests that actually rely on
those will skip/xfail rather than error at import time."""

from __future__ import annotations

import contextlib
import enum
import functools
import sys
from io import StringIO

import pytest

from hypothesis import Phase as _Phase

# Upstream's tests.common.utils.no_shrink: every phase EXCEPT shrink (so a test still
# generates + finds a failure, just doesn't minimise it). Was wrongly `()` (which means
# "no phases at all" -> SkipTest("ran no examples")).
no_shrink = tuple(p for p in _Phase if p != _Phase.shrink)


def fails_with(expected, *, match=None):
    def accept(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            with pytest.raises(expected, match=match):
                f(*args, **kwargs)

        return inner

    return accept


fails = fails_with(AssertionError)


def flaky(max_runs, min_passes):
    assert 0 < min_passes <= max_runs <= 50

    def accept(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            runs = passes = 0
            while passes < min_passes:
                runs += 1
                try:
                    func(*args, **kwargs)
                    passes += 1
                except BaseException:
                    if runs >= max_runs:
                        raise

        return inner

    return accept


def counts_calls(func):
    @functools.wraps(func)
    def _inner(*args, **kwargs):
        _inner.calls += 1
        return func(*args, **kwargs)

    _inner.calls = 0
    return _inner


@contextlib.contextmanager
def capture_out():
    old = sys.stdout
    try:
        new = StringIO()
        sys.stdout = new
        yield new
    finally:
        sys.stdout = old


def run_test_for_falsifying_example(test_fn):
    with pytest.raises(AssertionError) as err:
        test_fn()
    return "\n".join(getattr(err.value, "__notes__", [])).strip()


def assert_output_contains_failure(output, test, **kwargs):
    assert test.__name__ + "(" in output
    for k, v in kwargs.items():
        assert f"{k}={v!r}" in output, (f"{k}={v!r}", output)


class Why(enum.Enum):
    symbolic_outside_context = "symbolic"
    nested_given = "nested @given"
    undiscovered = "undiscovered"
    other = "other"


def xfail_on_crosshair(why, /, *, strict=True, as_marks=False):
    # we never run the crosshair backend, so this is a pass-through.
    if as_marks:
        return ()
    return lambda fn: fn


def skipif_threading(f):
    return f


def xfail_if_gil_disabled(f):
    # GIL is enabled on our CPython build, so this is a pass-through (upstream xfails
    # only on free-threaded builds).
    return f


def skipif_emscripten(f):
    return pytest.mark.skipif(sys.platform == "emscripten", reason="no threads")(f)


def checks_deprecated_behaviour(func):
    # mirror real hypothesis: wrap the body so it must emit a HypothesisDeprecationWarning.
    @functools.wraps(func)
    def _inner(*args, **kwargs):
        with validate_deprecation():
            return func(*args, **kwargs)

    return _inner


@contextlib.contextmanager
def temp_registered(type_, strat_or_factory):
    from hypothesis.strategies._internal.types import _global_type_lookup
    from hypothesis.strategies._internal.utils import clear_strategy_cache

    from hypothesis_fast import native_strategies as _ns
    from hypothesis_fast import strategies as st

    had = type_ in _global_type_lookup
    prev = _global_type_lookup.get(type_)
    # snapshot our own registry too — register_type_strategy below populates both
    # _global_type_lookup AND st._TYPE_REGISTRY (identity lookup), so we have to
    # restore both on exit to avoid leaking stale registrations into the next
    # test (which would re-resolve to our strategy instead of the default).
    our_had = type_ in st._TYPE_REGISTRY
    our_prev = st._TYPE_REGISTRY.get(type_)
    # And the NATIVE registry consulted by the Rust from_type / subclass walk under
    # HP_NATIVE_DEFAULT — register_type_strategy populates it there, and if we don't
    # restore it the leaked registration makes unrelated later tests flaky (the walk
    # scans the whole registry). This is the registry the native engine actually reads.
    nat_had = type_ in _ns._NATIVE_TYPE_REGISTRY
    nat_prev = _ns._NATIVE_TYPE_REGISTRY.get(type_)
    nat_user = type_ in _ns._USER_REGISTERED
    # native register_type_strategy ALSO keys on get_origin(type_) (so typing.Sequence
    # resolves Sequence[int] whose origin is abc.Sequence) — snapshot+restore that too.
    import typing as _typing

    _origin = _typing.get_origin(type_)
    if _origin is type_:
        _origin = None
    if _origin is not None:
        oh = _origin in _ns._NATIVE_TYPE_REGISTRY
        op = _ns._NATIVE_TYPE_REGISTRY.get(_origin)
        ou = _origin in _ns._USER_REGISTERED
    st.register_type_strategy(type_, strat_or_factory)
    # hypothesis's @cacheable strategy builders memoize by their argument tuple —
    # `from_type(list[int])` inside the `with` caches a list-of-sentinels strategy
    # that would persist after exit and contaminate the next call. Clearing the
    # strategy cache on enter AND exit guarantees fresh resolution both ways.
    clear_strategy_cache()
    try:
        yield
    finally:
        _global_type_lookup.pop(type_, None)
        if had:
            _global_type_lookup[type_] = prev
        st._TYPE_REGISTRY.pop(type_, None)
        if our_had:
            st._TYPE_REGISTRY[type_] = our_prev
        _ns._NATIVE_TYPE_REGISTRY.pop(type_, None)
        if nat_had:
            _ns._NATIVE_TYPE_REGISTRY[type_] = nat_prev
        _ns._USER_REGISTERED.discard(type_)
        if nat_user:
            _ns._USER_REGISTERED.add(type_)
        if _origin is not None:
            _ns._NATIVE_TYPE_REGISTRY.pop(_origin, None)
            if oh:
                _ns._NATIVE_TYPE_REGISTRY[_origin] = op
            _ns._USER_REGISTERED.discard(_origin)
            if ou:
                _ns._USER_REGISTERED.add(_origin)
        clear_strategy_cache()
        # The native registry was restored by direct pops above (not via
        # register_type_strategy), so explicitly drop the native resolution caches keyed on
        # it — else a resolution computed inside the `with` leaks into the next test.
        _ns._GENERIC_SUBSET_CACHE = None
        _ns._e._clear_resolution_caches()


# --- additional symbols upstream framework tests import (lazy real-internal deps) ---


class NotDeprecated(Exception):
    pass


class ExcInfo:
    pass


@contextlib.contextmanager
def validate_deprecation():
    import warnings

    from hypothesis.errors import HypothesisDeprecationWarning

    try:
        warnings.simplefilter("always", HypothesisDeprecationWarning)
        with warnings.catch_warnings(record=True) as w:
            yield
    finally:
        warnings.simplefilter("error", HypothesisDeprecationWarning)
        if not any(e.category == HypothesisDeprecationWarning for e in w):
            raise NotDeprecated(f"Expected a deprecation warning but got {[e.category for e in w]!r}")


@contextlib.contextmanager
def raises_warning(expected_warning, match=None):
    import warnings

    with pytest.raises(expected_warning, match=match) as r, warnings.catch_warnings():
        warnings.simplefilter("error", category=expected_warning)
        yield r


@contextlib.contextmanager
def capture_observations(*, choices=None):
    from hypothesis.internal import observability
    from hypothesis.internal.observability import (
        add_observability_callback,
        remove_observability_callback,
    )

    ls: list = []
    add_observability_callback(ls.append)
    old = None
    if choices is not None:
        old = observability.OBSERVABILITY_CHOICES
        observability.OBSERVABILITY_CHOICES = choices
    try:
        yield ls
    finally:
        remove_observability_callback(ls.append)
        if choices is not None:
            observability.OBSERVABILITY_CHOICES = old


def all_values(db):
    return {v for vs in db.data.values() for v in vs}


def non_covering_examples(database):
    return {v for k, vs in database.data.items() if not k.endswith(b".pareto") for v in vs}


def assert_falsifying_output(test, example_type="Falsifying", expected_exception=AssertionError, **kwargs):
    with capture_out() as out:
        if expected_exception is None:
            test()
            msg = ""
        else:
            with pytest.raises(expected_exception) as exc_info:
                test()
            notes = "\n".join(getattr(exc_info.value, "__notes__", []))
            msg = str(exc_info.value) + "\n" + notes
    output = out.getvalue() + msg
    assert f"{example_type} example:" in output
    assert_output_contains_failure(output, test, **kwargs)


def skipif_time_unpatched(f):
    # our compat suite never monkeypatches the clock, so upstream tests that rely on a
    # patched time (body does time.sleep(1000), "takes forever" otherwise) would hang —
    # always skip them.
    return pytest.mark.skipif(
        True, reason="time is not patched in the compat suite (sleep-based test would hang)"
    )(f)


@contextlib.contextmanager
def restore_recursion_limit():
    original = sys.getrecursionlimit()
    try:
        yield
    finally:
        sys.setrecursionlimit(original)


def run_concurrently(function, *, n):
    from threading import Barrier, Thread

    barrier = Barrier(n)

    def run():
        barrier.wait()
        function()

    threads = [Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)


def wait_for(condition, *, timeout=1, interval=0.01):
    import math
    import time

    for _ in range(math.ceil(timeout / interval)):
        if condition():
            return
        time.sleep(interval)
    raise Exception("timed out waiting for condition")


try:
    from hypothesis.internal.floats import next_down as _next_down

    PYTHON_FTZ = _next_down(sys.float_info.min) == 0.0
except Exception:
    PYTHON_FTZ = False
