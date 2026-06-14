"""@given / @example / find — the test-driving layer over the Rust engine.

`given` translates strategy arguments to engine specs and, when every strategy
and option is engine-native, runs/shrinks the test in Rust. If any strategy is
unsupported (or is a foreign real-`hypothesis` strategy), the whole test is
delegated to real `hypothesis` instead, so unsupported features keep working.

A test may also take pytest fixtures: those params stay in the wrapper's
signature so pytest still injects them, while @given params come from the engine.
Async bodies are driven to completion on an event loop per example.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import sys
import threading
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NoReturn, TypeVar

from . import _engine
from .control import (
    _clear_native_3819,
    _context_stack_nonempty,
    _enter_test_context,
    _exit_test_context,
    _native_3819_note,
    current_notes,
    drain_events,
    drain_target_collection,
    set_engine_active,
    start_target_collection,
)
from .errors import InvalidArgument, UnsatisfiedAssumption
from .settings import HealthCheck, Verbosity, apply_settings
from .settings import settings as _settings_cls
from .strategies import SearchStrategy, _to_hyp


class _ThreadLocal(threading.local):
    # Defaults to None (not AttributeError) so real entropy.deterministic_PRNG — which
    # reads `hypothesis.core.threadlocal._hypothesis_global_random` (aliased to us) — can
    # lazily create + register it. Holds the engine's master PRNG, advanced once per run.
    _hypothesis_global_random: Any = None


_OUR_THREADLOCAL = _ThreadLocal()


def _resolve_threadlocal() -> Any:
    """The RNG threadlocal, shared with real hypothesis when available so native runs (our
    engine) and fallback runs (real engine) agree on `_hypothesis_global_random` — which
    the cover suite reads as `hypothesis.core.threadlocal` (test_random_module)."""
    try:
        from .strategies import _real_hypothesis

        rh = _real_hypothesis()
        if rh is not None:
            return rh.core.threadlocal
    except Exception:  # noqa: BLE001 - real hypothesis not registered (standalone use)
        pass
    return _OUR_THREADLOCAL


def __getattr__(name: str) -> Any:
    # `core.threadlocal` resolves to the shared (real-hypothesis) RNG threadlocal.
    if name == "threadlocal":
        return _resolve_threadlocal()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _global_random() -> Any:
    """The master PRNG used to derive per-run seeds (advancing it each run, so repeated
    runs explore different examples — test_given/find_does_not_pollute_state)."""
    threadlocal = _resolve_threadlocal()
    r = threadlocal._hypothesis_global_random
    if r is None:
        import random as _random

        r = _random.Random()
        threadlocal._hypothesis_global_random = r
        try:
            from hypothesis.internal.entropy import register_random

            register_random(r)
        except Exception:  # noqa: BLE001 - real entropy not importable (standalone use)
            pass
    return r


def _deterministic_prng() -> Any:
    """Real hypothesis's deterministic_PRNG: seeds all registered Randoms (seed 0) for the
    example and restores their state afterwards, so a test's use of global/registered
    randomness is reproducible and doesn't pollute global state (test_random_module)."""
    try:
        from hypothesis.internal.entropy import deterministic_PRNG

        return deterministic_PRNG()
    except Exception:  # noqa: BLE001 - real entropy not importable
        return contextlib.nullcontext()


def _managed_randoms() -> list[Any]:
    """Real hypothesis's registered Randoms (the global `random` module, numpy's PRNG if
    imported, and any user `register_random`), weakrefs resolved. Mirrors
    get_seeder_and_restorer's lazy numpy registration. Used to pin them to seed-0 cheaply
    per example without deterministic_PRNG's per-example save/dummy-Random/hash dance."""
    try:
        from hypothesis.internal import entropy
    except Exception:  # noqa: BLE001 - real entropy not importable
        return []
    if "numpy" in sys.modules and getattr(entropy, "NP_RANDOM", None) is None:
        try:
            entropy.get_seeder_and_restorer(0)  # side effect: registers numpy's PRNG
        except Exception:  # noqa: BLE001
            pass
    out: list[Any] = []
    try:
        for ref in entropy.RANDOMS_TO_MANAGE.data.copy().values():
            r = ref()
            if r is not None:
                out.append(r)
    except Exception:  # noqa: BLE001 - registry shape changed upstream
        pass
    return out


_SEED0_STATE: Any = None


def _seed0_state() -> Any:
    """The Mersenne-Twister state of a Random seeded with 0, cached. Applying it via
    setstate() is ~2.6x faster than seed(0) (skips the key schedule) and yields an
    identical state — used to pin managed PRNGs per example cheaply."""
    global _SEED0_STATE
    if _SEED0_STATE is None:
        import random as _random_mod

        _d = _random_mod.Random()
        _d.seed(0)
        _SEED0_STATE = _d.getstate()
    return _SEED0_STATE


def _format_call(name: str, vals: tuple[Any, ...]) -> str:
    return f"{name}({', '.join(repr(v) for v in vals)})"


def _attach_draw_notes(exc: BaseException, drawn: tuple[Any, ...]) -> None:
    """Attach each st.data() interactive draw ('Draw N: ...') as a note on a failing
    exception, so failures report what was drawn (test_arbitrary_data). The log lives on
    the shared ConjectureData, so two data() args dedupe to one numbered sequence."""
    existing = list(getattr(exc, "__notes__", None) or [])
    seen_lists: list[int] = []
    for arg in drawn:
        notes = getattr(arg, "_drawn_notes", None)
        if notes is None or id(notes) in seen_lists:
            continue
        seen_lists.append(id(notes))
        for n in notes:
            if n not in existing:
                exc.add_note(n)
                existing.append(n)


_BE = TypeVar("_BE", bound=BaseException)


def _flatten_group(eg: BaseExceptionGroup[_BE]) -> list[_BE]:
    out: list[_BE] = []
    for exc in eg.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            out.extend(_flatten_group(exc))
        else:
            out.append(exc)
    return out


def _unwrap_markers_from_group(excgroup: BaseExceptionGroup) -> NoReturn:
    """Process a BaseExceptionGroup raised by a test body, mirroring hypothesis's
    unwrap_markers_from_group (test_exceptiongroup): strip Frozen; if any genuine user
    exception remains, reraise the whole group; otherwise surface a lone marker, or the
    StopTest with the lowest testcounter. Never returns — always raises."""
    from .errors import Frozen, HypothesisException, StopTest

    _frozen, non_frozen = excgroup.split(Frozen)
    # Only Frozen — reraise; the engine converts it to a StopTest once it sees the data
    # is frozen (see the runner's frozen-data handling).
    if non_frozen is None:
        raise excgroup
    _, user_exceptions = non_frozen.split(
        lambda e: isinstance(e, (StopTest, HypothesisException))
    )
    # A real user exception is present — keep the whole group for debugging context.
    if user_exceptions is not None:
        raise excgroup
    flat = _flatten_group(non_frozen)
    if len(flat) == 1:
        e = flat[0]
        raise e from e.__cause__
    stoptests, non_stoptests = non_frozen.split(StopTest)
    # A non-StopTest marker (e.g. Flaky) wins over StopTest — reraise the first.
    if non_stoptests is not None:
        e = _flatten_group(non_stoptests)[0]
        raise e from e.__cause__
    assert stoptests is not None
    raise min(_flatten_group(stoptests), key=lambda s: s.testcounter)


def _frozen_testcounter(drawn: tuple[Any, ...]) -> int | None:
    """If a st.data() arg's ConjectureData has been frozen, return its testcounter; else
    None. A body that freezes its own data and then errors should end via a StopTest the
    engine swallows, not surface as a failure (test_exceptiongroup test_discard_frozen)."""
    for arg in drawn:
        cd = getattr(arg, "conjecture_data", None)
        if cd is None:
            continue
        try:
            if cd.frozen:
                return cd.testcounter
        except AttributeError:
            continue
    return None


def _falsifying_example_note(
    fn_name: str, names: list[str], vals: tuple[Any, ...], explain: bool,
    explicit: bool = False, varargs: tuple[Any, ...] = (),
) -> str:
    """Upstream's multi-line keyword falsifying-example format (test_regex_output):
        Falsifying example: test(
            1,                                         <- positional *args, when present
            x=1,
            s='00',  # or any other generated value   <- only when Phase.explain ran
        )
    An explicit @example uses the "Falsifying explicit example:" prefix."""
    prefix = "Falsifying explicit example:" if explicit else "Falsifying example:"
    if not vals and not varargs:
        return f"{prefix} {fn_name}()"
    suffix = "  # or any other generated value" if explain else ""
    # Positional *args are user-supplied at call time (not drawn), so the explain
    # "or any other generated value" suffix applies only to the generated keyword args.
    body = [f"    {v!r}," for v in varargs]
    body += [f"    {n}={v!r},{suffix}" for n, v in zip(names, vals)]
    return f"{prefix} {fn_name}(\n" + "\n".join(body) + "\n)"


def _raise_multiple_bugs(
    result: list[tuple[tuple[Any, ...], BaseException]],
    fn: Any,
    given_names: list[str],
    pnames: Any,
    active: Any,
) -> None:
    """Raise an ExceptionGroup of the distinct per-origin falsifying exceptions found
    under report_multiple_bugs=True (hypothesis's MultipleFailures). Each sub-exception
    carries its own Falsifying-example note and trimmed traceback."""
    explain_on = pnames is None or "explain" in pnames
    verbosity = getattr(active, "verbosity", None)
    excs: list[BaseException] = []
    for falsifying, exc in result:
        try:
            exc.add_note(
                _falsifying_example_note(fn.__name__, given_names, falsifying, explain_on)
            )
        except Exception:  # noqa: BLE001 - <3.11 or non-standard exc
            pass
        _trim_internal_tb(exc, verbosity)
        excs.append(exc)
    msg = f"Hypothesis found {len(excs)} distinct failures."
    only_exceptions = [e for e in excs if isinstance(e, Exception)]
    if len(only_exceptions) == len(excs):
        raise ExceptionGroup(msg, only_exceptions)
    raise BaseExceptionGroup(msg, excs)


def _suppressed_checks(s: Any) -> frozenset[Any]:
    """The set of HealthCheck members suppressed by settings `s` (empty if None)."""
    raw = getattr(s, "suppress_health_check", ()) if s is not None else ()
    return frozenset(raw or ())


def _native_db_key(fn: Any) -> bytes:
    """A stable per-test database key. Hypothesis derives this from `function_digest`;
    we only need a value that is stable across runs of the SAME test function so that a
    saved minimal example is found again on replay. Hash module+qualname+bytecode."""
    import hashlib

    h = hashlib.sha384()
    h.update((getattr(fn, "__module__", "") or "").encode("utf-8", "surrogatepass"))
    h.update(b".")
    h.update((getattr(fn, "__qualname__", getattr(fn, "__name__", "")) or "").encode("utf-8", "surrogatepass"))
    code = getattr(fn, "__code__", None)
    if code is not None:
        h.update(code.co_code)
    return h.digest()


def _trim_internal_tb(exc: BaseException, verbosity: Any) -> None:
    """Trim hypothesis_fast-internal frames from a falsifying exception's traceback,
    so the user's trace shows the test entry + one internal frame + the failing line
    (mirrors upstream's elision; skipped in debug verbosity or with
    HYPOTHESIS_NO_TRACEBACK_TRIM). Called right before the `raise <exc>` re-raise: we drop
    LEADING internal frames WHILE the next is ALSO internal (keeping the last internal frame
    adjacent to user code); the re-raise then prepends exactly one entry frame, landing on
    the minimal trace."""
    import os

    if os.environ.get("HYPOTHESIS_NO_TRACEBACK_TRIM"):
        return
    try:
        from .settings import Verbosity

        if verbosity is not None and verbosity >= Verbosity.debug:
            return
    except Exception:  # noqa: BLE001
        return
    tb = exc.__traceback__
    if tb is None:
        return
    import hypothesis_fast as _hp

    pkg = os.path.dirname(os.path.abspath(_hp.__file__))

    def internal(t: Any) -> bool:
        return os.path.abspath(t.tb_frame.f_code.co_filename).startswith(pkg)

    while tb.tb_next is not None and internal(tb) and internal(tb.tb_next):
        tb = tb.tb_next
    exc.__traceback__ = tb


def _raise_flaky(message: str, exceptions: list[BaseException]) -> None:
    """Raise a FlakyFailure (an exception group) the way hypothesis does when a
    falsifying example doesn't reproduce consistently on replay."""
    from .errors import FlakyFailure

    raise FlakyFailure(message, exceptions)


_LOOP: asyncio.AbstractEventLoop | None = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    return _LOOP


def _parse_params(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]):
    sig = inspect.signature(fn)
    plist = list(sig.parameters.values())
    names = [p.name for p in plist]
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in plist)
    has_self = bool(names) and names[0] in ("self", "cls")
    # body = explicitly-named fillable params (exclude self/cls, *args and **kw)
    _fillable = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    body = [
        p.name
        for i, p in enumerate(plist)
        if p.kind in _fillable and not (has_self and i == 0)
    ]
    # `@given(...)` (sole Ellipsis) means "infer every parameter" — expand it to a keyword
    # infer for each positional-or-keyword / keyword-only param (matching hypothesis).
    if len(args) == 1 and args[0] is ... and not kwargs:
        infer_params = [
            p.name
            for i, p in enumerate(plist)
            if p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and not (has_self and i == 0)
        ]
        args = ()
        kwargs = {name: ... for name in infer_params}
    if args and kwargs:
        raise InvalidArgument("given() does not mix positional and keyword strategies")
    if not args and not kwargs:
        raise InvalidArgument("given() requires at least one strategy")
    if args:
        # `...` (infer) may only be passed by keyword — except the sole-argument form
        # `@given(...)`, which means "infer every parameter" (handled below).
        if ... in args and not (len(args) == 1 and not kwargs):
            raise InvalidArgument(
                "... was passed as a positional argument to @given, but may only be "
                "passed as a keyword argument or as the sole argument of @given"
            )
        # Positional strategies require every parameter to be POSITIONAL_OR_KEYWORD;
        # varargs/varkeywords/positional-only/keyword-only make positional assignment
        # ambiguous, so hypothesis rejects them (use keyword strategies instead).
        if any(p.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for p in plist):
            raise InvalidArgument(
                "positional arguments to @given are not supported with varargs, "
                "varkeywords, positional-only, or keyword-only arguments"
            )
        if len(args) > len(body):
            raise InvalidArgument("given() got more strategies than the test has parameters")
        given_names = body[len(body) - len(args):]
        strategies = list(args)
    else:
        # keyword given: names matching explicit params first, then any extras that
        # land in **kw (only valid when the test actually has **kw).
        body_given = [n for n in body if n in kwargs]
        extra_given = [n for n in kwargs if n not in body]
        if extra_given and not has_varkw:
            # common typo: extra kwarg name collides with a `settings(...)` field.
            # Suggest @settings instead, matching hypothesis' helpful error.
            from .settings import _DEFAULTS as _SETTINGS_FIELDS  # local: avoid cycle

            misnamed = [n for n in extra_given if n in _SETTINGS_FIELDS]
            if misnamed:
                hint = ", ".join(f"{n}={kwargs[n]!r}" for n in misnamed)
                raise InvalidArgument(
                    f"{sorted(misnamed)!r} look like @settings parameters, not "
                    f"strategies. Did you mean @settings({hint})?"
                )
            raise InvalidArgument(
                f"given() got strategies for non-parameters: {sorted(extra_given)}"
            )
        given_names = body_given + extra_given
        strategies = [kwargs[n] for n in given_names]
    # a parameter @given fills must not also have a default value (hypothesis)
    for name in given_names:
        p = sig.parameters.get(name)
        if p is not None and p.default is not inspect.Parameter.empty:
            raise InvalidArgument(
                f"Cannot use @given to fill parameter {name!r}: it has a default value"
            )
        # @given fills params by keyword at call time; a positional-only param
        # cannot receive a kwarg → mark as a usage error, matching hypothesis.
        if p is not None and p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise InvalidArgument(
                f"Cannot use @given to fill positional-only parameter {name!r}"
            )
    return has_self, body, given_names, strategies


def _native_frontend_active() -> bool:
    """True when `hypothesis_fast.strategies` is the all-native frontend (the package
    attribute the native-default config rebinds to `native_strategies`). Only then is
    `_engine.from_type` the correct resolver for `@given(...)` inference — under the legacy
    frontend, `st.*` produces legacy strategies, so inference must defer to the fallback."""
    import hypothesis_fast as _hp
    from hypothesis_fast import native_strategies as _ns

    return _hp.strategies is _ns


def _resolve_infer_native(
    fn: Callable[..., Any], given_names: list[str], strategies: list[Any]
) -> list[Any] | None:
    """Resolve `@given(...)`/`infer` (Ellipsis) strategies to native ones via from_type
    of each parameter's type annotation. Returns the fully-native strategy list, or None
    to signal "fall back to real hypothesis" (any param failed to resolve to a native
    SearchStrategy — e.g. a type our from_type defers to legacy). Raises InvalidArgument
    for a `...` param with no annotation (matches hypothesis)."""
    import typing

    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:  # noqa: BLE001 - unresolvable annotations -> let fallback handle it
        return None
    out: list[Any] = []
    for name, s in zip(given_names, strategies):
        if s is ...:
            if name not in hints:
                raise InvalidArgument(
                    f"passed ... for {name}, but {name} has no type annotation"
                )
            try:
                resolved = _engine.from_type(hints[name])
            except Exception:  # noqa: BLE001 - native from_type can't handle it -> fallback
                return None
            if not isinstance(resolved, _engine.SearchStrategy):
                return None
            out.append(resolved)
        elif isinstance(s, _engine.SearchStrategy):
            out.append(s)
        else:
            return None
    return out


def _deferred_error_test(fn: Callable[..., Any], exc: InvalidArgument) -> Callable[..., Any]:
    """A @given whose arguments are invalid. hypothesis defers such errors to test
    INVOCATION (decorating must never raise, so imports/collection don't break), so
    we return a callable that re-raises when run, presenting no params to pytest."""

    @functools.wraps(fn)
    def wrapper(*pargs: Any, **pkw: Any) -> None:
        raise exc

    wrapper.__signature__ = inspect.Signature([])  # type: ignore[attr-defined]
    wrapper._hypothesis_internal_use_settings = getattr(  # type: ignore[attr-defined]
        fn, "_hypothesis_internal_use_settings", None
    )
    return wrapper


def given(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        # @given on a class is a usage error: hypothesis only decorates test FUNCTIONS.
        # Raise immediately (NOT deferred) — `@given(...) class C:` is evaluated at
        # class-definition time and never produces a callable invocation site for
        # the deferred error to fire from, so deferring would silently swallow it.
        if isinstance(fn, type):
            raise InvalidArgument(
                "@given cannot be applied to a class — apply it to test functions instead"
            )
        # @given @given is a usage error too (the inner @given already turned `fn`
        # into a wrapper that does its own example generation; stacking another is
        # never what the user means).
        if is_hypothesis_test(fn):
            return _deferred_error_test(
                fn,
                InvalidArgument(
                    "You have applied @given to the test "
                    f"{getattr(fn, '__name__', fn)!r} more than once, which "
                    "wraps the test multiple times and is extremely slow. A "
                    "similar effect can be gained by combining the arguments of "
                    "the two calls to given. For example, instead of "
                    "@given(booleans()) @given(integers()), you could write "
                    "@given(booleans(), integers())."
                ),
            )
        is_async = inspect.iscoroutinefunction(fn)
        try:
            _has_self, _body, given_names, strategies = _parse_params(fn, args, kwargs)
        except InvalidArgument as exc:
            return _deferred_error_test(fn, exc)
        # `@given(...)` / infer: resolve Ellipsis params to native strategies via from_type.
        # Only when the native frontend is active (else from_type's `st.*`-backed resolution
        # is wrong, e.g. grouped-metadata yields legacy strategies). If every param resolves
        # native, the native engine runs it; otherwise leave the Ellipsis so the fallback
        # path routes inference through real hypothesis.
        if any(s is ... for s in strategies) and _native_frontend_active():
            try:
                inferred = _resolve_infer_native(fn, given_names, strategies)
            except InvalidArgument as exc:
                return _deferred_error_test(fn, exc)
            if inferred is not None:
                strategies = inferred
        inner_settings = getattr(fn, "_hypothesis_internal_use_settings", None)

        # Native path: when every strategy is an engine-native SearchStrategy (built by
        # the Rust `_engine.*` constructors) the whole test runs through the Rust
        # generate+shrink engine; async test bodies are driven on an event loop inside
        # the native runner. Anything else (a real-hypothesis strategy passed directly,
        # or an unresolved `@given(...)` infer) goes to the real-hypothesis fallback —
        # the proptest engine is no longer used.
        if strategies and all(isinstance(s, _engine.SearchStrategy) for s in strategies):
            return _native_engine_given(
                fn, given_names, strategies, inner_settings, is_async
            )
        return _fallback_given(fn, given_names, strategies, is_async, inner_settings)

    return decorate


def _has_unreplayable_arg(args: Any) -> bool:
    """Args whose draws can't survive a `runner(*args)` replay: an interactive data()'s
    DataObject (frozen after the run) or a randoms() object (its draw-state has advanced
    and the underlying data is frozen). Such failures are re-raised, not replayed."""
    if any(isinstance(a, _engine.DataObject) for a in args):
        return True
    try:
        from hypothesis.strategies._internal.random import HypothesisRandom
    except Exception:  # noqa: BLE001 - real hypothesis unavailable -> only DataObject matters
        return False
    return any(isinstance(a, HypothesisRandom) for a in args)


# Per-thread statistics collection. `.cases` is a list of per-test-case dicts while the
# generate phase runs under an active hypothesis.statistics.collector, else None.
_stats_tls = threading.local()

# Per-thread seed source override: find(random=...) sets this so generation is driven by
# the caller's Random (deterministic across repeated find() calls — test_find_uses_provided_random).
_random_override = threading.local()


def _stats_collector() -> Any:
    """The active statistics collector callback (hypothesis.statistics.collector.value),
    or None. The cover suite sets it via `collector.with_value(...)`."""
    mod = sys.modules.get("hypothesis.statistics")
    return mod.collector.value if mod is not None else None


def _push_statistics(
    cases: list[dict[str, Any]],
    duration: float,
    max_examples: int,
    targets: dict[str, float],
    stopped_override: str | None = None,
) -> None:
    callback = _stats_collector()
    if callback is None:
        return
    from collections import Counter

    statuses = Counter(c["status"] for c in cases)
    total = sum(statuses.values())
    valid = statuses.get("valid", 0)
    interesting = statuses.get("interesting", 0)
    if stopped_override is not None:
        stopped = stopped_override
    elif interesting:
        stopped = "nothing left to do"
    elif total and valid < max(1, total // 100):
        stopped = (
            f"settings.max_examples={max_examples}, "
            "but < 1% of examples satisfied assumptions"
        )
    else:
        stopped = f"settings.max_examples={max_examples}"
    callback(
        {
            "generate-phase": {
                "test-cases": cases,
                "duration-seconds": duration,
                "distinct-failures": interesting,
                "shrinks-successful": 0,
            },
            "targets": targets,
            "stopped-because": stopped,
        }
    )


def _report(text: str) -> None:
    """Send `text` to the active hypothesis reporter (verbose/progress output)."""
    rep = sys.modules.get("hypothesis.reporting")
    if rep is None:
        return
    try:
        rep.current_reporter()(text)
    except Exception:  # noqa: BLE001 - reporting must never break a test run
        pass


def _emit_shrink_report(rep: tuple[int, int, bool, list[tuple[str, int, int]]]) -> None:
    """Print hypothesis's debug-verbosity shrink-pass profiling: a total line plus, for
    each pass that ran, "<name> made N call(s) of which M shrank" (test_reports_passes,
    test_debug_information). Named passes mirror upstream (minimize_individual_choices, …)."""
    total_calls, total_shrinks, _capped, passes = rep

    def _s(n: int) -> str:
        return "" if n == 1 else "s"

    _report(
        f"Shrink pass profiling — {total_shrinks} shrinks in {total_calls} "
        f"call{_s(total_calls)}:"
    )
    for name, calls, shrinks in passes:
        if calls == 0:
            continue
        _report(f"  * {name} made {calls} call{_s(calls)} of which {shrinks} shrank.")


def _reproduce_failure_note(active: Any) -> str | None:
    """The `@reproduce_failure(version, blob)` line to attach to a falsifying example when
    print_blob is set: encodes the minimal choice sequence (via hypothesis.core's
    encode_failure) so the failure can be pinned and replayed (test_prints_reproduction)."""
    if not getattr(active, "print_blob", False):
        return None
    try:
        from . import __version__

        choices = _engine.minimal_choices()
        core_mod = sys.modules.get("hypothesis.core")
        if not choices or core_mod is None or not hasattr(core_mod, "encode_failure"):
            return None
        blob = core_mod.encode_failure(choices)
        return f"@reproduce_failure({__version__!r}, {blob!r})"
    except Exception:  # noqa: BLE001 - reproduction blob is best-effort, never fatal
        return None


def _describe_targets(best: dict[str, float]) -> list[str]:
    """The "Highest target score:" report line(s) for the observed target() scores, via
    real hypothesis's formatter (exact text/format parity); empty if unavailable."""
    mod = sys.modules.get("hypothesis.statistics")
    if mod is not None:
        try:
            return list(mod.describe_targets(best))
        except Exception:  # noqa: BLE001 - reporting is best-effort, never fatal
            return []
    return []


def _make_build_context(fn: Callable[..., Any]) -> Any:
    """Build ONE real hypothesis BuildContext to wrap the native test body, so the cover
    suite's real control APIs (note/event/current_build_context/cleanup/
    currently_in_test_context) and stateful's is_final work inside it. Returns the
    BuildContext (NOT entered) or None if real hypothesis isn't importable (standalone use).

    Built ONCE PER RUN and reused for every example by the runner: the body never draws from
    this cd (args are drawn in Rust before the body runs), so the same throwaway cd anchors
    the context for the whole run. The runner resets the per-example mutable state
    (is_final / tasks / known_object_printers / _label_path) and re-enters it per example via
    BuildContext.__enter__/__exit__. Reusing one context instead of constructing
    `ConjectureData.for_choices([]) + BuildContext` every example was ~35% of the trivial-body
    per-example wall (profiled 2026-06-14)."""
    cd_mod = sys.modules.get("hypothesis.internal.conjecture.data")
    ctrl_mod = sys.modules.get("hypothesis.control")
    if cd_mod is None or ctrl_mod is None:
        return None
    try:
        data = cd_mod.ConjectureData.for_choices([])
        return ctrl_mod.BuildContext(data, is_final=False, wrapped_test=fn)
    except Exception:  # noqa: BLE001 - any construction failure -> run without a context
        return None


def _real_verbosity_settings(verbosity: Any) -> Any:
    """Enter real hypothesis's local_settings at `verbosity` for the duration of the
    native body. Real note()/report() gate on `settings.default.verbosity` (its global
    default), not on our per-test `active`; setting the real default makes them honour
    @settings(verbosity=...) so every example's notes print in verbose/debug mode."""
    s_mod = sys.modules.get("hypothesis._settings")
    if s_mod is None or verbosity is None:
        return contextlib.nullcontext()
    try:
        name = verbosity.name if hasattr(verbosity, "name") else Verbosity(verbosity).name
        rv = s_mod.Verbosity[name]
        base = s_mod.settings.default
        new = (
            s_mod.settings(base, verbosity=rv)
            if base is not None
            else s_mod.settings(verbosity=rv)
        )
        return s_mod.local_settings(new)
    except Exception:  # noqa: BLE001 - reporting verbosity is best-effort, never fatal
        return contextlib.nullcontext()


def _example_executor(outer_names: list[str], bound: dict[str, Any]) -> Any:
    """The executor `self` for a @given method: the first non-strategy arg (self/cls) when
    it defines any of setup_example/teardown_example/execute_example, else None."""
    if not outer_names or outer_names[0] not in ("self", "cls"):
        return None
    cand = bound.get(outer_names[0])
    if cand is None:
        return None
    if any(
        getattr(cand, m, None) is not None
        for m in ("setup_example", "teardown_example", "execute_example")
    ):
        return cand
    return None


@contextlib.contextmanager
def _fake_subTest(self: Any, msg: Any = None, **__: Any) -> Any:
    """Replacement for unittest.TestCase.subTest during @given: subTest reports each example
    as a separate sub-result, which is meaningless when Hypothesis runs hundreds of them, so
    we warn and no-op it for the duration of the run (test_subTest)."""
    import warnings

    from .errors import HypothesisWarning

    warnings.warn(
        "subTest per-example reporting interacts badly with Hypothesis trying "
        "hundreds of examples, so we disable it for the duration of any test that "
        "uses `@given`.",
        HypothesisWarning,
        stacklevel=2,
    )
    yield


# differing_executors health check: a @given wrapper records the executor `self` it was
# called with; a later call with a DIFFERENT self is flagged (replaying from the database
# across executors gives nonreproducible errors). The recorded self is tagged with a session
# counter, bumped per pytest session by _reset_executor_tracking (called from the conftest's
# pytest_sessionstart), so the recorded self can't leak across runs under the resident
# pytest-fast daemon — which would otherwise fire on a re-run's very first call
# (test_differing_executors_fails_health_check).
_executor_session = 0


def _reset_executor_tracking() -> None:
    """Invalidate every wrapper's recorded executor (called once per pytest session)."""
    global _executor_session
    _executor_session += 1


def _native_engine_given(
    fn: Callable[..., Any],
    given_names: list[str],
    strategies: list[Any],
    inner_settings: Any,
    is_async: bool = False,
) -> Callable[..., Any]:
    max_examples = 100
    if inner_settings is not None:
        try:
            max_examples = int(getattr(inner_settings, "max_examples", 100) or 100)
        except Exception:  # noqa: BLE001
            pass

    given_set = set(given_names)
    try:
        sig = inspect.signature(fn)
        outer_params = [
            p for name, p in sig.parameters.items() if name not in given_set
        ]
        outer_names = [p.name for p in outer_params]
        # Positional-only params (always user-supplied — @given can't fill them, see the
        # validation in given()) must be passed positionally, not as keywords, when we
        # invoke fn (test_given_works_with_positional_only_params).
        posonly_names = [
            name
            for name, p in sig.parameters.items()
            if p.kind is inspect.Parameter.POSITIONAL_ONLY
        ]
        # A *args param: extra positional args at call time go here and must be splatted
        # positionally when invoking fn (test_vararg_output).
        varargs_name = next(
            (
                name
                for name, p in sig.parameters.items()
                if p.kind is inspect.Parameter.VAR_POSITIONAL
            ),
            None,
        )
    except (ValueError, TypeError):
        outer_params = None
        outer_names = []
        posonly_names = []
        varargs_name = None

    # @example cases pinned onto the inner test (e.g. `@given(...) @example(...) def f`).
    inner_examples = list(getattr(fn, "_hypothesis_explicit_examples", []))

    # Names of the fixed positional outer params that precede *args (passed positionally,
    # in order, ahead of the collected varargs when invoking fn).
    _fixed_pos_names = (
        outer_names[: outer_names.index(varargs_name)]
        if varargs_name is not None and varargs_name in outer_names
        else outer_names
    )

    @functools.wraps(fn)
    def wrapper(*pargs: Any, **pkw: Any) -> None:
        # pargs/pkw are the non-strategy params pytest supplied (self + fixtures), plus any
        # positional varargs at call time.
        if varargs_name is not None:
            bound = dict(zip(_fixed_pos_names, pargs))
            varargs_vals = tuple(pargs[len(_fixed_pos_names):])
        else:
            bound = dict(zip(outer_names, pargs))
            varargs_vals = ()
        bound.update(pkw)

        # Executor protocol: setup_example/teardown_example wrap the whole example (incl.
        # arg generation) in the native engine; execute_example wraps just the post-draw
        # test call and consumes its return value, so it's applied here in the runner.
        _executor = _example_executor(outer_names, bound)
        _execute = (
            getattr(_executor, "execute_example", None) if _executor is not None else None
        )
        # unittest.TestCase.subTest is disabled for the duration of a @given run (it reports
        # each of the hundreds of examples as a separate sub-result); patch it to a warning
        # no-op per example in the runner below (test_subTest).
        _tc_self = (
            bound.get(outer_names[0])
            if outer_names and outer_names[0] in ("self", "cls")
            else None
        )
        if _tc_self is not None:
            import unittest as _ut

            if not isinstance(_tc_self, _ut.TestCase) or not hasattr(_tc_self, "subTest"):
                _tc_self = None
        # Read the inner test FRESH from the (mutable) `wrapper.hypothesis.inner_test`
        # handle rather than closing over the original `fn`: pytest-asyncio / pytest-trio
        # substitute it at collection time with a synchronous driver that runs the coroutine
        # on their event loop. Re-deriving async-ness from the current handle means a swapped-in
        # driver is sync and runs normally, while a bare coroutine (no such plugin) stays async
        # and — with no executor — errors, matching upstream (test_specific_error_for_
        # coroutine_functions).
        _inner = getattr(getattr(wrapper, "hypothesis", None), "inner_test", fn)
        _inner_async = inspect.iscoroutinefunction(_inner)
        # A coroutine test needs an executor (execute_example) that knows how to run it on
        # an event loop; without one Hypothesis can't run it. test_asyncio supplies such an
        # executor, so it still runs.
        if _inner_async and _execute is None:
            raise InvalidArgument(
                "Hypothesis doesn't know how to run async test functions like "
                f"{getattr(fn, '__name__', 'test')}. You'll need to write a custom "
                "executor, or use a library such as pytest-asyncio or pytest-trio which "
                "can decorate a test function and then drive it through Hypothesis."
            )

        # Managed PRNGs (global random / numpy / register_random), captured once per run
        # below as (random, original_state, seed0_state) triples. The outer save/restore is
        # hoisted to the run boundary (`_run_rng`). Per-random seed-0 states are required
        # because the state format differs by PRNG (a stdlib MT19937 state can't be
        # setstate'd onto numpy's PRNG).
        #
        # `_run_rng` = ALL managed PRNGs, for the run-level save/restore.
        # `_run_rng_pin` = the SUBSET re-pinned to seed-0 per example by `runner`. It
        # EXCLUDES the master `_hypothesis_global_random`: the master derives the run seed
        # exactly once (getrandbits(64) at run setup, before any example) and is never read
        # inside a test body, so pinning it per example is wasted work. Its only invariant —
        # advance once per run, restore to the advanced state afterward — is enforced by the
        # run-level getrandbits/save/restore, not by the per-example pin. Dropping it halves
        # the per-example setstate cost (2 PRNGs -> 1: the global `random` module).
        _run_rng: list[Any] = []
        _run_rng_pin: list[Any] = []
        # The run-level real BuildContext, built ONCE in run setup and reused by `runner` for
        # every example (a 1-element box so the closure reads the value set in run setup).
        # None when real hypothesis isn't importable -> runner uses a nullcontext.
        _bc: list[Any] = [None]

        def runner(*drawn: Any) -> Any:
            call = dict(bound)
            call.update(zip(given_names, drawn))
            # With a *args param, the fixed positional params precede the collected varargs,
            # all passed positionally; the rest (given/keyword-only) stay keywords.
            if varargs_name is not None:
                pos = [call.pop(n) for n in _fixed_pos_names if n in call] + list(varargs_vals)
            else:
                # Positional-only params must be passed positionally (in signature order),
                # the rest by keyword.
                pos = [call.pop(n) for n in posonly_names if n in call]
            # Push an _ExampleContext so in-test-context APIs (target(), and its
            # label-dedup / outside-test guards) work inside the native test body.
            _stats_cases = getattr(_stats_tls, "cases", None)
            _stat_t0 = time.perf_counter() if _stats_cases is not None else 0.0
            _stat_status = "valid"
            _verbose = _run_verbose
            if _verbose:
                _args_repr = ", ".join(
                    f"{n}={v!r}" for n, v in zip(given_names, drawn)
                )
                _report(f"Trying example: {fn.__name__}({_args_repr})")
            _enter_test_context()
            _saved_subtest = None
            if _tc_self is not None:
                import types as _types

                _saved_subtest = _tc_self.subTest
                _tc_self.subTest = _types.MethodType(_fake_subTest, _tc_self)
            try:
                # per-example deadline (None = disabled, our default), hoisted to a run
                # constant. Args are drawn in Rust BEFORE runner, so timing fn() naturally
                # excludes arg-draw time — only the test body (incl. st.data() draws) is timed.
                deadline_s = _run_deadline_s
                t0 = time.perf_counter() if deadline_s is not None else 0.0
                # Draw time so far (arg draws); interactive st.data() draws during the body
                # add to this, and are excluded from the deadline (which times test
                # execution, not generation — test_slow_generation_inline).
                _draw0 = _engine.draw_secs() if deadline_s is not None else 0.0
                _is_final = getattr(_stats_tls, "final_replay", False)
                # In verbose/debug mode, set real hypothesis's default verbosity so its
                # note()/report() (gated on settings.default.verbosity) print every
                # example's notes, not only the final replay (test_prints_all_notes_…).
                _vctx = (
                    _real_verbosity_settings(getattr(active, "verbosity", None))
                    if _verbose
                    else contextlib.nullcontext()
                )
                # Pin global/registered PRNGs to their seed-0 state for this example
                # (anti-pollution, reproducible incidental random use). Save/restore is at
                # run level, and we setstate() each PRNG's own precomputed seed-0 state
                # instead of seed(0) — ~2.6x faster and identical — replacing
                # deterministic_PRNG's per-example save+dummy-Random+hash dance. The master
                # (run-seed source, never read in a body) is omitted from `_run_rng_pin`.
                for _r, _s0 in _run_rng_pin:
                    _r.setstate(_s0)
                # Reuse the run-level BuildContext (built once in run setup) instead of
                # constructing a fresh real ConjectureData+BuildContext every example (was
                # ~35% of the trivial-body per-example wall, profiled 2026-06-14). Reset the
                # per-example mutable state and re-enter via the real __enter__/__exit__, which
                # preserve the current_build_context push and the cleanup()-task teardown
                # contract. _label_path is always balanced in our flow (never pushed — args are
                # drawn in Rust) but is cleared defensively.
                _bcx = _bc[0]
                if _bcx is None:
                    _body_ctx: Any = contextlib.nullcontext()
                else:
                    _bcx.is_final = _is_final
                    if _bcx.tasks:
                        _bcx.tasks.clear()
                    if _bcx.known_object_printers:
                        _bcx.known_object_printers.clear()
                    if _bcx._label_path:
                        _bcx._label_path.clear()
                    _body_ctx = _bcx
                with _vctx, _body_ctx:
                    try:
                        if _execute is not None:
                            res = _execute(lambda: _inner(*pos, **call))
                        else:
                            res = _inner(*pos, **call)
                        if _inner_async:
                            res = _ensure_loop().run_until_complete(res)
                    except BaseExceptionGroup as _eg:
                        # Surface StopTest/Frozen markers wrapped in a group, matching
                        # hypothesis's unwrap_markers_from_group (test_exceptiongroup).
                        _unwrap_markers_from_group(_eg)
                if deadline_s is not None and (
                    elapsed := time.perf_counter() - t0 - (_engine.draw_secs() - _draw0)
                ) > deadline_s:
                    import datetime as _dt

                    from .errors import DeadlineExceeded

                    raise DeadlineExceeded(
                        _dt.timedelta(seconds=elapsed), _dt.timedelta(seconds=deadline_s)
                    )
                # return_value health check: a test that returns non-None is almost
                # always a typo'd assertion (e.g. `return x == y`); reject unless
                # suppressed (test_returning_non_none_is_forbidden).
                if res is not None and HealthCheck.return_value not in _suppressed_checks(active):
                    from .errors import FailedHealthCheck

                    raise FailedHealthCheck(
                        f"Test returned {res!r} (a non-None value) instead of None; "
                        "did you mean to assert something? (HealthCheck.return_value)"
                    )
                return res
            except BaseException as exc:
                # #3819: if a sampled_from-of-strategies was drawn this example and a
                # TypeError mentioning SearchStrategy escaped, attach the "Was one_of
                # intended?" note before the engine captures the exception.
                _stat_status = (
                    "invalid" if isinstance(exc, UnsatisfiedAssumption) else "interesting"
                )
                # If the body froze its own ConjectureData and then errored, resume normal
                # operation with a StopTest the engine swallows (test_exceptiongroup).
                from .errors import StopTest as _StopTest

                if not isinstance(exc, (_StopTest, UnsatisfiedAssumption)):
                    _ftc = _frozen_testcounter(drawn)
                    if _ftc is not None:
                        _stat_status = "invalid"
                        raise _StopTest(_ftc) from exc
                if _verbose and not isinstance(exc, UnsatisfiedAssumption):
                    _report(f"{type(exc).__name__}: {exc}")
                _native_3819_note(exc)
                if not isinstance(exc, UnsatisfiedAssumption):
                    _attach_draw_notes(exc, drawn)
                    _ex_notes = current_notes()
                    if _ex_notes:
                        _existing = getattr(exc, "__notes__", None) or []
                        for _n in _ex_notes:
                            if _n not in _existing and hasattr(exc, "add_note"):
                                exc.add_note(_n)
                raise
            finally:
                # Always drain the filter draw-event buffer (even when not collecting) so it
                # can't grow unboundedly across examples under the resident daemon. Resolve the
                # module via sys.modules (a dict lookup) rather than re-running the `from .
                # import` statement machinery every example (~1.5µs/example); None only before
                # native_strategies is imported — no native draws yet, so nothing to drain.
                _ns_draw = sys.modules.get("hypothesis_fast.native_strategies")
                _draw_evs = _ns_draw.drain_draw_events() if _ns_draw is not None else []
                if _stats_cases is not None:
                    _evs = drain_events()
                    for _de in _draw_evs:
                        if _de not in _evs:
                            _evs.append(_de)
                    # drawtime: argument + interactive st.data() draw time for THIS example
                    # (measured in Rust via the mockable perf_counter clock). _stat_t0 was
                    # taken after args were drawn, so runtime here is body-only — add drawtime
                    # back so "runtime, of which N in data generation" stays consistent
                    # (test_draw_timing).
                    _drawtime = _engine.draw_secs()
                    _stats_cases.append(
                        {
                            "status": _stat_status,
                            "runtime": (time.perf_counter() - _stat_t0) + _drawtime,
                            "drawtime": _drawtime,
                            "gctime": 0.0,
                            "events": _evs,
                        }
                    )
                if _tc_self is not None:
                    _tc_self.subTest = _saved_subtest
                _clear_native_3819()
                _exit_test_context()

        s = getattr(wrapper, "_hypothesis_internal_use_settings", None) or inner_settings
        active = s if s is not None else _settings_cls()
        # Run-constant per-example settings, hoisted out of the hot `runner` (read once here
        # rather than via getattr(active, ...) on every example).
        _run_verbose = getattr(active, "verbosity", Verbosity.normal) >= Verbosity.verbose
        _dl0 = getattr(active, "deadline", None)
        if _dl0 is None:
            _run_deadline_s: float | None = None
        elif hasattr(_dl0, "total_seconds"):
            _run_deadline_s = _dl0.total_seconds()
        else:
            _run_deadline_s = float(_dl0) / 1000.0
        # nested_given health check: a @given running inside another @given's body (a live
        # example context is on the stack) fires nested_given — keyed on the OUTER test's
        # currently-active default settings (set via apply_settings during its body), which
        # is why suppressing it on the inner test doesn't help (test_cant_suppress_*_inner).
        if _context_stack_nonempty() and HealthCheck.nested_given not in _suppressed_checks(
            _settings_cls.default
        ):
            from .errors import FailedHealthCheck

            raise FailedHealthCheck(
                "Cannot nest @given tests inside other @given tests "
                "(HealthCheck.nested_given). Use data() or refactor; or suppress "
                "HealthCheck.nested_given on the OUTER test."
            )
        # differing_executors health check: if this same wrapped method is invoked from a
        # DIFFERENT executor `self` than a previous call this session, flag it (see
        # _reset_executor_tracking). cur_self is the bound instance only when the wrapper is
        # genuinely the method on its type (not e.g. a mock).
        _cur_self = (
            bound.get(outer_names[0])
            if outer_names and outer_names[0] in ("self", "cls")
            else None
        )
        # not_a_test_method health check: @given on a unittest.TestCase lifecycle method
        # (setUp/tearDown/...) is almost always a mistake — those aren't tests
        # (test_given_on_setUp_fails_health_check). Use the RAW bound self (BEFORE the
        # wrapper-identity reset below), because such methods are often further wrapped (e.g.
        # @fails_with). An ordinary `test_*` method isn't in dir(TestCase), so real @given test
        # methods are unaffected.
        if _cur_self is not None:
            import unittest as _ut

            if (
                isinstance(_cur_self, _ut.TestCase)
                and fn.__name__ in dir(_ut.TestCase)
                and HealthCheck.not_a_test_method not in (_suppressed_checks(active) or ())
            ):
                from .errors import FailedHealthCheck

                raise FailedHealthCheck(
                    f"{getattr(fn, '__qualname__', fn.__name__)} is a method of a "
                    f"unittest.TestCase, but {fn.__name__} is not a test method — applying "
                    "@given to it is almost certainly a mistake (HealthCheck.not_a_test_method)."
                )
        if _cur_self is not None and getattr(type(_cur_self), fn.__name__, None) is not wrapper:
            _cur_self = None
        # Recorded per-THREAD (a threading.local on the wrapper) so concurrent threads each
        # track their own executor and don't cross-fire, and tagged with the session counter
        # so a per-test reset invalidates it.
        _exec_local = getattr(wrapper, "_hp_exec_local", None)
        if _exec_local is None:
            import threading

            _exec_local = threading.local()
            wrapper._hp_exec_local = _exec_local  # type: ignore[attr-defined]
        _prev_exec = getattr(_exec_local, "prev", None)
        if _prev_exec is None or _prev_exec[1] != _executor_session:
            _exec_local.prev = (_cur_self, _executor_session)
        elif _cur_self is not _prev_exec[0] and HealthCheck.differing_executors not in (
            _suppressed_checks(active) or ()
        ):
            from .errors import FailedHealthCheck

            raise FailedHealthCheck(
                f"The method {getattr(fn, '__qualname__', fn.__name__)} was called from "
                "multiple different executors. This may lead to flaky tests and "
                "nonreproducible errors when replaying from the database. Unlike most health "
                "checks, HealthCheck.differing_executors warns about a correctness issue with "
                "your test; we recommend fixing it, but you can suppress it with "
                "@settings(suppress_health_check=[HealthCheck.differing_executors])."
            )
        # @reproduce_failure: replay exactly the one encoded example, nothing else.
        repro = getattr(wrapper, "_hypothesis_internal_use_reproduce_failure", None)
        if repro is not None:
            with apply_settings(active):
                _run_reproduce_failure(fn, runner, list(strategies), repro)
            return
        # honour `phases`: explicit -> run @example cases; generate -> engine generation.
        phases = getattr(active, "phases", None)
        pnames = {getattr(p, "name", str(p)) for p in phases} if phases is not None else None
        run_explicit = pnames is None or "explicit" in pnames
        run_generate = pnames is None or "generate" in pnames
        ex_max = int(getattr(active, "max_examples", max_examples) or max_examples)
        _ran_any = False  # whether any example actually executed (else -> SkipTest)

        with apply_settings(active):
            if run_explicit:
                # @example cases can also be pinned ON the wrapper (e.g.
                # `@example(...) @given(...) def f`). Decorators apply bottom-up, so the
                # collected (applied) order is reversed from the source (top-to-bottom)
                # order hypothesis runs them in (test_examples_are_tried_in_order).
                examples = list(
                    reversed(
                        inner_examples
                        + list(getattr(wrapper, "_hypothesis_explicit_examples", []))
                    )
                )
                example_failures: list[tuple[BaseException, tuple[int, str]]] = []
                for ex in examples:
                    # Positional @example args are ambiguous when the test has positional-only
                    # params (which @given can't fill), so they're rejected (test_lookup_py38
                    # test_example_argument_validation).
                    if ex.args and posonly_names:
                        raise InvalidArgument(
                            "Cannot pass positional arguments to @example() when decorating "
                            "a test function which has positional-only parameters."
                        )
                    # an example must agree with the number/names of @given parameters
                    if len(ex.args) > len(given_names):
                        raise InvalidArgument(
                            f"example has {len(ex.args)} arguments but the test takes "
                            f"{len(given_names)} @given parameter(s)"
                        )
                    bad_kw = [k for k in ex.kwargs if k not in given_names]
                    if bad_kw:
                        given_kws = ", ".join(repr(k) for k in sorted(given_names))
                        example_kws = ", ".join(repr(k) for k in sorted(ex.kwargs))
                        raise InvalidArgument(
                            f"Inconsistent args: @given() got strategies for {given_kws}, "
                            f"but @example() got arguments for {example_kws}"
                        )
                    gen_vals: list[Any] = [None] * len(given_names)
                    for i, v in enumerate(ex.args):
                        gen_vals[i] = v
                    for k, v in ex.kwargs.items():
                        gen_vals[given_names.index(k)] = v
                    _ran_any = True
                    # Compute the note BEFORE running so it captures the original arg reprs
                    # even if the body mutates them (test_captures_original_repr_of_example).
                    _ex_note = _falsifying_example_note(
                        fn.__name__, given_names, tuple(gen_vals), False, explicit=True,
                        varargs=varargs_vals,
                    )
                    if getattr(active, "verbosity", Verbosity.normal) >= Verbosity.verbose:
                        _ar = ", ".join(
                            f"{n}={v!r}" for n, v in zip(given_names, gen_vals)
                        )
                        _report(f"Trying explicit example: {fn.__name__}({_ar})")
                    failed = _eval_explicit_example(ex, gen_vals, given_names, runner)
                    if failed is not None:
                        try:
                            failed.add_note(_ex_note)
                        except Exception:  # noqa: BLE001 - <3.11 / non-standard exc
                            pass
                        example_failures.append(
                            (failed, _example_sort_key(tuple(gen_vals)))
                        )
                if example_failures:
                    _raise_collected_failures(
                        example_failures,
                        getattr(active, "report_multiple_bugs", True),
                        getattr(active, "verbosity", Verbosity.normal) >= Verbosity.verbose,
                    )

            if not run_generate:
                # No generation phase and nothing ran explicitly (e.g. phases without
                # explicit/generate, or reuse with an empty DB) — hypothesis raises
                # SkipTest (test_non_executed_tests_raise_skipped).
                if not _ran_any:
                    import unittest

                    raise unittest.SkipTest(
                        "Hypothesis ran no examples (no generate phase and no explicit "
                        "examples to run)."
                    )
                return

            # An empty strategy (e.g. nothing()) can't generate anything — surface the
            # same Unsatisfiable hypothesis raises, after explicit examples have run.
            for nm, strat in zip(given_names, strategies):
                if strat.is_empty:
                    from .errors import Unsatisfiable

                    raise Unsatisfiable(
                        f"Cannot generate examples from empty strategy: {nm}={strat!r}"
                    )

            # derandomize: derive a stable seed from the test's identity so repeated runs
            # reproduce the same examples (test_can_derandomize). Otherwise pick fresh entropy.
            _seed_random = getattr(_random_override, "random", None)
            if _seed_random is not None:
                # find(random=...) — seed generation from the caller's Random so repeated
                # find() calls with the same Random reproduce (test_find_uses_provided_random).
                seed = _seed_random.getrandbits(64)
            elif getattr(active, "derandomize", False):
                import zlib

                _ident = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', fn.__name__)}"
                seed = zlib.adler32(_ident.encode("utf-8", "replace"))
            else:
                # Derive from the master PRNG (advancing it) so repeated runs differ and the
                # global random state isn't polluted (test_random_module).
                seed = _global_random().getrandbits(64)
            from . import native_strategies as _ns

            _ns.reset_random_calls()
            db = getattr(active, "database", None)
            db_key = _native_db_key(fn) if db is not None else None
            report_multi = bool(getattr(active, "report_multiple_bugs", True))
            # setup_example/teardown_example run around each example (incl. arg generation)
            # by the native engine; execute_example is applied in the runner above.
            # Statistics: when a collector is active, gather per-test-case records during
            # the generate phase and push one stats dict afterwards (hypothesis parity).
            _collecting = _stats_collector() is not None
            _stats_t0 = 0.0
            # Always track target() scores (not only under a statistics collector): the
            # falsifying-example report includes "Highest target score:" notes when targets
            # were used (test_shows_target_scores_with_multiple_failures).
            start_target_collection()
            if _collecting:
                _stats_tls.cases = []
                _stats_t0 = time.perf_counter()
                # Time draws via Python's (mockable) perf_counter while collecting stats, so
                # a frozen-clock run reports generation time (test_draw_timing). Off otherwise
                # — the engine times draws with a Rust Instant to avoid a per-draw Python call
                # on the hot shrink path.
                _engine.set_use_py_clock(True)
            # Health-check suppression + the give-up threshold (INVALID_THRESHOLD_BASE+1)
            # let a fully-rejected run report Unsatisfiable with exact reject counts when
            # filter_too_much is suppressed (test_notes_high_filter_rates).
            _suppress_names = [
                getattr(h, "name", str(h)) for h in (_suppressed_checks(active) or ())
            ]
            try:
                from hypothesis.internal.conjecture.engine import (
                    INVALID_THRESHOLD_BASE as _itb,
                )
            except Exception:  # noqa: BLE001 - real hypothesis internals not importable
                import math as _math

                _itb = _math.ceil(_math.log(1 - 0.99) / _math.log(1 - 0.01)) - 1
            _test_name = getattr(fn, "__name__", "test")
            # Deadline in ms (0 -> the engine uses its default too_slow allowance).
            _dl = getattr(active, "deadline", None)
            if _dl is None:
                _deadline_ms = 0.0
            elif hasattr(_dl, "total_seconds"):
                _deadline_ms = _dl.total_seconds() * 1000.0
            else:
                _deadline_ms = float(_dl)
            from .errors import FlakyFailure as _FlakyFailure

            # upstream MAX_SHRINKS (tests monkeypatch the real engine module's constant to
            # cap/disable shrinking — test_stops_after_x_shrinks).
            _eng_mod = sys.modules.get("hypothesis.internal.conjecture.engine")
            _max_shrinks = int(getattr(_eng_mod, "MAX_SHRINKS", 500)) if _eng_mod else 500
            # phases without Phase.shrink => don't shrink at all (the failing example is found
            # then replayed once == runs exactly twice; large data isn't minimised). Reuses the
            # max_shrinks=0 disable path (test_when_set_to_no_simplifies_runs_failing_example_twice,
            # test_does_not_print_reproduction_for_large_data_examples_by_default).
            if pnames is not None and "shrink" not in pnames:
                _max_shrinks = 0
            set_engine_active(True)
            _stats_payload: tuple[list[dict[str, Any]], float, dict[str, float]] | None = (
                None
            )
            _stopped_override: str | None = None
            _targets: dict[str, float] = {}
            # Snapshot each managed PRNG once as (random, original_state, seed0_state):
            # save its current state (restored in the finally), seed it to 0 and capture
            # THAT state (its own format), so `runner` only needs a per-example setstate —
            # hoisting deterministic_PRNG's per-example save/restore/hash out of the hot loop
            # (it was ~60% of trivial-body time). Also record the global random's seed-0
            # state hash once (for random_module reseed detection).
            _run_rng[:] = []
            _run_rng_pin[:] = []
            # The master is never read inside a body (it only sourced `seed` above); exclude
            # it from the per-example pin while still saving/restoring it at run level.
            _master = _resolve_threadlocal()._hypothesis_global_random
            for _r in _managed_randoms():
                try:
                    _orig = _r.getstate()
                    _r.seed(0)
                    _s0 = _r.getstate()
                    _run_rng.append((_r, _orig, _s0))
                    if _r is not _master:
                        _run_rng_pin.append((_r, _s0))
                except Exception:  # noqa: BLE001 - a managed PRNG without usable state; skip
                    pass
            if _run_rng:
                try:
                    from hypothesis.internal.entropy import (
                        _known_random_state_hashes as _krsh,
                    )

                    _krsh.add(hash(_seed0_state()))
                except Exception:  # noqa: BLE001 - best-effort reseed-detection parity
                    pass
            # Build the per-run BuildContext once; `runner` reuses it for every example.
            _bc[0] = _make_build_context(fn)
            try:
                try:
                    result = _engine.run_native(
                        runner, list(strategies), ex_max, seed, db, db_key, report_multi,
                        list(given_names), _executor, _suppress_names, _itb + 1, _test_name,
                        _deadline_ms, _max_shrinks,
                    )
                finally:
                    set_engine_active(False)
                    for _r, _orig, _ in _run_rng:
                        try:
                            _r.setstate(_orig)
                        except Exception:  # noqa: BLE001 - politeness restore, never fatal
                            pass
                    if _collecting:
                        _engine.set_use_py_clock(False)
                    _targets = drain_target_collection()
                    if _collecting:
                        # Stash the per-case records; the actual push is deferred to the OUTER
                        # finally so it runs on EVERY exit path — a flaky exit (detected by the
                        # minimal-example replay below) or a run_native raise (e.g. Unsatisfiable
                        # when everything was filtered) must still emit one statistics record.
                        _stats_payload = (
                            _stats_tls.cases,
                            time.perf_counter() - _stats_t0,
                            _targets,
                        )
                        _stats_tls.cases = None
                if result:
                    # Shrinker reporting: emit the debug-verbosity pass profiling and, if
                    # the run hit MAX_SHRINKS, record the "shrunk example N times" stop reason.
                    _shrink_rep = _engine.shrink_report()
                    if getattr(active, "verbosity", Verbosity.normal) >= Verbosity.debug:
                        _emit_shrink_report(_shrink_rep)
                    if _shrink_rep[2]:
                        _stopped_override = f"shrunk example {_max_shrinks} times"
                    # Attach "Highest target score:" notes when target() was used, so the
                    # falsifying report shows them (test_shows_target_scores_with_multiple_failures).
                    if _targets:
                        for _tl in _describe_targets(_targets):
                            for _fals, _texc in result:
                                if hasattr(_texc, "add_note") and _tl not in (
                                    getattr(_texc, "__notes__", None) or []
                                ):
                                    _texc.add_note(_tl)
                    # Multiple distinct bugs (report_multiple_bugs): assemble an
                    # ExceptionGroup of the per-origin falsifying exceptions, each carrying
                    # its own Falsifying-example note (matching hypothesis's MultipleFailures).
                    if len(result) > 1:
                        # Reproduce each minimal example once (as the single-bug path replays
                        # runner(*falsifying)) so a replayed N-bug run invokes the body twice
                        # per bug — find/replay + reproduce — matching upstream's call count
                        # (test_does_not_shrink_on_replay_with_multiple_bugs). The exceptions
                        # are already captured, so a divergent reproduce is simply ignored.
                        _stats_tls.final_replay = True
                        for _fals, _ in result:
                            if not _has_unreplayable_arg(_fals):
                                try:
                                    runner(*_fals)
                                except BaseException:  # noqa: BLE001 - reproduce only
                                    pass
                        _stats_tls.final_replay = False
                        _raise_multiple_bugs(result, fn, given_names, pnames, active)
                    falsifying, exc = result[0]
                    explain_on = pnames is None or "explain" in pnames
                    note = _falsifying_example_note(
                        fn.__name__, given_names, falsifying, explain_on,
                        varargs=varargs_vals,
                    )
                    # @reproduce_failure(version, blob) line, when print_blob is set.
                    _repro_note = _reproduce_failure_note(active)
                    # Stateful args can't be replayed by re-calling runner(*falsifying): an
                    # interactive data()'s DataObject is frozen after the run, and a
                    # randoms() object's draw-state has advanced (re-drawing would hit the
                    # frozen data). Re-raise the exception run_native already captured,
                    # attaching any note_method_calls recorded during the run (deduped).
                    if _has_unreplayable_arg(falsifying):
                        # A test that suppresses the default given-args note (stateful's
                        # run_state_machine sets _hypothesis_internal_print_given_args=False)
                        # builds its OWN trace instead of `run_state_machine(data=...)`. Drive
                        # a final (is_final=True) replay of the minimal example so the body
                        # re-emits + attaches that trace, then transplant its notes onto exc.
                        _suppress_default = (
                            getattr(fn, "_hypothesis_internal_print_given_args", True) is False
                        )
                        _stateful_flaky: tuple[str, list[Any]] | None = None
                        # A FlakyStrategyDefinition raised by the runner during the replay
                        # (a precondition/draw diverged between generation and replay) is
                        # propagated AS-IS so run_state_machine_as_test can annotate it.
                        _stateful_flaky_def: BaseException | None = None
                        from hypothesis.errors import (
                            FlakyStrategyDefinition as _FlakyStratDef,
                            InvalidDefinition as _InvalidDef,
                        )
                        try:
                            if _suppress_default:
                                _stats_tls.final_replay = True
                                try:
                                    _replayed = _engine.reproduce_native(
                                        runner, list(strategies), _engine.minimal_choices()
                                    )
                                finally:
                                    _stats_tls.final_replay = False
                                if _replayed is None:
                                    # Failed during generation but NOT on the minimal replay
                                    # (e.g. a rule body keyed on is_final): flaky.
                                    _stateful_flaky = (
                                        "Falsified on the first call but did not on a "
                                        "subsequent one",
                                        [exc],
                                    )
                                else:
                                    _rexc = _replayed[1]
                                    if isinstance(_rexc, _FlakyStratDef) and not isinstance(
                                        exc, _InvalidDef
                                    ):
                                        # Replay's rule selection / draws diverged from
                                        # generation (a flaky precondition or is_final-keyed
                                        # draw): propagate the FlakyStrategyDefinition so
                                        # run_state_machine_as_test can annotate it.
                                        _stateful_flaky_def = _rexc
                                    elif not isinstance(_rexc, _FlakyStratDef) and type(
                                        _rexc
                                    ) is not type(exc):
                                        # Replay raised a genuinely different error type (the
                                        # runner's own no-valid-rule FlakyStrategyDefinition,
                                        # when the original was itself InvalidDefinition, just
                                        # means gen and replay agree — fall through to
                                        # transplant + re-raise the original).
                                        _stateful_flaky = (
                                            "Inconsistent results from replaying a test case!",
                                            [exc, _rexc],
                                        )
                                    else:
                                        _exc_notes = getattr(exc, "__notes__", None) or []
                                        for _n in getattr(_rexc, "__notes__", None) or []:
                                            if _n not in _exc_notes:
                                                exc.add_note(_n)
                            else:
                                seen: set = set()
                                for _call in _ns.drain_random_calls():
                                    head = _call.split(" -> ")[0]
                                    if head not in seen:
                                        seen.add(head)
                                        exc.add_note(_call)
                                exc.add_note(note)
                            if _repro_note and _stateful_flaky is None and _stateful_flaky_def is None:
                                exc.add_note(_repro_note)
                        except Exception:  # noqa: BLE001 - <3.11 or non-standard exc
                            pass
                        # Raise OUTSIDE the note-attaching try (which swallows Exception).
                        if _stateful_flaky_def is not None:
                            raise _stateful_flaky_def
                        if _stateful_flaky is not None:
                            _raise_flaky(_stateful_flaky[0], _stateful_flaky[1])
                        _trim_internal_tb(exc, getattr(active, "verbosity", None))
                        raise exc
                    # Replay the minimal example to confirm it reproduces. Hypothesis treats
                    # a divergent replay as a FlakyFailure: a replay that now PASSES means
                    # "Falsified on the first call but did not on a subsequent one"; one that
                    # rejects (assume) or raises a DIFFERENT type means "Inconsistent results
                    # from replaying". A matching replay is the genuine failure (re-raised
                    # with the falsifying-example note).
                    # is_final marks the minimal-example replay so real note() prints.
                    _stats_tls.final_replay = True
                    try:
                        try:
                            runner(*falsifying)
                        except UnsatisfiedAssumption as replayed:
                            _raise_flaky(
                                "Inconsistent results from replaying a test case!",
                                [exc, replayed],
                            )
                        except BaseException as replayed:  # noqa: BLE001 - re-raised below
                            if type(replayed) is not type(exc):
                                _raise_flaky(
                                    "Inconsistent results from replaying a test case!",
                                    [exc, replayed],
                                )
                            try:
                                # Carry over the #3819 sampled_from-of-strategies note that
                                # generation attached to `exc`: the replay passes drawn
                                # values directly, so it doesn't re-draw the sampled_from
                                # (the `indirect` case: sampled_from in a drawn collection).
                                replayed_notes = getattr(replayed, "__notes__", [])
                                for n in getattr(exc, "__notes__", []):
                                    if (
                                        n.startswith(
                                            "sampled_from was given a collection of strategies"
                                        )
                                        and n not in replayed_notes
                                    ):
                                        replayed.add_note(n)
                                replayed.add_note(note)
                                if _repro_note:
                                    replayed.add_note(_repro_note)
                            except Exception:  # noqa: BLE001 - <3.11 or non-standard exc
                                pass
                            _trim_internal_tb(replayed, getattr(active, "verbosity", None))
                            raise replayed
                        # A deadline-exceeded failure that didn't reproduce on replay gets a
                        # deadline-specific "Unreliable test timings!" message, not the generic
                        # one (test_gives_a_deadline_specific_flaky_error_message).
                        from .errors import DeadlineExceeded as _DeadlineExceeded

                        if isinstance(exc, _DeadlineExceeded):
                            def _ms(td: Any) -> float:
                                return td.total_seconds() * 1000 if hasattr(td, "total_seconds") else 0.0

                            _df = _FlakyFailure(
                                "Hypothesis test produced unreliable results", [exc]
                            )
                            try:
                                _df.add_note(
                                    "Unreliable test timings! On an initial run, this test took "
                                    f"{_ms(getattr(exc, 'runtime', None)):.2f}ms, which exceeded the "
                                    f"deadline of {_ms(getattr(exc, 'deadline', None)):.2f}ms, but on a "
                                    "subsequent run it took less time and did not. Either make your "
                                    "test faster, or use settings(deadline=None) to disable the deadline."
                                )
                            except Exception:  # noqa: BLE001 - <3.11 / non-standard exc
                                pass
                            raise _df
                        _raise_flaky(
                            "Falsified on the first call but did not on a subsequent one",
                            [exc],
                        )
                    finally:
                        _stats_tls.final_replay = False
            except _FlakyFailure:
                # A flaky exit is only known here (after the minimal-example replay), so the
                # statistics push was deferred to reflect it (test_flaky_exit).
                _stopped_override = "test was flaky"
                raise
            finally:
                if _stats_payload is not None:
                    _push_statistics(
                        _stats_payload[0],
                        _stats_payload[1],
                        ex_max,
                        _stats_payload[2],
                        stopped_override=_stopped_override,
                    )

    # @functools.wraps copied fn.__dict__ (incl. the inner test's pinned examples)
    # onto wrapper; drop that copy so `inner_examples` isn't double-counted and a
    # genuinely-outer `@example(...) @given(...)` decorator starts from a clean list.
    wrapper.__dict__.pop("_hypothesis_explicit_examples", None)
    # Strip the @given params from the wrapper's signature so pytest doesn't try
    # to inject them as fixtures (it only supplies the remaining self/fixtures). The
    # wrapped test always returns None, so its return annotation becomes None too.
    if outer_params is not None:
        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
            parameters=outer_params, return_annotation=None
        )
    wrapper.is_hypothesis_fast_test = True  # type: ignore[attr-defined]
    # Real-hypothesis attribute name too: pytest-asyncio / pytest-trio gate their async
    # driver on `getattr(func, "is_hypothesis_test", False)` (the literal attribute, not our
    # is_hypothesis_test() helper) before swapping in `wrapper.hypothesis.inner_test`.
    wrapper.is_hypothesis_test = True  # type: ignore[attr-defined]
    wrapper.hypothesis = SimpleNamespace(inner_test=fn)  # type: ignore[attr-defined]
    return wrapper


def _fallback_given(
    fn: Callable[..., Any],
    given_names: list[str],
    strategies: list[Any],
    is_async: bool,
    inner_settings: Any,
) -> Callable[..., Any]:
    from .strategies import _real_hypothesis, _real_strategies

    real_hyp = _real_hypothesis()
    r = _real_strategies()
    real_map = {name: _to_hyp(s, r) for name, s in zip(given_names, strategies)}

    if is_async:

        @functools.wraps(fn)
        def async_target(*a: Any, **k: Any) -> Any:
            return _ensure_loop().run_until_complete(fn(*a, **k))

        async_target.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        target = async_target
    else:
        target = fn

    # In the fallback path the test is managed entirely by real hypothesis, which would
    # reject OUR @settings-without-@given marker as a double-applied @settings. Strip it from
    # both the user function and the real wrapper before real settings runs
    # (test_compatible_nested_shared_strategies_do_not_warn).
    def _strip_settings_marker(obj: Any) -> None:
        try:
            delattr(obj, "_hypothesis_internal_settings_applied")
        except (AttributeError, TypeError):
            pass

    _strip_settings_marker(target)
    wrapped = real_hyp.given(**real_map)(target)
    _strip_settings_marker(wrapped)
    if inner_settings is not None:
        wrapped = real_hyp.settings(
            max_examples=int(inner_settings.max_examples), deadline=None
        )(wrapped)
    wrapped._hp_is_fallback = True  # type: ignore[attr-defined]
    return wrapped


def _article(name: str) -> str:
    return "an" if name[:1].upper() in "AEIOU" else "a"


def _expected_phrase(raises: tuple[type[BaseException], ...]) -> str:
    if raises == (BaseException,):
        return "an exception"
    names = [exc.__name__ for exc in raises]
    if len(names) == 1:
        return f"{_article(names[0])} {names[0]}"
    return f"{_article(names[0])} " + ", ".join(names[:-1]) + ", or " + names[-1]


def _example_repr(ex: Any, given_names: list[str]) -> str:
    parts = [f"{given_names[i]}={v!r}" for i, v in enumerate(ex.args)]
    parts += [f"{k}={v!r}" for k, v in ex.kwargs.items()]
    return f"@example({', '.join(parts)})"


def _example_values_contain_strategy(vals: list[Any]) -> bool:
    """True if any value in `vals` is a SearchStrategy — `@example(text())` is
    almost always a usage mistake (the user meant to draw, not pin the strategy)."""
    from .strategies import SearchStrategy, _hypothesis_base

    base = _hypothesis_base()
    for v in vals:
        if isinstance(v, (SearchStrategy, _engine.SearchStrategy)):
            return True
        if base is not object and isinstance(v, base):
            return True
    return False


def _eval_explicit_example(
    ex: Any, gen_vals: list[Any], given_names: list[str], call_body: Callable[..., Any]
) -> BaseException | None:
    """Run one explicit @example, returning its failure exception (for multi-bug
    collection) or None if it passed / was skipped (assume) / satisfied its xfail.
    An invalid xfail (non-Exception expected) still raises immediately — that's a
    usage error, not a test failure."""
    if ex._xfail is None:
        try:
            call_body(*gen_vals)
        except UnsatisfiedAssumption:
            return None  # assume(False)/reject() in an explicit example -> skip
        except Exception as exc:  # noqa: BLE001 - collected for multi-bug reporting
            # @example(some_strategy)-style misuse: the value is a strategy, not
            # a concrete one. Hypothesis wraps the resulting error in a
            # HypothesisWarning whose __cause__ points at the original — mirror
            # that so callers see the helpful warning, not a bare AssertionError.
            if _example_values_contain_strategy(gen_vals):
                try:
                    from hypothesis.errors import HypothesisWarning  # type: ignore[import-not-found]
                except Exception:  # pragma: no cover - hypothesis absent in this branch
                    return exc
                wrapped = HypothesisWarning(
                    "@example was passed a strategy object instead of a concrete "
                    "value — strategies must be drawn from inside the test, not "
                    "pinned as explicit examples."
                )
                wrapped.__cause__ = exc
                return wrapped
            return exc
        return None
    condition, _reason, raises = ex._xfail
    if not condition:
        try:
            call_body(*gen_vals)
        except UnsatisfiedAssumption:
            return None
        except Exception as exc:  # noqa: BLE001 - collected for multi-bug reporting
            return exc
        return None
    try:
        call_body(*gen_vals)
    except raises as exc:  # the expected failure
        if not isinstance(exc, Exception):
            raise InvalidArgument(
                f"{_example_repr(ex, given_names)} raised an expected {exc!r}, but "
                "Hypothesis does not treat this as a test failure"
            ) from None
        return None  # xfail satisfied
    # body did not raise an expected exception
    return AssertionError(
        f"Expected {_expected_phrase(raises)} from {_example_repr(ex, given_names)}, "
        "but no exception was raised."
    )


def _exc_signature(exc: BaseException) -> tuple[type, int]:
    """(exception type, innermost-frame line number) — hypothesis treats two failures
    as the same bug iff they share an exception type AND a raised-at location."""
    tb = exc.__traceback__
    lineno = -1
    while tb is not None:
        lineno = tb.tb_lineno
        tb = tb.tb_next
    return (type(exc), lineno)


def _example_sort_key(values: tuple[Any, ...]) -> tuple[int, str]:
    """Order explicit examples simplest-first: shorter total repr, then lexicographic — so
    the 'simplest' member of a same-error group (e.g. x=1 over x=100/x=1000) is the one
    reported (test_simplifies_multiple_examples_with_same_error)."""
    r = repr(values)
    return (len(r), r)


def _raise_collected_failures(
    failures: list[tuple[BaseException, tuple[int, str]]],
    report_multiple_bugs: bool,
    verbose: bool = False,
) -> None:
    """Report explicit-example failures. With report_multiple_bugs: group failures by
    interesting origin (type+location), report only the SIMPLEST of each group — noting how
    many other examples shared its error — and raise distinct groups together as an
    ExceptionGroup. In verbose mode every failing example is shown (no same-error dedup),
    matching test_shows_all_examples_at_verbose."""
    excs = [f for f, _ in failures]
    if report_multiple_bugs and verbose:
        if len(excs) > 1:
            raise BaseExceptionGroup(
                "Hypothesis found multiple distinct failing explicit examples", excs
            )
        raise excs[0]
    if report_multiple_bugs:
        groups: dict[tuple[type, int], list[tuple[BaseException, tuple[int, str]]]] = {}
        for f, key in failures:
            groups.setdefault(_exc_signature(f), []).append((f, key))
        chosen: list[BaseException] = []
        for members in groups.values():
            members.sort(key=lambda m: m[1])
            simplest = members[0][0]
            others = len(members) - 1
            if others:
                try:
                    simplest.add_note(
                        f"{others} other explicit example{'s' if others != 1 else ''} "
                        "also failed with the same error"
                    )
                except Exception:  # noqa: BLE001 - <3.11 / non-standard exc
                    pass
            chosen.append(simplest)
        if len(chosen) > 1:
            # BaseExceptionGroup accepts BaseException members and returns an
            # ExceptionGroup instance when they're all Exceptions, so pytest.raises(
            # ExceptionGroup) still matches the common case.
            raise BaseExceptionGroup(
                "Hypothesis found multiple distinct failing explicit examples", chosen
            )
        raise chosen[0]
    raise excs[0]


def is_hypothesis_test(thing: object) -> bool:
    """Whether `thing` is a property-based test (engine-native or fallback)."""
    return bool(
        getattr(thing, "is_hypothesis_fast_test", False)
        or getattr(thing, "is_hypothesis_test", False)
        or getattr(thing, "_hp_is_fallback", False)
    )


class _FindFound(Exception):
    pass


def find(
    specifier: SearchStrategy,
    condition: Callable[[Any], object],
    *,
    settings: Any = None,
    random: Any = None,
    database_key: Any = None,
) -> Any:
    """Return the minimal example of `specifier` for which `condition` is truthy."""
    from .errors import FailedHealthCheck, NoSuchExample

    found: list[Any] = []

    @given(specifier)
    def runner(value: Any) -> None:
        if condition(value):
            found.append(value)
            raise _FindFound

    if settings is not None:
        runner = settings(runner)
    _prev_random = getattr(_random_override, "random", None)
    if random is not None:
        _random_override.random = random
    try:
        try:
            runner()
        except _FindFound:
            return found[-1]
        except FailedHealthCheck:
            # find() is exploratory: a search where every example was rejected is
            # Unsatisfiable, not a filter_too_much health-check failure
            # (test_stops_after_ten_times_max_examples_if_not_satisfying).
            from .errors import Unsatisfiable

            raise Unsatisfiable(
                f"Unable to satisfy the condition for find({specifier!r}, ...): "
                "every example was rejected."
            ) from None
        raise NoSuchExample(f"No example of {specifier!r} satisfied the condition")
    finally:
        _random_override.random = _prev_random


class example:
    """Attach an explicit example to a @given test (run before generated ones).

    Mirrors hypothesis's `example`: callable as a decorator, with chainable
    `.via()` / `.xfail()` builders. `.xfail()` makes the example assert that the
    body raises (one of) the given exception type(s).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and kwargs:
            raise InvalidArgument(
                "Cannot pass both positional and keyword arguments to example()"
            )
        if not args and not kwargs:
            raise InvalidArgument("An example must provide at least one argument")
        self.args = args
        self.kwargs = kwargs
        self._xfail: tuple[bool, str, tuple[type[BaseException], ...]] | None = None
        # populated when this example is decorated onto ANOTHER example object
        # rather than directly onto a test function — applied when the chain
        # finally lands on the test (see `__call__`).
        self._pending: list[example] = []

    def __call__(self, test: Callable[..., Any]) -> Callable[..., Any]:
        # `@example(...) @example(...) def f(): ...` is the obvious case; but the
        # user may also nest by hand: `example("outer")(example("inner"))(f)`.
        # When `test` is itself an example object, we attach ourselves to it and
        # return it — the outer `(f)` call will then re-apply both examples in
        # the right order (inner first, then us). Matches hypothesis' behaviour
        # so test_stop_silently_dropping_examples_… sees both examples on `f`.
        if isinstance(test, example):
            test._pending.append(self)
            return test  # type: ignore[return-value]
        existing = getattr(test, "_hypothesis_explicit_examples", None)
        if existing is None:
            existing = []
            test._hypothesis_explicit_examples = existing  # type: ignore[attr-defined]
        existing.append(self)
        # apply any examples that piggy-backed on us via the isinstance(test, example)
        # branch above — they were waiting for the real test function to land here.
        for pending in self._pending:
            existing.append(pending)
        return test

    def via(self, whence: str) -> example:
        if not isinstance(whence, str):
            raise InvalidArgument(f".via() must be passed a string, got {whence!r}")
        return self

    def xfail(
        self,
        condition: bool = True,
        *,
        reason: str = "",
        raises: type[BaseException] | tuple[type[BaseException], ...] = BaseException,
    ) -> example:
        if not isinstance(condition, bool):
            raise InvalidArgument(f"condition={condition!r} must be a bool")
        if not isinstance(reason, str):
            raise InvalidArgument(f"reason={reason!r} must be a string")
        raises_tuple = raises if isinstance(raises, tuple) else (raises,)
        if not raises_tuple:
            raise InvalidArgument("raises=() must be a (non-empty) exception type or tuple")
        for exc in raises_tuple:
            if not (isinstance(exc, type) and issubclass(exc, BaseException)):
                raise InvalidArgument(f"raises={raises!r} must contain exception types, got {exc!r}")
        self._xfail = (condition, reason, raises_tuple)
        return self


def seed(value: object) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Compatibility no-op: pin a seed. (Determinism not yet wired through.)"""

    def attach(test: Callable[..., Any]) -> Callable[..., Any]:
        return test

    return attach


def reproduce_failure(
    version: str, blob: bytes
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Replay the single example encoded in `blob` (created by encode_failure), to
    reproduce a specific failure. Errors with InvalidArgument if `version` differs
    from ours; raises DidNotReproduce if the blob doesn't reproduce a failure."""

    def attach(test: Callable[..., Any]) -> Callable[..., Any]:
        test._hypothesis_internal_use_reproduce_failure = (version, blob)  # type: ignore[attr-defined]
        return test

    return attach


def _decode_failure_blob(blob: bytes) -> list[Any]:
    """base64 + optional-zlib + flat-choice decode (mirrors hypothesis.core.decode_failure),
    raising InvalidArgument on any malformed input."""
    import base64
    import zlib

    from ._engine import choices_from_bytes

    try:
        decoded = base64.b64decode(blob)
    except Exception:  # noqa: BLE001
        raise InvalidArgument(f"Invalid base64 encoded string: {blob!r}") from None
    prefix, body = decoded[:1], decoded[1:]
    if prefix == b"\0":
        pass
    elif prefix == b"\1":
        try:
            body = zlib.decompress(body)
        except zlib.error as err:
            raise InvalidArgument(f"Invalid zlib compression for blob {blob!r}") from err
    else:
        raise InvalidArgument(
            f"Could not decode blob {blob!r}: Invalid start byte {prefix!r}"
        )
    choices = choices_from_bytes(body)
    if choices is None:
        raise InvalidArgument(f"Invalid serialized choice sequence for blob {blob!r}")
    return list(choices)


def _run_reproduce_failure(
    fn: Callable[..., Any],
    runner: Callable[..., Any],
    strategies: list[Any],
    repro: tuple[str, bytes],
) -> None:
    """Replay the @reproduce_failure blob through the native engine exactly once."""
    from . import __version__
    from .errors import DidNotReproduce

    version, blob = repro
    if version != __version__:
        raise InvalidArgument(
            f"Attempting to reproduce a failure from a different version of Hypothesis. "
            f"This failure is from {version}, but you are currently running {__version__!r}."
        )
    choices = _decode_failure_blob(blob)
    result = _engine.reproduce_native(runner, list(strategies), choices)
    if result is None:
        raise DidNotReproduce(
            "Expected the test to raise an error, but it completed successfully."
        )
    falsifying, exc = result
    rendered = ", ".join(repr(a) for a in falsifying)
    try:
        exc.add_note(f"Falsifying example: {fn.__name__}({rendered})")
    except Exception:  # noqa: BLE001
        pass
    raise exc


settings = _settings_cls
