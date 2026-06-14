"""All-native strategies frontend — a drop-in for `hypothesis.strategies`.

Every constructor returns an engine-native `_engine.SearchStrategy` drawn entirely
in Rust. This is the module the cover suite's `hypothesis.strategies` alias will
eventually point at (replacing the legacy proptest+fallback `strategies.py`). It is
kept separate so the swap can be done and measured deliberately.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable

from hypothesis_fast import _engine as _e


@functools.cache
def _unicode_charmap() -> dict[str, list[tuple[int, int]]]:
    """Map each 2-letter general-category code to merged codepoint intervals, built from the
    RUNNING interpreter's ``unicodedata`` (called once by the Rust engine's charmap, cached).

    Derived from `unicodedata.category` rather than a fixed table baked into the extension, so
    `characters(categories=...)` agrees with `unicodedata.category` on whatever Unicode version
    THIS CPython ships (it differs by release: 3.11→14.0, 3.12→15.0, 3.13→15.1, 3.14→16.0).
    A fixed table can't match every version at once. `chr()`/`unicodedata.category` handle the
    surrogate range (U+D800..U+DFFF → 'Cs') directly, so no special-casing is needed.
    """
    import unicodedata

    cat = unicodedata.category
    out: dict[str, list[tuple[int, int]]] = {}
    last = 0x10FFFF
    cur = cat(chr(0))
    start = 0
    cp = 1
    while cp <= last + 1:
        c = cat(chr(cp)) if cp <= last else ""
        if c != cur:
            out.setdefault(cur, []).append((start, cp - 1))
            cur = c
            start = cp
        cp += 1
    return out


class _PrettyIter:
    """The value drawn by st.iterables(): an iterator over `_values` with a useful repr
    (`iter([...])`), matching hypothesis (test_iterables_repr_is_useful)."""

    def __init__(self, values: Any) -> None:
        self._values = values
        self._it = iter(values)

    def __iter__(self) -> "_PrettyIter":
        return self

    def __next__(self) -> Any:
        return next(self._it)

    def __repr__(self) -> str:
        return f"iter({self._values!r})"


# note_method_calls buffer: a randoms(note_method_calls=True) object records each method
# call here; the native @given wrapper drains it and attaches the calls to a falsifying
# example's exception notes (see core._native_engine_given).
_RANDOM_CALL_LOG = threading.local()


def _record_random_call(msg: str) -> None:
    log = getattr(_RANDOM_CALL_LOG, "calls", None)
    if log is None:
        log = []
        _RANDOM_CALL_LOG.calls = log
    log.append(msg)


def reset_random_calls() -> None:
    _RANDOM_CALL_LOG.calls = []


def drain_random_calls() -> list:
    log = getattr(_RANDOM_CALL_LOG, "calls", None)
    _RANDOM_CALL_LOG.calls = []
    return log or []


# Draw-time events buffer: a native .filter() that retries records its
# "Retried draw from <repr> to satisfy filter" event here (upstream writes the same to
# ConjectureData.events). Argument generation happens in Rust BEFORE the per-example
# build context exists, so these can't go through control.event(); the @given runner
# drains them into the per-test-case statistics events (test_has_lambdas_in_output).
_DRAW_EVENTS = threading.local()


def _record_draw_event(msg: str) -> None:
    log = getattr(_DRAW_EVENTS, "events", None)
    if log is None:
        log = set()
        _DRAW_EVENTS.events = log
    log.add(msg)


def drain_draw_events() -> list:
    log = getattr(_DRAW_EVENTS, "events", None)
    if not log:
        return []
    _DRAW_EVENTS.events = set()
    return sorted(log)


SearchStrategy = _e.SearchStrategy

# --- strategy construction cache (upstream @cacheable) -----------------------------
# Constructors with identical arguments return the SAME strategy object, so
# `text() is text()` (test_caching). Key = (constructor, arg value-keys, kwarg
# value-keys); floats are keyed by their bit pattern so 0.0, -0.0 and int 0 are
# distinct (test_caches_floats_sensitively). Unhashable args/kwargs skip the cache
# rather than erroring (test_does_not_error_on_unhashable_kwarg). Once-per-construction
# frontend glue, not a per-draw hot path.
_STRATEGY_CACHE: dict[Any, Any] = {}
_MISSING = object()


def _value_key(v: Any) -> Any:
    import struct

    if isinstance(v, float):
        return (float, struct.unpack("<q", struct.pack("<d", v))[0])
    return (type(v), v)


def _cacheable(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def cached(*args: Any, **kwargs: Any) -> Any:
        try:
            cache_key = (
                fn,
                tuple(_value_key(v) for v in args),
                frozenset((k, _value_key(v)) for k, v in kwargs.items()),
            )
            hash(cache_key)
        except TypeError:
            # an unhashable arg/kwarg can't be cached — just build it
            return fn(*args, **kwargs)
        cached_result = _STRATEGY_CACHE.get(cache_key, _MISSING)
        if cached_result is not _MISSING:
            return cached_result
        result = fn(*args, **kwargs)
        if getattr(result, "is_cacheable", True):
            _STRATEGY_CACHE[cache_key] = result
        return result

    return cached


integers = _cacheable(_e.integers)
booleans = _cacheable(_e.booleans)
def _floats(
    *args: Any, exclude_min: Any = False, exclude_max: Any = False, **kwargs: Any
) -> Any:
    # exclude_min/exclude_max must be bools. pyo3 collapses an explicit `None` into a missing
    # Option arg, so the Rust engine can't tell exclude_min=None from omitted — validate it in
    # the Python frontend, where it can (test_validates_keyword_arguments[floats(exclude_min=None)]).
    from .errors import InvalidArgument

    for _name, _val in (("exclude_min", exclude_min), ("exclude_max", exclude_max)):
        if not isinstance(_val, bool):
            raise InvalidArgument(f"Expected {_name} to be a bool, but got {_val!r}")
    return _e.floats(*args, exclude_min=exclude_min, exclude_max=exclude_max, **kwargs)


floats = _cacheable(_floats)
none = _cacheable(_e.none)
just = _cacheable(_e.just)
nothing = _cacheable(_e.nothing)
sampled_from = _cacheable(_e.sampled_from)
one_of = _cacheable(_e.one_of)
tuples = _cacheable(_e.tuples)
lists = _cacheable(_e.lists)
sets = _cacheable(_e.sets)
frozensets = _cacheable(_e.frozensets)
dictionaries = _cacheable(_e.dictionaries)
fixed_dictionaries = _cacheable(_e.fixed_dictionaries)
def _text(alphabet: Any = None, *, min_size: Any = None, max_size: Any = None) -> Any:
    # A builds()/map()/etc. alphabet can't be resolved to a static character set the way
    # str/list/characters()/just()/sampled_from() can — draw from it per character and
    # validate each is a single-character string (test_validates_keyword_arguments).
    if isinstance(alphabet, _e.SearchStrategy):
        try:
            _e._regex_alphabet_intervals(alphabet)
        except Exception:  # noqa: BLE001 - not a static char-set strategy (builds/map/...)
            return _text_from_char_strategy(alphabet, min_size, max_size)
    return _e.text(alphabet, min_size=min_size, max_size=max_size)


def _text_from_char_strategy(alphabet: Any, min_size: Any, max_size: Any) -> Any:
    """text() over a strategy alphabet that isn't a static char-set (builds/map/...): draw a
    list of values from it and join, validating each is a single-character string at draw time
    (mirrors upstream's OneCharStringStrategy)."""
    from .errors import InvalidArgument

    def _join(chars: Any) -> str:
        out: list[str] = []
        for ch in chars:
            if not isinstance(ch, str):
                raise InvalidArgument(f"Got non-string {ch!r} (type {type(ch).__name__})")
            if len(ch) != 1:
                raise InvalidArgument(f"Got {ch!r} (length {len(ch)} != 1)")
            out.append(ch)
        return "".join(out)

    return lists(alphabet, min_size=min_size or 0, max_size=max_size).map(_join)


text = _cacheable(_text)
characters = _cacheable(_e.characters)
binary = _cacheable(_e.binary)
uuids = _cacheable(_e.uuids)
permutations = _cacheable(_e.permutations)
# builds() is NOT cached: it infers missing args via from_type, so its result depends on
# the mutable type registry. Caching would return a registry-independent resolution built
# in an earlier context (the native cache isn't cleared by real-hypothesis's
# clear_strategy_cache that temp_registered calls), breaking test_generic_origin_*.
builds = _e.builds
dates = _cacheable(_e.dates)


def _attach_timezone(base: Any, timezones: Any, allow_imaginary: Any) -> Any:
    """Compose a naive-datetime/time strategy with a drawn timezone: draw (value, tz) and
    attach via `.replace(tzinfo=tz)`. The native datetimes()/times() generate naive values
    (tz handling lives in Python, where tz objects do); this honors the `timezones=` arg
    that the engine otherwise drops. With allow_imaginary False, filter out datetimes that
    don't exist in their timezone (DST gaps), matching upstream's datetime_does_not_exist."""
    from datetime import timezone as _timezone

    def _attach(pair: Any) -> Any:
        value, tz = pair
        return value.replace(tzinfo=tz)

    strat = _e.tuples(base, timezones).map(_attach)
    if allow_imaginary is False:

        def _exists(d: Any) -> bool:
            tz = d.tzinfo
            if tz is None:
                return True
            try:
                roundtrip = d.astimezone(_timezone.utc).astimezone(tz)
            except (OverflowError, OSError, ValueError):
                return False
            return d == roundtrip

        strat = strat.filter(_exists)
    return strat


def _datetimes(
    min_value: Any = None,
    max_value: Any = None,
    *,
    timezones: Any = None,
    allow_imaginary: Any = None,
) -> Any:
    # allow_imaginary must be a bool; pyo3 collapses an explicit None into a missing arg, so
    # the frontend validates it (test_validates_keyword_arguments[datetimes(allow_imaginary=0)]).
    if allow_imaginary is not None and not isinstance(allow_imaginary, bool):
        from .errors import InvalidArgument

        raise InvalidArgument(
            f"allow_imaginary={allow_imaginary!r} must be a boolean (or None)."
        )
    base = _e.datetimes(min_value=min_value, max_value=max_value)
    if timezones is None:
        return base
    return _attach_timezone(base, timezones, allow_imaginary)


def _times(
    min_value: Any = None,
    max_value: Any = None,
    *,
    timezones: Any = None,
) -> Any:
    base = _e.times(min_value=min_value, max_value=max_value)
    if timezones is None:
        return base
    return _attach_timezone(base, timezones, None)


times = _cacheable(_times)
datetimes = _cacheable(_datetimes)
timedeltas = _cacheable(_e.timedeltas)
ip_addresses = _cacheable(_e.ip_addresses)
slices = _cacheable(_e.slices)
fractions = _cacheable(_e.fractions)
# Lazy / stateful constructors are NOT cached (each call must build fresh state).
recursive = _e.recursive
deferred = _e.deferred
data = _e.data
iterables = _e.iterables
emails = _e.emails

# The reflective native exports (from_type/randoms/from_regex/register_type_strategy/
# check_strategy) are wired at the END of this module, after the helpers they reference
# are defined.


# Native type registry consulted by the Rust `from_type` (it looks up
# `native_strategies._NATIVE_TYPE_REGISTRY`). Maps a type/typing-thing to either a
# native SearchStrategy or a factory `(type) -> SearchStrategy`. Kept native so
# `register_type_strategy(T, native_strat)` doesn't hit real-hypothesis's validator
# (which rejects native objects via isinstance(real SearchStrategy)).
_NATIVE_TYPE_REGISTRY: dict[Any, Any] = {}

# The subset of _NATIVE_TYPE_REGISTRY keys registered by the USER (register_type_strategy /
# temp_registered), as opposed to the built-in/abc fallbacks pre-populated by
# _populate_native_registry. Only USER registrations pre-empt built-in container handling
# for a parametrized-generic request (see _resolve_generic_subtypes), so the element-aware
# abc fallbacks (Iterator[str] etc.) are not shadowed.
_USER_REGISTERED: set[Any] = set()
# Cache of (len(_USER_REGISTERED), [generic types in it]). Only generic registered types can
# be subtypes of a parametrized from_type request, so _resolve_generic_subtypes scans this
# subset instead of the whole registry; recomputed when the registry size changes.
_GENERIC_SUBSET_CACHE: tuple[int, list[Any]] | None = None


def _check_strategy(arg: Any, name: str = "") -> None:
    """Native `check_strategy`: accept a native SearchStrategy, mirroring hypothesis's
    error shape otherwise (incl. the sampled_from hint for list/tuple)."""
    from hypothesis_fast.errors import InvalidArgument

    assert isinstance(name, str)
    if not isinstance(arg, SearchStrategy):
        hint = ""
        if isinstance(arg, (list, tuple)):
            hint = ", such as st.sampled_from({}),".format(name or "...")
        if name:
            name += "="
        raise InvalidArgument(
            f"Expected a SearchStrategy{hint} but got {name}{arg!r} "
            f"(type={type(arg).__name__})"
        )


def _is_a_type(thing: Any) -> bool:
    """Mirror hypothesis's `types.is_a_type`: a plain class, a generic alias, or a
    typing special form (Union/Optional/etc.) or TypeVar."""
    import typing as _typing

    if isinstance(thing, type):
        return True
    if _typing.get_origin(thing) is not None:
        return True
    # TypeVar, generic alias (__args__), a NewType (__supertype__), or a ForwardRef
    # (registrable so `register_type_strategy(ForwardRef("A"), ...)` works) are accepted.
    return (
        isinstance(thing, (_typing.TypeVar, _typing.ForwardRef))
        or hasattr(thing, "__args__")
        or hasattr(thing, "__supertype__")
        # PEP 695 type alias (`type A = int`): a TypeAliasType, identified by carrying
        # both __value__ and __type_params__, is registrable (test_can_register_typealias).
        or (hasattr(thing, "__value__") and hasattr(thing, "__type_params__"))
    )


def _strategy_types() -> tuple:
    """isinstance tuple accepting our native SearchStrategy AND, when available, the real
    hypothesis SearchStrategy — real strategies are now drawable via the interop bridge, so
    they're valid registry values (e.g. `register_type_strategy(T, real_strategy)`)."""
    try:
        from hypothesis.strategies._internal.strategies import SearchStrategy as _RealSS

        return (SearchStrategy, _RealSS)
    except Exception:  # noqa: BLE001 - real hypothesis absent
        return (SearchStrategy,)


def _register_type_strategy(thing: Any, strategy: Any) -> Any:
    """Native `register_type_strategy`: store a native strategy/factory in the native
    registry (no real-hypothesis isinstance validation, which would reject natives)."""
    import typing as _typing

    from hypothesis_fast.errors import InvalidArgument

    if not _is_a_type(thing):
        raise InvalidArgument(f"thing={thing!r} must be a type")
    # A parametrized generic with concrete (non-TypeVar) args can't be registered —
    # hypothesis directs you to register a function for its origin instead. All-TypeVar
    # args (e.g. list[T]) ARE allowed.
    args = _typing.get_args(thing)
    if args and not all(isinstance(a, _typing.TypeVar) for a in args):
        raise InvalidArgument(
            f"Cannot register generic type {thing!r}, because it has type arguments "
            f"which would not be handled.  Instead, register a function for "
            f"{_typing.get_origin(thing)!r} which can inspect specific type objects "
            "and return a strategy."
        )
    # Accept a native SearchStrategy, a factory, OR a real-hypothesis SearchStrategy
    # (now drawable via the interop bridge — e.g. registering `types._global_type_lookup[set]`
    # in test_register_generic_typing_strats).
    _any_strategy = _strategy_types()
    if not isinstance(strategy, _any_strategy) and not callable(strategy):
        raise InvalidArgument(
            f"strategy={strategy!r} must be a SearchStrategy, or a function that takes "
            "a generic type and returns a specific SearchStrategy"
        )
    if isinstance(strategy, _any_strategy) and getattr(strategy, "is_empty", False):
        raise InvalidArgument(f"Cannot register empty strategy {strategy!r}")
    _NATIVE_TYPE_REGISTRY[thing] = strategy
    _USER_REGISTERED.add(thing)
    # Also key on the origin so `register_type_strategy(typing.Sequence, ...)` resolves
    # `Sequence[int]` whose `get_origin` is `collections.abc.Sequence` (typing.Sequence is
    # NOT abc.Sequence). Upstream stores under both `type_` and `get_origin(type_) or type_`.
    # `temp_registered` restores this origin key too (so it doesn't leak across tests).
    origin = _typing.get_origin(thing)
    if origin is not None and origin is not thing:
        _NATIVE_TYPE_REGISTRY[origin] = strategy
        _USER_REGISTERED.add(origin)
    # NOTE: external code that reads `types._global_type_lookup` DIRECTLY (e.g. the returns
    # plugin's look_up_strategy/law machinery) needs our registrations mirrored there. That
    # mirror is done in the HF_SHIM (_external/shim/sitecustomize.py), NOT here — writing native
    # strategies into the real lookup pollutes the parity suite's own real-resolution paths
    # (abc subtype resolution reads _global_type_lookup and chokes on a native strategy).
    # Registry changed → drop the cached generic-subset (recomputed lazily in
    # _resolve_generic_subtypes). With its length check this catches both adds (here) and
    # removes (len shrinks via temp_registered/clear), so an equal-length content swap
    # across tests can't yield a stale subset.
    global _GENERIC_SUBSET_CACHE
    _GENERIC_SUBSET_CACHE = None
    # The native from_type/builds resolution caches depend on the registry too — drop them
    # so a resolution computed under the old registry isn't reused (native, not a fallback).
    _e._clear_resolution_caches()
    return strategy


def _resolve_entry(entry: Any, thing: Any) -> Any:
    """Resolve a registry entry for request `thing`: a SearchStrategy is returned as-is; a
    factory is called with the full requested type (so it can inspect get_args/get_origin) —
    NotImplemented means "declined" (-> None), a non-strategy result is ignored (-> None)."""
    _any = _strategy_types()
    if isinstance(entry, _any):
        return entry
    result = entry(thing)
    if result is NotImplemented or not isinstance(result, _any):
        return None
    return result


def _resolve_generic_subtypes(thing: Any) -> Any:
    """Port of upstream from_typing_type's registry resolution for a PARAMETRIZED generic
    (e.g. Sequence[int], Container[str]). Engages ONLY when a USER-registered generic is a
    subtype of the request's origin (else None -> the Rust built-in element-aware handling
    runs). Builds a mapping of {registered-or-builtin generic subtype -> strategy}, applies
    the maximal filter (drop t when a registered supertype of t is also in the mapping, so a
    registered abstract type masks its concrete subtypes), and one_of's the survivors."""
    import typing as _typing

    origin = _typing.get_origin(thing)
    if not isinstance(origin, type):
        return None

    def k_origin(k: Any) -> Any:
        return _typing.get_origin(k) or k

    # Engage only when a USER-registered GENERIC type is a subtype of the REQUEST (the full
    # parametrized alias). Use upstream's `try_issubclass`, which checks generic-arg
    # compatibility — so a bare tuple subclass is NOT a subtype of Sequence[int] (issue
    # #3767) while CustomContainer(Container[T]) IS a subtype of Container[str].
    try:
        from hypothesis.strategies._internal.types import (
            is_generic_type as _is_generic,
            try_issubclass as _try_issub,
        )
    except Exception:  # noqa: BLE001 - real hypothesis absent
        return None

    # Only generic registered types can be subtypes of a parametrized request — scan the
    # cached generic subset instead of _is_generic()-checking the WHOLE registry on every
    # call (hot for from_type-heavy code registering many non-generic types, e.g. libcst:
    # was ~25% of generation time). Recompute the subset only when the registry size changes.
    global _GENERIC_SUBSET_CACHE
    _n = len(_USER_REGISTERED)
    if _GENERIC_SUBSET_CACHE is None or _GENERIC_SUBSET_CACHE[0] != _n:
        _GENERIC_SUBSET_CACHE = (_n, [k for k in _USER_REGISTERED if _is_generic(k)])
    user_subs = [k for k in _GENERIC_SUBSET_CACHE[1] if _try_issub(k, thing)]
    if not user_subs:
        return None

    args = _typing.get_args(thing)

    def el(i: int) -> Any:
        return _e.from_type(args[i]) if i < len(args) else one_of(
            _e.none(), _e.booleans(), _e.integers(), _e.floats(), _e.text()
        )

    # Built-in container generics, element-aware on `thing`'s args (so the maximal filter
    # has the concrete subtypes to mask). The strategy only matters if it survives maximal.
    builtin = {
        list: lambda: _e.lists(el(0)),
        set: lambda: _e.sets(el(0)),
        frozenset: lambda: _e.frozensets(el(0)),
        tuple: lambda: _e.lists(el(0)).map(tuple),
        dict: lambda: _e.dictionaries(el(0), el(1)),
    }
    mapping = {}
    for k, build in builtin.items():
        if _try_issub(k, thing):
            mapping[k] = build()
    for k in user_subs:
        ko = k_origin(k)
        s = _resolve_entry(_NATIVE_TYPE_REGISTRY[k], thing)
        if s is not None:
            mapping[ko] = s  # a registered subtype overrides the built-in for that origin

    # Drop tuple-subtypes (incl. namedtuples) when any non-tuple alternative exists — a
    # tuple subclass is not treated as a generic sequence (issue #3767). Mirrors upstream.
    import collections as _collections
    import collections.abc as _abc2

    tuple_types = [
        t
        for t in mapping
        if (isinstance(t, type) and issubclass(t, tuple)) or _typing.get_origin(t) is tuple
    ]
    if len(mapping) > len(tuple_types):
        for tt in tuple_types:
            mapping.pop(tt, None)
    if {dict, set}.intersection(mapping):
        mapping.pop(_abc2.ItemsView, None)
    if _collections.deque in mapping and len(mapping) > 1:
        mapping.pop(_collections.deque, None)

    keys = list(mapping)
    maximal = [t for t in keys if sum(1 for tt in keys if _safe_issubclass(t, tt)) == 1]
    chosen = [mapping[t] for t in maximal]
    if not chosen:
        return None
    return chosen[0] if len(chosen) == 1 else one_of(*chosen)


def _safe_issubclass(a: Any, b: Any) -> bool:
    try:
        return issubclass(a, b)
    except TypeError:
        return False


def _filter_makes_empty(
    func: Any, kind: str, lo: Any, hi: Any, allow_nan: bool, allow_inf: bool
) -> bool:
    """CONSERVATIVE filter-rewriting emptiness: True only when `base.filter(func)` is
    PROVABLY unsatisfiable for a numeric base (int / float in [lo, hi]); False on any
    uncertainty (so it can only fix tests, never wrongly mark a satisfiable filter
    empty). Handles math.isinf/isnan and functools.partial(operator.OP, N)."""
    import math
    import operator
    from functools import partial
    from numbers import Real

    if func is math.isnan:
        return True if kind == "int" else (not allow_nan)
    if func is math.isinf:
        if kind == "int":
            return True
        # a float +/-inf is producible only if a bound is infinite AND inf is allowed
        return not (allow_inf and (lo is None or hi is None or lo == -math.inf or hi == math.inf))
    if not (isinstance(func, partial) and not func.keywords and len(func.args) == 1):
        return False
    op = func.func
    (n,) = func.args
    if isinstance(n, bool) or not isinstance(n, Real):
        return False  # can't reason numerically (e.g. eq against a string)
    if isinstance(n, float) and math.isnan(n):
        return op is not operator.ne  # any comparison with NaN is False
    inf = math.inf
    blo = -inf if lo is None else lo
    bhi = inf if hi is None else hi
    # {x : op(n, x)} as an interval (low, low_incl, high, high_incl)
    table = {
        operator.lt: (n, False, inf, False),  # n < x
        operator.le: (n, True, inf, False),  # n <= x
        operator.gt: (-inf, False, n, False),  # n > x
        operator.ge: (-inf, False, n, True),  # n >= x
        operator.eq: (n, True, n, True),  # x == n
    }
    if op not in table:
        return False  # ne / unknown -> don't claim empty
    clo, cli, chi, chi_in = table[op]
    if clo > blo:
        low, low_in = clo, cli
    elif clo < blo:
        low, low_in = blo, True
    else:
        low, low_in = blo, cli
    if chi < bhi:
        high, high_in = chi, chi_in
    elif chi > bhi:
        high, high_in = bhi, True
    else:
        high, high_in = bhi, chi_in
    if low > high:
        return True
    if low == high and not (low_in and high_in):
        return True
    if kind == "int":
        lo_i = math.ceil(low) if low_in else math.floor(low) + 1
        hi_i = math.floor(high) if high_in else math.ceil(high) - 1
        return lo_i > hi_i
    # float: a single required point is reachable only if it's exactly a float
    if low == high:
        return float(n) != n
    return False


def _any_callable(*args: Any, **kwargs: Any) -> None:
    """A flexible `like` for functions() built from Callable[...] (accepts any args), so
    the generated callable can be called however the annotation implies."""


def _arity_callable(n: int) -> Any:
    """A `like` for functions() with exactly `n` positional parameters, so a function
    generated for `Callable[[t1, ..., tn], R]` rejects calls with the wrong argument
    count (e.g. `Callable[[], R]` makes `f(1)` raise TypeError)."""
    import inspect

    def _f(*args: Any, **kwargs: Any) -> None: ...

    fn: Any = _f
    fn.__signature__ = inspect.Signature(
        [
            inspect.Parameter(f"a{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for i in range(n)
        ]
    )
    return fn


def _make_generator(pair: Any) -> Any:
    """Assemble the generator drawn for `from_type(Generator[Y, S, R])`: yield each drawn
    Y value, then return the drawn R (surfaced through StopIteration.value)."""
    values, retval = pair

    def _gen() -> Any:
        yield from values
        return retval

    return _gen()


def _state_for_seed(cd: Any, seed: Any) -> Any:
    """Per-cd RandomState for a seed (shared across randoms drawn from one st.data(),
    since they wrap the same cd) — so seeding two of them alike synchronises them."""
    from hypothesis.strategies._internal.random import RandomState

    sts = getattr(cd, "seeds_to_states", None)
    if sts is None:
        sts = {}
        cd.seeds_to_states = sts
    st = sts.get(seed)
    if st is None:
        st = RandomState()
        sts[seed] = st
    return st


@functools.lru_cache(maxsize=1)
def _artificial_random_cls() -> Any:
    """Build the native ArtificialRandom class lazily (imported here, not at module
    load, to avoid a circular import with real hypothesis under the conftest alias).
    Subclasses the REAL HypothesisRandom — reusing its per-method bindings, signatures
    and convert_kwargs — and only overrides the data-drawing core with NATIVE strategies
    plus the RandomState determinism machine (seed/getstate/setstate/copy)."""
    import math

    from hypothesis.strategies._internal.random import (
        HypothesisRandom,
        RandomState,
        convert_kwargs,
        normalize_zero,
    )

    class _ArtificialRandom(HypothesisRandom):  # type: ignore[misc]
        def __init__(self, *, note_method_calls: bool, data: Any) -> None:
            super().__init__(note_method_calls=note_method_calls)
            self._data = data
            self._state = RandomState()

        def __repr__(self) -> str:
            return "HypothesisRandom(generated data)"

        def __copy__(self) -> Any:
            result = _ArtificialRandom(note_method_calls=self._note_method_calls, data=self._data)
            result.setstate(self.getstate())
            return result

        def _hypothesis_log_random(self, method: Any, kwargs: Any, result: Any) -> None:
            if not self._note_method_calls:
                return
            args, kw = convert_kwargs(method, kwargs)
            argstr = ", ".join(list(map(repr, args)) + [f"{k}={v!r}" for k, v in kw.items()])
            _record_random_call(f"{self!r}.{method}({argstr}) -> {result!r}")

        def __convert_result(self, method: Any, kwargs: Any, result: Any) -> Any:
            if method == "choice":
                return kwargs.get("seq")[result]
            if method in ("choices", "sample"):
                seq = kwargs["population"]
                return [seq[i] for i in result]
            if method == "shuffle":
                seq = kwargs["x"]
                original = list(seq)
                for i, i2 in enumerate(result):
                    seq[i] = original[i2]
                return None
            return result

        def _hypothesis_do_random(self, method: Any, kwargs: Any) -> Any:
            if method == "choices":
                key = (method, len(kwargs["population"]), kwargs.get("k"))
            elif method == "choice":
                key = (method, len(kwargs["seq"]))
            elif method == "shuffle":
                key = (method, len(kwargs["x"]))
            else:
                key = (method, *sorted(kwargs))

            try:
                result, self._state = self._state.next_states[key]
            except KeyError:
                pass
            else:
                return self.__convert_result(method, kwargs, result)

            data = self._data
            if method == "_randbelow":
                result = data.draw(integers(0, kwargs["n"] - 1))
            elif method == "random":
                result = data.draw(floats(0, 1, exclude_max=True))
            elif method == "betavariate":
                result = data.draw(floats(0, 1))
            elif method == "uniform":
                a = normalize_zero(kwargs["a"])
                b = normalize_zero(kwargs["b"])
                result = data.draw(floats(a, b))
            elif method in ("weibullvariate", "gammavariate"):
                result = data.draw(floats(min_value=0.0, allow_infinity=False))
            elif method in ("gauss", "normalvariate"):
                mu = kwargs["mu"]
                result = mu + data.draw(floats(allow_nan=False, allow_infinity=False))
            elif method == "vonmisesvariate":
                result = data.draw(floats(0, 2 * math.pi))
            elif method == "randrange":
                if kwargs["stop"] is None:
                    stop = kwargs["start"]
                    start = 0
                else:
                    start = kwargs["start"]
                    stop = kwargs["stop"]
                step = kwargs["step"]
                if start == stop:
                    raise ValueError(f"empty range for randrange({start}, {stop}, {step})")
                if step != 1:
                    endpoint = (stop - start) // step
                    if (start - stop) % step == 0:
                        endpoint -= 1
                    i = data.draw(integers(0, endpoint))
                    result = start + i * step
                else:
                    result = data.draw(integers(start, stop - 1))
            elif method == "randint":
                result = data.draw(integers(kwargs["a"], kwargs["b"]))
            elif method == "binomialvariate":
                result = data.draw(integers(0, kwargs["n"]))
            elif method == "choice":
                seq = kwargs["seq"]
                result = data.draw(integers(0, len(seq) - 1))
            elif method == "choices":
                k = kwargs["k"]
                result = data.draw(
                    lists(integers(0, len(kwargs["population"]) - 1), min_size=k, max_size=k)
                )
            elif method == "sample":
                k = kwargs["k"]
                seq = kwargs["population"]
                if k > len(seq) or k < 0:
                    raise ValueError(f"Sample size {k} not in expected range 0 <= k <= {len(seq)}")
                if k == 0:
                    result = []
                else:
                    result = data.draw(
                        lists(sampled_from(range(len(seq))), min_size=k, max_size=k, unique=True)
                    )
            elif method == "getrandbits":
                result = data.draw(integers(0, 2 ** kwargs["n"] - 1))
            elif method == "triangular":
                low = normalize_zero(kwargs["low"])
                high = normalize_zero(kwargs["high"])
                mode = normalize_zero(kwargs["mode"]) if kwargs["mode"] is not None else None
                if mode is None:
                    result = data.draw(floats(low, high))
                elif data.draw(booleans()):
                    result = data.draw(floats(mode, high))
                else:
                    result = data.draw(floats(low, mode))
            elif method in ("paretovariate", "expovariate", "lognormvariate"):
                result = data.draw(floats(min_value=0.0))
            elif method == "shuffle":
                result = data.draw(permutations(range(len(kwargs["x"]))))
            elif method == "randbytes":
                n = int(kwargs["n"])
                result = data.draw(binary(min_size=n, max_size=n))
            else:
                raise NotImplementedError(method)

            new_state = RandomState()
            self._state.next_states[key] = (result, new_state)
            self._state = new_state
            return self.__convert_result(method, kwargs, result)

        def seed(self, seed: Any) -> None:
            self._state = _state_for_seed(self._data.cd, seed)

        def getstate(self) -> Any:
            if self._state.state_id is not None:
                return self._state.state_id
            cd = self._data.cd
            sfi = getattr(cd, "states_for_ids", None)
            if sfi is None:
                sfi = {}
                cd.states_for_ids = sfi
            self._state.state_id = len(sfi)
            sfi[self._state.state_id] = self._state
            return self._state.state_id

        def setstate(self, state: Any) -> None:
            self._state = self._data.cd.states_for_ids[state]

    return _ArtificialRandom


def _byte_category(cat: Any) -> set[int]:
    """The set of byte values matched by a regex category opcode in bytes mode
    (ASCII semantics, matching how CPython's re treats \\d/\\s/\\w on bytes)."""
    import re

    c = re._constants
    digit = set(range(48, 58))
    space = {9, 10, 11, 12, 13, 32}
    word = set(range(48, 58)) | set(range(65, 91)) | set(range(97, 123)) | {95}
    full = set(range(256))
    return {
        c.CATEGORY_DIGIT: digit,
        c.CATEGORY_NOT_DIGIT: full - digit,
        c.CATEGORY_SPACE: space,
        c.CATEGORY_NOT_SPACE: full - space,
        c.CATEGORY_WORD: word,
        c.CATEGORY_NOT_WORD: full - word,
    }.get(cat, set())


def _byte_choice(allowed: set[int]) -> Any:
    """A 1-byte strategy drawing from `allowed` (any byte if empty)."""
    values = sorted(allowed) if allowed else list(range(256))
    return sampled_from(values).map(lambda i: bytes([i]))


def _ignorecase_variants(av: int, is_bytes: bool) -> Any:
    """The single-char literals matching codepoint/byte `av` under re.IGNORECASE:
    [c] or [c, c.swapcase()] (only when the swap is a genuine case-insensitive match)."""
    import re

    if is_bytes:
        b = bytes([av])
        sw = b.swapcase()
        return [b, sw] if sw != b and len(sw) == 1 else [b]
    ch = chr(av)
    sw = ch.swapcase()
    if sw != ch and len(sw) == 1 and re.match(re.escape(ch), sw, re.IGNORECASE) is not None:
        return [ch, sw]
    return [ch]


def _regex_charset(items: Any, is_bytes: bool, flags: int) -> Any:
    """A 1-char strategy for a regex character set (the IN opcode's items)."""
    import re

    c = re._constants
    ignorecase = bool(flags & re.IGNORECASE)
    conv = (lambda v: bytes([v])) if is_bytes else chr
    negate = bool(items) and items[0][0] == c.NEGATE
    body = items[1:] if negate else items
    if is_bytes:
        allowed: set[int] = set()
        for op, av in body:
            if op == c.LITERAL:
                allowed.add(av)
                if ignorecase:
                    allowed.update(bytes([av]).swapcase())  # iterating bytes yields ints
            elif op == c.RANGE:
                allowed.update(range(av[0], min(av[1], 255) + 1))
            elif op == c.CATEGORY:
                allowed |= _byte_category(av)
        if negate:
            allowed = set(range(256)) - allowed
        return _byte_choice(allowed)
    # Precise \d \w \s (and \D \W \S): a charset of exactly one category opcode. Build the
    # native characters() spec directly via the (not-category XOR outer-negate) duality, so
    # `\s` yields exactly whitespace, `\D` exactly non-digits, honouring re.ASCII. Mirrors
    # upstream CharactersBuilder.add_category.
    if len(body) == 1 and body[0][0] == c.CATEGORY:
        ascii_mode = bool(flags & re.ASCII)
        space_chars = " \t\n\r\f\v"
        space_white = space_chars if ascii_mode else space_chars + "\x1c\x1d\x1e\x1f\x85"
        spec = {
            c.CATEGORY_DIGIT: (False, ["Nd"], ""),
            c.CATEGORY_NOT_DIGIT: (True, ["Nd"], ""),
            c.CATEGORY_SPACE: (False, ["Z"], space_white),
            c.CATEGORY_NOT_SPACE: (True, ["Z"], space_white),
            c.CATEGORY_WORD: (False, ["L", "N"], "_"),
            c.CATEGORY_NOT_WORD: (True, ["L", "N"], "_"),
        }.get(body[0][1])
        if spec is not None:
            inverted, cats, white = spec
            kw = {}
            if ascii_mode:
                kw["max_codepoint"] = 127
            if inverted ^ negate:  # the complement of (cats ∪ white)
                kw["exclude_categories"] = cats
                if white:
                    kw["exclude_characters"] = white
            else:
                kw["categories"] = cats
                if white:
                    kw["include_characters"] = white
            return characters(**kw)
    if negate:
        # any 1 char NOT matched by the set — re-test membership via a real charset.
        def _excluded(ch: Any) -> bool:
            for op, av in body:
                if op == c.LITERAL and ord(ch) == av:
                    return True
                if op == c.RANGE and av[0] <= ord(ch) <= av[1]:
                    return True
                if op == c.CATEGORY:
                    return False  # approximate: don't exclude category members
            return False

        return characters().filter(lambda ch: not _excluded(ch))
    branches = []
    for op, av in body:
        if op == c.LITERAL:
            branches.append(sampled_from(_ignorecase_variants(av, is_bytes)) if ignorecase else just(conv(av)))
        elif op == c.RANGE:
            branches.append(characters(min_codepoint=av[0], max_codepoint=av[1]))
        elif op == c.CATEGORY:
            cats = {
                c.CATEGORY_DIGIT: ("Nd",),
                c.CATEGORY_SPACE: ("Zs", "Cc"),
                c.CATEGORY_WORD: ("L", "N"),
            }.get(av)
            branches.append(characters(categories=list(cats)) if cats else characters())
    if not branches:
        return characters()
    return branches[0] if len(branches) == 1 else one_of(*branches)


def _regex_node(op: Any, av: Any, is_bytes: bool, flags: int) -> Any:
    """A strategy producing the (sub)string matched by one parsed regex opcode.
    `flags` carries re.DOTALL (widens ANY to match newline) and re.IGNORECASE
    (a literal generates both case variants)."""
    import re

    c = re._constants
    empty = b"" if is_bytes else ""
    conv = (lambda v: bytes([v])) if is_bytes else chr
    dotall = bool(flags & re.DOTALL)
    if op == c.LITERAL:
        if flags & re.IGNORECASE:
            variants = _ignorecase_variants(av, is_bytes)
            if len(variants) > 1:
                return sampled_from(variants)
        return just(conv(av))
    if op == c.NOT_LITERAL:
        if is_bytes:
            return _byte_choice(set(range(256)) - {av})
        return characters().filter(lambda ch: ord(ch) != av)
    if op == c.ANY:
        if is_bytes:
            return _byte_choice(set(range(256)) if dotall else set(range(256)) - {10})
        return characters() if dotall else characters().filter(lambda ch: ch != "\n")
    if op == c.IN:
        return _regex_charset(av, is_bytes, flags)
    if op in (c.MAX_REPEAT, c.MIN_REPEAT):
        mn, mx, sub = av
        mx = 4 if mx is c.MAXREPEAT else min(mx, 8)
        elem = _regex_seq(list(sub), is_bytes, flags)
        return lists(elem, min_size=mn, max_size=max(mn, mx)).map(lambda xs: empty.join(xs))
    if op == c.SUBPATTERN:
        return _regex_seq(list(av[-1]), is_bytes, flags)
    if op == c.BRANCH:
        return one_of(*[_regex_seq(list(a), is_bytes, flags) for a in av[1]])
    if op == c.ASSERT:
        # Positive lookahead/lookbehind '(?=...)'/'(?<=...)': emit the asserted
        # content so the surrounding pattern can match (port of upstream recurse).
        return _regex_seq(list(av[1]), is_bytes, flags)
    # AT (anchors), ASSERT_NOT (negative lookaround), GROUPREF/etc. -> contribute
    # nothing; matching is resolved by the search/fullmatch filter (best-effort).
    return just(empty)


def _regex_seq(seq: Any, is_bytes: bool, flags: int) -> Any:
    empty = b"" if is_bytes else ""
    parts = [_regex_node(op, av, is_bytes, flags) for (op, av) in seq]
    if not parts:
        return just(empty)
    if len(parts) == 1:
        return parts[0]
    return tuples(*parts).map(lambda xs: empty.join(xs))


def _maybe_pad_impl(draw: Any, regex: Any, strategy: Any, left_pad_strategy: Any, right_pad_strategy: Any) -> Any:
    """Attempt to insert padding around the result of a regex draw (port of
    hypothesis.regex.maybe_pad): only keep a pad if the whole still matches.
    Wrapped with `composite` at call time (composite is defined later in this module)."""
    result = draw(strategy)
    left_pad = draw(left_pad_strategy)
    if left_pad and regex.search(left_pad + result):
        result = left_pad + result
    right_pad = draw(right_pad_strategy)
    if right_pad and regex.search(result + right_pad):
        result += right_pad
    return result


def _regex_has_backref(seq: Any) -> bool:
    """Whether a parsed regex contains a backreference (\\1 / (?P=name) / (?(id)...))."""
    import re

    c = re._constants
    for op, av in seq:
        if op in (c.GROUPREF, c.GROUPREF_EXISTS):
            return True
        if op == c.SUBPATTERN and _regex_has_backref(list(av[-1])):
            return True
        if op in (c.MAX_REPEAT, c.MIN_REPEAT) and _regex_has_backref(list(av[2])):
            return True
        if op == c.BRANCH and any(_regex_has_backref(list(a)) for a in av[1]):
            return True
        if op == c.ASSERT and _regex_has_backref(list(av[1])):
            return True
    return False


def _regex_imperative_impl(draw: Any, parsed: Any, is_bytes: bool, flags: int) -> Any:
    """Imperative regex draw: walks the parsed pattern, recording each capturing group's
    emitted text and re-emitting it at backreferences — so `(['"])[a-z]+\\1` produces a
    matching quote-pair. Used only when the pattern has backrefs; leaves draw the same
    per-node strategies as the static path."""
    import re

    c = re._constants
    empty = b"" if is_bytes else ""
    groups: dict = {}

    def emit_seq(seq: Any) -> Any:
        return empty.join(emit(op, av) for (op, av) in seq)

    def emit(op: Any, av: Any) -> Any:
        if op == c.GROUPREF:
            return groups.get(av, empty)
        if op == c.GROUPREF_EXISTS:
            ref, yes, no = av
            return emit_seq(list(yes if groups.get(ref) else (no or [])))
        if op == c.SUBPATTERN:
            gid = av[0]
            val = emit_seq(list(av[-1]))
            if gid is not None:
                groups[gid] = val
            return val
        if op == c.BRANCH:
            branches = av[1]
            return emit_seq(list(branches[draw(integers(0, len(branches) - 1))]))
        if op in (c.MAX_REPEAT, c.MIN_REPEAT):
            mn, mx, sub = av
            mx = 4 if mx is c.MAXREPEAT else min(mx, 8)
            sub = list(sub)
            return empty.join(emit_seq(sub) for _ in range(draw(integers(mn, max(mn, mx)))))
        if op == c.ASSERT:
            return emit_seq(list(av[1]))  # positive lookaround: emit content
        if op in (c.AT, c.ASSERT_NOT):
            return empty  # anchors / negative lookaround contribute nothing
        return draw(_regex_node(op, av, is_bytes, flags))

    return emit_seq(parsed)


def _regex_base(parsed: Any, is_bytes: bool, flags: int) -> Any:
    """Strategy for the regex body: imperative (group-tracking) when it has backrefs,
    else the static node-tree composition."""
    if _regex_has_backref(parsed):
        return composite(_regex_imperative_impl)(parsed, is_bytes, flags)
    return _regex_seq(parsed, is_bytes, flags)


def _native_from_regex(regex: Any, *, fullmatch: bool = False, alphabet: Any = None) -> Any:
    """Native from_regex: parse the pattern (re._parser) into a strategy composing
    native char/repeat/branch strategies. Non-fullmatch draws are padded with arbitrary
    text on either side unless the pattern is anchored (\\A/^ and \\Z/$), matching
    hypothesis. Falls back to legacy for patterns the native compiler can't structure."""
    import re

    from .errors import InvalidArgument

    if not isinstance(regex, (str, bytes, re.Pattern)):
        raise InvalidArgument(f"regex={regex!r} must be a string, bytes, or compiled pattern")
    if not isinstance(fullmatch, bool):
        raise InvalidArgument(f"fullmatch={fullmatch!r} must be a boolean")
    c = re._constants
    compiled = re.compile(regex) if isinstance(regex, (str, bytes)) else regex
    pattern = compiled.pattern
    is_bytes = isinstance(pattern, bytes)
    if alphabet is not None and is_bytes:
        raise InvalidArgument("alphabet= is not supported for bytestrings")
    src = pattern.decode("latin-1") if is_bytes else pattern
    flags = compiled.flags
    empty_val = b"" if is_bytes else ""
    newline_val = b"\n" if is_bytes else "\n"
    # Reject an unsatisfiable alphabet up front (BEFORE the broad fallback try below, which
    # would otherwise swallow the error): resolve the alphabet to a codepoint-membership
    # predicate (raising InvalidArgument for an invalid alphabet type), then check that the
    # pattern's required characters are coverable within it.
    if alphabet is not None and not is_bytes:
        _in_alpha, _in_range = _regex_alphabet_membership(alphabet)
        try:
            _parsed_chk = list(re._parser.parse(src, flags))
        except re.error:
            _parsed_chk = None
        if _parsed_chk and not _regex_seq_satisfiable(_parsed_chk, _in_alpha, _in_range, flags):
            raise InvalidArgument(
                f"The pattern {regex!r} requires characters that are not in "
                f"alphabet={alphabet!r}."
            )
    try:
        parsed = list(re._parser.parse(src, flags))
        if fullmatch:
            if not parsed:
                return just(empty_val)
            return _regex_base(parsed, is_bytes, flags).filter(
                lambda s: compiled.fullmatch(s) is not None
            )
        if not parsed:
            return binary() if is_bytes else _native_text_alphabet(alphabet)

        base_pad = binary() if is_bytes else _native_text_alphabet(alphabet)
        just_empty = just(empty_val)
        just_newline = just(newline_val)
        right_pad = base_pad
        left_pad = base_pad

        last_op, last_av = parsed[-1]
        if last_op == c.AT:
            if last_av == c.AT_END_STRING:  # \Z
                right_pad = just_empty
            elif last_av == c.AT_END:  # $
                if flags & re.MULTILINE:
                    right_pad = one_of(just_empty, base_pad.map(lambda s: newline_val + s))
                else:
                    right_pad = one_of(just_empty, just_newline)

        first_op, first_av = parsed[0]
        if first_op == c.AT:
            if first_av == c.AT_BEGINNING_STRING:  # \A
                left_pad = just_empty
            elif first_av == c.AT_BEGINNING:  # ^
                if flags & re.MULTILINE:
                    left_pad = one_of(just_empty, base_pad.map(lambda s: s + newline_val))
                else:
                    left_pad = just_empty

        base = _regex_base(parsed, is_bytes, flags).filter(
            lambda s: compiled.search(s) is not None
        )
        return composite(_maybe_pad_impl)(compiled, base, left_pad, right_pad)
    except Exception:  # noqa: BLE001 - construct the native compiler can't structure
        # Self-contained fallback (no legacy proptest frontend): draw arbitrary text/
        # bytes and filter to a match. Slower / may filter heavily for exotic patterns,
        # but keeps from_regex fully native.
        matcher = compiled.fullmatch if fullmatch else compiled.search
        base = binary() if is_bytes else _native_text_alphabet(alphabet)
        return base.filter(lambda s: matcher(s) is not None)


def _regex_alphabet_membership(alphabet: Any) -> Any:
    """(in_alpha, in_range) codepoint-membership predicates for a from_regex alphabet. Raises
    InvalidArgument for an invalid alphabet type (only str / sampled_from() / characters())."""
    if isinstance(alphabet, str):
        cps = {ord(ch) for ch in alphabet}
        return (lambda cp: cp in cps), (lambda lo, hi: any(lo <= x <= hi for x in cps))
    # _regex_alphabet_intervals raises InvalidArgument unless this is characters()/sampled_from().
    ivs = tuple(_e._regex_alphabet_intervals(alphabet).intervals)
    return (
        (lambda cp: any(a <= cp <= b for a, b in ivs)),
        (lambda lo, hi: any(not (hi < a or lo > b) for a, b in ivs)),
    )


def _regex_seq_satisfiable(seq: Any, in_alpha: Any, in_range: Any, flags: int) -> bool:
    """Whether every opcode in a parsed regex sequence can be matched within the alphabet."""
    return all(_regex_node_satisfiable(op, av, in_alpha, in_range, flags) for op, av in seq)


def _regex_node_satisfiable(op: Any, av: Any, in_alpha: Any, in_range: Any, flags: int) -> bool:
    import re

    c = re._constants
    if op == c.LITERAL:
        if flags & re.IGNORECASE:
            return any(in_alpha(ord(v)) for v in _ignorecase_variants(av, False))
        return in_alpha(av)
    if op == c.NOT_LITERAL:
        return in_range(0, av - 1) or in_range(av + 1, 0x10FFFF)
    if op == c.ANY:
        return in_range(0, 0x10FFFF)
    if op == c.IN:
        if av and av[0][0] == c.NEGATE:
            return True  # negated charset: conservatively assume some alphabet char lies outside
        for sub_op, val in av:
            if sub_op == c.LITERAL and in_alpha(val):
                return True
            if sub_op == c.RANGE and in_range(val[0], val[1]):
                return True
            if sub_op not in (c.LITERAL, c.RANGE):
                return True  # CATEGORY etc.: conservative
        return False
    if op in (c.MAX_REPEAT, c.MIN_REPEAT):
        mn, _mx, sub = av
        return mn == 0 or _regex_seq_satisfiable(list(sub), in_alpha, in_range, flags)
    if op == c.SUBPATTERN:
        return _regex_seq_satisfiable(list(av[-1]), in_alpha, in_range, flags)
    if op == c.BRANCH:
        return any(_regex_seq_satisfiable(list(a), in_alpha, in_range, flags) for a in av[1])
    if op == c.ASSERT:
        return _regex_seq_satisfiable(list(av[1]), in_alpha, in_range, flags)
    return True  # AT (anchors) / ASSERT_NOT / GROUPREF / etc. contribute no required character


def _native_text_alphabet(alphabet: Any) -> Any:
    """text() honouring a from_regex `alphabet=` argument (str/IntervalSet/characters
    strategy / list-of-chars), else the default text()."""
    if alphabet is None:
        return text()
    return text(alphabet=alphabet)


@functools.lru_cache(maxsize=1)
def _true_random_cls() -> Any:
    """A real hypothesis TrueRandom subclass whose method-call logging routes to our
    native note buffer (the base would use real hypothesis's report(), which isn't
    wired into the native engine)."""
    from hypothesis.strategies._internal.random import TrueRandom, convert_kwargs

    class _NativeTrueRandom(TrueRandom):  # type: ignore[misc]
        def _hypothesis_log_random(self, method: Any, kwargs: Any, result: Any) -> None:
            if not self._note_method_calls:
                return
            args, kw = convert_kwargs(method, kwargs)
            argstr = ", ".join(list(map(repr, args)) + [f"{k}={v!r}" for k, v in kw.items()])
            _record_random_call(f"{self!r}.{method}({argstr}) -> {result!r}")

    return _NativeTrueRandom


def _build_random(data: Any, use_true_random: bool, note_method_calls: bool) -> Any:
    """Construct the object `randoms()` draws: a seeded random.Random (TrueRandom; seed
    drawn from the data) when use_true_random, else the native ArtificialRandom (draws
    shrink/replay, deterministic per seed/state). Both route note-logging to our buffer."""
    if use_true_random:
        seed = data.draw(integers(0, (1 << 64) - 1))
        return _true_random_cls()(seed=seed, note_method_calls=note_method_calls)
    return _artificial_random_cls()(note_method_calls=note_method_calls, data=data)


def _require_real_magnitude(value: Any, name: str) -> None:
    """Raise InvalidArgument if `value` isn't a real number (e.g. a complex magnitude). Kept
    separate from complex_numbers so the narrowing to numbers.Real stays on this param."""
    from numbers import Real

    from hypothesis_fast.errors import InvalidArgument

    if not isinstance(value, Real):
        raise InvalidArgument(f"{name}={value!r} must be a real number.")


def complex_numbers(
    *,
    min_magnitude: Any = 0,
    max_magnitude: Any = None,
    allow_infinity: Any = None,
    allow_nan: Any = None,
    allow_subnormal: bool = True,
    width: int = 128,
) -> Any:
    """Port of hypothesis.complex_numbers — composes native floats/builds/composite
    with magnitude constraints (via upstream's pure cathetus/float_of helpers)."""
    import math

    from hypothesis.internal.cathetus import cathetus
    from hypothesis.internal.floats import float_of
    from hypothesis_fast.errors import InvalidArgument

    # Catch non-real magnitudes before the `< 0` comparisons below would raise a bare TypeError
    # (None < 0, or (1+2j) < 0). Checked via a helper so the locals stay `Any` — narrowing them
    # to numbers.Real here makes pyright reject the float ops / cathetus() calls downstream.
    if min_magnitude is None:
        raise InvalidArgument("Use min_magnitude=0 or omit the argument entirely.")
    _require_real_magnitude(min_magnitude, "min_magnitude")
    if max_magnitude is not None:
        _require_real_magnitude(max_magnitude, "max_magnitude")
    if min_magnitude < 0:
        raise InvalidArgument(f"min_magnitude={min_magnitude!r} must be non-negative")
    if max_magnitude is not None and max_magnitude < 0:
        raise InvalidArgument(f"max_magnitude={max_magnitude!r} must be non-negative")
    if max_magnitude is not None and min_magnitude > max_magnitude:
        raise InvalidArgument(f"min_magnitude={min_magnitude!r} > max_magnitude={max_magnitude!r}")
    if max_magnitude == math.inf:
        max_magnitude = None

    if allow_infinity is None:
        allow_infinity = max_magnitude is None
    elif allow_infinity and max_magnitude is not None:
        raise InvalidArgument(f"Cannot have allow_infinity=True with max_magnitude={max_magnitude!r}")
    if allow_nan is None:
        allow_nan = bool(min_magnitude == 0 and max_magnitude is None)
    elif allow_nan and not (min_magnitude == 0 and max_magnitude is None):
        raise InvalidArgument(
            f"Cannot have allow_nan=True, min_magnitude={min_magnitude!r}, max_magnitude={max_magnitude!r}"
        )
    if not isinstance(allow_subnormal, bool):
        raise InvalidArgument(f"allow_subnormal={allow_subnormal!r} must be a bool")
    if width not in (32, 64, 128):
        raise InvalidArgument(f"width={width!r}, but must be 32, 64 or 128")

    component_width = width // 2
    allow_kw = {
        "allow_nan": allow_nan,
        "allow_infinity": allow_infinity,
        "allow_subnormal": None if allow_subnormal else allow_subnormal,
        "width": component_width,
    }

    if min_magnitude == 0 and max_magnitude is None:
        return builds(complex, floats(**allow_kw), floats(**allow_kw))

    @composite
    def constrained_complex(draw: Any) -> complex:
        if max_magnitude is None:
            zi = draw(floats(**allow_kw))
            rmax = None
        else:
            zi = draw(
                floats(
                    -float_of(max_magnitude, component_width),
                    float_of(max_magnitude, component_width),
                    **allow_kw,
                )
            )
            rmax = float_of(cathetus(max_magnitude, zi), component_width)
        if min_magnitude == 0 or math.fabs(zi) >= min_magnitude:
            zr = draw(floats(None if rmax is None else -rmax, rmax, **allow_kw))
        else:
            rmin = float_of(cathetus(min_magnitude, zi), component_width)
            zr = draw(floats(rmin, rmax, **allow_kw))
        if min_magnitude > 0 and draw(booleans()) and math.fabs(zi) <= min_magnitude:
            zr = -zr
        return complex(zr, zi)

    return constrained_complex()


def _as_finite_decimal(value: Any, name: str, allow_infinity: Any, places: Any = None) -> Any:
    import math
    from decimal import Context, Decimal, InvalidOperation, localcontext

    from hypothesis_fast.errors import InvalidArgument

    if value is None:
        return None
    from fractions import Fraction

    old = value
    if isinstance(value, Fraction):
        # Decimal() can't convert a Fraction directly. A Fraction is *exactly* a terminating
        # decimal iff its lowest-terms denominator factors into only 2s and 5s (3/20, 1/8 are
        # fine; 1/3, 2/3 are not and must raise InvalidArgument).
        den = value.denominator
        while den % 2 == 0:
            den //= 2
        while den % 5 == 0:
            den //= 5
        if den != 1:
            raise InvalidArgument(
                f"{name}={value!r} cannot be exactly represented as a Decimal value"
            )
        value = Decimal(value.numerator) / Decimal(value.denominator)
    if not isinstance(value, Decimal):
        # Convert in a fresh context so the error message is identical regardless
        # of the caller's ambient decimal context/traps (test_consistent_decimal_error).
        try:
            with localcontext(Context()):
                value = Decimal(value)
        except (ValueError, InvalidOperation, TypeError):
            raise InvalidArgument(
                f"{name}={value!r} is not a valid Decimal value"
            ) from None
    # A NaN bound is invalid (and must be rejected before the inexact check below,
    # which calls math.isfinite — that raises InvalidOperation on a signalling NaN).
    if value.is_nan():
        raise InvalidArgument(f"Invalid {name}={value!r}")
    # Passing a bound that can't be exactly represented as a decimal (e.g. an
    # inexact float like 1e-100) is deprecated — the drawn values won't honour the
    # bound the user thinks they wrote. Mirrors real hypothesis core._as_finite_decimal.
    finitude_old = value if isinstance(old, str) else old
    if math.isfinite(finitude_old) != math.isfinite(value) or (
        value.is_finite() and Fraction(str(old)) != Fraction(str(value))
    ):
        import warnings

        from hypothesis_fast.errors import HypothesisDeprecationWarning

        warnings.warn(
            f"{old!r} cannot be exactly represented as a decimal with {places=}",
            HypothesisDeprecationWarning,
            stacklevel=2,
        )
    if value.is_finite():
        return value
    if value.is_infinite() and (value < 0 if "min" in name else value > 0):
        if allow_infinity or allow_infinity is None:
            return None
        raise InvalidArgument(f"allow_infinity=False, but {name}={value!r}")
    raise InvalidArgument(f"Invalid {name}={value!r}")


def decimals(
    min_value: Any = None,
    max_value: Any = None,
    *,
    allow_nan: Any = None,
    allow_infinity: Any = None,
    places: Any = None,
) -> Any:
    """Port of hypothesis.decimals — composes native integers/fractions +
    sampled_from(special) for inf/NaN. The per-draw Decimal build is the
    irreducible Python boundary."""
    import math
    from decimal import Context, Decimal

    from hypothesis_fast.errors import InvalidArgument

    if places is not None and (not isinstance(places, int) or places < 0):
        raise InvalidArgument(f"places={places!r} may not be negative or non-integer")
    min_value = _as_finite_decimal(min_value, "min_value", allow_infinity, places)
    max_value = _as_finite_decimal(max_value, "max_value", allow_infinity, places)
    if min_value is not None and max_value is not None and min_value > max_value:
        raise InvalidArgument(f"min_value={min_value!r} > max_value={max_value!r}")
    if allow_infinity and (None not in (min_value, max_value)):
        raise InvalidArgument("Cannot allow infinity between finite bounds")

    if places is not None:
        factor = Decimal(10) ** -places

        def ctx(val: Any) -> Context:
            precision = math.ceil(math.log10(abs(val) or 1)) + places + 1
            return Context(prec=max(precision, 1))

        def int_to_decimal(val: int) -> Decimal:
            c = ctx(val)
            return c.quantize(c.multiply(val, factor), factor)

        min_num = None if min_value is None else math.ceil(ctx(min_value).divide(min_value, factor))
        max_num = None if max_value is None else math.floor(ctx(max_value).divide(max_value, factor))
        if min_num is not None and max_num is not None and min_num > max_num:
            raise InvalidArgument(
                f"There are no decimals with {places} places between "
                f"min_value={min_value!r} and max_value={max_value!r}"
            )
        strat = integers(min_num, max_num).map(int_to_decimal)
    else:
        def fraction_to_decimal(val: Any) -> Decimal:
            precision = (
                math.ceil(math.log10(abs(val.numerator) or 1) + math.log10(val.denominator)) + 1
            )
            return Context(prec=precision or 1).divide(Decimal(val.numerator), val.denominator)

        strat = fractions(min_value, max_value).map(fraction_to_decimal)

    special = []
    if allow_infinity or (allow_infinity is None and max_value is None):
        special.append(Decimal("Infinity"))
    if allow_infinity or (allow_infinity is None and min_value is None):
        special.append(Decimal("-Infinity"))
    if allow_nan or (allow_nan is None and (None in (min_value, max_value))):
        special.extend(map(Decimal, ("NaN", "-NaN", "sNaN", "-sNaN")))
    return strat | sampled_from(special) if special else strat


_RETURN_TYPE_STRATEGIES: dict[Any, Callable[[], Any]] = {
    type(None): none,
    bool: booleans,
    int: integers,
    float: floats,
    str: text,
    bytes: binary,
}


def _infer_return_strategy(return_type: Any) -> Any:
    factory = _RETURN_TYPE_STRATEGIES.get(return_type)
    if factory is not None:
        return factory()
    return none()


def _functions_default_like() -> None:
    return None


def functions(
    *, like: Callable[..., Any] = _functions_default_like, returns: Any = ..., pure: bool = False
) -> Any:
    """Port of hypothesis.functions — draws a callable mimicking `like`'s signature
    whose return value is drawn from `returns` against the live data. Native node;
    the per-call drawing/purity/InvalidState logic lives in `_build_function`."""
    import typing

    from hypothesis_fast.errors import InvalidArgument

    # Track which kwargs were explicitly passed so the repr mirrors hypothesis's LazyStrategy
    # (shows only the given args) — `functions(returns=booleans())`, not all three.
    like_explicit = like is not _functions_default_like
    returns_explicit = returns is not ... and returns is not None
    if not isinstance(pure, bool):
        raise InvalidArgument(f"pure={pure!r} must be a bool")
    if not callable(like):
        raise InvalidArgument(
            "The first argument to functions() must be a callable to imitate, "
            f"but got non-callable like={like!r}"
        )
    if returns is None or returns is ...:
        try:
            hints = typing.get_type_hints(like)
        except Exception:  # noqa: BLE001 - unresolved annotations -> fall back to none
            hints = {}
        returns = _infer_return_strategy(hints.get("return", type(None)))
    if not isinstance(returns, SearchStrategy):
        raise InvalidArgument(
            f"returns={returns!r} must be a SearchStrategy but is a {type(returns).__name__}"
        )
    strat = _e._functions_strategy(like, returns, pure)
    parts = []
    if like_explicit:
        from hypothesis.internal.reflection import get_pretty_function_description

        parts.append(f"like={get_pretty_function_description(like)}")
    if returns_explicit:
        parts.append(f"returns={returns!r}")
    if pure:
        parts.append("pure=True")
    strat._hf_repr_override = f"functions({', '.join(parts)})"
    return strat


def _build_function(like: Callable[..., Any], returns: Any, pure: bool, data: Any) -> Any:
    """Construct the callable that functions() draws: it mimics `like`'s signature
    (so a mismatched call raises TypeError), draws its result from `returns` against
    the live ConjectureData, and raises InvalidState once that data is frozen."""
    import functools
    import inspect

    from hypothesis_fast.errors import InvalidState

    import sys

    sig = inspect.signature(like)
    cache: dict[Any, Any] = {}

    def _report_call(args: Any, kwargs: Any, result: Any) -> None:
        # In verbose/debug mode, report each (uncached) call — pure functions report only
        # on the first call with given args (test_functions_note_*_to_*_functions).
        from .settings import Verbosity
        from .settings import settings as _settings_cls

        active = _settings_cls.default
        if active is None or active.verbosity < Verbosity.verbose:
            return
        rep = sys.modules.get("hypothesis.reporting")
        if rep is None:
            return
        name = getattr(like, "__name__", "f")
        arglist = ", ".join(
            [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        )
        rep.current_reporter()(f"Called {name}({arglist}) -> {result!r}")

    @functools.wraps(like)
    def inner(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind(*args, **kwargs)  # raises TypeError on a signature mismatch
        if data.is_frozen():
            raise InvalidState(
                f"This generated {getattr(like, '__name__', like)!r} function can only "
                "be called within the scope of the @given that created it."
            )
        if pure:
            bound.apply_defaults()
            try:
                key: Any = frozenset(bound.arguments.items())
            except TypeError:
                # An unhashable argument value (e.g. a dict/list — returns passes these): key
                # by a stable repr so the pure cache still returns the same value for
                # equal-repr args within a run, instead of crashing on `unhashable type`.
                key = tuple(sorted((k, repr(v)) for k, v in bound.arguments.items()))
            if key not in cache:
                cache[key] = data.draw(returns)
                _report_call(args, kwargs, cache[key])
            return cache[key]
        result = data.draw(returns)
        _report_call(args, kwargs, result)
        return result

    return inner


def composite(f: Any) -> Any:
    """`@composite` — wraps a `def f(draw, *args)` into a strategy factory. Validates
    that `f` takes a positional `draw` parameter (no default) and warns if it never
    calls it, matching hypothesis. `f` may be a classmethod/staticmethod object, in which
    case the produced factory is re-wrapped in the same descriptor (so the return type
    spans factory-callable and descriptor)."""
    import dis
    import inspect

    from hypothesis_fast.errors import HypothesisDeprecationWarning, InvalidArgument

    # `@composite` may wrap a classmethod/staticmethod object (when it's the OUTER
    # decorator, e.g. `@composite @classmethod def f(draw, cls)`). Unwrap to the plain
    # function for inspection, and re-apply the descriptor to the produced factory so the
    # class/instance access still binds correctly (test_applying_composite_decorator_to_methods).
    special_method = None
    if isinstance(f, (classmethod, staticmethod)):
        special_method = type(f)
        f = f.__func__

    import typing as _typing

    params = list(inspect.signature(f).parameters.values())
    # The draw parameter may be any positional kind, INCLUDING *args (pure-positional
    # composites use args[0] as draw) — match upstream's `"POSITIONAL" in kind.name`.
    if not params or "POSITIONAL" not in params[0].kind.name:
        raise InvalidArgument(
            "Functions wrapped with @composite must take at least one positional "
            f"argument (the draw function), but {f.__name__} does not."
        )
    _is_var_draw = params[0].kind == inspect.Parameter.VAR_POSITIONAL
    if not _is_var_draw and params[0].default is not inspect.Parameter.empty:
        raise InvalidArgument(
            f"@composite draw parameter {params[0].name!r} cannot have a default value."
        )
    # Warn if the function never calls its draw argument — but skip a *args draw (the name
    # isn't a fixed local we can scan for) and the typing.overload placeholder (no body).
    try:
        _overload_dummy = _typing._overload_dummy
    except AttributeError:
        _overload_dummy = None
    if not _is_var_draw and f is not _overload_dummy:
        draw_name = params[0].name
        loaded = {i.argval for i in dis.get_instructions(f)}
        if draw_name not in loaded:
            import warnings

            warnings.warn(
                f"@composite function {f.__name__} never calls its draw argument "
                f"{draw_name!r}; it should draw at least one value.",
                HypothesisDeprecationWarning,
                stacklevel=2,
            )

    # The factory drops the leading `draw` parameter (but NOT a *args draw, which stays so
    # the caller can pass the pure positional values) and returns a SearchStrategy of the
    # wrapped function's return type (matching hypothesis's @composite signature editing).
    _sig = inspect.signature(f)
    _ret = _sig.return_annotation
    # A @composite function returns a VALUE, not a strategy; a `-> SearchStrategy[...]`
    # return annotation is almost certainly a mistake (test_warns_on_strategy_annotation).
    if _typing.get_origin(_ret) is SearchStrategy:
        import warnings

        from hypothesis_fast.errors import HypothesisWarning

        warnings.warn(
            f"Return-type annotation is `{_ret!r}`, but the decorated function "
            "should return a value (not a strategy)",
            HypothesisWarning,
            stacklevel=2,
        )
    _factory_params = list(_sig.parameters.values())
    if not _is_var_draw:
        _factory_params = _factory_params[1:]
    _factory_sig = _sig.replace(
        parameters=_factory_params,
        return_annotation=(
            _e.SearchStrategy[_ret] if _ret is not inspect.Signature.empty else _e.SearchStrategy
        ),
    )

    @functools.wraps(f)
    def inner(*args: Any, **kwargs: Any) -> Any:
        # Enforce the factory signature so e.g. passing a positional-only argument by
        # keyword raises TypeError, exactly as calling the original function would.
        _factory_sig.bind(*args, **kwargs)
        return _e._composite_strategy(f, args, kwargs)

    inner.__signature__ = _factory_sig  # type: ignore[attr-defined]
    if special_method is not None:
        return special_method(inner)
    return inner


def _populate_native_registry() -> None:
    """Pre-register native strategies for the stdlib types hypothesis keeps in its
    `_global_type_lookup`, so native `from_type` resolves them without a legacy
    fallback (which the native engine cannot draw). Mirrors hypothesis's built-in
    registrations. Uses setdefault so explicit user registrations always win."""
    import collections.abc as _abc
    import datetime as _dt
    import ipaddress as _ip
    import numbers as _num
    import os as _osmod
    import pathlib as _pl
    import random as _random
    import re as _re
    import types as _ty

    reg = _NATIVE_TYPE_REGISTRY.setdefault

    # random.Random -> randoms() (mirrors hypothesis's built-in registration; gives the
    # 'randoms()' repr rather than builds(Random) from __init__ introspection).
    reg(_random.Random, lambda t: randoms())

    # numbers ABCs
    reg(_num.Integral, lambda t: _e.integers())
    reg(_num.Rational, lambda t: _e.fractions())
    reg(_num.Real, lambda t: _e.floats())
    reg(_num.Complex, lambda t: complex_numbers())
    reg(_num.Number, lambda t: _e.one_of(_e.integers(), _e.floats(), complex_numbers()))

    # ip addresses / networks / interfaces
    reg(_ip.IPv4Address, lambda t: _e.ip_addresses(v=4))
    reg(_ip.IPv6Address, lambda t: _e.ip_addresses(v=6))
    reg(_ip.IPv4Network, lambda t: _e.ip_addresses(v=4).map(lambda a: _ip.IPv4Network(int(a))))
    reg(_ip.IPv6Network, lambda t: _e.ip_addresses(v=6).map(lambda a: _ip.IPv6Network(int(a))))
    reg(_ip.IPv4Interface, lambda t: _e.ip_addresses(v=4).map(lambda a: _ip.IPv4Interface(int(a))))
    reg(_ip.IPv6Interface, lambda t: _e.ip_addresses(v=6).map(lambda a: _ip.IPv6Interface(int(a))))

    # datetime.timezone (offset must lie in the open (-24h, 24h) interval)
    _tzmax = _dt.timedelta(hours=23, minutes=59)
    reg(_dt.timezone, lambda t: _e.one_of(
        _e.just(_dt.timezone.utc),
        _e.builds(_dt.timezone, _e.timedeltas(min_value=-_tzmax, max_value=_tzmax)),
    ))

    # Unicode errors carry specific constructor signatures
    _idx = _e.integers(min_value=0, max_value=8)
    reg(UnicodeDecodeError, lambda t: _e.builds(UnicodeDecodeError, _e.just("utf-8"), _e.binary(), _idx, _idx, _e.just("reason")))
    reg(UnicodeEncodeError, lambda t: _e.builds(UnicodeEncodeError, _e.just("utf-8"), _e.text(), _idx, _idx, _e.just("reason")))
    reg(UnicodeTranslateError, lambda t: _e.builds(UnicodeTranslateError, _e.text(), _idx, _idx, _e.just("reason")))

    # re.Pattern / re.Match — escape arbitrary text so compile/match always succeed
    # re.Pattern[str|bytes] / re.Match[str|bytes] — respect the type arg (default str).
    def _re_is_bytes(thing: Any) -> bool:
        import typing as _ty

        args = _ty.get_args(thing)
        return bool(args) and args[0] is bytes

    def _pattern_strat(thing: Any) -> Any:
        if _re_is_bytes(thing):
            return _e.binary().map(lambda s: _re.compile(_re.escape(s)))
        return _e.text().map(lambda s: _re.compile(_re.escape(s)))

    def _match_strat(thing: Any) -> Any:
        if _re_is_bytes(thing):
            return _e.binary().map(lambda s: _re.match(_re.escape(s) or b"", s))
        return _e.text().map(lambda s: _re.match(_re.escape(s), s))

    reg(type(_re.compile("")), _pattern_strat)
    reg(type(_re.match("", "")), _match_strat)

    # callables
    reg(_ty.FunctionType, lambda t: functions())

    # collections.abc views / iterators / generators / mutable set (bare, elements unchecked)
    # Hashable -> a union of always-hashable scalar types (so set[Hashable] /
    # dict[Hashable, V] draw hashable elements). typing.Hashable (origin abc.Hashable)
    # resolves to this via the generic-origin registry lookup in Rust from_type.
    # decimals() is included so from_type(Hashable) can yield Decimal('snan'),
    # which is a Hashable subclass instance that is NOT hashable (issue #2320);
    # set[Hashable]/dict[Hashable, V] filter those out via can_hash at resolution.
    reg(_abc.Hashable, lambda t: _e.one_of(
        _e.none(), _e.booleans(), _e.integers(), _e.floats(), complex_numbers(),
        _e.text(), _e.binary(), decimals(),
    ))
    # Buffer (PEP 688, 3.12+): bytes/bytearray are buffers on every version; memoryview
    # becomes a registered Buffer subtype on 3.14, so from_type(Buffer) can yield all three.
    _buffer = getattr(_abc, "Buffer", None)
    if _buffer is not None:
        reg(_buffer, lambda t: _e.one_of(
            _e.binary(), _e.binary().map(bytearray), _e.binary().map(memoryview),
        ))
    # ByteString (deprecated 3.12, removed 3.14): bytes/bytearray are ByteString instances.
    # Merely reading the attribute warns on 3.12+, so suppress that during the lookup.
    import warnings as _warn
    with _warn.catch_warnings():
        _warn.simplefilter("ignore", DeprecationWarning)
        _bytestring = getattr(_abc, "ByteString", None)
    if _bytestring is not None:
        reg(_bytestring, lambda t: _e.one_of(_e.binary(), _e.binary().map(bytearray)))
    # Sized: anything with __len__ — a list works (isinstance(list, typing.Sized) is True).
    reg(_abc.Sized, lambda t: _e.lists(_e.integers()))
    reg(_abc.Iterator, lambda t: _e.lists(_e.integers()).map(iter))
    reg(_abc.Iterable, lambda t: _e.lists(_e.integers()))
    reg(_abc.MutableSet, lambda t: _e.sets(_e.integers()))
    reg(_abc.Generator, lambda t: _e.lists(_e.integers()).map(lambda xs: (x for x in xs)))
    reg(_abc.ItemsView, lambda t: _e.dictionaries(_e.integers(), _e.integers()).map(lambda d: d.items()))
    reg(_abc.KeysView, lambda t: _e.dictionaries(_e.integers(), _e.integers()).map(lambda d: d.keys()))
    reg(_abc.ValuesView, lambda t: _e.dictionaries(_e.integers(), _e.integers()).map(lambda d: d.values()))

    # builtins constructible via builds
    reg(enumerate, lambda t: _e.builds(enumerate, _e.lists(_e.integers())))
    # reversed(list) is a list_reverseiterator; reverse a tuple to get a `reversed` object
    reg(reversed, lambda t: _e.lists(_e.integers()).map(lambda xs: reversed(tuple(xs))))
    reg(filter, lambda t: _e.builds(filter, _e.just(bool), _e.lists(_e.integers())))
    reg(map, lambda t: _e.builds(map, _e.just(str), _e.lists(_e.integers())))
    # type -> a sample of concrete types (so from_type(type) / typing.Type produce a class)
    reg(type, lambda t: _e.sampled_from(
        [bool, int, float, complex, str, bytes, bytearray, list, dict, set, frozenset, tuple]
    ))
    reg(classmethod, lambda t: _e.builds(classmethod, functions()))
    reg(staticmethod, lambda t: _e.builds(staticmethod, functions()))
    # super -> a valid bound super object (super(int, 0); 0 is an int instance)
    reg(super, lambda t: _e.just(super(int, 0)))
    # bare Callable -> a generated function (parametrized Callable[..] handled in from_type)
    reg(_abc.Callable, lambda t: functions())
    # bare defaultdict -> a dict wrapped as a defaultdict
    import collections as _collections
    reg(_collections.defaultdict, lambda t: _e.dictionaries(_e.integers(), _e.integers()).map(
        lambda d: _collections.defaultdict(None, d)
    ))

    # os.PathLike
    reg(_osmod.PathLike, lambda t: _e.text().map(_pl.PurePosixPath))

    # typing.Supports* protocols -> unions of types implementing / castable to them
    # (mirrors hypothesis's _global_type_lookup registrations).
    import typing as _typing

    reg(_typing.SupportsAbs, lambda t: _e.one_of(
        _e.booleans(), _e.integers(), _e.floats(), complex_numbers(),
        _e.fractions(), decimals(), _e.timedeltas(),
    ))
    reg(_typing.SupportsRound, lambda t: _e.one_of(
        _e.booleans(), _e.integers(), _e.floats(), decimals(), _e.fractions(),
    ))
    reg(_typing.SupportsComplex, lambda t: _e.one_of(
        _e.booleans(), _e.integers(), _e.floats(), complex_numbers(), decimals(), _e.fractions(),
    ))
    reg(_typing.SupportsFloat, lambda t: _e.one_of(
        _e.booleans(), _e.integers(), _e.floats(), decimals(), _e.fractions(),
        _e.floats().map(str),
    ))
    reg(_typing.SupportsInt, lambda t: _e.one_of(
        _e.booleans(), _e.integers(), _e.floats(), _e.uuids(), decimals(),
        _native_from_regex(r"\A-?\d+\Z"),
    ))
    reg(_typing.SupportsIndex, lambda t: _e.one_of(_e.integers(), _e.booleans()))
    reg(_typing.SupportsBytes, lambda t: _e.one_of(
        _e.booleans(), _e.binary(), _e.integers(min_value=0, max_value=255),
        _e.lists(_e.integers(min_value=0, max_value=255)).map(tuple),
    ))

    # zoneinfo (optional; available on 3.9+). available_timezones() walks the tz DB on
    # disk, so defer it into the factory (lazy — only paid if from_type(ZoneInfo) is
    # actually used) instead of reading it at every import.
    try:
        import zoneinfo as _zi

        reg(
            _zi.ZoneInfo,
            lambda t: _e.sampled_from(sorted(_zi.available_timezones())).map(_zi.ZoneInfo),
        )
    except Exception:  # noqa: BLE001 - zoneinfo/tzdata unavailable -> leave to legacy
        pass


# Native frontend public exports (the reflective surface: from_type / randoms /
# from_regex / register_type_strategy / check_strategy), wired now that their helpers
# are all defined above. (External plugins that call `st.from_type.__clear_cache()` — real
# hypothesis's lru_cached from_type exposes it — get that hook installed by the HF_SHIM, which
# wraps from_type; the bare native builtin can't carry an attribute.)
from_type = _e.from_type
randoms = _e.randoms
from_regex = _native_from_regex


def register_type_strategy(thing: Any, strategy: Any) -> Any:
    return _register_type_strategy(thing, strategy)


def check_strategy(arg: Any, name: str = "") -> None:
    _check_strategy(arg, name)


def _warn_then_identity(msg: str) -> Callable[[Any], Any]:
    """A map-pack that emits `msg` as a HypothesisWarning the first time it draws, then
    returns the value unchanged. Used by the regex finditer/split filter rewrite, whose
    warning must fire lazily at draw time (matching real hypothesis's LazyStrategy)."""
    import warnings

    from hypothesis_fast.errors import HypothesisWarning

    def f(v: Any) -> Any:
        warnings.warn(msg, HypothesisWarning, stacklevel=2)
        return v

    return f


_populate_native_registry()


_NATIVE_NAMES = frozenset(name for name in dir() if not name.startswith("_"))


def __getattr__(name: str) -> Any:
    # Drop-in completeness: names we haven't made native yet fall back to the
    # legacy strategies module (which may itself fall back to real hypothesis).
    # Resolve the legacy module via sys.modules, NOT `from hypothesis_fast import
    # strategies` — when the package attribute `hypothesis_fast.strategies` is
    # rebound to THIS module (native-default mode), the attribute path resolves back
    # here and recurses forever. The sys.modules entry stays the real legacy submodule.
    import importlib
    import sys

    _legacy = sys.modules.get("hypothesis_fast.strategies")
    if _legacy is None or _legacy is sys.modules.get(__name__):
        _legacy = importlib.import_module("hypothesis_fast.strategies")
    return getattr(_legacy, name)


__all__ = list(_NATIVE_NAMES)  # type: ignore[reportUnsupportedDunderAll]  # dynamic export list

