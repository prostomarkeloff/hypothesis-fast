"""hypothesis-compatible strategy constructors.

Each public function returns a `SearchStrategy`. A strategy is either:

* **engine-native** — it carries a *spec* (a tagged tuple the Rust engine turns
  into a proptest strategy), and `@given` runs/shrinks it in Rust; or
* **fallback** — something the engine can't do (an unsupported strategy or
  option, or a foreign real-`hypothesis` strategy). It carries a thunk that
  builds the equivalent real-`hypothesis` strategy, and `@given` delegates the
  whole test to real `hypothesis`.

Every native strategy can *also* produce its real-`hypothesis` equivalent (so a
fully-supported strategy still converts if a sibling in the same `@given` forces
fallback). Spec-expressible strategies reconstruct generically via
`_spec_to_hyp`; opaque ones (`@composite`, `flatmap`, `data`) carry an explicit
thunk.
"""

from __future__ import annotations

import datetime as _dt
import inspect
import warnings
from collections.abc import Callable, Hashable, Iterable, Sequence
from math import copysign as _copysign
from typing import Any, TypeVar

from .errors import InvalidArgument

T = TypeVar("T")

# i64 bounds (the engine generates integers as i64). Symmetric to avoid abs overflow.
_INT_MIN = -(2**63 - 1)
_INT_MAX = 2**63 - 1
_FLOAT_MIN = -1e308
_FLOAT_MAX = 1e308
_MAX_CODEPOINT = 0x10FFFF
_DEFAULT_MAX_SIZE = 10
# sampled_from above this many elements is delegated to real hypothesis so we
# never materialise an enormous range into the Rust engine's PyAny list.
_SAMPLED_FROM_NATIVE_CAP = 10_000

_ABSENT = object()  # marks an omitted optional fixed_dictionaries key


def _rewrite_numeric_filter(strat: SearchStrategy, predicate: Callable[[Any], Any]) -> SearchStrategy | None:
    """Native filter-rewriting for integers/floats. Returns a tightened strategy
    if the predicate is a simple comparison we can decode, else None (caller
    falls through to a runtime filter).

    Patterns recognised (matching hypothesis's filtering analyzer):

    * `functools.partial(operator.OP, N)` — compares as `OP(N, x)`. We invert to
      `(N < x)` ⇒ `min_value=N, exclude_min` etc.
    * `lambda x: x OP N` and `lambda x: N OP x` — same idea, parsed via the AST.
    * `math.isfinite` — drops ±inf bounds.
    """
    import functools
    import math
    import operator

    assert strat._spec is not None  # only called for native int/float specs
    tag = strat._spec[0]
    cur_min: Any = strat._spec[1]
    cur_max: Any = strat._spec[2]

    new_min: Any = None
    new_max: Any = None
    exclude_min = False
    exclude_max = False

    # functools.partial(op, N) — op(N, x)
    if isinstance(predicate, functools.partial) and not predicate.keywords and len(predicate.args) == 1:
        op = predicate.func
        n = predicate.args[0]
        if isinstance(n, (int, float)):
            # NaN bound: any comparison with NaN is False → impossible filter.
            if isinstance(n, float) and math.isnan(n):
                return nothing()
            if op is operator.lt:
                new_min, exclude_min = n, True
            elif op is operator.le:
                new_min = n
            elif op is operator.eq:
                new_min, new_max = n, n
            elif op is operator.ge:
                new_max = n
            elif op is operator.gt:
                new_max, exclude_max = n, True

    if predicate is math.isfinite and tag == "float":
        # floats already exclude inf when bounded; nothing to rewrite for tightening.
        return strat

    if new_min is None and new_max is None:
        return None

    # Fold rewriting into native int spec.
    if tag == "int":
        lo = cur_min
        hi = cur_max
        if new_min is not None:
            # ±inf / out-of-i64 bounds map to "all i64 above/below" — i.e. no
            # tightening on the low side, OR collapse to nothing if the bound
            # excludes EVERY current value.
            if isinstance(new_min, float):
                if new_min == math.inf:
                    return nothing()
                if new_min == -math.inf:
                    pass  # no lower-tightening
                else:
                    n = math.ceil(new_min) + (1 if exclude_min and new_min == int(new_min) else 0)
                    lo = max(lo, n)
            else:
                n = int(new_min) + (1 if exclude_min else 0)
                lo = max(lo, n)
        if new_max is not None:
            if isinstance(new_max, float):
                if new_max == -math.inf:
                    return nothing()
                if new_max == math.inf:
                    pass  # no upper-tightening
                else:
                    n = math.floor(new_max) - (1 if exclude_max and new_max == int(new_max) else 0)
                    hi = min(hi, n)
            else:
                n = int(new_max) - (1 if exclude_max else 0)
                hi = min(hi, n)
        if lo > hi:
            return nothing()
        return SearchStrategy(("int", lo, hi))

    # Tag is "float" — keep a runtime filter for now; tightening floats needs
    # next_up/next_down handling for exclude semantics.
    return None


def _is_identity_function(f: Callable[..., Any]) -> bool:
    """True if `f` is `lambda x: x` (or a def equivalent) — single-arg passthrough.

    Compares the bytecode of `f.__code__` against a freshly-built `lambda x: x`:
    same shape across CPython versions (just `LOAD_FAST 0; RETURN_VALUE`-style ops),
    and immune to closure-cell renames since we compare the immutable `.co_code`.
    Falls back to False for anything callable that isn't a simple Python function.
    """
    code = getattr(f, "__code__", None)
    if code is None or code.co_argcount != 1 or code.co_kwonlyargcount:
        return False
    return code.co_code == _IDENTITY_LAMBDA.__code__.co_code


_IDENTITY_LAMBDA = lambda x: x  # noqa: E731 — used by `_is_identity_function`

Spec = tuple[Any, ...]  # an engine tagged tuple, e.g. ("int", 0, 10)
HypFactory = Callable[[Any], Any]  # (real hypothesis.strategies module) -> strategy


# The real hypothesis modules, used for fallback. Normally resolved by importing
# `hypothesis`; the compat test harness (which aliases sys.modules['hypothesis']
# to this package) registers them here so fallback still reaches the real package.
_REAL_HYP: Any = None
_REAL_ST: Any = None


def register_real_hypothesis(module: Any) -> None:
    """Point fallback at an explicit real-hypothesis module (advanced/shadow use)."""
    global _REAL_HYP, _REAL_ST
    _REAL_HYP = module
    _REAL_ST = module.strategies


def _missing_hypothesis(exc: Exception) -> InvalidArgument:
    return InvalidArgument(
        "This strategy or option is not supported by the hypothesis_fast "
        "engine and falls back to the real `hypothesis` package, which is not "
        "installed. Install it with: pip install hypothesis"
    )


def _real_strategies() -> Any:
    if _REAL_ST is not None:
        return _REAL_ST
    try:
        import hypothesis.strategies as real
    except ImportError as exc:  # pragma: no cover - exercised only without hypothesis
        raise _missing_hypothesis(exc) from exc
    return real


def _real_hypothesis() -> Any:
    if _REAL_HYP is not None:
        return _REAL_HYP
    try:
        import hypothesis as real
    except ImportError as exc:  # pragma: no cover - exercised only without hypothesis
        raise _missing_hypothesis(exc) from exc
    return real


def _hypothesis_base() -> type:
    # Subclass the real hypothesis SearchStrategy when available, so our strategies
    # are accepted by real hypothesis (e.g. interactive `data.draw(our_strategy)`
    # when a test has fully fallen back). Without hypothesis we use a plain object.
    try:
        from hypothesis.strategies._internal.strategies import SearchStrategy as _Base
    except ImportError:
        return object
    return _Base


_BASE = _hypothesis_base()
_HAS_BASE = _BASE is not object


class SearchStrategy(_BASE):  # type: ignore[misc,valid-type]
    """A property-test strategy. Build with the constructors in this module.

    When real hypothesis is installed this is a subclass of its SearchStrategy, so
    real hypothesis can consume our objects directly. For real-hypothesis
    consumption (do_draw/label/is_empty/validate) we delegate to the equivalent
    real strategy; our own engine path uses `_spec` and never touches these.
    """

    def __init__(
        self,
        spec: Spec | None,
        *,
        supported: bool = True,
        hyp: HypFactory | None = None,
    ) -> None:
        if _HAS_BASE:
            super().__init__()
        self._spec = spec
        self._supported = supported and spec is not None
        self._hyp = hyp
        self._real_cached: Any = None

    def __class_getitem__(cls, item: Any) -> Any:
        # `SearchStrategy[str]` — yield the REAL parametrized generic (we subclass it)
        # so type-resolution and repr match real hypothesis (our concrete subclass
        # otherwise isn't generic-subscriptable on 3.14).
        if _HAS_BASE:
            return _BASE[item]  # type: ignore[index]  # real SearchStrategy is generic-subscriptable at runtime
        import types as _types

        return _types.GenericAlias(cls, item)

    def _real(self) -> Any:
        # cache so recursive/self-referential strategies (deferred) resolve every
        # self-reference to the SAME real object, as real hypothesis requires.
        if self._real_cached is None:
            self._real_cached = _build_real(self, _real_strategies())
        return self._real_cached

    # --- real-hypothesis consumption (only used when drawn by real hypothesis) ---
    def do_draw(self, data: Any) -> Any:
        return self._real().do_draw(data)

    @property
    def label(self) -> Any:
        return self._real().label

    # hypothesis computes these from a strategy's structure; the base class would
    # compute them on our (structure-less to hypothesis) object, so delegate.
    @property
    def has_reusable_values(self) -> Any:
        return self._real().has_reusable_values

    @property
    def is_cacheable(self) -> Any:
        return self._real().is_cacheable

    @property
    def branches(self) -> Any:
        return self._real().branches

    # --- composition ---
    def map(self, pack: Callable[[Any], Any]) -> SearchStrategy:
        # mapping an empty strategy stays empty — there's nothing to map.
        if isinstance(self, _NothingStrategy):
            return self
        # identity-map is a noop: `s.map(lambda x: x) is s`. Mirrors hypothesis'
        # own optimization (matters for code that builds map chains generically).
        if _is_identity_function(pack):
            return self
        if self._supported:
            return SearchStrategy(("map", self._spec, pack))
        return _fallback(lambda r: _to_hyp(self, r).map(pack))

    def filter(self, condition: Callable[[Any], object]) -> SearchStrategy:
        # filtering an empty strategy stays empty.
        if isinstance(self, _NothingStrategy):
            return self
        # Filter rewriting for numeric primitives: when the predicate is a simple
        # comparison (`partial(operator.lt, N)`, `lambda x: x > N`, math.isfinite,
        # …), rewrite into tightened bounds via our own analyzer. Falls through
        # to a runtime filter for anything we don't understand. All native — no
        # delegation to real hypothesis.
        if self._supported and self._spec is not None and self._spec[0] in ("int", "float"):
            rewritten = _rewrite_numeric_filter(self, condition)
            if rewritten is not None:
                return rewritten
        if self._supported:
            return SearchStrategy(("filter", self._spec, condition))
        return _fallback(lambda r: _to_hyp(self, r).filter(condition))

    def flatmap(self, expand: Callable[[Any], SearchStrategy]) -> SearchStrategy:
        # flatmap of empty stays empty.
        if isinstance(self, _NothingStrategy):
            return self
        base = self

        def _producer() -> Any:
            drawn = _draw_one(base)
            return _draw_one(expand(drawn))

        return SearchStrategy(
            ("composite", _producer),
            hyp=lambda r: _to_hyp(base, r).flatmap(lambda v: _to_hyp(expand(v), r)),
        )

    def example(self) -> Any:
        """Generate a single example. For debugging only — not for use in tests."""
        return _real_example(_to_hyp(self, _real_strategies()))

    def validate(self) -> None:
        """Check the strategy is well-formed (delegated to the real equivalent)."""
        if _HAS_BASE:
            self._real().validate()

    @property
    def is_empty(self) -> bool:
        if _HAS_BASE:
            return bool(self._real().is_empty)
        return False

    def __or__(self, other: SearchStrategy) -> SearchStrategy:
        # `s | x` only makes sense if x is also a strategy; matching hypothesis,
        # we surface a ValueError early instead of silently building a one_of()
        # over non-strategy values (which would then explode in generation).
        _base = _hypothesis_base()
        if not isinstance(other, SearchStrategy) and not (
            _base is not object and isinstance(other, _base)
        ):
            raise ValueError(
                f"Cannot | a SearchStrategy with {other!r} — both sides must be strategies"
            )
        # flatten one_of chains, as hypothesis does for `a | b | c`
        if (
            self._supported
            and self._spec is not None
            and self._spec[0] == "one_of"
            and isinstance(other, SearchStrategy)
            and other._supported
        ):
            return SearchStrategy(("one_of", list(self._spec[1]) + [other._spec]))
        return one_of(self, other)

    def __repr__(self) -> str:
        # mirror hypothesis's repr by delegating to the real equivalent
        if _HAS_BASE:
            try:
                return repr(self._real())
            except Exception:  # noqa: BLE001 - repr must never raise
                pass
        kind = self._spec[0] if self._spec is not None else "fallback"
        return f"SearchStrategy({kind!r})"


# --- support / fallback plumbing -----------------------------------------

def _is_supported(strategy: Any) -> bool:
    return isinstance(strategy, SearchStrategy) and strategy._supported


def _all_supported(strategies: Iterable[Any]) -> bool:
    return all(_is_supported(s) for s in strategies)


def _elements_all_strategies(elements: Any) -> bool:
    """True if `elements` is a non-empty collection of (only) SearchStrategy."""
    try:
        items = list(elements)
    except TypeError:
        return False
    return bool(items) and all(isinstance(x, SearchStrategy) for x in items)


def scan_spec_for_sampled_strategies(spec: Any) -> tuple[Any, ...] | None:
    """Find a nested ``sampled_from`` whose elements are ALL strategies (#3819).

    Walks a strategy / spec tree generically (so containers like lists/tuples are
    covered without knowing each layout) and returns that sampled_from's elements,
    or None. SearchStrategy elements are NOT descended into — they are the leaves.
    """
    if isinstance(spec, SearchStrategy):
        return scan_spec_for_sampled_strategies(spec._spec)
    if isinstance(spec, tuple):
        if len(spec) >= 2 and spec[0] == "sampled_from" and _elements_all_strategies(spec[1]):
            return tuple(spec[1])
        for item in spec:
            found = scan_spec_for_sampled_strategies(item)
            if found is not None:
                return found
    elif isinstance(spec, list):
        for item in spec:
            found = scan_spec_for_sampled_strategies(item)
            if found is not None:
                return found
    return None


def value_contains_strategy(value: Any) -> bool:
    """True if a generated value (recursively) contains a SearchStrategy — i.e. a
    sampled_from-of-strategies actually yielded a strategy this example (#3819)."""
    if isinstance(value, SearchStrategy):
        return True
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(value_contains_strategy(x) for x in value)
    if isinstance(value, dict):
        return any(value_contains_strategy(x) for x in value.values())
    return False


def _fallback(hyp: HypFactory) -> SearchStrategy:
    return SearchStrategy(None, supported=False, hyp=hyp)


def _real_example(hyp_strategy: Any) -> Any:
    # we deliberately use .example() to draw single values internally; silence
    # hypothesis's interactive-use warning for these intentional calls.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return hyp_strategy.example()


def _draw_one(strategy: Any) -> Any:
    """Draw a single value, using the engine if supported else real hypothesis.

    If the engine can't generate (e.g. an element filter rejects everything), fall
    back to real hypothesis, which handles such degenerate cases.
    """
    # validation: callers like flatmap may receive a non-strategy from an `expand`
    # function (e.g. `just(100).flatmap(lambda n: "a")` — expand returns str). Catch
    # that at draw time with the same error hypothesis raises rather than letting it
    # detonate inside the engine as a confusing RuntimeError.
    if not isinstance(strategy, SearchStrategy) and not (_HAS_BASE and isinstance(strategy, _BASE)):
        raise InvalidArgument(
            f"Expected a SearchStrategy, got {type(strategy).__name__}: {strategy!r}"
        )
    # Drawing from a known-empty strategy (`lists(nothing(), min_size=1)`, …):
    # surface the stored validation error rather than spinning in the engine's
    # reject loop. This is the "validation happens on draw" contract.
    if isinstance(strategy, _NothingStrategy):
        if strategy._validation_error is not None:
            raise InvalidArgument(strategy._validation_error)
        raise InvalidArgument("Cannot draw from nothing() — the strategy has no values")
    return _real_example(_to_hyp(strategy, _real_strategies()))


def _to_spec(x: SearchStrategy) -> Spec:
    if isinstance(x, SearchStrategy) and x._spec is not None:
        return x._spec
    raise InvalidArgument(f"Expected an engine-native SearchStrategy, got {x!r}")


def _to_hyp(strategy: Any, r: Any) -> Any:
    """Convert any strategy to its real-hypothesis equivalent (cached per object)."""
    if not isinstance(strategy, SearchStrategy):
        return strategy  # already a real hypothesis strategy (foreign)
    return strategy._real()


def _build_real(strategy: SearchStrategy, r: Any) -> Any:
    if strategy._hyp is not None:
        return strategy._hyp(r)
    if strategy._spec is None:
        raise InvalidArgument(f"No hypothesis fallback available for {strategy!r}")
    return _spec_to_hyp(strategy._spec, r)


def _spec_to_hyp(spec: Spec, r: Any) -> Any:
    tag = spec[0]
    if tag == "int":
        # our spec stores wide i64 sentinels for "unbounded"; pass None to real
        # hypothesis so its repr/validation matches the user-visible API call
        # (`integers()` not `integers(min_value=-9223372036854775807, ...)`).
        lo = None if spec[1] <= _INT_MIN else spec[1]
        hi = None if spec[2] >= _INT_MAX else spec[2]
        return r.integers(lo, hi)
    if tag == "bool":
        return r.booleans()
    if tag == "float":
        # our spec stores wide sentinels for "unbounded"; real floats() needs None
        # there (it rejects allow_nan together with explicit bounds). Read the bounds
        # BEFORE the len()-gated optionals — a `len(spec) > N` guard narrows `spec` to
        # include shorter tuples, which would make pyright flag spec[1]/spec[2] after.
        lo = None if spec[1] <= _FLOAT_MIN else spec[1]
        hi = None if spec[2] >= _FLOAT_MAX else spec[2]
        nan = spec[3] if len(spec) > 3 else False
        inf = spec[4] if len(spec) > 4 else False
        return r.floats(lo, hi, allow_nan=nan, allow_infinity=inf)
    if tag == "text":
        return r.text(min_size=spec[1], max_size=spec[2])
    if tag == "characters":
        # collapse our defaults (0, 0x10FFFF) to omitted-kwargs so real-hypothesis's
        # repr matches the user-visible API call (`characters()` not
        # `characters(min_codepoint=0, max_codepoint=1114111)`).
        lo = spec[1] if len(spec) > 1 else 0
        hi = spec[2] if len(spec) > 2 else _MAX_CODEPOINT
        kw: dict[str, Any] = {}
        if lo > 0:
            kw["min_codepoint"] = lo
        if hi < _MAX_CODEPOINT:
            kw["max_codepoint"] = hi
        return r.characters(**kw)
    if tag == "none":
        return r.none()
    if tag == "just":
        return r.just(spec[1])
    if tag == "sampled_from":
        return r.sampled_from(spec[1])
    if tag == "one_of":
        return r.one_of(*[_spec_to_hyp(s, r) for s in spec[1]])
    if tag == "tuple":
        return r.tuples(*[_spec_to_hyp(s, r) for s in spec[1]])
    if tag == "list":
        return r.lists(_spec_to_hyp(spec[1], r), min_size=spec[2], max_size=spec[3])
    if tag == "dict":
        return r.dictionaries(
            _spec_to_hyp(spec[1], r), _spec_to_hyp(spec[2], r),
            min_size=spec[3], max_size=spec[4],
        )
    if tag == "fixed_dict":
        return r.fixed_dictionaries({k: _spec_to_hyp(v, r) for k, v in spec[1]})
    if tag == "map":
        return _spec_to_hyp(spec[1], r).map(spec[2])
    if tag == "filter":
        return _spec_to_hyp(spec[1], r).filter(spec[2])
    if tag == "datetime":
        base, span_us = spec[1], spec[2]
        return r.integers(0, max(span_us, 0)).map(
            lambda us: base + _dt.timedelta(microseconds=us)
        )
    if tag == "regex":
        return r.from_regex(spec[1])
    if tag == "nothing":
        return r.nothing()
    if tag == "composite":
        producer = spec[1]

        @r.composite
        def _wrap(draw, _producer=producer):  # noqa: ARG001 — `draw` unused
            return _producer()

        return _wrap()
    raise InvalidArgument(f"Cannot reconstruct real strategy for spec tag {tag!r}")


def _max(max_size: int | None, min_size: int = 0) -> int:
    # unbounded collections: cap generation near hypothesis's small default mean (the
    # uniform 0..cap gives mean cap/2), but never below min_size so the range is valid.
    if max_size is not None:
        return int(max_size)
    return max(int(min_size), _DEFAULT_MAX_SIZE)


# Strategy-construction cache — `st.text() is st.text()` (and likewise for the
# common primitives) is a hypothesis-documented identity invariant. Each builder
# below `@_cached_strategy` looks up by its hashable kwargs; non-hashable args
# silently bypass the cache (matches hypothesis' best-effort caching policy).
_STRATEGY_CACHE: dict[tuple[Any, ...], SearchStrategy] = {}


def _cached_strategy(fn: Callable[..., SearchStrategy]) -> Callable[..., SearchStrategy]:
    name = fn.__name__

    def wrapper(*args: Any, **kwargs: Any) -> SearchStrategy:
        try:
            # Distinguish bool from int (sensitive caching: `floats(min_value=0)`
            # is NOT `floats(min_value=0.0)` — different concrete types).
            key = (name, _typed(args), tuple((k, _typed(v)) for k, v in sorted(kwargs.items())))
            hash(key)
        except TypeError:
            return fn(*args, **kwargs)
        cached = _STRATEGY_CACHE.get(key)
        if cached is not None:
            return cached
        s = fn(*args, **kwargs)
        _STRATEGY_CACHE[key] = s
        return s

    wrapper.__name__ = name
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _typed(v: Any) -> Any:
    """Type-sensitive wrapping for cache keys so `int` vs `float` don't collide
    (hypothesis caches `floats(min_value=0)` distinctly from `floats(min_value=0.0)`),
    and so that `+0.0` / `-0.0` aren't conflated either."""
    if isinstance(v, float):
        # use bit pattern: +0.0/-0.0 collapse otherwise; NaN handled as a string.
        if v != v:  # NaN
            return ("float", "nan")
        import struct
        return ("float", struct.pack("!d", v))
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, tuple):
        return tuple(_typed(x) for x in v)
    return v


# --- primitives -----------------------------------------------------------

@_cached_strategy
def integers(min_value: int | None = None, max_value: int | None = None) -> SearchStrategy:
    # Non-int bounds (float/Decimal/nan/complex/str) need real hypothesis's
    # inward-rounding + validation (e.g. integers(0.1, 0.2) -> InvalidArgument, no int
    # in range); int(...) here would crash on nan or silently truncate Decimal("1.5").
    if not (
        (min_value is None or isinstance(min_value, int))
        and (max_value is None or isinstance(max_value, int))
    ):
        return _fallback(lambda r: r.integers(min_value, max_value))
    lo = _INT_MIN if min_value is None else int(min_value)
    hi = _INT_MAX if max_value is None else int(max_value)
    if lo > hi:
        raise InvalidArgument(f"min_value={lo} > max_value={hi}")
    # values outside i64 fall back to real hypothesis (true bigints)
    if lo < _INT_MIN or hi > _INT_MAX:
        return _fallback(lambda r: r.integers(min_value, max_value))
    return SearchStrategy(("int", lo, hi))


@_cached_strategy
def booleans() -> SearchStrategy:
    return SearchStrategy(("bool",))


@_cached_strategy
def floats(
    min_value: float | None = None,
    max_value: float | None = None,
    *,
    allow_nan: bool | None = None,
    allow_infinity: bool | None = None,
    allow_subnormal: bool | None = None,
    width: int = 64,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> SearchStrategy:
    # Native: the engine's HypFloat ValueTree shrinks toward nice values (0/1/ints)
    # like hypothesis. (exclude_min/max, width<64 and subnormal control are not yet
    # honoured natively.) Native validation of the parameter types so we don't reach
    # the real-hypothesis fallback just to surface an obvious usage error.
    if not isinstance(exclude_min, bool):
        raise InvalidArgument(f"exclude_min={exclude_min!r} must be a bool")
    if not isinstance(exclude_max, bool):
        raise InvalidArgument(f"exclude_max={exclude_max!r} must be a bool")
    if width not in (16, 32, 64):
        raise InvalidArgument(f"width={width!r} must be one of 16, 32, 64")

    def _float_bound_native(v: Any) -> bool:
        return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))

    if not (_float_bound_native(min_value) and _float_bound_native(max_value)):
        # Decimal / Fraction bound coercion and validation (e.g. "0" string is
        # not a valid bound) is hypothesis's job — delegate the rest of the
        # validation path so the user sees the precise diagnostic message.
        return _fallback(
            lambda r: r.floats(
                min_value, max_value, allow_nan=allow_nan, allow_infinity=allow_infinity,
                allow_subnormal=allow_subnormal, width=width,
                exclude_min=exclude_min, exclude_max=exclude_max,
            )
        )
    has_bounds = min_value is not None or max_value is not None
    lo = _FLOAT_MIN if min_value is None else float(min_value)
    hi = _FLOAT_MAX if max_value is None else float(max_value)
    # +0.0 min with -0.0 max is empty under sign-aware ordering -> real validates
    if lo == 0.0 and hi == 0.0 and _copysign(1.0, lo) > _copysign(1.0, hi):
        return _fallback(lambda r: r.floats(min_value, max_value))
    # NaN bounds would panic the Rust range strategy; reject like hypothesis does.
    if lo != lo or hi != hi:
        raise InvalidArgument("min_value and max_value cannot be NaN")
    # subnormal/width/exclusive and INFINITE bounds aren't native yet -> real
    # hypothesis (which also constructs fine and defers the precise error, e.g.
    # "allow_infinity=False excludes min_value=inf", to .validate()).
    _inf = float("inf")
    bound_is_infinite = lo in (_inf, -_inf) or hi in (_inf, -_inf)
    # allow_infinity=True with finite bounds: real hypothesis validates the
    # (in)compatibility (e.g. both bounds finite → no infinity possible → error);
    # our native generator can't, so delegate.
    if (
        allow_subnormal
        or width != 64
        or exclude_min
        or exclude_max
        or bound_is_infinite
        or (allow_infinity is True and has_bounds)
    ):
        return _fallback(
            lambda r: r.floats(
                min_value, max_value, allow_nan=allow_nan, allow_infinity=allow_infinity,
                allow_subnormal=allow_subnormal, width=width,
                exclude_min=exclude_min, exclude_max=exclude_max,
            )
        )
    if lo > hi:
        raise InvalidArgument(f"min_value={lo} > max_value={hi}")
    nan = allow_nan if allow_nan is not None else not has_bounds
    inf = allow_infinity if allow_infinity is not None else not has_bounds
    if nan and has_bounds:
        raise InvalidArgument("Cannot have allow_nan=True together with min_value/max_value bounds")
    return SearchStrategy(("float", lo, hi, bool(nan), bool(inf)))


@_cached_strategy
def none() -> SearchStrategy:
    return SearchStrategy(("none",))


def just(value: object) -> SearchStrategy:
    return SearchStrategy(("just", value))


_NOTHING_SPEC: Spec = ("nothing",)


class _NothingStrategy(SearchStrategy):
    """Native empty strategy. `is_empty` is true; any attempt to generate fails."""

    def __init__(self, validation_error: str | None = None) -> None:
        super().__init__(_NOTHING_SPEC, hyp=lambda r: r.nothing())
        # When a builder constructs an unsatisfiable strategy lazily (e.g.
        # `lists(nothing(), min_size=1)`), hypothesis raises on `.validate()`
        # rather than at construction. We carry the message and re-raise from
        # validate to match that contract — no real-hypothesis fallback needed.
        self._validation_error = validation_error

    @property
    def is_empty(self) -> bool:  # type: ignore[override]
        return True

    def validate(self) -> None:  # type: ignore[override]
        if self._validation_error is not None:
            raise InvalidArgument(self._validation_error)

    def _real(self) -> Any:  # type: ignore[override]
        # Propagate stored validation errors when ANY consumer asks for the real
        # equivalent (e.g. a wrapping fallback strategy is being built that will
        # call `_to_hyp(self, r)` — without this, our error would be silently
        # replaced by a real `nothing()` and the outer construction would
        # succeed instead of raising at validate time).
        if self._validation_error is not None:
            raise InvalidArgument(self._validation_error)
        return super()._real()


_NOTHING_SINGLETON: SearchStrategy | None = None


def nothing() -> SearchStrategy:
    global _NOTHING_SINGLETON
    if _NOTHING_SINGLETON is None:
        _NOTHING_SINGLETON = _NothingStrategy()
    return _NOTHING_SINGLETON


def sampled_from(elements: Sequence[Any]) -> SearchStrategy:
    # set / frozenset / dict_keys are unordered: reject with the same error
    # hypothesis raises, since shrinking would be non-deterministic. Mappings'
    # `.keys()` view inherits set semantics; ordered dicts pass via items.
    if isinstance(elements, (set, frozenset)):
        raise InvalidArgument(
            f"Cannot sample from {type(elements).__name__}, because the order of "
            "elements is arbitrary; pass list(...) or tuple(...) instead"
        )
    # don't materialize the sequence to a list when we can avoid it — `repr` should
    # show whichever type the user passed (`sampled_from(range(0, N))` /
    # `sampled_from((0, 1))` / `[...]`). For a `range` of more than _SAMPLED_FROM_NATIVE_CAP
    # elements we delegate to real hypothesis rather than expand it into the engine —
    # `sampled_from(range(10**100))` must not allocate.
    if hasattr(elements, "__len__"):
        try:
            n = len(elements)
        except OverflowError:
            n = -1  # sentinel: too big to hold
        if n == 0:
            raise InvalidArgument("sampled_from() requires at least one element")
        if n < 0 or n > _SAMPLED_FROM_NATIVE_CAP:
            return _fallback(lambda r: r.sampled_from(elements))
        # snapshot mutable sequences so post-construction mutation of the caller's
        # list/dict doesn't change generated values (hypothesis issue #2507).
        # Immutable: tuple, range, str, bytes — keep as-is to preserve repr.
        if isinstance(elements, list):
            elements = list(elements)
    else:
        elements = list(elements)
        if not elements:
            raise InvalidArgument("sampled_from() requires at least one element")
    return SearchStrategy(("sampled_from", elements))


def one_of(*args: Any) -> SearchStrategy:
    if len(args) == 1 and isinstance(args[0], Iterable) and not isinstance(args[0], SearchStrategy):
        strategies = list(args[0])
    else:
        strategies = list(args)
    if not strategies:
        return nothing()
    # one_of(values…) is a common typo for sampled_from(values); fire the same
    # helpful error hypothesis does when none of the args are strategies.
    _hyp_base = _hypothesis_base()
    if all(
        not isinstance(s, SearchStrategy) and not (_hyp_base is not object and isinstance(s, _hyp_base))
        for s in strategies
    ):
        raise InvalidArgument(
            f"Did you mean st.sampled_from({strategies!r})? "
            "one_of() takes strategies; sampled_from() takes a collection of values."
        )
    if len(strategies) == 1 and isinstance(strategies[0], SearchStrategy):
        return strategies[0]
    if _all_supported(strategies):
        return SearchStrategy(("one_of", [_to_spec(s) for s in strategies]))
    return _fallback(lambda r: r.one_of(*[_to_hyp(s, r) for s in strategies]))


@_cached_strategy
def tuples(*args: SearchStrategy) -> SearchStrategy:
    # An empty-element tuple is empty itself: any nothing() in the slot collapses
    # the whole product. Only check the syntactic-nothing case here — calling
    # `.is_empty` on an arbitrary SearchStrategy can recurse forever on a
    # deferred/self-referential strategy (test_deferred_strategies.test_binary_tree).
    if any(isinstance(s, _NothingStrategy) for s in args):
        return nothing()
    if _all_supported(args):
        return SearchStrategy(("tuple", [_to_spec(s) for s in args]))
    return _fallback(lambda r: r.tuples(*[_to_hyp(s, r) for s in args]))


# --- text -----------------------------------------------------------------

def characters(
    *,
    min_codepoint: int | None = None,
    max_codepoint: int | None = None,
    categories: Any = None,
    exclude_categories: Any = None,
    whitelist_categories: Any = None,
    blacklist_categories: Any = None,
    include_characters: Any = None,
    exclude_characters: Any = None,
    whitelist_characters: Any = None,
    blacklist_characters: Any = None,
    codec: str | None = None,
) -> SearchStrategy:
    kwargs = {
        k: v
        for k, v in dict(
            min_codepoint=min_codepoint, max_codepoint=max_codepoint,
            categories=categories, exclude_categories=exclude_categories,
            whitelist_categories=whitelist_categories,
            blacklist_categories=blacklist_categories,
            include_characters=include_characters, exclude_characters=exclude_characters,
            whitelist_characters=whitelist_characters,
            blacklist_characters=blacklist_characters, codec=codec,
        ).items()
        if v is not None
    }
    category_filters = (
        categories, exclude_categories, whitelist_categories, blacklist_categories,
        include_characters, exclude_characters, whitelist_characters,
        blacklist_characters, codec,
    )
    if any(v is not None for v in category_filters):
        # category/codec/include/exclude filtering needs hypothesis's IntervalSet
        # (union/diff over codepoint ranges). Until we port that, delegate.
        kwargs = {
            k: v for k, v in dict(
                min_codepoint=min_codepoint, max_codepoint=max_codepoint,
                categories=categories, exclude_categories=exclude_categories,
                whitelist_categories=whitelist_categories,
                blacklist_categories=blacklist_categories,
                include_characters=include_characters, exclude_characters=exclude_characters,
                whitelist_characters=whitelist_characters,
                blacklist_characters=blacklist_characters, codec=codec,
            ).items() if v is not None
        }
        return _fallback(lambda r: r.characters(**kwargs))
    # invalid codepoints (non-int, negative, out of range) -> real validates
    def _cp_native(v: Any) -> bool:
        return v is None or (
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= _MAX_CODEPOINT
        )

    if not (_cp_native(min_codepoint) and _cp_native(max_codepoint)):
        return _fallback(
            lambda r: r.characters(min_codepoint=min_codepoint, max_codepoint=max_codepoint)
        )
    lo = 0 if min_codepoint is None else int(min_codepoint)
    hi = _MAX_CODEPOINT if max_codepoint is None else int(max_codepoint)
    if lo > hi:
        raise InvalidArgument(f"Cannot have max_codepoint={hi} < min_codepoint={lo}")
    return SearchStrategy(("characters", lo, hi))


def _sizes_are_int(min_size: Any, max_size: Any) -> bool:
    """True if min/max_size are engine-native (real non-negative ints with min<=max).
    Anything else (non-int types, negative, min>max) falls back to real hypothesis,
    which defers the precise InvalidArgument to .validate() — hypothesis never coerces
    or rejects sizes at construction."""
    ok_min = isinstance(min_size, int) and not isinstance(min_size, bool)
    ok_max = max_size is None or (isinstance(max_size, int) and not isinstance(max_size, bool))
    if not (ok_min and ok_max):
        return False
    if min_size < 0 or (max_size is not None and (max_size < 0 or min_size > max_size)):
        return False
    return True


@_cached_strategy
def text(
    alphabet: str | SearchStrategy | None = None,
    *,
    min_size: int = 0,
    max_size: int | None = None,
) -> SearchStrategy:
    # Hypothesis defers size validation to `.validate()` / draw time, NOT
    # construction. Build a placeholder that re-raises on validate when sizes
    # are not valid non-negative ints.
    def _bad_size(name: str, v: Any) -> str | None:
        if v is None and name == "max_size":
            return None
        if not isinstance(v, int) or isinstance(v, bool):
            return f"{name}={v!r} must be a non-negative int"
        if v < 0:
            return f"{name}={v} must be a non-negative int"
        return None

    err = _bad_size("min_size", min_size) or _bad_size("max_size", max_size)
    if err is None and max_size is not None and max_size < min_size:
        err = f"max_size={max_size} cannot be smaller than min_size={min_size}"
    if err is not None:
        return _NothingStrategy(err)
    mn = int(min_size)
    mx = _max(max_size, mn)
    if alphabet is None:
        return SearchStrategy(("text", mn, mx, 0, _MAX_CODEPOINT))
    if isinstance(alphabet, str):
        if not alphabet:
            if mn > 0:
                raise InvalidArgument("Cannot create non-empty text from empty alphabet")
            return just("")
        # Codec-name alphabets are very likely user error (`text("utf-8")`
        # rather than `characters(codec="utf-8")`). Warn and treat the string
        # as literal alphabet — same observable behaviour, no fallback.
        import codecs

        try:
            codecs.lookup(alphabet)
        except LookupError:
            pass
        else:
            import warnings

            try:
                from hypothesis.errors import HypothesisWarning  # type: ignore[import-not-found]
            except Exception:  # pragma: no cover
                HypothesisWarning = UserWarning  # type: ignore[misc,assignment]
            warnings.warn(
                f"it seems like you are trying to use the codec {alphabet!r} as a text alphabet — "
                f"if so, pass it as `characters(codec={alphabet!r})` instead. "
                "Treating the argument as a literal alphabet for now.",
                HypothesisWarning,
                stacklevel=2,
            )
        # sort the alphabet by codepoint so shrinking the index toward 0 yields
        # the smallest-codepoint char (text("FEDCBA") → "AAA" min).
        chars = "".join(sorted(set(alphabet)))
        return SearchStrategy(
            ("map", ("list", ("int", 0, len(chars) - 1), mn, mx),
             lambda idx: "".join(chars[i] for i in idx))
        )
    # Iterable-but-not-string alphabet (set, tuple of chars, etc.): flatten into
    # a string. Each element must be a single character.
    if not isinstance(alphabet, SearchStrategy) and hasattr(alphabet, "__iter__"):
        flat = []
        for ch in alphabet:
            if not isinstance(ch, str) or len(ch) != 1:
                raise InvalidArgument(
                    f"text() alphabet must contain only single-character strings, got {ch!r}"
                )
            flat.append(ch)
        if not flat:
            if mn > 0:
                raise InvalidArgument("Cannot create non-empty text from empty alphabet")
            return just("")
        chars = "".join(sorted(set(flat)))
        return SearchStrategy(
            ("map", ("list", ("int", 0, len(chars) - 1), mn, mx),
             lambda idx: "".join(chars[i] for i in idx))
        )
    if isinstance(alphabet, SearchStrategy):
        # Strategy alphabet: each draw must be a 1-char string. Build at draw
        # time and assemble via composite producer — `_draw_one` handles the
        # underlying engine call.
        alphabet_strat = alphabet

        def _producer(_alph: SearchStrategy = alphabet_strat, _mn: int = mn, _mx: int = mx) -> str:
            # uniform length within [mn,mx] — engine int shrink pulls toward _mn.
            n = _draw_one(integers(_mn, _mx))
            parts: list[str] = []
            for _ in range(n):
                ch = _draw_one(_alph)
                if not isinstance(ch, str) or len(ch) != 1:
                    raise InvalidArgument(
                        f"text() alphabet must yield single-character strings, got {ch!r}"
                    )
                parts.append(ch)
            return "".join(parts)

        return SearchStrategy(
            ("composite", _producer),
            hyp=lambda r: r.text(_to_hyp(alphabet_strat, r), min_size=mn, max_size=mx),
        )
    raise InvalidArgument(
        f"text() alphabet must be a string, SearchStrategy, or None — got {type(alphabet).__name__}"
    )


def from_regex(regex: Any, *, fullmatch: bool = False, alphabet: Any = None) -> SearchStrategy:
    # Full Python `re` semantics (anchors, look-around, backrefs, unicode \w/\d,
    # inline flags, bytes patterns, fullmatch, alphabet restriction) is a large
    # parser/interpreter — delegate to real hypothesis until we have our own.
    if not isinstance(fullmatch, bool):
        raise InvalidArgument(f"fullmatch={fullmatch!r} must be a bool")
    kw: dict[str, Any] = {"fullmatch": fullmatch}
    if alphabet is not None:
        kw["alphabet"] = (
            _to_hyp(alphabet, _real_strategies())
            if isinstance(alphabet, SearchStrategy)
            else alphabet
        )
    return _fallback(lambda r: r.from_regex(regex, **kw))


def binary(*, min_size: int = 0, max_size: int | None = None) -> SearchStrategy:
    if not _sizes_are_int(min_size, max_size):
        return _fallback(lambda r: r.binary(min_size=min_size, max_size=max_size))
    mx = _max(max_size, min_size)
    return SearchStrategy(("map", ("list", ("int", 0, 255), int(min_size), mx), bytes))


# --- collections ----------------------------------------------------------

def lists(
    elements: SearchStrategy,
    *,
    min_size: int = 0,
    max_size: int | None = None,
    unique: bool = False,
    unique_by: Callable[[Any], Hashable] | tuple[Callable[[Any], Hashable], ...] | None = None,
) -> SearchStrategy:
    # lists of nothing(): with min_size==0 the only satisfiable list is [],
    # with min_size>0 the constraint is unsatisfiable → nothing(). Detect the
    # empty-element case up front so generation doesn't loop trying to draw
    # an impossible item.
    if isinstance(elements, _NothingStrategy):
        if (min_size and min_size > 0) or (max_size is not None and max_size > 0):
            # constructed-but-invalid: error surfaces on .validate() or at draw
            # time. Includes the "has no values" phrasing hypothesis uses so
            # callers `pytest.raises(InvalidArgument, match="has no values")`.
            return _NothingStrategy(
                f"lists(elements=nothing()) has no values to generate: "
                f"the element strategy is empty (min_size={min_size}, max_size={max_size})"
            )
        return just([])
    # size validation is deferred to validate() per hypothesis contract.
    if not _sizes_are_int(min_size, max_size):
        err = f"lists(min_size={min_size!r}, max_size={max_size!r}) requires non-negative ints with min<=max"
        return _NothingStrategy(err)
    mn = int(min_size)
    mx = _max(max_size, mn)

    # Unique lists with min_size>0 / unique_by need hypothesis's index-tracking
    # shrinker to converge — our native composite-loop overshoots the minimum
    # length and is brittle on coupon-collector-style strategies. Delegate.
    if unique or unique_by is not None:
        return _fallback(
            lambda r: r.lists(
                _to_hyp(elements, r), min_size=mn, max_size=max_size,
                unique=unique, unique_by=unique_by,
            )
        )
    if not _is_supported(elements):
        return _fallback(
            lambda r: r.lists(_to_hyp(elements, r), min_size=min_size, max_size=max_size)
        )
    return SearchStrategy(("list", _to_spec(elements), mn, mx))


def sets(elements: SearchStrategy, *, min_size: int = 0, max_size: int | None = None) -> SearchStrategy:
    if isinstance(elements, _NothingStrategy):
        return just(set()) if not min_size else nothing()
    if min_size == 0:
        return lists(elements, max_size=max_size).map(set)
    return _fallback(lambda r: r.sets(_to_hyp(elements, r), min_size=min_size, max_size=max_size))


def frozensets(elements: SearchStrategy, *, min_size: int = 0, max_size: int | None = None) -> SearchStrategy:
    if isinstance(elements, _NothingStrategy):
        return just(frozenset()) if not min_size else nothing()
    if min_size == 0:
        return lists(elements, max_size=max_size).map(frozenset)
    return _fallback(lambda r: r.frozensets(_to_hyp(elements, r), min_size=min_size, max_size=max_size))


def iterables(
    elements: SearchStrategy, *, min_size: int = 0, max_size: int | None = None,
    unique: bool = False, unique_by: Any = None,
) -> SearchStrategy:
    return lists(
        elements, min_size=min_size, max_size=max_size, unique=unique, unique_by=unique_by
    ).map(iter)


def dictionaries(
    keys: SearchStrategy, values: SearchStrategy, *, min_size: int = 0, max_size: int | None = None,
) -> SearchStrategy:
    if min_size == 0:
        return lists(tuples(keys, values), max_size=max_size).map(dict)
    return _fallback(
        lambda r: r.dictionaries(
            _to_hyp(keys, r), _to_hyp(values, r), min_size=min_size, max_size=max_size
        )
    )


def fixed_dictionaries(
    mapping: dict[Any, SearchStrategy], *, optional: dict[Any, SearchStrategy] | None = None,
) -> SearchStrategy:
    # any REQUIRED value being nothing() collapses the whole product → nothing.
    # optional nothing()-values just never appear in the output dict.
    if isinstance(mapping, dict) and any(
        isinstance(v, _NothingStrategy) for v in mapping.values()
    ):
        return nothing()
    # invalid mapping/optional (non-dict, dict subclass like OrderedDict, or keys
    # shared between mapping and optional) -> real hypothesis validates and raises.
    if (
        type(mapping) is not dict
        or (optional is not None and type(optional) is not dict)
        or (
            isinstance(mapping, dict)
            and isinstance(optional, dict)
            and set(mapping) & set(optional)
        )
    ):
        return _fallback(lambda r: r.fixed_dictionaries(mapping, optional=optional))
    # optional nothing()-keys would simply never appear — drop them from the
    # optional dict so we don't ask the engine to draw an unsatisfiable strategy.
    if isinstance(optional, dict):
        optional = {k: v for k, v in optional.items() if not isinstance(v, _NothingStrategy)}
        if not optional:
            optional = None
    required = list(mapping.items())
    all_strats = [v for _, v in required] + ([v for _, v in optional.items()] if optional else [])
    if not _all_supported(all_strats):
        return _fallback(
            lambda r: r.fixed_dictionaries(
                {k: _to_hyp(v, r) for k, v in required},
                optional=({k: _to_hyp(v, r) for k, v in optional.items()} if optional else None),
            )
        )
    if not optional:
        return SearchStrategy(("fixed_dict", [(k, _to_spec(v)) for k, v in required]))

    opt = list(optional.items())
    req_keys = [k for k, _ in required]
    opt_keys = [k for k, _ in opt]
    nreq = len(req_keys)
    specs = [_to_spec(v) for _, v in required] + [
        ("one_of", [("just", _ABSENT), _to_spec(v)]) for _, v in opt
    ]

    def _build(vals: list[Any]) -> dict[Any, Any]:
        out = dict(zip(req_keys, vals[:nreq]))
        for key, val in zip(opt_keys, vals[nreq:]):
            if val is not _ABSENT:
                out[key] = val
        return out

    return SearchStrategy(("map", ("tuple", specs), _build))


# --- builders -------------------------------------------------------------

def _is_union_like(target: Any) -> bool:
    """True if `target` is a Union/Optional (typing.Union[...] / X | Y) — these are
    never constructible directly and must be resolved via `from_type(...)`."""
    import typing
    import types as _types

    origin = typing.get_origin(target)
    if origin is typing.Union:
        return True
    # `int | str` syntax on 3.10+ produces `types.UnionType`.
    if isinstance(target, getattr(_types, "UnionType", ())):
        return True
    return False


def _is_uninstantiable_generic(target: Any) -> bool:
    """True if `target` is a `typing`-module generic alias (like `typing.List`,
    `typing.Dict[...]`, `typing.Optional[int]`) — these can't be instantiated by
    direct call; users almost certainly mean `from_type(...)`."""
    import typing

    if target is type(None):  # NoneType is fine to call (returns None)
        return False
    # typing.List / typing.Dict / Generic[...] alias objects
    if isinstance(
        target,
        (typing._SpecialForm, getattr(typing, "_GenericAlias", type(None))),  # type: ignore[attr-defined]
    ):
        return True
    # `list[int]` etc — types.GenericAlias on 3.9+ — also rejects direct calls
    import types as _types

    if isinstance(target, _types.GenericAlias):
        return True
    return False


def _builds_needs_inference(target: Any, n_pos: int, kw_names: set[str]) -> bool:
    """True if `target` has required parameters not covered by the provided
    strategies — real hypothesis infers those from annotations (and raises helpful
    errors for uninspectable targets), so we delegate."""
    try:
        params = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return True  # uninspectable -> let real hypothesis handle/explain
    required = [
        p
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    return any(
        i >= n_pos and p.name not in kw_names for i, p in enumerate(required)
    )


def builds(*args: Any, **kwargs: SearchStrategy) -> SearchStrategy:
    # signature is (callable, /, *arg_strategies, **kwarg_strategies) — the callable
    # is the FIRST POSITIONAL (so `target=` stays free as a kwarg for callables that
    # happen to have a `target` parameter, e.g. a namedtuple field).
    if not args:
        raise TypeError(
            "builds() must be passed a callable as the first positional argument"
        )
    target = args[0]
    pos = args[1:]
    # `Union`/`Optional` are NEVER valid for builds (they name a choice of types,
    # not a constructor) AND they aren't callable, so we have to handle them
    # BEFORE the generic callable-check below — otherwise the user sees the
    # generic "must be callable" error instead of the helpful from_type hint.
    if _is_union_like(target):
        raise InvalidArgument(
            f"Cannot build({target!r}): Union/Optional names a choice of types. "
            f"Try using from_type({target!r}) "
            f"(try using from_type({target!r}) instead of builds({target!r}))."
        )
    if not callable(target):
        raise InvalidArgument(
            f"The first positional argument to builds() must be callable, got {target!r}"
        )
    # typing generics (`List`, `Dict`, …) ARE callable but raise TypeError on
    # direct instantiation. Surface the same hint hypothesis does — but only
    # when no positional/keyword strategies were supplied; with extra args the
    # user might legitimately mean `builds(List, st.lists(...))` and we let it
    # crash at draw time with the underlying TypeError.
    if not pos and not kwargs and _is_uninstantiable_generic(target):
        raise InvalidArgument(
            f"Cannot build({target!r}): generic alias has no concrete constructor. "
            f"Try using from_type({target!r}) "
            f"(try using from_type({target!r}) instead of builds({target!r}))."
        )
    if (
        not (_all_supported(pos) and _all_supported(kwargs.values()))
        or _builds_needs_inference(target, len(pos), set(kwargs))
    ):
        return _fallback(
            lambda r: r.builds(
                target, *[_to_hyp(a, r) for a in pos],
                **{k: _to_hyp(v, r) for k, v in kwargs.items()},
            )
        )
    arg_specs = [_to_spec(a) for a in pos]
    kw_items = list(kwargs.items())
    kw_keys = [k for k, _ in kw_items]
    all_specs = arg_specs + [_to_spec(v) for _, v in kw_items]
    n = len(arg_specs)
    # explicit `hyp` factory so `repr(s)` (which goes through `_real()`) shows
    # `builds(target, …)` — the user-visible API — rather than the internal
    # `tuple(...).map(<_call>)` spec we use for the native fast path.
    _hyp_factory = lambda r: r.builds(  # noqa: E731
        target,
        *[_to_hyp(a, r) for a in pos],
        **{k: _to_hyp(v, r) for k, v in kwargs.items()},
    )
    if not all_specs:
        return SearchStrategy(("map", ("none",), lambda _: target()), hyp=_hyp_factory)

    def _call(vals: list[Any]) -> Any:
        return target(*vals[:n], **dict(zip(kw_keys, vals[n:])))

    return SearchStrategy(("map", ("tuple", all_specs), _call), hyp=_hyp_factory)


def composite(f: Callable[..., T]) -> Callable[..., SearchStrategy]:
    """Decorator turning a `def s(draw, ...)` function into a strategy factory.

    Delegated to real hypothesis: shrinking the internal draws (and `@composite`'s
    own signature validation / reprs) needs the conjecture machinery. The user's
    `draw()` calls work because our strategies subclass hypothesis's SearchStrategy.
    """
    real_factory = _real_strategies().composite(f)  # validates f's signature now

    def make_strategy(*args: Any, **kwargs: Any) -> SearchStrategy:
        return _fallback(lambda r: real_factory(*args, **kwargs))

    make_strategy.__name__ = getattr(f, "__name__", "composite")
    make_strategy.__doc__ = getattr(f, "__doc__", None)
    return make_strategy


def deferred(definition: Callable[[], SearchStrategy]) -> SearchStrategy:
    # Recursive/self-referential strategies need depth tracking + emptiness
    # detection (otherwise generation runs forever). Real hypothesis owns that
    # machinery — we delegate `deferred` until our engine grows the same
    # depth-budget primitives. Validation is still ours.
    if not callable(definition):
        raise InvalidArgument(
            f"deferred() expected a callable, got {type(definition).__name__}: {definition!r}"
        )
    return _fallback(lambda r: r.deferred(lambda: _to_hyp(definition(), r)))


def recursive(
    base: SearchStrategy,
    extend: Callable[[SearchStrategy], SearchStrategy],
    **kwargs: Any,
) -> SearchStrategy:
    # defer to real hypothesis (handles depth/leaf bounds and shrinking properly)
    return _fallback(
        lambda r: r.recursive(_to_hyp(base, r), lambda s: _to_hyp(extend(s), r), **kwargs)
    )


# --- date / time ----------------------------------------------------------

# Native: an integer OFFSET is mapped onto the value so the engine's shrink-toward-0
# reproduces hypothesis's canonical minimum — dates/datetimes offset from 2000-01-01,
# times microseconds-from-midnight, timedeltas microseconds-from-zero. datetimes/times
# also draw a `fold` bit (hypothesis varies it even for naive values). Edge cases ->
# real hypothesis: a single value (min==max, must preserve object identity), non-
# date/time bounds (so 'fish' -> InvalidArgument), tz, and out-of-i64 ranges.

_ORD_2000 = _dt.date(2000, 1, 1).toordinal()
_DT_2000 = _dt.datetime(2000, 1, 1)
_US = _dt.timedelta(microseconds=1)


def _time_to_us(t: _dt.time) -> int:
    return ((t.hour * 60 + t.minute) * 60 + t.second) * 1_000_000 + t.microsecond


def _time_from_us(us: int) -> _dt.time:
    return _dt.time(
        us // 3_600_000_000, (us // 60_000_000) % 60, (us // 1_000_000) % 60, us % 1_000_000
    )


def _within_i64(*xs: int) -> bool:
    return all(_INT_MIN <= x <= _INT_MAX for x in xs)


def dates(min_value: _dt.date = _dt.date.min, max_value: _dt.date = _dt.date.max) -> SearchStrategy:
    if not (isinstance(min_value, _dt.date) and isinstance(max_value, _dt.date)):
        return _fallback(lambda r: r.dates(min_value, max_value))  # real validates
    if min_value == max_value:
        return just(min_value)
    lo, hi = min_value.toordinal() - _ORD_2000, max_value.toordinal() - _ORD_2000
    return integers(lo, hi).map(lambda off: _dt.date.fromordinal(off + _ORD_2000))


def datetimes(
    min_value: _dt.datetime = _dt.datetime.min, max_value: _dt.datetime = _dt.datetime.max,
    *, timezones: Any = None, allow_imaginary: bool = True,
) -> SearchStrategy:
    native = (
        timezones is None
        and isinstance(allow_imaginary, bool)  # non-bool -> real raises InvalidArgument
        and isinstance(min_value, _dt.datetime)
        and isinstance(max_value, _dt.datetime)
    )
    if native and min_value == max_value:
        return just(min_value)
    if native:
        lo, hi = int((min_value - _DT_2000) / _US), int((max_value - _DT_2000) / _US)
        if _within_i64(lo, hi):
            return tuples(integers(lo, hi), booleans()).map(
                lambda ob: (_DT_2000 + _dt.timedelta(microseconds=ob[0])).replace(fold=int(ob[1]))
            )
    return _fallback(
        lambda r: r.datetimes(
            min_value, max_value,
            timezones=(r.none() if timezones is None else _to_hyp(timezones, r)),
            allow_imaginary=allow_imaginary,
        )
    )


def times(min_value: _dt.time = _dt.time.min, max_value: _dt.time = _dt.time.max, *, timezones: Any = None) -> SearchStrategy:
    native = (
        timezones is None
        and isinstance(min_value, _dt.time)
        and isinstance(max_value, _dt.time)
    )
    if native and min_value == max_value:
        return just(min_value)
    if native:
        return tuples(integers(_time_to_us(min_value), _time_to_us(max_value)), booleans()).map(
            lambda ob: _time_from_us(ob[0]).replace(fold=int(ob[1]))
        )
    return _fallback(
        lambda r: r.times(
            min_value, max_value,
            timezones=(r.none() if timezones is None else _to_hyp(timezones, r)),
        )
    )


def timedeltas(min_value: _dt.timedelta = _dt.timedelta.min, max_value: _dt.timedelta = _dt.timedelta.max) -> SearchStrategy:
    # native type check — hypothesis raises InvalidArgument for non-timedelta args.
    if not isinstance(min_value, _dt.timedelta):
        raise InvalidArgument(
            f"min_value={min_value!r} must be a datetime.timedelta instance"
        )
    if not isinstance(max_value, _dt.timedelta):
        raise InvalidArgument(
            f"max_value={max_value!r} must be a datetime.timedelta instance"
        )
    if min_value > max_value:
        raise InvalidArgument(
            f"min_value={min_value!r} cannot exceed max_value={max_value!r}"
        )
    if min_value == max_value:
        return just(min_value)
    lo = int(min_value / _US)
    hi = int(max_value / _US)
    if _within_i64(lo, hi):
        return integers(lo, hi).map(lambda us: _dt.timedelta(microseconds=us))
    # Out-of-i64 microsecond span (full timedelta.min..max) — split into days and
    # microseconds within each day; recombine. Keeps generation/shrinking native.
    day_us = 24 * 60 * 60 * 1_000_000
    lo_d, lo_r = divmod(lo, day_us)
    hi_d, hi_r = divmod(hi, day_us)

    def _mk(parts: tuple[int, int], _lo_d: int = lo_d, _hi_d: int = hi_d, _lo_r: int = lo_r, _hi_r: int = hi_r) -> _dt.timedelta:
        d, r = parts
        total = d * day_us + r
        # clamp into [lo, hi] (shrink may overshoot end-of-day boundary)
        total = max(int(min_value / _US), min(int(max_value / _US), total))
        return _dt.timedelta(microseconds=total)

    return tuples(integers(lo_d, hi_d), integers(0, day_us - 1)).map(_mk)


# --- numbers --------------------------------------------------------------

def complex_numbers(
    *,
    min_magnitude: float = 0,
    max_magnitude: float | None = None,
    allow_nan: bool | None = None,
    allow_infinity: bool | None = None,
    allow_subnormal: bool | None = None,
    width: int = 128,
) -> SearchStrategy:
    # magnitude constraints need the joint (re,im) distribution -> still fall back;
    # the common unconstrained case is native (built on native floats). allow_subnormal
    # must fall back when EXPLICITLY set (incl. False): native floats emit subnormals by
    # default, so only real hypothesis can honour an explicit allow_subnormal=False.
    # width != 128 (incl. invalid like None/16/196/256) needs real hypothesis (it
    # validates the width and splits it across the two components).
    if (
        min_magnitude
        or min_magnitude is None
        or max_magnitude is not None
        or allow_nan
        or allow_infinity
        or allow_subnormal is not None
        or width != 128
    ):
        kw: dict[str, Any] = {"min_magnitude": min_magnitude, "max_magnitude": max_magnitude, "width": width}
        if allow_nan is not None:
            kw["allow_nan"] = allow_nan
        if allow_infinity is not None:
            kw["allow_infinity"] = allow_infinity
        if allow_subnormal is not None:
            kw["allow_subnormal"] = allow_subnormal
        return _fallback(lambda r: r.complex_numbers(**kw))
    # unconstrained: native (re, im) tuple of finite floats -> complex. The float
    # shrink (toward 0/±1) gives the canonical minimum, e.g. minimal nonzero-imaginary
    # == 1j (real shrinks to 0.0, imag to the smallest nonzero == 1.0).
    part = floats(allow_nan=False, allow_infinity=False)
    return tuples(part, part).map(lambda p: complex(p[0], p[1]))


def fractions(
    min_value: Any = None,
    max_value: Any = None,
    *,
    max_denominator: int | None = None,
) -> SearchStrategy:
    """Generate `fractions.Fraction` values in `[min_value, max_value]`.

    Native path: numerator × inverse-denominator pair, clamped into the requested
    range. Shrinks toward small numerators / denominator=1 so the canonical
    `Fraction(0, 1)` is reachable from inside any range that contains zero.
    Bounds and `max_denominator` arithmetic with arbitrary-precision rationals
    are real hypothesis's domain — we delegate for those constrained cases.
    """
    from fractions import Fraction

    if max_denominator is not None and (
        not isinstance(max_denominator, int) or isinstance(max_denominator, bool) or max_denominator < 1
    ):
        raise InvalidArgument(
            f"max_denominator={max_denominator!r} must be a positive int or None"
        )
    from decimal import Decimal
    import math as _math

    for name, v in (("min_value", min_value), ("max_value", max_value)):
        if v is not None and not isinstance(v, (int, float, Fraction, Decimal, str)):
            raise InvalidArgument(
                f"{name}={v!r} must be int, float, Fraction, Decimal, str or None"
            )
        if isinstance(v, float) and _math.isnan(v):
            raise InvalidArgument(f"{name}={v!r} cannot be NaN")
    # Unconstrained / int-bounded case: build natively. Coerce any Decimal/
    # Fraction bound to its integer floor/ceil for the numerator range — losing
    # exact bound precision is OK for the *engine* path (the .filter clause
    # below tightens the output back to the requested range).
    def _to_int_bound(v: Any, ceil: bool) -> int | None:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            import math as _m

            return _m.ceil(v) if ceil else _m.floor(v)
        # Fraction / Decimal
        return int(v.__ceil__()) if ceil else int(v.__floor__())

    if max_denominator is None:
        lo_int = _to_int_bound(min_value, ceil=True)
        hi_int = _to_int_bound(max_value, ceil=False)
        lo = _INT_MIN if lo_int is None else lo_int
        hi = _INT_MAX if hi_int is None else hi_int
        if (
            _INT_MIN <= lo <= _INT_MAX
            and _INT_MIN <= hi <= _INT_MAX
            and lo <= hi
        ):
            # filter so the post-multiplication value lands in the precise range.
            min_frac = Fraction(min_value) if min_value is not None and not isinstance(min_value, str) else None
            max_frac = Fraction(max_value) if max_value is not None and not isinstance(max_value, str) else None

            def _ok(f: Fraction, _mn: Fraction | None = min_frac, _mx: Fraction | None = max_frac) -> bool:
                if _mn is not None and f < _mn:
                    return False
                return not (_mx is not None and f > _mx)

            return tuples(integers(lo, hi), integers(0, 999)).map(
                lambda parts: Fraction(parts[0], parts[1] + 1)
            ).filter(_ok)
        if lo > hi:
            raise InvalidArgument(
                f"min_value={min_value!r} cannot exceed max_value={max_value!r}"
            )
    # Float bounds / max_denominator / Fraction bounds — real hypothesis's
    # rational arithmetic is more accurate than anything we can write here in
    # under a hundred lines, so delegate until we port that.
    return _fallback(
        lambda r: r.fractions(min_value, max_value, max_denominator=max_denominator)
    )


def decimals(
    min_value: Any = None,
    max_value: Any = None,
    *,
    allow_nan: bool | None = None,
    allow_infinity: bool | None = None,
    places: int | None = None,
) -> SearchStrategy:
    # places/allow_nan/allow_infinity/bounds + context-precision validation are
    # intricate; delegate to real hypothesis for fidelity.
    return _fallback(
        lambda r: r.decimals(
            min_value,
            max_value,
            allow_nan=allow_nan,
            allow_infinity=allow_infinity,
            places=places,
        )
    )


def uuids(*, version: int | None = None, allow_nil: bool = False) -> SearchStrategy:
    import uuid

    # native validation — hypothesis raises InvalidArgument for unsupported versions.
    if version is not None and (
        not isinstance(version, int) or isinstance(version, bool) or version not in (1, 2, 3, 4, 5)
    ):
        raise InvalidArgument(
            f"version={version!r} is not supported — must be one of 1..5 or None"
        )
    if not isinstance(allow_nil, bool):
        raise InvalidArgument(f"allow_nil={allow_nil!r} must be a bool")
    nil = uuid.UUID(int=0)

    def _producer(_v: int | None = version, _nil: uuid.UUID = nil, _allow_nil: bool = allow_nil) -> uuid.UUID:
        import secrets

        raw = secrets.token_bytes(16)
        if _v is None:
            u = uuid.UUID(bytes=raw)
        else:
            u = uuid.UUID(bytes=raw, version=_v)
        # `allow_nil` ⇒ include the nil UUID with low probability so shrink can
        # reach it; ⇒ False ⇒ never emit it (reject and resample).
        if not _allow_nil and u == _nil:
            # vanishingly rare draw with 128 random bits; emit a 1-bit replacement.
            u = uuid.UUID(int=1) if _v is None else uuid.UUID(int=1, version=_v)
        return u

    return SearchStrategy(
        ("composite", _producer),
        hyp=lambda r: r.uuids(version=version, allow_nil=allow_nil),
    )


# --- network --------------------------------------------------------------

def ip_addresses(*, v: int | None = None, network: Any = None) -> SearchStrategy:
    import ipaddress

    # native validation: v must be int 4 or 6 (not 4.0/'4'/5).
    if v is not None and not (isinstance(v, int) and not isinstance(v, bool) and v in (4, 6)):
        raise InvalidArgument(f"v={v!r} must be 4, 6, or None")
    # network-restricted addresses: sample an integer offset within the subnet
    # and add to network_address. This handles both IPv4 and IPv6 uniformly.
    if network is not None:
        if isinstance(network, str):
            try:
                network = ipaddress.ip_network(network)
            except ValueError as exc:
                raise InvalidArgument(f"network={network!r} is not a valid network") from exc
        if not isinstance(network, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
            raise InvalidArgument(
                f"network={network!r} must be an ip_network, IPv4Network, or IPv6Network instance"
            )
        if v is not None and network.version != v:
            raise InvalidArgument(
                f"v={v} conflicts with network={network!r} which is IPv{network.version}"
            )
        net = network
        addr_cls = ipaddress.IPv4Address if net.version == 4 else ipaddress.IPv6Address
        base = int(net.network_address)
        last = int(net.broadcast_address)
        span = last - base
        if span <= 0:
            return just(addr_cls(base))
        # split into i64-sized chunks if needed (IPv6 /0 = 128-bit span > i64).
        if span <= _INT_MAX:
            return integers(0, span).map(lambda i: addr_cls(base + i))
        # 128-bit fallback: draw two i64 halves and combine.
        hi = (span >> 63) + 1
        return tuples(integers(0, hi - 1), integers(0, _INT_MAX)).map(
            lambda parts: addr_cls(base + min(span, (parts[0] << 63) | parts[1]))
        )
    if v == 6:
        return tuples(*(integers(0, 255) for _ in range(16))).map(lambda o: ipaddress.IPv6Address(bytes(o)))
    return tuples(*(integers(0, 255) for _ in range(4))).map(lambda o: ipaddress.IPv4Address(bytes(o)))


def emails() -> SearchStrategy:
    return from_regex(r"[a-z]{1,8}@[a-z]{1,8}\.[a-z]{2,3}")


def permutations(values: Sequence[Any]) -> SearchStrategy:
    # Sets/dicts/etc. have arbitrary iteration order — permuting them is
    # undefined, so reject up front (hypothesis raises the same error).
    if isinstance(values, (set, frozenset, dict)):
        raise InvalidArgument(
            f"Cannot permute a {type(values).__name__} — pass a list/tuple/range"
        )
    # Native permutation: generate a permutation as a list of indices into the
    # original sequence and rebuild. The engine's int shrink pulls each index
    # toward 0, which biases the permutation toward identity (= the input order),
    # mirroring hypothesis's preferred minimum.
    snapshot = list(values)
    n = len(snapshot)
    if n <= 1:
        return just(list(snapshot))
    # generate via a list of integers and `random.shuffle`-like selection.
    # We use the engine's `lists(integers(0, n-1), min=n, max=n)` and map to a
    # permutation by ordering the seen indices first then back-filling.
    base = lists(integers(0, n - 1), min_size=n, max_size=n)

    def _to_perm(idx_list: list[int], _snap: list[Any] = snapshot, _n: int = n) -> list[Any]:
        seen: list[int] = []
        seen_set: set[int] = set()
        for i in idx_list:
            if i not in seen_set:
                seen.append(i)
                seen_set.add(i)
        # back-fill any missing indices in ascending order — keeps the shrink
        # bias toward identity (low indices come out first).
        for i in range(_n):
            if i not in seen_set:
                seen.append(i)
        return [_snap[i] for i in seen]

    return base.map(_to_perm)


def slices(size: int) -> SearchStrategy:
    # Native port of hypothesis's `slices` composite (strategies/_internal/core.py).
    # Builds (start, stop, step, neg_step?, neg_start?, neg_stop?) and folds them
    # via `.map` — keeps the whole pipeline on our engine, no fallback.
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise InvalidArgument(f"slices() size must be a non-negative int, got {size!r}")
    if size == 0:
        # only the step can vary; None or any non-zero int.
        return one_of(none(), integers().filter(bool)).map(
            lambda step: slice(None, None, step)
        )

    int_in = one_of(integers(0, size - 1), none())
    int_out = one_of(integers(0, size), none())
    # We need to sample step based on start/stop, so use a tuple+map. We draw a
    # larger envelope for step (1..size) then clamp inside _mk to mirror real.
    return tuples(
        int_in,                # start
        int_out,               # stop
        integers(1, size),     # step magnitude
        booleans(),            # negate step?
        booleans(),            # offset start by -size?
        booleans(),            # offset stop by -size?
    ).map(lambda t: _mk_slice(t, size))


def _mk_slice(t: tuple[Any, Any, int, bool, bool, bool], size: int) -> slice:
    start, stop, step_mag, neg_step, neg_start, neg_stop = t
    if start is None and stop is None:
        max_step = size
    elif start is None:
        max_step = stop  # type: ignore[assignment]
    elif stop is None:
        max_step = start
    else:
        max_step = abs(start - stop)
    step = min(step_mag, max_step or 1)
    if (neg_step and start == stop) or ((stop or 0) < (start or 0)):
        step = -step
    if neg_start and start is not None:
        start -= size
    if neg_stop and stop is not None:
        stop -= size
    return slice(start, stop, step)


# --- interactive ----------------------------------------------------------

class DataObject:
    """Returned by data(); `.draw(strategy)` produces a value at test time."""

    __slots__ = ()

    def draw(self, strategy: SearchStrategy, label: str | None = None) -> Any:
        base = _hypothesis_base()
        if not (isinstance(strategy, SearchStrategy) or (base is not object and isinstance(strategy, base))):
            raise InvalidArgument(
                f"Cannot draw from {strategy!r}: it is not a SearchStrategy. "
                "(Did you mean to pass a strategy, not a value?)"
            )
        result = _draw_one(strategy)
        # #3819: drawing a sampled_from-of-strategies yields a strategy — record so
        # an escaping TypeError mentioning SearchStrategy gets the one_of hint.
        if (
            isinstance(strategy, SearchStrategy)
            and strategy._spec is not None
            and strategy._spec[0] == "sampled_from"
            and _elements_all_strategies(strategy._spec[1])
        ):
            from . import control

            control.record_sampled_from_strategies(strategy._spec[1])
        return result

    def __repr__(self) -> str:
        return "data(...)"


def data() -> SearchStrategy:
    return SearchStrategy(("just", DataObject()), hyp=lambda r: r.data())


# --- type-driven ----------------------------------------------------------

# Our own type registry, keyed by EXACT type identity (not equality, since types
# compare by identity anyway). `from_type(T)` checks here first and returns the
# registered SearchStrategy as-is — preserves `from_type(T) is registered_strat`
# identity that hypothesis tests rely on (see test_issue_2951_regression). For
# anything not in our registry (generics, abstract bases, forward refs, registered
# resolvers like Sequence[int]) we still delegate to real hypothesis.
_TYPE_REGISTRY: dict[type, SearchStrategy] = {}


def register_type_strategy(custom_type: type, strategy: Any) -> Any:
    # Mirror the registration into the real-hypothesis registry — its generic
    # resolution (e.g. `from_type(List[T])` looking up `T`) only sees what's
    # registered there. Register in real FIRST so its validation (nothing() →
    # InvalidArgument, etc) runs before we touch our own identity registry;
    # otherwise a failed registration would leak into `_TYPE_REGISTRY`.
    real_st = _real_strategies()
    real_strategy = _to_hyp(strategy, real_st) if isinstance(strategy, SearchStrategy) else strategy
    result = real_st.register_type_strategy(custom_type, real_strategy)
    # Store OUR strategy by exact type identity so `from_type(T) is registered_strat`
    # round-trips. We guard against unhashable type objects (custom metaclass with
    # `__hash__ = None`) — they simply can't sit in a dict, so identity round-trip
    # doesn't apply and from_type falls through to the real-hypothesis path.
    if isinstance(strategy, SearchStrategy):
        try:
            _TYPE_REGISTRY[custom_type] = strategy
        except TypeError:
            pass  # unhashable custom_type — skip the identity cache
    return result


def from_type(thing: Any) -> SearchStrategy:
    # Identity short-circuit: if a SearchStrategy was registered for THIS exact
    # type, return that strategy object directly so `from_type(T) is strat` and
    # `from_type(T) == strat` both hold. Hypothesis tests assert this invariant.
    if isinstance(thing, type):
        try:
            cached = _TYPE_REGISTRY.get(thing)
        except TypeError:
            cached = None  # unhashable type — no identity cache for it
        if cached is not None:
            return cached
        # Built-in primitives: serve natively so the most common `from_type(int)`
        # / `from_type(str)` / etc. paths don't reach the real-hypothesis lookup
        # graph at all. Anything else (generics, abstract bases, forward refs,
        # registered resolvers) still goes through `_fallback`.
        if thing is bool:
            return booleans()
        if thing is int:
            return integers()
        if thing is float:
            return floats()
        if thing is bytes:
            return binary()
        if thing is str:
            return text()
        if thing is type(None):
            return none()
    # Generics, abstract bases, forward refs, registered callable resolvers — real
    # hypothesis owns the resolution graph; we just delegate.
    return _fallback(lambda r: r.from_type(thing))


def __getattr__(name: str) -> Any:
    """Drop-in fallback: strategy names we don't define natively defer to the
    real hypothesis.strategies. Native ones above shadow this."""
    try:
        real = _real_strategies()
    except Exception:
        raise AttributeError(name) from None
    try:
        return getattr(real, name)
    except AttributeError:
        raise AttributeError(name) from None
