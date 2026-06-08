"""Native ``LazyStrategy`` + ``defines_strategy`` for the ``extra.*`` frontends.

Real hypothesis's ``defines_strategy`` wraps a function so that calling it returns
a real ``LazyStrategy`` — whose ``do_validate`` asserts ``isinstance(result, real
SearchStrategy)`` and so rejects our native strategies. The extra ports therefore
use this native equivalent: a thin subclass of the native ``SearchStrategy`` that
defers the function call to draw time and draws the resulting *native* strategy
through our engine. This keeps laziness (deferred validation + clean repr) and the
``force_reusable_values`` override that array fill-inference relies on, while every
draw stays on the native fast path.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from hypothesis_fast._engine import SearchStrategy


def _arg_repr(args: tuple, kwargs: dict) -> str:
    bits = [repr(a) for a in args]
    bits += [f"{k}={v!r}" for k, v in kwargs.items()]
    return ", ".join(bits)


class LazyStrategy(SearchStrategy):
    """Defer ``function(*args, **kwargs)`` (which returns a native strategy) until
    draw time, then draw it. Mirrors the public surface of real hypothesis's
    LazyStrategy that the extra modules touch (``has_reusable_values``, ``is_empty``,
    ``wrapped_strategy``, ``repr``)."""

    def __init__(
        self,
        function: Callable[..., Any],
        args: tuple,
        kwargs: dict,
        *,
        force_repr: str | None = None,
        force_reusable: bool | None = None,
    ) -> None:
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._force_repr = force_repr
        self._force_reusable = force_reusable
        self._wrapped: Any = None

    @property
    def wrapped_strategy(self) -> Any:
        if self._wrapped is None:
            self._wrapped = self._function(*self._args, **self._kwargs)
        return self._wrapped

    def do_draw(self, data: Any) -> Any:
        return data.draw(self.wrapped_strategy)

    @property
    def is_empty(self) -> bool:
        return bool(self.wrapped_strategy.is_empty)

    @property
    def has_reusable_values(self) -> bool:
        if self._force_reusable is not None:
            return self._force_reusable
        return bool(self.wrapped_strategy.has_reusable_values)

    def __repr__(self) -> str:
        if self._force_repr is not None:
            return self._force_repr
        return f"{self._function.__name__}({_arg_repr(self._args, self._kwargs)})"


def defines_strategy(
    *, force_reusable_values: bool = False, try_non_lazy: bool = False
) -> Callable[[Callable[..., Any]], Callable[..., LazyStrategy]]:
    """Native analogue of ``hypothesis.strategies._internal.utils.defines_strategy``:
    turns a strategy-returning function into one that returns a lazily-evaluated
    native ``LazyStrategy``. ``try_non_lazy`` is accepted for signature parity and
    ignored (we always go lazy)."""

    def decorator(strategy_definition: Callable[..., Any]) -> Callable[..., LazyStrategy]:
        @functools.wraps(strategy_definition)
        def inner(*args: Any, **kwargs: Any) -> LazyStrategy:
            return LazyStrategy(
                strategy_definition,
                args,
                kwargs,
                force_reusable=True if force_reusable_values else None,
            )

        return inner

    return decorator


def unwrap_strategies(s: Any) -> Any:
    """Peel our ``LazyStrategy`` wrappers to the concrete native strategy. Used by
    ``arrays()`` for a fast-path optimization that is simply skipped for native
    strategies (they are not real ``MappedStrategy`` instances)."""
    seen: set[int] = set()
    while isinstance(s, LazyStrategy) and id(s) not in seen:
        seen.add(id(s))
        s = s.wrapped_strategy
    return s
