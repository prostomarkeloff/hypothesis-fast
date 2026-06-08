"""Transparent fallback to real hypothesis where the engine can't help.

Each test here uses a strategy/option the Rust engine doesn't support; @given
should delegate the whole test to real hypothesis and still pass.
"""

import pytest

from hypothesis_fast import given
from hypothesis_fast import strategies as st
from hypothesis_fast.strategies import _is_supported


class Point:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


def test_unsupported_strategies_are_marked():
    # fractions, recursive, from_type(custom), guaranteed-min_size sets still fall back
    assert not _is_supported(st.fractions())
    assert not _is_supported(st.sets(st.integers(), min_size=1))
    assert not _is_supported(st.recursive(st.integers(), st.lists))
    assert not _is_supported(st.from_type(Point))
    # ...but plain ones are — incl. floats, sets/dicts (min_size=0), and date/time
    assert _is_supported(st.integers())
    assert _is_supported(st.lists(st.integers()))
    assert _is_supported(st.floats(allow_nan=True))
    assert _is_supported(st.sets(st.integers()))
    assert _is_supported(st.dictionaries(st.integers(), st.integers()))
    assert _is_supported(st.dates())
    assert _is_supported(st.datetimes())


@given(st.floats(allow_nan=True, allow_infinity=True))
def test_floats_with_nan_is_native(f):
    assert isinstance(f, float)


@given(st.from_type(int))
def test_from_type_builtin(n):
    assert isinstance(n, int)


@given(st.recursive(st.booleans(), lambda kids: st.lists(kids, max_size=3)))
def test_recursive_falls_back(value):
    # either a leaf bool or a (possibly nested) list of them
    def ok(v):
        return isinstance(v, bool) or (isinstance(v, list) and all(ok(x) for x in v))

    assert ok(value)


@given(st.lists(st.sets(st.integers(), max_size=2), max_size=3))
def test_unsupported_child_propagates_to_container(xs):
    # a container of an unsupported element also falls back
    assert isinstance(xs, list)
    assert all(isinstance(x, set) for x in xs)


def test_container_with_unsupported_child_is_unsupported():
    assert not _is_supported(st.lists(st.fractions()))
    assert not _is_supported(st.tuples(st.integers(), st.recursive(st.integers(), st.lists)))


@given(st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=90), min_size=1, max_size=3))
def test_characters_filtering_falls_back(s):
    # capital ASCII letters only, via real hypothesis (filtered characters alphabet)
    assert s and all("A" <= c <= "Z" for c in s)


def test_foreign_real_strategy_runs_via_fallback():
    real_st = pytest.importorskip("hypothesis.strategies")

    @given(real_st.integers(0, 5))
    def prop(n):
        assert 0 <= n <= 5

    prop()
