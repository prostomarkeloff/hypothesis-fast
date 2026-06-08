"""Each strategy generates values of the right shape and respects its bounds."""

import datetime as dt
from decimal import Decimal
from fractions import Fraction

from hypothesis_fast import given
from hypothesis_fast import strategies as st


@given(st.integers(0, 10))
def test_integers_bounds(n):
    assert isinstance(n, int)
    assert 0 <= n <= 10


@given(st.booleans())
def test_booleans(b):
    assert b is True or b is False


@given(st.floats(-1.0, 1.0))
def test_floats_bounds(f):
    assert isinstance(f, float)
    assert -1.0 <= f <= 1.0


@given(st.text(min_size=1, max_size=5))
def test_text_size(s):
    assert isinstance(s, str)
    assert 1 <= len(s) <= 5


@given(st.text(alphabet="abc", min_size=3, max_size=3))
def test_text_alphabet(s):
    assert len(s) == 3
    assert set(s) <= set("abc")


@given(st.none())
def test_none(x):
    assert x is None


@given(st.just(42))
def test_just(x):
    assert x == 42


@given(st.sampled_from([1, 2, 3]))
def test_sampled_from(x):
    assert x in (1, 2, 3)


@given(st.one_of(st.integers(0, 5), st.none()))
def test_one_of(x):
    assert x is None or 0 <= x <= 5


@given(st.tuples(st.integers(0, 3), st.booleans()))
def test_tuples(t):
    assert isinstance(t, tuple) and len(t) == 2
    assert 0 <= t[0] <= 3 and isinstance(t[1], bool)


@given(st.lists(st.integers(0, 9), min_size=2, max_size=4))
def test_lists(xs):
    assert isinstance(xs, list)
    assert 2 <= len(xs) <= 4
    assert all(0 <= x <= 9 for x in xs)


@given(st.lists(st.integers(0, 100), unique=True, max_size=5))
def test_lists_unique(xs):
    assert len(xs) == len(set(xs))


@given(st.sets(st.integers(0, 50), max_size=5))
def test_sets(s):
    assert isinstance(s, set)
    assert len(s) <= 5


@given(st.frozensets(st.integers(0, 50), max_size=5))
def test_frozensets(s):
    assert isinstance(s, frozenset)


@given(st.dictionaries(st.integers(0, 20), st.text(max_size=3), max_size=4))
def test_dictionaries(d):
    assert isinstance(d, dict)
    assert len(d) <= 4
    assert all(isinstance(k, int) for k in d)


@given(st.fixed_dictionaries({"a": st.integers(0, 3), "b": st.booleans()}))
def test_fixed_dictionaries(d):
    assert set(d) == {"a", "b"}
    assert 0 <= d["a"] <= 3


@given(st.fixed_dictionaries({"a": st.integers(0, 3)}, optional={"b": st.booleans()}))
def test_fixed_dictionaries_optional(d):
    assert "a" in d
    assert set(d) <= {"a", "b"}


@given(st.binary(min_size=1, max_size=4))
def test_binary(b):
    assert isinstance(b, bytes)
    assert 1 <= len(b) <= 4


@given(st.builds(complex, st.floats(-1, 1), st.floats(-1, 1)))
def test_builds(c):
    assert isinstance(c, complex)


@given(st.dates(dt.date(2000, 1, 1), dt.date(2001, 1, 1)))
def test_dates(d):
    assert dt.date(2000, 1, 1) <= d <= dt.date(2001, 1, 1)


@given(st.datetimes(dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 2)))
def test_datetimes(d):
    assert dt.datetime(2000, 1, 1) <= d <= dt.datetime(2000, 1, 2)


@given(st.from_regex(r"[a-f]{2,4}"))
def test_from_regex(s):
    import re

    assert re.search(r"[a-f]{2,4}", s)


@given(st.complex_numbers(max_magnitude=10))
def test_complex_numbers(c):
    assert isinstance(c, complex)


@given(st.fractions(max_denominator=20))
def test_fractions(f):
    assert isinstance(f, Fraction)


@given(st.decimals())
def test_decimals(d):
    assert isinstance(d, Decimal)


@given(st.uuids())
def test_uuids(u):
    import uuid

    assert isinstance(u, uuid.UUID)


@given(st.ip_addresses())
def test_ip_addresses(ip):
    import ipaddress

    assert isinstance(ip, ipaddress.IPv4Address)


@given(st.integers(0, 5).map(lambda n: n * 2))
def test_map(n):
    assert n % 2 == 0 and 0 <= n <= 10


@given(st.integers(0, 100).filter(lambda n: n % 2 == 0))
def test_filter(n):
    assert n % 2 == 0


@given(st.integers(1, 5).flatmap(lambda n: st.lists(st.just(n), min_size=n, max_size=n)))
def test_flatmap(xs):
    assert len(xs) == xs[0]


@st.composite
def _pair_with_sum(draw):
    total = draw(st.integers(0, 10))
    part = draw(st.integers(0, total))
    return (total, part)


@given(_pair_with_sum())
def test_composite(pair):
    total, part = pair
    assert 0 <= part <= total <= 10


@given(st.data())
def test_data_object(data):
    n = data.draw(st.integers(0, 5))
    assert 0 <= n <= 5
