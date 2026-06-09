"""given() driver behaviour: pass/fail, shrinking, assume, settings, example,
fixtures, async."""

import asyncio

import pytest

from hypothesis_fast import HealthCheck, assume, example, given, settings
from hypothesis_fast import strategies as st


@given(st.integers(), st.integers())
def test_addition_commutes(a, b):
    assert a + b == b + a


def test_failing_property_reports_falsifying_example():
    @given(st.integers(0, 1000))
    def prop(n):
        assert n < 10

    with pytest.raises(AssertionError) as exc:
        prop()
    # the original exception is re-raised, with the falsifying example as a note;
    # shrinking drives n to the minimal failing boundary value, 10
    notes = getattr(exc.value, "__notes__", [])
    assert any("Falsifying example" in n for n in notes)
    assert any("prop(10)" in n for n in notes)


def test_shrinking_finds_minimal():
    seen = []

    @given(st.integers(0, 10_000))
    def prop(n):
        if n >= 100:
            seen.append(n)
            raise AssertionError("too big")

    with pytest.raises(AssertionError):
        prop()
    # the last (shrunk) reproduction should be the minimal boundary value
    assert seen[-1] == 100


@given(st.integers())
def test_assume(n):
    assume(n % 2 == 0)
    assert n % 2 == 0


@settings(max_examples=5)
@given(st.integers())
def test_settings_max_examples(n):
    assert isinstance(n, int)


@example(0)
@example(7)
@given(st.integers(0, 100))
def test_example_runs_explicit(n):
    assert 0 <= n <= 100


@given(n=st.integers(0, 5))
def test_keyword_strategy(n):
    assert 0 <= n <= 5


@pytest.fixture
def offset():
    return 1000


@given(st.integers(0, 10))
def test_with_fixture(offset, n):
    assert offset == 1000
    assert 0 <= n <= 10


@given(st.integers(0, 10))
async def test_async_body(n):
    await asyncio.sleep(0)
    assert 0 <= n <= 10


class TestMethods:
    @given(st.integers(0, 10))
    def test_method(self, n):
        assert 0 <= n <= 10


def test_healthcheck_shares_identity_with_upstream():
    """When the real `hypothesis` is importable, our HealthCheck must BE its enum.

    The upstream pytest plugin gates its function-scoped-fixture warning on
    `HealthCheck.function_scoped_fixture in settings.suppress_health_check`, compared by
    enum identity. Two distinct enums make that membership test silently false, so the
    plugin re-flags suppressed fixtures in any consumer suite that keeps the upstream
    plugin enabled. Regression for that cross-package enum-identity mismatch.
    """
    hypothesis = pytest.importorskip("hypothesis")

    assert HealthCheck is hypothesis.HealthCheck
    suppressed = settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    assert hypothesis.HealthCheck.function_scoped_fixture in suppressed.suppress_health_check


def test_function_scoped_fixture_suppression_respected_under_upstream_plugin(pytester, monkeypatch):
    """End-to-end: with the upstream `hypothesis` pytest plugin enabled, suppressing the
    function-scoped-fixture health check through our `@settings` must actually suppress it.

    Our own `make test` runs with `-p no:hypothesispytest`, so this drives a clean
    subprocess pytest (PYTEST_ADDOPTS cleared) where the plugin auto-loads via its entry
    point — exactly the consumer configuration that surfaced the bug.
    """
    pytest.importorskip("hypothesis")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    pytester.makepyfile(
        """
        import pytest

        from hypothesis_fast import HealthCheck, given, settings
        from hypothesis_fast import strategies as st


        @pytest.fixture
        def offset():
            return 1000


        @settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
        @given(n=st.integers(0, 10))
        def test_uses_function_scoped_fixture(offset, n):
            assert offset == 1000
            assert 0 <= n <= 10
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)
