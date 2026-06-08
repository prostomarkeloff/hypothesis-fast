"""settings + the HealthCheck/Phase/Verbosity enums.

A hypothesis-compatible surface. Only `max_examples` currently changes engine
behaviour; the rest are accepted and stored for API compatibility so existing
`@settings(...)` decorations import and run unchanged.
"""

from __future__ import annotations

import contextlib
import enum
from collections.abc import Callable, Iterator
from typing import Any

from .errors import InvalidArgument


def _note_deprecation(message: str, *, since: str, has_codemod: bool = False, stacklevel: int = 2) -> None:
    """Emit a HypothesisDeprecationWarning. Delegates to real hypothesis's note_deprecation
    (kept real by the shim / present in a plain install) so the warning class + filtering
    match exactly; degrades to a plain warning if it's unavailable."""
    try:
        from hypothesis.utils.deprecation import note_deprecation as _nd

        _nd(message, since=since, has_codemod=has_codemod, stacklevel=stacklevel + 1)
    except Exception:  # noqa: BLE001 - real hypothesis absent
        import warnings

        from .errors import HypothesisDeprecationWarning

        warnings.warn(message, HypothesisDeprecationWarning, stacklevel=stacklevel + 1)


class _IntDeprecatedMeta(enum.EnumMeta):
    """Real hypothesis (2025-11-05) re-keyed Verbosity/Phase/HealthCheck off strings;
    constructing one from a bare int (`Verbosity(2)`, `Phase(0)`, `HealthCheck(1)`) is
    the deprecated integer pathway. We keep the int values — our engine's `verbosity >=`
    comparisons and the Rust side rely on them — but warn on int construction to match.
    A genuine member (already an int under IntEnum) and bools are exempt."""

    def __call__(cls, value: Any, *args: Any, **kwargs: Any) -> Any:
        member = super().__call__(value, *args, **kwargs)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and not isinstance(value, cls)
        ):
            _note_deprecation(
                f"Passing {cls.__name__}({value}) as an integer is deprecated. "
                f"Hypothesis now treats {cls.__name__} values as strings, not integers. "
                f"Use {cls.__name__}.{member.name} instead.",
                since="2025-11-05",
            )
        return member


class _HealthCheckMeta(_IntDeprecatedMeta):
    # return_value / not_a_test_method are deprecated: still accessible as
    # HealthCheck.return_value, but excluded from iteration (and so from list()/all()).
    def __iter__(cls):  # type of `cls` is the enum class
        deprecated = ("return_value", "not_a_test_method")
        # iterate __members__ (typed by the concrete enum) so the yielded type is
        # HealthCheck, not the metaclass — keeps list(HealthCheck)/all() well-typed.
        return iter(m for name, m in HealthCheck.__members__.items() if name not in deprecated)


class HealthCheck(enum.Enum, metaclass=_HealthCheckMeta):
    data_too_large = 1
    filter_too_much = 2
    too_slow = 3
    return_value = 5
    large_base_example = 7
    not_a_test_method = 8
    function_scoped_fixture = 9
    differing_executors = 10
    nested_given = 11

    @classmethod
    def all(cls) -> list[HealthCheck]:
        _note_deprecation(
            "`HealthCheck.all()` is deprecated; use `list(HealthCheck)` instead.",
            since="2023-04-16",
            has_codemod=True,
        )
        return list(cls)


# return_value / not_a_test_method: always-an-error, so suppressing them is deprecated.
_DEPRECATED_HEALTH_CHECKS = (HealthCheck.return_value, HealthCheck.not_a_test_method)


class Phase(enum.IntEnum, metaclass=_IntDeprecatedMeta):
    explicit = 0
    reuse = 1
    generate = 2
    target = 3
    shrink = 4
    explain = 5


class Verbosity(enum.IntEnum, metaclass=_IntDeprecatedMeta):
    quiet = 0
    normal = 1
    verbose = 2
    debug = 3


_NOT_SET = object()

# Field name -> default. Mirrors the hypothesis settings surface we accept.
_DEFAULTS: dict[str, Any] = {
    "max_examples": 100,
    "derandomize": False,
    "deadline": None,
    "database": None,
    "suppress_health_check": (),
    "verbosity": Verbosity.normal,
    "phases": tuple(Phase),
    "stateful_step_count": 50,
    "report_multiple_bugs": True,
    "print_blob": False,
    "backend": "hypothesis",
}


# Active-settings stack (top = settings of the currently-running @given test) and
# registered profiles. `settings.default` reads the top of the stack during a test,
# else the current profile's settings — mirroring hypothesis's settings context.
_SETTINGS_STACK: list[settings] = []
_PROFILES: dict[str, settings] = {}
_current_profile_name = "default"
_base_settings: settings | None = None


class _SettingsMeta(type):
    @property
    def default(cls) -> settings:
        if _SETTINGS_STACK:
            return _SETTINGS_STACK[-1]
        prof = _PROFILES.get(_current_profile_name)
        return prof if prof is not None else _base_settings  # type: ignore[return-value]

    @property
    def _current_profile(cls) -> str:
        # hypothesis exposes the active profile name as a class attribute on
        # `settings`; upstream tests read `settings._current_profile` directly.
        return _current_profile_name

    def __setattr__(cls, name: str, value: Any) -> None:
        if name in _DEFAULTS:
            raise AttributeError(
                f"Cannot assign {name!r} on the settings class; pass it to settings(...) "
                "or settings.register_profile(...) instead"
            )
        super().__setattr__(name, value)


class settings(metaclass=_SettingsMeta):
    """Decorator/holder for per-test configuration.

    Usage mirrors hypothesis::

        @settings(max_examples=500)
        @given(st.integers())
        def test_x(n): ...

    `settings.default` is the settings of the currently-running test (or the active
    profile's settings outside a test).
    """

    def __init__(self, parent: settings | None = None, **kwargs: Any) -> None:
        if parent is None:
            base = dict(_DEFAULTS)
        elif hasattr(parent, "_kwargs"):
            base = dict(parent._kwargs)
        else:
            # Interop: real hypothesis constructs `hypothesis.settings(parent=<real
            # settings>, ...)` for internal health-check suppression; under native-default
            # `hypothesis.settings` is aliased to us, so the parent is a real settings with
            # no `_kwargs`. Read each known field off it by attribute (defaulting on miss).
            base = {k: getattr(parent, k, v) for k, v in _DEFAULTS.items()}
        for key, value in kwargs.items():
            if key not in _DEFAULTS:
                raise InvalidArgument(f"Invalid argument to settings(): {key!r}")
            base[key] = value
        # max_examples / stateful_step_count must be positive integers (a run with zero of
        # either can't do anything). Validate only when passed explicitly.
        for _name in ("max_examples", "stateful_step_count"):
            if _name in kwargs:
                _v = kwargs[_name]
                if not isinstance(_v, int) or isinstance(_v, bool) or _v < 1:
                    raise InvalidArgument(
                        f"{_name}={_v!r} must be at least one. You can disable example "
                        f"generation with the `phases` setting instead."
                        if _name == "max_examples"
                        else f"{_name}={_v!r} must be at least one."
                    )
        dl = base.get("deadline")
        if dl is not None and not isinstance(dl, bool) and not (
            isinstance(dl, (int, float)) or hasattr(dl, "total_seconds")
        ):
            raise InvalidArgument(
                f"deadline={dl!r} must be a number of milliseconds, a timedelta, or None"
            )
        shc = base.get("suppress_health_check")
        if shc is not None:
            try:
                raw = list(shc)
            except TypeError:
                raise InvalidArgument(
                    f"suppress_health_check={shc!r} must be an iterable of HealthCheck members"
                ) from None
            members = []
            for member in raw:
                if isinstance(member, HealthCheck):
                    members.append(member)
                elif isinstance(member, int) and not isinstance(member, bool):
                    # the deprecated integer pathway — HealthCheck(int) emits the warning
                    members.append(HealthCheck(member))
                else:
                    raise InvalidArgument(
                        f"suppress_health_check={shc!r} must contain only HealthCheck members; "
                        f"got {member!r}"
                    )
            # return_value / not_a_test_method are always-an-error, so suppressing them
            # is deprecated (real hypothesis _settings.py:481).
            for member in members:
                if member in _DEPRECATED_HEALTH_CHECKS:
                    _note_deprecation(
                        f"The {member.name} health check is deprecated, because this is "
                        "always an error.",
                        since="2023-03-15",
                    )
            base["suppress_health_check"] = tuple(members)
        # verbosity: coerce the deprecated integer pathway (Verbosity(int) warns).
        if "verbosity" in kwargs:
            vb = kwargs["verbosity"]
            if isinstance(vb, int) and not isinstance(vb, bool) and not isinstance(vb, Verbosity):
                base["verbosity"] = Verbosity(vb)
        # Validate/normalise phases only when the user passed them explicitly — values
        # inherited from a real-hypothesis parent (interop) use the real Phase enum.
        if "phases" in kwargs:
            ph = kwargs["phases"]
            try:
                raw_phases = list(ph)
            except TypeError:
                raise InvalidArgument(
                    f"phases={ph!r} must be a collection of Phase members"
                ) from None
            phase_members = []
            for member in raw_phases:
                if isinstance(member, Phase):
                    phase_members.append(member)
                elif isinstance(member, int) and not isinstance(member, bool):
                    # the deprecated integer pathway — Phase(int) emits the warning
                    phase_members.append(Phase(member))
                else:
                    raise InvalidArgument(
                        f"phases={ph!r} must contain only Phase members; got {member!r}"
                    )
            # Phase is an IntEnum, so sorting orders by the canonical phase sequence;
            # set() dedupes (matching hypothesis's settings(phases=...) normalisation).
            base["phases"] = tuple(sorted(set(phase_members)))
        self._kwargs = base

    def __getattr__(self, name: str) -> Any:
        # only reached for names not set as instance attrs
        try:
            return self.__dict__["_kwargs"][name]
        except KeyError:
            raise AttributeError(name) from None

    def __call__(self, test: Callable[..., Any]) -> Callable[..., Any]:
        if getattr(test, "_hp_is_fallback", False):
            # the test was delegated to real hypothesis; map our settings onto it
            from .strategies import _real_hypothesis

            real_hyp = _real_hypothesis()
            wrapped = real_hyp.settings(max_examples=self.max_examples, deadline=None)(test)
            wrapped._hp_is_fallback = True  # type: ignore[attr-defined]
            return wrapped
        # both orders (above/below @given) just stash the settings on the callable;
        # the engine wrapper reads it at call time, given() picks it up otherwise.
        test._hypothesis_internal_use_settings = self  # type: ignore[attr-defined]
        # Marker the pytest plugin checks to reject `@settings` on a function WITHOUT
        # `@given` ("completely pointless") — test_settings_alone runs a sub-pytest where the
        # real hypothesis plugin is active and looks for exactly this attribute.
        test._hypothesis_internal_settings_applied = True  # type: ignore[attr-defined]
        return test

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in self._kwargs.items())
        return f"settings({inner})"

    def show_changed(self) -> str:
        """The settings that differ from the defaults, as `name=value` pairs (shortest
        first). Mirrors hypothesis's `settings.show_changed()` — the real pytest plugin's
        `pytest_report_header` calls `settings.default.show_changed()` on every session, so a
        nested pytest run that auto-loads the hypothesis plugin needs this to not crash."""
        bits = []
        for name, default_value in _DEFAULTS.items():
            value = self._kwargs.get(name, default_value)
            if value != default_value:
                bits.append(f"{name}={value!r}")
        return ", ".join(sorted(bits, key=len))

    @classmethod
    def register_profile(
        cls, name: str, parent: settings | None = None, **kwargs: Any
    ) -> None:
        if not isinstance(name, str):
            raise InvalidArgument(f"name={name!r} must be a string")
        # Registering a profile while a test is running (an @settings decorator has
        # pushed its settings onto the stack) is deprecated: the change can't take
        # effect mid-run and usually signals a misplaced call. Real hypothesis
        # _settings.py:1129 warns; register at module level instead.
        if _SETTINGS_STACK:
            _note_deprecation(
                "Cannot register a settings profile when the current settings differ "
                "from the current profile (usually due to an @settings decorator). "
                "Register profiles at module level instead.",
                since="2025-11-15",
            )
        _PROFILES[name] = settings(parent, **kwargs)

    @classmethod
    def get_profile(cls, name: str) -> settings:
        try:
            return _PROFILES[name]
        except KeyError:
            raise InvalidArgument(f"No profile called {name!r}") from None

    @classmethod
    def load_profile(cls, name: str) -> None:
        global _current_profile_name
        if name not in _PROFILES:
            raise InvalidArgument(f"No profile called {name!r}")
        _current_profile_name = name

    @classmethod
    def get_current_profile_name(cls) -> str:
        return _current_profile_name


_base_settings = settings()
_PROFILES["default"] = _base_settings


@contextlib.contextmanager
def apply_settings(s: Any) -> Iterator[None]:
    """Make `s` the active `settings.default` for the duration of a test run.

    For our own settings, push the local stack. For a real-hypothesis settings
    object (cover files import `settings` from `hypothesis._settings`), defer to
    real hypothesis's own `local_settings` so *its* `settings.default` updates too.
    """
    if not isinstance(s, settings):
        # resolve real hypothesis's context manager WITHOUT wrapping the yield (so a
        # failing test body's exception isn't swallowed by an `except` around it).
        cm = None
        try:
            from hypothesis._settings import local_settings

            cm = local_settings(s)
        except Exception:  # noqa: BLE001 - no real hypothesis -> plain no-op context
            cm = None
        if cm is not None:
            with cm:
                yield
        else:
            yield
        return
    _SETTINGS_STACK.append(s)
    try:
        yield
    finally:
        _SETTINGS_STACK.pop()
