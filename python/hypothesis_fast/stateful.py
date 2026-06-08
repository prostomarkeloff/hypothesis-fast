"""Native stateful testing (RuleBasedStateMachine), running on the native engine.

This is a from-scratch reimplementation of `hypothesis.stateful` that drives the
step loop against OUR native `ConjectureData` — so a rule's strategy draws go
through the fast native `do_draw` instead of crossing the real-hypothesis
`draw_node_foreign` interop wall (which is what the real-stateful fallback hit on
`randoms()`, `builds()`, etc.). The user-facing surface (the machine class, the
decorators, `Bundle`) is necessarily Python; the generation and shrinking stay in
the Rust engine, reached via `core.given(st.data())`.

Mirrors upstream `hypothesis/stateful.py` semantics: `@rule`/`@initialize`/
`@invariant`/`@precondition`, `Bundle`/`consumes`/`multiple`, `target=`/`targets=`,
the per-step draw structure (forced stop-boolean, init-then-regular rule
selection, bundle-index `shrink_towards=len`), and the `state = M(); v1 =
state.rule(...); state.teardown()` reproduction trace.
"""

from __future__ import annotations

import collections
import dataclasses
import inspect
import io
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar
from unittest import TestCase

from . import strategies as st
from ._engine import SearchStrategy as _NativeSearchStrategy
from .core import given
from .errors import FailedHealthCheck, InvalidArgument
from .settings import HealthCheck, Verbosity, settings as Settings

# InvalidDefinition isn't one of errors.py's own classes (it's resolved lazily via
# its __getattr__, which pyright sees as `object`), so import it from the real module
# directly — kept real by the alias shim and present in a plain install, and the same
# class the cover suite raises.
from hypothesis.errors import FlakyStrategyDefinition, InvalidDefinition

# Real-hypothesis reporting/build-context hooks (kept real by the alias shim and in
# a plain install). Used only for output gating + the is_final trace; best-effort —
# fall back to no output if reporting is somehow unavailable.
try:  # pragma: no cover - import wiring
    from hypothesis.control import current_build_context as _current_build_context
    from hypothesis.reporting import current_verbosity as _current_verbosity
    from hypothesis.reporting import report as _report
except Exception:  # noqa: BLE001 - degrade to no output if reporting is unavailable
    _current_build_context = None
    _current_verbosity = None
    _report = None

# Real hypothesis's pretty-printer, used to render a bundle value by its variable name (via
# singleton_pprinters keyed on id) — kept real by the shim and present in a plain install.
try:  # pragma: no cover - import wiring
    from hypothesis.vendor.pretty import RepresentationPrinter as _RepresentationPrinter
except Exception:  # noqa: BLE001
    _RepresentationPrinter = None


def _is_singleton(obj: object) -> bool:
    """True if two separately created instances of `obj` share an id (interned), so it's
    unsafe to key a name-printer on id(obj). Mirrors upstream."""
    if isinstance(obj, int) and -5 <= obj <= 256:
        return True
    return isinstance(obj, bool) or obj is None

try:  # pragma: no cover - import wiring
    from hypothesis.internal.conjecture.engine import BUFFER_SIZE
except Exception:  # noqa: BLE001
    BUFFER_SIZE = 8 * 1024


# A real-hypothesis settings object may reach us (upstream's test_settings.py imports
# `settings` from the un-aliased `hypothesis._settings`, and a class-level @settings on a
# machine then carries it). Recognise it, and coerce it into our own Settings.
try:  # pragma: no cover - import wiring
    from hypothesis._settings import settings as _RealSettings
except Exception:  # noqa: BLE001
    _RealSettings = None


def _is_settings(obj: Any) -> bool:
    return isinstance(obj, Settings) or (
        _RealSettings is not None and isinstance(obj, _RealSettings)
    )


def _coerce_settings(s: Any) -> Any:
    """Return `s` if it's already our Settings; if it's a real-hypothesis settings, read the
    fields our step loop needs into our Settings (mapping enums by name)."""
    if isinstance(s, Settings):
        return s
    return Settings(
        max_examples=s.max_examples,
        stateful_step_count=s.stateful_step_count,
        print_blob=getattr(s, "print_blob", False),
        report_multiple_bugs=getattr(s, "report_multiple_bugs", True),
        deadline=None,
        verbosity=Verbosity[s.verbosity.name],
        suppress_health_check=[HealthCheck[h.name] for h in s.suppress_health_check],
    )


# Drive the step loop from the native Rust StatefulRunner (src/stateful.rs). The Python
# _run_body driver is kept for reference + differential testing (flip this to compare).
_USE_RUST_STATEFUL = True


def _add_note(exc: BaseException, note: str) -> None:
    if hasattr(exc, "add_note"):
        try:
            exc.add_note(note)
        except Exception:  # noqa: BLE001 - non-standard exc
            pass


def _is_final() -> bool:
    if _current_build_context is None:
        return False
    try:
        return bool(_current_build_context().is_final)
    except Exception:  # noqa: BLE001 - no build context (shouldn't happen inside @given)
        return False


_VERBOSITY_RANK = {"quiet": 0, "normal": 1, "verbose": 2, "debug": 3}


def _verbosity_at_least(level: Verbosity) -> bool:
    # current_verbosity() may be real hypothesis's Verbosity enum (cross-enum compares
    # break), so rank by name instead of comparing enum members directly.
    if _current_verbosity is None:
        return False
    name = getattr(_current_verbosity(), "name", "")
    if not isinstance(name, str):
        name = ""
    return _VERBOSITY_RANK.get(name, -1) >= _VERBOSITY_RANK.get(level.name, 99)


# ----------------------------------------------------------------------------- #
# Markers + setup state
# ----------------------------------------------------------------------------- #

RULE_MARKER = "hypothesis_stateful_rule"
INITIALIZE_RULE_MARKER = "hypothesis_stateful_initialize_rule"
PRECONDITIONS_MARKER = "hypothesis_stateful_preconditions"
INVARIANT_MARKER = "hypothesis_stateful_invariant"


def _rule_qualname(f: Any) -> str:
    return f.__qualname__.rsplit("<locals>.", 1)[-1]


@dataclass(frozen=True)
class _SetupState:
    rules: list[Rule]
    invariants: list[Invariant]
    initializers: list[Rule]


# ----------------------------------------------------------------------------- #
# Bundles, multiple(), VarReference
# ----------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VarReference:
    name: str


def _machine_for(data: Any) -> RuleBasedStateMachine:
    """The state machine currently driving `data`. Stashed on the native cd (which has
    a __dict__) at run start, so bundle draws nested arbitrarily deep inside native
    collections — e.g. `lists(consumes(b))` — can reach it. Per-cd, so a rule body that
    starts a nested @given/state-machine (a fresh cd) can't see the outer machine."""
    machine = getattr(data, "_hf_stateful_machine", None)
    if machine is None:
        raise InvalidArgument(
            "Bundles can only be used inside a running RuleBasedStateMachine"
        )
    return machine


class BundleReferenceStrategy(_NativeSearchStrategy):
    """Draws a *reference* (a VarReference naming a stored value) from a bundle. The
    step loop resolves references to their values before invoking the rule, and prints
    them by variable name. `consumes(b)` and top-level `Bundle` rule args draw references."""

    def __init__(self, name: str, *, consume: bool = False) -> None:
        super().__init__()
        self.name = name
        self.consume = consume

    def do_draw(self, data: Any) -> Any:
        machine = _machine_for(data)
        bundle = machine.bundle(self.name)
        if not bundle:
            data.mark_invalid(f"Cannot draw from empty bundle {self.name!r}")
        # Shrink towards the right so earlier-produced values are easier to delete.
        position = data.draw_integer(0, len(bundle) - 1, shrink_towards=len(bundle))
        if self.consume:
            return bundle.pop(position)
        return bundle[position]

    @property
    def is_empty(self) -> bool:
        # A bundle is assumed to grow over time, so its reference strategy is never empty
        # (matches upstream Bundle.calc_is_empty) — else native lists()/one_of() would
        # treat it as nothing() and refuse to draw.
        return False

    @property
    def has_reusable_values(self) -> bool:
        return False

    def validate(self) -> None:
        return None


class Bundle(_NativeSearchStrategy):
    """A named collection of values produced and consumed by rules, usable as a
    strategy: a rule's `target(s)` add their return value(s) to the bundle, and the
    Bundle (or `consumes(bundle)`, or `lists(consumes(bundle))`, ...) draws from it.

    A native `SearchStrategy` subclass, so it composes inside native collections and
    `.flatmap`/`.map` like any other strategy; `do_draw` resolves a drawn reference to
    its concrete value via the running machine (reached through the cd)."""

    def __init__(
        self, name: str, *, consume: bool = False, draw_references: bool = True
    ) -> None:
        super().__init__()
        self.name = name
        self.consume = consume
        self.draw_references = draw_references
        self._reference_strategy = BundleReferenceStrategy(name, consume=consume)

    def do_draw(self, data: Any) -> Any:
        # Always return the concrete VALUE (so a Bundle composed into builds()/lists()/etc.
        # yields real values). Printing a bundle value by its variable name is handled by the
        # machine's RepresentationPrinter (singleton_pprinters keyed by id), not here. The
        # draw_references flag only steers flatmap (see below).
        machine = _machine_for(data)
        reference = data.draw(self._reference_strategy)
        return machine.names_to_values[reference.name]

    @property
    def is_empty(self) -> bool:
        return False

    @property
    def has_reusable_values(self) -> bool:
        return False

    def validate(self) -> None:
        return None

    def flatmap(self, expand: Any) -> Any:
        # Drawing a value (not a reference) before flatmapping, so `b.flatmap(f)` sees
        # the concrete value — mirrors upstream Bundle.flatmap's draw_references toggle.
        if self.draw_references:
            return type(self)(
                self.name, consume=self.consume, draw_references=False
            ).flatmap(expand)
        return super().flatmap(expand)

    def __repr__(self) -> str:
        if not self.consume:
            return f"Bundle(name={self.name!r})"
        return f"Bundle(name={self.name!r}, consume=True)"

    def __hash__(self) -> int:
        # Hashable so st.sampled_from's rule-selection label calc hits its fast path.
        return hash(("Bundle", self.name))


class BundleConsumer(Bundle):
    def __init__(self, bundle: Bundle) -> None:
        super().__init__(bundle.name, consume=True)


def consumes(bundle: Bundle) -> Any:
    """Mark a bundle as consumed: each value drawn from it (here or nested in a
    collection) is removed from the bundle. Returns a strategy usable like any other."""
    if not isinstance(bundle, Bundle):
        raise TypeError("Argument to be consumed must be a bundle.")
    return BundleConsumer(bundle)


@dataclass(frozen=True)
class MultipleResults:
    values: tuple[Any, ...]

    def __iter__(self) -> Any:
        return iter(self.values)


def multiple(*args: Any) -> MultipleResults:
    """Return multiple values from a rule, to be routed to its target bundle(s).

    `return multiple()` (no args) ends a rule without producing any value.
    """
    return MultipleResults(args)


# ----------------------------------------------------------------------------- #
# Rule / Invariant
# ----------------------------------------------------------------------------- #


@dataclass
class Rule:
    targets: tuple[str, ...]
    function: Any
    arguments: dict[str, Any]
    preconditions: tuple[Any, ...]
    bundles: tuple[Bundle, ...] = field(init=False, default=())
    # name -> native strategy OR _BundleArg, in sorted-key order.
    arguments_strategies: dict[str, Any] = field(init=False, default_factory=dict)
    _cached_hash: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        bundles: list[Bundle] = []
        for k, v in sorted(self.arguments.items()):
            # A top-level Bundle arg draws a REFERENCE (resolved to its value before the
            # rule runs, and printed by variable name); the bundle is also recorded so the
            # rule is only valid while that bundle is non-empty. A Bundle nested inside a
            # collection (lists(consumes(b))) stays as-is and draws values inline.
            if isinstance(v, Bundle):
                bundles.append(v)
                self.arguments_strategies[k] = BundleReferenceStrategy(
                    v.name, consume=isinstance(v, BundleConsumer)
                )
            else:
                self.arguments_strategies[k] = v
        self.bundles = tuple(bundles)

    def __hash__(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = hash(
                (self.targets, self.function, tuple(self.arguments), self.preconditions)
            )
        return self._cached_hash


@dataclass(frozen=True)
class Invariant:
    function: Any
    preconditions: tuple[Any, ...]
    check_during_init: bool


def _convert_targets(targets: Any, target: Any) -> tuple[str, ...]:
    if target is not None:
        if targets:
            raise InvalidArgument(
                f"Passing both targets={targets!r} and target={target!r} is redundant - "
                f"pass targets={(*targets, target)!r} instead."
            )
        targets = (target,)
    converted: list[str] = []
    for t in targets:
        if not isinstance(t, Bundle):
            msg = (
                f"Got invalid target {t!r} of type {type(t)!r}, but all targets must "
                "be Bundles."
            )
            if isinstance(t, _NativeSearchStrategy):
                msg += (
                    "\nIt looks like you passed `one_of(a, b)` or `a | b` as a target. "
                    "You should instead pass `targets=(a, b)` to add the return value of "
                    "this rule to both the `a` and `b` bundles, or define a rule for each "
                    "target if it should be added to exactly one."
                )
            raise InvalidArgument(msg)
        if isinstance(t, BundleConsumer):
            from hypothesis.utils.deprecation import note_deprecation

            note_deprecation(
                f"Using consumes({t.name}) doesn't makes sense in this context.  "
                "This will be an error in a future version of Hypothesis.",
                since="2021-09-08",
                has_codemod=False,
                stacklevel=2,
            )
        converted.append(t.name)
    return tuple(converted)


# ----------------------------------------------------------------------------- #
# Decorators
# ----------------------------------------------------------------------------- #


def _proxy(f: Any) -> Any:
    """A thin functools.wraps-style proxy so the wrapper looks like the rule fn."""
    import functools

    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return f(*args, **kwargs)

    return wrapper


def rule(*, targets: Any = (), target: Any = None, **kwargs: Any) -> Any:
    """Decorator marking a method as a state-machine rule. `target(s)` name the
    bundle(s) the return value is added to; kwargs are argument strategies (or a
    Bundle / consumes(bundle) to draw a previously-produced value)."""
    converted_targets = _convert_targets(targets, target)
    for k, v in kwargs.items():
        if not isinstance(v, (Bundle, _NativeSearchStrategy)):
            raise InvalidArgument(f"Expected a SearchStrategy but got {k}={v!r}")

    def accept(f: Any) -> Any:
        if getattr(f, INVARIANT_MARKER, None):
            raise InvalidDefinition(
                f"{_rule_qualname(f)} is used with both @rule and @invariant, "
                "which is not allowed."
            )
        if getattr(f, RULE_MARKER, None) is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with @rule twice, "
                "which is not allowed."
            )
        if getattr(f, INITIALIZE_RULE_MARKER, None) is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @rule and "
                "@initialize, which is not allowed."
            )
        preconditions = getattr(f, PRECONDITIONS_MARKER, ())
        r = Rule(
            targets=converted_targets,
            arguments=kwargs,
            function=f,
            preconditions=preconditions,
        )
        wrapper = _proxy(f)
        setattr(wrapper, RULE_MARKER, r)
        return wrapper

    return accept


def initialize(*, targets: Any = (), target: Any = None, **kwargs: Any) -> Any:
    """Like @rule, but runs once per run before any @rule, in arbitrary order, and
    may not have a @precondition."""
    converted_targets = _convert_targets(targets, target)
    for k, v in kwargs.items():
        if not isinstance(v, (Bundle, _NativeSearchStrategy)):
            raise InvalidArgument(f"Expected a SearchStrategy but got {k}={v!r}")

    def accept(f: Any) -> Any:
        if getattr(f, INVARIANT_MARKER, None):
            raise InvalidDefinition(
                f"{_rule_qualname(f)} is used with both @initialize and @invariant, "
                "which is not allowed."
            )
        if getattr(f, RULE_MARKER, None) is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @rule and "
                "@initialize, which is not allowed."
            )
        if getattr(f, INITIALIZE_RULE_MARKER, None) is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with @initialize twice, "
                "which is not allowed."
            )
        if getattr(f, PRECONDITIONS_MARKER, ()):
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @initialize and "
                "@precondition, which is not allowed."
            )
        r = Rule(
            targets=converted_targets,
            arguments=kwargs,
            function=f,
            preconditions=(),
        )
        wrapper = _proxy(f)
        setattr(wrapper, INITIALIZE_RULE_MARKER, r)
        return wrapper

    return accept


def precondition(precond: Any) -> Any:
    """Add a precondition predicate `precond(self) -> bool` to a @rule or @invariant.
    The rule/invariant is only a valid step when all its preconditions return True."""

    def decorator(f: Any) -> Any:
        wrapper = _proxy(f)
        if getattr(f, INITIALIZE_RULE_MARKER, None) is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @initialize and "
                "@precondition, which is not allowed."
            )
        r = getattr(f, RULE_MARKER, None)
        invar = getattr(f, INVARIANT_MARKER, None)
        if r is not None:
            new_rule = dataclasses.replace(
                r, preconditions=(*r.preconditions, precond)
            )
            setattr(wrapper, RULE_MARKER, new_rule)
        elif invar is not None:
            new_invar = dataclasses.replace(
                invar, preconditions=(*invar.preconditions, precond)
            )
            setattr(wrapper, INVARIANT_MARKER, new_invar)
        else:
            setattr(
                wrapper,
                PRECONDITIONS_MARKER,
                (*getattr(f, PRECONDITIONS_MARKER, ()), precond),
            )
        return wrapper

    return decorator


def invariant(*, check_during_init: bool = False) -> Any:
    """Decorator marking a method run after every rule (and after initialization).
    May raise to indicate a failed invariant. With check_during_init=True it also
    runs during the initialization phase."""
    if not isinstance(check_during_init, bool):
        raise InvalidArgument("check_during_init must be a bool")

    def accept(f: Any) -> Any:
        if getattr(f, RULE_MARKER, None) or getattr(f, INITIALIZE_RULE_MARKER, None):
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with both @invariant and "
                "@rule, which is not allowed."
            )
        if getattr(f, INVARIANT_MARKER, None) is not None:
            raise InvalidDefinition(
                f"{_rule_qualname(f)} has been decorated with @invariant twice, "
                "which is not allowed."
            )
        preconditions = getattr(f, PRECONDITIONS_MARKER, ())
        invar = Invariant(
            function=f,
            preconditions=preconditions,
            check_during_init=check_during_init,
        )
        wrapper = _proxy(f)
        setattr(wrapper, INVARIANT_MARKER, invar)
        return wrapper

    return accept


# ----------------------------------------------------------------------------- #
# RuleBasedStateMachine
# ----------------------------------------------------------------------------- #


class StateMachineMeta(type):
    def __setattr__(cls, name: str, value: Any) -> None:
        if name == "settings" and _is_settings(value):
            raise AttributeError(
                f"Assigning {cls.__name__}.settings = ... does nothing. Assign to "
                f"{cls.__name__}.TestCase.settings, or use @settings(...) as a decorator."
            )
        super().__setattr__(name, value)


class TestCaseProperty:
    def __get__(self, obj: Any, typ: Any = None) -> Any:
        if obj is not None:
            typ = type(obj)
        return typ._to_test_case()

    def __set__(self, obj: Any, value: Any) -> None:
        raise AttributeError("Cannot set TestCase")

    def __delete__(self, obj: Any) -> None:
        raise AttributeError("Cannot delete TestCase")


def _rule_sort_key(r: Rule) -> Any:
    return (sorted(r.targets), len(r.arguments), r.function.__name__)


class RuleBasedStateMachine(metaclass=StateMachineMeta):
    """Base class for stateful tests: subclass it, define @rule/@initialize/
    @invariant methods (and optional Bundles), and run via its `.TestCase` or
    `run_state_machine_as_test`."""

    _setup_state_per_class: ClassVar[dict[type, _SetupState]] = {}

    def __init__(self) -> None:
        setup_state = self.setup_state()
        if not setup_state.rules:
            raise InvalidDefinition(
                f"State machine {type(self).__name__} defines no rules"
            )
        if _is_settings(vars(type(self)).get("settings")):
            tname = type(self).__name__
            raise InvalidDefinition(
                f"Assigning settings = ... as a class attribute does nothing. Assign "
                f"to {tname}.TestCase.settings, or use @settings(...) as a decorator "
                f"on the {tname} class."
            )

        self.rules = setup_state.rules
        self.invariants = setup_state.invariants
        self._initialize_rules_to_run = list(setup_state.initializers)
        self._sorted_rules = sorted(self.rules, key=_rule_sort_key)

        self.bundles: dict[str, list[VarReference]] = {}
        self.names_counters: collections.Counter = collections.Counter()
        self.names_list: list[str] = []
        self.names_to_values: dict[str, Any] = {}
        self.__stream = io.StringIO()
        self.__printer: Any = None

    def _printer(self) -> Any:
        # Lazily build the RepresentationPrinter so a machine instantiated outside a run
        # (M() in a test) doesn't need a build context. singleton_pprinters (registered in
        # _add_results_to_targets) make a bundle value render as its variable name.
        if self.__printer is None and _RepresentationPrinter is not None:
            ctx = None
            if _current_build_context is not None:
                try:
                    ctx = _current_build_context()
                except Exception:  # noqa: BLE001 - no build context outside a run
                    ctx = None
            self.__printer = _RepresentationPrinter(self.__stream, context=ctx)
        return self.__printer

    def _pretty_print(self, value: Any) -> str:
        if isinstance(value, VarReference):
            return value.name
        if isinstance(value, list) and value and all(
            isinstance(item, VarReference) for item in value
        ):
            return "[" + ", ".join(item.name for item in value) + "]"
        printer = self._printer()
        if printer is None:
            return repr(value)
        self.__stream.seek(0)
        self.__stream.truncate(0)
        printer.output_width = 0
        printer.buffer_width = 0
        printer.buffer.clear()
        printer.pretty(value)
        printer.flush()
        return self.__stream.getvalue()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.bundles!r})"

    def _new_name(self, target: str) -> str:
        result = f"{target}_{self.names_counters[target]}"
        self.names_counters[target] += 1
        self.names_list.append(result)
        return result

    def _last_names(self, n: int) -> list[str]:
        return self.names_list[len(self.names_list) - n :]

    def bundle(self, name: str) -> list[VarReference]:
        return self.bundles.setdefault(name, [])

    @classmethod
    def setup_state(cls) -> _SetupState:
        try:
            return cls._setup_state_per_class[cls]
        except KeyError:
            pass
        rules: list[Rule] = []
        initializers: list[Rule] = []
        invariants: list[Invariant] = []
        for _name, f in inspect.getmembers(cls):
            r = getattr(f, RULE_MARKER, None)
            initr = getattr(f, INITIALIZE_RULE_MARKER, None)
            invar = getattr(f, INVARIANT_MARKER, None)
            if r is not None:
                rules.append(r)
            if initr is not None:
                initializers.append(initr)
            if invar is not None:
                invariants.append(invar)
            if (
                getattr(f, PRECONDITIONS_MARKER, None) is not None
                and r is None
                and invar is None
            ):
                raise InvalidDefinition(
                    f"{_rule_qualname(f)} has been decorated with @precondition, but "
                    "not @rule (or @invariant), which is not allowed."
                )
        state = _SetupState(rules=rules, initializers=initializers, invariants=invariants)
        cls._setup_state_per_class[cls] = state
        return state

    def _is_valid(self, rule: Rule) -> bool:
        for b in rule.bundles:
            if not self.bundle(b.name):
                return False
        for pred in rule.preconditions:
            if not pred(self):
                return False
        return True

    def _add_results_to_targets(self, targets: tuple[str, ...], results: Any) -> None:
        for target in targets:
            for result in results:
                name = self._new_name(target)

                def _printer(obj: Any, p: Any, cycle: Any, name: str = name) -> Any:
                    return p.text(name)

                # Register a name-printer keyed on the value's id, so when this value later
                # appears as a rule argument it prints as its variable name. Skip interned
                # singletons (small ints / None / bool) where id collisions would misname
                # unrelated values.
                printer = self._printer()
                if printer is not None and not _is_singleton(result):
                    printer.singleton_pprinters.setdefault(id(result), _printer)
                self.names_to_values[name] = result
                self.bundles.setdefault(target, []).append(VarReference(name))

    def _repr_step(self, rule: Rule, data: dict[str, str], result: Any) -> str:
        output_assignment = ""
        extra_lines: list[str] = []
        if rule.targets:
            n_results = len(result.values) if isinstance(result, MultipleResults) else 1
            last_names = self._last_names(len(rule.targets) * n_results)
            if isinstance(result, MultipleResults):
                if len(result.values) == 1:
                    output_assignment = (
                        " = ".join(f"({name},)" for name in last_names) + " = "
                    )
                elif result.values:
                    per_target = [
                        last_names[i : i + n_results]
                        for i in range(0, len(last_names), n_results)
                    ]
                    first = ", ".join(per_target[0])
                    output_assignment = first + " = "
                    for other in per_target[1:]:
                        extra_lines.append(", ".join(other) + " = " + first)
            else:
                output_assignment = " = ".join(last_names) + " = "
        args = ", ".join(f"{k}={v}" for k, v in data.items())
        line = f"{output_assignment}state.{rule.function.__name__}({args})"
        return "\n".join([line] + extra_lines)

    def check_invariants(self, settings: Any, output: Any) -> None:
        for invar in self.invariants:
            if self._initialize_rules_to_run and not invar.check_during_init:
                continue
            if not all(precond(self) for precond in invar.preconditions):
                continue
            name = invar.function.__name__
            if _is_final() or _verbosity_at_least(Verbosity.debug):
                output(f"state.{name}()")
            result = invar.function(self)
            if result is not None:
                _fail_health_check(
                    settings,
                    "The return value of an @invariant is always ignored, but "
                    f"{invar.function.__qualname__} returned {result!r} instead of None",
                )

    def teardown(self) -> None:
        """Called once after a run finishes (even on failure) to clean up. No-op by default."""

    TestCase = TestCaseProperty()

    @classmethod
    @lru_cache
    def _to_test_case(cls) -> type:
        machine_cls = cls

        class StateMachineTestCase(TestCase):
            # A class-level `@settings(...)` on the machine wins; else the permissive default.
            settings = getattr(
                cls, "_hypothesis_internal_use_settings", None
            ) or Settings(deadline=None, suppress_health_check=list(HealthCheck))

            def runTest(self) -> None:
                run_state_machine_as_test(machine_cls, settings=self.settings)

            runTest.is_hypothesis_test = True  # type: ignore[attr-defined]

        StateMachineTestCase.__name__ = cls.__name__ + ".TestCase"
        StateMachineTestCase.__qualname__ = cls.__qualname__ + ".TestCase"
        return StateMachineTestCase


def _fail_health_check(settings: Any, message: str) -> None:
    suppressed = getattr(settings, "suppress_health_check", ()) or ()
    if HealthCheck.return_value in suppressed:
        return
    raise FailedHealthCheck(
        message + "\nSee https://hypothesis.readthedocs.io/en/latest/reference/api.html"
        "#hypothesis.HealthCheck for more information about this. "
        "If you want to disable just this health check, add HealthCheck.return_value "
        "to the suppress_health_check settings for this test."
    )


# ----------------------------------------------------------------------------- #
# The runner
# ----------------------------------------------------------------------------- #


def _draw_args(machine: RuleBasedStateMachine, cd: Any, rule: Rule) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for k, strat in rule.arguments_strategies.items():
        # Every arg is a native strategy now: a BundleReferenceStrategy (top-level bundle
        # arg) draws a reference, a Bundle nested in a collection draws values, everything
        # else draws normally. Bundle draws reach the machine via cd._hf_stateful_machine.
        try:
            data[k] = cd.draw(strat)
        except Exception as err:  # noqa: BLE001 - annotate then re-raise
            _add_note(
                err,
                f"while generating {k!r} from {strat!r} for rule "
                f"{rule.function.__name__}",
            )
            raise
    return data


# The step loop is split into these small helpers so BOTH the Python driver (_run_body)
# and the Rust driver (StatefulRunner.__call__, P3) execute the exact same per-step logic.


def _emit(printed: list[str], print_steps: bool, s: str) -> None:
    """Record a step-program line + (when enabled) report it live. The Rust loop binds this
    with functools.partial(printed, print_steps) to get the same `output` callable the
    Python loop builds as a closure."""
    if print_steps:
        printed.append(s)
        if _report is not None:
            _report(s)


def _must_stop(steps_run: int, min_steps: int, max_steps: int, cd_length: int) -> Any:
    """The `forced=` value for the per-step stop-boolean: True once we've taken enough
    steps (or the buffer is nearly full), False while below min_steps, else None (free)."""
    if steps_run >= max_steps:
        return True
    if steps_run <= min_steps:
        return False
    if cd_length > (0.8 * BUFFER_SIZE):
        return True
    return None


def _select_rule(machine: RuleBasedStateMachine, cd: Any, flaky_state: Any) -> Rule:
    """Pick the next rule to run: an outstanding @initialize first, else a valid @rule.
    Marks flaky_state['selecting_rule'] so a divergence here is classified as a flaky
    precondition (see run_state_machine_as_test)."""
    if machine._initialize_rules_to_run:
        rule = cd.draw(st.sampled_from(machine._initialize_rules_to_run))
        machine._initialize_rules_to_run.remove(rule)
        return rule
    flaky_state["selecting_rule"] = True
    if not any(machine._is_valid(r) for r in machine._sorted_rules):
        # On the final (is_final) replay, "no valid rule" where generation made progress
        # means a precondition flipped between runs — a flaky DEFINITION, not a stuck
        # machine. selecting_rule stays True so the caller adds the precondition note.
        if _is_final():
            raise FlakyStrategyDefinition(
                "Inconsistent rule availability between generation and replay"
            )
        names = ", ".join(r.function.__name__ for r in machine._sorted_rules)
        raise InvalidDefinition(
            f"No progress can be made from state {machine!r}, because no available rule "
            f"had a True precondition. rules: {names}"
        )
    rule = cd.draw(st.sampled_from(machine._sorted_rules).filter(machine._is_valid))
    flaky_state["selecting_rule"] = False
    return rule


def _run_step(
    machine: RuleBasedStateMachine,
    cd: Any,
    rule: Rule,
    settings: Any,
    output: Any,
    print_steps: bool,
) -> None:
    """Draw a rule's args, run it, route its result to target bundle(s), and print the
    step. Pretty-printing happens BEFORE reference resolution so an argument that is also a
    return value prints with its original variable name (upstream #2341)."""
    argdata = _draw_args(machine, cd, rule)
    data_to_print = (
        {k: machine._pretty_print(v) for k, v in argdata.items()} if print_steps else {}
    )
    result: Any = multiple()
    try:
        resolved = dict(argdata)
        for k, v in list(resolved.items()):
            if isinstance(v, VarReference):
                resolved[k] = machine.names_to_values[v.name]
            elif isinstance(v, list) and v and all(
                isinstance(item, VarReference) for item in v
            ):
                resolved[k] = [machine.names_to_values[item.name] for item in v]
        result = rule.function(machine, **resolved)
        if rule.targets:
            if isinstance(result, MultipleResults):
                machine._add_results_to_targets(rule.targets, result.values)
            else:
                machine._add_results_to_targets(rule.targets, [result])
        elif result is not None:
            _fail_health_check(
                settings,
                "Rules should return None if they have no target bundle, but "
                f"{rule.function.__qualname__} returned {result!r}",
            )
    finally:
        if print_steps:
            output(machine._repr_step(rule, data_to_print, result))


def get_state_machine_test(
    state_machine_factory: Any,
    *,
    settings: Any = None,
    _min_steps: int = 0,
    _flaky_state: Any = None,
) -> Any:
    # Shared with run_state_machine_as_test: records whether a FlakyStrategyDefinition was
    # raised while SELECTING a rule (→ flaky precondition) vs inside a rule body.
    if _flaky_state is None:
        _flaky_state = {"selecting_rule": False}
    if settings is None:
        # A class-level `@settings(...)` on the machine (e.g. stateful_step_count=5) wins
        # over the TestCase default.
        settings = getattr(
            state_machine_factory, "_hypothesis_internal_use_settings", None
        )
    if settings is None:
        try:
            settings = state_machine_factory.TestCase.settings
        except AttributeError:
            settings = Settings(deadline=None, suppress_health_check=list(HealthCheck))
    if not _is_settings(settings):
        raise InvalidArgument(f"settings={settings!r} must be a Settings instance")
    settings = _coerce_settings(settings)
    if not isinstance(_min_steps, int) or isinstance(_min_steps, bool) or _min_steps < 0:
        raise InvalidArgument(
            f"_min_steps={_min_steps!r} must be a non-negative integer."
        )

    def _run_body(data: Any) -> None:
        cd = data.conjecture_data
        machine = state_machine_factory()
        if not isinstance(machine, RuleBasedStateMachine):
            raise InvalidArgument(
                f"state_machine_factory() must return a RuleBasedStateMachine, got "
                f"{machine!r}"
            )
        # Stash the machine on the cd so bundle draws — including ones nested deep inside
        # native collections (lists(consumes(b))) — can reach it. Per-cd, so a nested run
        # gets its own cd and can't see this machine.
        cd._hf_stateful_machine = machine

        print_steps = _is_final() or _verbosity_at_least(Verbosity.debug)
        # Collect the reproducing program (state = M(); v1 = state.rule(...); teardown())
        # so it can be attached to the failing exception as notes — that's what carries
        # the trace to the user's test (upstream prints it via report() during the final
        # replay; we mirror that AND attach to __notes__). Only populated when print_steps.
        printed: list[str] = []

        def output(s: str) -> None:
            if print_steps:
                printed.append(s)
                if _report is not None:
                    _report(s)

        try:
            output(f"state = {machine.__class__.__name__}()")
            machine.check_invariants(settings, output)
            max_steps = settings.stateful_step_count
            steps_run = 0

            while True:
                must_stop = _must_stop(steps_run, _min_steps, max_steps, cd.length)
                if cd.draw_boolean(p=2**-16, forced=must_stop):
                    break
                steps_run += 1
                rule = _select_rule(machine, cd, _flaky_state)
                _run_step(machine, cd, rule, settings, output, print_steps)
                machine.check_invariants(settings, output)
        finally:
            output("state.teardown()")
            machine.teardown()
            # On the final (minimal) replay the body re-runs with is_final True; attach the
            # collected program (incl. this teardown line) as notes on the in-flight failure
            # so the falsifying example shows the reproducing steps.
            _inflight = sys.exc_info()[1]
            if _inflight is not None and _is_final():
                # Upstream prints a "Falsifying example:" header before the program; the
                # printing tests index notes from it (notes[0]).
                _add_note(_inflight, "Falsifying example:")
                for line in printed:
                    _add_note(_inflight, line)

    # Set the print-suppression flag on the BODY (what @given wraps and core reads at the
    # falsifying-example site) so core skips the default `run_state_machine(data=...)` note
    # and instead lets our trace-via-notes through. Then wrap with given()+settings().
    _run_body._hypothesis_internal_print_given_args = False  # type: ignore[attr-defined]

    if _USE_RUST_STATEFUL:
        # The native step loop (src/stateful.rs): drives the SAME per-step helpers as
        # _run_body above (kept for reference / differential testing), so behaviour is
        # identical — only the loop control runs in Rust. A thin Python wrapper keeps a
        # normal signature for given()'s introspection.
        from ._engine import StatefulRunner

        _native_runner = StatefulRunner(
            state_machine_factory, settings, _min_steps, _flaky_state
        )

        def _run_body(data: Any) -> None:  # noqa: F811 - swap in the native driver
            _native_runner(data)

        _run_body._hypothesis_internal_print_given_args = False  # type: ignore[attr-defined]

    run_state_machine = settings(given(st.data())(_run_body))

    # @seed / @reproduce_failure "just work" by copying the markers across.
    run_state_machine._hypothesis_internal_use_seed = getattr(  # type: ignore[attr-defined]
        state_machine_factory, "_hypothesis_internal_use_seed", None
    )
    run_state_machine._hypothesis_internal_use_reproduce_failure = getattr(  # type: ignore[attr-defined]
        state_machine_factory, "_hypothesis_internal_use_reproduce_failure", None
    )
    run_state_machine._hypothesis_internal_print_given_args = False  # type: ignore[attr-defined]
    return run_state_machine


def run_state_machine_as_test(
    state_machine_factory: Any, *, settings: Any = None, _min_steps: int = 0
) -> None:
    """Run a state machine definition as a test: silently pass, or raise with a
    minimal failing program. `state_machine_factory` is anything returning a
    RuleBasedStateMachine when called with no arguments (a class or function)."""
    flaky_state = {"selecting_rule": False}
    test = get_state_machine_test(
        state_machine_factory,
        settings=settings,
        _min_steps=_min_steps,
        _flaky_state=flaky_state,
    )
    try:
        test()
    except FlakyStrategyDefinition as err:
        if flaky_state["selecting_rule"]:
            _add_note(
                err,
                "while selecting a rule to run. This is usually caused by a flaky "
                "precondition, or a bundle that was unexpectedly empty.",
            )
        raise


__all__ = [
    "Bundle",
    "BundleConsumer",
    "Invariant",
    "MultipleResults",
    "RuleBasedStateMachine",
    "Rule",
    "VarReference",
    "consumes",
    "get_state_machine_test",
    "initialize",
    "invariant",
    "multiple",
    "precondition",
    "rule",
    "run_state_machine_as_test",
]
