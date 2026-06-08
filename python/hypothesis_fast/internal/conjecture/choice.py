"""Typed choice-sequence model.

All the math is Rust (`hypothesis_fast._engine`): `choice_to_index`,
`choice_from_index`, `choice_permitted`, `choice_key`, `choices_key`,
`choice_equal`, `zigzag_*`. This module keeps only the thin Python containers the
test suite constructs directly — `ChoiceNode`, `ChoiceTemplate`, the constraint
TypedDicts, and the constraints-key helpers (which key/compare constraint dicts,
not a per-draw path).
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict, cast

from hypothesis_fast._engine import (
    choice_equal,
    choice_from_index,
    choice_key,
    choice_permitted,
    choice_to_index,
    choices_key,
    zigzag_index,
    zigzag_value,
)
from hypothesis_fast.internal.floats import float_to_int
from hypothesis_fast.internal.intervalsets import IntervalSet

__all__ = [
    "ChoiceNode",
    "ChoiceTemplate",
    "IntegerConstraints",
    "FloatConstraints",
    "StringConstraints",
    "BytesConstraints",
    "BooleanConstraints",
    "ChoiceT",
    "ChoiceConstraintsT",
    "ChoiceTypeT",
    "ChoiceKeyT",
    "choice_to_index",
    "choice_from_index",
    "choice_permitted",
    "choice_key",
    "choices_key",
    "choice_equal",
    "choice_constraints_key",
    "choice_constraints_equal",
    "zigzag_index",
    "zigzag_value",
    "choices_size",
]


class IntegerConstraints(TypedDict):
    min_value: int | None
    max_value: int | None
    weights: dict[int, float] | None
    shrink_towards: int


class FloatConstraints(TypedDict):
    min_value: float
    max_value: float
    allow_nan: bool
    smallest_nonzero_magnitude: float


class StringConstraints(TypedDict):
    intervals: IntervalSet
    min_size: int
    max_size: int


class BytesConstraints(TypedDict):
    min_size: int
    max_size: int


class BooleanConstraints(TypedDict):
    p: float


ChoiceT: TypeAlias = int | str | bool | float | bytes
ChoiceConstraintsT: TypeAlias = (
    IntegerConstraints
    | FloatConstraints
    | StringConstraints
    | BytesConstraints
    | BooleanConstraints
)
ChoiceTypeT: TypeAlias = Literal["integer", "string", "boolean", "float", "bytes"]
ChoiceKeyT: TypeAlias = (
    int | str | bytes | tuple[Literal["bool"], bool] | tuple[Literal["float"], int]
)


@dataclass(slots=True, frozen=False)
class ChoiceTemplate:
    type: Literal["simplest"]
    count: int | None

    def __post_init__(self) -> None:
        if self.count is not None:
            assert self.count > 0


@dataclass(slots=True, frozen=False)
class ChoiceNode:
    type: ChoiceTypeT
    value: ChoiceT
    constraints: ChoiceConstraintsT
    was_forced: bool
    index: int | None = None

    def copy(
        self,
        *,
        with_value: ChoiceT | None = None,
        with_constraints: ChoiceConstraintsT | None = None,
    ) -> "ChoiceNode":
        if self.was_forced:
            assert with_value is None, "modifying a forced node doesn't make sense"
        return ChoiceNode(
            type=self.type,
            value=self.value if with_value is None else with_value,
            constraints=(
                self.constraints if with_constraints is None else with_constraints
            ),
            was_forced=self.was_forced,
        )

    @property
    def trivial(self) -> bool:
        if self.was_forced:
            return True

        if self.type != "float":
            zero_value = choice_from_index(0, self.type, self.constraints)
            return choice_equal(self.value, zero_value)
        else:
            constraints = cast(FloatConstraints, self.constraints)
            min_value = constraints["min_value"]
            max_value = constraints["max_value"]
            shrink_towards = 0.0

            if min_value == -math.inf and max_value == math.inf:
                return choice_equal(self.value, shrink_towards)

            if (
                not math.isinf(min_value)
                and not math.isinf(max_value)
                and math.ceil(min_value) <= math.floor(max_value)
            ):
                shrink_towards = max(math.ceil(min_value), shrink_towards)
                shrink_towards = min(math.floor(max_value), shrink_towards)
                return choice_equal(self.value, float(shrink_towards))

            return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChoiceNode):
            return NotImplemented

        return (
            self.type == other.type
            and choice_equal(self.value, other.value)
            and choice_constraints_equal(self.type, self.constraints, other.constraints)
            and self.was_forced == other.was_forced
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.type,
                choice_key(self.value),
                choice_constraints_key(self.type, self.constraints),
                self.was_forced,
            )
        )

    def __repr__(self) -> str:
        forced_marker = " [forced]" if self.was_forced else ""
        return f"{self.type} {self.value!r}{forced_marker} {self.constraints!r}"


def choice_constraints_equal(
    choice_type: ChoiceTypeT,
    constraints1: ChoiceConstraintsT,
    constraints2: ChoiceConstraintsT,
) -> bool:
    return choice_constraints_key(choice_type, constraints1) == choice_constraints_key(
        choice_type, constraints2
    )


def choice_constraints_key(
    choice_type: ChoiceTypeT, constraints: ChoiceConstraintsT
) -> tuple[Hashable, ...]:
    if choice_type == "float":
        constraints = cast(FloatConstraints, constraints)
        return (
            float_to_int(constraints["min_value"]),
            float_to_int(constraints["max_value"]),
            constraints["allow_nan"],
            constraints["smallest_nonzero_magnitude"],
        )
    if choice_type == "integer":
        constraints = cast(IntegerConstraints, constraints)
        return (
            constraints["min_value"],
            constraints["max_value"],
            None if constraints["weights"] is None else tuple(constraints["weights"]),
            constraints["shrink_towards"],
        )
    return tuple(constraints[key] for key in sorted(constraints))  # type: ignore


def choices_size(choices: Iterable[ChoiceT]) -> int:
    from hypothesis_fast.database import choices_to_bytes

    return len(choices_to_bytes(choices))
