"""Vendored helpers from upstream `tests/nocover/test_stateful.py` that the cover
`test_stateful.py` imports (`from tests.nocover.test_stateful import DepthMachine`).
Loaded under that dotted name by conftest's `_load`, AFTER `hypothesis.stateful` is
routed to our native reimplementation — so these run on the native engine."""

from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule


class DepthCharge:
    def __init__(self, value):
        if value is None:
            self.depth = 0
        else:
            self.depth = value.depth + 1


class DepthMachine(RuleBasedStateMachine):
    charges = Bundle("charges")

    @rule(targets=(charges,), child=charges)
    def charge(self, child):
        return DepthCharge(child)

    @rule(targets=(charges,))
    def none_charge(self):
        return DepthCharge(None)

    @rule(check=charges)
    def is_not_too_deep(self, check):
        assert check.depth < 3
