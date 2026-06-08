"""Scope numpy ``from_type`` registration to the numpy parity tests only.

``hypothesis_fast.extra.numpy`` resolves numpy types for ``st.from_type`` by registering
~24 hashable numpy scalar types into the *shared* native type registry. Done globally (at
import, in the forkserver) that registration leaks into every worker and structurally
changes ``from_type(collections.abc.Hashable)`` resolution for the rest of the parity suite
— e.g. ``test_lookup`` could no longer find an unhashable ``Decimal``. This autouse fixture
registers the numpy types for the duration of each numpy test and restores the registry
afterwards, so the registration stays local to the tests that actually need it.
"""

import pytest


@pytest.fixture(autouse=True)
def _scoped_numpy_type_registry():
    from hypothesis_fast import native_strategies as _ns
    from hypothesis_fast.extra import numpy as _np

    registry_snapshot = dict(_ns._NATIVE_TYPE_REGISTRY)
    user_snapshot = set(_ns._USER_REGISTERED)
    _np._register_from_type()
    try:
        yield
    finally:
        _np._unregister_from_type(registry_snapshot, user_snapshot)
