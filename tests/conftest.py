"""Top-level test config.

The hypothesis_compat/ suite aliases sys.modules['hypothesis'] to our package
(global, process-wide), which would contaminate the native package tests
(tests/test_*.py) if collected in the same session. So a bare `pytest tests/`
skips it; it's run on its own via `make test` (which targets the directory
explicitly), with its own conftest doing the aliasing.
"""

collect_ignore = ["hypothesis_compat"]
