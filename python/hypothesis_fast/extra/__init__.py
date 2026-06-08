"""``hypothesis_fast.extra`` — optional frontends that depend on third-party
packages (numpy, ...), ported to run on the native engine.

These mirror ``hypothesis.extra.*`` but are built on the native
``hypothesis_fast`` strategies, so user-supplied ``elements``/``fill``
strategies compose natively (no real-hypothesis ``ConjectureData`` interop
boundary). Import a submodule explicitly, e.g. ``hypothesis_fast.extra.numpy``;
nothing is imported eagerly here because the third-party deps may be absent.
"""
