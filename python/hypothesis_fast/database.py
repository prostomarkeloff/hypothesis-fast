"""Choice-sequence (de)serialization — re-exported from the Rust engine.

`choices_to_bytes` / `choices_from_bytes` implement hypothesis's flat custom
format (used by the database + reproduce_failure subsystems). The ExampleDatabase
classes (in-memory / directory) are still to be ported.
"""

from __future__ import annotations

from hypothesis_fast._engine import choices_from_bytes, choices_to_bytes

__all__ = ["choices_to_bytes", "choices_from_bytes"]
