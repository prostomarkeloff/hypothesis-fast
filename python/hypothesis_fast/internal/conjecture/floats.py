"""Lexical float encoding — re-exported from the Rust engine.

`float_to_lex` / `lex_to_float` / `is_simple` are implemented in Rust
(`hypothesis_fast._engine`); they define the shrink-friendly encoding of
non-negative floats used by `choice_to_index`.
"""

from __future__ import annotations

from hypothesis_fast._engine import float_to_lex, is_simple, lex_to_float

__all__ = ["float_to_lex", "lex_to_float", "is_simple"]
