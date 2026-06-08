"""ConjectureRunner engine — minimal for now.

Phase 4 builds the real generate/reuse/shrink/target loop here. For now this only
provides the few constants that lower layers (choice.py) reference.
"""

from __future__ import annotations

# Upstream's conjecture byte-buffer size; used as a sanity cap on collection sizes
# in choice.collection_value (a size that would exceed it is treated as overrun).
BUFFER_SIZE: int = 8 * 1024
