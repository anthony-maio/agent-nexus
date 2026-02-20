from __future__ import annotations

from continuity_core.memory.system import TieredMemorySystem


_MEMORY_SYSTEM: TieredMemorySystem | None = None


def get_memory_system() -> TieredMemorySystem:
    global _MEMORY_SYSTEM
    if _MEMORY_SYSTEM is None:
        _MEMORY_SYSTEM = TieredMemorySystem()
    return _MEMORY_SYSTEM
