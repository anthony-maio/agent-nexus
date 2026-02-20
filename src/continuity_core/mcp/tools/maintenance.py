from __future__ import annotations

from typing import Any, Dict

from continuity_core.mcp.tools.introspect import _normalize_graph
from continuity_core.services.night_cycle import NightCycle
from continuity_core.services.runtime import get_memory_system


def maintenance(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Run a Night Cycle maintenance pass.

    Performs decay, pruning, stress recomputation, void scanning,
    and harmonic integration tracking in one structured pass.
    """
    mem = get_memory_system()
    graph_raw = arguments.get("graph") or {}
    graph = _normalize_graph(graph_raw) if graph_raw else None

    cycle = NightCycle(
        memory_system=mem,
        prune_threshold=float(arguments.get("prune_threshold", 0.05)),
        statement_window=int(arguments.get("statement_window", 30)),
    )
    result = cycle.run(graph=graph)

    return {
        "decay_applied": result.decay_applied,
        "items_pruned": result.items_pruned,
        "stress_before": result.stress_before,
        "stress_after": result.stress_after,
        "stress_delta": result.stress_after - result.stress_before,
        "contradictions_found": result.contradictions_found,
        "deep_tensions_found": result.deep_tensions_found,
        "voids_found": result.voids_found,
        "resolutions": result.resolutions,
        "harmonic_reward": result.harmonic_reward,
        "duration_sec": result.duration_sec,
    }
