from __future__ import annotations

from typing import Any, Dict

from continuity_core.mra import EpistemicStressMonitor, VoidDetector
from continuity_core.services.runtime import get_memory_system


def _normalize_graph(raw: Dict[str, Any]) -> Dict[str, set]:
    graph: Dict[str, set] = {}
    for node, neighbors in raw.items():
        graph[str(node)] = {str(n) for n in (neighbors or [])}
    return graph


def _graph_sparsity(graph: Dict[str, set]) -> float:
    """Compute 1 - density of the graph.  Returns 1.0 (maximally sparse) when empty."""
    nodes = set(graph.keys())
    for neighbors in graph.values():
        nodes.update(neighbors)
    n = len(nodes)
    if n < 2:
        return 1.0
    max_edges = n * (n - 1)  # directed
    actual_edges = sum(len(nbrs) for nbrs in graph.values())
    density = actual_edges / max_edges
    return 1.0 - density


def introspect(arguments: Dict[str, Any]) -> Dict[str, Any]:
    statements = arguments.get("statements") or []
    concept_contexts = arguments.get("concept_contexts") or {}
    graph_raw = arguments.get("graph") or {}

    mem = get_memory_system()
    graph = _normalize_graph(graph_raw)
    sparsity = _graph_sparsity(graph) if graph else 1.0

    monitor = EpistemicStressMonitor(
        embed_fn=mem.embedder.embed,
    )
    stress = monitor.compute(statements, concept_contexts=concept_contexts, graph_sparsity=sparsity)

    void_detector = VoidDetector()
    voids = void_detector.detect_voids(graph) if graph else None

    # Update the MRA cache so context pipeline can use these signals.
    mem.update_mra_cache(stress, voids)

    return {
        "stress": {
            "s_omega": stress.s_omega,
            "d_log": stress.d_log,
            "d_sem": stress.d_sem,
            "v_top": stress.v_top,
            "components": stress.components,
            "should_trigger": stress.should_trigger,
            "contradictions": stress.contradictions,
            "deep_tensions": [
                {"s1": s1, "s2": s2, "score": sc, "similarity": sim}
                for s1, s2, sc, sim in stress.deep_tensions
            ],
        },
        "voids": {
            "pairs": [
                {"a": list(a), "b": list(b), "edges_between": edges}
                for a, b, edges in (voids.void_pairs if voids else [])
            ],
            "questions": (voids.questions if voids else []),
        },
    }
