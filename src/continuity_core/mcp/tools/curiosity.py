from __future__ import annotations

from typing import Any, Dict, List

from continuity_core.mra import EpistemicStressMonitor, VoidDetector
from continuity_core.services.runtime import get_memory_system


def curiosity(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Return prioritized epistemic tensions and bridging questions.

    Pull-based complement to the push-based MRA injection in the context
    gatherer.  If the MRA cache is stale the tool runs a fresh introspect
    cycle from recent event-log statements.
    """
    mem = get_memory_system()
    mra = mem.get_mra_signals()

    # If cache is stale, run a fresh cycle from recent events.
    if mra is None:
        events = mem.event_log.tail(n=20)
        statements = [e.output for e in events if e.output]
        if not statements:
            return {
                "stress_level": 0.0,
                "contradictions": [],
                "deep_tensions": [],
                "bridging_questions": [],
                "suggested_action": "expand_knowledge",
            }

        monitor = EpistemicStressMonitor(embed_fn=mem.embedder.embed)
        stress = monitor.compute(statements)

        void_detector = VoidDetector()
        voids = None  # no graph available in auto-mode

        mem.update_mra_cache(stress, voids)
        mra = mem.get_mra_signals()

    stress = mra.last_stress
    voids = mra.last_voids

    contradictions: List[Dict[str, Any]] = []
    deep_tensions: List[Dict[str, Any]] = []
    bridging_questions: List[str] = []

    if stress is not None:
        for s1, s2, score in stress.contradictions:
            contradictions.append({"s1": s1, "s2": s2, "score": score})
        for s1, s2, score, sim in stress.deep_tensions:
            deep_tensions.append({"s1": s1, "s2": s2, "score": score, "similarity": sim})

    if voids is not None:
        bridging_questions = list(voids.questions)

    # Heuristic: choose a suggested action based on what the MRA found.
    suggested_action = "expand_knowledge"
    if deep_tensions:
        suggested_action = "resolve_deep_tension"
    elif contradictions:
        suggested_action = "resolve_contradiction"
    elif bridging_questions:
        suggested_action = "explore_gap"

    return {
        "stress_level": stress.s_omega if stress else 0.0,
        "contradictions": contradictions,
        "deep_tensions": deep_tensions,
        "bridging_questions": bridging_questions,
        "suggested_action": suggested_action,
    }
