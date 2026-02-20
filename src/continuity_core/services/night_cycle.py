"""Night Cycle — the background maintenance loop.

Runs decay sweeps, stress recomputation, void scanning, consolidation
pruning, and harmonic integration tracking in a single structured pass.
Can be triggered on-demand via the ``c2.maintenance`` MCP tool or
scheduled externally (cron, timer, idle-GPU hook).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from continuity_core.memory.system import TieredMemorySystem
from continuity_core.mra.stress import EpistemicStressMonitor, StressResult
from continuity_core.mra.voids import VoidDetector, VoidReport

logger = logging.getLogger(__name__)


@dataclass
class NightCycleResult:
    """Summary of one maintenance pass."""
    decay_applied: bool = False
    items_pruned: int = 0
    stress_before: float = 0.0
    stress_after: float = 0.0
    contradictions_found: int = 0
    deep_tensions_found: int = 0
    voids_found: int = 0
    resolutions: List[Dict[str, Any]] = field(default_factory=list)
    harmonic_reward: float = 0.0
    duration_sec: float = 0.0


class NightCycle:
    """Structured maintenance pass over the memory system.

    The cycle performs, in order:

    1. **Decay sweep** — reduce salience of stale items.
    2. **Prune** — remove items whose salience fell below threshold.
    3. **Stress recomputation** — scan recent statements for contradictions.
    4. **Void scan** — detect weakly-connected clusters (when graph available).
    5. **Harmonic integration** — compare stress before/after to detect
       resolutions (contradictions that went away since last cycle).
    6. **Cache update** — store fresh MRA results for next context call.
    7. **Log** — write a maintenance event to the event log.
    """

    def __init__(
        self,
        memory_system: TieredMemorySystem,
        prune_threshold: float = 0.05,
        statement_window: int = 30,
    ) -> None:
        self._mem = memory_system
        self._prune_threshold = prune_threshold
        self._statement_window = statement_window

    def run(self, graph: Optional[Dict[str, set]] = None) -> NightCycleResult:
        t0 = time.time()
        result = NightCycleResult()

        # Capture stress *before* this cycle for harmonic integration.
        prev_cache = self._mem.get_mra_signals()
        if prev_cache is not None and prev_cache.last_stress is not None:
            result.stress_before = prev_cache.last_stress.s_omega
            prev_contradictions = {
                (s1, s2) for s1, s2, _ in prev_cache.last_stress.contradictions
            }
        else:
            prev_contradictions = set()

        # 1. Decay
        self._mem._run_decay()
        result.decay_applied = True

        # 2. Prune
        if self._mem._fallback is not None:
            result.items_pruned = self._mem._fallback.prune(self._prune_threshold)

        # 3. Stress recomputation from recent event log
        events = self._mem.event_log.tail(n=self._statement_window)
        statements = [e.output for e in events if e.output]
        monitor = EpistemicStressMonitor(embed_fn=self._mem.embedder.embed)

        from continuity_core.mcp.tools.introspect import _graph_sparsity
        sparsity = _graph_sparsity(graph) if graph else 1.0
        stress = monitor.compute(statements, graph_sparsity=sparsity)
        result.stress_after = stress.s_omega
        result.contradictions_found = len(stress.contradictions)
        result.deep_tensions_found = len(stress.deep_tensions)

        # 4. Void scan
        void_detector = VoidDetector()
        voids: Optional[VoidReport] = None
        if graph:
            voids = void_detector.detect_voids(graph)
            result.voids_found = len(voids.void_pairs)

        # 5. Harmonic integration — detect resolved contradictions
        if prev_contradictions:
            current_contradictions = {
                (s1, s2) for s1, s2, _ in stress.contradictions
            }
            resolved = prev_contradictions - current_contradictions
            for s1, s2 in resolved:
                result.resolutions.append({"s1": s1, "s2": s2, "type": "contradiction_resolved"})

        # 5b. Distribute harmonic integration reward
        stress_delta = result.stress_after - result.stress_before
        result.harmonic_reward = self._mem.harmonic_integration(
            stress_delta=stress_delta,
            resolutions=result.resolutions,
        )

        # 6. Update MRA cache
        self._mem.update_mra_cache(stress, voids)

        # 7. Log the maintenance event
        result.duration_sec = time.time() - t0
        self._mem.write_event(
            actor="system",
            intent="maintenance",
            inp="night_cycle",
            out=(
                f"decay=True pruned={result.items_pruned} "
                f"stress={result.stress_after:.3f} "
                f"contradictions={result.contradictions_found} "
                f"deep_tensions={result.deep_tensions_found} "
                f"voids={result.voids_found} "
                f"resolutions={len(result.resolutions)} "
                f"duration={result.duration_sec:.2f}s"
            ),
            tags=["maintenance", "night_cycle"],
        )

        return result
