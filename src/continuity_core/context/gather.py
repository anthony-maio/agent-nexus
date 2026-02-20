from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from continuity_core.context.composer import Candidate
from continuity_core.event_log import EventLog
from continuity_core.memory.system import MRACache, ScoredMemory, TieredMemorySystem
from continuity_core.storage.neo4j import Neo4jGraphStore

MAX_MRA_INJECTIONS = 5


class CandidateGatherer:
    def __init__(self, memory_system: TieredMemorySystem, max_mra_injections: int = MAX_MRA_INJECTIONS) -> None:
        self._memory = memory_system
        self._max_mra_injections = max_mra_injections

    def gather(self, query: str, thread_id: Optional[str] = None, top_k: int = 8) -> Tuple[List[Candidate], List[Dict[str, Any]]]:
        candidates: List[Candidate] = []
        working_context: List[Dict[str, Any]] = []
        now = time.time()

        if thread_id:
            working_context = self._memory.get_working_context(thread_id, limit=12)
            for msg in working_context:
                text = str(msg.get("content", ""))
                candidates.append(_candidate_from_text("working", text, now, base_relevance=0.9))

        memories = self._memory.recall(query, top_k=top_k)
        for mem in memories:
            candidates.append(_candidate_from_memory(mem, now))

        if self._memory.neo4j is not None:
            candidates.extend(self._graph_candidates(query, self._memory.neo4j, now))

        events = self._memory.event_log.query(limit=8)
        for event in events:
            text = f"{event.intent}: {event.output}"
            candidates.append(_candidate_from_text("event", text, now, base_relevance=0.4))

        # Inject MRA signals (contradictions, voids, deep tensions) so the
        # pilot agent sees epistemic pressure in its context window.
        mra = self._memory.get_mra_signals()
        if mra is not None:
            candidates.extend(self._mra_candidates(mra, now))
            # Reduce confidence for candidates that appear in contradictions.
            self._adjust_confidence_from_mra(candidates, mra)

        return candidates, working_context

    def _mra_candidates(self, mra: MRACache, now: float) -> List[Candidate]:
        out: List[Candidate] = []
        stress = mra.last_stress
        if stress is None or not stress.should_trigger:
            return out

        # Deep tensions first — highest priority MRA signals.
        for s1, s2, score, sim in stress.deep_tensions:
            if len(out) >= self._max_mra_injections:
                break
            text = (
                f"[Deep Tension] \"{s1}\" vs \"{s2}\" "
                f"(contradiction: {score:.2f}, similarity: {sim:.2f}). "
                "These views are tightly coupled yet contradictory — resolve or investigate."
            )
            out.append(_candidate_from_text("mra", text, now, base_relevance=0.95, salience=0.95, confidence=0.9))

        # Regular contradictions (skip those already covered by deep tensions).
        deep_pairs: Set[Tuple[str, str]] = {(s1, s2) for s1, s2, _, _ in stress.deep_tensions}
        for s1, s2, score in stress.contradictions:
            if len(out) >= self._max_mra_injections:
                break
            if (s1, s2) in deep_pairs:
                continue
            text = f"[Contradiction] \"{s1}\" vs \"{s2}\" (score: {score:.2f})"
            out.append(_candidate_from_text("mra", text, now, base_relevance=0.9, salience=0.9, confidence=0.85))

        # Bridging questions from void detection.
        voids = mra.last_voids
        if voids is not None:
            for question in voids.questions:
                if len(out) >= self._max_mra_injections:
                    break
                text = f"[Knowledge Gap] {question}"
                out.append(_candidate_from_text("mra", text, now, base_relevance=0.85, salience=0.85, confidence=0.8))

        return out

    def _graph_candidates(self, query: str, graph: Neo4jGraphStore, now: float) -> List[Candidate]:
        nodes = graph.query_nodes(text=query, node_types=None, limit=8)
        out: List[Candidate] = []
        for node in nodes:
            name = node.get("name", "")
            desc = node.get("description", "") or ""
            text = name if not desc else f"{name} - {desc}"
            # Derive centrality from graph degree if available.
            degree = int(node.get("degree", 0))
            centrality = min(1.0, degree / 20.0) if degree > 0 else 0.3
            out.append(Candidate(
                id=_stable_id("graph", text),
                text=text,
                store="graph",
                token_cost=_token_cost(text),
                relevance=0.6,
                recency_sec=0.0,
                centrality=centrality,
                confidence=0.7,
                task_match=0.7,
                salience=0.6,
            ))
        return out

    def _adjust_confidence_from_mra(self, candidates: List[Candidate], mra: MRACache) -> None:
        """Reduce confidence for candidates whose content appears in contradictions."""
        stress = mra.last_stress
        if stress is None:
            return
        contradicted_texts: Dict[str, float] = {}
        for s1, s2, score in stress.contradictions:
            contradicted_texts[s1] = max(contradicted_texts.get(s1, 0.0), score)
            contradicted_texts[s2] = max(contradicted_texts.get(s2, 0.0), score)
        if not contradicted_texts:
            return
        for cand in candidates:
            if cand.store == "mra":
                continue  # Don't adjust MRA signals themselves
            for text_fragment, score in contradicted_texts.items():
                if text_fragment in cand.text:
                    cand.confidence = max(0.1, cand.confidence * (1.0 - 0.3 * score))
                    break


def _candidate_from_memory(mem: ScoredMemory, now: float) -> Candidate:
    text = mem.content
    recency = now - float(mem.payload.get("last_accessed", now))
    return Candidate(
        id=mem.id,
        text=text,
        store=mem.memory_type,
        token_cost=_token_cost(text),
        relevance=mem.score,
        recency_sec=recency,
        centrality=0.4,
        confidence=float(mem.payload.get("confidence", 0.8)),
        task_match=0.8,
        salience=float(mem.payload.get("importance", 5)) / 10.0,
    )


def _candidate_from_text(store: str, text: str, now: float, base_relevance: float = 0.5,
                          salience: float = 0.6, confidence: float = 0.7) -> Candidate:
    return Candidate(
        id=_stable_id(store, text),
        text=text,
        store=store,
        token_cost=_token_cost(text),
        relevance=base_relevance,
        recency_sec=0.0,
        centrality=0.3,
        confidence=confidence,
        task_match=0.7,
        salience=salience,
    )


def _token_cost(text: str) -> int:
    return max(1, int(len(text) * 0.25))


def _stable_id(store: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{store}:{digest}"
