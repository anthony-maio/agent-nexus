"""Contradiction classifier and resolution engine for the MRA.

Implements Section 3 of the MRA theory — the Autonomous Resolution Loop.
Most contradictions detected by the stress monitor are artifacts (truncation,
duplicates, outdated info).  The classifier identifies these so the resolution
engine can auto-resolve them, leaving only genuine contradictions for
structured investigation by task agents.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ContradictionType(enum.Enum):
    """Classification of a detected contradiction."""

    TRUNCATION = "truncation"
    OUTDATED = "outdated"
    DUPLICATE = "duplicate"
    GENUINE = "genuine"
    VOID = "void"


@dataclass
class ClassifiedContradiction:
    """A contradiction with its classification and metadata."""

    s1: str
    s2: str
    score: float
    similarity: float
    type: ContradictionType
    confidence: float  # How confident the classifier is in this label
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolutionResult:
    """Outcome of attempting to resolve a classified contradiction."""

    contradiction: ClassifiedContradiction
    resolved: bool
    action_taken: str  # e.g. "superseded_shorter", "merged", "escalated"
    summary: str  # Human-readable description
    graph_ops: List[Dict[str, Any]] = field(default_factory=list)


class ContradictionClassifier:
    """Classifies detected contradictions by likely cause.

    Most "contradictions" from the stress monitor are artifacts:
    - **Truncation:** One statement is a cut-off version of the other
      (Discord's 2000-char limit creates these constantly).
    - **Duplicate:** Near-identical statements scored as contradictory
      due to minor wording differences.
    - **Outdated:** Same topic discussed at different times; the older
      version is stale.
    - **Genuine:** Real disagreement that needs investigation.
    """

    def __init__(
        self,
        truncation_word_overlap: float = 0.80,
        truncation_length_ratio: float = 2.0,
        duplicate_similarity: float = 0.85,
        duplicate_max_contradiction: float = 0.6,
        outdated_time_diff_sec: float = 3600.0,
        outdated_topic_overlap: float = 0.40,
        genuine_min_score: float = 0.5,
    ) -> None:
        self.truncation_word_overlap = truncation_word_overlap
        self.truncation_length_ratio = truncation_length_ratio
        self.duplicate_similarity = duplicate_similarity
        self.duplicate_max_contradiction = duplicate_max_contradiction
        self.outdated_time_diff_sec = outdated_time_diff_sec
        self.outdated_topic_overlap = outdated_topic_overlap
        self.genuine_min_score = genuine_min_score

    def classify(
        self,
        s1: str,
        s2: str,
        score: float,
        similarity: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClassifiedContradiction:
        """Classify a contradiction pair."""
        meta = metadata or {}

        # Check truncation first — most common artifact
        if self._is_truncation(s1, s2):
            return ClassifiedContradiction(
                s1=s1, s2=s2, score=score, similarity=similarity,
                type=ContradictionType.TRUNCATION, confidence=0.9,
                metadata=meta,
            )

        # Check duplicate
        if self._is_duplicate(score, similarity):
            return ClassifiedContradiction(
                s1=s1, s2=s2, score=score, similarity=similarity,
                type=ContradictionType.DUPLICATE, confidence=0.85,
                metadata=meta,
            )

        # Check outdated (requires timestamps in metadata)
        if self._is_outdated(s1, s2, meta):
            return ClassifiedContradiction(
                s1=s1, s2=s2, score=score, similarity=similarity,
                type=ContradictionType.OUTDATED, confidence=0.75,
                metadata=meta,
            )

        # Genuine contradiction
        if score >= self.genuine_min_score:
            return ClassifiedContradiction(
                s1=s1, s2=s2, score=score, similarity=similarity,
                type=ContradictionType.GENUINE, confidence=score,
                metadata=meta,
            )

        # Below threshold — treat as duplicate/noise
        return ClassifiedContradiction(
            s1=s1, s2=s2, score=score, similarity=similarity,
            type=ContradictionType.DUPLICATE, confidence=0.6,
            metadata=meta,
        )

    def _is_truncation(self, s1: str, s2: str) -> bool:
        """Check if one statement is a truncated version of the other."""
        # Direct substring check
        if s1 in s2 or s2 in s1:
            return True

        # Truncation marker check
        truncation_markers = ("...", "(truncated)", "[truncated]", "…")
        has_marker = any(s1.rstrip().endswith(m) or s2.rstrip().endswith(m)
                        for m in truncation_markers)

        # Length ratio + high word overlap = truncation
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return False

        length_ratio = max(len1, len2) / min(len1, len2)
        if length_ratio < self.truncation_length_ratio:
            return False

        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        shorter_words = words1 if len(words1) <= len(words2) else words2
        if not shorter_words:
            return False

        overlap = len(words1 & words2) / len(shorter_words)
        if overlap >= self.truncation_word_overlap:
            return True

        # Has truncation marker + moderate overlap
        if has_marker and overlap >= 0.5:
            return True

        return False

    def _is_duplicate(self, score: float, similarity: float) -> bool:
        """High similarity + low contradiction = duplicate, not conflict."""
        return (similarity >= self.duplicate_similarity
                and score < self.duplicate_max_contradiction)

    def _is_outdated(self, s1: str, s2: str, meta: Dict[str, Any]) -> bool:
        """Same topic at different times — older version is stale."""
        t1 = meta.get("timestamp_s1", 0.0)
        t2 = meta.get("timestamp_s2", 0.0)
        if not t1 or not t2:
            return False

        time_diff = abs(float(t1) - float(t2))
        if time_diff < self.outdated_time_diff_sec:
            return False

        # Check topic overlap
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        union = words1 | words2
        if not union:
            return False

        overlap = len(words1 & words2) / len(union)
        return overlap >= self.outdated_topic_overlap


class ResolutionEngine:
    """Resolves classified contradictions automatically where possible.

    - TRUNCATION → supersede the shorter fragment
    - DUPLICATE → merge into the longer/primary version
    - OUTDATED → supersede the older version
    - GENUINE → escalate for structured investigation
    """

    def resolve(self, classified: ClassifiedContradiction) -> ResolutionResult:
        """Attempt to resolve a classified contradiction."""
        handlers = {
            ContradictionType.TRUNCATION: self._resolve_truncation,
            ContradictionType.DUPLICATE: self._resolve_duplicate,
            ContradictionType.OUTDATED: self._resolve_outdated,
            ContradictionType.GENUINE: self._escalate,
            ContradictionType.VOID: self._escalate,
        }
        handler = handlers.get(classified.type, self._escalate)
        return handler(classified)

    def _resolve_truncation(self, c: ClassifiedContradiction) -> ResolutionResult:
        """Supersede the shorter (truncated) fragment."""
        shorter, longer = (c.s1, c.s2) if len(c.s1) <= len(c.s2) else (c.s2, c.s1)
        return ResolutionResult(
            contradiction=c,
            resolved=True,
            action_taken="superseded_shorter",
            summary=f"Truncation artifact resolved: kept complete version ({len(longer)} chars), "
                    f"superseded fragment ({len(shorter)} chars).",
            graph_ops=[{
                "op": "supersede",
                "superseded_text": shorter,
                "superseding_text": longer,
                "reason": "truncation",
            }],
        )

    def _resolve_duplicate(self, c: ClassifiedContradiction) -> ResolutionResult:
        """Merge duplicates — keep the longer/more complete version."""
        primary, secondary = (c.s1, c.s2) if len(c.s1) >= len(c.s2) else (c.s2, c.s1)
        return ResolutionResult(
            contradiction=c,
            resolved=True,
            action_taken="merged",
            summary=f"Duplicate resolved: merged into primary version ({len(primary)} chars).",
            graph_ops=[{
                "op": "supersede",
                "superseded_text": secondary,
                "superseding_text": primary,
                "reason": "duplicate",
            }],
        )

    def _resolve_outdated(self, c: ClassifiedContradiction) -> ResolutionResult:
        """Supersede the older version."""
        t1 = c.metadata.get("timestamp_s1", 0.0)
        t2 = c.metadata.get("timestamp_s2", 0.0)
        if float(t1) >= float(t2):
            newer, older = c.s1, c.s2
        else:
            newer, older = c.s2, c.s1
        return ResolutionResult(
            contradiction=c,
            resolved=True,
            action_taken="superseded_older",
            summary="Outdated info resolved: superseded older version, kept newer.",
            graph_ops=[{
                "op": "supersede",
                "superseded_text": older,
                "superseding_text": newer,
                "reason": "outdated",
            }],
        )

    def _escalate(self, c: ClassifiedContradiction) -> ResolutionResult:
        """Escalate genuine contradictions for structured investigation."""
        return ResolutionResult(
            contradiction=c,
            resolved=False,
            action_taken="escalated",
            summary=f"Genuine contradiction (score={c.score:.2f}): "
                    f"\"{c.s1[:80]}\" vs \"{c.s2[:80]}\"",
            graph_ops=[],
        )
