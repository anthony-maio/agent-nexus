"""MRA Resolution Pipeline — the missing Section 3.

Orchestrates the full detect → classify → resolve → summarise flow.
Takes raw stress results and produces a resolution report with:
- Auto-resolved count (truncation, duplicate, outdated artifacts)
- Escalations (genuine contradictions needing investigation)
- Human-readable summary suitable for context injection at low salience
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from continuity_core.mra.resolver import (
    ContradictionClassifier,
    ResolutionEngine,
    ResolutionResult,
)
from continuity_core.mra.stress import StressResult
from continuity_core.mra.voids import VoidReport


@dataclass
class MRAResolutionReport:
    """Summary of a resolution pipeline run."""

    auto_resolved: int = 0
    escalated: int = 0
    total_processed: int = 0
    results: List[ResolutionResult] = field(default_factory=list)
    escalations: List[ResolutionResult] = field(default_factory=list)
    graph_ops: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    void_count: int = 0
    bridging_questions: List[str] = field(default_factory=list)

    @property
    def has_escalations(self) -> bool:
        return len(self.escalations) > 0


class MRAResolutionPipeline:
    """Run classification and resolution over MRA stress results.

    Usage::

        pipeline = MRAResolutionPipeline()
        report = pipeline.run(stress_result, voids_result)
        # report.summary → inject into context at low salience
        # report.escalations → create structured investigation tasks
        # report.graph_ops → apply to knowledge graph
    """

    def __init__(
        self,
        classifier: Optional[ContradictionClassifier] = None,
        engine: Optional[ResolutionEngine] = None,
    ) -> None:
        self.classifier = classifier or ContradictionClassifier()
        self.engine = engine or ResolutionEngine()

    def run(
        self,
        stress: StressResult,
        voids: Optional[VoidReport] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MRAResolutionReport:
        """Process all contradictions from a stress result.

        1. Deep tensions first (most serious).
        2. Regular contradictions (skip those already in deep tensions).
        3. Classify each → resolve or escalate.
        4. Collect graph operations.
        5. Build summary text.
        """
        meta = metadata or {}
        report = MRAResolutionReport()

        # Track which pairs we've already processed
        seen_pairs: set[tuple[str, str]] = set()

        # Process deep tensions first — these are the most serious
        for s1, s2, score, similarity in stress.deep_tensions:
            pair = (s1, s2)
            seen_pairs.add(pair)
            classified = self.classifier.classify(
                s1,
                s2,
                score,
                similarity,
                metadata=meta,
            )
            result = self.engine.resolve(classified)
            self._record(report, result)

        # Process regular contradictions (skip those covered by deep tensions)
        for s1, s2, score in stress.contradictions:
            if (s1, s2) in seen_pairs:
                continue
            seen_pairs.add((s1, s2))
            # We don't have similarity pre-computed for regular contradictions,
            # so we approximate from score (higher contradiction → lower implicit
            # similarity, unless it's a truncation artifact).
            similarity = 0.0  # Will be refined if classifier needs it
            classified = self.classifier.classify(
                s1,
                s2,
                score,
                similarity,
                metadata=meta,
            )
            result = self.engine.resolve(classified)
            self._record(report, result)

        # Record void information
        if voids is not None:
            report.void_count = len(voids.void_pairs)
            report.bridging_questions = list(voids.questions[:5])

        # Build summary
        report.summary = self._build_summary(report)
        return report

    def _record(self, report: MRAResolutionReport, result: ResolutionResult) -> None:
        """Record a resolution result in the report."""
        report.total_processed += 1
        report.results.append(result)

        if result.resolved:
            report.auto_resolved += 1
            report.graph_ops.extend(result.graph_ops)
        else:
            report.escalated += 1
            report.escalations.append(result)

    def _build_summary(self, report: MRAResolutionReport) -> str:
        """Build a concise, human-readable summary for context injection."""
        if report.total_processed == 0 and report.void_count == 0:
            return ""

        parts: list[str] = []

        if report.auto_resolved > 0:
            # Breakdown by type
            type_counts: dict[str, int] = {}
            for r in report.results:
                if r.resolved:
                    t = r.contradiction.type.value
                    type_counts[t] = type_counts.get(t, 0) + 1

            breakdown = ", ".join(f"{count} {typ}" for typ, count in type_counts.items())
            parts.append(
                f"Knowledge maintenance: {report.auto_resolved} issues auto-resolved ({breakdown})."
            )

        if report.escalated > 0:
            parts.append(
                f"{report.escalated} genuine contradiction(s) escalated for investigation."
            )

        if report.void_count > 0:
            parts.append(f"{report.void_count} knowledge gap(s) detected.")

        return " ".join(parts)
