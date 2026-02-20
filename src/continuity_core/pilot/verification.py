"""Consciousness Pilot — the four-step verification protocol.

The Pilot sits between the subconscious (MRA + memory system) and conscious
execution.  It does NOT generate content — it *decides* whether content
produced by the system is safe to emit, needs to be downgraded to a question,
or should be aborted entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from continuity_core.context.composer import Candidate


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

class PilotVerdict(str, Enum):
    COMMIT = "commit"           # Safe to execute
    DOWNGRADE = "downgrade"     # Too uncertain — convert action to question
    ABORT = "abort"             # Violates safety constraint


@dataclass
class VerificationResult:
    verdict: PilotVerdict
    confidence: float
    reasons: List[str] = field(default_factory=list)
    original_action: str = ""
    revised_action: Optional[str] = None


# ---------------------------------------------------------------------------
# Safety constraints (pluggable)
# ---------------------------------------------------------------------------

class SafetyConstraint(Protocol):
    """A callable that returns a violation reason or None if safe."""
    def __call__(self, action: str, context: Dict[str, Any]) -> Optional[str]:
        ...


class PatternConstraint:
    """Block actions matching a regex pattern."""

    def __init__(self, pattern: str, reason: str) -> None:
        self._pattern = re.compile(pattern, re.IGNORECASE)
        self._reason = reason

    def __call__(self, action: str, context: Dict[str, Any]) -> Optional[str]:
        if self._pattern.search(action):
            return self._reason
        return None


# Default constraints — prevent common failure modes.
DEFAULT_CONSTRAINTS: List[SafetyConstraint] = [
    PatternConstraint(
        r"(?:drop\s+table|delete\s+from|truncate)\b",
        "Destructive database operation detected",
    ),
    PatternConstraint(
        r"(?:rm\s+-rf|rmdir\s+/|format\s+c:)",
        "Destructive filesystem operation detected",
    ),
]


# ---------------------------------------------------------------------------
# The Pilot
# ---------------------------------------------------------------------------

class ConsciousnessPilot:
    """Four-step verification: Intent → Safety → Uncertainty → Commit/Abort.

    Parameters
    ----------
    safety_constraints :
        List of callables that inspect an action string and return a
        violation reason (str) or None.
    uncertainty_threshold :
        If average confidence of chosen candidates is below this, the
        Pilot downgrades the action to a question.
    min_candidates :
        If fewer than this many candidates were selected, treat as
        high-uncertainty (the system doesn't have enough context).
    goal_keywords :
        Optional set of keywords representing current goals.  If provided,
        the Pilot checks that at least one goal keyword appears in the
        chosen context — otherwise it flags low intent alignment.
    """

    def __init__(
        self,
        safety_constraints: Optional[List[SafetyConstraint]] = None,
        uncertainty_threshold: float = 0.4,
        min_candidates: int = 1,
        goal_keywords: Optional[List[str]] = None,
    ) -> None:
        self.safety_constraints = safety_constraints or list(DEFAULT_CONSTRAINTS)
        self.uncertainty_threshold = uncertainty_threshold
        self.min_candidates = min_candidates
        self.goal_keywords = [kw.lower() for kw in (goal_keywords or [])]

    def verify(
        self,
        action: str,
        chosen_candidates: List[Candidate],
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """Run the four-step protocol against a proposed action."""
        context = context or {}
        reasons: List[str] = []

        # Step 1 — Review Intent
        intent_ok = self._check_intent(chosen_candidates, reasons)

        # Step 2 — Check Safety
        safety_ok = self._check_safety(action, context, reasons)
        if not safety_ok:
            return VerificationResult(
                verdict=PilotVerdict.ABORT,
                confidence=0.0,
                reasons=reasons,
                original_action=action,
            )

        # Step 3 — Uncertainty Check
        avg_confidence = self._average_confidence(chosen_candidates)
        low_context = len(chosen_candidates) < self.min_candidates
        uncertain = avg_confidence < self.uncertainty_threshold or low_context

        if uncertain:
            if low_context:
                reasons.append(
                    f"Insufficient context: {len(chosen_candidates)} candidates "
                    f"(minimum: {self.min_candidates})"
                )
            if avg_confidence < self.uncertainty_threshold:
                reasons.append(
                    f"Low confidence: {avg_confidence:.2f} "
                    f"(threshold: {self.uncertainty_threshold:.2f})"
                )
            return VerificationResult(
                verdict=PilotVerdict.DOWNGRADE,
                confidence=avg_confidence,
                reasons=reasons,
                original_action=action,
                revised_action=self._downgrade_to_question(action),
            )

        # Step 4 — Commit
        if not intent_ok:
            reasons.append("Proceeding despite low intent alignment")

        return VerificationResult(
            verdict=PilotVerdict.COMMIT,
            confidence=avg_confidence,
            reasons=reasons,
            original_action=action,
        )

    # -- Step implementations -----------------------------------------------

    def _check_intent(self, candidates: List[Candidate], reasons: List[str]) -> bool:
        """Check whether chosen context aligns with known goals."""
        if not self.goal_keywords:
            return True  # No goals configured — pass by default
        all_text = " ".join(c.text.lower() for c in candidates)
        matched = [kw for kw in self.goal_keywords if kw in all_text]
        if not matched:
            reasons.append(
                f"No goal keywords found in context "
                f"(looking for: {', '.join(self.goal_keywords)})"
            )
            return False
        return True

    def _check_safety(
        self, action: str, context: Dict[str, Any], reasons: List[str]
    ) -> bool:
        """Run all safety constraints.  Returns False on first violation."""
        for constraint in self.safety_constraints:
            violation = constraint(action, context)
            if violation is not None:
                reasons.append(f"Safety violation: {violation}")
                return False
        return True

    def _average_confidence(self, candidates: List[Candidate]) -> float:
        if not candidates:
            return 0.0
        return sum(c.confidence for c in candidates) / len(candidates)

    def _downgrade_to_question(self, action: str) -> str:
        """Convert an action statement into a clarifying question."""
        action = action.strip().rstrip(".")
        return (
            f"I'm not confident enough to proceed with: \"{action}\". "
            "Could you confirm or provide more context?"
        )
