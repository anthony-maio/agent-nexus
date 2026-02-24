"""Anti-hallucination guardrails for the Agent Nexus orchestrator.

Three automated safeguards that prevent LLM hallucination from
propagating through the swarm feedback loop:

1. **Entity Grounding** -- Drops decision actions that reference specific
   external entities (PR numbers, Jira tickets, URLs) not found in the
   actual gathered state.

2. **Task Output Validation** -- Screens task agent (1.2B model) outputs
   for hallucination signals: off-topic content, fabricated specifics,
   generic filler, unsubstantiated confidence claims.

3. **Idle-Loop Detection** -- Tracks topic similarity across consecutive
   orchestrator cycles and suppresses auto-dispatch when the swarm is
   recycling the same content with no new external input.
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)


# =====================================================================
# Guardrail 1: Entity Grounding
# =====================================================================

# Patterns for specific external entities the bot has no way to verify.
_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:PR|pull request)\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,10}-\d{2,6}\b"),  # Jira-style: SEC-2847
    re.compile(r"https?://\S+"),  # URLs
    re.compile(r"\bcommit\s+[0-9a-f]{7,40}\b", re.IGNORECASE),
    re.compile(r"\bqueue\s+position\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\b(?:build|pipeline|workflow)\s*#\d+\b", re.IGNORECASE),
    re.compile(
        r"\b(?:review(?:er|ed)\s+by)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
    ),
]


def check_entity_grounding(
    actions: list[dict[str, Any]],
    state_text: str,
) -> list[dict[str, Any]]:
    """Remove actions that reference entities not present in state.

    Scans each action's ``description`` for entity-like patterns (PR
    numbers, Jira tickets, URLs, commit SHAs, etc.).  If any match is
    not found verbatim in ``state_text``, the action is dropped.

    Args:
        actions: Parsed action list from ``_parse_actions()``.
        state_text: The full decision prompt built from gathered state.

    Returns:
        Filtered list containing only grounded actions.
    """
    grounded: list[dict[str, Any]] = []
    state_lower = state_text.lower()

    for action in actions:
        description = action.get("description", "")
        ungrounded: list[str] = []

        for pattern in _ENTITY_PATTERNS:
            for match in pattern.findall(description):
                if match.lower() not in state_lower:
                    ungrounded.append(match)

        if ungrounded:
            log.warning(
                "GUARDRAIL: Dropping ungrounded action: '%.100s' "
                "-- references not in state: %s",
                description,
                ungrounded[:5],
            )
        else:
            grounded.append(action)

    dropped = len(actions) - len(grounded)
    if dropped:
        log.info(
            "GUARDRAIL: Entity grounding dropped %d of %d action(s).",
            dropped,
            len(actions),
        )

    return grounded


# =====================================================================
# Guardrail 2: Task Agent Output Validation
# =====================================================================

# Vocabulary sets that indicate off-topic hallucination.
_OFFTOPIC_MEDICAL: set[str] = {
    "tavp", "transcatheter", "aortic", "cardiac", "cardiovascular",
    "myocardial", "endoscopic", "catheter", "arterial", "ventricular",
    "angioplasty", "hemodynamic", "intravenous", "pathology",
    "prognosis", "surgical", "clinical",
}

_OFFTOPIC_LEGAL: set[str] = {
    "plaintiff", "defendant", "subpoena", "deposition", "tort",
    "jurisprudence", "litigation", "arbitration", "indictment",
    "arraignment", "statutory",
}

_FABRICATED_SPECIFICS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:PR|pull request)\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,10}-\d{2,6}\b"),  # Jira-style
    re.compile(r"\bqueue\s+position\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\b\d+\.\d{3,}%"),  # Overly precise: "99.847%"
    re.compile(
        r"\b(?:reviewed by|assigned to|approved by)\s+[A-Z][a-z]+",
        re.IGNORECASE,
    ),
]

# Patterns indicating fabricated infrastructure/system access.
# Task agents are LLMs with no filesystem, DB, or infra access.
_FABRICATED_INFRA: list[re.Pattern[str]] = [
    # File paths the agent claims to have accessed
    re.compile(r"(?:/app/|/var/|/etc/|/config/|/bin/)\S+", re.IGNORECASE),
    # Shell commands the agent claims to have run
    re.compile(r"\b(?:cat|ls|grep|curl|docker|kubectl)\s+\S+"),
    # Database queries the agent claims to have executed
    re.compile(
        r"\b(?:SELECT|INSERT|CALL|MATCH|CREATE)\s+\w+",
        re.IGNORECASE,
    ),
    # Log IDs and process IDs the agent fabricated
    re.compile(r"\bLog ID\s*#?\d+\b", re.IGNORECASE),
    re.compile(r"\bPID\s*:?\s*\d+\b"),
    # Error codes from systems the agent can't access
    re.compile(r"\bError\s+\d{3}\b"),
    # Fabricated agent names
    re.compile(
        r"\bAgent\s+[A-Z][a-z]+-(?:Alpha|Beta|Gamma|\d+)\b",
    ),
]

_FILLER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\(\d+\s*words?\)", re.IGNORECASE),
]

_CONFIDENCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bconfirmed\b", re.IGNORECASE),
    re.compile(r"\bverified\b", re.IGNORECASE),
    re.compile(r"\bvalidated\b", re.IGNORECASE),
    re.compile(r"\baction completed per protocol\b", re.IGNORECASE),
    re.compile(
        r"\bsuccessfully (?:resolved|implemented|deployed|executed)\b",
        re.IGNORECASE,
    ),
]


def validate_task_output(
    result_text: str,
    task_description: str,
) -> tuple[bool, str]:
    """Check task agent output for hallucination signals.

    Args:
        result_text: The raw text from the task agent.
        task_description: The original task description.

    Returns:
        ``(True, "")`` if the output looks valid, or
        ``(False, reason)`` describing the detected signal.
    """
    words = set(re.findall(r"\b\w+\b", result_text.lower()))
    reasons: list[str] = []

    # 1. Off-topic vocabulary
    medical_hits = words & _OFFTOPIC_MEDICAL
    if len(medical_hits) >= 3:
        reasons.append(f"off-topic medical terminology: {sorted(medical_hits)[:5]}")

    legal_hits = words & _OFFTOPIC_LEGAL
    if len(legal_hits) >= 3:
        reasons.append(f"off-topic legal terminology: {sorted(legal_hits)[:5]}")

    # 2. Fabricated specifics
    for pattern in _FABRICATED_SPECIFICS:
        matches = pattern.findall(result_text)
        if matches:
            reasons.append(f"fabricated specifics: {matches[:3]}")
            break

    # 2b. Fabricated infrastructure access
    infra_hits: list[str] = []
    for pattern in _FABRICATED_INFRA:
        matches = pattern.findall(result_text)
        infra_hits.extend(matches[:2])
    if len(infra_hits) >= 2:
        reasons.append(
            f"fabricated infrastructure access: {infra_hits[:4]}"
        )

    # 3. Filler padding
    for pattern in _FILLER_PATTERNS:
        if pattern.search(result_text):
            reasons.append("generic filler/padding detected")
            break

    # 4. Confidence without evidence (need 2+ hits)
    confidence_count = sum(
        1 for p in _CONFIDENCE_PATTERNS if p.search(result_text)
    )
    if confidence_count >= 2:
        reasons.append(
            "multiple unsubstantiated confidence claims "
            "(agent has no verification capability)"
        )

    if reasons:
        combined = "; ".join(reasons)
        log.warning(
            "GUARDRAIL: Task output failed validation: %s | "
            "Task: '%.80s' | Output: '%.200s'",
            combined,
            task_description,
            result_text,
        )
        return False, f"Hallucination detected: {combined}"

    return True, ""


# =====================================================================
# Guardrail 3: Idle-Loop Detection
# =====================================================================

# Words filtered out when extracting topic terms from action descriptions.
_STOP_WORDS: set[str] = {
    "the", "and", "for", "with", "from", "that", "this", "these",
    "those", "have", "has", "had", "been", "being", "will", "would",
    "could", "should", "might", "shall", "about", "into", "over",
    "after", "before", "between", "under", "above", "below", "each",
    "every", "both", "some", "such", "only", "also", "just", "very",
    "well", "back", "like", "more", "most", "other", "than", "then",
    "what", "when", "where", "which", "while", "does", "done",
    # Domain stop words (too generic to distinguish topics)
    "task", "agent", "swarm", "nexus", "analyze", "research",
    "summarize", "review", "current", "recent", "based", "determine",
    "identify", "ensure", "check", "verify", "assess", "evaluate",
}


class IdleLoopDetector:
    """Detects when the orchestrator is recycling the same topics.

    Tracks significant terms from action descriptions per cycle.  When
    term overlap with the previous cycle exceeds a threshold for several
    consecutive cycles, suppresses auto-dispatch.

    Args:
        overlap_threshold: Minimum term overlap fraction to count as
            stale (default: 0.60).
        stale_cycle_limit: Consecutive stale cycles before suppression
            (default: 3).
    """

    def __init__(
        self,
        overlap_threshold: float = 0.60,
        stale_cycle_limit: int = 3,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.stale_cycle_limit = stale_cycle_limit
        self._prev_terms: set[str] = set()
        self._staleness_counter: int = 0
        self._suppressed: bool = False

    def check_cycle(
        self,
        actions: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> bool:
        """Check whether this cycle's actions are stale.

        Call after entity grounding but before dispatch.

        Args:
            actions: The proposed actions for this cycle.
            state: The gathered state dict.

        Returns:
            ``True`` if the cycle should be suppressed (skip dispatch),
            ``False`` if it should proceed normally.
        """
        current_terms = self._extract_terms(actions)

        # Fresh external input resets staleness
        if self._has_fresh_input(state):
            if self._staleness_counter > 0:
                log.info(
                    "GUARDRAIL: Staleness counter reset (fresh input).",
                )
            self._staleness_counter = 0
            self._suppressed = False
            self._prev_terms = current_terms
            return False

        # Compute overlap with previous cycle
        if self._prev_terms and current_terms:
            intersection = self._prev_terms & current_terms
            smaller = min(len(self._prev_terms), len(current_terms))
            overlap = len(intersection) / smaller if smaller > 0 else 0.0

            if overlap >= self.overlap_threshold:
                self._staleness_counter += 1
                log.debug(
                    "GUARDRAIL: Stale cycle detected (overlap=%.2f, "
                    "counter=%d/%d, shared=%s).",
                    overlap,
                    self._staleness_counter,
                    self.stale_cycle_limit,
                    sorted(intersection)[:8],
                )
            else:
                # New topic — reset
                self._staleness_counter = 0
                self._suppressed = False

        self._prev_terms = current_terms

        if self._staleness_counter >= self.stale_cycle_limit:
            if not self._suppressed:
                log.warning(
                    "GUARDRAIL: Idle loop detected — suppressing dispatch "
                    "(%d consecutive stale cycles).",
                    self._staleness_counter,
                )
            self._suppressed = True
            return True

        return False

    # -- Internals --------------------------------------------------------

    def _extract_terms(self, actions: list[dict[str, Any]]) -> set[str]:
        """Extract significant terms from action descriptions."""
        text = " ".join(a.get("description", "") for a in actions).lower()
        words = set(re.findall(r"\b[a-z]{4,}\b", text))
        return words - _STOP_WORDS

    def _has_fresh_input(self, state: dict[str, Any]) -> bool:
        """Check if the state contains genuinely new external input."""
        for msg in state.get("recent_messages", []):
            if msg.get("author") == "human":
                return True

        # Only count non-stale activity as fresh input.
        activity = state.get("activity")
        if activity is not None and not getattr(activity, "is_stale", False):
            return True

        return False

    def reset(self) -> None:
        """Manually reset the detector."""
        self._prev_terms = set()
        self._staleness_counter = 0
        self._suppressed = False

    @property
    def staleness_counter(self) -> int:
        """Current number of consecutive stale cycles."""
        return self._staleness_counter

    @property
    def is_suppressed(self) -> bool:
        """Whether dispatch is currently suppressed."""
        return self._suppressed
