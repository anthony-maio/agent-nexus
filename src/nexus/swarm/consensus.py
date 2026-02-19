"""Multi-model consensus protocol for the Agent Nexus swarm.

For decisions that benefit from diverse opinion -- tool approval, code
review, security checks -- the ``ConsensusProtocol`` fans a structured
vote prompt out to multiple models in parallel and tallies the results
into an actionable ``ConsensusResult``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

log = logging.getLogger(__name__)


class ConsensusOutcome(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_HUMAN = "needs_human"
    TIMEOUT = "timeout"


@dataclass
class Vote:
    model_id: str
    decision: str        # "approve", "reject", or "abstain"
    reasoning: str
    confidence: float    # 0.0-1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConsensusResult:
    question: str
    votes: list[Vote]
    outcome: ConsensusOutcome
    agreement_ratio: float  # 0.0-1.0
    summary: str


class ConsensusProtocol:
    """Multi-model voting for decisions requiring diverse opinion.

    Used for: tool approval, code review, security concerns, any action
    where multiple perspectives improve safety.
    """

    def __init__(self, threshold: float = 0.5, timeout: float = 60.0) -> None:
        self.threshold = threshold  # Agreement ratio required
        self.timeout = timeout      # Seconds to wait for votes
        self._active_votes: dict[str, list[Vote]] = {}  # question_id -> votes

    async def request_consensus(
        self,
        question: str,
        model_ids: list[str],
        call_model_fn,  # async fn(model_id, prompt) -> str
    ) -> ConsensusResult:
        """Request a consensus vote from multiple models.

        Args:
            question: The decision to vote on
            model_ids: Which models should vote
            call_model_fn: Async function to call each model. Takes (model_id, prompt) -> response string.

        Returns:
            ConsensusResult with the outcome
        """
        prompt = (
            "You are participating in a multi-model consensus vote.\n\n"
            f"QUESTION: {question}\n\n"
            "Respond with EXACTLY this format:\n"
            "DECISION: approve | reject | abstain\n"
            "CONFIDENCE: 0.0-1.0\n"
            "REASONING: your explanation\n"
        )

        votes: list[Vote] = []

        async def get_vote(model_id: str) -> Vote | None:
            try:
                response = await asyncio.wait_for(
                    call_model_fn(model_id, prompt),
                    timeout=self.timeout,
                )
                return self._parse_vote(model_id, response)
            except asyncio.TimeoutError:
                log.warning(f"Consensus timeout for {model_id}")
                return None
            except Exception as e:
                log.error(f"Consensus error from {model_id}: {e}")
                return None

        # Collect votes in parallel
        tasks = [get_vote(mid) for mid in model_ids]
        results = await asyncio.gather(*tasks)
        votes = [v for v in results if v is not None]

        if not votes:
            return ConsensusResult(
                question=question,
                votes=[],
                outcome=ConsensusOutcome.TIMEOUT,
                agreement_ratio=0.0,
                summary="No models responded within the timeout.",
            )

        # Calculate outcome
        approvals = sum(1 for v in votes if v.decision == "approve")
        rejections = sum(1 for v in votes if v.decision == "reject")
        total_decisive = approvals + rejections

        if total_decisive == 0:
            outcome = ConsensusOutcome.NEEDS_HUMAN
            agreement_ratio = 0.0
        else:
            agreement_ratio = max(approvals, rejections) / total_decisive
            if agreement_ratio >= self.threshold:
                outcome = ConsensusOutcome.APPROVED if approvals > rejections else ConsensusOutcome.REJECTED
            else:
                outcome = ConsensusOutcome.NEEDS_HUMAN

        summary_parts = [f"{len(votes)}/{len(model_ids)} models voted."]
        if approvals:
            summary_parts.append(f"{approvals} approved.")
        if rejections:
            summary_parts.append(f"{rejections} rejected.")
        summary_parts.append(f"Agreement: {agreement_ratio:.0%}. Outcome: {outcome.value}.")

        return ConsensusResult(
            question=question,
            votes=votes,
            outcome=outcome,
            agreement_ratio=agreement_ratio,
            summary=" ".join(summary_parts),
        )

    def _parse_vote(self, model_id: str, response: str) -> Vote:
        """Parse a model's vote response."""
        decision = "abstain"
        confidence = 0.5
        reasoning = response

        lines = response.strip().split("\n")
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith("decision:"):
                val = line_lower.split(":", 1)[1].strip()
                if "approve" in val:
                    decision = "approve"
                elif "reject" in val:
                    decision = "reject"
                else:
                    decision = "abstain"
            elif line_lower.startswith("confidence:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            elif line_lower.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return Vote(
            model_id=model_id,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
        )
