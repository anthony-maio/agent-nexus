from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from continuity_core.context.composer import Candidate, ContextComposer
from continuity_core.context.gather import CandidateGatherer
from continuity_core.memory.system import TieredMemorySystem
from continuity_core.pilot.verification import ConsciousnessPilot, VerificationResult
from continuity_core.config import C2Config, load_config


@dataclass
class ContextResult:
    chosen: List[Candidate]
    working_context: List[Dict[str, Any]]
    pilot_verdict: Optional[VerificationResult] = None


class ContextPipeline:
    def __init__(self, memory_system: Optional[TieredMemorySystem] = None,
                 config: Optional[C2Config] = None,
                 pilot: Optional[ConsciousnessPilot] = None) -> None:
        self.config = config or load_config()
        self.memory_system = memory_system or TieredMemorySystem(self.config)
        self.gatherer = CandidateGatherer(self.memory_system)
        self.composer = ContextComposer(
            token_budget=self.config.token_budget,
            epsilon=self.config.epsilon,
            lambda_penalty=self.config.lambda_penalty,
        )
        self.pilot = pilot or ConsciousnessPilot()

    def run(self, query: str, thread_id: Optional[str] = None, top_k: int = 8) -> ContextResult:
        candidates, working_context = self.gatherer.gather(query, thread_id=thread_id, top_k=top_k)
        chosen = self.composer.select(candidates)

        verdict = self.pilot.verify(
            action=query,
            chosen_candidates=chosen,
        )

        return ContextResult(
            chosen=chosen,
            working_context=working_context,
            pilot_verdict=verdict,
        )
