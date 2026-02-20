from __future__ import annotations

from typing import Any, Dict, List

from continuity_core.context import Candidate, ContextComposer, ContextPipeline
from continuity_core.config import load_config
from continuity_core.services.runtime import get_memory_system


_DEF_CONFIG = load_config()


def _token_cost(text: str) -> int:
    return max(1, int(len(text) * 0.25))


def build_context(arguments: Dict[str, Any]) -> Dict[str, Any]:
    candidates_in = arguments.get("candidates") or []
    query = arguments.get("query")
    use_pipeline = arguments.get("use_pipeline", True)
    token_budget = int(arguments.get("token_budget", _DEF_CONFIG.token_budget))

    if query and use_pipeline and not candidates_in:
        pipeline = ContextPipeline(memory_system=get_memory_system())
        result = pipeline.run(query=query, thread_id=arguments.get("thread_id"))
        return {
            "token_budget": token_budget,
            "working_context": result.working_context,
            "chosen": [
                {"id": c.id, "text": c.text, "store": c.store, "token_cost": c.token_cost}
                for c in result.chosen
            ],
        }

    composer = ContextComposer(
        token_budget=token_budget,
        epsilon=_DEF_CONFIG.epsilon,
        lambda_penalty=_DEF_CONFIG.lambda_penalty,
    )

    candidates: List[Candidate] = []
    for raw in candidates_in:
        text = str(raw.get("text", ""))
        candidates.append(
            Candidate(
                id=str(raw.get("id", "cand")),
                text=text,
                store=str(raw.get("store", "unknown")),
                token_cost=int(raw.get("token_cost", _token_cost(text))),
                relevance=float(raw.get("relevance", 0.5)),
                recency_sec=float(raw.get("recency_sec", 0.0)),
                centrality=float(raw.get("centrality", 0.3)),
                confidence=float(raw.get("confidence", 0.7)),
                task_match=float(raw.get("task_match", 0.7)),
                salience=float(raw.get("salience", 0.6)),
            )
        )

    chosen = composer.select(candidates)
    return {
        "token_budget": token_budget,
        "chosen": [
            {"id": c.id, "text": c.text, "store": c.store, "token_cost": c.token_cost}
            for c in chosen
        ],
    }
