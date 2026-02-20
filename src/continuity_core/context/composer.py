from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


@dataclass
class Candidate:
    id: str
    text: str
    store: str
    token_cost: int
    relevance: float
    recency_sec: float
    centrality: float
    confidence: float
    task_match: float
    salience: float


class ContextComposer:
    def __init__(self, token_budget: int = 2048, epsilon: float = 0.05, lambda_penalty: float = 0.001,
                 recency_half_life_sec: float = 6 * 3600) -> None:
        self.token_budget = token_budget
        self.epsilon = epsilon
        self.lambda_penalty = lambda_penalty
        self.recency_half_life_sec = recency_half_life_sec

    def _recency_weight(self, recency_sec: float) -> float:
        return math.exp(-math.log(2.0) * recency_sec / max(60.0, self.recency_half_life_sec))

    def expected_utility(self, cand: Candidate) -> float:
        rec = self._recency_weight(cand.recency_sec)
        score = (cand.relevance * rec * (0.5 + 0.5 * cand.centrality) *
                 (0.5 + 0.5 * cand.confidence) * (0.5 + 0.5 * cand.task_match))
        score *= (0.7 + 0.6 * cand.salience)
        penalty = self.lambda_penalty * cand.token_cost
        return score - penalty

    def select(self, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return []
        n_rand = max(1, int(len(candidates) * self.epsilon))
        rand_pick = random.sample(candidates, min(n_rand, len(candidates)))

        ranked = sorted(candidates, key=self.expected_utility, reverse=True)
        chosen: List[Candidate] = []
        used_tokens = 0
        for cand in ranked:
            if used_tokens + cand.token_cost > self.token_budget:
                continue
            chosen.append(cand)
            used_tokens += cand.token_cost
            if used_tokens >= self.token_budget:
                break

        for cand in rand_pick:
            if cand not in chosen and used_tokens + cand.token_cost <= self.token_budget:
                chosen.append(cand)
                used_tokens += cand.token_cost
        return chosen
