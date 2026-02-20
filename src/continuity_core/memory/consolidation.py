from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .gates import Gate, SmoothGate, AdaptiveGate


@dataclass
class ConsolidationFeedback:
    retrieval_accuracy: float
    coherence_gain: float
    eviction_count: int = 0


@dataclass
class ConsolidationPolicy:
    surprise_threshold: float = 0.7
    pattern_depth: int = 3
    abstraction_rate: float = 0.4
    cross_memory_integration: float = 0.6
    gate_bias: float = 0.0

    def update_from_feedback(self, feedback: ConsolidationFeedback) -> None:
        if feedback.retrieval_accuracy < 0.8:
            self.pattern_depth = min(12, int(self.pattern_depth * 1.2))
            self.cross_memory_integration = min(1.0, self.cross_memory_integration * 1.1)
        if feedback.coherence_gain < 0.1:
            self.abstraction_rate = max(0.1, self.abstraction_rate * 0.9)
        if feedback.eviction_count > 0:
            self.gate_bias = min(0.4, self.gate_bias + 0.05)


class RecallGatedConsolidator:
    def __init__(self, gate: Optional[Gate] = None, policy: Optional[ConsolidationPolicy] = None) -> None:
        self.gate = gate or SmoothGate(theta=0.5, sharpness=10.0)
        self.policy = policy or ConsolidationPolicy()

    def gate_value(self, recall_strength: float, occupancy: float, uncertainty: float = 0.0) -> float:
        # Adaptive plasticity: raise consolidation when uncertainty is high.
        bias = self.policy.gate_bias + (0.2 * max(0.0, min(1.0, uncertainty)))
        if isinstance(self.gate, AdaptiveGate):
            return min(1.0, max(0.0, self.gate(recall_strength, occupancy=occupancy) + bias))
        return min(1.0, max(0.0, self.gate(recall_strength) + bias))

    def should_consolidate(self, recall_strength: float, occupancy: float, uncertainty: float = 0.0) -> bool:
        return self.gate_value(recall_strength, occupancy, uncertainty) > 0.5

    def nested_learning_cycle(self, feedback: ConsolidationFeedback) -> None:
        self.policy.update_from_feedback(feedback)
