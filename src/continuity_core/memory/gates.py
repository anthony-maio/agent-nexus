from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class Gate(Protocol):
    def __call__(self, recall: float) -> float:
        ...


@dataclass(frozen=True)
class ThresholdGate:
    theta: float = 0.5

    def __call__(self, recall: float) -> float:
        return 1.0 if recall >= self.theta else 0.0


@dataclass(frozen=True)
class BandpassGate:
    low: float = 0.3
    high: float = 0.8

    def __call__(self, recall: float) -> float:
        return 1.0 if (self.low <= recall < self.high) else 0.0


@dataclass(frozen=True)
class SmoothGate:
    theta: float = 0.5
    sharpness: float = 12.0

    def __call__(self, recall: float) -> float:
        x = self.sharpness * (recall - self.theta)
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)


@dataclass(frozen=True)
class AdaptiveGate:
    theta_base: float = 0.5
    sharpness: float = 12.0
    pressure_sensitivity: float = 0.3

    def __call__(self, recall: float, occupancy: float = 0.5) -> float:
        theta = self.theta_base + self.pressure_sensitivity * (occupancy - 0.5)
        theta = max(0.1, min(0.9, theta))
        x = self.sharpness * (recall - theta)
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
