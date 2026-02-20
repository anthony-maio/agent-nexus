from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class DecayPolicy:
    decay_rate: float = 0.95
    time_unit_sec: float = 86400.0

    def decay_factor(self, last_access: float, now: float | None = None) -> float:
        now = time.time() if now is None else now
        elapsed = max(0.0, now - last_access)
        periods = elapsed / max(1.0, self.time_unit_sec)
        return self.decay_rate ** periods

    def apply(self, salience: float, last_access: float, now: float | None = None) -> float:
        return salience * self.decay_factor(last_access, now=now)
