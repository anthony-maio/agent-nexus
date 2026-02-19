"""Background orchestrator: gather state, decide actions, dispatch tasks.

The orchestrator package contains three modules:

- :mod:`~nexus.orchestrator.loop` -- The main background loop that drives
  autonomous orchestrator cycles.
- :mod:`~nexus.orchestrator.state` -- State gathering from conversation,
  memory, and activity sources.
- :mod:`~nexus.orchestrator.dispatch` -- Task dispatch to LiquidAI Tier 2
  agents with result reporting.
"""

from nexus.orchestrator.dispatch import TaskDispatcher, TaskResult
from nexus.orchestrator.loop import OrchestratorLoop
from nexus.orchestrator.state import StateGatherer

__all__ = [
    "OrchestratorLoop",
    "StateGatherer",
    "TaskDispatcher",
    "TaskResult",
]
