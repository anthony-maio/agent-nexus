"""LangGraph state schema for the Agent Nexus orchestrator.

Defines the :class:`NexusOrchestratorState` TypedDict that flows through
every node in the orchestrator graph.  Follows the LangGraph convention of
using ``Annotated[list, add]`` for accumulator fields that append across
node invocations rather than overwriting.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class NexusOrchestratorState(TypedDict):
    """State flowing through the LangGraph orchestrator pipeline.

    Fields are grouped into four categories:

    1. **Gathered state** -- sourced from :class:`~nexus.orchestrator.state.StateGatherer`.
    2. **C2 context** -- composed knowledge from Continuity Core.
    3. **Decision output** -- actions proposed/approved by the orchestrator model.
    4. **Agent results** -- accumulated outputs from tool-enabled task agents.
    5. **Control flow** -- bookkeeping for the dispatch loop.
    """

    # --- Gathered state (from StateGatherer.gather()) ---------------------
    timestamp: str
    cycle_count: int
    recent_messages: list[dict[str, Any]]
    memories: list[dict[str, Any]]
    activity: dict[str, Any] | None
    curiosity: dict[str, Any] | None
    task_results: list[dict[str, Any]]
    active_goals: str

    # --- C2 shared brain context ------------------------------------------
    c2_context: str

    # --- Orchestrator decision output -------------------------------------
    proposed_actions: list[dict[str, Any]]
    approved_actions: list[dict[str, Any]]

    # --- Agent execution (accumulators) -----------------------------------
    agent_results: Annotated[list[dict[str, Any]], add]
    tool_log: Annotated[list[dict[str, str]], add]

    # --- Control flow -----------------------------------------------------
    pending_action_index: int
    should_stop: bool
