"""LangGraph orchestrator graph for Agent Nexus.

Builds a :class:`~langgraph.graph.StateGraph` that replaces the manual
decide-dispatch loop in :class:`~nexus.orchestrator.loop.OrchestratorLoop`
with a structured pipeline:

.. code-block:: text

    START -> gather_state -> enrich_c2 -> orchestrator_decide -> guardrails
      -> autonomy_gate -> dispatch_agent -> post_results
      -> should_continue? --yes--> dispatch_agent
                          --no---> END

All node functions are async and receive dependencies via
``functools.partial()``, following the same pattern as the MRA reference
implementation.

Usage::

    graph = build_orchestrator_graph(
        bot=bot,
        orchestrator_llm=orchestrator_llm,
        agent_llm=agent_llm,
        tools=tools,
        checkpointer=checkpointer,
    )
    result = await graph.ainvoke(initial_state, config)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import partial
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from nexus.orchestrator.graph_state import NexusOrchestratorState

log = logging.getLogger(__name__)


# =====================================================================
# Orchestrator system prompt
# =====================================================================

_ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the orchestrator for an AI agent swarm called Agent Nexus. "
    "Based on the current state, decide what tasks to dispatch to the "
    "swarm's tool-enabled task agents. Respond with a JSON array of "
    "action objects. Each action has:\n"
    '  "type": one of "research", "code", "analyze", "summarize", '
    '"classify", "extract"\n'
    '  "description": a clear, specific task description\n'
    '  "priority": "high", "medium", or "low"\n'
    '  "goal_id": (optional) ID of an existing goal this relates to\n'
    '  "new_goal": (optional) object with "title" and "description" '
    "to create a new goal\n\n"
    "IMPORTANT CONSTRAINTS:\n"
    "- Task agents have tools: query_memory, query_c2_context, "
    "query_c2_curiosity, write_c2_event, get_active_goals, "
    "get_recent_c2_events. They can query knowledge but CANNOT access "
    "files, run commands, or interact with infrastructure.\n"
    "- Do NOT create meta-goals about the swarm itself (e.g. 'summarize "
    "cycle results', 'compile status report', 'review escalation').\n"
    "- Do NOT create goals about diagnosing or fixing the swarm's own "
    "infrastructure or task agent failures.\n"
    "- Tasks should be grounded in the USER's actual work -- projects "
    "they are working on, code they are writing, topics they are "
    "researching.\n"
    "- Return an empty array [] if there is nothing useful to do. "
    "Doing nothing is BETTER than inventing busywork.\n"
    "- Do not duplicate actions already in active goals or recent results."
)


# =====================================================================
# Node functions
# =====================================================================

async def gather_state_node(
    state: NexusOrchestratorState,
    *,
    bot: Any,
) -> dict[str, Any]:
    """Gather current system state from all sources."""
    gathered = await bot.state_gatherer.gather()

    # Serialise ActivityDigest to dict if present.
    activity = gathered.get("activity")
    activity_dict = None
    if activity is not None and hasattr(activity, "summary"):
        activity_dict = {
            "summary": getattr(activity, "summary", ""),
            "recent_focus": getattr(activity, "recent_focus", ""),
            "projects": getattr(activity, "projects", []),
            "active_apps": getattr(activity, "active_apps", []),
            "is_stale": getattr(activity, "is_stale", False),
            "age_description": getattr(activity, "age_description", ""),
        }

    return {
        "timestamp": gathered.get("timestamp", ""),
        "recent_messages": gathered.get("recent_messages", []),
        "memories": gathered.get("memories", []),
        "activity": activity_dict,
        "curiosity": gathered.get("curiosity"),
        "task_results": gathered.get("task_results", []),
        "active_goals": gathered.get("active_goals", ""),
    }


async def enrich_c2_node(
    state: NexusOrchestratorState,
    *,
    bot: Any,
) -> dict[str, Any]:
    """Compose C2 context pack from recent conversation."""
    c2 = getattr(bot, "c2", None)
    if c2 is None or not c2.is_running:
        return {"c2_context": ""}

    # Build a query from recent conversation + curiosity signals.
    query_parts: list[str] = []
    for msg in state.get("recent_messages", [])[-3:]:
        content = msg.get("content", "")
        if content:
            query_parts.append(content[:200])
    curiosity = state.get("curiosity")
    if curiosity:
        suggested = curiosity.get("suggested_action", "")
        if suggested:
            query_parts.append(suggested)

    query = " ".join(query_parts)[:500] or "current tasks and priorities"

    try:
        result = await c2.get_context(query, token_budget=2048)
        if result is None:
            return {"c2_context": ""}
        chosen = result.get("chosen", [])
        if not chosen:
            return {"c2_context": ""}
        context_text = "\n".join(
            c.get("text", "")[:300] for c in chosen
        )
        return {"c2_context": context_text}
    except Exception:
        log.debug("C2 context enrichment failed.", exc_info=True)
        return {"c2_context": ""}


async def orchestrator_decide_node(
    state: NexusOrchestratorState,
    *,
    orchestrator_llm: Any,
    bot: Any,
) -> dict[str, Any]:
    """Use the dedicated orchestrator model to decide actions."""
    prompt = _build_decision_prompt(state, bot)

    try:
        response = await orchestrator_llm.ainvoke([
            SystemMessage(content=_ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        actions = _parse_actions(response.content)
        return {"proposed_actions": actions}
    except Exception:
        log.error("Orchestrator decision failed.", exc_info=True)
        return {"proposed_actions": []}


async def guardrails_node(
    state: NexusOrchestratorState,
    *,
    bot: Any,
) -> dict[str, Any]:
    """Apply entity grounding, capability filter, meta-goal filter, idle-loop."""
    from nexus.orchestrator.guardrails import check_capability, check_entity_grounding

    actions = list(state.get("proposed_actions", []))
    state_text = json.dumps(state, default=str)

    # Entity grounding.
    actions = check_entity_grounding(actions, state_text)

    # Capability filter (reject impossible tasks).
    actions = check_capability(actions)

    # Meta-goal hard filter (reuse patterns from loop.py).
    actions = [a for a in actions if not _is_meta_goal(a.get("description", ""))]

    # Idle-loop detection (stateful detector lives on the orchestrator loop).
    orch_loop = getattr(bot, "orchestrator", None)
    if orch_loop is not None:
        idle = getattr(orch_loop, "_idle_detector", None)
        if idle is not None and idle.check_cycle(actions, state):
            log.info("GUARDRAIL: Idle loop detected -- suppressing all actions.")
            actions = []

    return {"approved_actions": actions, "pending_action_index": 0}


async def autonomy_gate_node(
    state: NexusOrchestratorState,
    *,
    bot: Any,
) -> dict[str, Any]:
    """Filter actions through the autonomy gate."""
    gate = getattr(bot, "autonomy_gate", None)
    if gate is None:
        return {}  # No gate -- all actions pass through.

    approved: list[dict[str, Any]] = []
    for action in state.get("approved_actions", []):
        if gate.should_auto_execute(action):
            approved.append(action)
        elif gate.should_escalate(action):
            try:
                ok = await gate.propose_and_wait(bot, action)
                if ok:
                    approved.append(action)
            except Exception:
                log.debug("Autonomy gate escalation failed.", exc_info=True)
        # else: dropped silently.

    return {"approved_actions": approved}


async def dispatch_agent_node(
    state: NexusOrchestratorState,
    *,
    agent_llm: Any,
    tools: list,
    bot: Any,
) -> dict[str, Any]:
    """Dispatch the next pending action to a tool-enabled agent."""
    idx = state.get("pending_action_index", 0)
    actions = state.get("approved_actions", [])

    if idx >= len(actions):
        return {"should_stop": True}

    action = actions[idx]
    action_type = action.get("type", "analyze")
    description = action.get("description", "")

    log.info(
        "LangGraph dispatching action %d/%d (%s): %.100s",
        idx + 1, len(actions), action_type, description,
    )

    # Build agent messages with C2 context.
    c2_context = state.get("c2_context", "")
    system_prompt = _build_agent_system_prompt(action, c2_context)

    # Bind tools and run the ReAct loop.
    agent_with_tools = agent_llm.bind_tools(tools)
    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=description),
    ]

    agent_error = False
    try:
        result_text, tool_calls_log = await _react_loop(
            agent_with_tools, tools, messages, max_steps=3,
        )
    except Exception as exc:
        log.error("Agent dispatch failed: %s", exc, exc_info=True)
        result_text = f"Agent error: {exc}"
        tool_calls_log = []
        agent_error = True

    # Validate result via guardrails (skip if already failed).
    from nexus.orchestrator.guardrails import validate_task_output

    if agent_error:
        is_valid = False
        reason = result_text
    else:
        is_valid, reason = validate_task_output(result_text, description)

    agent_result: dict[str, Any] = {
        "action": action,
        "result": result_text if is_valid else reason,
        "success": is_valid,
        "tool_calls": tool_calls_log,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not is_valid:
        log.info(
            "GUARDRAIL: Agent result failed validation: %.100s",
            description,
        )

    return {
        "agent_results": [agent_result],
        "tool_log": tool_calls_log,
        "pending_action_index": idx + 1,
    }


async def post_results_node(
    state: NexusOrchestratorState,
    *,
    bot: Any,
) -> dict[str, Any]:
    """Post the latest agent result to #nexus, log to C2, update goals."""
    results = state.get("agent_results", [])
    if not results:
        return {}

    latest = results[-1]
    action = latest.get("action", {})
    description = action.get("description", "")
    success = latest.get("success", False)
    result_text = latest.get("result", "")

    # Post to #nexus.
    await _post_to_nexus(bot, latest)

    # Log to C2.
    c2 = getattr(bot, "c2", None)
    if c2 is not None and c2.is_running:
        try:
            await c2.write_event(
                actor="task_agent",
                intent="task_result",
                inp=description[:200],
                out=result_text[:300],
                tags=["langgraph", action.get("type", "unknown")],
            )
        except Exception:
            pass

    # Update linked goal.
    goal_id = action.get("goal_id", "")
    if goal_id:
        goal_store = getattr(bot, "goal_store", None)
        if goal_store is not None:
            try:
                snippet = result_text[:100].replace("\n", " ")
                status = "OK" if success else "FAIL"
                await goal_store.add_progress_note(
                    goal_id,
                    f"{status}: {description[:60]} -> {snippet}",
                )
            except Exception:
                log.debug("Failed to update goal %s.", goal_id)

    return {}


def should_continue(state: NexusOrchestratorState) -> str:
    """Conditional edge: more actions to dispatch?"""
    if state.get("should_stop", False):
        return "end"
    idx = state.get("pending_action_index", 0)
    actions = state.get("approved_actions", [])
    if idx >= len(actions):
        return "end"
    return "continue"


# =====================================================================
# Graph builder
# =====================================================================

def build_orchestrator_graph(
    bot: Any,
    orchestrator_llm: Any,
    agent_llm: Any,
    tools: list,
    checkpointer: Any = None,
) -> Any:
    """Build and compile the LangGraph orchestrator graph.

    Args:
        bot: The ``NexusBot`` instance.
        orchestrator_llm: ChatOpenAI for the dedicated orchestrator model.
        agent_llm: ChatOpenAI for tool-enabled task agents.
        tools: List of LangChain tools from ``build_tools()``.
        checkpointer: LangGraph checkpointer (Redis or MemorySaver).

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``ainvoke()``.
    """
    graph = StateGraph(NexusOrchestratorState)

    # Add nodes with dependencies bound via partial.
    graph.add_node("gather_state", partial(gather_state_node, bot=bot))
    graph.add_node("enrich_c2", partial(enrich_c2_node, bot=bot))
    graph.add_node(
        "orchestrator_decide",
        partial(orchestrator_decide_node, orchestrator_llm=orchestrator_llm, bot=bot),
    )
    graph.add_node("guardrails", partial(guardrails_node, bot=bot))
    graph.add_node("autonomy_gate", partial(autonomy_gate_node, bot=bot))
    graph.add_node(
        "dispatch_agent",
        partial(dispatch_agent_node, agent_llm=agent_llm, tools=tools, bot=bot),
    )
    graph.add_node("post_results", partial(post_results_node, bot=bot))

    # Edges: linear pipeline with loop-back for multi-action dispatch.
    graph.add_edge(START, "gather_state")
    graph.add_edge("gather_state", "enrich_c2")
    graph.add_edge("enrich_c2", "orchestrator_decide")
    graph.add_edge("orchestrator_decide", "guardrails")
    graph.add_edge("guardrails", "autonomy_gate")
    graph.add_edge("autonomy_gate", "dispatch_agent")
    graph.add_edge("dispatch_agent", "post_results")

    # Conditional loop: more actions pending -> dispatch_agent, else END.
    graph.add_conditional_edges(
        "post_results",
        should_continue,
        {"continue": "dispatch_agent", "end": END},
    )

    compiled = graph.compile(checkpointer=checkpointer)
    log.info("LangGraph orchestrator graph compiled (%d nodes).", len(graph.nodes))
    return compiled


# =====================================================================
# ReAct loop
# =====================================================================

async def _react_loop(
    agent_with_tools: Any,
    tools: list,
    messages: list[Any],
    max_steps: int = 3,
) -> tuple[str, list[dict[str, str]]]:
    """Run a simple ReAct loop: invoke -> tool_call -> observe -> invoke.

    Args:
        agent_with_tools: LLM with tools bound via ``bind_tools()``.
        tools: List of LangChain tool objects.
        messages: Initial message list (system + user).
        max_steps: Maximum tool-calling rounds.

    Returns:
        ``(final_text, tool_log)`` where ``tool_log`` captures each
        tool invocation for visibility in #nexus.
    """
    tool_map = {t.name: t for t in tools}
    tool_log: list[dict[str, str]] = []

    for step in range(max_steps):
        response: AIMessage = await agent_with_tools.ainvoke(messages)

        # No tool calls -- return the final text response.
        if not response.tool_calls:
            return response.content or "", tool_log

        # Execute each tool call.
        messages.append(response)
        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            log_entry: dict[str, str] = {
                "step": str(step),
                "tool": tool_name,
                "args": json.dumps(tool_args, default=str)[:200],
            }

            if tool_name in tool_map:
                try:
                    result = await tool_map[tool_name].ainvoke(tool_args)
                except Exception as exc:
                    result = f"Tool error: {exc}"
            else:
                result = f"Unknown tool: {tool_name}"

            log_entry["result_preview"] = str(result)[:100]
            tool_log.append(log_entry)

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tc["id"])
            )

    # Max steps reached -- get final response.
    response = await agent_with_tools.ainvoke(messages)
    return response.content or "", tool_log


# =====================================================================
# Helpers
# =====================================================================

def _build_decision_prompt(
    state: NexusOrchestratorState,
    bot: Any,
) -> str:
    """Build the decision prompt from graph state.

    Mirrors the structure of ``OrchestratorLoop._build_decision_prompt()``
    but works from the graph state dict instead of the raw gathered state.
    """
    parts: list[str] = [
        f"Timestamp: {state.get('timestamp', 'unknown')}",
        f"Cycle: #{state.get('cycle_count', 0)}",
        "",
        "=== Current Swarm State ===",
    ]

    # Active goals.
    active_goals = state.get("active_goals", "")
    if active_goals:
        parts.append("\n--- Active Goals ---")
        parts.append(active_goals)

    # Recent task results.
    task_results = state.get("task_results", [])
    if task_results:
        parts.append(f"\n--- Recent Task Results ({len(task_results)}) ---")
        for tr in task_results:
            status = "OK" if tr.get("success") else "FAILED"
            parts.append(
                f"  [{status}] {tr.get('type', '?')}: "
                f"{tr.get('description', '')[:100]}"
            )

    # Recent conversation.
    recent_msgs = state.get("recent_messages", [])
    if recent_msgs:
        parts.append(f"\n--- Recent Conversation ({len(recent_msgs)} messages) ---")
        for msg in recent_msgs[-5:]:
            author = msg.get("author", "unknown")
            content = msg.get("content", "")[:1500]
            parts.append(f"  [{author}]: {content}")
    else:
        parts.append("\n--- Recent Conversation ---")
        parts.append("  (no recent messages)")

    # Memories.
    memories = state.get("memories", [])
    if memories:
        parts.append(f"\n--- Relevant Memories ({len(memories)}) ---")
        for mem in memories[:3]:
            content = mem.get("content", "")[:1500]
            source = mem.get("source", "unknown")
            score = mem.get("score", 0.0)
            parts.append(f"  [{source}, relevance={score:.2f}]: {content}")

    # PiecesOS activity.
    activity = state.get("activity")
    if activity is not None:
        age = activity.get("age_description", "")
        stale = activity.get("is_stale", False)
        header = "Recent User Activity (PiecesOS)"
        if age:
            header += f" [{age}]"
        if stale:
            header += " (STALE)"
        parts.append(f"\n--- {header} ---")
        if activity.get("recent_focus"):
            parts.append(f"  Current focus: {activity['recent_focus'][:300]}")
        if activity.get("projects"):
            parts.append(f"  Active projects: {', '.join(activity['projects'])}")
        if activity.get("active_apps"):
            parts.append(
                f"  Active apps: {', '.join(activity['active_apps'][:5])}"
            )

    # C2 context.
    c2_context = state.get("c2_context", "")
    if c2_context:
        parts.append("\n--- C2 Knowledge Context ---")
        parts.append(f"  {c2_context[:800]}")

    # C2 curiosity signals.
    curiosity = state.get("curiosity")
    if curiosity:
        parts.append("\n--- Epistemic Signals (C2 Curiosity) ---")
        stress = curiosity.get("stress_level", 0)
        parts.append(f"  Stress level: {stress:.3f}")
        contradictions = curiosity.get("contradictions", [])
        if contradictions:
            parts.append(f"  Contradictions ({len(contradictions)}):")
            for c in contradictions[:3]:
                parts.append(
                    f"    - {c.get('s1', '')[:80]} vs {c.get('s2', '')[:80]}"
                )

    parts.append("\n=== End State ===")
    parts.append(
        "\nBased on this state, what tasks should be dispatched? "
        "Return a JSON array of actions (or empty array [])."
    )
    return "\n".join(parts)


def _build_agent_system_prompt(
    action: dict[str, Any],
    c2_context: str,
) -> str:
    """Build the system prompt for a tool-enabled task agent."""
    parts: list[str] = [
        "You are a task agent in Agent Nexus with access to tools. "
        "Use tools to query memory and knowledge before answering. "
        "Never fabricate data -- if a tool returns nothing, say so.",
        "",
    ]

    if c2_context:
        parts.append("## Shared Knowledge (from Continuity Core)")
        parts.append(c2_context[:1000])
        parts.append("")

    parts.append("## Your Task")
    parts.append(f"Type: {action.get('type', 'analyze')}")
    parts.append(f"Priority: {action.get('priority', 'medium')}")
    parts.append("")
    parts.append(
        "Respond with a clear, actionable answer. "
        "Keep your response under 500 words."
    )

    return "\n".join(parts)


# Meta-goal patterns (mirrors OrchestratorLoop._META_GOAL_PATTERNS).
_META_GOAL_PATTERNS: list[str] = [
    "summarize all completed goals",
    "summarize all goals",
    "compile escalation",
    "compile status report",
    "cycle completion summary",
    "cycle #",
    "status report for cycle",
    "review the compiled",
    "dispatch the compiled",
    "decommission the agent",
    "task agent failure",
    "task agent non-functional",
    "output path failure",
    "output serialization",
    "swarm cannot self-repair",
    "swarm self-diagnosis",
    "infra-level intervention",
    "infra-level review",
    "escalation message",
    # Self-diagnosis / hallucination spiral patterns
    "empty result",
    "empty extraction",
    "diagnose the bug",
    "root cause analysis",
    "root cause of the",
    "investigate the failure",
    "investigate why the",
    "trace the data flow",
    "audit the findings",
    "cross-reference the",
    "synthesize the findings",
    "analyze the task",
    "taskresult",
    "validation logic",
    "extraction code",
    "extraction tool",
    "agent cannot access",
    "agent returned empty",
    "context caching hypothesis",
    "holding pattern",
]


def _is_meta_goal(description: str) -> bool:
    """Return True if the description is self-referential busywork."""
    desc_lower = description.lower()
    return any(p in desc_lower for p in _META_GOAL_PATTERNS)


def _parse_actions(raw_response: str) -> list[dict[str, Any]]:
    """Parse the orchestrator model's response into action dicts."""
    import re

    text = raw_response.strip()

    # Strip markdown code fences.
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Strip <think> blocks.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try direct parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _validate_actions(parsed)
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON array from surrounding text.
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        try:
            parsed = json.loads(text[bracket_start:bracket_end + 1])
            if isinstance(parsed, list):
                return _validate_actions(parsed)
        except json.JSONDecodeError:
            pass

    log.warning("Could not parse orchestrator decision: %.200s", text)
    return []


def _validate_actions(
    parsed: list[Any],
    max_actions: int = 5,
) -> list[dict[str, Any]]:
    """Validate and normalise a parsed list of action dicts."""
    valid_types = {
        "research", "code", "analyze", "summarize", "classify", "extract",
    }
    valid_priorities = {"high", "medium", "low"}
    actions: list[dict[str, Any]] = []

    for item in parsed[:max_actions]:
        if not isinstance(item, dict):
            continue
        action_type = item.get("type", "analyze")
        if action_type not in valid_types:
            action_type = "analyze"
        priority = item.get("priority", "medium")
        if priority not in valid_priorities:
            priority = "medium"
        description = str(item.get("description", "")).strip()
        if not description:
            continue
        if _is_meta_goal(description):
            log.warning("GUARDRAIL: Dropping meta-goal: '%.100s'", description)
            continue

        action: dict[str, Any] = {
            "type": action_type,
            "description": description,
            "priority": priority,
        }
        goal_id = item.get("goal_id")
        if isinstance(goal_id, str) and goal_id.strip():
            action["goal_id"] = goal_id.strip()
        new_goal = item.get("new_goal")
        if isinstance(new_goal, dict):
            action["new_goal"] = new_goal

        actions.append(action)

    return actions


async def _post_to_nexus(bot: Any, agent_result: dict[str, Any]) -> None:
    """Post an agent result to #nexus with tool usage visibility."""
    router = getattr(bot, "router", None)
    if router is None or getattr(router, "nexus", None) is None:
        return

    try:
        from nexus.channels.formatter import MessageFormatter

        action = agent_result.get("action", {})
        status = "completed" if agent_result.get("success") else "FAILED"
        result_text = agent_result.get("result", "")

        body_parts: list[str] = [
            f"**Task [{status}]:** {action.get('description', '')[:300]}",
            "",
            result_text,
        ]
        body = "\n".join(body_parts)

        model_id = bot.settings.TASK_AGENT_MODEL
        embed = MessageFormatter.format_response(model_id, body)

        # Add tool usage field if tools were called.
        tool_calls = agent_result.get("tool_calls", [])
        if tool_calls:
            tool_summary = "\n".join(
                f"`{tc.get('tool', '?')}` -> "
                f"{tc.get('result_preview', '...')[:60]}"
                for tc in tool_calls
            )
            embed.add_field(
                name=f"Tools Used ({len(tool_calls)})",
                value=tool_summary[:1024],
                inline=False,
            )

        await router.nexus.send(embed=embed)
    except Exception:
        log.error("Failed to post agent result to #nexus.", exc_info=True)
