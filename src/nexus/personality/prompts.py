"""System prompts for each model in the Agent Nexus swarm.

These prompts shape how each model behaves inside the Discord conversation.
Every swarm member receives a base prompt injected with its identity (name,
role, personality) and a roster of the other active members so it can
address them by name.

Usage::

    from nexus.personality.prompts import build_system_prompt

    prompt = build_system_prompt(
        model_id="minimax/minimax-m2.5",
        swarm_model_ids=["minimax/minimax-m2.5", "z-ai/glm-5"],
    )
"""

from __future__ import annotations

from nexus.personality.identities import get_identity


# ---------------------------------------------------------------------------
# Base prompt -- injected into every Tier-1 / Tier-1-premium model
# ---------------------------------------------------------------------------

SWARM_BASE_PROMPT = """You are {name}, a member of Agent Nexus - a multi-model AI swarm that collaborates through Discord.

Your role: {role}
Your personality: {personality}

## Context
- You are in a shared channel (#nexus) with other AI models. You can see their messages.
- Messages from other models appear as "[emoji name]: content"
- A human user interacts through #human channel. Their messages are forwarded to you.
- Be concise. Discord messages have a 2000 character limit.
- When you disagree with another model, say so clearly and explain why.
- When you agree, build on their ideas rather than repeating them.
- If asked to vote on a decision, respond with DECISION/CONFIDENCE/REASONING format.

## Your Swarm Members
{swarm_roster}

## Guidelines
- Stay in character as {name} the {role}.
- Keep responses focused and actionable.
- If you don't know something, say so - don't fabricate.
- Reference other models by name when responding to their ideas.
"""


def build_system_prompt(model_id: str, swarm_model_ids: list[str]) -> str:
    """Build the full system prompt for a swarm model.

    The prompt includes the model's own identity details and a roster
    describing every *other* model currently active in the swarm so it
    can collaborate effectively.

    Args:
        model_id: The model's OpenRouter ID (e.g. ``"minimax/minimax-m2.5"``).
        swarm_model_ids: All active swarm model IDs, including *model_id*
            itself.  The roster will list everyone except the target model.

    Returns:
        A fully-formatted system prompt string ready to be used as the
        ``system`` message in an OpenAI-compatible chat request.
    """
    identity = get_identity(model_id)

    # Build roster of other swarm members
    roster_lines: list[str] = []
    for mid in swarm_model_ids:
        if mid == model_id:
            continue
        other = get_identity(mid)
        roster_lines.append(
            f"- {other.emoji} {other.name} ({other.role}): {other.personality}"
        )

    roster = (
        "\n".join(roster_lines)
        if roster_lines
        else "You are the only model in the swarm."
    )

    def _safe(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    return SWARM_BASE_PROMPT.format(
        name=_safe(identity.name),
        role=_safe(identity.role),
        personality=_safe(identity.personality),
        swarm_roster=_safe(roster),
    )


# ---------------------------------------------------------------------------
# Task agent prompt -- lightweight Tier-2 models (LiquidAI routing/reasoning)
# ---------------------------------------------------------------------------

TASK_AGENT_PROMPT = """You are a task agent in the Agent Nexus system. You execute specific tasks and return results.

Your job: {task_description}

Be concise and actionable. Return only the result, no preamble.
"""


def build_task_prompt(task_description: str) -> str:
    """Build a system prompt for a Tier-2 task agent.

    Task agents are lightweight models (e.g. LiquidAI LFM) that handle
    routing, quick reasoning, or other narrowly-scoped jobs dispatched by
    the orchestrator.

    Args:
        task_description: A plain-English description of the task the agent
            should perform.

    Returns:
        A formatted system prompt string.
    """
    # Note: .format() does NOT re-interpret braces inside substitution
    # values, so escaping the task_description is unnecessary and would
    # produce doubled braces in the LLM prompt.
    return TASK_AGENT_PROMPT.format(task_description=task_description)
