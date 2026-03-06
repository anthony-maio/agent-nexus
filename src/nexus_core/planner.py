"""Browser-first run planning for app-first execution."""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError

from nexus_core.models import RunMode, StepDefinition, StepExecutionResult
from nexus_core.policy import is_high_risk_action

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s)>]+", re.IGNORECASE)
_WORKFLOW_HINTS: tuple[str, ...] = (
    "fill out",
    "fill in",
    "contact form",
    "submit",
    "apply",
    "sign up",
    "signup",
    "register",
    "book",
    "reserve",
    "schedule",
    "send message",
    "login",
    "log in",
    "checkout",
    "purchase",
    "buy",
    "form",
)
_ALLOWED_FOLLOW_UP_ACTIONS: frozenset[str] = frozenset(
    {
        "navigate",
        "inspect",
        "scroll",
        "extract",
        "read",
        "click",
        "type",
        "wait",
        "write",
        "export",
        "submit",
    }
)
_BLOCKED_MODEL_ACTIONS: frozenset[str] = frozenset(
    {
        "delete",
        "purchase",
        "promote",
        "send",
    }
)
_MAX_FOLLOW_UP_STEPS = 4


def plan_steps_for_objective(objective: str) -> list[StepDefinition]:
    """Return a browser-first run plan for the given objective."""
    cleaned = " ".join(objective.split())
    if _looks_like_workflow(cleaned):
        return _workflow_steps(cleaned)
    return _research_steps(cleaned)


def plan_follow_up_steps(
    objective: str,
    completed_step: dict[str, Any],
    result: StepExecutionResult,
    existing_steps: list[dict[str, Any]],
) -> list[StepDefinition]:
    """Return dynamic follow-up steps based on execution result."""
    metadata_steps = _metadata_follow_up_steps(result)
    if metadata_steps:
        return metadata_steps

    action = str(completed_step.get("action_type", "")).strip().lower()
    if action != "extract":
        return []
    if result.citations:
        return []
    if _count_adaptive_extracts(existing_steps) >= 2:
        return []

    step_index = int(completed_step.get("step_index", -1))
    if _has_action_after(existing_steps, step_index, "scroll"):
        return []

    return [
        StepDefinition(
            action_type="scroll",
            instruction=(
                "Adaptive follow-up: gather additional page context because the last "
                f"extraction returned no citations. Objective: {objective}"
            ),
        ),
        StepDefinition(
            action_type="extract",
            instruction=(
                "Adaptive follow-up: re-run extraction after scrolling and include "
                f"citations for: {objective}"
            ),
        ),
    ]


class AdaptivePlanner(Protocol):
    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
    ) -> list[StepDefinition]:
        """Return proposed follow-up steps for current run context."""


class RuleAdaptivePlanner:
    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
    ) -> list[StepDefinition]:
        return plan_follow_up_steps(
            objective=objective,
            completed_step=completed_step,
            result=result,
            existing_steps=existing_steps,
        )


class CompositeAdaptivePlanner:
    def __init__(self, planners: list[AdaptivePlanner]) -> None:
        self.planners = planners

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
    ) -> list[StepDefinition]:
        for planner in self.planners:
            try:
                proposed = await planner.propose_follow_up(
                    objective=objective,
                    completed_step=completed_step,
                    result=result,
                    existing_steps=existing_steps,
                )
            except Exception as exc:
                log.warning("Adaptive planner failed (%s): %s", type(planner).__name__, exc)
                continue
            if proposed:
                return proposed
        return []


def apply_follow_up_policy(
    steps: list[StepDefinition],
    mode: RunMode,
) -> list[StepDefinition]:
    """Enforce hard policy checks for follow-up steps before insertion."""
    sanitized: list[StepDefinition] = []
    for step in steps:
        action = step.action_type.strip().lower()
        instruction = step.instruction.strip()
        if not instruction:
            continue
        if action not in _ALLOWED_FOLLOW_UP_ACTIONS:
            continue
        if action in _BLOCKED_MODEL_ACTIONS:
            continue
        if mode != RunMode.AUTOPILOT and action == "write":
            # Keep write paths explicit in v1 via type/click/submit+approval.
            continue
        if mode == RunMode.MANUAL and is_high_risk_action(action, instruction):
            # Manual mode should not add extra risky actions autonomously.
            continue
        sanitized.append(StepDefinition(action_type=action, instruction=instruction))
        if len(sanitized) >= _MAX_FOLLOW_UP_STEPS:
            break
    return sanitized


def _looks_like_workflow(objective: str) -> bool:
    lowered = objective.lower()
    return any(hint in lowered for hint in _WORKFLOW_HINTS)


def _extract_url(objective: str) -> str:
    match = _URL_RE.search(objective)
    return match.group(0) if match else ""


def _navigation_instruction(objective: str) -> str:
    url = _extract_url(objective)
    if url:
        return (
            f"Open a browser session and navigate directly to {url}. "
            f"Objective: {objective}"
        )
    return f"Open a browser session and locate the best starting pages for: {objective}"


def _research_steps(objective: str) -> list[StepDefinition]:
    return [
        StepDefinition(
            action_type="navigate",
            instruction=_navigation_instruction(objective),
        ),
        StepDefinition(
            action_type="inspect",
            instruction=(
                "Inspect the current page structure, candidate sources, and "
                f"relevant sections for: {objective}"
            ),
        ),
        StepDefinition(
            action_type="scroll",
            instruction=(
                "Scroll through relevant content and gather more evidence for: "
                f"{objective}"
            ),
        ),
        StepDefinition(
            action_type="extract",
            instruction=(
                "Summarize findings with citations and call out remaining open "
                f"questions for: {objective}"
            ),
        ),
        StepDefinition(
            action_type="export",
            instruction=(
                "Prepare an exportable report artifact for promotion into the "
                f"canonical workspace. Objective: {objective}"
            ),
        ),
    ]


def _workflow_steps(objective: str) -> list[StepDefinition]:
    return [
        StepDefinition(
            action_type="navigate",
            instruction=_navigation_instruction(objective),
        ),
        StepDefinition(
            action_type="inspect",
            instruction=(
                "Inspect the page layout, required fields, and safe next controls "
                f"before any write action for: {objective}"
            ),
        ),
        StepDefinition(
            action_type="extract",
            instruction=(
                "Summarize the workflow state, required inputs, and upcoming "
                f"approval points with citations for: {objective}"
            ),
        ),
        StepDefinition(
            action_type="type",
            instruction=(
                "Enter only the minimum draft input required to continue this "
                f"workflow without final submission for: {objective}"
            ),
        ),
        StepDefinition(
            action_type="click",
            instruction=(
                "Click the next non-destructive control required to continue "
                f"the workflow for: {objective}"
            ),
        ),
        StepDefinition(
            action_type="wait",
            instruction=f"Wait for the page state to settle and verify the result for: {objective}",
        ),
        StepDefinition(
            action_type="submit",
            instruction=f"Submit the workflow only after approval for: {objective}",
        ),
        StepDefinition(
            action_type="export",
            instruction=(
                "Prepare an exportable workflow report artifact for promotion into "
                f"the canonical workspace. Objective: {objective}"
            ),
        ),
    ]


def _metadata_follow_up_steps(result: StepExecutionResult) -> list[StepDefinition]:
    raw = result.metadata.get("next_steps")
    if not isinstance(raw, list):
        return []
    planned: list[StepDefinition] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            planned.append(
                StepDefinition(
                    action_type=str(item.get("action_type", "")),
                    instruction=str(item.get("instruction", "")),
                )
            )
        except ValidationError:
            continue
    return planned


def _count_adaptive_extracts(existing_steps: list[dict[str, Any]]) -> int:
    count = 0
    for step in existing_steps:
        if str(step.get("action_type", "")).strip().lower() != "extract":
            continue
        instruction = str(step.get("instruction", ""))
        if instruction.startswith("Adaptive follow-up:"):
            count += 1
    return count


def _has_action_after(
    existing_steps: list[dict[str, Any]],
    after_step_index: int,
    action_type: str,
) -> bool:
    for step in existing_steps:
        try:
            step_index = int(step.get("step_index", -1))
        except (TypeError, ValueError):
            step_index = -1
        if step_index <= after_step_index:
            continue
        if str(step.get("action_type", "")).strip().lower() == action_type:
            return True
    return False
