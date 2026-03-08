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
        "search_web",
        "fetch_url",
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
    """Return bootstrap steps for an autonomous tool loop."""
    cleaned = " ".join(objective.split())
    if _looks_like_workflow(cleaned):
        return _workflow_bootstrap_steps(cleaned)
    return _research_bootstrap_steps(cleaned)


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
    step_index = int(completed_step.get("step_index", -1))
    if action == "search_web":
        top_url = _top_result_url(result)
        if not top_url:
            return []
        next_action = "navigate" if _looks_like_workflow(objective) else "fetch_url"
        return [
            StepDefinition(
                action_type=next_action,
                instruction=f"Use grounded result {top_url} for: {objective}",
            )
        ]

    if action in {"fetch_url", "navigate"}:
        next_action = "inspect" if _looks_like_workflow(objective) else "extract"
        if _has_action_after(existing_steps, step_index, next_action):
            return []
        instruction = (
            "Inspect the current page structure, controls, and relevant sections "
            f"for: {objective}"
            if next_action == "inspect"
            else f"Summarize the fetched page with citations for: {objective}"
        )
        return [StepDefinition(action_type=next_action, instruction=instruction)]

    if action == "inspect":
        if _looks_like_workflow(objective):
            steps: list[StepDefinition] = []
            if not _has_action_after(existing_steps, step_index, "extract"):
                steps.append(
                    StepDefinition(
                        action_type="extract",
                        instruction=(
                            "Summarize the workflow state, required inputs, and next "
                            f"approval point for: {objective}"
                        ),
                    )
                )
            if not _has_action_after(existing_steps, step_index, "type"):
                steps.append(
                    StepDefinition(
                        action_type="type",
                        instruction=(
                            "Enter only the minimum draft input required to continue "
                            f"the workflow for: {objective}"
                        ),
                    )
                )
            return steps
        if _has_action_after(existing_steps, step_index, "extract"):
            return []
        return [
            StepDefinition(
                action_type="extract",
                instruction=f"Summarize findings with citations for: {objective}",
            )
        ]

    if action == "type":
        if _has_action_after(existing_steps, step_index, "click"):
            return []
        return [
            StepDefinition(
                action_type="click",
                instruction=f"Click the next non-destructive control for: {objective}",
            ),
            StepDefinition(
                action_type="wait",
                instruction=f"Wait for the page state to settle for: {objective}",
            ),
            StepDefinition(
                action_type="extract",
                instruction=(
                    "Summarize the updated workflow state and the next approval "
                    f"point for: {objective}"
                ),
            ),
        ]

    if action == "extract":
        if not result.citations:
            if _count_adaptive_extracts(existing_steps) >= 2:
                return []
            if _has_action_after(existing_steps, step_index, "scroll"):
                return []
            return [
                StepDefinition(
                    action_type="scroll",
                    instruction=(
                        "Adaptive follow-up: gather additional page context because "
                        f"the last extraction returned no citations. Objective: {objective}"
                    ),
                ),
                StepDefinition(
                    action_type="extract",
                    instruction=(
                        "Adaptive follow-up: re-run extraction after scrolling and "
                        f"include citations for: {objective}"
                    ),
                ),
            ]
        if _looks_like_workflow(objective) and _has_action_before(existing_steps, step_index, "type"):
            if _has_action_after(existing_steps, step_index, "submit"):
                return []
            return [
                StepDefinition(
                    action_type="submit",
                    instruction=f"Submit the workflow only after approval for: {objective}",
                )
            ]
        if _has_action_after(existing_steps, step_index, "export"):
            return []
        return [
            StepDefinition(
                action_type="export",
                instruction=f"Prepare an exportable report artifact for: {objective}",
            )
        ]

    if action == "submit":
        if _has_action_after(existing_steps, step_index, "export"):
            return []
        return [
            StepDefinition(
                action_type="export",
                instruction=f"Prepare an exportable workflow report for: {objective}",
            )
        ]

    return []


class AdaptivePlanner(Protocol):
    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
    ) -> list[StepDefinition]:
        """Return the initial bootstrap steps for a run."""

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
    ) -> list[StepDefinition]:
        """Return proposed follow-up steps for current run context."""


class RuleAdaptivePlanner:
    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
    ) -> list[StepDefinition]:
        _ = mode
        return plan_steps_for_objective(objective)

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

    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
    ) -> list[StepDefinition]:
        for planner in self.planners:
            try:
                proposed = await planner.plan_initial_steps(objective=objective, mode=mode)
            except Exception as exc:
                log.warning("Adaptive planner failed (%s): %s", type(planner).__name__, exc)
                continue
            if proposed:
                return proposed
        return []

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
        return f"Open a browser session and navigate directly to {url}. Objective: {objective}"
    return f"Open a browser session and locate the best starting pages for: {objective}"


def _research_bootstrap_steps(objective: str) -> list[StepDefinition]:
    return [
        StepDefinition(
            action_type="search_web",
            instruction=f"Search the web for the best grounded sources for: {objective}",
        ),
    ]


def _workflow_bootstrap_steps(objective: str) -> list[StepDefinition]:
    url = _extract_url(objective)
    if not url:
        return [
            StepDefinition(
                action_type="search_web",
                instruction=f"Find the best starting page for this workflow: {objective}",
            )
        ]
    return [
        StepDefinition(
            action_type="navigate",
            instruction=f"Navigate directly to {url} and begin this workflow: {objective}",
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


def _top_result_url(result: StepExecutionResult) -> str:
    raw_results = result.metadata.get("search_results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict) and str(item.get("url", "")).strip():
                return str(item["url"])
    if result.citations:
        return result.citations[0].url
    return ""


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


def _has_action_before(
    existing_steps: list[dict[str, Any]],
    before_step_index: int,
    action_type: str,
) -> bool:
    for step in existing_steps:
        try:
            step_index = int(step.get("step_index", -1))
        except (TypeError, ValueError):
            step_index = -1
        if step_index >= before_step_index:
            continue
        if str(step.get("action_type", "")).strip().lower() == action_type:
            return True
    return False
