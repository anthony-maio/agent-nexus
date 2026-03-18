"""Browser-first run planning for app-first execution."""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError

from nexus_core.models import RunMode, StepDefinition, StepExecutionResult
from nexus_core.policy import is_high_risk_action

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://[^\s)>]+", re.IGNORECASE)
_WORKSPACE_PATH_RE = re.compile(
    r"(?P<path>(?:workspace[\\/])?[A-Za-z0-9_.\\/-]+\.(?:txt|md|markdown|json|csv|tsv|yaml|yml|xml|html|log|py|go|js|jsx|ts|tsx))",
    re.IGNORECASE,
)
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
_API_HINTS: tuple[str, ...] = (
    "api",
    "endpoint",
    "json response",
    "rest",
    "graphql",
    "webhook",
    "http request",
)
_CODE_ACTION_HINTS: tuple[str, ...] = (
    "implement",
    "fix",
    "debug",
    "refactor",
    "patch",
    "modify",
    "edit",
    "update",
    "rename",
    "add test",
    "add tests",
    "write test",
    "write tests",
)
_CODE_TARGET_HINTS: tuple[str, ...] = (
    "repo",
    "repository",
    "codebase",
    "project",
    "workspace",
    "function",
    "class",
    "module",
    "handler",
    "service",
    "endpoint",
    "test",
    "tests",
    "bug",
)
_REPORT_HINTS: tuple[str, ...] = (
    "report",
    "brief",
    "memo",
    "writeup",
    "write-up",
    "document",
)
_CHART_HINTS: tuple[str, ...] = (
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualise",
    "dashboard",
)
_IMAGE_HINTS: tuple[str, ...] = (
    "image",
    "hero image",
    "illustration",
    "graphic",
    "poster",
    "cover art",
    "visual",
)
_ALLOWED_FOLLOW_UP_ACTIONS: frozenset[str] = frozenset(
    {
        "search_web",
        "fetch_url",
        "call_api",
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "execute_code",
        "delegate",
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
        "generate_report",
        "generate_chart",
        "generate_image",
        "submit",
    }
)
_ALLOWED_INITIAL_ACTIONS: frozenset[str] = frozenset(
    {
        "search_web",
        "fetch_url",
        "call_api",
        "navigate",
        "inspect",
        "read",
        "list_files",
        "read_file",
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


def annotate_planner_steps(
    steps: list[StepDefinition],
    *,
    planner_source: str,
    planner_phase: str,
) -> list[StepDefinition]:
    """Attach planner provenance metadata to each step."""

    annotated: list[StepDefinition] = []
    for step in steps:
        metadata = dict(step.metadata)
        metadata["planner_source"] = planner_source
        metadata["planner_phase"] = planner_phase
        annotated.append(
            StepDefinition(
                action_type=step.action_type,
                instruction=step.instruction,
                metadata=metadata,
            )
        )
    return annotated


def annotate_planner_fallback(
    steps: list[StepDefinition],
    *,
    fallback_reason: str,
) -> list[StepDefinition]:
    """Attach fallback metadata to planner-produced steps."""

    annotated: list[StepDefinition] = []
    for step in steps:
        metadata = dict(step.metadata)
        metadata["planner_fallback_reason"] = fallback_reason
        annotated.append(
            StepDefinition(
                action_type=step.action_type,
                instruction=step.instruction,
                metadata=metadata,
            )
        )
    return annotated


def plan_steps_for_objective(
    objective: str,
    skill_context: list[dict[str, Any]] | None = None,
) -> list[StepDefinition]:
    """Return bootstrap steps for an autonomous tool loop."""
    cleaned = " ".join(objective.split())
    if not _extract_url(cleaned):
        workspace_path = _workspace_path_from_objective(cleaned)
        if workspace_path:
            return [
                StepDefinition(
                    action_type="read_file",
                    instruction=_workspace_instruction(workspace_path),
                )
            ]
        if _looks_like_code_task(cleaned):
            return [
                StepDefinition(
                    action_type="list_files",
                    instruction=_workspace_listing_instruction(),
                )
            ]
        preferred_action = _preferred_initial_action_from_skills(skill_context)
        if preferred_action == "read_file":
            return [
                StepDefinition(
                    action_type="list_files",
                    instruction=_workspace_listing_instruction(),
                )
            ]
        if preferred_action == "list_files":
            return [
                StepDefinition(
                    action_type="list_files",
                    instruction=_workspace_listing_instruction(),
                )
            ]
        if preferred_action == "call_api":
            return [
                StepDefinition(
                    action_type="call_api",
                    instruction=_api_instruction(cleaned),
                )
            ]
        if preferred_action == "navigate":
            return _workflow_bootstrap_steps(cleaned)
        if preferred_action == "search_web":
            return _research_bootstrap_steps(cleaned)
    if _looks_like_workflow(cleaned):
        return _workflow_bootstrap_steps(cleaned)
    return _research_bootstrap_steps(cleaned)


def plan_follow_up_steps(
    objective: str,
    completed_step: dict[str, Any],
    result: StepExecutionResult,
    existing_steps: list[dict[str, Any]],
    skill_context: list[dict[str, Any]] | None = None,
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
        next_action = _preferred_follow_up_action_from_skills(
            skill_context,
            allowed={"navigate", "fetch_url"},
        )
        if not next_action:
            next_action = "navigate" if _looks_like_workflow(objective) else "fetch_url"
        return [
            StepDefinition(
                action_type=next_action,
                instruction=f"Use grounded result {top_url} for: {objective}",
            )
        ]

    if action == "list_files":
        next_path = _preferred_workspace_path(objective, result, metadata_key="files")
        if not next_path or _has_action_after(existing_steps, step_index, "read_file"):
            return []
        return [
            StepDefinition(
                action_type="read_file",
                instruction=_workspace_instruction(next_path),
            )
        ]

    if action == "read_file":
        if _looks_like_code_task(objective):
            completed_metadata = _step_metadata(completed_step)
            if completed_metadata.get("code_follow_up") == "failed_test_diagnostic":
                if _has_action_after(existing_steps, step_index, "extract"):
                    return []
                return [
                    StepDefinition(
                        action_type="extract",
                        instruction=(
                            "Summarize the failing code path, likely fix location, and "
                            f"next code change for: {objective}"
                        ),
                    )
                ]
            command = _preferred_code_execution_command(result, objective=objective)
            if command and not _has_action_after(existing_steps, step_index, "execute_code"):
                return [
                    StepDefinition(
                        action_type="execute_code",
                        instruction=json.dumps({"command": command}),
                    )
                ]
            if _has_action_after(existing_steps, step_index, "execute_code"):
                return []
        if _has_action_after(existing_steps, step_index, "extract"):
            return []
        return [
            StepDefinition(
                action_type="extract",
                instruction=f"Summarize the file contents and key evidence for: {objective}",
            )
        ]

    if action in {"write_file", "edit_file"}:
        next_path = _preferred_workspace_path(objective, result, metadata_key="file_path")
        if not next_path or _has_action_after(existing_steps, step_index, "read_file"):
            return []
        return [
            StepDefinition(
                action_type="read_file",
                instruction=_workspace_instruction(next_path),
            )
        ]

    if action == "execute_code":
        diagnostic_path = _diagnostic_workspace_path(result)
        if diagnostic_path and not _has_action_after(existing_steps, step_index, "read_file"):
            return [
                StepDefinition(
                    action_type="read_file",
                    instruction=_workspace_instruction(diagnostic_path),
                    metadata={"code_follow_up": "failed_test_diagnostic"},
                )
            ]
        next_path = _preferred_workspace_path(objective, result, metadata_key="touched_files")
        if next_path and not _has_action_after(existing_steps, step_index, "read_file"):
            return [
                StepDefinition(
                    action_type="read_file",
                    instruction=_workspace_instruction(next_path),
                )
            ]
        if _has_action_after(existing_steps, step_index, "extract"):
            return []
        return [
            StepDefinition(
                action_type="extract",
                instruction=f"Summarize the code execution result for: {objective}",
            )
        ]

    if action in {"fetch_url", "navigate", "call_api"}:
        next_action = _preferred_follow_up_action_from_skills(
            skill_context,
            allowed={"inspect", "extract"},
        )
        if not next_action:
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
            input_hints = _input_field_hints(result, max_items=3)
            if input_hints and not _has_action_after(existing_steps, step_index, "type"):
                field_text = ", ".join(input_hints)
                return [
                    StepDefinition(
                        action_type="type",
                        instruction=(
                            "Enter only the minimum draft input required in the grounded "
                            f"fields ({field_text}) to continue the workflow for: {objective}"
                        ),
                    )
                ]
            if not _has_action_after(existing_steps, step_index, "extract"):
                return [
                    StepDefinition(
                        action_type="extract",
                        instruction=(
                            "Summarize the workflow state, required inputs, and next "
                            f"approval point for: {objective}"
                        ),
                    )
                ]
            return []
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
        button_hints = _button_hints(result, max_items=1)
        button_instruction = (
            f"Click the grounded `{button_hints[0]}` control to continue for: {objective}"
            if button_hints
            else f"Click the next non-destructive control for: {objective}"
        )
        return [
            StepDefinition(
                action_type="click",
                instruction=button_instruction,
            ),
        ]

    if action == "click":
        if _has_action_after(existing_steps, step_index, "wait"):
            return []
        return [
            StepDefinition(
                action_type="wait",
                instruction=f"Wait for the page state to settle for: {objective}",
            )
        ]

    if action == "wait":
        if _has_action_after(existing_steps, step_index, "extract"):
            return []
        return [
            StepDefinition(
                action_type="extract",
                instruction=(
                    "Summarize the updated workflow state and the next approval "
                    f"point for: {objective}"
                ),
            )
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
                )
            ]
        if _looks_like_workflow(objective) and _has_action_before(existing_steps, step_index, "type"):
            if _has_action_after(existing_steps, step_index, "submit"):
                return []
            button_hints = _button_hints(result, max_items=1)
            instruction = (
                f"Submit the workflow via the grounded `{button_hints[0]}` control only after approval for: {objective}"
                if button_hints
                else f"Submit the workflow only after approval for: {objective}"
            )
            return [
                StepDefinition(
                    action_type="submit",
                    instruction=instruction,
                )
            ]
        preferred_artifact_action = _preferred_follow_up_action_from_skills(
            skill_context,
            allowed={"generate_report", "generate_chart", "generate_image", "export"},
        )
        if preferred_artifact_action == "generate_chart":
            chart_payload = _chart_instruction_payload(objective, result)
            if (
                chart_payload
                and not _has_action_after(existing_steps, step_index, "generate_chart")
            ):
                return [
                    StepDefinition(
                        action_type="generate_chart",
                        instruction=json.dumps(chart_payload),
                    )
                ]
        if preferred_artifact_action == "generate_image" and not _has_action_after(
            existing_steps, step_index, "generate_image"
        ):
            return [
                StepDefinition(
                    action_type="generate_image",
                    instruction=_image_instruction_payload(objective, result),
                )
            ]
        if preferred_artifact_action == "generate_report" and not _has_action_after(
            existing_steps, step_index, "generate_report"
        ):
            return [
                StepDefinition(
                    action_type="generate_report",
                    instruction=json.dumps(_report_instruction_payload(objective, result)),
                )
            ]
        if preferred_artifact_action == "export" and not _has_action_after(
            existing_steps, step_index, "export"
        ):
            return [
                StepDefinition(
                    action_type="export",
                    instruction=f"Export the grounded findings for: {objective}",
                )
            ]
        if _wants_chart_artifact(objective):
            chart_payload = _chart_instruction_payload(objective, result)
            if (
                chart_payload
                and not _has_action_after(existing_steps, step_index, "generate_chart")
                and not _has_action_after(existing_steps, step_index, "export")
            ):
                return [
                    StepDefinition(
                        action_type="generate_chart",
                        instruction=json.dumps(chart_payload),
                    )
                ]
        if _wants_image_artifact(objective):
            if _has_action_after(existing_steps, step_index, "generate_image") or _has_action_after(
                existing_steps, step_index, "export"
            ):
                return []
            return [
                StepDefinition(
                    action_type="generate_image",
                    instruction=json.dumps(_image_instruction_payload(objective, result)),
                )
            ]
        if _wants_report_artifact(objective) and _supports_explicit_report(
            existing_steps, step_index, result
        ):
            if _has_action_after(existing_steps, step_index, "generate_report") or _has_action_after(
                existing_steps, step_index, "export"
            ):
                return []
            return [
                StepDefinition(
                    action_type="generate_report",
                    instruction=json.dumps(_report_instruction_payload(objective, result)),
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

    if action == "scroll":
        if _has_action_after(existing_steps, step_index, "extract"):
            return []
        return [
            StepDefinition(
                action_type="extract",
                instruction=(
                    "Adaptive follow-up: re-run extraction after scrolling and "
                    f"include citations for: {objective}"
                ),
            )
        ]

    if action == "submit":
        if _wants_report_artifact(objective) and _supports_explicit_report(
            existing_steps, step_index, result
        ):
            if _has_action_after(existing_steps, step_index, "generate_report") or _has_action_after(
                existing_steps, step_index, "export"
            ):
                return []
            return [
                StepDefinition(
                    action_type="generate_report",
                    instruction=json.dumps(_report_instruction_payload(objective, result)),
                )
            ]
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
    async def plan_next_steps(
        self,
        objective: str,
        mode: RunMode,
        existing_steps: list[dict[str, Any]],
        completed_step: dict[str, Any] | None = None,
        result: StepExecutionResult | None = None,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        """Return the next steps for either bootstrap or follow-up planning."""

    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        """Return the initial bootstrap steps for a run."""

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        """Return proposed follow-up steps for current run context."""


async def request_next_steps(
    planner: Any,
    *,
    objective: str,
    mode: RunMode,
    existing_steps: list[dict[str, Any]],
    completed_step: dict[str, Any] | None = None,
    result: StepExecutionResult | None = None,
    skill_context: list[dict[str, str]] | None = None,
) -> list[StepDefinition]:
    """Dispatch to the planner's shared next-step contract when available."""

    if hasattr(planner, "plan_next_steps"):
        return await _call_planner_method(
            planner.plan_next_steps,
            objective=objective,
            mode=mode,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
        )
    if completed_step is None or result is None:
        return await _call_planner_method(
            planner.plan_initial_steps,
            objective=objective,
            mode=mode,
            skill_context=skill_context,
        )
    return await _call_planner_method(
        planner.propose_follow_up,
        objective=objective,
        completed_step=completed_step,
        result=result,
        existing_steps=existing_steps,
        skill_context=skill_context,
    )


async def _call_planner_method(method: Any, **kwargs: Any) -> list[StepDefinition]:
    signature = inspect.signature(method)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return await method(**kwargs)
    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return await method(**filtered_kwargs)


class RuleAdaptivePlanner:
    async def plan_next_steps(
        self,
        objective: str,
        mode: RunMode,
        existing_steps: list[dict[str, Any]],
        completed_step: dict[str, Any] | None = None,
        result: StepExecutionResult | None = None,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        _ = mode
        if completed_step is None or result is None:
            return annotate_planner_steps(
                plan_steps_for_objective(objective, skill_context=skill_context),
                planner_source="rule",
                planner_phase="initial",
            )
        return annotate_planner_steps(
            plan_follow_up_steps(
                objective=objective,
                completed_step=completed_step,
                result=result,
                existing_steps=existing_steps,
            ),
            planner_source="rule",
            planner_phase="follow_up",
        )

    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=mode,
            existing_steps=[],
            skill_context=skill_context,
        )

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=RunMode.MANUAL,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
        )


class CompositeAdaptivePlanner:
    def __init__(self, planners: list[AdaptivePlanner]) -> None:
        self.planners = planners

    async def plan_next_steps(
        self,
        objective: str,
        mode: RunMode,
        existing_steps: list[dict[str, Any]],
        completed_step: dict[str, Any] | None = None,
        result: StepExecutionResult | None = None,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        fallback_reason = ""
        for planner in self.planners:
            try:
                proposed = await request_next_steps(
                    planner,
                    objective=objective,
                    mode=mode,
                    existing_steps=existing_steps,
                    completed_step=completed_step,
                    result=result,
                    skill_context=skill_context,
                )
            except Exception as exc:
                log.warning("Adaptive planner failed (%s): %s", type(planner).__name__, exc)
                fallback_reason = "planner_error"
                continue
            if proposed:
                if fallback_reason:
                    return annotate_planner_fallback(
                        proposed,
                        fallback_reason=fallback_reason,
                    )
                return proposed
            fallback_reason = "no_steps"
        return []

    async def plan_initial_steps(
        self,
        objective: str,
        mode: RunMode,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=mode,
            existing_steps=[],
            skill_context=skill_context,
        )

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, Any],
        result: StepExecutionResult,
        existing_steps: list[dict[str, Any]],
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        return await self.plan_next_steps(
            objective=objective,
            mode=RunMode.MANUAL,
            existing_steps=existing_steps,
            completed_step=completed_step,
            result=result,
            skill_context=skill_context,
        )


def apply_initial_plan_policy(
    steps: list[StepDefinition],
    mode: RunMode,
) -> list[StepDefinition]:
    """Enforce a safe bootstrap surface for model-proposed initial steps."""

    sanitized: list[StepDefinition] = []
    for step in steps:
        action = step.action_type.strip().lower()
        instruction = step.instruction.strip()
        if not instruction:
            continue
        if action not in _ALLOWED_INITIAL_ACTIONS:
            continue
        if action in _BLOCKED_MODEL_ACTIONS:
            continue
        if mode == RunMode.MANUAL and is_high_risk_action(action, instruction):
            continue
        sanitized.append(
            StepDefinition(
                action_type=action,
                instruction=instruction,
                metadata=dict(step.metadata),
            )
        )
        break
    return sanitized


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
        sanitized.append(
            StepDefinition(
                action_type=action,
                instruction=instruction,
                metadata=dict(step.metadata),
            )
        )
        if len(sanitized) >= _MAX_FOLLOW_UP_STEPS:
            break
    return sanitized


def _looks_like_workflow(objective: str) -> bool:
    lowered = objective.lower()
    return any(hint in lowered for hint in _WORKFLOW_HINTS)


def _wants_report_artifact(objective: str) -> bool:
    lowered = objective.lower()
    return any(hint in lowered for hint in _REPORT_HINTS)


def _wants_chart_artifact(objective: str) -> bool:
    lowered = objective.lower()
    return any(hint in lowered for hint in _CHART_HINTS)


def _looks_like_api_objective(objective: str) -> bool:
    lowered = objective.lower()
    return bool(_extract_url(objective)) and any(hint in lowered for hint in _API_HINTS)


def _wants_image_artifact(objective: str) -> bool:
    lowered = objective.lower()
    return any(hint in lowered for hint in _IMAGE_HINTS)


def _extract_url(objective: str) -> str:
    match = _URL_RE.search(objective)
    return match.group(0) if match else ""


def _navigation_instruction(objective: str) -> str:
    url = _extract_url(objective)
    if url:
        return f"Open a browser session and navigate directly to {url}. Objective: {objective}"
    return f"Open a browser session and locate the best starting pages for: {objective}"


def _api_instruction(objective: str) -> str:
    payload: dict[str, Any] = {
        "url": _extract_url(objective),
        "method": "GET",
    }
    if "json" in objective.lower():
        payload["headers"] = {"Accept": "application/json"}
    return json.dumps(payload)


def _workspace_path_from_objective(objective: str) -> str:
    match = _WORKSPACE_PATH_RE.search(objective)
    if not match:
        return ""
    return _normalize_workspace_path(match.group("path"))


def _workspace_instruction(path: str) -> str:
    return json.dumps({"path": path})


def _workspace_listing_instruction(path: str = ".") -> str:
    return json.dumps({"path": path})


def _preferred_initial_action_from_skills(
    skill_context: list[dict[str, Any]] | None,
) -> str:
    if not isinstance(skill_context, list):
        return ""
    for skill in skill_context:
        if not isinstance(skill, dict):
            continue
        raw_actions = skill.get("preferred_initial_actions")
        if not isinstance(raw_actions, list):
            continue
        for raw_action in raw_actions:
            action = str(raw_action).strip().lower()
            if action in _ALLOWED_INITIAL_ACTIONS:
                return action
    return ""


def _preferred_follow_up_action_from_skills(
    skill_context: list[dict[str, Any]] | None,
    *,
    allowed: set[str] | frozenset[str] | None = None,
) -> str:
    if not isinstance(skill_context, list):
        return ""
    normalized_allowed = {item.strip().lower() for item in (allowed or set()) if item}
    for skill in skill_context:
        if not isinstance(skill, dict):
            continue
        raw_actions = skill.get("preferred_follow_up_actions")
        if not isinstance(raw_actions, list):
            continue
        for raw_action in raw_actions:
            action = str(raw_action).strip().lower()
            if not action:
                continue
            if normalized_allowed and action not in normalized_allowed:
                continue
            return action
    return ""


def _research_bootstrap_steps(objective: str) -> list[StepDefinition]:
    if _looks_like_api_objective(objective):
        return [
            StepDefinition(
                action_type="call_api",
                instruction=_api_instruction(objective),
            ),
        ]
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


def _page_affordances(result: StepExecutionResult) -> dict[str, Any]:
    raw = result.metadata.get("page_affordances")
    return raw if isinstance(raw, dict) else {}


def _input_field_hints(result: StepExecutionResult, *, max_items: int) -> list[str]:
    raw_fields = _page_affordances(result).get("input_fields")
    if not isinstance(raw_fields, list):
        return []
    hints: list[str] = []
    for item in raw_fields:
        if not isinstance(item, dict):
            continue
        label = _normalize_affordance_hint(
            item.get("label")
            or item.get("name")
            or item.get("placeholder")
            or item.get("type")
            or item.get("tag")
        )
        if not label or label in hints:
            continue
        hints.append(label)
        if len(hints) >= max(1, max_items):
            break
    return hints


def _button_hints(result: StepExecutionResult, *, max_items: int) -> list[str]:
    raw_buttons = _page_affordances(result).get("buttons")
    if not isinstance(raw_buttons, list):
        return []
    hints: list[str] = []
    for item in raw_buttons:
        if not isinstance(item, dict):
            continue
        label = _normalize_affordance_hint(
            item.get("text") or item.get("label") or item.get("name") or item.get("type")
        )
        if not label or label in hints:
            continue
        hints.append(label)
        if len(hints) >= max(1, max_items):
            break
    return hints


def _normalize_affordance_hint(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _count_adaptive_extracts(existing_steps: list[dict[str, Any]]) -> int:
    count = 0
    for step in existing_steps:
        if str(step.get("action_type", "")).strip().lower() != "extract":
            continue
        instruction = str(step.get("instruction", ""))
        if instruction.startswith("Adaptive follow-up:"):
            count += 1
    return count


def _report_instruction_payload(
    objective: str,
    result: StepExecutionResult,
) -> dict[str, Any]:
    slug = _artifact_slug(objective, fallback="report")
    payload: dict[str, Any] = {
        "path": f"reports/{slug}.md",
        "title": _artifact_title(objective, fallback="Grounded Report"),
        "objective": objective,
    }
    if result.citations:
        payload["sources"] = [
            {"url": citation.url, "title": citation.title, "snippet": citation.snippet}
            for citation in result.citations[:5]
        ]
    return payload


def _chart_instruction_payload(
    objective: str,
    result: StepExecutionResult,
) -> dict[str, Any] | None:
    raw_data = result.metadata.get("chart_data")
    if not isinstance(raw_data, list):
        return None
    data = [item for item in raw_data if isinstance(item, dict)]
    if not data:
        return None
    x_key, y_key = _infer_chart_keys(data)
    if not x_key or not y_key:
        return None
    slug = _artifact_slug(objective, fallback="chart")
    payload: dict[str, Any] = {
        "path": f"charts/{slug}.html",
        "title": str(result.metadata.get("chart_title") or _artifact_title(objective, fallback="Grounded Chart")),
        "chart_type": str(result.metadata.get("chart_type") or "bar"),
        "x_key": x_key,
        "y_key": y_key,
        "data": data[:12],
    }
    if result.citations:
        payload["sources"] = [
            {"url": citation.url, "title": citation.title, "snippet": citation.snippet}
            for citation in result.citations[:5]
        ]
    return payload


def _image_instruction_payload(
    objective: str,
    result: StepExecutionResult,
) -> dict[str, Any]:
    slug = _artifact_slug(objective, fallback="image")
    payload: dict[str, Any] = {
        "path": f"images/{slug}.svg",
        "title": _artifact_title(objective, fallback="Generated Image"),
        "prompt": objective,
        "objective": objective,
    }
    if result.citations:
        payload["sources"] = [
            {"url": citation.url, "title": citation.title, "snippet": citation.snippet}
            for citation in result.citations[:5]
        ]
    return payload


def _artifact_slug(objective: str, *, fallback: str) -> str:
    parts = re.findall(r"[a-z0-9]+", objective.lower())
    if not parts:
        return fallback
    return "-".join(parts[:6])


def _artifact_title(objective: str, *, fallback: str) -> str:
    compact = " ".join(objective.split()).strip()
    if not compact:
        return fallback
    return compact[:80]


def _infer_chart_keys(data: list[dict[str, Any]]) -> tuple[str, str]:
    sample = data[0] if data else {}
    x_key = ""
    y_key = ""
    for key, value in sample.items():
        if not x_key and not isinstance(value, (int, float)):
            x_key = str(key)
        if not y_key and isinstance(value, (int, float)):
            y_key = str(key)
    return x_key, y_key


def _has_grounded_report_context(result: StepExecutionResult) -> bool:
    if result.citations:
        return True
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    for key in ("current_url", "page_title", "page_excerpt", "search_results"):
        value = metadata.get(key)
        if value:
            return True
    return False


def _supports_explicit_report(
    existing_steps: list[dict[str, Any]],
    step_index: int,
    result: StepExecutionResult,
) -> bool:
    if not _has_grounded_report_context(result):
        return False
    browser_actions = {"search_web", "fetch_url", "navigate", "inspect", "read", "scroll", "submit"}
    return any(_has_action_before(existing_steps, step_index, action) for action in browser_actions)


def _top_result_url(result: StepExecutionResult) -> str:
    raw_results = result.metadata.get("search_results")
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict) and str(item.get("url", "")).strip():
                return str(item["url"])
    if result.citations:
        return result.citations[0].url
    return ""


def _preferred_workspace_path(
    objective: str,
    result: StepExecutionResult,
    *,
    metadata_key: str,
) -> str:
    objective_path = _workspace_path_from_objective(objective)
    if objective_path:
        return objective_path
    raw_value = result.metadata.get(metadata_key)
    if isinstance(raw_value, str) and raw_value.strip():
        return _normalize_workspace_path(raw_value)
    if isinstance(raw_value, list):
        normalized_candidates: list[str] = []
        for item in raw_value:
            normalized = _normalize_workspace_path(str(item))
            if normalized:
                normalized_candidates.append(normalized)
        if _looks_like_code_task(objective):
            preferred = _preferred_code_workspace_path(normalized_candidates)
            if preferred:
                return preferred
        if normalized_candidates:
            return normalized_candidates[0]
    return ""


def _normalize_workspace_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/").lstrip("./")
    if not normalized:
        return ""
    if normalized.lower().startswith("workspace/"):
        normalized = normalized[len("workspace/") :]
    return normalized


def _preferred_code_workspace_path(candidates: list[str]) -> str:
    source_candidates = [candidate for candidate in candidates if not _is_test_workspace_path(candidate)]
    for candidate in source_candidates:
        if _looks_like_source_workspace_path(candidate):
            return candidate
    if source_candidates:
        return source_candidates[0]
    return candidates[0] if candidates else ""


def _is_test_workspace_path(path: str) -> bool:
    lowered = path.lower()
    filename = lowered.rsplit("/", 1)[-1]
    return (
        lowered.startswith("tests/")
        or "/tests/" in lowered
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith("_test.go")
        or filename.endswith(".spec.ts")
        or filename.endswith(".spec.tsx")
        or filename.endswith(".test.ts")
        or filename.endswith(".test.tsx")
        or filename.endswith(".test.js")
        or filename.endswith(".test.jsx")
    )


def _looks_like_source_workspace_path(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(("src/", "app/", "cmd/", "lib/", "pkg/", "internal/")) or lowered.endswith(
        (".py", ".go", ".js", ".jsx", ".ts", ".tsx", ".rs", ".java", ".rb", ".php", ".cs")
    )


def _diagnostic_workspace_path(result: StepExecutionResult) -> str:
    if not bool(result.metadata.get("command_failed")):
        return ""

    candidates: list[str] = []
    for field in ("stderr", "stdout", "output_text"):
        raw_text = result.output_text if field == "output_text" else result.metadata.get(field, "")
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        for match in _WORKSPACE_PATH_RE.finditer(raw_text):
            normalized = _normalize_workspace_path(match.group("path"))
            if not normalized or normalized in candidates:
                continue
            candidates.append(normalized)
    if not candidates:
        return ""
    for candidate in candidates:
        lowered = candidate.lower()
        if not lowered.startswith("tests/") and not lowered.startswith("test_"):
            return candidate
    return candidates[0]


def _looks_like_code_task(objective: str) -> bool:
    lowered = objective.lower()
    return any(hint in lowered for hint in _CODE_ACTION_HINTS) and any(
        hint in lowered for hint in _CODE_TARGET_HINTS
    )


def _step_metadata(step: dict[str, Any]) -> dict[str, Any]:
    raw = step.get("metadata")
    return raw if isinstance(raw, dict) else {}


def _preferred_code_execution_command(
    result: StepExecutionResult,
    *,
    objective: str,
) -> list[str]:
    file_path = ""
    raw_path = result.metadata.get("file_path")
    if isinstance(raw_path, str) and raw_path.strip():
        file_path = _normalize_workspace_path(raw_path)
    if not file_path:
        file_path = _workspace_path_from_objective(objective)
    lowered_path = file_path.lower()
    if lowered_path.endswith(".py"):
        return ["python", "-m", "pytest", "-q"]
    if lowered_path.endswith(".go"):
        return ["go", "test", "./..."]
    if lowered_path.endswith((".js", ".jsx", ".ts", ".tsx")):
        return ["npm", "test", "--", "--runInBand"]
    return []


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
