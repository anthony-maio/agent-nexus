"""Browser-first run planning for app-first execution."""

from __future__ import annotations

import re

from nexus_core.models import StepDefinition

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


def plan_steps_for_objective(objective: str) -> list[StepDefinition]:
    """Return a browser-first run plan for the given objective."""
    cleaned = " ".join(objective.split())
    if _looks_like_workflow(cleaned):
        return _workflow_steps(cleaned)
    return _research_steps(cleaned)


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
