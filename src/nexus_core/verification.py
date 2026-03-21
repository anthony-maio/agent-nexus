"""Run-level completion verification for autonomous kernel outcomes."""

from __future__ import annotations

from typing import Any

from nexus_core.models import RunVerificationRecord, RunStatus, StepStatus

_ARTIFACT_ACTIONS = {"export", "generate_report", "generate_chart", "generate_image"}
_WORKFLOW_ACTIONS = {"submit"}
_CODING_MUTATION_ACTIONS = {"write_file", "edit_file", "write"}
_RESEARCH_ACTIONS = {"search_web", "fetch_url", "extract", "call_api"}
_EXPLICIT_RESEARCH_KEYWORDS = {
    "research",
    "competitor",
    "pricing",
    "citation",
    "citations",
    "source",
    "sources",
    "evidence",
    "docs",
    "documentation",
}


def evaluate_run_completion(
    *,
    strategy: str,
    run: dict[str, Any],
    citations: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> RunVerificationRecord:
    """Evaluate whether a finished run is verified, provisional, or blocked."""

    steps = run.get("steps")
    completed_steps = [
        step
        for step in steps
        if isinstance(steps, list)
        and isinstance(step, dict)
        and str(step.get("status", "")).strip().lower() == StepStatus.COMPLETED.value
    ]
    completed_actions = {
        str(step.get("action_type", "")).strip().lower()
        for step in completed_steps
        if str(step.get("action_type", "")).strip()
    }
    child_runs = run.get("child_runs")
    child_statuses = [
        str(child.get("status", "")).strip().lower()
        for child in child_runs
        if isinstance(child_runs, list) and isinstance(child, dict)
    ]
    objective_text = str(run.get("objective", "")).strip().lower()
    explicit_research_intent = any(keyword in objective_text for keyword in _EXPLICIT_RESEARCH_KEYWORDS)
    successful_execute_code_count = 0
    for step in completed_steps:
        action = str(step.get("action_type", "")).strip().lower()
        if action != "execute_code":
            continue
        metadata = step.get("metadata")
        if not isinstance(metadata, dict):
            successful_execute_code_count += 1
            continue
        if str(metadata.get("verification_result", "")).strip().lower() == "failed":
            continue
        successful_execute_code_count += 1

    signals: dict[str, Any] = {
        "completed_step_count": len(completed_steps),
        "citation_count": len(citations),
        "artifact_count": len(artifacts),
        "completed_actions": sorted(completed_actions),
        "child_run_count": len(child_statuses),
        "completed_child_run_count": sum(status == RunStatus.COMPLETED.value for status in child_statuses),
        "failed_child_run_count": sum(status == RunStatus.FAILED.value for status in child_statuses),
        "successful_execute_code_count": successful_execute_code_count,
        "mutating_code_action_count": sum(action in _CODING_MUTATION_ACTIONS for action in completed_actions),
    }

    if not completed_steps:
        return RunVerificationRecord(
            strategy=strategy,
            result="blocked",
            reason="Run ended without completing any steps.",
            signals=signals,
        )

    if completed_actions & _ARTIFACT_ACTIONS:
        if not artifacts:
            return RunVerificationRecord(
                strategy=strategy,
                result="blocked",
                reason="Run finished with a terminal artifact action but produced no artifacts.",
                signals=signals,
            )
        if strategy == "research" and not citations:
            return RunVerificationRecord(
                strategy=strategy,
                result="blocked",
                reason="Research run produced an artifact without grounded citations.",
                signals=signals,
            )
        return RunVerificationRecord(
            strategy=strategy,
            result="verified",
            reason="Run produced a terminal artifact output.",
            signals=signals,
        )

    if completed_actions & _WORKFLOW_ACTIONS:
        return RunVerificationRecord(
            strategy=strategy,
            result="verified",
            reason="Workflow run completed a terminal submit action.",
            signals=signals,
        )

    if signals["mutating_code_action_count"] > 0 and successful_execute_code_count > 0:
        return RunVerificationRecord(
            strategy=strategy,
            result="verified",
            reason="Run changed files and completed code execution successfully.",
            signals=signals,
        )

    if signals["mutating_code_action_count"] > 0 or successful_execute_code_count > 0:
        return RunVerificationRecord(
            strategy=strategy,
            result="provisional",
            reason="Run made mutating or executable progress without a stronger strategy-specific verifier.",
            signals=signals,
        )

    if "delegate" in completed_actions and signals["child_run_count"] > 0:
        return RunVerificationRecord(
            strategy=strategy,
            result="provisional",
            reason="Run completed through delegation without a stronger terminal verifier.",
            signals=signals,
        )

    if strategy == "research":
        if not explicit_research_intent or not (completed_actions & _RESEARCH_ACTIONS):
            return RunVerificationRecord(
                strategy=strategy,
                result="provisional",
                reason="Run finished without a strategy-specific terminal verifier.",
                signals=signals,
            )
        if not citations:
            return RunVerificationRecord(
                strategy=strategy,
                result="blocked",
                reason="Research run finished without grounded citations.",
                signals=signals,
            )
        if run.get("parent_run_id") or run.get("delegation"):
            return RunVerificationRecord(
                strategy=strategy,
                result="provisional",
                reason="Delegated research run gathered grounded evidence without a terminal artifact.",
                signals=signals,
            )
        return RunVerificationRecord(
            strategy=strategy,
            result="blocked",
            reason="Research run finished without a terminal synthesis or output step.",
            signals=signals,
        )

    if strategy == "workflow":
        return RunVerificationRecord(
            strategy=strategy,
            result="provisional",
            reason="Workflow run ended without a terminal submit action.",
            signals=signals,
        )

    return RunVerificationRecord(
        strategy=strategy,
        result="provisional",
        reason="Run finished without a strategy-specific terminal verifier.",
        signals=signals,
    )
