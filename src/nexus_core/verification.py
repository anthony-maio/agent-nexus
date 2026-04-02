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
    pending_tool_sequence = _pending_tool_follow_up_sequence(completed_steps)
    child_runs = run.get("child_runs")
    child_statuses = [
        str(child.get("status", "")).strip().lower()
        for child in (child_runs or [])
        if isinstance(child_runs, list) and isinstance(child, dict)
    ]
    objective_text = str(run.get("objective", "")).strip().lower()
    explicit_research_intent = any(keyword in objective_text for keyword in _EXPLICIT_RESEARCH_KEYWORDS)
    metadata = run.get("metadata")
    capability_state = (
        metadata.get("capability_state")
        if isinstance(metadata, dict) and isinstance(metadata.get("capability_state"), dict)
        else {}
    )
    required_signals = _normalized_string_list(capability_state.get("verification_signals"))
    required_artifact_kinds = _normalized_string_list(capability_state.get("required_artifact_kinds"))
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
        "artifact_kinds": sorted(
            {
                str(item.get("kind", "")).strip().lower()
                for item in artifacts
                if isinstance(item, dict) and str(item.get("kind", "")).strip()
            }
        ),
        "completed_actions": sorted(completed_actions),
        "child_run_count": len(child_statuses),
        "completed_child_run_count": sum(status == RunStatus.COMPLETED.value for status in child_statuses),
        "failed_child_run_count": sum(status == RunStatus.FAILED.value for status in child_statuses),
        "successful_execute_code_count": successful_execute_code_count,
        "mutating_code_action_count": sum(action in _CODING_MUTATION_ACTIONS for action in completed_actions),
    }
    if pending_tool_sequence:
        signals["pending_tool_follow_up_sequence"] = pending_tool_sequence
    if required_signals:
        signals["required_verification_signals"] = list(required_signals)
    if required_artifact_kinds:
        signals["required_artifact_kinds"] = list(required_artifact_kinds)

    if not completed_steps:
        return RunVerificationRecord(
            strategy=strategy,
            result="blocked",
            reason="Run ended without completing any steps.",
            signals=signals,
        )

    if pending_tool_sequence:
        remaining_actions = pending_tool_sequence.get("remaining_actions") or []
        readable_remaining = ", ".join(str(item) for item in remaining_actions) or "follow-up work"
        return RunVerificationRecord(
            strategy=strategy,
            result="blocked",
            reason=(
                "Run ended before completing the capability-declared tool follow-up "
                f"sequence: {readable_remaining}."
            ),
            signals=signals,
        )

    capability_missing = _missing_capability_requirements(
        required_signals=required_signals,
        required_artifact_kinds=required_artifact_kinds,
        completed_actions=completed_actions,
        citations=citations,
        artifacts=artifacts,
        signals=signals,
    )
    if capability_missing:
        return RunVerificationRecord(
            strategy=strategy,
            result="blocked",
            reason=(
                "Run did not satisfy skill-defined completion requirements: "
                + ", ".join(capability_missing)
            ),
            signals=signals,
        )
    if required_signals or required_artifact_kinds:
        return RunVerificationRecord(
            strategy=strategy,
            result="verified",
            reason="Run satisfied skill-defined completion requirements.",
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


def _pending_tool_follow_up_sequence(completed_steps: list[dict[str, Any]]) -> dict[str, Any]:
    for step in reversed(completed_steps):
        metadata = step.get("metadata")
        if not isinstance(metadata, dict):
            continue
        raw_remaining = metadata.get("tool_follow_up_sequence_remaining")
        if not isinstance(raw_remaining, list):
            continue
        remaining_actions = [str(item).strip().lower() for item in raw_remaining if str(item).strip()]
        if not remaining_actions:
            continue
        payload = {
            "step_action": str(step.get("action_type", "")).strip().lower(),
            "remaining_actions": remaining_actions,
        }
        external_tool = metadata.get("external_tool")
        if isinstance(external_tool, dict):
            tool_name = str(external_tool.get("name", "")).strip()
            if tool_name:
                payload["tool_name"] = tool_name
        return payload
    return {}


def _missing_capability_requirements(
    *,
    required_signals: list[str],
    required_artifact_kinds: list[str],
    completed_actions: set[str],
    citations: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    signals: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    artifact_kinds = set(signals.get("artifact_kinds", []))
    for signal in required_signals:
        if signal == "citations" and not citations:
            missing.append("citations")
        elif signal == "artifact" and not artifacts:
            missing.append("artifact")
        elif signal == "submit" and not (completed_actions & _WORKFLOW_ACTIONS):
            missing.append("submit")
        elif signal == "execute_code" and int(signals.get("successful_execute_code_count", 0)) <= 0:
            missing.append("execute_code")
        elif signal in {"mutating_code", "file_mutation"} and int(
            signals.get("mutating_code_action_count", 0)
        ) <= 0:
            missing.append("mutating_code")
    missing_artifact_kinds = [kind for kind in required_artifact_kinds if kind not in artifact_kinds]
    for kind in missing_artifact_kinds:
        missing.append(f"artifact_kind:{kind}")
    return missing


def _normalized_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = str(item).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        items.append(value)
    return items
