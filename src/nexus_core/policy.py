"""Risk-tier policy for supervised automation."""

from __future__ import annotations

import json
from typing import Any

from nexus_core.models import RiskTier

_RESEARCHER_ACTIONS: frozenset[str] = frozenset(
    {
        "search_web",
        "fetch_url",
        "call_api",
        "navigate",
        "inspect",
        "scroll",
        "read",
        "extract",
        "list_files",
        "read_file",
        "export",
        "generate_report",
        "generate_chart",
        "generate_image",
    }
)
_OPERATOR_ACTIONS: frozenset[str] = frozenset(
    {
        "call_api",
        "navigate",
        "inspect",
        "scroll",
        "read",
        "extract",
        "click",
        "wait",
        "type",
        "submit",
        "list_files",
        "read_file",
        "write_file",
        "edit_file",
        "export",
        "generate_report",
        "generate_chart",
        "generate_image",
    }
)
_DELEGATED_ROLE_ACTIONS: dict[str, frozenset[str]] = {
    "researcher": _RESEARCHER_ACTIONS,
    "operator": _OPERATOR_ACTIONS,
}

_HIGH_RISK_ACTIONS: frozenset[str] = frozenset(
    {
        "submit",
        "type",
        "write",
        "write_file",
        "edit_file",
        "execute_code",
        "export",
        "generate_report",
        "generate_chart",
        "generate_image",
        "external_tool",
        "promote",
        "send",
        "purchase",
        "delete",
    }
)
_HIGH_RISK_CLICK_HINTS: tuple[str, ...] = (
    "submit",
    "confirm",
    "purchase",
    "buy",
    "delete",
    "remove",
    "send",
    "post",
    "apply",
    "checkout",
    "place order",
)
_LOW_RISK_EXECUTE_CODE_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "pytest"),
    ("pytest",),
    ("python", "-m", "unittest"),
    ("go", "test"),
    ("cargo", "test"),
    ("npm", "test"),
    ("pnpm", "test"),
    ("yarn", "test"),
    ("vitest",),
    ("npx", "vitest"),
    ("jest",),
    ("npx", "jest"),
)
_SHELL_CONTROL_TOKENS: frozenset[str] = frozenset({"&&", "||", ";", "|", ">", ">>", "<"})


def risk_tier_for_action(action_type: str, instruction: str = "") -> RiskTier:
    """Classify step action type into low/high risk."""
    normalized_action = action_type.strip().lower()
    normalized_instruction = instruction.strip().lower()
    if normalized_action == "delegate" and _delegate_instruction_is_high_risk(instruction):
        return RiskTier.HIGH
    if normalized_action == "execute_code":
        if _execute_code_instruction_is_low_risk(instruction):
            return RiskTier.LOW
        return RiskTier.HIGH
    if normalized_action == "call_api":
        if _call_api_instruction_is_low_risk(instruction):
            return RiskTier.LOW
        return RiskTier.HIGH
    if normalized_action in _HIGH_RISK_ACTIONS:
        return RiskTier.HIGH
    if normalized_action == "click" and any(
        hint in normalized_instruction for hint in _HIGH_RISK_CLICK_HINTS
    ):
        return RiskTier.HIGH
    return RiskTier.LOW


def is_high_risk_action(action_type: str, instruction: str = "") -> bool:
    """Convenience helper for approval checks."""
    return risk_tier_for_action(action_type, instruction) == RiskTier.HIGH


def delegated_role_allows_action(role: str, action_type: str) -> bool:
    """Return whether a delegated worker role may execute the action."""
    allowed_actions = _DELEGATED_ROLE_ACTIONS.get(role.strip().lower())
    if allowed_actions is None:
        return False
    return action_type.strip().lower() in allowed_actions


def delegated_workspace_path_allowed(
    allowed_paths: list[str] | tuple[str, ...],
    target_path: str,
) -> bool:
    """Return whether a delegated workspace path stays within handed-off scope."""
    normalized_target = _normalize_workspace_path(target_path)
    if not normalized_target:
        return False

    normalized_allowed = [
        candidate
        for candidate in (_normalize_workspace_path(path) for path in allowed_paths)
        if candidate
    ]
    if not normalized_allowed:
        return False

    for candidate in normalized_allowed:
        if candidate == ".":
            return True
        if normalized_target == candidate or normalized_target.startswith(f"{candidate}/"):
            return True
    return False


def delegated_output_contract_violations(
    context: dict[str, Any] | None,
    citations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[str]:
    """Return unmet output-contract requirements for a delegated run."""
    if not isinstance(context, dict):
        return []

    violations: list[str] = []
    required_citation_count = _required_citation_count(context.get("required_citation_count"))
    actual_citation_count = len(citations)
    if required_citation_count and actual_citation_count < required_citation_count:
        violations.append(
            f"required at least {required_citation_count} citation(s), received {actual_citation_count}"
        )

    required_artifact_kinds = _required_artifact_kinds(context.get("required_artifact_kinds"))
    if required_artifact_kinds:
        actual_artifact_kinds = {
            kind
            for kind in (_artifact_kind(artifact) for artifact in artifacts)
            if kind
        }
        for kind in required_artifact_kinds:
            if kind not in actual_artifact_kinds:
                violations.append(f"required artifact kind `{kind}` was not produced")

    return violations


def _normalize_workspace_path(raw_path: str) -> str:
    value = str(raw_path or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.strip("/")
    return value or "."


def _required_citation_count(raw_value: Any) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 0
    return max(value, 0)


def _required_artifact_kinds(raw_value: Any) -> list[str]:
    if not isinstance(raw_value, list):
        return []

    kinds: list[str] = []
    seen: set[str] = set()
    for item in raw_value:
        normalized = str(item or "").strip().lower()
        if not normalized or normalized in seen:
            continue
        kinds.append(normalized)
        seen.add(normalized)
    return kinds


def _artifact_kind(artifact: dict[str, Any]) -> str:
    return str(artifact.get("kind", "") or "").strip().lower()


def _delegate_instruction_is_high_risk(instruction: str) -> bool:
    try:
        payload = json.loads(instruction)
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, dict):
        return False

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        role = str(payload.get("role", "")).strip().lower()
        return role not in {"", "researcher"}

    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        action_type = str(raw_step.get("action_type", "")).strip().lower()
        child_instruction = str(raw_step.get("instruction", ""))
        if risk_tier_for_action(action_type, child_instruction) == RiskTier.HIGH:
            return True
    return False


def _execute_code_instruction_is_low_risk(instruction: str) -> bool:
    command = _command_from_execute_code_instruction(instruction)
    if not command:
        return False

    normalized = [part.strip().lower() for part in command if str(part).strip()]
    if not normalized:
        return False
    if any(token in _SHELL_CONTROL_TOKENS for token in normalized):
        return False

    return any(
        tuple(normalized[: len(prefix)]) == prefix for prefix in _LOW_RISK_EXECUTE_CODE_PREFIXES
    )


def _call_api_instruction_is_low_risk(instruction: str) -> bool:
    payload: Any
    try:
        payload = json.loads(instruction)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        return False
    method = str(payload.get("method", "GET")).strip().upper() or "GET"
    has_body = payload.get("json") is not None or payload.get("body") is not None
    return method in {"GET", "HEAD"} and not has_body


def _command_from_execute_code_instruction(instruction: str) -> list[str]:
    payload: Any
    try:
        payload = json.loads(instruction)
    except json.JSONDecodeError:
        payload = instruction

    command = payload.get("command") if isinstance(payload, dict) else payload
    if isinstance(command, list):
        return [str(part) for part in command if str(part).strip()]
    if isinstance(command, str):
        return [part for part in command.split() if part]
    return []
