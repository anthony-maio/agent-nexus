"""Risk-tier policy for supervised automation."""

from __future__ import annotations

from nexus_core.models import RiskTier

_RESEARCHER_ACTIONS: frozenset[str] = frozenset(
    {
        "search_web",
        "fetch_url",
        "navigate",
        "inspect",
        "scroll",
        "read",
        "extract",
        "list_files",
        "read_file",
        "export",
    }
)
_OPERATOR_ACTIONS: frozenset[str] = frozenset(
    {
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


def risk_tier_for_action(action_type: str, instruction: str = "") -> RiskTier:
    """Classify step action type into low/high risk."""
    normalized_action = action_type.strip().lower()
    normalized_instruction = instruction.strip().lower()
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


def _normalize_workspace_path(raw_path: str) -> str:
    value = str(raw_path or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.strip("/")
    return value or "."
