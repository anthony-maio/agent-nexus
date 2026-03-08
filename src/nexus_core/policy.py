"""Risk-tier policy for supervised automation."""

from __future__ import annotations

from nexus_core.models import RiskTier

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
