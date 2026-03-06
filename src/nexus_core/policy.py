"""Risk-tier policy for supervised automation."""

from __future__ import annotations

from nexus_core.models import RiskTier

_HIGH_RISK_ACTIONS: frozenset[str] = frozenset(
    {
        "submit",
        "write",
        "export",
        "promote",
        "send",
        "purchase",
        "delete",
    }
)


def risk_tier_for_action(action_type: str) -> RiskTier:
    """Classify step action type into low/high risk."""
    if action_type.strip().lower() in _HIGH_RISK_ACTIONS:
        return RiskTier.HIGH
    return RiskTier.LOW


def is_high_risk_action(action_type: str) -> bool:
    """Convenience helper for approval checks."""
    return risk_tier_for_action(action_type) == RiskTier.HIGH
