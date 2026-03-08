"""Tests for browser action risk-tier policy."""

from __future__ import annotations

import pytest

from nexus_core.models import RiskTier
from nexus_core.policy import is_high_risk_action, risk_tier_for_action


def test_type_actions_are_treated_as_high_risk() -> None:
    assert risk_tier_for_action("type") == RiskTier.HIGH
    assert is_high_risk_action("type")


def test_click_risk_depends_on_instruction_intent() -> None:
    assert risk_tier_for_action("click", "Click the next non-destructive button") == RiskTier.LOW
    assert risk_tier_for_action("click", "Click the submit order button") == RiskTier.HIGH


@pytest.mark.parametrize("action_type", ["list_files", "read_file"])
def test_workspace_read_actions_are_low_risk(action_type: str) -> None:
    assert risk_tier_for_action(action_type) == RiskTier.LOW
    assert not is_high_risk_action(action_type)


@pytest.mark.parametrize("action_type", ["write_file", "edit_file", "execute_code"])
def test_workspace_mutation_and_code_actions_are_high_risk(action_type: str) -> None:
    assert risk_tier_for_action(action_type) == RiskTier.HIGH
    assert is_high_risk_action(action_type)
