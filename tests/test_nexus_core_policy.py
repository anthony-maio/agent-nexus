"""Tests for browser action risk-tier policy."""

from __future__ import annotations

import json

import pytest

from nexus_core.models import RiskTier
from nexus_core.policy import (
    delegated_role_allows_action,
    delegated_output_contract_violations,
    delegated_workspace_path_allowed,
    is_high_risk_action,
    risk_tier_for_action,
)


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


def test_delegate_risk_depends_on_child_plan() -> None:
    safe_delegate = json.dumps(
        {
            "role": "researcher",
            "objective": "Collect references",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "search_web",
                    "instruction": "collect evidence",
                }
            ],
        }
    )
    risky_delegate = json.dumps(
        {
            "role": "operator",
            "objective": "Update report",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "write_file",
                    "instruction": json.dumps({"path": "reports/summary.md", "content": "updated"}),
                }
            ],
        }
    )

    assert risk_tier_for_action("delegate", safe_delegate) == RiskTier.LOW
    assert risk_tier_for_action("delegate", risky_delegate) == RiskTier.HIGH
    assert is_high_risk_action("delegate", risky_delegate)


@pytest.mark.parametrize(
    ("role", "action_type"),
    [
        ("researcher", "search_web"),
        ("researcher", "read_file"),
        ("operator", "write_file"),
        ("operator", "type"),
    ],
)
def test_delegate_roles_allow_expected_actions(role: str, action_type: str) -> None:
    assert delegated_role_allows_action(role, action_type)


@pytest.mark.parametrize(
    ("role", "action_type"),
    [
        ("researcher", "write_file"),
        ("researcher", "edit_file"),
        ("researcher", "execute_code"),
        ("unknown", "read_file"),
    ],
)
def test_delegate_roles_reject_disallowed_actions(role: str, action_type: str) -> None:
    assert not delegated_role_allows_action(role, action_type)


def test_delegate_workspace_paths_allow_exact_or_nested_matches() -> None:
    assert delegated_workspace_path_allowed(["reports"], "reports")
    assert delegated_workspace_path_allowed(["reports"], "reports/summary.md")
    assert delegated_workspace_path_allowed(["reports/summary.md"], "reports/summary.md")


def test_delegate_workspace_paths_reject_out_of_scope_targets() -> None:
    assert not delegated_workspace_path_allowed(["reports/summary.md"], "reports")
    assert not delegated_workspace_path_allowed(["reports"], "notes/private.md")
    assert not delegated_workspace_path_allowed([], "reports/summary.md")


def test_delegate_output_contract_accepts_satisfied_requirements() -> None:
    violations = delegated_output_contract_violations(
        {
            "required_citation_count": 1,
            "required_artifact_kinds": ["text"],
        },
        citations=[{"url": "https://example.com"}],
        artifacts=[{"kind": "text", "name": "report.txt"}],
    )
    assert violations == []


def test_delegate_output_contract_reports_missing_citations_and_artifacts() -> None:
    violations = delegated_output_contract_violations(
        {
            "required_citation_count": 2,
            "required_artifact_kinds": ["text", "image"],
        },
        citations=[{"url": "https://example.com"}],
        artifacts=[{"kind": "text", "name": "report.txt"}],
    )
    assert "required at least 2 citation(s), received 1" in violations
    assert "required artifact kind `image` was not produced" in violations
