from __future__ import annotations

from nexus_core.verification import evaluate_run_completion


def _completed_step(action_type: str, *, metadata: dict | None = None) -> dict:
    return {
        "action_type": action_type,
        "status": "completed",
        "metadata": metadata or {},
    }


def test_skill_defined_code_verification_blocks_incomplete_run() -> None:
    run = {
        "objective": "Fix the payment retry bug in the repository",
        "steps": [
            _completed_step("list_files"),
            _completed_step("read_file"),
            _completed_step("extract"),
        ],
        "metadata": {
            "capability_state": {
                "verification_signals": ["mutating_code", "execute_code"],
            }
        },
    }

    verification = evaluate_run_completion(
        strategy="coding",
        run=run,
        citations=[],
        artifacts=[],
    )

    assert verification.result == "blocked"
    assert "mutating_code" in verification.reason
    assert "execute_code" in verification.reason


def test_skill_defined_artifact_requirements_verify_matching_output() -> None:
    run = {
        "objective": "Generate a chart from sales data",
        "steps": [
            _completed_step("generate_chart"),
        ],
        "metadata": {
            "capability_state": {
                "verification_signals": ["artifact"],
                "required_artifact_kinds": ["chart"],
            }
        },
    }

    verification = evaluate_run_completion(
        strategy="artifact",
        run=run,
        citations=[{"url": "https://example.com", "title": "Source", "snippet": "Grounded"}],
        artifacts=[
            {
                "kind": "chart",
                "name": "sales.html",
                "rel_path": "charts/sales.html",
                "sandbox_path": "D:/tmp/charts/sales.html",
                "sha256": "abc",
            }
        ],
    )

    assert verification.result == "verified"
    assert "skill-defined" in verification.reason
    assert verification.signals["artifact_kinds"] == ["chart"]


def test_pending_tool_follow_up_sequence_blocks_completion() -> None:
    run = {
        "objective": "Implement retry backoff in the repository",
        "steps": [
            _completed_step(
                "read_file",
                metadata={"tool_follow_up_sequence_remaining": ["execute_code"]},
            ),
        ],
        "metadata": {},
    }

    verification = evaluate_run_completion(
        strategy="coding",
        run=run,
        citations=[],
        artifacts=[],
    )

    assert verification.result == "blocked"
    assert "follow-up sequence" in verification.reason
    assert verification.signals["pending_tool_follow_up_sequence"]["remaining_actions"] == [
        "execute_code"
    ]
