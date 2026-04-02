from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nexus_core.engine import RunEngine


def _engine(tmp_path: Path) -> RunEngine:
    return RunEngine(
        repository=SimpleNamespace(),
        execution=SimpleNamespace(),
        interaction=SimpleNamespace(),
        events=SimpleNamespace(),
        canonical_workspace=tmp_path / "workspace",
    )


def test_completion_recovery_step_for_tool_sequence_continues_declared_chain(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    completed_step = {
        "id": "step-2",
        "action_type": "read_file",
        "output_text": "def retry_backoff():\n    return 2",
        "metadata": {
            "file_path": "src/payments/retry.py",
            "tool_follow_up_sequence_remaining": ["execute_code"],
        },
        "status": "completed",
    }
    run = {
        "objective": "Implement retry backoff in the repo",
        "steps": [completed_step],
    }

    recovery_step = engine._completion_recovery_step_for_tool_sequence(
        run=run,
        completed_step=completed_step,
    )

    assert recovery_step is not None
    assert recovery_step.action_type == "execute_code"
    assert recovery_step.instruction == '{"command": ["python", "-m", "pytest", "-q"]}'


def test_kernel_strategy_snapshot_surfaces_pending_tool_sequence_tactic() -> None:
    snapshot = RunEngine._kernel_strategy_snapshot(
        objective="Implement retry backoff in the repo",
        phase="running",
        steps=[
            {
                "id": "step-2",
                "action_type": "read_file",
                "status": "completed",
                "metadata": {"tool_follow_up_sequence_remaining": ["execute_code"]},
            }
        ],
        capability_state={},
        kernel_focus={},
    )

    assert snapshot["tactic"] == "sequence_follow_up"
    assert "execute_code" in snapshot["tactic_reason"]
