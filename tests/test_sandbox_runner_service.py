"""Tests for sandbox-runner execution service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nexus_sandbox_runner.app import create_app


def test_execute_step_creates_artifact_for_extract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app())

    payload = {
        "run_id": "run123",
        "step_id": "step123",
        "action_type": "extract",
        "instruction": "summarize sources",
    }
    resp = client.post("/execute-step", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "output_text" in data
    assert len(data["citations"]) == 1
    assert len(data["artifacts"]) == 1

    artifact = data["artifacts"][0]
    assert Path(artifact["sandbox_path"]).exists()
    assert artifact["sha256"]


def test_execute_step_no_artifact_for_navigate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app())

    resp = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step124",
            "action_type": "navigate",
            "instruction": "open docs",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifacts"] == []
    assert len(data["citations"]) == 1
