"""Tests for sandbox-runner execution service."""

from __future__ import annotations

from pathlib import Path

import pytest
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
    assert data["metadata"]["executor_backend"] == "local"

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


def test_execute_step_rejects_unsupported_action(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app())

    resp = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step125",
            "action_type": "delete",
            "instruction": "remove production data",
        },
    )
    assert resp.status_code == 400
    assert "Unsupported action_type" in resp.json()["detail"]


def test_execute_step_requires_token_when_configured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_RUNNER_TOKEN", "sandbox-secret")
    client = TestClient(create_app())

    payload = {
        "run_id": "run123",
        "step_id": "step126",
        "action_type": "extract",
        "instruction": "summarize sources",
    }
    unauthorized = client.post("/execute-step", json=payload)
    assert unauthorized.status_code == 401

    authorized = client.post(
        "/execute-step",
        json=payload,
        headers={"X-Sandbox-Token": "sandbox-secret"},
    )
    assert authorized.status_code == 200


def test_health_reports_executor_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_EXECUTION_BACKEND", "local")
    client = TestClient(create_app())

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["executor_backend"] == "local"
    assert resp.json()["preflight_status"] == "ok"


def test_docker_backend_fails_fast_when_docker_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_EXECUTION_BACKEND", "docker")
    monkeypatch.setenv(
        "SANDBOX_DOCKER_IMAGE",
        "python:3.13-slim@sha256:8bc60ca09afaa8ea0d6d1220bde073bacfedd66a4bf8129cbdc8ef0e16c8a952",
    )
    monkeypatch.setenv(
        "SANDBOX_DOCKER_ALLOWED_IMAGES",
        "python:3.13-slim@sha256:8bc60ca09afaa8ea0d6d1220bde073bacfedd66a4bf8129cbdc8ef0e16c8a952",
    )
    monkeypatch.setattr("nexus_sandbox_runner.executors.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="preflight failed"):
        create_app()
