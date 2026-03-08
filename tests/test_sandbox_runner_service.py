"""Tests for sandbox-runner execution service."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexus_sandbox_runner.app import create_app
from nexus_sandbox_runner.executors import StepResult


class _StubExecutor:
    backend_name = "stub"

    def execute(self, request, sandbox_root: Path) -> StepResult:
        return StepResult(
            output_text=f"handled {request.action_type}",
            citations=[],
            artifacts=[],
            metadata={
                "handled_action": request.action_type,
                "sandbox_root": str(sandbox_root),
            },
        )


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
    assert data["citations"] == []
    assert len(data["artifacts"]) == 1
    assert data["metadata"]["executor_backend"] == "local"
    assert data["metadata"]["session_path"]

    artifact = data["artifacts"][0]
    assert Path(artifact["sandbox_path"]).exists()
    assert artifact["sha256"]


def test_execute_step_no_artifact_for_navigate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "nexus_sandbox_runner.executors._fetch_url_content",
        lambda url, timeout_sec=10.0: {
            "url": url,
            "title": "Open Docs",
            "text": "Grounded page body",
            "snippet": "Grounded page body",
        },
    )
    client = TestClient(create_app())

    resp = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step124",
            "action_type": "navigate",
            "instruction": "open https://docs.example.org",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifacts"] == []
    assert len(data["citations"]) == 1
    assert data["citations"][0]["url"] == "https://docs.example.org"


def test_execute_step_supports_inspect_action(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "nexus_sandbox_runner.executors._fetch_url_content",
        lambda url, timeout_sec=10.0: {
            "url": url,
            "title": "Inspect Page",
            "text": "Form fields and confirmation copy",
            "snippet": "Form fields and confirmation copy",
        },
    )
    client = TestClient(create_app())

    navigate = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step124a",
            "action_type": "navigate",
            "instruction": "open https://docs.example.org",
        },
    )
    assert navigate.status_code == 200

    resp = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step124b",
            "action_type": "inspect",
            "instruction": "inspect the current page structure",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifacts"] == []
    assert len(data["citations"]) == 1
    assert "inspect" in data["output_text"].lower()
    assert data["metadata"]["current_url"] == "https://docs.example.org"


def test_execute_step_supports_type_action(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app())

    resp = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step124c",
            "action_type": "type",
            "instruction": "enter a draft response into the first field",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifacts"] == []
    assert data["citations"] == []
    assert "type" in data["output_text"].lower()


def test_execute_step_supports_grounded_search_and_fetch_actions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "nexus_sandbox_runner.executors._search_web",
        lambda query, max_results=5: [
            {
                "url": "https://docs.example.org/start",
                "title": f"Search result for {query}",
                "snippet": "Grounded result",
            }
        ],
    )
    monkeypatch.setattr(
        "nexus_sandbox_runner.executors._fetch_url_content",
        lambda url, timeout_sec=10.0: {
            "url": url,
            "title": "Fetched page",
            "text": "Grounded fetched text",
            "snippet": "Grounded fetched text",
        },
    )
    client = TestClient(create_app())

    search = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step-search",
            "action_type": "search_web",
            "instruction": "grounded runtime docs",
        },
    )
    assert search.status_code == 200
    search_data = search.json()
    assert search_data["citations"][0]["url"] == "https://docs.example.org/start"

    fetch = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": "step-fetch",
            "action_type": "fetch_url",
            "instruction": "open the best result from the current session",
        },
    )
    assert fetch.status_code == 200
    fetch_data = fetch.json()
    assert fetch_data["citations"][0]["url"] == "https://docs.example.org/start"
    assert fetch_data["metadata"]["current_url"] == "https://docs.example.org/start"


@pytest.mark.parametrize(
    "action_type,instruction",
    [
        ("list_files", "list the workspace files"),
        ("read_file", "read workspace/notes.txt"),
        ("write_file", "write workspace/report.txt with the latest summary"),
        ("edit_file", "replace TODO with done in workspace/report.txt"),
        ("execute_code", "run python -c \"print('ok')\""),
    ],
)
def test_execute_step_supports_workspace_and_code_action_contract(
    tmp_path: Path,
    monkeypatch,
    action_type: str,
    instruction: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "nexus_sandbox_runner.app.build_executor_from_env",
        lambda env: _StubExecutor(),
    )
    monkeypatch.setattr(
        "nexus_sandbox_runner.app.run_executor_preflight",
        lambda executor: {"status": "ok", "backend": executor.backend_name},
    )
    client = TestClient(create_app())

    resp = client.post(
        "/execute-step",
        json={
            "run_id": "run123",
            "step_id": f"step-{action_type}",
            "action_type": action_type,
            "instruction": instruction,
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["output_text"] == f"handled {action_type}"
    assert data["metadata"]["handled_action"] == action_type


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
