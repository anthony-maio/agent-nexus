"""Integration tests for Nexus API run lifecycle."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient

from nexus_api.app import create_app
from nexus_api.config import ApiSettings
from nexus_api.service import build_context
from nexus_core.models import ArtifactRecord, CitationRecord, StepExecutionResult


class FakeExecutionAdapter:
    """Deterministic execution adapter for API integration tests."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        run_dir = self.base_dir / "sandbox" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        citations = [
            CitationRecord(
                url="https://example.com",
                title="Example",
                snippet=f"{action_type}: {instruction[:60]}",
            )
        ]
        artifacts = []
        if action_type in {"extract", "export", "write"}:
            name = f"{step_id}-{action_type}.txt"
            path = run_dir / name
            path.write_text(f"{action_type} output", encoding="utf-8")
            artifacts.append(
                ArtifactRecord(
                    kind="text",
                    name=name,
                    rel_path=f"{run_id}/{name}",
                    sandbox_path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        return StepExecutionResult(
            output_text=f"done:{action_type}",
            citations=citations,
            artifacts=artifacts,
        )


class FailingExecutionAdapter(FakeExecutionAdapter):
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "export":
            raise RuntimeError("simulated export failure")
        return await super().execute_step(run_id, step_id, action_type, instruction)


def _client(tmp_path: Path, execution_adapter: FakeExecutionAdapter | None = None) -> TestClient:
    settings = ApiSettings(
        APP_DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
        APP_CANONICAL_WORKSPACE=str(tmp_path / "workspace"),
        APP_SANDBOX_ARTIFACT_ROOT=str(tmp_path / "sandbox"),
        APP_ADMIN_USERNAME="admin",
        APP_ADMIN_PASSWORD="secret",
        APP_SESSION_TTL_HOURS=24,
    )
    ctx = build_context(settings)
    ctx.execution_adapter = execution_adapter or FakeExecutionAdapter(tmp_path)
    return TestClient(create_app(ctx))


def _auth_header(client: TestClient) -> dict[str, str]:
    resp = client.post("/sessions", json={"username": "admin", "password": "secret"})
    assert resp.status_code == 200
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_run_lifecycle_with_approval_and_promotion(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Prepare competitor research report",
            "mode": "supervised",
            "steps": [
                {"action_type": "navigate", "instruction": "find sources"},
                {"action_type": "extract", "instruction": "summarize findings"},
                {"action_type": "export", "instruction": "export report artifact"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()
    run_id = run["id"]
    assert run["status"] == "pending_approval"

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    items = pending.json()["items"]
    assert len(items) == 1
    step_id = items[0]["step_id"]

    details = client.get(f"/runs/{run_id}", headers=headers)
    assert details.status_code == 200
    run_details = details.json()
    assert len(run_details["citations"]) >= 2
    assert len(run_details["artifacts"]) >= 1

    timeline = client.get(f"/runs/{run_id}/timeline", headers=headers)
    assert timeline.status_code == 200
    assert len(timeline.json()["timeline"]) >= 1

    artifact_id = run_details["artifacts"][0]["id"]
    promote = client.post(
        f"/runs/{run_id}/artifacts/{artifact_id}/promote",
        headers=headers,
        json={"promoted_by": "admin"},
    )
    assert promote.status_code == 200
    assert Path(promote.json()["target_path"]).exists()

    approve = client.post(
        f"/runs/{run_id}/approvals/{step_id}",
        headers=headers,
        json={"decision": "approve", "reason": "safe export"},
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "completed"

    citations = client.get(f"/runs/{run_id}/citations", headers=headers)
    assert citations.status_code == 200
    assert len(citations.json()["citations"]) >= 3


def test_promote_rejects_integrity_mismatch(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Prepare extract artifact",
            "mode": "manual",
            "steps": [
                {"action_type": "extract", "instruction": "summarize findings"},
            ],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["id"]

    details = client.get(f"/runs/{run_id}", headers=headers)
    assert details.status_code == 200
    artifact = details.json()["artifacts"][0]
    Path(artifact["sandbox_path"]).write_text("tampered", encoding="utf-8")

    promote = client.post(
        f"/runs/{run_id}/artifacts/{artifact['id']}/promote",
        headers=headers,
        json={"promoted_by": "admin"},
    )
    assert promote.status_code == 403
    assert "integrity check failed" in promote.json()["detail"]


def test_promote_rejects_user_spoof_and_repeat_promote(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Prepare extract artifact",
            "mode": "manual",
            "steps": [
                {"action_type": "extract", "instruction": "summarize findings"},
            ],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["id"]

    details = client.get(f"/runs/{run_id}", headers=headers)
    assert details.status_code == 200
    artifact_id = details.json()["artifacts"][0]["id"]

    spoof = client.post(
        f"/runs/{run_id}/artifacts/{artifact_id}/promote",
        headers=headers,
        json={"promoted_by": "different-user"},
    )
    assert spoof.status_code == 403
    assert "must match authenticated user" in spoof.json()["detail"]

    first = client.post(
        f"/runs/{run_id}/artifacts/{artifact_id}/promote",
        headers=headers,
        json={"promoted_by": "admin"},
    )
    assert first.status_code == 200

    second = client.post(
        f"/runs/{run_id}/artifacts/{artifact_id}/promote",
        headers=headers,
        json={"promoted_by": "admin"},
    )
    assert second.status_code == 400
    assert "already promoted" in second.json()["detail"]


def test_autopilot_high_risk_artifact_can_promote_without_approval_record(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Autopilot export run",
            "mode": "autopilot",
            "steps": [
                {"action_type": "export", "instruction": "export report"},
            ],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["id"]
    assert create.json()["status"] == "completed"

    details = client.get(f"/runs/{run_id}", headers=headers)
    assert details.status_code == 200
    artifact_id = details.json()["artifacts"][0]["id"]

    promote = client.post(
        f"/runs/{run_id}/artifacts/{artifact_id}/promote",
        headers=headers,
        json={"promoted_by": "admin"},
    )
    assert promote.status_code == 200


def test_run_stream_ready_event(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)
    token = headers["Authorization"].split(" ", 1)[1]

    create = client.post(
        "/runs",
        headers=headers,
        json={"objective": "quick run", "mode": "manual", "steps": []},
    )
    assert create.status_code == 200
    run_id = create.json()["id"]

    with client.websocket_connect(f"/runs/{run_id}/stream?token={token}") as websocket:
        ready = websocket.receive_json()
        assert ready["event_type"] == "stream.ready"
        assert ready["run_id"] == run_id


def test_approved_step_failure_marks_run_failed(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=FailingExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Export report with failure",
            "mode": "supervised",
            "steps": [
                {"action_type": "navigate", "instruction": "find sources"},
                {"action_type": "extract", "instruction": "summarize findings"},
                {"action_type": "export", "instruction": "export report artifact"},
            ],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["id"]
    assert create.json()["status"] == "pending_approval"

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    step_id = pending.json()["items"][0]["step_id"]

    approve = client.post(
        f"/runs/{run_id}/approvals/{step_id}",
        headers=headers,
        json={"decision": "approve", "reason": "continue"},
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "failed"
