"""Integration tests for Nexus API run lifecycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from nexus_api.app import create_app
from nexus_api.config import ApiSettings
from nexus_api.service import build_context
from nexus_core.models import ArtifactRecord, CitationRecord, StepDefinition, StepExecutionResult


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


class AdaptiveExecutionAdapter(FakeExecutionAdapter):
    def __init__(self, base_dir: Path) -> None:
        super().__init__(base_dir)
        self._extract_calls = 0

    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "extract":
            self._extract_calls += 1
            if self._extract_calls == 1:
                return StepExecutionResult(
                    output_text="done:extract-no-citations",
                    citations=[],
                    artifacts=[],
                )
        return await super().execute_step(run_id, step_id, action_type, instruction)


class FlakyExportExecutionAdapter(FakeExecutionAdapter):
    def __init__(self, base_dir: Path) -> None:
        super().__init__(base_dir)
        self._export_calls = 0

    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "export":
            self._export_calls += 1
            if self._export_calls == 1:
                raise RuntimeError("transient export failure")
        return await super().execute_step(run_id, step_id, action_type, instruction)


class ModelSuggestionPlanner:
    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, str],
        result: StepExecutionResult,
        existing_steps: list[dict[str, str]],
    ) -> list[StepDefinition]:
        _ = objective, completed_step, result, existing_steps
        return [
            StepDefinition(
                action_type="delete",
                instruction="delete all source documents",
            ),
            StepDefinition(
                action_type="submit",
                instruction="submit the prepared workflow changes",
            ),
        ]


class WorkspaceDiscoveryExecutionAdapter(FakeExecutionAdapter):
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "list_files":
            return StepExecutionResult(
                output_text="done:list_files",
                citations=[],
                artifacts=[],
                metadata={"files": ["reports/summary.md"]},
            )
        return await super().execute_step(run_id, step_id, action_type, instruction)


class CodeArtifactExecutionAdapter(FakeExecutionAdapter):
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "execute_code":
            return StepExecutionResult(
                output_text="done:execute_code",
                citations=[],
                artifacts=[],
                metadata={"touched_files": ["reports/generated.md"]},
            )
        return await super().execute_step(run_id, step_id, action_type, instruction)


def _client(tmp_path: Path, execution_adapter: FakeExecutionAdapter | None = None) -> TestClient:
    settings = ApiSettings(
        APP_DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
        APP_CANONICAL_WORKSPACE=str(tmp_path / "workspace"),
        APP_SANDBOX_ARTIFACT_ROOT=str(tmp_path / "sandbox"),
        APP_ADMIN_USERNAME="admin",
        APP_ADMIN_PASSWORD="secret",
        APP_SESSION_TTL_HOURS=24,
        APP_ENABLE_MODEL_REPLANNER=False,
    )
    ctx = build_context(settings)
    ctx.execution_adapter = execution_adapter or FakeExecutionAdapter(tmp_path)
    return TestClient(create_app(ctx))


def _client_with_planner(
    tmp_path: Path,
    adaptive_planner: object,
    execution_adapter: FakeExecutionAdapter | None = None,
) -> TestClient:
    settings = ApiSettings(
        APP_DATABASE_URL=f"sqlite:///{tmp_path / 'app.db'}",
        APP_CANONICAL_WORKSPACE=str(tmp_path / "workspace"),
        APP_SANDBOX_ARTIFACT_ROOT=str(tmp_path / "sandbox"),
        APP_ADMIN_USERNAME="admin",
        APP_ADMIN_PASSWORD="secret",
        APP_SESSION_TTL_HOURS=24,
        APP_ENABLE_MODEL_REPLANNER=False,
    )
    ctx = build_context(settings)
    ctx.execution_adapter = execution_adapter or FakeExecutionAdapter(tmp_path)
    ctx.adaptive_planner = adaptive_planner
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


def test_default_research_run_bootstraps_autonomous_tool_loop(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Research payroll automation competitors and capture citations",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "search_web",
        "fetch_url",
        "extract",
        "export",
    ]
    assert [step["status"] for step in run["steps"]] == [
        "completed",
        "completed",
        "completed",
        "pending_approval",
    ]
    assert run["status"] == "pending_approval"


def test_default_workflow_run_gates_on_first_autonomous_write_action(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Fill out the contact form at https://example.com/contact",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "navigate",
        "inspect",
        "extract",
        "type",
    ]
    assert [step["status"] for step in run["steps"]] == [
        "completed",
        "completed",
        "completed",
        "pending_approval",
    ]
    assert run["status"] == "pending_approval"

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    items = pending.json()["items"]
    assert len(items) == 1
    assert items[0]["run_id"] == run["id"]
    assert items[0]["action_type"] == "type"


def test_default_workspace_file_run_bootstraps_with_read_file(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Read workspace/brief.txt and summarize the key points",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "read_file",
        "extract",
        "export",
    ]
    assert run["steps"][0]["status"] == "completed"
    assert run["steps"][1]["status"] == "completed"
    assert run["steps"][2]["status"] == "pending_approval"
    assert "brief.txt" in run["steps"][0]["instruction"]


def test_run_adapts_list_files_into_read_file(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=WorkspaceDiscoveryExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Review the discovered report file and summarize it",
            "mode": "supervised",
            "steps": [
                {
                    "action_type": "list_files",
                    "instruction": json.dumps({"path": "."}),
                }
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "list_files",
        "read_file",
        "extract",
        "export",
    ]
    assert run["steps"][1]["status"] == "completed"
    assert "reports/summary.md" in run["steps"][1]["instruction"]
    assert run["steps"][3]["status"] == "pending_approval"


def test_run_adapts_execute_code_into_read_file(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=CodeArtifactExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Generate a report file and summarize it",
            "mode": "autopilot",
            "steps": [
                {
                    "action_type": "execute_code",
                    "instruction": json.dumps(
                        {"command": ["python", "-c", "print('generate report')"]}
                    ),
                }
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "execute_code",
        "read_file",
        "extract",
        "export",
    ]
    assert run["steps"][1]["status"] == "completed"
    assert "reports/generated.md" in run["steps"][1]["instruction"]
    assert run["steps"][3]["status"] == "completed"


def test_parent_child_run_persistence_and_delegation_summary(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    parent = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Primary operator objective",
            "mode": "manual",
            "steps": [],
        },
    )
    assert parent.status_code == 200
    parent_id = parent.json()["id"]

    child = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Research competitor docs",
            "mode": "manual",
            "parent_run_id": parent_id,
            "delegation": {
                "role": "researcher",
                "objective": "Collect relevant docs",
                "status": "completed",
                "summary": "Collected 3 relevant docs",
                "context": {
                    "handoff_note": "Start from prior research",
                    "workspace_paths": ["reports/summary.md"],
                },
            },
            "steps": [
                {"action_type": "search_web", "instruction": "collect relevant docs"},
            ],
        },
    )
    assert child.status_code == 200
    child_run = child.json()
    assert child_run["parent_run_id"] == parent_id
    assert child_run["delegation"]["role"] == "researcher"
    assert child_run["delegation"]["summary"] == "Collected 3 relevant docs"
    assert child_run["delegation"]["context"]["handoff_note"] == "Start from prior research"

    parent_detail = client.get(f"/runs/{parent_id}", headers=headers)
    assert parent_detail.status_code == 200
    child_runs = parent_detail.json()["child_runs"]
    assert len(child_runs) == 1
    assert child_runs[0]["id"] == child_run["id"]
    assert child_runs[0]["delegation_role"] == "researcher"
    assert child_runs[0]["delegation_status"] == "completed"
    assert child_runs[0]["delegation_summary"] == "Collected 3 relevant docs"
    assert child_runs[0]["delegation_context"]["workspace_paths"] == ["reports/summary.md"]
    assert child_runs[0]["steps"]
    assert child_runs[0]["steps"][0]["action_type"] == "search_web"
    assert child_runs[0]["steps"][0]["instruction"] == "collect relevant docs"


def test_delegate_step_inherits_parent_context_snapshot(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with inherited context",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "extract",
                    "instruction": "summarize parent evidence",
                },
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Collect competitor docs",
                            "mode": "manual",
                            "steps": [
                                {
                                    "action_type": "search_web",
                                    "instruction": "collect competitor docs",
                                }
                            ],
                        }
                    ),
                },
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()
    assert len(run["child_runs"]) == 1

    child = run["child_runs"][0]
    context = child["delegation_context"]
    assert context["parent_run_id"] == run["id"]
    assert context["parent_objective"] == "Parent run with inherited context"
    assert context["citations"][0]["url"] == "https://example.com"
    assert context["artifacts"][0]["kind"] == "text"

    child_detail = client.get(f"/runs/{child['id']}", headers=headers)
    assert child_detail.status_code == 200
    assert child_detail.json()["delegation"]["context"]["parent_run_id"] == run["id"]


def test_delegate_researcher_role_blocks_workspace_mutation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with constrained researcher",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Update workspace file",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports/allowed.md"]},
                            "steps": [
                                {
                                    "action_type": "write_file",
                                    "instruction": json.dumps(
                                        {
                                            "path": "reports/allowed.md",
                                            "content": "updated by delegate",
                                        }
                                    ),
                                }
                            ],
                        }
                    ),
                },
                {"action_type": "navigate", "instruction": "open fallback page"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()
    assert run["status"] == "completed"
    assert "failed" in run["steps"][0]["output_text"].lower()
    assert run["steps"][1]["status"] == "completed"
    assert run["child_runs"][0]["status"] == "failed"

    child_detail = client.get(f"/runs/{run['child_runs'][0]['id']}", headers=headers)
    assert child_detail.status_code == 200
    assert "does not allow action `write_file`" in child_detail.json()["steps"][0]["error_text"]


def test_delegate_workspace_reads_stay_within_handoff_scope(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    blocked = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with scoped reader",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Read delegated workspace file",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports/allowed.md"]},
                            "steps": [
                                {
                                    "action_type": "read_file",
                                    "instruction": json.dumps({"path": "notes/private.md"}),
                                }
                            ],
                        }
                    ),
                }
            ],
        },
    )
    assert blocked.status_code == 200
    blocked_run = blocked.json()
    assert blocked_run["child_runs"][0]["status"] == "failed"

    blocked_child = client.get(
        f"/runs/{blocked_run['child_runs'][0]['id']}",
        headers=headers,
    )
    assert blocked_child.status_code == 200
    assert "outside delegated workspace scope" in blocked_child.json()["steps"][0]["error_text"]

    allowed = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with allowed reader",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Read delegated workspace file",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports/allowed.md"]},
                            "steps": [
                                {
                                    "action_type": "read_file",
                                    "instruction": json.dumps({"path": "reports/allowed.md"}),
                                }
                            ],
                        }
                    ),
                }
            ],
        },
    )
    assert allowed.status_code == 200
    allowed_run = allowed.json()
    assert allowed_run["child_runs"][0]["status"] == "completed"


def test_delegate_step_creates_child_run_merges_result_and_emits_events(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with delegation",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Collect competitor docs",
                            "mode": "manual",
                            "summary": "Collected competitor docs",
                            "steps": [
                                {
                                    "action_type": "search_web",
                                    "instruction": "collect competitor docs",
                                }
                            ],
                        }
                    ),
                }
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()
    assert run["status"] == "completed"
    assert run["steps"][0]["status"] == "completed"
    assert "Collected competitor docs" in run["steps"][0]["output_text"]
    assert len(run["child_runs"]) == 1
    assert run["child_runs"][0]["status"] == "completed"

    details = client.get(f"/runs/{run['id']}", headers=headers)
    assert details.status_code == 200
    assert len(details.json()["citations"]) >= 1

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    event_types = [item["type"] for item in timeline.json()["timeline"]]
    assert "delegate.started" in event_types
    assert "delegate.completed" in event_types


def test_delegate_step_records_child_failure_and_parent_continues(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=FailingExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with failing delegate",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Export delegated artifact",
                            "mode": "manual",
                            "steps": [
                                {
                                    "action_type": "export",
                                    "instruction": "export delegated artifact",
                                }
                            ],
                        }
                    ),
                },
                {"action_type": "navigate", "instruction": "open fallback page"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()
    assert run["status"] == "completed"
    assert run["steps"][0]["status"] == "completed"
    assert "failed" in run["steps"][0]["output_text"].lower()
    assert run["steps"][1]["status"] == "completed"
    assert len(run["child_runs"]) == 1
    assert run["child_runs"][0]["status"] == "failed"
    assert run["child_runs"][0]["delegation_status"] == "failed"

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    event_types = [item["type"] for item in timeline.json()["timeline"]]
    assert "delegate.started" in event_types
    assert "delegate.failed" in event_types


def test_run_adapts_when_extract_returns_no_citations(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=AdaptiveExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Research competitor pricing pages",
            "mode": "manual",
            "steps": [
                {"action_type": "navigate", "instruction": "start with homepage"},
                {"action_type": "extract", "instruction": "extract relevant pricing evidence"},
                {"action_type": "export", "instruction": "export report artifact"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()
    assert run["status"] == "completed"

    assert [step["action_type"] for step in run["steps"]] == [
        "navigate",
        "extract",
        "scroll",
        "extract",
        "export",
    ]


def test_list_runs_returns_recent_first(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    first = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "first objective",
            "mode": "manual",
            "steps": [],
        },
    )
    assert first.status_code == 200
    second = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "second objective",
            "mode": "manual",
            "steps": [],
        },
    )
    assert second.status_code == 200

    resp = client.get("/runs", headers=headers)
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) >= 2
    assert items[0]["id"] == second.json()["id"]
    assert items[1]["id"] == first.json()["id"]
    assert items[0]["status"] == "completed"


def test_list_runs_supports_filters_search_and_pagination(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    first = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "alpha invoice reconciliation",
            "mode": "manual",
            "steps": [],
        },
    )
    assert first.status_code == 200
    second = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "beta workflow submit",
            "mode": "supervised",
            "steps": [{"action_type": "type", "instruction": "enter draft values"}],
        },
    )
    assert second.status_code == 200
    assert second.json()["status"] == "pending_approval"

    pending = client.get("/runs?status=pending_approval", headers=headers)
    assert pending.status_code == 200
    pending_payload = pending.json()
    assert pending_payload["total"] == 1
    assert pending_payload["items"][0]["id"] == second.json()["id"]

    filtered = client.get("/runs?mode=manual&search=invoice", headers=headers)
    assert filtered.status_code == 200
    filtered_payload = filtered.json()
    assert filtered_payload["total"] == 1
    assert filtered_payload["items"][0]["id"] == first.json()["id"]

    paged = client.get("/runs?limit=1&offset=1", headers=headers)
    assert paged.status_code == 200
    paged_payload = paged.json()
    assert paged_payload["limit"] == 1
    assert paged_payload["offset"] == 1
    assert len(paged_payload["items"]) == 1


def test_list_runs_supports_created_at_range_filters(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "created range objective",
            "mode": "manual",
            "steps": [],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["id"]

    baseline = client.get("/runs?created_after=2000-01-01T00:00:00Z", headers=headers)
    assert baseline.status_code == 200
    baseline_payload = baseline.json()
    assert baseline_payload["total"] >= 1
    assert any(item["id"] == run_id for item in baseline_payload["items"])

    future = client.get("/runs?created_after=2100-01-01T00:00:00Z", headers=headers)
    assert future.status_code == 200
    assert future.json()["total"] == 0

    past = client.get("/runs?created_before=2000-01-01T00:00:00Z", headers=headers)
    assert past.status_code == 200
    assert past.json()["total"] == 0


def test_list_runs_rejects_invalid_filters(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    invalid_status = client.get("/runs?status=invalid-status", headers=headers)
    assert invalid_status.status_code == 400
    assert "status" in invalid_status.json()["detail"]

    invalid_mode = client.get("/runs?mode=invalid-mode", headers=headers)
    assert invalid_mode.status_code == 400
    assert "mode" in invalid_mode.json()["detail"]

    invalid_after = client.get("/runs?created_after=not-a-date", headers=headers)
    assert invalid_after.status_code == 400
    assert "created_after" in invalid_after.json()["detail"]

    invalid_before = client.get("/runs?created_before=still-not-a-date", headers=headers)
    assert invalid_before.status_code == 400
    assert "created_before" in invalid_before.json()["detail"]


def test_resume_run_continues_after_rejected_step(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "workflow requiring approval",
            "mode": "supervised",
            "steps": [
                {"action_type": "navigate", "instruction": "open page"},
                {"action_type": "type", "instruction": "enter draft"},
                {"action_type": "export", "instruction": "export report artifact"},
            ],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["id"]
    assert create.json()["status"] == "pending_approval"

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    type_step_id = pending.json()["items"][0]["step_id"]

    reject = client.post(
        f"/runs/{run_id}/approvals/{type_step_id}",
        headers=headers,
        json={"decision": "reject", "reason": "skip typing"},
    )
    assert reject.status_code == 200
    assert reject.json()["status"] == "paused"

    resume = client.post(f"/runs/{run_id}/resume", headers=headers)
    assert resume.status_code == 200
    assert resume.json()["status"] == "pending_approval"
    assert resume.json()["steps"][1]["status"] == "rejected"

    pending_after = client.get("/approvals/pending", headers=headers)
    assert pending_after.status_code == 200
    assert pending_after.json()["items"][0]["run_id"] == run_id
    assert pending_after.json()["items"][0]["action_type"] == "export"


def test_retry_failed_steps_reexecutes_run(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=FlakyExportExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "retry export after transient failure",
            "mode": "manual",
            "steps": [
                {"action_type": "navigate", "instruction": "open docs"},
                {"action_type": "export", "instruction": "export report artifact"},
            ],
        },
    )
    assert create.status_code == 200
    run_id = create.json()["id"]
    assert create.json()["status"] == "failed"

    retry = client.post(f"/runs/{run_id}/retry", headers=headers)
    assert retry.status_code == 200
    assert retry.json()["status"] == "completed"
    assert retry.json()["steps"][1]["status"] == "completed"


def test_model_replanner_proposals_are_policy_checked_and_gated(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=ModelSuggestionPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Review workflow and continue",
            "mode": "supervised",
            "steps": [
                {"action_type": "navigate", "instruction": "open start page"},
                {"action_type": "extract", "instruction": "summarize current state"},
                {"action_type": "export", "instruction": "export report artifact"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert run["status"] == "pending_approval"
    assert [step["action_type"] for step in run["steps"]] == [
        "navigate",
        "extract",
        "submit",
        "export",
    ]
    assert run["steps"][2]["status"] == "pending_approval"
