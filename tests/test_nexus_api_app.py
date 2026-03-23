"""Integration tests for Nexus API run lifecycle."""

from __future__ import annotations

import hashlib
import json
import shutil
import weakref
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from nexus_api.adapters import ExternalToolDispatchExecutionAdapter
from nexus_api.app import create_app
from nexus_api.config import ApiSettings
from nexus_api.external_tools import StdioExternalToolInvoker, parse_external_tool_config
import nexus_api.service as api_service
from nexus_api.service import build_context
from nexus_core.models import ArtifactRecord, CitationRecord, StepDefinition, StepExecutionResult
from nexus_core.planner import RuleAdaptivePlanner


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
        metadata: dict[str, object] = {}
        if action_type in {"inspect", "type"}:
            metadata["page_affordances"] = {
                "forms_count": 1,
                "input_fields": [
                    {"tag": "input", "type": "email", "name": "email"},
                    {"tag": "textarea", "name": "message"},
                ],
                "buttons": [{"text": "Send message", "type": "submit"}],
            }
        return StepExecutionResult(
            output_text=f"done:{action_type}",
            citations=citations,
            artifacts=artifacts,
            metadata=metadata,
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
                metadata={"planner_source": "model", "planner_phase": "follow_up"},
            ),
            StepDefinition(
                action_type="submit",
                instruction="submit the prepared workflow changes",
                metadata={"planner_source": "model", "planner_phase": "follow_up"},
            ),
        ]


class BlockedFollowUpPlanner:
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
                instruction="delete the current source set",
                metadata={"planner_source": "model", "planner_phase": "follow_up"},
            )
        ]


class SafeInitialSearchPlanner:
    def __init__(self) -> None:
        self.rule = RuleAdaptivePlanner()

    async def plan_initial_steps(self, objective: str, mode: object) -> list[StepDefinition]:
        _ = mode
        return [
            StepDefinition(
                action_type="search_web",
                instruction=f"Model bootstrap search for: {objective}",
                metadata={"planner_source": "model", "planner_phase": "initial"},
            )
        ]

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, str],
        result: StepExecutionResult,
        existing_steps: list[dict[str, str]],
    ) -> list[StepDefinition]:
        return await self.rule.propose_follow_up(
            objective=objective,
            completed_step=completed_step,
            result=result,
            existing_steps=existing_steps,
        )


class UnsafeInitialTypePlanner:
    def __init__(self) -> None:
        self.rule = RuleAdaptivePlanner()

    async def plan_initial_steps(self, objective: str, mode: object) -> list[StepDefinition]:
        _ = objective, mode
        return [
            StepDefinition(
                action_type="type",
                instruction="enter draft values immediately",
                metadata={"planner_source": "model", "planner_phase": "initial"},
            )
        ]

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, str],
        result: StepExecutionResult,
        existing_steps: list[dict[str, str]],
    ) -> list[StepDefinition]:
        return await self.rule.propose_follow_up(
            objective=objective,
            completed_step=completed_step,
            result=result,
            existing_steps=existing_steps,
        )


class DelegateReplanApprovalPlanner:
    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, str],
        result: StepExecutionResult,
        existing_steps: list[dict[str, str]],
    ) -> list[StepDefinition]:
        _ = result, existing_steps
        if (
            objective == "Collect references via replanning"
            and completed_step["action_type"] == "navigate"
        ):
            return [
                StepDefinition(
                    action_type="write_file",
                    instruction=json.dumps(
                        {"path": "reports/summary.md", "content": "delegated replan"}
                    ),
                    metadata={"planner_source": "model", "planner_phase": "follow_up"},
                )
            ]
        return []


class MultiStepInitialPlanner:
    def __init__(self) -> None:
        self.rule = RuleAdaptivePlanner()

    async def plan_initial_steps(self, objective: str, mode: object) -> list[StepDefinition]:
        _ = objective, mode
        return [
            StepDefinition(
                action_type="search_web",
                instruction="collect grounded sources",
                metadata={"planner_source": "model", "planner_phase": "initial"},
            ),
            StepDefinition(
                action_type="fetch_url",
                instruction="open the first grounded source",
                metadata={"planner_source": "model", "planner_phase": "initial"},
            ),
        ]

    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, str],
        result: StepExecutionResult,
        existing_steps: list[dict[str, str]],
    ) -> list[StepDefinition]:
        return await self.rule.propose_follow_up(
            objective=objective,
            completed_step=completed_step,
            result=result,
            existing_steps=existing_steps,
        )


class NoFollowUpPlanner:
    async def propose_follow_up(
        self,
        objective: str,
        completed_step: dict[str, str],
        result: StepExecutionResult,
        existing_steps: list[dict[str, str]],
    ) -> list[StepDefinition]:
        _ = objective, completed_step, result, existing_steps
        return []


class SkillAwarePlanner:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def plan_next_steps(
        self,
        objective: str,
        mode: object,
        existing_steps: list[dict[str, object]],
        completed_step: dict[str, object] | None = None,
        result: StepExecutionResult | None = None,
        skill_context: list[dict[str, str]] | None = None,
    ) -> list[StepDefinition]:
        _ = objective, mode, existing_steps, completed_step, result
        self.calls.append(skill_context or [])
        if completed_step is not None or result is not None:
            return []
        return [
            StepDefinition(
                action_type="search_web",
                instruction="Use the resolved skill guidance before acting.",
                metadata={"planner_source": "model", "planner_phase": "initial"},
            )
        ]


class UnifiedNextStepPlanner:
    def __init__(self) -> None:
        self.rule = RuleAdaptivePlanner()

    async def plan_next_steps(
        self,
        objective: str,
        mode: object,
        existing_steps: list[dict[str, str]],
        completed_step: dict[str, str] | None = None,
        result: StepExecutionResult | None = None,
    ) -> list[StepDefinition]:
        _ = mode
        if completed_step is None or result is None:
            return [
                StepDefinition(
                    action_type="search_web",
                    instruction=f"Unified bootstrap search for: {objective}",
                    metadata={"planner_source": "model", "planner_phase": "initial"},
                )
            ]
        return await self.rule.propose_follow_up(
            objective=objective,
            completed_step=completed_step,
            result=result,
            existing_steps=existing_steps,
        )


class EndlessFollowUpPlanner:
    async def plan_next_steps(
        self,
        objective: str,
        mode: object,
        existing_steps: list[dict[str, str]],
        completed_step: dict[str, str] | None = None,
        result: StepExecutionResult | None = None,
    ) -> list[StepDefinition]:
        _ = mode, existing_steps, result
        if completed_step is None:
            return [
                StepDefinition(
                    action_type="search_web",
                    instruction=f"bootstrap {objective}",
                    metadata={"planner_source": "model", "planner_phase": "initial"},
                )
            ]
        return [
            StepDefinition(
                action_type="extract",
                instruction="loop again",
                metadata={"planner_source": "model", "planner_phase": "follow_up"},
            )
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


class ChartWorkspaceDiscoveryExecutionAdapter(FakeExecutionAdapter):
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "list_files":
            return StepExecutionResult(
                output_text="done:list_files",
                citations=[],
                artifacts=[],
                metadata={"files": ["data/sales.csv"]},
            )
        if action_type == "read_file":
            return StepExecutionResult(
                output_text="date,revenue\n2026-03-01,120\n2026-03-02,140",
                citations=[],
                artifacts=[],
                metadata={"file_path": "data/sales.csv"},
            )
        return await super().execute_step(run_id, step_id, action_type, instruction)


class CodeWorkspaceDiscoveryExecutionAdapter(FakeExecutionAdapter):
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "list_files":
            return StepExecutionResult(
                output_text="done:list_files",
                citations=[],
                artifacts=[],
                metadata={"files": ["src/payments/retry.py", "tests/test_retry.py"]},
            )
        if action_type == "read_file":
            return StepExecutionResult(
                output_text="def retry_backoff():\n    return 1",
                citations=[],
                artifacts=[],
                metadata={"file_path": "src/payments/retry.py"},
            )
        return await super().execute_step(run_id, step_id, action_type, instruction)


class ReversedCodeWorkspaceDiscoveryExecutionAdapter(FakeExecutionAdapter):
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "list_files":
            return StepExecutionResult(
                output_text="done:list_files",
                citations=[],
                artifacts=[],
                metadata={"files": ["tests/test_retry.py", "src/payments/retry.py"]},
            )
        if action_type == "read_file":
            return StepExecutionResult(
                output_text="def retry_backoff():\n    return 1",
                citations=[],
                artifacts=[],
                metadata={"file_path": "src/payments/retry.py"},
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


class FailingCodeWorkspaceExecutionAdapter(FakeExecutionAdapter):
    async def execute_step(
        self, run_id: str, step_id: str, action_type: str, instruction: str
    ) -> StepExecutionResult:
        if action_type == "list_files":
            return StepExecutionResult(
                output_text="done:list_files",
                citations=[],
                artifacts=[],
                metadata={"files": ["src/payments/retry.py", "tests/test_retry.py"]},
            )
        if action_type == "read_file":
            return StepExecutionResult(
                output_text="def retry_backoff(base_delay):\n    return base_delay",
                citations=[],
                artifacts=[],
                metadata={"file_path": "src/payments/retry.py"},
            )
        if action_type == "execute_code":
            return StepExecutionResult(
                output_text="tests failed",
                citations=[],
                artifacts=[],
                metadata={
                    "command_failed": True,
                    "exit_code": 1,
                    "stderr": "AssertionError in src/payments/retry.py:12",
                },
            )
        return await super().execute_step(run_id, step_id, action_type, instruction)


_API_DB_TEMPLATE_PATH: Path | None = None


def _sqlite_db_path(database_url: str) -> Path | None:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return None
    return Path(database_url.removeprefix(prefix))


def _ensure_api_db_template(settings: ApiSettings) -> Path | None:
    global _API_DB_TEMPLATE_PATH
    if _API_DB_TEMPLATE_PATH is not None:
        return _API_DB_TEMPLATE_PATH
    db_path = _sqlite_db_path(settings.APP_DATABASE_URL)
    if db_path is None:
        return None
    template_root = db_path.parent / ".template-db"
    template_root.mkdir(parents=True, exist_ok=True)
    template_db_path = template_root / "app-template.db"
    template_settings = settings.model_copy(
        update={
            "APP_DATABASE_URL": f"sqlite:///{template_db_path}",
            "APP_CANONICAL_WORKSPACE": str(template_root / "workspace"),
            "APP_SANDBOX_ARTIFACT_ROOT": str(template_root / "sandbox"),
        }
    )
    template_ctx = build_context(template_settings)
    template_ctx.db_engine.dispose()
    _API_DB_TEMPLATE_PATH = template_db_path
    return _API_DB_TEMPLATE_PATH


def _build_test_context(settings: ApiSettings):
    template_db_path = _ensure_api_db_template(settings)
    db_path = _sqlite_db_path(settings.APP_DATABASE_URL)
    if template_db_path is None or db_path is None:
        return build_context(settings)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    shutil.copy2(template_db_path, db_path)

    original_run_migrations = api_service.run_migrations
    try:
        api_service.run_migrations = lambda _: None
        return build_context(settings)
    finally:
        api_service.run_migrations = original_run_migrations


def _write_skill(
    root: Path,
    folder: str,
    *,
    name: str,
    description: str,
    preferred_initial_actions: str = "",
    preferred_follow_up_actions: str = "",
    verification_signals: str = "",
    required_artifact_kinds: str = "",
) -> Path:
    skill_dir = root / folder
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
    ]
    if preferred_initial_actions:
        frontmatter.append(f"preferred_initial_actions: {preferred_initial_actions}")
    if preferred_follow_up_actions:
        frontmatter.append(f"preferred_follow_up_actions: {preferred_follow_up_actions}")
    if verification_signals:
        frontmatter.append(f"verification_signals: {verification_signals}")
    if required_artifact_kinds:
        frontmatter.append(f"required_artifact_kinds: {required_artifact_kinds}")
    frontmatter.extend(["---", ""])
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            frontmatter
            + [
                f"# {name}",
                "",
                description,
            ]
        ),
        encoding="utf-8",
    )
    return skill_dir / "SKILL.md"


def _write_fake_synthesis_project(root: Path) -> None:
    package_dir = root / "synthesis"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "from .client import SynthesisClient\n",
        encoding="utf-8",
    )
    (package_dir / "client.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "",
                "",
                "class _Result:",
                "    def __init__(self, payload):",
                "        self._payload = payload",
                "",
                "    def to_dict(self):",
                "        return dict(self._payload)",
                "",
                "",
                "class SynthesisClient:",
                "    def __init__(self, provider_type='mock', canonical_repo_path=None, host_root=None, **kwargs):",
                "        _ = provider_type, canonical_repo_path, kwargs",
                "        self.host_root = Path(host_root).expanduser()",
                "        self.host_root.mkdir(parents=True, exist_ok=True)",
                "",
                "    async def acquire_skill(self, intent, requirements=''):",
                "        skill_dir = self.host_root / 'synthesized-skill'",
                "        skill_dir.mkdir(parents=True, exist_ok=True)",
                "        (skill_dir / 'SKILL.md').write_text(",
                "            '\\n'.join([",
                "                '---',",
                "                'name: synthesized-skill',",
                "                'description: Acquired via fake Synthesis bridge.',",
                "                '---',",
                "                '',",
                "                '# synthesized-skill',",
                "                '',",
                "                f'Intent: {intent}',",
                "                f'Requirements: {requirements}',",
                "            ]),",
                "            encoding='utf-8',",
                "        )",
                "        return _Result({",
                "            'success': True,",
                "            'method': 'canonical_skill',",
                "            'primary_skill': {'name': 'synthesized-skill'},",
                "            'activation_message': 'Installed via fake Synthesis',",
                "        })",
            ]
        ),
        encoding="utf-8",
    )


def _write_fake_stdio_mcp_server(root: Path) -> Path:
    script_path = root / "fake_mcp_server.py"
    script_path.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                "",
                "for line in sys.stdin:",
                "    raw = line.strip()",
                "    if not raw:",
                "        continue",
                "    req = json.loads(raw)",
                "    req_id = req.get('id')",
                "    method = req.get('method')",
                "    params = req.get('params') or {}",
                "    if method == 'initialize':",
                "        resp = {",
                "            'jsonrpc': '2.0',",
                "            'id': req_id,",
                "            'result': {",
                "                'protocolVersion': '2024-11-05',",
                "                'serverInfo': {'name': 'fake-mcp', 'version': '1.0'},",
                "                'capabilities': {'tools': {}},",
                "            },",
                "        }",
                "    elif method == 'tools/call':",
                "        resp = {",
                "            'jsonrpc': '2.0',",
                "            'id': req_id,",
                "            'result': {",
                "                'content': [",
                "                    {",
                "                        'type': 'text',",
                "                        'text': json.dumps({'neo4j': 'offline', 'echo': params.get('arguments') or {}}),",
                "                    }",
                "                ]",
                "            },",
                "        }",
                "    else:",
                "        resp = {'jsonrpc': '2.0', 'id': req_id, 'error': {'code': -32601, 'message': method}}",
                "    sys.stdout.write(json.dumps(resp) + '\\n')",
                "    sys.stdout.flush()",
            ]
        ),
        encoding="utf-8",
    )
    return script_path


class ManagedTestClient(TestClient):
    def __init__(self, app: FastAPI, cleanup: callable | None = None) -> None:
        super().__init__(app)
        self._nexus_cleanup = cleanup
        self._nexus_cleanup_finalizer = weakref.finalize(self, cleanup) if cleanup else None

    def close(self) -> None:
        try:
            super().close()
        finally:
            cleanup = self._nexus_cleanup
            self._nexus_cleanup = None
            if cleanup is not None:
                cleanup()
            finalizer = getattr(self, "_nexus_cleanup_finalizer", None)
            if finalizer is not None and finalizer.alive:
                finalizer.detach()


def _client(
    tmp_path: Path,
    execution_adapter: FakeExecutionAdapter | None = None,
    settings_overrides: dict[str, object] | None = None,
) -> TestClient:
    settings_payload: dict[str, object] = {
        "APP_DATABASE_URL": f"sqlite:///{tmp_path / 'app.db'}",
        "APP_CANONICAL_WORKSPACE": str(tmp_path / "workspace"),
        "APP_SANDBOX_ARTIFACT_ROOT": str(tmp_path / "sandbox"),
        "APP_ADMIN_USERNAME": "admin",
        "APP_ADMIN_PASSWORD": "secret",
        "APP_SESSION_TTL_HOURS": 24,
        "APP_ENABLE_MODEL_REPLANNER": False,
    }
    if settings_overrides:
        settings_payload.update(settings_overrides)
    settings = ApiSettings(**settings_payload)
    ctx = _build_test_context(settings)
    ctx.execution_adapter = execution_adapter or FakeExecutionAdapter(tmp_path)
    return ManagedTestClient(create_app(ctx), cleanup=ctx.db_engine.dispose)


def _client_with_planner(
    tmp_path: Path,
    adaptive_planner: object,
    execution_adapter: FakeExecutionAdapter | None = None,
    settings_overrides: dict[str, object] | None = None,
) -> TestClient:
    settings_payload: dict[str, object] = {
        "APP_DATABASE_URL": f"sqlite:///{tmp_path / 'app.db'}",
        "APP_CANONICAL_WORKSPACE": str(tmp_path / "workspace"),
        "APP_SANDBOX_ARTIFACT_ROOT": str(tmp_path / "sandbox"),
        "APP_ADMIN_USERNAME": "admin",
        "APP_ADMIN_PASSWORD": "secret",
        "APP_SESSION_TTL_HOURS": 24,
        "APP_ENABLE_MODEL_REPLANNER": False,
    }
    if settings_overrides:
        settings_payload.update(settings_overrides)
    settings = ApiSettings(**settings_payload)
    ctx = _build_test_context(settings)
    ctx.execution_adapter = execution_adapter or FakeExecutionAdapter(tmp_path)
    ctx.adaptive_planner = adaptive_planner
    return ManagedTestClient(create_app(ctx), cleanup=ctx.db_engine.dispose)


def _auth_header(client: TestClient) -> dict[str, str]:
    resp = client.post("/sessions", json={"username": "admin", "password": "secret"})
    assert resp.status_code == 200
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_client_helpers_reuse_migrated_template_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_API_DB_TEMPLATE_PATH", None, raising=False)
    monkeypatch.setitem(globals(), "create_app", lambda ctx: FastAPI())

    migration_calls: list[str] = []

    def fake_run_migrations(database_url: str) -> None:
        migration_calls.append(database_url)
        db_path = Path(database_url.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch()

    monkeypatch.setattr(api_service, "run_migrations", fake_run_migrations)

    first_client = _client(tmp_path / "one")
    second_client = _client(tmp_path / "two")

    first_client.close()
    second_client.close()

    assert len(migration_calls) == 1


def test_create_app_disposes_db_engine_on_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    disposed = False

    class FakeSession:
        def __enter__(self) -> FakeSession:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb
            self.close()

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeSessionFactory:
        def __call__(self) -> FakeSession:
            return FakeSession()

    class FakeEngine:
        def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    monkeypatch.setattr("nexus_api.app.ensure_admin_user", lambda *args, **kwargs: None)

    ctx = SimpleNamespace(
        settings=SimpleNamespace(APP_ADMIN_USERNAME="admin", APP_ADMIN_PASSWORD="secret"),
        session_factory=FakeSessionFactory(),
        db_engine=FakeEngine(),
        skill_registry=SimpleNamespace(list_manifests=lambda: []),
        capability_resolver=SimpleNamespace(resolve_matches=lambda _: []),
        synthesis_bridge=None,
        execution_adapter=object(),
        interaction_adapter=object(),
        events=SimpleNamespace(subscribe=lambda _: None, unsubscribe=lambda *_: None),
        adaptive_planner=object(),
    )

    with TestClient(create_app(ctx)):
        pass

    assert disposed is True


def test_client_helpers_close_and_gc_dispose_db_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disposed = False
    finalized: list[weakref.finalize] = []

    class FakeEngine:
        def dispose(self) -> None:
            nonlocal disposed
            disposed = True

    fake_ctx = SimpleNamespace(
        db_engine=FakeEngine(),
        execution_adapter=None,
    )

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_build_test_context", lambda settings: fake_ctx)
    monkeypatch.setitem(globals(), "create_app", lambda ctx: FastAPI())

    client = _client(tmp_path)
    finalizer = getattr(client, "_nexus_cleanup_finalizer", None)
    if finalizer is not None:
        finalized.append(finalizer)
    client.close()

    assert disposed is True
    assert finalized and finalized[0].alive is False


def test_list_skills_returns_discovered_runtime_skill_manifests(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    _write_skill(
        skill_root,
        "chart-maker",
        name="chart-maker",
        description="Generate charts from tabular data.",
    )
    client = _client(
        tmp_path,
        settings_overrides={"APP_SKILL_PATHS": str(skill_root)},
    )
    headers = _auth_header(client)

    response = client.get("/skills", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["name"] == "chart-maker"
    assert payload["items"][0]["description"] == "Generate charts from tabular data."


def test_list_tools_returns_registered_external_tools(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        settings_overrides={
            "APP_EXTERNAL_TOOL_CONFIG": json.dumps(
                [
                    {
                        "name": "mnemos.retrieve",
                        "description": "Retrieve scoped memory from Mnemos.",
                        "source": "mcp://mnemos",
                        "tags": ["memory", "retrieval"],
                    },
                    {
                        "name": "cartographer.map_repo",
                        "description": "Build a scoped repository map.",
                        "source": "mcp://cartographer",
                        "tags": ["repo", "context"],
                    },
                ]
            )
        },
    )
    headers = _auth_header(client)

    response = client.get("/tools", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["items"][0]["name"] == "cartographer.map_repo"
    assert payload["items"][0]["source"] == "mcp://cartographer"
    assert payload["items"][1]["name"] == "mnemos.retrieve"
    assert payload["items"][1]["tags"] == ["memory", "retrieval"]


def test_manual_run_executes_registered_stdio_external_tool(tmp_path: Path) -> None:
    script_path = _write_fake_stdio_mcp_server(tmp_path)
    tool_config = json.dumps(
        [
            {
                "name": "c2.status",
                "description": "Return continuity-core status.",
                "source": "mcp://continuity-core",
                "transport": {
                    "kind": "stdio",
                    "command": [sys.executable, str(script_path)],
                },
            }
        ]
    )
    client = _client(
        tmp_path,
        execution_adapter=ExternalToolDispatchExecutionAdapter(
            base_adapter=FakeExecutionAdapter(tmp_path),
            tool_registry=parse_external_tool_config(tool_config),
            tool_invoker=StdioExternalToolInvoker(timeout_sec=10.0),
        ),
        settings_overrides={
            "APP_EXTERNAL_TOOL_CONFIG": tool_config
        },
    )
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Inspect continuity-core status",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "external_tool",
                    "instruction": json.dumps({"tool_name": "c2.status", "arguments": {}}),
                }
            ],
        },
    )

    assert create.status_code == 200
    run = create.json()
    assert run["status"] == "completed"
    assert run["steps"][0]["action_type"] == "external_tool"
    assert run["steps"][0]["status"] == "completed"

    details = client.get(f"/runs/{run['id']}", headers=headers)
    assert details.status_code == 200
    payload = details.json()
    assert payload["citations"][0]["url"] == "mcp://continuity-core"
    assert payload["steps"][0]["metadata"]["external_tool"]["name"] == "c2.status"


def test_list_skills_includes_synthesis_host_and_canonical_roots(tmp_path: Path) -> None:
    synthesis_root = tmp_path / "synthesis-project"
    host_root = tmp_path / "installed-skills"
    canonical_root = tmp_path / "canonical-skills"
    _write_fake_synthesis_project(synthesis_root)
    _write_skill(
        host_root,
        "local-helper",
        name="local-helper",
        description="Installed local skill from Synthesis host root.",
    )
    _write_skill(
        canonical_root / "skills",
        "canonical-helper",
        name="canonical-helper",
        description="Curated canonical skill from Synthesis repo.",
    )

    client = _client(
        tmp_path,
        settings_overrides={
            "APP_ENABLE_SYNTHESIS": True,
            "APP_SYNTHESIS_ROOT": str(synthesis_root),
            "APP_SYNTHESIS_HOST_ROOT": str(host_root),
            "APP_SYNTHESIS_CANONICAL_REPO_PATH": str(canonical_root),
        },
    )
    headers = _auth_header(client)

    response = client.get("/skills", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    names = {item["name"] for item in payload["items"]}
    assert "local-helper" in names
    assert "canonical-helper" in names


def test_resolve_skills_returns_scored_runtime_matches(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    _write_skill(
        skill_root,
        "chart-maker",
        name="chart-maker",
        description="Generate charts from tabular data.",
        preferred_initial_actions="list_files, read_file",
    )
    _write_skill(
        skill_root,
        "browser-agent",
        name="browser-agent",
        description="Navigate websites and fill forms.",
    )
    client = _client(
        tmp_path,
        settings_overrides={"APP_SKILL_PATHS": str(skill_root)},
    )
    headers = _auth_header(client)

    response = client.get(
        "/skills/resolve",
        headers=headers,
        params={"objective": "Generate a chart from local sales data"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["objective"] == "Generate a chart from local sales data"
    assert payload["items"][0]["name"] == "chart-maker"
    assert payload["items"][0]["score"] > 0
    assert payload["items"][0]["preferred_initial_actions"] == ["list_files", "read_file"]


def test_acquire_skill_uses_synthesis_bridge_and_refreshes_registry(tmp_path: Path) -> None:
    synthesis_root = tmp_path / "synthesis-project"
    host_root = tmp_path / "installed-skills"
    _write_fake_synthesis_project(synthesis_root)
    client = _client(
        tmp_path,
        settings_overrides={
            "APP_ENABLE_SYNTHESIS": True,
            "APP_SYNTHESIS_ROOT": str(synthesis_root),
            "APP_SYNTHESIS_HOST_ROOT": str(host_root),
        },
    )
    headers = _auth_header(client)

    acquire = client.post(
        "/skills/acquire",
        headers=headers,
        json={
            "intent": "parse csv files",
            "requirements": "Prefer reusable skill packages",
        },
    )

    assert acquire.status_code == 200
    payload = acquire.json()
    assert payload["success"] is True
    assert payload["primary_skill"]["name"] == "synthesized-skill"

    listed = client.get("/skills", headers=headers)
    assert listed.status_code == 200
    names = {item["name"] for item in listed.json()["items"]}
    assert "synthesized-skill" in names


def test_run_auto_acquires_skill_from_synthesis_when_no_local_match_exists(tmp_path: Path) -> None:
    synthesis_root = tmp_path / "synthesis-project"
    host_root = tmp_path / "installed-skills"
    _write_fake_synthesis_project(synthesis_root)
    client = _client(
        tmp_path,
        settings_overrides={
            "APP_ENABLE_SYNTHESIS": True,
            "APP_SYNTHESIS_ROOT": str(synthesis_root),
            "APP_SYNTHESIS_HOST_ROOT": str(host_root),
            "APP_SYNTHESIS_PROVIDER_TYPE": "mock",
            "APP_AUTO_ACQUIRE_SKILLS": True,
            "APP_SKILL_PATHS": str(tmp_path / "empty-skills"),
        },
    )
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parse csv files into a reusable skill-backed workflow",
            "mode": "manual",
        },
    )

    assert create.status_code == 200
    run = create.json()
    assert run["metadata"]["capability_state"]["skill_source"] == "synthesis_acquisition"
    assert run["metadata"]["capability_state"]["skill_names"] == ["synthesized-skill"]

    listed = client.get("/skills", headers=headers)
    assert listed.status_code == 200
    names = {item["name"] for item in listed.json()["items"]}
    assert "synthesized-skill" in names


def test_run_planning_annotates_resolved_skill_context(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    _write_skill(
        skill_root,
        "chart-maker",
        name="chart-maker",
        description="Generate charts from tabular data.",
        verification_signals="artifact, citations",
        required_artifact_kinds="chart",
    )
    planner = SkillAwarePlanner()
    client = _client_with_planner(
        tmp_path,
        adaptive_planner=planner,
        settings_overrides={"APP_SKILL_PATHS": str(skill_root)},
    )
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Generate a chart from CSV sales data and summarize it",
            "mode": "supervised",
        },
    )

    assert create.status_code == 200
    run = create.json()
    assert planner.calls[0][0]["name"] == "chart-maker"
    assert run["steps"][0]["metadata"]["skill_source"] == "capability_resolver"
    assert run["steps"][0]["metadata"]["skill_names"] == ["chart-maker"]
    assert run["steps"][0]["metadata"]["verification_signals"] == ["artifact", "citations"]
    assert run["steps"][0]["metadata"]["required_artifact_kinds"] == ["chart"]
    assert run["metadata"]["capability_state"]["skill_source"] == "capability_resolver"
    assert run["metadata"]["capability_state"]["skill_names"] == ["chart-maker"]
    assert run["metadata"]["capability_state"]["resolved_skill_count"] == 1
    assert run["metadata"]["capability_state"]["verification_signals"] == ["artifact", "citations"]
    assert run["metadata"]["capability_state"]["required_artifact_kinds"] == ["chart"]

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    capability_events = [
        item for item in timeline.json()["timeline"] if item["type"] == "run.capability"
    ]
    assert capability_events[-1]["skill_names"] == ["chart-maker"]
    assert capability_events[-1]["verification_signals"] == ["artifact", "citations"]


def test_skill_preferred_initial_actions_bias_rule_bootstrap(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    _write_skill(
        skill_root,
        "chart-maker",
        name="chart-maker",
        description="Generate charts from local CSV files.",
        preferred_initial_actions="list_files, read_file",
    )
    client = _client(
        tmp_path,
        execution_adapter=ChartWorkspaceDiscoveryExecutionAdapter(tmp_path),
        settings_overrides={"APP_SKILL_PATHS": str(skill_root)},
    )
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Generate a chart from local sales data and summarize it",
            "mode": "supervised",
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
    assert run["steps"][0]["metadata"]["skill_names"] == ["chart-maker"]
    assert run["metadata"]["capability_state"]["skill_names"] == ["chart-maker"]


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
    failed_run = approve.json()
    assert failed_run["status"] == "failed"
    failed_step = next(step for step in failed_run["steps"] if step["id"] == step_id)
    assert failed_step["metadata"]["kernel_decision"] == "fail"
    assert failed_step["metadata"]["retryable"] is True


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
    assert run["steps"][0]["metadata"]["planner_source"] == "rule"
    assert run["steps"][0]["metadata"]["planner_phase"] == "initial"
    assert [step["metadata"]["planner_phase"] for step in run["steps"]] == [
        "initial",
        "follow_up",
        "follow_up",
        "follow_up",
    ]
    assert run["status"] == "pending_approval"
    assert run["metadata"]["kernel_state"]["phase"] == "awaiting_approval"
    assert run["metadata"]["kernel_state"]["completed_step_count"] == 3
    assert run["metadata"]["kernel_state"]["pending_step_count"] == 1

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    first_step_event = next(
        item
        for item in timeline.json()["timeline"]
        if item["step_id"] == run["steps"][0]["id"] and item["type"] == "step.completed"
    )
    assert first_step_event["planner_source"] == "rule"
    assert first_step_event["planner_phase"] == "initial"


def test_default_api_run_bootstraps_call_api_tool_loop(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Call the Circle sandbox API endpoint at https://api.example.org/v1/payments and inspect the JSON response",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "call_api",
        "extract",
        "export",
    ]
    assert [step["status"] for step in run["steps"]] == [
        "completed",
        "completed",
        "pending_approval",
    ]
    assert run["steps"][0]["metadata"]["planner_source"] == "rule"
    assert run["steps"][0]["metadata"]["planner_phase"] == "initial"
    assert run["steps"][0]["output_text"]


def test_kernel_step_budget_fails_non_converging_run(tmp_path: Path) -> None:
    client = _client_with_planner(
        tmp_path,
        adaptive_planner=EndlessFollowUpPlanner(),
        settings_overrides={"APP_KERNEL_MAX_AUTONOMOUS_STEPS": 3},
    )
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Loop forever",
            "mode": "manual",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert run["status"] == "failed"
    assert len([step for step in run["steps"] if step["status"] == "completed"]) == 3
    failed_step = next(step for step in run["steps"] if step["status"] == "failed")
    assert "autonomous step budget" in failed_step["error_text"]


def test_kernel_stops_identical_follow_up_streaks(tmp_path: Path) -> None:
    client = _client_with_planner(
        tmp_path,
        adaptive_planner=EndlessFollowUpPlanner(),
        settings_overrides={
            "APP_KERNEL_MAX_AUTONOMOUS_STEPS": 10,
            "APP_KERNEL_MAX_IDENTICAL_STEP_STREAK": 2,
        },
    )
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Loop forever",
            "mode": "manual",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert run["status"] == "failed"
    completed_actions = [step["action_type"] for step in run["steps"] if step["status"] == "completed"]
    assert completed_actions == ["search_web", "extract", "extract"]
    failed_step = next(step for step in run["steps"] if step["status"] == "failed")
    assert "repeated without progress" in failed_step["error_text"]


def test_default_research_run_timeline_includes_planner_decision_events(tmp_path: Path) -> None:
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

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    planner_events = [
        item for item in timeline.json()["timeline"] if item["type"] == "planner.decision"
    ]

    assert [item["action_type"] for item in planner_events] == [
        "search_web",
        "fetch_url",
        "extract",
        "export",
    ]
    assert planner_events[0]["planner_source"] == "rule"
    assert planner_events[0]["planner_phase"] == "initial"
    assert all(item["planner_phase"] == "follow_up" for item in planner_events[1:])
    kernel_events = [
        item for item in timeline.json()["timeline"] if item["type"] == "kernel.decision"
    ]
    assert kernel_events[0]["kernel_decision"] == "continue"
    assert kernel_events[0]["verification_result"] == "passed"
    run_kernel_events = [
        item for item in timeline.json()["timeline"] if item["type"] == "run.kernel"
    ]
    assert run_kernel_events[-1]["phase"] == "awaiting_approval"
    assert run_kernel_events[-1]["pending_step_count"] == 1


def test_run_kernel_state_includes_strategy_and_tactic(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Implement the payment retry backoff fix in the repo and update tests",
            "mode": "supervised",
        },
    )

    assert create.status_code == 200
    run = create.json()

    kernel_state = run["metadata"]["kernel_state"]
    assert kernel_state["strategy"] == "coding"
    assert kernel_state["tactic"] in {"observe", "approval", "done", "synthesize", "act"}
    assert "tactic_reason" in kernel_state

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    kernel_events = [item for item in timeline.json()["timeline"] if item["type"] == "kernel.decision"]
    assert kernel_events
    assert kernel_events[0]["kernel_decision"] in {"continue", "stop", "await_approval"}


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

    assert [step["action_type"] for step in run["steps"]] == ["navigate", "inspect", "type"]
    assert [step["status"] for step in run["steps"]] == [
        "completed",
        "completed",
        "pending_approval",
    ]
    assert [step["metadata"]["planner_phase"] for step in run["steps"]] == [
        "initial",
        "follow_up",
        "follow_up",
    ]
    assert run["status"] == "pending_approval"

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    items = pending.json()["items"]
    assert len(items) == 1
    assert items[0]["run_id"] == run["id"]
    assert items[0]["action_type"] == "type"


def test_initial_planner_safe_bootstrap_step_is_used(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=SafeInitialSearchPlanner())
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
        "search_web",
        "navigate",
        "inspect",
        "type",
    ]
    assert run["steps"][0]["instruction"].startswith("Model bootstrap search for:")
    assert run["steps"][0]["metadata"]["planner_source"] == "model"
    assert run["steps"][0]["metadata"]["planner_phase"] == "initial"
    assert [step["metadata"]["planner_source"] for step in run["steps"]] == [
        "model",
        "rule",
        "rule",
        "rule",
    ]
    assert [step["metadata"]["planner_phase"] for step in run["steps"]] == [
        "initial",
        "follow_up",
        "follow_up",
        "follow_up",
    ]
    assert run["steps"][3]["status"] == "pending_approval"
    assert run["status"] == "pending_approval"

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    first_step_event = next(
        item
        for item in timeline.json()["timeline"]
        if item["step_id"] == run["steps"][0]["id"] and item["type"] == "step.completed"
    )
    assert first_step_event["planner_source"] == "model"
    assert first_step_event["planner_phase"] == "initial"


def test_initial_planner_unsafe_bootstrap_step_falls_back_to_default_bootstrap(
    tmp_path: Path,
) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=UnsafeInitialTypePlanner())
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

    assert [step["action_type"] for step in run["steps"]] == ["navigate", "inspect", "type"]
    assert run["steps"][0]["instruction"].startswith("Navigate directly to https://example.com/contact")
    assert run["steps"][0]["metadata"]["planner_source"] == "rule"
    assert run["steps"][0]["metadata"]["planner_fallback_reason"] == "policy_rejected"
    assert [step["metadata"]["planner_phase"] for step in run["steps"]] == [
        "initial",
        "follow_up",
        "follow_up",
    ]
    assert run["steps"][2]["status"] == "pending_approval"
    assert run["status"] == "pending_approval"

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    first_step_event = next(
        item
        for item in timeline.json()["timeline"]
        if item["step_id"] == run["steps"][0]["id"] and item["type"] == "step.completed"
    )
    assert first_step_event["planner_fallback_reason"] == "policy_rejected"


def test_unified_next_step_planner_drives_initial_and_follow_up_paths(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=UnifiedNextStepPlanner())
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
    assert run["steps"][0]["instruction"].startswith("Unified bootstrap search for:")
    assert [step["metadata"]["planner_source"] for step in run["steps"]] == [
        "model",
        "rule",
        "rule",
        "rule",
    ]
    assert [step["metadata"]["planner_phase"] for step in run["steps"]] == [
        "initial",
        "follow_up",
        "follow_up",
        "follow_up",
    ]
    assert run["steps"][3]["status"] == "pending_approval"
    assert run["status"] == "pending_approval"


def test_explicit_user_steps_preserve_user_planner_provenance(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Open the provided workspace brief",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "read_file",
                    "instruction": json.dumps({"path": "workspace/brief.txt"}),
                }
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert run["steps"][0]["metadata"]["planner_source"] == "user"
    assert run["steps"][0]["metadata"]["planner_phase"] == "initial"

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    first_step_event = next(
        item
        for item in timeline.json()["timeline"]
        if item["step_id"] == run["steps"][0]["id"] and item["type"] == "step.completed"
    )
    assert first_step_event["planner_source"] == "user"
    assert first_step_event["planner_phase"] == "initial"


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


def test_default_code_task_run_bootstraps_with_workspace_discovery(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=WorkspaceDiscoveryExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Implement the payment retry backoff fix in the repo and update tests",
            "mode": "supervised",
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
    assert run["steps"][0]["instruction"] == '{"path": "."}'
    assert run["steps"][0]["status"] == "completed"
    assert run["steps"][1]["status"] == "completed"
    assert run["steps"][3]["status"] == "pending_approval"


def test_code_task_run_reads_code_then_executes_tests(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=CodeWorkspaceDiscoveryExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Implement the payment retry backoff fix in the repo and update tests",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "list_files",
        "read_file",
        "execute_code",
        "extract",
        "export",
    ]
    assert run["steps"][2]["instruction"] == '{"command": ["python", "-m", "pytest", "-q"]}'
    assert run["steps"][2]["status"] == "completed"
    assert run["steps"][4]["status"] == "pending_approval"


def test_code_task_run_prefers_source_file_when_tests_are_listed_first(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=ReversedCodeWorkspaceDiscoveryExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Implement the payment retry backoff fix in the repo and update tests",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "list_files",
        "read_file",
        "execute_code",
        "extract",
        "export",
    ]
    assert run["steps"][1]["instruction"] == '{"path": "src/payments/retry.py"}'
    assert run["steps"][1]["status"] == "completed"


def test_code_task_failed_tests_trigger_diagnostic_file_read(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=FailingCodeWorkspaceExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Implement the payment retry backoff fix in the repo and update tests",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert [step["action_type"] for step in run["steps"]] == [
        "list_files",
        "read_file",
        "execute_code",
        "read_file",
        "extract",
        "export",
    ]
    assert run["steps"][3]["instruction"] == '{"path": "src/payments/retry.py"}'
    assert run["steps"][3]["metadata"]["code_follow_up"] == "failed_test_diagnostic"
    assert run["steps"][4]["status"] == "completed"
    assert run["steps"][5]["status"] == "pending_approval"


def test_kernel_focus_switches_to_diagnose_after_failed_code_verification(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=FailingCodeWorkspaceExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Implement the payment retry backoff fix in the repo and update tests",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    diagnose_events = [
        item
        for item in timeline.json()["timeline"]
        if item["type"] == "run.kernel" and item.get("tactic") == "diagnose"
    ]

    assert diagnose_events
    assert "failed" in diagnose_events[-1]["tactic_reason"].lower()
    assert run["metadata"]["kernel_state"]["tactic"] == "approval"


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


def test_delegate_nested_delegation_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with nested delegation attempt",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Attempt nested delegation",
                            "mode": "manual",
                            "steps": [
                                {
                                    "action_type": "delegate",
                                    "instruction": json.dumps(
                                        {
                                            "role": "researcher",
                                            "objective": "Nested child",
                                            "mode": "manual",
                                            "steps": [
                                                {
                                                    "action_type": "search_web",
                                                    "instruction": "nested child search",
                                                }
                                            ],
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
    assert run["steps"][1]["status"] == "completed"
    assert run["child_runs"][0]["status"] == "failed"
    assert "nested delegation" in run["steps"][0]["output_text"].lower()

    child_detail = client.get(f"/runs/{run['child_runs'][0]['id']}", headers=headers)
    assert child_detail.status_code == 200
    assert "Nested delegation is not allowed" in child_detail.json()["steps"][0]["error_text"]


def test_delegate_output_contracts_enforce_required_evidence(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    blocked = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with evidence contract",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Collect evidence with contract",
                            "mode": "manual",
                            "context": {
                                "required_citation_count": 4,
                                "required_artifact_kinds": ["image"],
                            },
                            "steps": [
                                {
                                    "action_type": "search_web",
                                    "instruction": "collect evidence",
                                }
                            ],
                        }
                    ),
                },
                {"action_type": "navigate", "instruction": "open fallback page"},
            ],
        },
    )
    assert blocked.status_code == 200
    blocked_run = blocked.json()
    assert blocked_run["status"] == "completed"
    assert blocked_run["steps"][1]["status"] == "completed"
    assert blocked_run["child_runs"][0]["status"] == "failed"
    assert "required at least 4 citation" in blocked_run["steps"][0]["output_text"]
    assert "required artifact kind `image` was not produced" in blocked_run["steps"][0]["output_text"]

    allowed = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with satisfied evidence contract",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Collect evidence with contract",
                            "mode": "manual",
                            "context": {
                                "required_citation_count": 1,
                                "required_artifact_kinds": ["text"],
                            },
                            "steps": [
                                {
                                    "action_type": "extract",
                                    "instruction": "collect evidence",
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


def test_supervised_delegate_gates_when_child_plan_is_high_risk(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with risky delegate",
            "mode": "supervised",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "operator",
                            "objective": "Update delegated report",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports"]},
                            "steps": [
                                {
                                    "action_type": "write_file",
                                    "instruction": json.dumps(
                                        {"path": "reports/summary.md", "content": "delegated update"}
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
    assert run["status"] == "pending_approval"
    assert run["steps"][0]["status"] == "pending_approval"
    assert run["steps"][1]["status"] == "pending"
    assert run["child_runs"] == []

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    items = pending.json()["items"]
    assert len(items) == 1
    assert items[0]["action_type"] == "delegate"

    approve = client.post(
        f"/runs/{run['id']}/approvals/{items[0]['step_id']}",
        headers=headers,
        json={"decision": "approve", "reason": "bounded delegated write"},
    )
    assert approve.status_code == 200
    approved_run = approve.json()
    assert approved_run["steps"][0]["status"] == "completed"
    assert approved_run["child_runs"][0]["status"] == "completed"
    pending_after = client.get("/approvals/pending", headers=headers)
    assert pending_after.status_code == 200
    assert not any(
        item["run_id"] == run["id"] and item["action_type"] == "delegate"
        for item in pending_after.json()["items"]
    )


def test_supervised_delegate_executes_when_child_plan_is_low_risk(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with safe delegate",
            "mode": "supervised",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
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
                    ),
                },
                {"action_type": "navigate", "instruction": "open fallback page"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()
    assert run["status"] == "pending_approval"
    assert run["steps"][0]["status"] == "running"
    assert run["child_runs"][0]["status"] == "pending_approval"
    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    assert not any(
        item["run_id"] == run["id"] and item["action_type"] == "delegate"
        for item in pending.json()["items"]
    )


def test_delegate_child_autoplan_truncates_initial_plan_to_single_seed_step(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=MultiStepInitialPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run with delegated child bootstrap",
            "mode": "manual",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "researcher",
                            "objective": "Collect delegated grounded sources",
                            "mode": "manual",
                            "steps": [],
                        }
                    ),
                }
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert len(run["child_runs"]) == 1
    child = run["child_runs"][0]
    assert [step["action_type"] for step in child["steps"][:2]] == ["search_web", "fetch_url"]
    assert child["steps"][0]["metadata"]["planner_source"] == "model"
    assert child["steps"][0]["metadata"]["planner_phase"] == "initial"
    assert all(step["metadata"]["planner_phase"] == "follow_up" for step in child["steps"][1:])
    assert all(step["metadata"]["planner_source"] == "rule" for step in child["steps"][1:])


def test_supervised_delegate_replan_gates_child_approval_and_pauses_parent(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=DelegateReplanApprovalPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run waiting on delegated child approval",
            "mode": "supervised",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "operator",
                            "objective": "Collect references via replanning",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports"]},
                            "steps": [
                                {
                                    "action_type": "navigate",
                                    "instruction": "open delegated workspace context",
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
    assert run["status"] == "pending_approval"
    assert run["steps"][0]["status"] == "running"
    assert run["steps"][1]["status"] == "pending"
    assert run["child_runs"][0]["status"] == "pending_approval"

    child_id = run["child_runs"][0]["id"]
    child_detail = client.get(f"/runs/{child_id}", headers=headers)
    assert child_detail.status_code == 200
    child_run = child_detail.json()
    assert child_run["steps"][0]["status"] == "completed"
    assert child_run["steps"][1]["action_type"] == "write_file"
    assert child_run["steps"][1]["status"] == "pending_approval"

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    items = pending.json()["items"]
    assert len(items) == 1
    assert items[0]["run_id"] == child_id
    assert items[0]["action_type"] == "write_file"


def test_approving_replanned_child_step_resumes_parent_delegate_run_to_next_gate(
    tmp_path: Path,
) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=DelegateReplanApprovalPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run waiting on delegated child approval",
            "mode": "supervised",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "operator",
                            "objective": "Collect references via replanning",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports"]},
                            "steps": [
                                {
                                    "action_type": "navigate",
                                    "instruction": "open delegated workspace context",
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
    child_id = run["child_runs"][0]["id"]

    pending = client.get("/approvals/pending", headers=headers)
    assert pending.status_code == 200
    pending_item = pending.json()["items"][0]
    assert pending_item["run_id"] == child_id
    assert pending_item["action_type"] == "write_file"

    approve = client.post(
        f"/runs/{child_id}/approvals/{pending_item['step_id']}",
        headers=headers,
        json={"decision": "approve", "reason": "allow delegated follow-up write"},
    )
    assert approve.status_code == 200
    assert approve.json()["status"] == "completed"

    parent_detail = client.get(f"/runs/{run['id']}", headers=headers)
    assert parent_detail.status_code == 200
    parent_run = parent_detail.json()
    assert parent_run["status"] == "pending_approval"
    assert parent_run["steps"][0]["status"] == "completed"
    assert parent_run["steps"][1]["status"] == "completed"
    assert parent_run["steps"][2]["action_type"] == "extract"
    assert parent_run["steps"][2]["status"] == "completed"
    assert parent_run["steps"][2]["metadata"]["planner_fallback_reason"] == "no_steps"
    assert parent_run["steps"][3]["action_type"] == "export"
    assert parent_run["steps"][3]["status"] == "pending_approval"
    assert parent_run["child_runs"][0]["status"] == "completed"


def test_list_runs_counts_delegated_child_pending_approvals(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=DelegateReplanApprovalPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run waiting on delegated child approval",
            "mode": "supervised",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "operator",
                            "objective": "Collect references via replanning",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports"]},
                            "steps": [
                                {
                                    "action_type": "navigate",
                                    "instruction": "open delegated workspace context",
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

    listed = client.get("/runs", headers=headers)
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert listed.json()["total"] == 1

    parent_item = next(item for item in items if item["id"] == run["id"])
    assert parent_item["pending_approval_count"] == 1
    assert all(item["id"] != run["child_runs"][0]["id"] for item in items)

    with_children = client.get("/runs?include_children=true", headers=headers)
    assert with_children.status_code == 200
    child_items = with_children.json()["items"]
    assert with_children.json()["total"] == 2
    child_item = next(item for item in child_items if item["id"] == run["child_runs"][0]["id"])
    assert child_item["pending_approval_count"] == 1


def test_timeline_surfaces_delegate_pending_approval_state(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=DelegateReplanApprovalPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Parent run waiting on delegated child approval",
            "mode": "supervised",
            "steps": [
                {
                    "action_type": "delegate",
                    "instruction": json.dumps(
                        {
                            "role": "operator",
                            "objective": "Collect references via replanning",
                            "mode": "manual",
                            "context": {"workspace_paths": ["reports"]},
                            "steps": [
                                {
                                    "action_type": "navigate",
                                    "instruction": "open delegated workspace context",
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
    assert run["status"] == "pending_approval"

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    events = timeline.json()["timeline"]
    event_types = [item["type"] for item in events]
    assert "delegate.started" in event_types
    assert "delegate.pending_approval" in event_types

    pending_event = next(item for item in events if item["type"] == "delegate.pending_approval")
    assert "awaiting delegated approval" in pending_event["summary"]


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


def test_research_run_without_terminal_output_is_blocked_by_completion_verifier(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=NoFollowUpPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Research competitor pricing pages",
            "mode": "manual",
            "steps": [
                {"action_type": "search_web", "instruction": "find competitor pricing pages"},
                {"action_type": "fetch_url", "instruction": "open the grounded pricing result"},
                {"action_type": "extract", "instruction": "extract grounded pricing evidence"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert run["status"] == "failed"
    verification = run["metadata"]["run_verification"]
    assert verification["result"] == "blocked"
    assert verification["strategy"] == "research"
    assert "terminal" in verification["reason"].lower()

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    verification_events = [
        item for item in timeline.json()["timeline"] if item["type"] == "run.verification"
    ]
    assert verification_events
    assert verification_events[-1]["result"] == "blocked"


def test_research_run_with_terminal_output_is_verified_by_completion_layer(tmp_path: Path) -> None:
    client = _client(tmp_path)
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Research competitor pricing pages",
            "mode": "manual",
            "steps": [
                {"action_type": "search_web", "instruction": "find competitor pricing pages"},
                {"action_type": "fetch_url", "instruction": "open the grounded pricing result"},
                {"action_type": "extract", "instruction": "extract grounded pricing evidence"},
                {"action_type": "export", "instruction": "export the pricing brief"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert run["status"] == "completed"
    verification = run["metadata"]["run_verification"]
    assert verification["result"] == "verified"
    assert verification["strategy"] == "research"
    assert verification["signals"]["artifact_count"] >= 1
    assert verification["signals"]["citation_count"] >= 1

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    verification_events = [
        item for item in timeline.json()["timeline"] if item["type"] == "run.verification"
    ]
    assert verification_events
    assert verification_events[-1]["result"] == "verified"


def test_kernel_focus_switches_to_recover_after_evidence_gap(tmp_path: Path) -> None:
    client = _client(tmp_path, execution_adapter=AdaptiveExecutionAdapter(tmp_path))
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Research grounded browser runtime docs",
            "mode": "supervised",
        },
    )
    assert create.status_code == 200
    run = create.json()

    timeline = client.get(f"/runs/{run['id']}/timeline", headers=headers)
    assert timeline.status_code == 200
    recover_events = [
        item
        for item in timeline.json()["timeline"]
        if item["type"] == "run.kernel" and item.get("tactic") == "recover"
    ]

    assert recover_events
    assert "citation" in recover_events[-1]["tactic_reason"].lower()
    assert run["metadata"]["kernel_state"]["tactic"] in {"approval", "done"}


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
    retried_run = retry.json()
    assert retried_run["status"] == "completed"
    assert retried_run["steps"][1]["status"] == "completed"
    assert retried_run["steps"][1]["metadata"]["retry_count"] == 1
    assert retried_run["steps"][1]["metadata"]["kernel_decision"] == "stop"


def test_kernel_auto_retries_transient_step_failures(tmp_path: Path) -> None:
    client = _client(
        tmp_path,
        execution_adapter=FlakyExportExecutionAdapter(tmp_path),
        settings_overrides={"APP_KERNEL_MAX_STEP_RETRIES": 1},
    )
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Export a grounded report with transient failure recovery",
            "mode": "manual",
            "steps": [
                {"action_type": "search_web", "instruction": "find source"},
                {"action_type": "export", "instruction": "write final report"},
            ],
        },
    )
    assert create.status_code == 200
    run = create.json()

    assert run["status"] == "completed"
    assert run["metadata"]["kernel_state"]["phase"] == "completed"
    assert run["metadata"]["kernel_state"]["completed_step_count"] == 2
    export_step = next(step for step in run["steps"] if step["action_type"] == "export")
    assert export_step["status"] == "completed"
    assert export_step["metadata"]["retry_count"] == 1
    assert export_step["metadata"]["verification_result"] == "passed"


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


def test_model_follow_up_policy_rejection_falls_back_to_rule_replan(tmp_path: Path) -> None:
    client = _client_with_planner(tmp_path, adaptive_planner=BlockedFollowUpPlanner())
    headers = _auth_header(client)

    create = client.post(
        "/runs",
        headers=headers,
        json={
            "objective": "Research payroll automation competitors and capture citations",
            "mode": "supervised",
            "steps": [
                {"action_type": "search_web", "instruction": "find grounded sources"},
            ],
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
    assert run["steps"][1]["metadata"]["planner_source"] == "rule"
    assert run["steps"][1]["metadata"]["planner_phase"] == "follow_up"
    assert run["steps"][1]["metadata"]["planner_fallback_reason"] == "policy_rejected"
    assert run["steps"][3]["status"] == "pending_approval"
