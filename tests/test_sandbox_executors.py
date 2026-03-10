"""Unit tests for sandbox execution backends."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nexus_sandbox_runner.executors import (
    DEFAULT_DOCKER_IMAGE,
    DockerEphemeralExecutor,
    LocalEphemeralExecutor,
    StepRequest,
    _container_script,
    build_executor_from_env,
)


def test_build_executor_from_env_local() -> None:
    executor = build_executor_from_env({"SANDBOX_EXECUTION_BACKEND": "local"})
    assert isinstance(executor, LocalEphemeralExecutor)


def test_build_executor_from_env_docker_uses_default_pinned_image() -> None:
    executor = build_executor_from_env({"SANDBOX_EXECUTION_BACKEND": "docker"})
    assert isinstance(executor, DockerEphemeralExecutor)
    assert executor.image == DEFAULT_DOCKER_IMAGE


def test_build_executor_from_env_docker_includes_docker_host() -> None:
    executor = build_executor_from_env(
        {
            "SANDBOX_EXECUTION_BACKEND": "docker",
            "SANDBOX_DOCKER_IMAGE": DEFAULT_DOCKER_IMAGE,
            "SANDBOX_DOCKER_HOST": "tcp://nexus-sandbox-dind:2375",
        }
    )
    assert isinstance(executor, DockerEphemeralExecutor)
    assert executor.docker_host == "tcp://nexus-sandbox-dind:2375"


def test_build_executor_from_env_docker_empty_allowlist_uses_default() -> None:
    executor = build_executor_from_env(
        {
            "SANDBOX_EXECUTION_BACKEND": "docker",
            "SANDBOX_DOCKER_IMAGE": DEFAULT_DOCKER_IMAGE,
            "SANDBOX_DOCKER_ALLOWED_IMAGES": "",
        }
    )
    assert isinstance(executor, DockerEphemeralExecutor)
    assert DEFAULT_DOCKER_IMAGE in executor.allowed_images


def test_docker_executor_rejects_unpinned_image() -> None:
    with pytest.raises(ValueError, match="pinned by digest"):
        DockerEphemeralExecutor(image="python:3.13-slim")


def test_docker_executor_allows_local_unpinned_image_when_flag_enabled() -> None:
    executor = DockerEphemeralExecutor(
        image="agent-nexus-sandbox-step:local",
        allow_unpinned_local=True,
        allowed_images=["agent-nexus-sandbox-step:local"],
    )
    assert executor.image == "agent-nexus-sandbox-step:local"


def test_docker_executor_rejects_non_allowlisted_image() -> None:
    with pytest.raises(ValueError, match="not in SANDBOX_DOCKER_ALLOWED_IMAGES"):
        DockerEphemeralExecutor(
            image=DEFAULT_DOCKER_IMAGE,
            allowed_images=[
                "python:3.12-slim@sha256:1111111111111111111111111111111111111111111111111111111111111111"
            ],
        )


def test_docker_executor_rejects_invalid_browser_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported SANDBOX_BROWSER_MODE"):
        DockerEphemeralExecutor(
            image=DEFAULT_DOCKER_IMAGE,
            browser_mode="broken",
        )


def test_docker_executor_build_command_contains_isolation_flags(tmp_path: Path) -> None:
    executor = DockerEphemeralExecutor(
        image=DEFAULT_DOCKER_IMAGE,
        network="none",
        memory_limit="256m",
        cpu_limit="0.5",
    )
    request = StepRequest(
        run_id="run123",
        step_id="step123",
        action_type="extract",
        instruction="summarize sources",
    )
    workspace = tmp_path / "w"
    workspace.mkdir(parents=True, exist_ok=True)
    command = executor.build_command(
        request,
        tmp_path / "sandbox" / request.run_id,
        workspace,
        workspace / "result.json",
    )

    assert command[0] == "docker"
    assert "--rm" in command
    assert "--network" in command
    assert "none" in command
    assert "--memory" in command
    assert "256m" in command
    assert "--cpus" in command
    assert "0.5" in command
    assert "--read-only" in command
    assert "--security-opt" in command
    assert "no-new-privileges" in command
    assert "--cap-drop" in command
    assert "ALL" in command
    assert "--pids-limit" in command
    assert command.count("-v") == 2
    assert any(str(tmp_path / "sandbox" / request.run_id) in part for part in command)


def test_local_executor_persists_session_state_across_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = LocalEphemeralExecutor()
    sandbox_root = tmp_path / "sandbox"

    monkeypatch.setattr(
        "nexus_sandbox_runner.executors._search_web",
        lambda query, max_results=5: [
            {
                "url": "https://docs.example.org/start",
                "title": f"Result for {query}",
                "snippet": "Grounded search result",
            }
        ],
    )
    monkeypatch.setattr(
        "nexus_sandbox_runner.executors._fetch_url_content",
        lambda url, timeout_sec=10.0: {
            "url": url,
            "title": "Fetched page",
            "text": "This page contains grounded details for extraction.",
            "snippet": "This page contains grounded details for extraction.",
        },
    )

    search_result = executor.execute(
        StepRequest(
            run_id="run-persist",
            step_id="step-search",
            action_type="search_web",
            instruction="best docs for grounded runtime",
        ),
        sandbox_root,
    )
    fetch_result = executor.execute(
        StepRequest(
            run_id="run-persist",
            step_id="step-fetch",
            action_type="fetch_url",
            instruction="Open the best result from the current session",
        ),
        sandbox_root,
    )
    type_result = executor.execute(
        StepRequest(
            run_id="run-persist",
            step_id="step-type",
            action_type="type",
            instruction="enter email and password",
        ),
        sandbox_root,
    )

    session_path = Path(search_result.metadata["session_path"])
    session_payload = json.loads(session_path.read_text(encoding="utf-8"))

    assert search_result.citations[0]["url"] == "https://docs.example.org/start"
    assert fetch_result.citations[0]["url"] == "https://docs.example.org/start"
    assert fetch_result.metadata["current_url"] == "https://docs.example.org/start"
    assert type_result.citations[0]["url"] == "https://docs.example.org/start"
    assert type_result.metadata["current_url"] == "https://docs.example.org/start"
    assert type_result.metadata["page_title"] == "Fetched page"
    assert session_payload["current_url"] == "https://docs.example.org/start"
    assert session_payload["search_results"][0]["url"] == "https://docs.example.org/start"


@pytest.mark.parametrize("action_type", ["type", "click", "wait", "submit"])
def test_local_executor_requires_grounded_page_for_interactive_actions(
    tmp_path: Path,
    action_type: str,
) -> None:
    executor = LocalEphemeralExecutor()
    sandbox_root = tmp_path / "sandbox"

    with pytest.raises(RuntimeError, match="No grounded page available"):
        executor.execute(
            StepRequest(
                run_id="run-grounding",
                step_id=f"step-{action_type}",
                action_type=action_type,
                instruction=f"attempt {action_type} without a grounded page",
            ),
            sandbox_root,
        )


def test_local_executor_supports_workspace_file_tools(tmp_path: Path) -> None:
    executor = LocalEphemeralExecutor()
    sandbox_root = tmp_path / "sandbox"

    write_result = executor.execute(
        StepRequest(
            run_id="run-files",
            step_id="step-write",
            action_type="write_file",
            instruction=json.dumps({"path": "notes/todo.txt", "content": "TODO\nship it"}),
        ),
        sandbox_root,
    )
    list_result = executor.execute(
        StepRequest(
            run_id="run-files",
            step_id="step-list",
            action_type="list_files",
            instruction=json.dumps({"path": "."}),
        ),
        sandbox_root,
    )
    read_result = executor.execute(
        StepRequest(
            run_id="run-files",
            step_id="step-read",
            action_type="read_file",
            instruction=json.dumps({"path": "notes/todo.txt"}),
        ),
        sandbox_root,
    )
    edit_result = executor.execute(
        StepRequest(
            run_id="run-files",
            step_id="step-edit",
            action_type="edit_file",
            instruction=json.dumps(
                {"path": "notes/todo.txt", "old": "TODO", "new": "DONE"}
            ),
        ),
        sandbox_root,
    )
    reread_result = executor.execute(
        StepRequest(
            run_id="run-files",
            step_id="step-reread",
            action_type="read_file",
            instruction=json.dumps({"path": "notes/todo.txt"}),
        ),
        sandbox_root,
    )

    assert write_result.metadata["file_path"] == "notes/todo.txt"
    assert Path(write_result.artifacts[0]["sandbox_path"]).exists()
    assert "notes/todo.txt" in list_result.metadata["files"]
    assert "TODO" in read_result.output_text
    assert edit_result.metadata["changed"] is True
    assert edit_result.metadata["file_path"] == "notes/todo.txt"
    assert "DONE" in reread_result.output_text


def test_local_executor_execute_code_creates_workspace_artifacts(tmp_path: Path) -> None:
    executor = LocalEphemeralExecutor()
    sandbox_root = tmp_path / "sandbox"

    result = executor.execute(
        StepRequest(
            run_id="run-code",
            step_id="step-code",
            action_type="execute_code",
            instruction=json.dumps(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; "
                            "Path('generated.txt').write_text('artifact', encoding='utf-8'); "
                            "print('code-ok')"
                        ),
                    ]
                }
            ),
        ),
        sandbox_root,
    )
    list_result = executor.execute(
        StepRequest(
            run_id="run-code",
            step_id="step-list",
            action_type="list_files",
            instruction=json.dumps({"path": "."}),
        ),
        sandbox_root,
    )

    assert "code-ok" in result.output_text
    assert result.metadata["exit_code"] == 0
    assert "generated.txt" in result.metadata["touched_files"]
    assert len(result.artifacts) == 1
    assert Path(result.artifacts[0]["sandbox_path"]).read_text(encoding="utf-8") == "artifact"
    assert "generated.txt" in list_result.metadata["files"]


def test_docker_executor_preflight_fails_real_mode_when_browser_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DockerEphemeralExecutor(
        image=DEFAULT_DOCKER_IMAGE,
        browser_mode="real",
    )

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["docker", "info", "--format"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="27.1.0",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'playwright'",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nexus_sandbox_runner.executors.shutil.which", lambda _: "docker")
    with pytest.raises(RuntimeError, match="Playwright-capable"):
        executor.preflight()


def test_docker_executor_preflight_reports_browser_support_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = DockerEphemeralExecutor(
        image=DEFAULT_DOCKER_IMAGE,
        browser_mode="auto",
    )

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["docker", "info", "--format"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="27.1.0",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'playwright'",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nexus_sandbox_runner.executors.shutil.which", lambda _: "docker")
    preflight = executor.preflight()
    assert preflight["browser_mode"] == "auto"
    assert preflight["browser_support"].startswith("missing-playwright:")


def test_build_executor_from_env_allows_local_unpinned_image_when_enabled() -> None:
    executor = build_executor_from_env(
        {
            "SANDBOX_EXECUTION_BACKEND": "docker",
            "SANDBOX_DOCKER_IMAGE": "agent-nexus-sandbox-step:local",
            "SANDBOX_DOCKER_ALLOWED_IMAGES": "agent-nexus-sandbox-step:local",
            "SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL": "1",
        }
    )
    assert isinstance(executor, DockerEphemeralExecutor)
    assert executor.image == "agent-nexus-sandbox-step:local"


def test_docker_executor_execute_collects_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = DockerEphemeralExecutor(image=DEFAULT_DOCKER_IMAGE)
    request = StepRequest(
        run_id="run123",
        step_id="step123",
        action_type="export",
        instruction="export report",
    )

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mount = cmd[cmd.index("-v") + 1]
        host_workspace = Path(mount.rsplit(":", 1)[0])
        host_workspace.mkdir(parents=True, exist_ok=True)
        artifact_name = "step123-export.txt"
        (host_workspace / artifact_name).write_text("artifact", encoding="utf-8")
        (host_workspace / "result.json").write_text(
            json.dumps(
                {
                    "output_text": "done:export",
                    "citations": [],
                    "artifacts": [
                        {
                            "kind": "text",
                            "name": artifact_name,
                            "path": artifact_name,
                            "workspace": str(host_workspace),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nexus_sandbox_runner.executors.shutil.which", lambda _: "docker")
    result = executor.execute(request, tmp_path / "sandbox")

    assert result.output_text == "done:export"
    assert result.metadata["executor_backend"] == "docker"
    assert len(result.artifacts) == 1
    artifact_path = Path(result.artifacts[0]["sandbox_path"])
    assert artifact_path.exists()
    assert artifact_path.name == "step123-export.txt"


def test_docker_executor_sets_docker_host_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = DockerEphemeralExecutor(
        image=DEFAULT_DOCKER_IMAGE,
        docker_host="tcp://nexus-sandbox-dind:2375",
    )
    request = StepRequest(
        run_id="run123",
        step_id="step123",
        action_type="export",
        instruction="export report",
    )
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["kwargs"] = kwargs
        mount = cmd[cmd.index("-v") + 1]
        host_workspace = Path(mount.rsplit(":", 1)[0])
        host_workspace.mkdir(parents=True, exist_ok=True)
        artifact_name = "step123-export.txt"
        (host_workspace / artifact_name).write_text("artifact", encoding="utf-8")
        (host_workspace / "result.json").write_text(
            json.dumps(
                {
                    "output_text": "done:export",
                    "citations": [],
                    "artifacts": [
                        {
                            "kind": "text",
                            "name": artifact_name,
                            "path": artifact_name,
                            "workspace": str(host_workspace),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nexus_sandbox_runner.executors.shutil.which", lambda _: "docker")
    result = executor.execute(request, tmp_path / "sandbox")

    assert result.output_text == "done:export"
    run_kwargs = captured["kwargs"]
    assert isinstance(run_kwargs, dict)
    env = run_kwargs.get("env")
    assert isinstance(env, dict)
    assert env.get("DOCKER_HOST") == "tcp://nexus-sandbox-dind:2375"


def _run_container_script_step(
    tmp_path: Path,
    *,
    action_type: str,
    instruction: str,
    session: dict[str, object] | None = None,
    step_id: str = "step",
) -> tuple[dict[str, object], dict[str, object]]:
    run_dir = tmp_path / "run-data"
    workspace_dir = tmp_path / "work"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    session_path = run_dir / "session.json"
    if session is not None:
        session_path.write_text(json.dumps(session), encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "NEXUS_ACTION": action_type,
            "NEXUS_INSTRUCTION": instruction,
            "NEXUS_STEP_ID": step_id,
            "NEXUS_OUTPUT_JSON": "result.json",
            "NEXUS_SESSION_STATE": str(session_path),
            "NEXUS_BROWSER_MODE": "simulated",
            "NEXUS_BROWSER_TIMEOUT_MS": "5000",
            "NEXUS_CAPTURE_SCREENSHOT": "0",
        }
    )
    subprocess.run(
        [sys.executable, "-c", _container_script()],
        check=True,
        capture_output=True,
        text=True,
        cwd=workspace_dir,
        env=env,
        timeout=30,
    )
    payload = json.loads((workspace_dir / "result.json").read_text(encoding="utf-8"))
    saved_session = json.loads(session_path.read_text(encoding="utf-8"))
    return payload, saved_session


@pytest.mark.parametrize(
    ("action_type", "instruction"),
    [
        ("type", "enter email and password"),
        ("click", "click continue"),
        ("wait", "wait for navigation"),
        ("submit", "submit form"),
    ],
)
def test_container_script_interactive_actions_use_grounded_session_page(
    tmp_path: Path,
    action_type: str,
    instruction: str,
) -> None:
    payload, saved_session = _run_container_script_step(
        tmp_path,
        action_type=action_type,
        instruction=instruction,
        step_id=f"step-{action_type}",
        session={
            "current_url": "https://docs.example.org/start",
            "last_title": "Fetched page",
            "last_page_text": "Grounded page context for interactive actions.",
            "search_results": [
                {
                    "url": "https://docs.example.org/start",
                    "title": "Grounded result",
                    "snippet": "Grounded page context for interactive actions.",
                }
            ],
            "draft_inputs": [],
            "submitted": False,
        },
    )

    citations = payload["citations"]
    assert isinstance(citations, list)
    assert citations[0]["url"] == "https://docs.example.org/start"
    metadata = payload["metadata"]
    assert metadata["current_url"] == "https://docs.example.org/start"
    assert metadata["page_title"] == "Fetched page"
    if action_type == "type":
        assert saved_session["draft_inputs"][0]["instruction"] == instruction
    if action_type == "submit":
        assert saved_session["submitted"] is True


def test_docker_executor_ignores_artifacts_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = DockerEphemeralExecutor(image=DEFAULT_DOCKER_IMAGE)
    request = StepRequest(
        run_id="run123",
        step_id="step123",
        action_type="export",
        instruction="export report",
    )
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mount = cmd[cmd.index("-v") + 1]
        host_workspace = Path(mount.rsplit(":", 1)[0])
        host_workspace.mkdir(parents=True, exist_ok=True)
        (host_workspace / "result.json").write_text(
            json.dumps(
                {
                    "output_text": "done:export",
                    "citations": [],
                    "artifacts": [
                        {
                            "kind": "text",
                            "name": "outside.txt",
                            "path": str(outside),
                            "workspace": str(host_workspace),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nexus_sandbox_runner.executors.shutil.which", lambda _: "docker")
    result = executor.execute(request, tmp_path / "sandbox")

    assert result.artifacts == []
