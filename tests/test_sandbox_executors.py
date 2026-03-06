"""Unit tests for sandbox execution backends."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nexus_sandbox_runner.executors import (
    DEFAULT_DOCKER_IMAGE,
    DockerEphemeralExecutor,
    LocalEphemeralExecutor,
    StepRequest,
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
    command = executor.build_command(request, workspace, workspace / "result.json")

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
