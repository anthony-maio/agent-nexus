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
    browser_mode: str = "simulated",
    extra_env: dict[str, str] | None = None,
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
            "NEXUS_BROWSER_MODE": browser_mode,
            "NEXUS_BROWSER_TIMEOUT_MS": "5000",
            "NEXUS_CAPTURE_SCREENSHOT": "0",
        }
    )
    if extra_env:
        env.update(extra_env)
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


def test_container_script_real_browser_persists_and_reuses_storage_state(
    tmp_path: Path,
) -> None:
    fake_pkg = tmp_path / "fake_playwright"
    playwright_dir = fake_pkg / "playwright"
    playwright_dir.mkdir(parents=True, exist_ok=True)
    (playwright_dir / "__init__.py").write_text("", encoding="utf-8")
    (playwright_dir / "sync_api.py").write_text(
        """
import json
import os
from pathlib import Path


class _FakePage:
    def __init__(self, context):
        self._context = context
        self.url = "https://example.test/"
        self.mouse = self

    def goto(self, url, wait_until=None, timeout=None):
        self.url = url

    def wheel(self, x, y):
        return None

    def title(self):
        return "Fake title"

    def inner_text(self, selector):
        return "Fake body text for storage-state coverage."

    def screenshot(self, path, full_page=True):
        Path(path).write_bytes(b"fake-image")


class _FakeContext:
    def __init__(self, storage_state_path):
        self._storage_state_path = storage_state_path
        self._page = _FakePage(self)
        log_path = os.environ.get("FAKE_PLAYWRIGHT_LOG", "")
        if log_path:
            entry = {
                "storage_state_path": storage_state_path or "",
                "storage_state_exists": bool(storage_state_path and Path(storage_state_path).exists()),
            }
            with Path(log_path).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\\n")

    def new_page(self):
        return self._page

    def storage_state(self, path):
        Path(path).write_text(
            json.dumps({"cookies": [{"name": "session", "value": "abc"}], "origins": []}),
            encoding="utf-8",
        )

    def close(self):
        return None


class _FakeBrowser:
    def new_context(self, storage_state=None):
        return _FakeContext(storage_state)

    def close(self):
        return None


class _FakeChromium:
    def launch(self, headless=True):
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self):
        self.chromium = _FakeChromium()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def sync_playwright():
    return _FakePlaywright()
""".strip(),
        encoding="utf-8",
    )

    log_path = tmp_path / "playwright.log"
    pythonpath_parts = [str(fake_pkg)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    step_one, session_one = _run_container_script_step(
        tmp_path,
        action_type="navigate",
        instruction="open https://docs.example.org/storage",
        step_id="step-one",
        browser_mode="real",
        extra_env={
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "FAKE_PLAYWRIGHT_LOG": str(log_path),
        },
    )
    step_two, session_two = _run_container_script_step(
        tmp_path,
        action_type="inspect",
        instruction="inspect current page",
        step_id="step-two",
        browser_mode="real",
        extra_env={
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "FAKE_PLAYWRIGHT_LOG": str(log_path),
        },
    )

    assert step_one["metadata"]["current_url"] == "https://docs.example.org/storage"
    assert step_two["metadata"]["current_url"] == "https://docs.example.org/storage"
    storage_state_path = Path(str(session_two.get("browser_storage_state_path", "")))
    assert storage_state_path.exists()
    assert storage_state_path.name == "browser-storage.json"
    assert session_one.get("browser_storage_state_path") == session_two.get("browser_storage_state_path")

    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(logs) >= 2
    assert logs[0]["storage_state_path"] == ""
    assert logs[1]["storage_state_path"].endswith("browser-storage.json")
    assert logs[1]["storage_state_exists"] is True


@pytest.mark.parametrize(
    ("action_type", "instruction", "expected_event"),
    [
        ("type", "enter email into the first field", "fill"),
        ("click", "click continue", "click"),
        ("wait", "wait for navigation", "wait_for_load_state"),
        ("submit", "submit the form", "submit"),
    ],
)
def test_container_script_real_browser_executes_interactive_actions(
    tmp_path: Path,
    action_type: str,
    instruction: str,
    expected_event: str,
) -> None:
    fake_pkg = tmp_path / "fake_playwright_interactions"
    playwright_dir = fake_pkg / "playwright"
    playwright_dir.mkdir(parents=True, exist_ok=True)
    (playwright_dir / "__init__.py").write_text("", encoding="utf-8")
    (playwright_dir / "sync_api.py").write_text(
        """
import json
import os
from pathlib import Path


def _append_log(payload):
    log_path = os.environ.get("FAKE_PLAYWRIGHT_LOG", "")
    if not log_path:
        return
    with Path(log_path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\\n")


class _FakeLocator:
    def __init__(self, selector):
        self._selector = selector
        self.first = self

    def click(self, timeout=None):
        _append_log({"event": "click", "selector": self._selector, "timeout": timeout})

    def fill(self, value, timeout=None):
        _append_log(
            {"event": "fill", "selector": self._selector, "value": value, "timeout": timeout}
        )

    def evaluate(self, expression):
        _append_log({"event": "submit", "selector": self._selector, "expression": expression})
        return None


class _FakePage:
    def __init__(self):
        self.url = "https://docs.example.org/start"
        self.mouse = self

    def goto(self, url, wait_until=None, timeout=None):
        self.url = url
        _append_log({"event": "goto", "url": url, "wait_until": wait_until, "timeout": timeout})

    def wheel(self, x, y):
        _append_log({"event": "wheel", "x": x, "y": y})

    def title(self):
        return "Fake interactive page"

    def inner_text(self, selector):
        return "Interactive body text"

    def screenshot(self, path, full_page=True):
        Path(path).write_bytes(b"fake-image")

    def wait_for_load_state(self, state="load", timeout=None):
        _append_log({"event": "wait_for_load_state", "state": state, "timeout": timeout})

    def locator(self, selector):
        return _FakeLocator(selector)


class _FakeContext:
    def __init__(self, storage_state_path):
        self._storage_state_path = storage_state_path
        self._page = _FakePage()

    def new_page(self):
        return self._page

    def storage_state(self, path):
        Path(path).write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")

    def close(self):
        return None


class _FakeBrowser:
    def new_context(self, storage_state=None):
        _append_log({"event": "new_context", "storage_state": storage_state or ""})
        return _FakeContext(storage_state)

    def close(self):
        return None


class _FakeChromium:
    def launch(self, headless=True):
        _append_log({"event": "launch", "headless": headless})
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self):
        self.chromium = _FakeChromium()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def sync_playwright():
    return _FakePlaywright()
""".strip(),
        encoding="utf-8",
    )

    log_path = tmp_path / f"{action_type}-playwright.log"
    pythonpath_parts = [str(fake_pkg)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    payload, session = _run_container_script_step(
        tmp_path,
        action_type=action_type,
        instruction=instruction,
        step_id=f"step-{action_type}",
        browser_mode="real",
        session={
            "current_url": "https://docs.example.org/start",
            "last_title": "Grounded page",
            "last_page_text": "Grounded page text",
            "search_results": [],
            "draft_inputs": [],
            "submitted": False,
        },
        extra_env={
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            "FAKE_PLAYWRIGHT_LOG": str(log_path),
        },
    )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert any(event["event"] == "new_context" for event in events)
    assert any(event["event"] == expected_event for event in events)
    assert payload["metadata"]["current_url"] == "https://docs.example.org/start"
    if action_type == "type":
        assert any(event.get("value") == instruction for event in events if event["event"] == "fill")
        assert session["draft_inputs"][0]["instruction"] == instruction
    if action_type == "submit":
        assert session["submitted"] is True


def test_container_script_search_web_returns_grounded_results(tmp_path: Path) -> None:
    fake_site = tmp_path / "fake_search"
    fake_site.mkdir(parents=True, exist_ok=True)
    (fake_site / "sitecustomize.py").write_text(
        """
import urllib.request
from email.message import Message


class _FakeResponse:
    def __init__(self, body, url):
        self._body = body.encode("utf-8")
        self._url = url
        self.headers = Message()
        self.headers["Content-Type"] = "text/html; charset=utf-8"

    def read(self, amount=-1):
        if amount is None or amount < 0:
            return self._body
        return self._body[:amount]

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_REAL_URLOPEN = urllib.request.urlopen


def _fake_urlopen(request, timeout=10):
    url = request.full_url if hasattr(request, "full_url") else str(request)
    if url.startswith("https://duckduckgo.com/html/?q="):
        body = '''
        <html><body>
            <a class="result__a" href="/l/?uddg=https%3A%2F%2Fdocs.example.org%2Falpha">Alpha docs</a>
            <div class="result__snippet">Alpha grounded snippet</div>
            <a class="result__a" href="https://docs.example.org/beta">Beta docs</a>
            <div class="result__snippet">Beta grounded snippet</div>
        </body></html>
        '''
        return _FakeResponse(body, url)
    return _REAL_URLOPEN(request, timeout=timeout)


urllib.request.urlopen = _fake_urlopen
""".strip(),
        encoding="utf-8",
    )

    pythonpath_parts = [str(fake_site)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    payload, session = _run_container_script_step(
        tmp_path,
        action_type="search_web",
        instruction="grounded browser runtime docs",
        step_id="step-search",
        extra_env={"PYTHONPATH": os.pathsep.join(pythonpath_parts)},
    )

    assert payload["metadata"]["top_url"] == "https://docs.example.org/alpha"
    assert [citation["url"] for citation in payload["citations"]] == [
        "https://docs.example.org/alpha",
        "https://docs.example.org/beta",
    ]
    assert session["search_results"][0]["title"] == "Alpha docs"
    assert session["search_results"][1]["snippet"] == "Beta grounded snippet"


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


def test_docker_executor_reuses_run_workspace_across_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = DockerEphemeralExecutor(image=DEFAULT_DOCKER_IMAGE)
    request_one = StepRequest(
        run_id="run123",
        step_id="step-one",
        action_type="export",
        instruction="export one",
    )
    request_two = StepRequest(
        run_id="run123",
        step_id="step-two",
        action_type="export",
        instruction="export two",
    )
    seen_workspaces: list[str] = []

    def fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        mount = cmd[cmd.index("-v") + 1]
        host_workspace = Path(mount.rsplit(":", 1)[0])
        seen_workspaces.append(str(host_workspace))
        host_workspace.mkdir(parents=True, exist_ok=True)
        (host_workspace / "result.json").write_text(
            json.dumps({"output_text": "done:export", "citations": [], "artifacts": []}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("nexus_sandbox_runner.executors.shutil.which", lambda _: "docker")

    executor.execute(request_one, tmp_path / "sandbox")
    executor.execute(request_two, tmp_path / "sandbox")

    assert len(seen_workspaces) == 2
    assert seen_workspaces[0] == seen_workspaces[1]
    assert Path(seen_workspaces[0]).name == "workspace"
