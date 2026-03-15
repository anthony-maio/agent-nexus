"""Execution backends for sandbox-runner."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from textwrap import dedent
from typing import Any, Protocol

_ARTIFACT_ACTIONS: frozenset[str] = frozenset(
    {"extract", "write", "export", "generate_report", "generate_chart", "generate_image"}
)
_DEFAULT_SEARCH_RESULTS = 5
DEFAULT_DOCKER_IMAGE = (
    "python:3.13-slim@sha256:8bc60ca09afaa8ea0d6d1220bde073bacfedd66a4bf8129cbdc8ef0e16c8a952"
)
VALID_BROWSER_MODES: frozenset[str] = frozenset({"simulated", "auto", "real"})

_URL_RE = re.compile(r"https?://[^\s)>]+", re.IGNORECASE)
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>'
    r"(?P<title>.*?)</a>.*?"
    r'(?:<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>|'
    r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>)'
    r"(?P<snippet>.*?)</(?:a|div)>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style).*?</\1>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")
_LOW_RISK_EXECUTE_CODE_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "pytest"),
    ("pytest",),
    ("python", "-m", "unittest"),
    ("go", "test"),
    ("cargo", "test"),
    ("npm", "test"),
    ("pnpm", "test"),
    ("yarn", "test"),
    ("vitest",),
    ("npx", "vitest"),
    ("jest",),
    ("npx", "jest"),
)
_SHELL_CONTROL_TOKENS: frozenset[str] = frozenset({"&&", "||", ";", "|", ">", ">>", "<"})


class _PageAffordanceParser(HTMLParser):
    def __init__(self, *, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.forms_count = 0
        self.input_fields: list[dict[str, str]] = []
        self.buttons: list[dict[str, str]] = []
        self.links: list[dict[str, str]] = []
        self._current_button: dict[str, str] | None = None
        self._current_button_text: list[str] = []
        self._current_link: dict[str, str] | None = None
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered_tag = tag.lower()
        attr_map = {
            str(key).lower(): str(value or "").strip()
            for key, value in attrs
            if key is not None
        }
        if lowered_tag == "form":
            self.forms_count += 1
        if lowered_tag in {"input", "textarea", "select"}:
            self._capture_input(lowered_tag, attr_map)
        if lowered_tag == "button":
            self._current_button = {
                key: value
                for key, value in {
                    "tag": "button",
                    "type": attr_map.get("type", "button"),
                }.items()
                if value
            }
            self._current_button_text = []
        if lowered_tag == "input" and attr_map.get("type", "").lower() in {"submit", "button"}:
            button = {
                key: value
                for key, value in {
                    "tag": "input",
                    "type": attr_map.get("type", ""),
                    "text": attr_map.get("value")
                    or attr_map.get("aria-label")
                    or attr_map.get("title")
                    or attr_map.get("name"),
                }.items()
                if value
            }
            if button and len(self.buttons) < 5:
                self.buttons.append(button)
        if lowered_tag == "a" and attr_map.get("href"):
            href = urllib.parse.urljoin(self.base_url, attr_map["href"])
            self._current_link = {"href": href}
            self._current_link_text = []

    def handle_endtag(self, tag: str) -> None:
        lowered_tag = tag.lower()
        if lowered_tag == "button" and self._current_button is not None:
            text = _clean_html(" ".join(self._current_button_text))
            if text:
                self._current_button["text"] = text
            if self._current_button and len(self.buttons) < 5:
                self.buttons.append(self._current_button)
            self._current_button = None
            self._current_button_text = []
        if lowered_tag == "a" and self._current_link is not None:
            text = _clean_html(" ".join(self._current_link_text))
            if text:
                self._current_link["text"] = text
            if self._current_link and len(self.links) < 5:
                self.links.append(self._current_link)
            self._current_link = None
            self._current_link_text = []

    def handle_data(self, data: str) -> None:
        if self._current_button is not None:
            self._current_button_text.append(data)
        if self._current_link is not None:
            self._current_link_text.append(data)

    def _capture_input(self, tag: str, attrs: dict[str, str]) -> None:
        if len(self.input_fields) >= 5:
            return
        input_type = attrs.get("type", "text" if tag == "input" else tag).lower()
        if tag == "input" and input_type in {
            "hidden",
            "submit",
            "button",
            "checkbox",
            "radio",
            "image",
            "reset",
        }:
            return
        field = {
            key: value
            for key, value in {
                "tag": tag,
                "type": input_type,
                "name": attrs.get("name", ""),
                "placeholder": attrs.get("placeholder", ""),
                "label": attrs.get("aria-label", "") or attrs.get("title", ""),
            }.items()
            if value
        }
        if field:
            self.input_fields.append(field)


def _extract_page_affordances(body: str, *, base_url: str) -> dict[str, Any]:
    parser = _PageAffordanceParser(base_url=base_url)
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        return {
            "forms_count": 0,
            "input_fields": [],
            "buttons": [],
            "links": [],
        }
    return {
        "forms_count": parser.forms_count,
        "input_fields": parser.input_fields,
        "buttons": parser.buttons,
        "links": parser.links,
    }


@dataclass(frozen=True)
class StepRequest:
    run_id: str
    step_id: str
    action_type: str
    instruction: str


@dataclass(frozen=True)
class StepResult:
    output_text: str
    citations: list[dict[str, str]]
    artifacts: list[dict[str, str]]
    metadata: dict[str, Any]


class StepExecutor(Protocol):
    """Execution backend contract."""

    def execute(self, request: StepRequest, sandbox_root: Path) -> StepResult:
        """Execute one step and return output artifacts/citations."""


class LocalEphemeralExecutor:
    """In-process executor with persistent run-scoped workspace state."""

    backend_name = "local"

    def preflight(self) -> dict[str, str]:
        return {"status": "ok", "backend": self.backend_name}

    def execute(self, request: StepRequest, sandbox_root: Path) -> StepResult:
        run_dir = sandbox_root / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = run_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).isoformat()
        session = _load_session_state(run_dir)
        result, session = _execute_local_action(
            request=request,
            run_dir=run_dir,
            workspace_dir=workspace_dir,
            session=session,
        )
        _save_session_state(run_dir, session)

        metadata = dict(result.metadata)
        metadata.update(
            {
                "timestamp": ts,
                "sandbox_root": str(sandbox_root),
                "executor_backend": self.backend_name,
                "session_path": str(_session_path(run_dir)),
                "workspace_dir": str(workspace_dir),
                "current_url": str(session.get("current_url", "")),
            }
        )
        _write_step_metadata(
            run_dir=run_dir,
            request=request,
            timestamp=ts,
            output_text=result.output_text,
            citations=result.citations,
            artifacts=result.artifacts,
            metadata=metadata,
        )
        return StepResult(
            output_text=result.output_text,
            citations=result.citations,
            artifacts=result.artifacts,
            metadata=metadata,
        )


class DockerEphemeralExecutor:
    """Executes each step inside a throwaway Docker container."""

    backend_name = "docker"

    def __init__(
        self,
        image: str,
        timeout_sec: int = 120,
        docker_bin: str = "docker",
        docker_host: str = "",
        docker_tls_verify: bool = False,
        docker_cert_path: str = "",
        allowed_images: list[str] | None = None,
        allow_unpinned_local: bool = False,
        browser_mode: str = "auto",
        browser_timeout_ms: int = 15_000,
        capture_screenshot: bool = True,
        network: str = "none",
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 128,
    ) -> None:
        self.image = image.strip()
        self.timeout_sec = timeout_sec
        self.docker_bin = docker_bin
        self.docker_host = docker_host.strip()
        self.docker_tls_verify = docker_tls_verify
        self.docker_cert_path = docker_cert_path.strip()
        self.network = network.strip()
        self.memory_limit = memory_limit.strip()
        self.cpu_limit = cpu_limit.strip()
        self.pids_limit = pids_limit
        self.browser_mode = browser_mode.strip().lower()
        self.browser_timeout_ms = browser_timeout_ms
        self.capture_screenshot = capture_screenshot
        self.allowed_images = allowed_images or []
        self.allow_unpinned_local = allow_unpinned_local
        if not self.image:
            raise ValueError("Docker executor requires SANDBOX_DOCKER_IMAGE")
        if "@sha256:" not in self.image and not self.allow_unpinned_local:
            raise ValueError("SANDBOX_DOCKER_IMAGE must be pinned by digest (`@sha256:...`).")
        if self.allowed_images and self.image not in self.allowed_images:
            raise ValueError(
                f"Sandbox image `{self.image}` is not in SANDBOX_DOCKER_ALLOWED_IMAGES."
            )
        if self.browser_mode not in VALID_BROWSER_MODES:
            raise ValueError(f"Unsupported SANDBOX_BROWSER_MODE: {self.browser_mode}")

    def preflight(self) -> dict[str, str]:
        if shutil.which(self.docker_bin) is None:
            raise RuntimeError(
                f"Docker binary `{self.docker_bin}` is not available for sandbox backend."
            )
        try:
            proc = subprocess.run(
                [self.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
                env=self._docker_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Unable to reach Docker daemon (timed out after 10s).") from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip() or "docker info returned non-zero"
            raise RuntimeError(f"Unable to reach Docker daemon: {err}")

        version = (proc.stdout or "").strip() or "unknown"
        browser_support = "n/a"
        if self.browser_mode in {"auto", "real"}:
            browser_support = self._probe_browser_support()
            if self.browser_mode == "real" and browser_support != "ready":
                raise RuntimeError(
                    "SANDBOX_BROWSER_MODE=real requires a Playwright-capable "
                    "SANDBOX_DOCKER_IMAGE."
                )
        return {
            "status": "ok",
            "backend": self.backend_name,
            "docker_server_version": version,
            "sandbox_image": self.image,
            "browser_mode": self.browser_mode,
            "browser_support": browser_support,
        }

    def execute(self, request: StepRequest, sandbox_root: Path) -> StepResult:
        run_dir = sandbox_root / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = run_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        if shutil.which(self.docker_bin) is None:
            raise RuntimeError(
                f"Docker binary `{self.docker_bin}` is not available for sandbox backend."
            )
        result_file = workspace_dir / "result.json"
        if result_file.exists():
            result_file.unlink()
        command = self.build_command(request, run_dir, workspace_dir, result_file)
        try:
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env=self._docker_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Docker step execution timed out after {self.timeout_sec}s"
            ) from exc
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()[:500]
            raise RuntimeError(
                f"Docker step execution failed (exit={proc.returncode}): {stderr}"
            )
        if not result_file.exists():
            raise RuntimeError("Docker step execution did not produce result.json")

        payload = json.loads(result_file.read_text(encoding="utf-8"))
        output = str(payload.get("output_text", ""))
        citations = _normalize_citations(payload.get("citations", []))
        artifacts = self._collect_artifacts(
            run_dir=run_dir,
            run_id=request.run_id,
            step_workspace=workspace_dir,
            artifact_specs=payload.get("artifacts", []),
        )

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(
            {
                "timestamp": ts,
                "sandbox_root": str(sandbox_root),
                "executor_backend": self.backend_name,
                "container_image": self.image,
                "browser_mode": self.browser_mode,
            }
        )
        _write_step_metadata(
            run_dir=run_dir,
            request=request,
            timestamp=ts,
            output_text=output,
            citations=citations,
            artifacts=artifacts,
            metadata=metadata,
        )
        return StepResult(output, citations, artifacts, metadata)

    def build_command(
        self,
        request: StepRequest,
        run_dir: Path,
        step_workspace: Path,
        result_file: Path,
    ) -> list[str]:
        script_path = _ensure_container_script(run_dir)
        return [
            self.docker_bin,
            "run",
            "--rm",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.pids_limit),
            "--memory",
            self.memory_limit,
            "--cpus",
            self.cpu_limit,
            "--network",
            self.network,
            "--tmpfs",
            "/tmp:rw,size=64m,noexec,nosuid,nodev",
            "-e",
            f"NEXUS_ACTION={request.action_type.strip().lower()}",
            "-e",
            f"NEXUS_INSTRUCTION={request.instruction}",
            "-e",
            f"NEXUS_STEP_ID={request.step_id}",
            "-e",
            f"NEXUS_OUTPUT_JSON={result_file.name}",
            "-e",
            f"NEXUS_BROWSER_MODE={self.browser_mode}",
            "-e",
            f"NEXUS_BROWSER_TIMEOUT_MS={self.browser_timeout_ms}",
            "-e",
            f"NEXUS_CAPTURE_SCREENSHOT={1 if self.capture_screenshot else 0}",
            "-e",
            "NEXUS_SESSION_STATE=/run-data/session.json",
            "-v",
            f"{step_workspace}:/work",
            "-v",
            f"{run_dir}:/run-data",
            "-w",
            "/work",
            self.image,
            "python",
            f"/run-data/{script_path.name}",
        ]

    def _docker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.docker_host:
            env["DOCKER_HOST"] = self.docker_host
        env["DOCKER_TLS_VERIFY"] = "1" if self.docker_tls_verify else "0"
        if self.docker_cert_path:
            env["DOCKER_CERT_PATH"] = self.docker_cert_path
        return env

    def _collect_artifacts(
        self,
        run_dir: Path,
        run_id: str,
        step_workspace: Path,
        artifact_specs: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        artifacts: list[dict[str, str]] = []
        workspace_root = step_workspace.resolve()
        run_root = run_dir.resolve()
        for spec in artifact_specs:
            source_name_raw = str(spec.get("name", "")).strip()
            if not source_name_raw:
                continue
            source_name = Path(source_name_raw).name
            source = Path(str(spec.get("path", source_name)))
            if not source.is_absolute():
                source = workspace_root / source
            source = source.resolve()
            if not any(root == source or root in source.parents for root in (workspace_root, run_root)):
                continue
            if not source.exists():
                continue
            if run_root == source or run_root in source.parents:
                final_path = source
                rel_suffix = source.relative_to(run_root).as_posix()
            else:
                final_path = run_dir / source_name
                shutil.copy2(source, final_path)
                rel_suffix = source_name
            artifacts.append(
                {
                    "kind": str(spec.get("kind", _artifact_kind_for_path(final_path))),
                    "name": source_name,
                    "rel_path": f"{run_id}/{rel_suffix}",
                    "sandbox_path": str(final_path),
                    "sha256": _sha256(final_path),
                }
            )
        return artifacts

    def _probe_browser_support(self) -> str:
        try:
            proc = subprocess.run(
                [
                    self.docker_bin,
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    self.image,
                    "python",
                    "-c",
                    "from playwright.sync_api import sync_playwright",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
                env=self._docker_env(),
            )
        except subprocess.TimeoutExpired:
            return "probe-timeout"
        if proc.returncode == 0:
            return "ready"
        stderr = (proc.stderr or proc.stdout or "").strip() or "import failed"
        return f"missing-playwright:{stderr[:120]}"


def build_executor_from_env(env: dict[str, str]) -> StepExecutor:
    backend = env.get("SANDBOX_EXECUTION_BACKEND", "local").strip().lower()
    if backend in {"", "local"}:
        return LocalEphemeralExecutor()
    if backend == "docker":
        image = env.get("SANDBOX_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE).strip()
        allowed_images = _parse_allowed_images(
            env.get("SANDBOX_DOCKER_ALLOWED_IMAGES", DEFAULT_DOCKER_IMAGE)
        )
        return DockerEphemeralExecutor(
            image=image,
            timeout_sec=int(env.get("SANDBOX_STEP_TIMEOUT_SEC", "120")),
            docker_bin=env.get("SANDBOX_DOCKER_BIN", "docker").strip() or "docker",
            docker_host=env.get("SANDBOX_DOCKER_HOST", "").strip(),
            docker_tls_verify=(env.get("SANDBOX_DOCKER_TLS_VERIFY", "1").strip() != "0"),
            docker_cert_path=env.get("SANDBOX_DOCKER_CERT_PATH", "").strip(),
            allowed_images=allowed_images,
            allow_unpinned_local=(
                env.get("SANDBOX_DOCKER_ALLOW_UNPINNED_LOCAL", "0").strip() == "1"
            ),
            browser_mode=env.get("SANDBOX_BROWSER_MODE", "auto").strip().lower() or "auto",
            browser_timeout_ms=int(env.get("SANDBOX_BROWSER_TIMEOUT_MS", "15000")),
            capture_screenshot=(env.get("SANDBOX_CAPTURE_SCREENSHOT", "1").strip() != "0"),
            network=env.get("SANDBOX_DOCKER_NETWORK", "none").strip() or "none",
            memory_limit=env.get("SANDBOX_DOCKER_MEMORY", "512m").strip() or "512m",
            cpu_limit=env.get("SANDBOX_DOCKER_CPUS", "1.0").strip() or "1.0",
            pids_limit=int(env.get("SANDBOX_DOCKER_PIDS", "128")),
        )
    raise ValueError(f"Unsupported SANDBOX_EXECUTION_BACKEND: {backend}")


def run_executor_preflight(executor: StepExecutor) -> dict[str, str]:
    preflight = getattr(executor, "preflight", None)
    if callable(preflight):
        result = preflight()
        if isinstance(result, dict):
            return {str(k): str(v) for k, v in result.items()}
    return {"status": "ok", "backend": "unknown"}


def _parse_allowed_images(raw: str) -> list[str]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or [DEFAULT_DOCKER_IMAGE]


def _execute_local_action(
    request: StepRequest,
    run_dir: Path,
    workspace_dir: Path,
    session: dict[str, Any],
) -> tuple[StepResult, dict[str, Any]]:
    action = request.action_type.strip().lower()
    instruction = request.instruction.strip()
    citations: list[dict[str, str]] = []
    artifacts: list[dict[str, str]] = []
    metadata: dict[str, Any] = {}

    if action == "search_web":
        results = _search_web(instruction, max_results=_DEFAULT_SEARCH_RESULTS)
        session["search_results"] = results
        metadata["search_results"] = results
        metadata["top_url"] = results[0]["url"] if results else ""
        output = f"[sandbox-search] Found {len(results)} result(s) for: {instruction}"
        _append_history(session, action, instruction)
        return StepResult(output, _normalize_citations(results), artifacts, metadata), session

    if action in {"fetch_url", "navigate", "inspect", "scroll", "read"}:
        page = _resolve_or_fetch_page(
            action,
            instruction,
            session,
            refresh=action in {"fetch_url", "navigate", "scroll"},
        )
        if page:
            _record_page(session, page)
            citations = [_citation_from_page(page)]
            metadata.update(_page_metadata(page))
            verb = {
                "fetch_url": "Fetched",
                "navigate": "Navigated to",
                "inspect": "Inspected",
                "scroll": "Scrolled",
                "read": "Read",
            }[action]
            output = f"[sandbox-browser] {verb} {page['url']} ({page['title']})"
        else:
            output = f"[sandbox-browser] No grounded page available for `{instruction}`."
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "extract":
        page = _resolve_or_fetch_page(action, instruction, session, refresh=False)
        if page:
            _record_page(session, page)
            citations = [_citation_from_page(page)]
            metadata.update(_page_metadata(page))
            output = (
                f"[sandbox-browser] Evidence summary for {page['url']}: "
                f"{_text_summary(page['text'], 260)}"
            )
        else:
            output = f"[sandbox-browser] Evidence summary prepared for: {instruction}"
        artifacts.append(
            _write_text_artifact(run_dir, request.run_id, f"{request.step_id}-extract.txt", output)
        )
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "list_files":
        payload = _parse_instruction_payload(instruction)
        target = _resolve_workspace_path(
            workspace_dir,
            payload,
            default_path=".",
            must_exist=False,
            allow_directory=True,
        )
        target.mkdir(parents=True, exist_ok=True)
        files = _list_workspace_files(target, workspace_dir)
        metadata["path"] = _relative_workspace_path(target, workspace_dir, directory_hint=True)
        metadata["files"] = files
        output = "\n".join(files) if files else "[sandbox-workspace] No files found."
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "read_file":
        payload = _parse_instruction_payload(instruction)
        target = _resolve_workspace_path(
            workspace_dir,
            payload,
            must_exist=True,
            allow_directory=False,
        )
        content = target.read_text(encoding="utf-8")
        metadata["file_path"] = _relative_workspace_path(target, workspace_dir)
        metadata["bytes_read"] = len(content.encode("utf-8"))
        output = content
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "write_file":
        payload = _parse_instruction_payload(instruction)
        target = _resolve_workspace_path(
            workspace_dir,
            payload,
            must_exist=False,
            allow_directory=False,
        )
        content = str(payload.get("content", ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        metadata["file_path"] = _relative_workspace_path(target, workspace_dir)
        metadata["bytes_written"] = target.stat().st_size
        metadata["changed"] = True
        artifacts.append(_workspace_file_artifact(target, workspace_dir, request.run_id))
        output = f"[sandbox-workspace] Wrote {metadata['file_path']}"
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "edit_file":
        payload = _parse_instruction_payload(instruction)
        target = _resolve_workspace_path(
            workspace_dir,
            payload,
            must_exist=True,
            allow_directory=False,
        )
        original = target.read_text(encoding="utf-8")
        old = str(payload.get("old", ""))
        new = str(payload.get("new", ""))
        if old:
            updated = original.replace(old, new)
        else:
            updated = str(payload.get("content", original))
        changed = updated != original
        if changed:
            target.write_text(updated, encoding="utf-8")
            artifacts.append(_workspace_file_artifact(target, workspace_dir, request.run_id))
        metadata["file_path"] = _relative_workspace_path(target, workspace_dir)
        metadata["bytes_written"] = len(updated.encode("utf-8"))
        metadata["changed"] = changed
        output = (
            f"[sandbox-workspace] Updated {metadata['file_path']}"
            if changed
            else f"[sandbox-workspace] No changes applied to {metadata['file_path']}"
        )
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "execute_code":
        payload = _parse_instruction_payload(instruction)
        command = _command_from_payload(payload, instruction)
        before = _snapshot_workspace(workspace_dir)
        completed = _run_workspace_command(command, workspace_dir)
        touched_files = _changed_workspace_files(before, _snapshot_workspace(workspace_dir))
        for rel_path in touched_files:
            artifacts.append(
                _workspace_file_artifact(workspace_dir / Path(rel_path), workspace_dir, request.run_id)
            )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        metadata["command"] = _stringify_command(command)
        metadata["exit_code"] = completed.returncode
        metadata["touched_files"] = touched_files
        metadata["stdout"] = stdout
        metadata["stderr"] = stderr
        if completed.returncode != 0:
            detail = stderr or stdout or "command returned non-zero exit status"
            if _is_low_risk_execute_code_command(command):
                metadata["command_failed"] = True
                metadata["failure_mode"] = "observation"
                output = (
                    f"[sandbox-code] Test command failed (exit={completed.returncode}): "
                    f"{detail[:500]}"
                )
                _append_history(session, action, instruction)
                return StepResult(output, citations, artifacts, metadata), session
            raise RuntimeError(
                f"Sandbox code execution failed (exit={completed.returncode}): {detail[:500]}"
            )
        output = stdout or stderr or "[sandbox-code] Command completed."
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "generate_report":
        payload = _parse_instruction_payload(instruction)
        target = _resolve_workspace_path(
            workspace_dir,
            payload,
            default_path=f"reports/{request.step_id}.md",
            must_exist=False,
            allow_directory=False,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        title = str(payload.get("title") or _artifact_title_from_payload(payload, fallback="Grounded Report"))
        sources = _artifact_sources(payload, session)
        citations = _normalize_citations(sources)
        report_body = _render_report_markdown(
            title=title,
            objective=str(payload.get("objective") or instruction),
            sections=payload.get("sections", []),
            sources=sources,
            session=session,
        )
        target.write_text(report_body, encoding="utf-8")
        metadata["file_path"] = _relative_workspace_path(target, workspace_dir)
        metadata["bytes_written"] = len(report_body.encode("utf-8"))
        metadata["artifact_kind"] = "report"
        metadata["report_title"] = title
        metadata["source_citation_count"] = len(citations)
        artifacts.append(
            _workspace_file_artifact(target, workspace_dir, request.run_id, kind_override="report")
        )
        output = (
            f"[sandbox-artifact] Generated grounded report {metadata['file_path']} "
            f"from {len(citations)} source(s)."
        )
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "generate_chart":
        payload = _parse_instruction_payload(instruction)
        target = _resolve_workspace_path(
            workspace_dir,
            payload,
            default_path=f"charts/{request.step_id}.html",
            must_exist=False,
            allow_directory=False,
        )
        data = _chart_rows_from_payload(payload)
        if not data:
            raise RuntimeError("No structured chart data provided for `generate_chart`.")
        target.parent.mkdir(parents=True, exist_ok=True)
        title = str(payload.get("title") or _artifact_title_from_payload(payload, fallback="Generated Chart"))
        chart_type = str(payload.get("chart_type") or "bar").strip().lower() or "bar"
        x_key, y_key = _chart_axis_keys(payload, data)
        sources = _artifact_sources(payload, session)
        citations = _normalize_citations(sources)
        chart_body = _render_chart_html(
            title=title,
            chart_type=chart_type,
            data=data,
            x_key=x_key,
            y_key=y_key,
            sources=sources,
        )
        target.write_text(chart_body, encoding="utf-8")
        metadata["file_path"] = _relative_workspace_path(target, workspace_dir)
        metadata["bytes_written"] = len(chart_body.encode("utf-8"))
        metadata["artifact_kind"] = "chart"
        metadata["chart_type"] = chart_type
        metadata["chart_title"] = title
        metadata["data_points"] = len(data)
        metadata["x_key"] = x_key
        metadata["y_key"] = y_key
        metadata["source_citation_count"] = len(citations)
        artifacts.append(
            _workspace_file_artifact(target, workspace_dir, request.run_id, kind_override="chart")
        )
        output = (
            f"[sandbox-artifact] Generated grounded chart {metadata['file_path']} "
            f"with {len(data)} data point(s)."
        )
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    if action == "generate_image":
        payload = _parse_instruction_payload(instruction)
        target = _resolve_workspace_path(
            workspace_dir,
            payload,
            default_path=f"images/{request.step_id}.svg",
            must_exist=False,
            allow_directory=False,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        title = str(payload.get("title") or _artifact_title_from_payload(payload, fallback="Generated Image"))
        prompt = str(payload.get("prompt") or payload.get("objective") or instruction).strip()
        sources = _artifact_sources(payload, session)
        citations = _normalize_citations(sources)
        image_body = _render_image_svg(title=title, prompt=prompt, sources=sources)
        target.write_text(image_body, encoding="utf-8")
        metadata["file_path"] = _relative_workspace_path(target, workspace_dir)
        metadata["bytes_written"] = len(image_body.encode("utf-8"))
        metadata["artifact_kind"] = "image"
        metadata["image_title"] = title
        metadata["image_prompt"] = prompt
        metadata["image_provider"] = "builtin-svg"
        metadata["source_citation_count"] = len(citations)
        artifacts.append(
            _workspace_file_artifact(target, workspace_dir, request.run_id, kind_override="image")
        )
        output = (
            f"[sandbox-artifact] Generated grounded image {metadata['file_path']} "
            f"from prompt and {len(citations)} source(s)."
        )
        _append_history(session, action, instruction)
        return StepResult(output, citations, artifacts, metadata), session

    interactive_page: dict[str, str] | None = None
    if action in {"type", "click", "wait", "submit"}:
        interactive_page = _resolve_or_fetch_page(action, instruction, session, refresh=False)
        if not interactive_page:
            raise RuntimeError(
                f"No grounded page available for `{action}` action. Navigate/search/fetch first."
            )
        _record_page(session, interactive_page)
        citations = [_citation_from_page(interactive_page)]
        metadata.update(_page_metadata(interactive_page))

    if action == "type":
        draft_inputs = session.setdefault("draft_inputs", [])
        draft_inputs.append({"step_id": request.step_id, "instruction": instruction})
        metadata["draft_input_count"] = len(draft_inputs)
        target = (
            interactive_page["url"]
            if interactive_page
            else str(session.get("current_url") or "current session")
        )
        output = (
            "[sandbox-browser] Typed draft input on "
            f"{target}: {instruction}"
        )
    elif action == "click":
        target = (
            interactive_page["url"]
            if interactive_page
            else str(session.get("current_url") or "current session")
        )
        output = (
            "[sandbox-browser] Clicked the requested control on "
            f"{target}: {instruction}"
        )
    elif action == "wait":
        target = (
            interactive_page["url"]
            if interactive_page
            else str(session.get("current_url") or "current session")
        )
        output = (
            "[sandbox-browser] Waited for the page state to settle on "
            f"{target}"
        )
    elif action == "submit":
        session["submitted"] = True
        target = (
            interactive_page["url"]
            if interactive_page
            else str(session.get("current_url") or "current session")
        )
        output = (
            "[sandbox-browser] Submitted the workflow on "
            f"{target}: {instruction}"
        )
    elif action in {"write", "export"}:
        target = session.get("current_url") or instruction
        label = "Workspace draft" if action == "write" else "Export artifact"
        output = f"[sandbox-workspace] {label} prepared for: {target}"
        artifacts.append(
            _write_text_artifact(
                run_dir,
                request.run_id,
                f"{request.step_id}-{action}.txt",
                output,
            )
        )
    else:
        output = f"[sandbox] Executed action `{action}` for: {instruction}"
    metadata["current_url"] = str(session.get("current_url", ""))
    _append_history(session, action, instruction)
    return StepResult(output, citations, artifacts, metadata), session


def _parse_instruction_payload(instruction: str) -> dict[str, Any]:
    try:
        payload = json.loads(instruction)
    except json.JSONDecodeError:
        return {"raw": instruction}
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"command": payload}
    return {"raw": instruction}


def _resolve_workspace_path(
    workspace_dir: Path,
    payload: dict[str, Any],
    *,
    default_path: str = "",
    must_exist: bool,
    allow_directory: bool,
) -> Path:
    raw_path = str(payload.get("path") or payload.get("file") or default_path).strip()
    if not raw_path:
        raise RuntimeError("Workspace action requires a `path` in the instruction payload.")
    candidate = workspace_dir if raw_path in {".", "./"} else workspace_dir / raw_path
    resolved = candidate.resolve()
    workspace_root = workspace_dir.resolve()
    if resolved != workspace_root and workspace_root not in resolved.parents:
        raise RuntimeError("Workspace path must stay within the run workspace.")
    if must_exist and not resolved.exists():
        raise RuntimeError(f"Workspace path does not exist: {raw_path}")
    if not allow_directory and resolved.exists() and resolved.is_dir():
        raise RuntimeError(f"Expected a file path, received a directory: {raw_path}")
    return resolved


def _relative_workspace_path(path: Path, workspace_dir: Path, *, directory_hint: bool = False) -> str:
    workspace_root = workspace_dir.resolve()
    resolved = path.resolve()
    if resolved == workspace_root:
        return "."
    rel_path = resolved.relative_to(workspace_root).as_posix()
    if directory_hint and not rel_path:
        return "."
    return rel_path


def _list_workspace_files(target: Path, workspace_dir: Path) -> list[str]:
    resolved = target.resolve()
    if resolved.is_file():
        return [_relative_workspace_path(resolved, workspace_dir)]
    files = [
        path.relative_to(workspace_dir.resolve()).as_posix()
        for path in resolved.rglob("*")
        if path.is_file()
    ]
    return sorted(files)


def _workspace_file_artifact(
    path: Path,
    workspace_dir: Path,
    run_id: str,
    *,
    kind_override: str | None = None,
) -> dict[str, str]:
    resolved = path.resolve()
    rel_path = resolved.relative_to(workspace_dir.resolve()).as_posix()
    return {
        "kind": kind_override or _artifact_kind_for_path(resolved),
        "name": resolved.name,
        "rel_path": f"{run_id}/workspace/{rel_path}",
        "sandbox_path": str(resolved),
        "sha256": _sha256(resolved),
    }


def _artifact_kind_for_path(path: Path) -> str:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return "image"
    return "text"


def _artifact_title_from_payload(payload: dict[str, Any], *, fallback: str) -> str:
    candidate = str(payload.get("raw") or payload.get("objective") or "").strip()
    if not candidate:
        return fallback
    compact = re.sub(r"\s+", " ", candidate)
    return compact[:80]


def _artifact_sources(payload: dict[str, Any], session: dict[str, Any]) -> list[dict[str, str]]:
    raw_sources = payload.get("sources")
    if isinstance(raw_sources, list):
        return _normalize_citations(raw_sources)
    current_url = str(session.get("current_url") or "").strip()
    current_title = str(session.get("page_title") or session.get("current_title") or "").strip()
    current_excerpt = str(session.get("page_excerpt") or "").strip()
    if current_url:
        return _normalize_citations(
            [{"url": current_url, "title": current_title or current_url, "snippet": current_excerpt}]
        )
    search_results = session.get("search_results")
    if isinstance(search_results, list):
        return _normalize_citations(search_results[:5])
    return []


def _render_report_markdown(
    *,
    title: str,
    objective: str,
    sections: Any,
    sources: list[dict[str, str]],
    session: dict[str, Any],
) -> str:
    lines = [f"# {title}", ""]
    objective_text = str(objective).strip()
    if objective_text and not objective_text.startswith("{"):
        lines.extend(["## Objective", objective_text, ""])
    rendered_sections = False
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            heading = str(section.get("heading") or section.get("title") or "").strip()
            body = str(section.get("body") or section.get("content") or "").strip()
            if not heading and not body:
                continue
            rendered_sections = True
            if heading:
                lines.append(f"## {heading}")
            if body:
                lines.append(body)
            lines.append("")
    if not rendered_sections:
        summary = str(session.get("page_excerpt") or "").strip()
        if summary:
            lines.extend(["## Summary", summary, ""])
    if sources:
        lines.append("## Sources")
        for source in sources:
            url = str(source.get("url", "")).strip()
            title_text = str(source.get("title", "")).strip() or url
            snippet = str(source.get("snippet", "")).strip()
            lines.append(f"- [{title_text}]({url})")
            if snippet:
                lines.append(f"  - {snippet}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _chart_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_data = payload.get("data", payload.get("chart_data"))
    if not isinstance(raw_data, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw_data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _chart_axis_keys(payload: dict[str, Any], data: list[dict[str, Any]]) -> tuple[str, str]:
    x_key = str(payload.get("x_key") or "").strip()
    y_key = str(payload.get("y_key") or "").strip()
    if x_key and y_key:
        return x_key, y_key
    sample = data[0] if data else {}
    for key, value in sample.items():
        if not x_key and not isinstance(value, (int, float)):
            x_key = str(key)
        if not y_key and isinstance(value, (int, float)):
            y_key = str(key)
    if not x_key:
        x_key = next(iter(sample.keys()), "label")
    if not y_key:
        y_key = next(iter(sample.keys()), "value")
    return x_key, y_key


def _render_chart_html(
    *,
    title: str,
    chart_type: str,
    data: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    sources: list[dict[str, str]],
) -> str:
    max_value = max(
        (
            float(row.get(y_key, 0))
            for row in data
            if isinstance(row.get(y_key), (int, float))
        ),
        default=1.0,
    )
    rows_markup: list[str] = []
    table_rows: list[str] = []
    for row in data:
        label = html.escape(str(row.get(x_key, "")))
        raw_value = row.get(y_key, 0)
        value = float(raw_value) if isinstance(raw_value, (int, float)) else 0.0
        width = 0.0 if max_value <= 0 else (value / max_value) * 100.0
        rows_markup.append(
            "<div class=\"bar-row\">"
            f"<div class=\"bar-label\">{label}</div>"
            "<div class=\"bar-track\">"
            f"<div class=\"bar-fill\" style=\"width:{width:.2f}%\"></div>"
            "</div>"
            f"<div class=\"bar-value\">{html.escape(str(raw_value))}</div>"
            "</div>"
        )
        table_rows.append(
            f"<tr><td>{label}</td><td>{html.escape(str(raw_value))}</td></tr>"
        )
    sources_markup = "".join(
        (
            "<li>"
            f"<a href=\"{html.escape(str(source.get('url', '')))}\">"
            f"{html.escape(str(source.get('title', '') or source.get('url', '')))}</a>"
            f"<span>{html.escape(str(source.get('snippet', '')))}</span>"
            "</li>"
        )
        for source in sources
    )
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\" />"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font-family:IBM Plex Sans,Arial,sans-serif;padding:32px;background:#f5f1e8;color:#17313b;}"
        "h1{margin:0 0 8px;} .meta{color:#47606a;margin-bottom:24px;}"
        ".chart{display:grid;gap:12px;margin:24px 0;}"
        ".bar-row{display:grid;grid-template-columns:160px 1fr 80px;gap:12px;align-items:center;}"
        ".bar-track{height:18px;background:#d7e5e2;border-radius:999px;overflow:hidden;}"
        ".bar-fill{height:100%;background:#2a8c82;}"
        "table{border-collapse:collapse;width:100%;margin-top:24px;}"
        "th,td{padding:8px 10px;border-bottom:1px solid #c8d4d1;text-align:left;}"
        "ul{padding-left:20px;} li span{display:block;color:#47606a;margin-top:4px;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class=\"meta\">Type: {html.escape(chart_type)} | X: {html.escape(x_key)} | Y: {html.escape(y_key)}</p>"
        f"<section class=\"chart\">{''.join(rows_markup)}</section>"
        "<section><h2>Data</h2><table><thead>"
        f"<tr><th>{html.escape(x_key)}</th><th>{html.escape(y_key)}</th></tr>"
        f"</thead><tbody>{''.join(table_rows)}</tbody></table></section>"
        f"<section><h2>Sources</h2><ul>{sources_markup}</ul></section>"
        "</body></html>\n"
    )


def _render_image_svg(
    *,
    title: str,
    prompt: str,
    sources: list[dict[str, str]],
) -> str:
    prompt_lines = _wrap_svg_text(prompt, width=44, limit=4)
    source_lines = [
        str(source.get("title") or source.get("url") or "").strip()
        for source in sources[:2]
        if str(source.get("title") or source.get("url") or "").strip()
    ]
    escaped_title = html.escape(title)
    prompt_markup = "".join(
        f'<text x="72" y="{240 + (index * 34)}" class="prompt">{html.escape(line)}</text>'
        for index, line in enumerate(prompt_lines)
    )
    source_markup = "".join(
        f'<text x="72" y="{470 + (index * 24)}" class="source">{html.escape(line)}</text>'
        for index, line in enumerate(source_lines)
    )
    source_heading = (
        '<text x="72" y="438" class="eyebrow">Grounded by sources</text>' if source_lines else ""
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" '
        f'aria-label="{escaped_title}">'
        "<defs>"
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#f6efe2" />'
        '<stop offset="100%" stop-color="#d2ebe6" />'
        "</linearGradient>"
        '<linearGradient id="card" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#17313b" stop-opacity="0.96" />'
        '<stop offset="100%" stop-color="#25505b" stop-opacity="0.92" />'
        "</linearGradient>"
        "</defs>"
        '<rect width="1200" height="630" fill="url(#bg)" />'
        '<circle cx="1020" cy="120" r="180" fill="#90c9bc" fill-opacity="0.35" />'
        '<circle cx="160" cy="520" r="220" fill="#f0c87c" fill-opacity="0.25" />'
        '<rect x="48" y="48" width="1104" height="534" rx="36" fill="url(#card)" />'
        '<style>'
        ".eyebrow{font:600 18px IBM Plex Sans,Arial,sans-serif;letter-spacing:0.12em;text-transform:uppercase;fill:#90c9bc;}"
        ".title{font:700 54px IBM Plex Sans,Arial,sans-serif;fill:#f6efe2;}"
        ".prompt{font:400 28px IBM Plex Sans,Arial,sans-serif;fill:#d9ece7;}"
        ".source{font:400 18px IBM Plex Sans,Arial,sans-serif;fill:#bdd7d0;}"
        "</style>"
        '<text x="72" y="110" class="eyebrow">Generated image artifact</text>'
        f'<text x="72" y="182" class="title">{escaped_title}</text>'
        f"{prompt_markup}"
        f"{source_heading}"
        f"{source_markup}"
        '<path d="M860 176c74 0 134 60 134 134s-60 134-134 134-134-60-134-134 60-134 134-134Zm0 52c-45 0-82 37-82 82s37 82 82 82 82-37 82-82-37-82-82-82Z" fill="#90c9bc" fill-opacity="0.18"/>'
        '<path d="M820 330c24-60 72-106 142-136 43 33 72 78 88 136-54-22-101-20-142 8-29 19-58 16-88-8Z" fill="#f0c87c" fill-opacity="0.82"/>'
        "</svg>\n"
    )


def _wrap_svg_text(text: str, *, width: int, limit: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return ["No prompt provided."]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
            if len(lines) >= limit:
                break
            continue
        current = candidate
    if len(lines) < limit and current:
        lines.append(current)
    if len(lines) > limit:
        lines = lines[:limit]
    if len(lines) == limit and sum(len(line.split()) for line in lines) < len(words):
        lines[-1] = lines[-1][: max(0, width - 1)].rstrip() + "…"
    return lines


def _snapshot_workspace(workspace_dir: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in workspace_dir.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[path.relative_to(workspace_dir.resolve()).as_posix()] = (
            stat.st_mtime_ns,
            stat.st_size,
        )
    return snapshot


def _changed_workspace_files(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> list[str]:
    return sorted(rel_path for rel_path, state in after.items() if before.get(rel_path) != state)


def _command_from_payload(payload: dict[str, Any], instruction: str) -> str | list[str]:
    command = payload.get("command")
    if isinstance(command, list):
        return [str(part) for part in command]
    if isinstance(command, str) and command.strip():
        return command.strip()
    return instruction.strip()


def _stringify_command(command: str | list[str]) -> str:
    if isinstance(command, list):
        return " ".join(command)
    return command


def _run_workspace_command(
    command: str | list[str],
    workspace_dir: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        cwd=workspace_dir,
        shell=isinstance(command, str),
        timeout=30,
    )


def _is_low_risk_execute_code_command(command: str | list[str]) -> bool:
    normalized = _normalized_command_tokens(command)
    if not normalized:
        return False
    if any(token in _SHELL_CONTROL_TOKENS for token in normalized):
        return False
    return any(
        tuple(normalized[: len(prefix)]) == prefix for prefix in _LOW_RISK_EXECUTE_CODE_PREFIXES
    )


def _normalized_command_tokens(command: str | list[str]) -> list[str]:
    raw_tokens = command if isinstance(command, list) else str(command).split()
    normalized: list[str] = []
    for index, part in enumerate(raw_tokens):
        token = str(part).strip().lower()
        if not token:
            continue
        if index == 0:
            executable = Path(token).name.lower()
            if executable.startswith("python"):
                token = "python"
            else:
                token = executable
        normalized.append(token)
    return normalized


def _search_web(query: str, max_results: int = _DEFAULT_SEARCH_RESULTS) -> list[dict[str, str]]:
    query = " ".join(query.split())
    if not query:
        return []
    request = urllib.request.Request(
        f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return []
    results: list[dict[str, str]] = []
    for match in _DDG_RESULT_RE.finditer(body):
        url = _normalize_search_url(html.unescape(match.group("href")))
        if not url:
            continue
        title = _clean_html(match.group("title")) or url
        snippet = _clean_html(match.group("snippet")) or title
        results.append({"url": url, "title": title, "snippet": snippet})
        if len(results) >= max(1, max_results):
            break
    return results


def _fetch_url_content(url: str, timeout_sec: float = 10.0) -> dict[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw = response.read(512_000)
        charset = response.headers.get_content_charset() or "utf-8"
        body = raw.decode(charset, errors="ignore")
        final_url = response.geturl()
    match = re.search(r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
    title = _clean_html(match.group(1)) if match else final_url
    text = _extract_visible_text(body)
    return {
        "url": final_url,
        "title": title or final_url,
        "text": text,
        "snippet": _text_summary(text, 240) or title or final_url,
        "affordances": _extract_page_affordances(body, base_url=final_url),
    }


def _resolve_or_fetch_page(
    action: str,
    instruction: str,
    session: dict[str, Any],
    *,
    refresh: bool,
) -> dict[str, str] | None:
    url = _resolve_target_url(action, instruction, session)
    if not url:
        current_url = str(session.get("current_url", ""))
        text = str(session.get("last_page_text", ""))
        if current_url and text:
            return {
                "url": current_url,
                "title": str(session.get("last_title", current_url)),
                "text": text,
                "snippet": _text_summary(text, 240),
                "affordances": session.get("last_page_affordances", {}) or {},
            }
        return None
    current_url = str(session.get("current_url", ""))
    current_text = str(session.get("last_page_text", ""))
    if not refresh and url == current_url and current_text:
        return {
            "url": url,
            "title": str(session.get("last_title", url)),
            "text": current_text,
            "snippet": _text_summary(current_text, 240),
            "affordances": session.get("last_page_affordances", {}) or {},
        }
    try:
        return _fetch_url_content(url)
    except Exception:
        if url == current_url and current_text:
            return {
                "url": url,
                "title": str(session.get("last_title", url)),
                "text": current_text,
                "snippet": _text_summary(current_text, 240),
                "affordances": session.get("last_page_affordances", {}) or {},
            }
        return None


def _resolve_target_url(action: str, instruction: str, session: dict[str, Any]) -> str:
    direct = _extract_url(instruction)
    if direct:
        return direct
    search_results = session.get("search_results")
    if action in {"fetch_url", "navigate"} and isinstance(search_results, list) and search_results:
        top = search_results[0]
        if isinstance(top, dict):
            return str(top.get("url", ""))
    if session.get("current_url"):
        return str(session["current_url"])
    return ""


def _record_page(session: dict[str, Any], page: dict[str, Any]) -> None:
    session["current_url"] = page.get("url", "")
    session["last_title"] = page.get("title", "")
    session["last_page_text"] = page.get("text", "")
    session["last_page_affordances"] = page.get("affordances", {}) or {}
    _append_history(session, "page", page.get("url", ""))


def _page_metadata(page: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "current_url": page.get("url", ""),
        "page_title": page.get("title", ""),
        "page_excerpt": page.get("snippet", ""),
    }
    affordances = page.get("affordances", {})
    if isinstance(affordances, dict):
        metadata["page_affordances"] = affordances
    return metadata


def _citation_from_page(page: dict[str, str]) -> dict[str, str]:
    return {
        "url": page.get("url", ""),
        "title": page.get("title", ""),
        "snippet": page.get("snippet", ""),
    }


def _session_path(run_dir: Path) -> Path:
    return run_dir / "session.json"


def _load_session_state(run_dir: Path) -> dict[str, Any]:
    path = _session_path(run_dir)
    if not path.exists():
        return _default_session_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _default_session_state()
    session = _default_session_state()
    if isinstance(payload, dict):
        session.update(payload)
    return session


def _save_session_state(run_dir: Path, session: dict[str, Any]) -> None:
    _session_path(run_dir).write_text(json.dumps(session, indent=2), encoding="utf-8")


def _default_session_state() -> dict[str, Any]:
    return {
        "current_url": "",
        "last_title": "",
        "last_page_text": "",
        "last_page_affordances": {},
        "search_results": [],
        "history": [],
        "draft_inputs": [],
        "submitted": False,
    }


def _append_history(session: dict[str, Any], action: str, detail: str) -> None:
    history = session.setdefault("history", [])
    if not isinstance(history, list):
        history = []
        session["history"] = history
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "detail": detail[:280],
        }
    )
    del history[:-25]


def _write_text_artifact(run_dir: Path, run_id: str, name: str, content: str) -> dict[str, str]:
    path = run_dir / Path(name).name
    path.write_text(content, encoding="utf-8")
    return {
        "kind": "text",
        "name": path.name,
        "rel_path": f"{run_id}/{path.name}",
        "sandbox_path": str(path),
        "sha256": _sha256(path),
    }


def _normalize_search_url(raw_href: str) -> str:
    if raw_href.startswith("//"):
        return f"https:{raw_href}"
    if raw_href.startswith("/"):
        parsed = urllib.parse.urlparse(raw_href)
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("uddg"):
            return urllib.parse.unquote(params["uddg"][0])
        return ""
    if raw_href.startswith("http://") or raw_href.startswith("https://"):
        return raw_href
    return ""


def _extract_url(text: str) -> str:
    match = _URL_RE.search(text)
    return match.group(0) if match else ""


def _clean_html(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value))).strip()


def _extract_visible_text(body: str) -> str:
    return _clean_html(_SCRIPT_RE.sub(" ", body))[:8_000]


def _text_summary(text: str, limit: int) -> str:
    compact = _WHITESPACE_RE.sub(" ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "…"


def _ensure_container_script(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "container_step.py"
    content = _container_script()
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    return path


def _container_script() -> str:
    return dedent(
        """
        import html
        import json
        import os
        import re
        import subprocess
        import urllib.parse
        import urllib.request
        from html.parser import HTMLParser
        from pathlib import Path

        action = os.environ.get("NEXUS_ACTION", "").strip().lower()
        instruction = os.environ.get("NEXUS_INSTRUCTION", "")
        step_id = os.environ.get("NEXUS_STEP_ID", "step")
        result_path = Path(os.environ.get("NEXUS_OUTPUT_JSON", "result.json"))
        session_path = Path(os.environ.get("NEXUS_SESSION_STATE", "/run-data/session.json"))
        browser_mode = os.environ.get("NEXUS_BROWSER_MODE", "auto").strip().lower()
        browser_timeout_ms = int(os.environ.get("NEXUS_BROWSER_TIMEOUT_MS", "15000"))
        capture_screenshot = os.environ.get("NEXUS_CAPTURE_SCREENSHOT", "1") != "0"
        run_dir = session_path.parent
        workspace_dir = run_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        ddg_result_re = re.compile(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>'
            r"(?P<title>.*?)</a>.*?"
            r'(?:<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>|'
            r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>)'
            r"(?P<snippet>.*?)</(?:a|div)>",
            re.IGNORECASE | re.DOTALL,
        )

        class PageAffordanceParser(HTMLParser):
            def __init__(self, base_url):
                super().__init__()
                self.base_url = base_url
                self.forms_count = 0
                self.input_fields = []
                self.buttons = []
                self.links = []
                self.current_button = None
                self.current_button_text = []
                self.current_link = None
                self.current_link_text = []

            def handle_starttag(self, tag, attrs):
                lowered_tag = str(tag).lower()
                attr_map = {
                    str(key).lower(): str(value or "").strip()
                    for key, value in attrs
                    if key is not None
                }
                if lowered_tag == "form":
                    self.forms_count += 1
                if lowered_tag in {"input", "textarea", "select"}:
                    self.capture_input(lowered_tag, attr_map)
                if lowered_tag == "button":
                    self.current_button = {
                        key: value
                        for key, value in {
                            "tag": "button",
                            "type": attr_map.get("type", "button"),
                        }.items()
                        if value
                    }
                    self.current_button_text = []
                if lowered_tag == "input" and attr_map.get("type", "").lower() in {"submit", "button"}:
                    button = {
                        key: value
                        for key, value in {
                            "tag": "input",
                            "type": attr_map.get("type", ""),
                            "text": attr_map.get("value")
                            or attr_map.get("aria-label")
                            or attr_map.get("title")
                            or attr_map.get("name"),
                        }.items()
                        if value
                    }
                    if button and len(self.buttons) < 5:
                        self.buttons.append(button)
                if lowered_tag == "a" and attr_map.get("href"):
                    self.current_link = {"href": urllib.parse.urljoin(self.base_url, attr_map["href"])}
                    self.current_link_text = []

            def handle_endtag(self, tag):
                lowered_tag = str(tag).lower()
                if lowered_tag == "button" and self.current_button is not None:
                    text = clean_html(" ".join(self.current_button_text))
                    if text:
                        self.current_button["text"] = text
                    if self.current_button and len(self.buttons) < 5:
                        self.buttons.append(self.current_button)
                    self.current_button = None
                    self.current_button_text = []
                if lowered_tag == "a" and self.current_link is not None:
                    text = clean_html(" ".join(self.current_link_text))
                    if text:
                        self.current_link["text"] = text
                    if self.current_link and len(self.links) < 5:
                        self.links.append(self.current_link)
                    self.current_link = None
                    self.current_link_text = []

            def handle_data(self, data):
                if self.current_button is not None:
                    self.current_button_text.append(data)
                if self.current_link is not None:
                    self.current_link_text.append(data)

            def capture_input(self, tag, attrs):
                if len(self.input_fields) >= 5:
                    return
                input_type = attrs.get("type", "text" if tag == "input" else tag).lower()
                if tag == "input" and input_type in {
                    "hidden",
                    "submit",
                    "button",
                    "checkbox",
                    "radio",
                    "image",
                    "reset",
                }:
                    return
                field = {
                    key: value
                    for key, value in {
                        "tag": tag,
                        "type": input_type,
                        "name": attrs.get("name", ""),
                        "placeholder": attrs.get("placeholder", ""),
                        "label": attrs.get("aria-label", "") or attrs.get("title", ""),
                    }.items()
                    if value
                }
                if field:
                    self.input_fields.append(field)

        def load_session():
            defaults = {
                "current_url": "",
                "last_title": "",
                "last_page_text": "",
                "last_page_affordances": {},
                "search_results": [],
                "draft_inputs": [],
                "submitted": False,
                "browser_storage_state_path": "",
            }
            if not session_path.exists():
                return dict(defaults)
            try:
                payload = json.loads(session_path.read_text(encoding="utf-8"))
            except Exception:
                return dict(defaults)
            if isinstance(payload, dict):
                defaults.update(payload)
            return defaults

        def save_session(session):
            run_dir.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")

        def first_url(text):
            match = re.search(r"https?://\\S+", text or "")
            return match.group(0) if match else ""

        def storage_state_path():
            raw = str(session.get("browser_storage_state_path", "")).strip()
            if raw:
                return Path(raw)
            return run_dir / "browser-storage.json"

        def summarize_text(text, limit):
            compact = re.sub(r"\\s+", " ", str(text or "")).strip()
            if len(compact) <= limit:
                return compact
            return compact[: max(0, limit - 1)].rstrip() + "…"

        def clean_html(value):
            return re.sub(r"\\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value or ""))).strip()

        def extract_affordances(body, base_url):
            parser = PageAffordanceParser(base_url)
            try:
                parser.feed(body or "")
                parser.close()
            except Exception:
                return {
                    "forms_count": 0,
                    "input_fields": [],
                    "buttons": [],
                    "links": [],
                }
            return {
                "forms_count": parser.forms_count,
                "input_fields": parser.input_fields,
                "buttons": parser.buttons,
                "links": parser.links,
            }

        def normalize_search_url(raw_href):
            href = str(raw_href or "").strip()
            if href.startswith("//"):
                return f"https:{href}"
            if href.startswith("/"):
                parsed = urllib.parse.urlparse(href)
                params = urllib.parse.parse_qs(parsed.query)
                if params.get("uddg"):
                    return urllib.parse.unquote(params["uddg"][0])
                return ""
            if href.startswith("http://") or href.startswith("https://"):
                return href
            return ""

        def search_web(query, max_results=5):
            compact_query = " ".join(str(query or "").split())
            if not compact_query:
                return []
            req = urllib.request.Request(
                f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(compact_query)}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    body = response.read().decode("utf-8", errors="ignore")
            except Exception:
                return []
            results = []
            for match in ddg_result_re.finditer(body):
                url = normalize_search_url(html.unescape(match.group("href")))
                if not url:
                    continue
                title = clean_html(match.group("title")) or url
                snippet = clean_html(match.group("snippet")) or title
                results.append({"url": url, "title": title, "snippet": snippet})
                if len(results) >= max(1, int(max_results)):
                    break
            return results

        def page_from_session():
            current_url = str(session.get("current_url", "")).strip()
            if not current_url:
                return None
            title = str(session.get("last_title", "")).strip() or current_url
            text = str(session.get("last_page_text", ""))
            snippet = summarize_text(text, 220) or title
            return {
                "url": current_url,
                "title": title,
                "text": text,
                "snippet": snippet,
                "affordances": session.get("last_page_affordances", {}) or {},
            }

        def fetch_page(url):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                body = response.read(500000).decode("utf-8", errors="ignore")
                final_url = response.geturl()
            title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
            title_raw = title_match.group(1) if title_match else final_url
            title = re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", title_raw)).strip() or final_url
            text = re.sub(r"\\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()[:8000]
            snippet = summarize_text(text, 220) or title
            return {
                "url": final_url,
                "title": title,
                "text": text,
                "snippet": snippet,
                "affordances": extract_affordances(body, final_url),
            }

        def resolve_target_url():
            direct = first_url(instruction)
            if direct:
                return direct
            current_url = str(session.get("current_url", "")).strip()
            if current_url:
                return current_url
            search_results = session.get("search_results")
            if isinstance(search_results, list) and search_results:
                top = search_results[0]
                if isinstance(top, dict):
                    return str(top.get("url", "")).strip()
            return ""

        def normalize_hint(value):
            return re.sub(r"\\s+", " ", str(value or "")).strip().strip("`'").strip('"')

        def unique_hints(values):
            ordered = []
            seen = set()
            for value in values:
                hint = normalize_hint(value)
                key = hint.lower()
                if not hint or key in seen:
                    continue
                seen.add(key)
                ordered.append(hint)
            return ordered

        def instruction_field_hints():
            hints = []
            affordances = session.get("last_page_affordances", {}) or {}
            fields = affordances.get("input_fields") if isinstance(affordances, dict) else []
            if isinstance(fields, list):
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    for key in ("name", "label", "placeholder", "type"):
                        value = normalize_hint(field.get(key, ""))
                        if value and re.search(rf"(?i)\\b{re.escape(value)}\\b", instruction):
                            hints.append(value)
                            break
            match = re.search(r"grounded fields?\\s*\\(([^)]*)\\)", instruction, re.IGNORECASE)
            if match:
                hints.extend(
                    part
                    for part in re.split(r",|\\band\\b", match.group(1))
                    if normalize_hint(part)
                )
            if hints:
                return unique_hints(hints)
            if isinstance(fields, list):
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    for key in ("name", "label", "placeholder", "type"):
                        value = normalize_hint(field.get(key, ""))
                        if value:
                            hints.append(value)
                            break
            return unique_hints(hints)

        def instruction_button_hint():
            affordances = session.get("last_page_affordances", {}) or {}
            buttons = affordances.get("buttons") if isinstance(affordances, dict) else []
            if isinstance(buttons, list):
                for button in buttons:
                    if not isinstance(button, dict):
                        continue
                    value = normalize_hint(button.get("text", ""))
                    if value and re.search(rf"(?i)\\b{re.escape(value)}\\b", instruction):
                        return value
            for match in re.finditer(r'[`"]([^`"]+)[`"]', instruction):
                value = normalize_hint(match.group(1))
                if value:
                    return value
            if isinstance(buttons, list):
                for button in buttons:
                    if not isinstance(button, dict):
                        continue
                    value = normalize_hint(button.get("text", ""))
                    if value:
                        return value
            return ""

        def submit_button_hint():
            affordances = session.get("last_page_affordances", {}) or {}
            buttons = affordances.get("buttons") if isinstance(affordances, dict) else []
            if not isinstance(buttons, list):
                return ""
            for button in buttons:
                if not isinstance(button, dict):
                    continue
                button_type = normalize_hint(button.get("type", "")).lower()
                value = normalize_hint(
                    button.get("text", "") or button.get("label", "") or button.get("name", "")
                )
                if button_type == "submit" and value:
                    return value
            for button in buttons:
                if not isinstance(button, dict):
                    continue
                value = normalize_hint(
                    button.get("text", "") or button.get("label", "") or button.get("name", "")
                )
                if value:
                    return value
            return ""

        def field_selector_from_hint(hint):
            value = normalize_hint(hint)
            if not value:
                return ""
            quoted = json.dumps(value)
            selectors = [
                f"input[name*={quoted} i]",
                f"textarea[name*={quoted} i]",
                f"select[name*={quoted} i]",
                f"[aria-label*={quoted} i]",
                f"[placeholder*={quoted} i]",
                f"[title*={quoted} i]",
            ]
            lowered = value.lower()
            if "email" in lowered:
                selectors.insert(0, "input[type=email]")
            if any(token in lowered for token in ("message", "comment", "description")):
                selectors.insert(0, "textarea")
            return ", ".join(selectors)

        def button_selector_from_hint(hint):
            value = normalize_hint(hint)
            if not value:
                return ""
            quoted = json.dumps(value)
            return ", ".join(
                [
                    f"button:has-text({quoted})",
                    f"[role='button']:has-text({quoted})",
                    f"a:has-text({quoted})",
                    f"input[type=submit][value*={quoted} i]",
                    f"input[type=button][value*={quoted} i]",
                    f"[aria-label*={quoted} i]",
                    f"[title*={quoted} i]",
                ]
            )

        def resolve_or_fetch_page(refresh=False):
            target_url = resolve_target_url()
            cached = page_from_session()
            if not target_url:
                return cached
            if not refresh and cached and cached.get("url") == target_url:
                return cached
            try:
                return fetch_page(target_url)
            except Exception:
                if cached and cached.get("url") == target_url:
                    return cached
                return cached

        def record_page(page):
            session["current_url"] = page.get("url", "")
            session["last_title"] = page.get("title", "")
            session["last_page_text"] = page.get("text", "")
            session["last_page_affordances"] = page.get("affordances", {}) or {}

        def page_citation(page):
            return {
                "url": page.get("url", ""),
                "title": page.get("title", ""),
                "snippet": page.get("snippet", ""),
            }

        def parse_instruction():
            try:
                payload = json.loads(instruction)
            except Exception:
                return {"raw": instruction}
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, list):
                return {"command": payload}
            return {"raw": instruction}

        def resolve_workspace_path(payload, default_path="", must_exist=False, allow_directory=False):
            raw_path = str(payload.get("path") or payload.get("file") or default_path).strip()
            if not raw_path:
                raise RuntimeError("Workspace action requires a `path` in the instruction payload.")
            target = workspace_dir if raw_path in {".", "./"} else workspace_dir / raw_path
            resolved = target.resolve()
            workspace_root = workspace_dir.resolve()
            if resolved != workspace_root and workspace_root not in resolved.parents:
                raise RuntimeError("Workspace path must stay within the run workspace.")
            if must_exist and not resolved.exists():
                raise RuntimeError(f"Workspace path does not exist: {raw_path}")
            if not allow_directory and resolved.exists() and resolved.is_dir():
                raise RuntimeError(f"Expected a file path, received a directory: {raw_path}")
            return resolved

        def relative_workspace_path(path):
            resolved = path.resolve()
            workspace_root = workspace_dir.resolve()
            if resolved == workspace_root:
                return "."
            return resolved.relative_to(workspace_root).as_posix()

        def list_workspace_files(target):
            resolved = target.resolve()
            if resolved.is_file():
                return [relative_workspace_path(resolved)]
            return sorted(
                path.relative_to(workspace_dir.resolve()).as_posix()
                for path in resolved.rglob("*")
                if path.is_file()
            )

        def snapshot_workspace():
            snapshot = {}
            for path in workspace_dir.rglob("*"):
                if not path.is_file():
                    continue
                stat = path.stat()
                snapshot[path.relative_to(workspace_dir.resolve()).as_posix()] = [
                    stat.st_mtime_ns,
                    stat.st_size,
                ]
            return snapshot

        def changed_workspace_files(before, after):
            return sorted(path for path, state in after.items() if before.get(path) != state)

        def artifact_kind(path):
            return "image" if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"} else "text"

        def workspace_artifact(path, kind_override=None):
            return {
                "kind": kind_override or artifact_kind(path),
                "name": path.name,
                "path": str(path.resolve()),
                "workspace": str(run_dir),
            }

        def normalize_citations(items):
            normalized = []
            if not isinstance(items, list):
                return normalized
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized.append(
                    {
                        "url": str(item.get("url", "")),
                        "title": str(item.get("title", "") or item.get("url", "")),
                        "snippet": str(item.get("snippet", "")),
                    }
                )
            return normalized

        def artifact_sources(payload):
            sources = normalize_citations(payload.get("sources"))
            if sources:
                return sources
            current_url = str(session.get("current_url", "")).strip()
            if not current_url:
                return []
            return normalize_citations(
                [
                    {
                        "url": current_url,
                        "title": str(session.get("last_title", "")).strip() or current_url,
                        "snippet": summarize_text(session.get("last_page_text", ""), 220),
                    }
                ]
            )

        def render_report(title, objective, sections, sources):
            lines = [f"# {title}", ""]
            objective_text = str(objective or "").strip()
            if objective_text and not objective_text.startswith("{"):
                lines.extend(["## Objective", objective_text, ""])
            rendered_sections = False
            if isinstance(sections, list):
                for section in sections:
                    if not isinstance(section, dict):
                        continue
                    heading = str(section.get("heading") or section.get("title") or "").strip()
                    body = str(section.get("body") or section.get("content") or "").strip()
                    if not heading and not body:
                        continue
                    rendered_sections = True
                    if heading:
                        lines.append(f"## {heading}")
                    if body:
                        lines.append(body)
                    lines.append("")
            if not rendered_sections:
                summary = summarize_text(session.get("last_page_text", ""), 320)
                if summary:
                    lines.extend(["## Summary", summary, ""])
            if sources:
                lines.append("## Sources")
                for source in sources:
                    lines.append(f"- [{source['title']}]({source['url']})")
                    snippet = str(source.get("snippet", "")).strip()
                    if snippet:
                        lines.append(f"  - {snippet}")
                lines.append("")
            return "\\n".join(lines).strip() + "\\n"

        def chart_rows(payload):
            raw_data = payload.get("data", payload.get("chart_data"))
            if not isinstance(raw_data, list):
                return []
            return [item for item in raw_data if isinstance(item, dict)]

        def chart_axis_keys(payload, data):
            x_key = str(payload.get("x_key", "")).strip()
            y_key = str(payload.get("y_key", "")).strip()
            if x_key and y_key:
                return x_key, y_key
            sample = data[0] if data else {}
            for key, value in sample.items():
                if not x_key and not isinstance(value, (int, float)):
                    x_key = str(key)
                if not y_key and isinstance(value, (int, float)):
                    y_key = str(key)
            return x_key or "label", y_key or "value"

        def render_chart(title, chart_type, data, x_key, y_key, sources):
            numeric_values = [
                float(row.get(y_key, 0))
                for row in data
                if isinstance(row.get(y_key), (int, float))
            ]
            max_value = max(numeric_values) if numeric_values else 1.0
            bar_rows = []
            table_rows = []
            for row in data:
                label = html.escape(str(row.get(x_key, "")))
                raw_value = row.get(y_key, 0)
                value = float(raw_value) if isinstance(raw_value, (int, float)) else 0.0
                width = 0.0 if max_value <= 0 else (value / max_value) * 100.0
                bar_rows.append(
                    "<div class=\\"bar-row\\">"
                    f"<div class=\\"bar-label\\">{label}</div>"
                    "<div class=\\"bar-track\\"><div class=\\"bar-fill\\" style=\\"width:"
                    f"{width:.2f}%\\"></div></div>"
                    f"<div class=\\"bar-value\\">{html.escape(str(raw_value))}</div></div>"
                )
                table_rows.append(
                    f"<tr><td>{label}</td><td>{html.escape(str(raw_value))}</td></tr>"
                )
            sources_markup = "".join(
                "<li>"
                f"<a href=\\"{html.escape(str(source.get('url', '')))}\\">"
                f"{html.escape(str(source.get('title', '') or source.get('url', '')))}</a>"
                f"<span>{html.escape(str(source.get('snippet', '')))}</span>"
                "</li>"
                for source in sources
            )
            return (
                "<!doctype html><html><head><meta charset=\\"utf-8\\" />"
                f"<title>{html.escape(title)}</title>"
                "<style>"
                "body{font-family:IBM Plex Sans,Arial,sans-serif;padding:32px;background:#f5f1e8;color:#17313b;}"
                ".chart{display:grid;gap:12px;margin:24px 0;}"
                ".bar-row{display:grid;grid-template-columns:160px 1fr 80px;gap:12px;align-items:center;}"
                ".bar-track{height:18px;background:#d7e5e2;border-radius:999px;overflow:hidden;}"
                ".bar-fill{height:100%;background:#2a8c82;}"
                "table{border-collapse:collapse;width:100%;margin-top:24px;}"
                "th,td{padding:8px 10px;border-bottom:1px solid #c8d4d1;text-align:left;}"
                "ul{padding-left:20px;} li span{display:block;color:#47606a;margin-top:4px;}"
                "</style></head><body>"
                f"<h1>{html.escape(title)}</h1>"
                f"<p>Type: {html.escape(chart_type)} | X: {html.escape(x_key)} | Y: {html.escape(y_key)}</p>"
                f"<section class=\\"chart\\">{''.join(bar_rows)}</section>"
                "<section><h2>Data</h2><table><thead>"
                f"<tr><th>{html.escape(x_key)}</th><th>{html.escape(y_key)}</th></tr>"
                f"</thead><tbody>{''.join(table_rows)}</tbody></table></section>"
                f"<section><h2>Sources</h2><ul>{sources_markup}</ul></section>"
                "</body></html>"
            )

        def wrap_svg_text(text, width=44, limit=4):
            words = str(text or "").split()
            if not words:
                return ["No prompt provided."]
            lines = []
            current = ""
            used_words = 0
            for word in words:
                candidate = f"{current} {word}".strip()
                if current and len(candidate) > width:
                    lines.append(current)
                    current = word
                    if len(lines) >= limit:
                        break
                    continue
                current = candidate
                used_words += 1
            if len(lines) < limit and current:
                lines.append(current)
            if len(lines) == limit and used_words < len(words):
                lines[-1] = lines[-1][: max(0, width - 1)].rstrip() + "…"
            return lines

        def render_image(title, prompt, sources):
            prompt_lines = wrap_svg_text(prompt, width=44, limit=4)
            source_lines = [
                str(source.get("title") or source.get("url") or "").strip()
                for source in sources[:2]
                if str(source.get("title") or source.get("url") or "").strip()
            ]
            prompt_markup = "".join(
                f'<text x="72" y="{240 + (index * 34)}" class="prompt">{html.escape(line)}</text>'
                for index, line in enumerate(prompt_lines)
            )
            source_markup = "".join(
                f'<text x="72" y="{470 + (index * 24)}" class="source">{html.escape(line)}</text>'
                for index, line in enumerate(source_lines)
            )
            source_heading = (
                '<text x="72" y="438" class="eyebrow">Grounded by sources</text>' if source_lines else ""
            )
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" '
                f'aria-label="{html.escape(title)}">'
                "<defs>"
                '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">'
                '<stop offset="0%" stop-color="#f6efe2" />'
                '<stop offset="100%" stop-color="#d2ebe6" />'
                "</linearGradient>"
                '<linearGradient id="card" x1="0%" y1="0%" x2="100%" y2="100%">'
                '<stop offset="0%" stop-color="#17313b" stop-opacity="0.96" />'
                '<stop offset="100%" stop-color="#25505b" stop-opacity="0.92" />'
                "</linearGradient>"
                "</defs>"
                '<rect width="1200" height="630" fill="url(#bg)" />'
                '<circle cx="1020" cy="120" r="180" fill="#90c9bc" fill-opacity="0.35" />'
                '<circle cx="160" cy="520" r="220" fill="#f0c87c" fill-opacity="0.25" />'
                '<rect x="48" y="48" width="1104" height="534" rx="36" fill="url(#card)" />'
                '<style>'
                ".eyebrow{font:600 18px IBM Plex Sans,Arial,sans-serif;letter-spacing:0.12em;text-transform:uppercase;fill:#90c9bc;}"
                ".title{font:700 54px IBM Plex Sans,Arial,sans-serif;fill:#f6efe2;}"
                ".prompt{font:400 28px IBM Plex Sans,Arial,sans-serif;fill:#d9ece7;}"
                ".source{font:400 18px IBM Plex Sans,Arial,sans-serif;fill:#bdd7d0;}"
                "</style>"
                '<text x="72" y="110" class="eyebrow">Generated image artifact</text>'
                f'<text x="72" y="182" class="title">{html.escape(title)}</text>'
                f"{prompt_markup}"
                f"{source_heading}"
                f"{source_markup}"
                '<path d="M860 176c74 0 134 60 134 134s-60 134-134 134-134-60-134-134 60-134 134-134Zm0 52c-45 0-82 37-82 82s37 82 82 82 82-37 82-82-37-82-82-82Z" fill="#90c9bc" fill-opacity="0.18"/>'
                '<path d="M820 330c24-60 72-106 142-136 43 33 72 78 88 136-54-22-101-20-142 8-29 19-58 16-88-8Z" fill="#f0c87c" fill-opacity="0.82"/>'
                "</svg>\\n"
            )

        def stringify_command(command):
            if isinstance(command, list):
                return " ".join(str(part) for part in command)
            return str(command)

        session = load_session()
        payload = parse_instruction()
        citations = []
        artifacts = []
        metadata = {"session_path": str(session_path)}
        output = f"[sandbox] Executed action `{action}`: {instruction}"

        if action == "search_web":
            citations = search_web(instruction, max_results=5)
            session["search_results"] = citations
            metadata["search_results"] = citations
            metadata["top_url"] = citations[0]["url"] if citations else ""
            output = f"[sandbox-search] Found {len(citations)} result(s) for: {instruction}"
        elif action == "list_files":
            target = resolve_workspace_path(payload, default_path=".", allow_directory=True)
            target.mkdir(parents=True, exist_ok=True)
            files = list_workspace_files(target)
            metadata["path"] = relative_workspace_path(target)
            metadata["files"] = files
            output = "\\n".join(files) if files else "[sandbox-workspace] No files found."
        elif action == "read_file":
            target = resolve_workspace_path(payload, must_exist=True, allow_directory=False)
            content = target.read_text(encoding="utf-8")
            metadata["file_path"] = relative_workspace_path(target)
            metadata["bytes_read"] = len(content.encode("utf-8"))
            output = content
        elif action == "write_file":
            target = resolve_workspace_path(payload, must_exist=False, allow_directory=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = str(payload.get("content", ""))
            target.write_text(content, encoding="utf-8")
            metadata["file_path"] = relative_workspace_path(target)
            metadata["bytes_written"] = target.stat().st_size
            metadata["changed"] = True
            artifacts.append(workspace_artifact(target))
            output = f"[sandbox-workspace] Wrote {metadata['file_path']}"
        elif action == "edit_file":
            target = resolve_workspace_path(payload, must_exist=True, allow_directory=False)
            original = target.read_text(encoding="utf-8")
            old = str(payload.get("old", ""))
            new = str(payload.get("new", ""))
            if old:
                updated = original.replace(old, new)
            else:
                updated = str(payload.get("content", original))
            changed = updated != original
            if changed:
                target.write_text(updated, encoding="utf-8")
                artifacts.append(workspace_artifact(target))
            metadata["file_path"] = relative_workspace_path(target)
            metadata["bytes_written"] = len(updated.encode("utf-8"))
            metadata["changed"] = changed
            output = (
                f"[sandbox-workspace] Updated {metadata['file_path']}"
                if changed
                else f"[sandbox-workspace] No changes applied to {metadata['file_path']}"
            )
        elif action == "execute_code":
            command = payload.get("command", instruction)
            if isinstance(command, list):
                command = [str(part) for part in command]
            before = snapshot_workspace()
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                cwd=workspace_dir,
                shell=isinstance(command, str),
                timeout=30,
            )
            touched_files = changed_workspace_files(before, snapshot_workspace())
            for rel_path in touched_files:
                artifacts.append(workspace_artifact(workspace_dir / rel_path))
            metadata["command"] = stringify_command(command)
            metadata["exit_code"] = proc.returncode
            metadata["touched_files"] = touched_files
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "command returned non-zero exit status").strip()
                raise RuntimeError(f"Sandbox code execution failed (exit={proc.returncode}): {detail[:500]}")
            output = (proc.stdout or "").strip() or (proc.stderr or "").strip() or "[sandbox-code] Command completed."
        elif action == "generate_report":
            target = resolve_workspace_path(payload, default_path=f"reports/{step_id}.md", must_exist=False, allow_directory=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            title = str(payload.get("title", "")).strip() or "Grounded Report"
            sources = artifact_sources(payload)
            citations = normalize_citations(sources)
            report_body = render_report(
                title,
                payload.get("objective", instruction),
                payload.get("sections", []),
                sources,
            )
            target.write_text(report_body, encoding="utf-8")
            metadata["file_path"] = relative_workspace_path(target)
            metadata["bytes_written"] = len(report_body.encode("utf-8"))
            metadata["artifact_kind"] = "report"
            metadata["report_title"] = title
            metadata["source_citation_count"] = len(citations)
            artifacts.append(workspace_artifact(target, kind_override="report"))
            output = f"[sandbox-artifact] Generated grounded report {metadata['file_path']} from {len(citations)} source(s)."
        elif action == "generate_chart":
            target = resolve_workspace_path(payload, default_path=f"charts/{step_id}.html", must_exist=False, allow_directory=False)
            data = chart_rows(payload)
            if not data:
                raise RuntimeError("No structured chart data provided for `generate_chart`.")
            target.parent.mkdir(parents=True, exist_ok=True)
            title = str(payload.get("title", "")).strip() or "Generated Chart"
            chart_type = str(payload.get("chart_type", "bar")).strip() or "bar"
            x_key, y_key = chart_axis_keys(payload, data)
            sources = artifact_sources(payload)
            citations = normalize_citations(sources)
            chart_body = render_chart(title, chart_type, data, x_key, y_key, sources)
            target.write_text(chart_body, encoding="utf-8")
            metadata["file_path"] = relative_workspace_path(target)
            metadata["bytes_written"] = len(chart_body.encode("utf-8"))
            metadata["artifact_kind"] = "chart"
            metadata["chart_type"] = chart_type
            metadata["chart_title"] = title
            metadata["data_points"] = len(data)
            metadata["x_key"] = x_key
            metadata["y_key"] = y_key
            metadata["source_citation_count"] = len(citations)
            artifacts.append(workspace_artifact(target, kind_override="chart"))
            output = f"[sandbox-artifact] Generated grounded chart {metadata['file_path']} with {len(data)} data point(s)."
        elif action == "generate_image":
            target = resolve_workspace_path(payload, default_path=f"images/{step_id}.svg", must_exist=False, allow_directory=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            title = str(payload.get("title", "")).strip() or "Generated Image"
            prompt = str(payload.get("prompt") or payload.get("objective") or instruction).strip()
            sources = artifact_sources(payload)
            citations = normalize_citations(sources)
            image_body = render_image(title, prompt, sources)
            target.write_text(image_body, encoding="utf-8")
            metadata["file_path"] = relative_workspace_path(target)
            metadata["bytes_written"] = len(image_body.encode("utf-8"))
            metadata["artifact_kind"] = "image"
            metadata["image_title"] = title
            metadata["image_prompt"] = prompt
            metadata["image_provider"] = "builtin-svg"
            metadata["source_citation_count"] = len(citations)
            artifacts.append(workspace_artifact(target, kind_override="image"))
            output = f"[sandbox-artifact] Generated grounded image {metadata['file_path']} from prompt and {len(citations)} source(s)."
        elif action in {"fetch_url", "navigate", "inspect", "scroll", "read", "extract"}:
            page_data = None
            target_url = resolve_target_url()
            if target_url:
                try:
                    if browser_mode != "simulated":
                        from playwright.sync_api import sync_playwright

                        with sync_playwright() as playwright:
                            browser = playwright.chromium.launch(headless=True)
                            state_path = storage_state_path()
                            context_kwargs = {}
                            if state_path.exists():
                                context_kwargs["storage_state"] = str(state_path)
                            context = browser.new_context(**context_kwargs)
                            page = context.new_page()
                            page.goto(target_url, wait_until="domcontentloaded", timeout=browser_timeout_ms)
                            if action == "scroll":
                                page.mouse.wheel(0, 1600)
                            resolved_url = page.url or target_url
                            title = page.title() or resolved_url
                            try:
                                page_text = page.inner_text("body")
                            except Exception:
                                page_text = ""
                            try:
                                page_html = page.inner_html("body")
                            except Exception:
                                page_html = ""
                            page_data = {
                                "url": resolved_url,
                                "title": title,
                                "text": page_text,
                                "snippet": summarize_text(page_text, 220) or title,
                                "affordances": extract_affordances(page_html, resolved_url),
                            }
                            output = f"[sandbox-browser-real] {action} {resolved_url}"
                            if capture_screenshot:
                                screenshot_name = f"{step_id}-screenshot.png"
                                page.screenshot(path=screenshot_name, full_page=True)
                                artifacts.append({"kind": "image", "name": screenshot_name, "path": screenshot_name})
                            context.storage_state(path=str(state_path))
                            session["browser_storage_state_path"] = str(state_path)
                            metadata["browser_storage_state_path"] = str(state_path)
                            context.close()
                            browser.close()
                    else:
                        page_data = resolve_or_fetch_page(
                            refresh=action in {"fetch_url", "navigate", "scroll"}
                        )
                        if page_data:
                            output = f"[sandbox-browser] {action} {page_data['url']}"
                except Exception:
                    if browser_mode == "real":
                        raise
                    page_data = resolve_or_fetch_page(refresh=action in {"fetch_url", "navigate", "scroll"})
                    if page_data:
                        output = f"[sandbox-browser] {action} {page_data['url']}"
            else:
                page_data = page_from_session()
            if page_data:
                record_page(page_data)
                citations = [page_citation(page_data)]
                metadata["current_url"] = page_data["url"]
                metadata["page_title"] = page_data["title"]
                metadata["page_excerpt"] = page_data["snippet"]
                if isinstance(page_data.get("affordances"), dict):
                    metadata["page_affordances"] = page_data["affordances"]
                if action == "extract":
                    output = (
                        f"[sandbox-browser] Evidence summary for {page_data['url']}: "
                        f"{summarize_text(page_data.get('text', ''), 260) or page_data['snippet']}"
                    )
            elif not target_url:
                output = f"[sandbox-browser] Unable to resolve grounded URL for: {instruction}"
            if action == "extract":
                artifact_name = f"{step_id}-extract.txt"
                Path(artifact_name).write_text(output, encoding="utf-8")
                artifacts.append({"kind": "text", "name": artifact_name, "path": artifact_name})
        elif action in {"type", "click", "wait", "submit"}:
            page_data = resolve_or_fetch_page(refresh=False)
            if not page_data:
                raise RuntimeError(
                    f"No grounded page available for `{action}` action. Navigate/search/fetch first."
                )
            if browser_mode != "simulated":
                from playwright.sync_api import sync_playwright

                target_url = page_data.get("url") or resolve_target_url()
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    state_path = storage_state_path()
                    context_kwargs = {}
                    if state_path.exists():
                        context_kwargs["storage_state"] = str(state_path)
                    context = browser.new_context(**context_kwargs)
                    page = context.new_page()
                    page.goto(target_url, wait_until="domcontentloaded", timeout=browser_timeout_ms)
                    if action == "type":
                        field_hints = instruction_field_hints()
                        selector = field_selector_from_hint(field_hints[0] if field_hints else "")
                        if not selector:
                            selector = (
                                "textarea, input:not([type=hidden]):not([type=submit]):not([type=button]):"
                                "not([type=checkbox]):not([type=radio]), [contenteditable='true']"
                            )
                        metadata["target_selector"] = selector
                        if field_hints:
                            metadata["target_hint"] = field_hints[0]
                        page.locator(selector).first.fill(instruction, timeout=browser_timeout_ms)
                    elif action == "click":
                        button_hint = instruction_button_hint()
                        selector = button_selector_from_hint(button_hint)
                        if not selector:
                            selector = "button, [role='button'], input[type=submit], input[type=button], a"
                        metadata["target_selector"] = selector
                        if button_hint:
                            metadata["target_hint"] = button_hint
                        page.locator(selector).first.click(timeout=browser_timeout_ms)
                    elif action == "wait":
                        page.wait_for_load_state("networkidle", timeout=browser_timeout_ms)
                    else:
                        button_hint = submit_button_hint()
                        selector = button_selector_from_hint(button_hint)
                        if selector:
                            metadata["target_selector"] = selector
                            metadata["target_hint"] = button_hint
                            page.locator(selector).first.click(timeout=browser_timeout_ms)
                        else:
                            metadata["target_selector"] = "form"
                            page.locator("form").first.evaluate("(form) => form.requestSubmit()")
                    resolved_url = page.url or target_url
                    title = page.title() or resolved_url
                    try:
                        page_text = page.inner_text("body")
                    except Exception:
                        page_text = ""
                    try:
                        page_html = page.inner_html("body")
                    except Exception:
                        page_html = ""
                    page_data = {
                        "url": resolved_url,
                        "title": title,
                        "text": page_text,
                        "snippet": summarize_text(page_text, 220) or title,
                        "affordances": extract_affordances(page_html, resolved_url),
                    }
                    context.storage_state(path=str(state_path))
                    session["browser_storage_state_path"] = str(state_path)
                    metadata["browser_storage_state_path"] = str(state_path)
                    context.close()
                    browser.close()
            record_page(page_data)
            citations = [page_citation(page_data)]
            metadata["current_url"] = page_data["url"]
            metadata["page_title"] = page_data["title"]
            metadata["page_excerpt"] = page_data["snippet"]
            if isinstance(page_data.get("affordances"), dict):
                metadata["page_affordances"] = page_data["affordances"]
            target = str(session.get("current_url") or "current session")
            if action == "type":
                draft_inputs = session.setdefault("draft_inputs", [])
                if not isinstance(draft_inputs, list):
                    draft_inputs = []
                    session["draft_inputs"] = draft_inputs
                draft_inputs.append({"step_id": step_id, "instruction": instruction})
                metadata["draft_input_count"] = len(draft_inputs)
                prefix = "[sandbox-browser-real]" if browser_mode != "simulated" else "[sandbox-browser]"
                output = f"{prefix} Typed draft input on {target}: {instruction}"
            elif action == "click":
                prefix = "[sandbox-browser-real]" if browser_mode != "simulated" else "[sandbox-browser]"
                output = f"{prefix} Clicked the requested control on {target}: {instruction}"
            elif action == "wait":
                prefix = "[sandbox-browser-real]" if browser_mode != "simulated" else "[sandbox-browser]"
                output = f"{prefix} Waited for the page state to settle on {target}"
            else:
                session["submitted"] = True
                prefix = "[sandbox-browser-real]" if browser_mode != "simulated" else "[sandbox-browser]"
                output = f"{prefix} Submitted the workflow on {target}: {instruction}"
        elif action in {"write", "export"}:
            artifact_name = f"{step_id}-{action}.txt"
            output = f"[sandbox-workspace] {action} prepared for: {session.get('current_url') or instruction}"
            Path(artifact_name).write_text(output, encoding="utf-8")
            artifacts.append({"kind": "text", "name": artifact_name, "path": artifact_name})

        save_session(session)
        result_path.write_text(
            json.dumps(
                {
                    "output_text": output,
                    "citations": citations,
                    "artifacts": artifacts,
                    "metadata": metadata,
                }
            ),
            encoding="utf-8",
        )
        """
    ).strip()


def _normalize_citations(citations: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        normalized.append(
            {
                "url": str(citation.get("url", "")),
                "title": str(citation.get("title", "")),
                "snippet": str(citation.get("snippet", "")),
            }
        )
    return normalized


def _write_step_metadata(
    run_dir: Path,
    request: StepRequest,
    timestamp: str,
    output_text: str,
    citations: list[dict[str, str]],
    artifacts: list[dict[str, str]],
    metadata: dict[str, Any],
) -> None:
    (run_dir / f"{request.step_id}.json").write_text(
        json.dumps(
            {
                "step_id": request.step_id,
                "run_id": request.run_id,
                "action_type": request.action_type.strip().lower(),
                "instruction": request.instruction,
                "timestamp": timestamp,
                "output_text": output_text,
                "citations": citations,
                "artifacts": artifacts,
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
