"""Execution backends for sandbox-runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Protocol
from urllib.parse import quote_plus

_ARTIFACT_ACTIONS: frozenset[str] = frozenset({"extract", "write", "export"})
_CITATION_ACTIONS: frozenset[str] = frozenset({"navigate", "extract"})
DEFAULT_DOCKER_IMAGE = (
    "python:3.13-slim@sha256:8bc60ca09afaa8ea0d6d1220bde073bacfedd66a4bf8129cbdc8ef0e16c8a952"
)
VALID_BROWSER_MODES: frozenset[str] = frozenset({"simulated", "auto", "real"})


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
    metadata: dict[str, str]


class StepExecutor(Protocol):
    """Execution backend contract."""

    def execute(self, request: StepRequest, sandbox_root: Path) -> StepResult:
        """Execute one step and return output artifacts/citations."""


class LocalEphemeralExecutor:
    """In-process executor with ephemeral per-step workspace."""

    backend_name = "local"

    def preflight(self) -> dict[str, str]:
        return {"status": "ok", "backend": self.backend_name}

    def execute(self, request: StepRequest, sandbox_root: Path) -> StepResult:
        run_dir = sandbox_root / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        action = request.action_type.strip().lower()
        ts = datetime.now(timezone.utc).isoformat()

        output = _step_output(action, request.instruction)
        citations = _citations(action, request.instruction)
        artifacts: list[dict[str, str]] = []

        with tempfile.TemporaryDirectory(
            prefix=f"step-{request.run_id}-{request.step_id}-",
            dir=sandbox_root,
        ) as temp_dir:
            step_workspace = Path(temp_dir)
            if action in _ARTIFACT_ACTIONS:
                name = f"{request.step_id}-{action}.txt"
                temp_artifact = step_workspace / name
                temp_artifact.write_text(output, encoding="utf-8")
                final_artifact = run_dir / name
                shutil.copy2(temp_artifact, final_artifact)
                artifacts.append(
                    {
                        "kind": "text",
                        "name": name,
                        "rel_path": f"{request.run_id}/{name}",
                        "sandbox_path": str(final_artifact),
                        "sha256": _sha256(final_artifact),
                    }
                )

        _write_step_metadata(run_dir, request, ts)
        return StepResult(
            output_text=output,
            citations=citations,
            artifacts=artifacts,
            metadata={
                "timestamp": ts,
                "sandbox_root": str(sandbox_root),
                "executor_backend": self.backend_name,
            },
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
        if not self.image:
            raise ValueError("Docker executor requires SANDBOX_DOCKER_IMAGE")
        if "@sha256:" not in self.image:
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
            err = (proc.stderr or proc.stdout or "").strip()
            if not err:
                err = "docker info returned non-zero status."
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
        ts = datetime.now(timezone.utc).isoformat()
        if shutil.which(self.docker_bin) is None:
            raise RuntimeError(
                f"Docker binary `{self.docker_bin}` is not available for sandbox backend."
            )

        with tempfile.TemporaryDirectory(
            prefix=f"step-{request.run_id}-{request.step_id}-",
            dir=sandbox_root,
        ) as temp_dir:
            step_workspace = Path(temp_dir)
            result_file = step_workspace / "result.json"
            command = self.build_command(request, step_workspace, result_file)
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
                step_workspace=step_workspace,
                artifact_specs=payload.get("artifacts", []),
            )

        _write_step_metadata(run_dir, request, ts)
        metadata = {
            "timestamp": ts,
            "sandbox_root": str(sandbox_root),
            "executor_backend": self.backend_name,
            "container_image": self.image,
            "browser_mode": self.browser_mode,
        }
        return StepResult(
            output_text=output,
            citations=citations,
            artifacts=artifacts,
            metadata=metadata,
        )

    def build_command(
        self,
        request: StepRequest,
        step_workspace: Path,
        result_file: Path,
    ) -> list[str]:
        script = _container_script()
        cmd = [
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
            "-v",
            f"{step_workspace}:/work",
            "-w",
            "/work",
            self.image,
            "python",
            "-c",
            script,
        ]
        return cmd

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
        for spec in artifact_specs:
            source_name_raw = spec.get("name", "").strip()
            if not source_name_raw:
                continue
            source_name = Path(source_name_raw).name
            source = Path(spec.get("path", source_name))
            if not source.is_absolute():
                source = workspace_root / source
            source = source.resolve()
            if workspace_root not in source.parents and source != workspace_root:
                continue
            if not source.exists():
                continue

            final_path = run_dir / source_name
            shutil.copy2(source, final_path)
            artifacts.append(
                {
                    "kind": spec.get("kind", "text"),
                    "name": source_name,
                    "rel_path": f"{run_id}/{source_name}",
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
        stderr = (proc.stderr or proc.stdout or "").strip()
        if not stderr:
            stderr = "import failed"
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
    if not items:
        return [DEFAULT_DOCKER_IMAGE]
    return items


def _container_script() -> str:
    return dedent(
        """
        import contextlib
        import json
        import os
        import re
        import urllib.parse

        action = os.environ.get("NEXUS_ACTION", "").strip().lower()
        instruction = os.environ.get("NEXUS_INSTRUCTION", "")
        step_id = os.environ.get("NEXUS_STEP_ID", "step")
        result_path = os.environ.get("NEXUS_OUTPUT_JSON", "result.json")
        browser_mode = os.environ.get("NEXUS_BROWSER_MODE", "auto").strip().lower()
        browser_timeout_ms = int(os.environ.get("NEXUS_BROWSER_TIMEOUT_MS", "15000"))
        capture_screenshot = os.environ.get("NEXUS_CAPTURE_SCREENSHOT", "1") != "0"

        if browser_mode not in {"simulated", "auto", "real"}:
            raise RuntimeError(f"Unsupported browser mode: {browser_mode}")

        def first_url(text: str) -> str:
            match = re.search(r"https?://\\S+", text)
            return match.group(0) if match else ""

        if action == "navigate":
            output = f"[sandbox-browser] Navigated and collected candidate pages for: {instruction}"
        elif action == "extract":
            output = (
                "[sandbox-browser] Evidence summary with citations prepared. "
                f"Focus: {instruction}"
            )
        elif action == "export":
            output = (
                "[sandbox-browser] Export artifact prepared for workspace promotion. "
                f"Request: {instruction}"
            )
        else:
            output = f"[sandbox-browser] Executed action `{action}`: {instruction}"

        citations = []
        if action in {"navigate", "extract"}:
            query = urllib.parse.quote_plus(instruction[:120])
            citations.append(
                {
                    "url": f"https://example.com/search?q={query}",
                    "title": "Sandbox Search Result",
                    "snippet": f"Evidence candidate for: {instruction[:120]}",
                }
            )

        artifacts = []
        target_url = first_url(instruction)
        wants_browser = action in {"navigate", "extract"} and target_url
        if wants_browser and browser_mode != "simulated":
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(target_url, wait_until="domcontentloaded", timeout=browser_timeout_ms)
                    title = page.title() or "Untitled"
                    body = ""
                    with contextlib.suppress(Exception):
                        body = page.inner_text("body")[:1200]
                    output = f"[sandbox-browser-real] Visited {target_url} ({title})"
                    citations = [
                        {
                            "url": target_url,
                            "title": title,
                            "snippet": body[:220] or f"Captured from {target_url}",
                        }
                    ]
                    if capture_screenshot:
                        screenshot_name = f"{step_id}-screenshot.png"
                        page.screenshot(path=screenshot_name, full_page=True)
                        artifacts.append(
                            {
                                "kind": "image",
                                "name": screenshot_name,
                                "path": screenshot_name,
                                "workspace": os.getcwd(),
                            }
                        )
                    browser.close()
            except Exception as exc:
                if browser_mode == "real":
                    raise
                output = f"{output} (browser fallback: {type(exc).__name__})"

        if action in {"extract", "write", "export"}:
            name = f"{step_id}-{action}.txt"
            with open(name, "w", encoding="utf-8") as fh:
                fh.write(output)
            artifacts.append(
                {
                    "kind": "text",
                    "name": name,
                    "path": name,
                    "workspace": os.getcwd(),
                }
            )

        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "output_text": output,
                    "citations": citations,
                    "artifacts": artifacts,
                },
                fh,
            )
        """
    ).strip()


def _step_output(action: str, instruction: str) -> str:
    if action == "navigate":
        return f"[sandbox-browser] Navigated and collected candidate pages for: {instruction}"
    if action == "extract":
        return (
            "[sandbox-browser] Evidence summary with citations prepared. "
            f"Focus: {instruction}"
        )
    if action == "export":
        return (
            "[sandbox-browser] Export artifact prepared for workspace promotion. "
            f"Request: {instruction}"
        )
    return f"[sandbox-browser] Executed action `{action}`: {instruction}"


def _citations(action: str, instruction: str) -> list[dict[str, str]]:
    if action not in _CITATION_ACTIONS:
        return []
    query = quote_plus(instruction[:120])
    return [
        {
            "url": f"https://example.com/search?q={query}",
            "title": "Sandbox Search Result",
            "snippet": f"Evidence candidate for: {instruction[:120]}",
        }
    ]


def _normalize_citations(citations: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for citation in citations:
        normalized.append(
            {
                "url": str(citation.get("url", "")),
                "title": str(citation.get("title", "")),
                "snippet": str(citation.get("snippet", "")),
            }
        )
    return normalized


def _write_step_metadata(run_dir: Path, request: StepRequest, timestamp: str) -> None:
    meta_path = run_dir / f"{request.step_id}.json"
    meta_path.write_text(
        json.dumps(
            {
                "step_id": request.step_id,
                "run_id": request.run_id,
                "action_type": request.action_type.strip().lower(),
                "instruction": request.instruction,
                "timestamp": timestamp,
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
