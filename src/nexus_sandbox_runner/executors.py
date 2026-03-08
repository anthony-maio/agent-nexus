"""Execution backends for sandbox-runner."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Any, Protocol

_ARTIFACT_ACTIONS: frozenset[str] = frozenset({"extract", "write", "export"})
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
            command = self.build_command(request, run_dir, step_workspace, result_file)
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
            "-c",
            _container_script(),
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
        for spec in artifact_specs:
            source_name_raw = str(spec.get("name", "")).strip()
            if not source_name_raw:
                continue
            source_name = Path(source_name_raw).name
            source = Path(str(spec.get("path", source_name)))
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
                    "kind": str(spec.get("kind", "text")),
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
    del workspace_dir
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

    if action == "type":
        draft_inputs = session.setdefault("draft_inputs", [])
        draft_inputs.append({"step_id": request.step_id, "instruction": instruction})
        metadata["draft_input_count"] = len(draft_inputs)
        output = (
            "[sandbox-browser] Typed draft input on "
            f"{session.get('current_url') or 'current session'}: {instruction}"
        )
    elif action == "click":
        output = (
            "[sandbox-browser] Clicked the requested control on "
            f"{session.get('current_url') or 'current session'}: {instruction}"
        )
    elif action == "wait":
        output = (
            "[sandbox-browser] Waited for the page state to settle on "
            f"{session.get('current_url') or 'current session'}"
        )
    elif action == "submit":
        session["submitted"] = True
        output = (
            "[sandbox-browser] Submitted the workflow on "
            f"{session.get('current_url') or 'current session'}: {instruction}"
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


def _record_page(session: dict[str, Any], page: dict[str, str]) -> None:
    session["current_url"] = page.get("url", "")
    session["last_title"] = page.get("title", "")
    session["last_page_text"] = page.get("text", "")
    _append_history(session, "page", page.get("url", ""))


def _page_metadata(page: dict[str, str]) -> dict[str, Any]:
    return {
        "current_url": page.get("url", ""),
        "page_title": page.get("title", ""),
        "page_excerpt": page.get("snippet", ""),
    }


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


def _container_script() -> str:
    return dedent(
        """
        import json
        import os
        import re
        import urllib.request
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

        def load_session():
            if not session_path.exists():
                return {"current_url": "", "search_results": []}
            try:
                return json.loads(session_path.read_text(encoding="utf-8"))
            except Exception:
                return {"current_url": "", "search_results": []}

        def save_session(session):
            run_dir.mkdir(parents=True, exist_ok=True)
            session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")

        def first_url(text):
            match = re.search(r"https?://\\S+", text or "")
            return match.group(0) if match else ""

        session = load_session()
        citations = []
        artifacts = []
        metadata = {"session_path": str(session_path)}
        output = f"[sandbox] Executed action `{action}`: {instruction}"

        if action == "search_web":
            top_url = first_url(instruction) or "https://duckduckgo.com/"
            citations = [{"url": top_url, "title": "Search result", "snippet": instruction[:220]}]
            session["search_results"] = citations
            metadata["top_url"] = top_url
            output = f"[sandbox-search] Collected grounded search results for: {instruction}"
        elif action in {"fetch_url", "navigate", "inspect", "scroll", "read", "extract"}:
            target_url = first_url(instruction) or session.get("current_url", "")
            if not target_url and session.get("search_results"):
                target_url = session["search_results"][0]["url"]
            if target_url:
                try:
                    if browser_mode != "simulated":
                        from playwright.sync_api import sync_playwright

                        with sync_playwright() as playwright:
                            browser = playwright.chromium.launch(headless=True)
                            page = browser.new_page()
                            page.goto(target_url, wait_until="domcontentloaded", timeout=browser_timeout_ms)
                            if action == "scroll":
                                page.mouse.wheel(0, 1600)
                            session["current_url"] = page.url or target_url
                            title = page.title() or session["current_url"]
                            try:
                                snippet = page.inner_text("body")[:220]
                            except Exception:
                                snippet = instruction[:220]
                            citations = [{"url": session["current_url"], "title": title, "snippet": snippet}]
                            metadata["current_url"] = session["current_url"]
                            output = f"[sandbox-browser-real] {action} {session['current_url']}"
                            if capture_screenshot:
                                screenshot_name = f"{step_id}-screenshot.png"
                                page.screenshot(path=screenshot_name, full_page=True)
                                artifacts.append({"kind": "image", "name": screenshot_name, "path": screenshot_name})
                            browser.close()
                    else:
                        req = urllib.request.Request(target_url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=10) as response:
                            session["current_url"] = response.geturl()
                        citations = [{"url": session["current_url"], "title": session["current_url"], "snippet": instruction[:220]}]
                        metadata["current_url"] = session["current_url"]
                        output = f"[sandbox-browser] {action} {session['current_url']}"
                except Exception:
                    if browser_mode == "real":
                        raise
                    try:
                        req = urllib.request.Request(target_url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=10) as response:
                            session["current_url"] = response.geturl()
                        citations = [{"url": session["current_url"], "title": session["current_url"], "snippet": instruction[:220]}]
                        metadata["current_url"] = session["current_url"]
                        output = f"[sandbox-browser] {action} {session['current_url']}"
                    except Exception:
                        output = f"[sandbox-browser] Unable to resolve grounded URL for: {instruction}"
            if action == "extract":
                artifact_name = f"{step_id}-extract.txt"
                Path(artifact_name).write_text(output, encoding="utf-8")
                artifacts.append({"kind": "text", "name": artifact_name, "path": artifact_name})
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
