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
        network: str = "none",
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        pids_limit: int = 128,
    ) -> None:
        self.image = image.strip()
        self.timeout_sec = timeout_sec
        self.docker_bin = docker_bin
        self.docker_host = docker_host.strip()
        self.network = network.strip()
        self.memory_limit = memory_limit.strip()
        self.cpu_limit = cpu_limit.strip()
        self.pids_limit = pids_limit
        if not self.image:
            raise ValueError("Docker executor requires SANDBOX_DOCKER_IMAGE")

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
                env = None
                if self.docker_host:
                    env = dict(os.environ)
                    env["DOCKER_HOST"] = self.docker_host
                proc = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    env=env,
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


def build_executor_from_env(env: dict[str, str]) -> StepExecutor:
    backend = env.get("SANDBOX_EXECUTION_BACKEND", "local").strip().lower()
    if backend in {"", "local"}:
        return LocalEphemeralExecutor()
    if backend == "docker":
        return DockerEphemeralExecutor(
            image=env.get("SANDBOX_DOCKER_IMAGE", "").strip(),
            timeout_sec=int(env.get("SANDBOX_STEP_TIMEOUT_SEC", "120")),
            docker_bin=env.get("SANDBOX_DOCKER_BIN", "docker").strip() or "docker",
            docker_host=env.get("SANDBOX_DOCKER_HOST", "").strip(),
            network=env.get("SANDBOX_DOCKER_NETWORK", "none").strip() or "none",
            memory_limit=env.get("SANDBOX_DOCKER_MEMORY", "512m").strip() or "512m",
            cpu_limit=env.get("SANDBOX_DOCKER_CPUS", "1.0").strip() or "1.0",
            pids_limit=int(env.get("SANDBOX_DOCKER_PIDS", "128")),
        )
    raise ValueError(f"Unsupported SANDBOX_EXECUTION_BACKEND: {backend}")


def _container_script() -> str:
    return dedent(
        """
        import json
        import os
        import urllib.parse

        action = os.environ.get("NEXUS_ACTION", "").strip().lower()
        instruction = os.environ.get("NEXUS_INSTRUCTION", "")
        step_id = os.environ.get("NEXUS_STEP_ID", "step")
        result_path = os.environ.get("NEXUS_OUTPUT_JSON", "result.json")

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
