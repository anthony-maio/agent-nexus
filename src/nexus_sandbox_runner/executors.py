"""Execution backends for sandbox-runner."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
        network: str = "none",
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
    ) -> None:
        self.image = image.strip()
        self.timeout_sec = timeout_sec
        self.docker_bin = docker_bin
        self.network = network.strip()
        self.memory_limit = memory_limit.strip()
        self.cpu_limit = cpu_limit.strip()
        if not self.image:
            raise ValueError("Docker executor requires SANDBOX_DOCKER_IMAGE")

    def execute(self, request: StepRequest, sandbox_root: Path) -> StepResult:
        run_dir = sandbox_root / request.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()

        with tempfile.TemporaryDirectory(
            prefix=f"step-{request.run_id}-{request.step_id}-",
            dir=sandbox_root,
        ) as temp_dir:
            step_workspace = Path(temp_dir)
            result_file = step_workspace / "result.json"
            command = self.build_command(request, step_workspace, result_file)
            proc = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
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

    def build_command(self, request: StepRequest, step_workspace: Path, result_file: Path) -> list[str]:
        script = _container_script()
        cmd = [
            self.docker_bin,
            "run",
            "--rm",
            "--memory",
            self.memory_limit,
            "--cpus",
            self.cpu_limit,
            "--network",
            self.network,
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
        artifact_specs: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        artifacts: list[dict[str, str]] = []
        for spec in artifact_specs:
            source_name = spec.get("name", "").strip()
            if not source_name:
                continue
            source = Path(spec.get("path", source_name))
            if not source.is_absolute():
                source = Path(spec.get("workspace", ".")) / source
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
            network=env.get("SANDBOX_DOCKER_NETWORK", "none").strip() or "none",
            memory_limit=env.get("SANDBOX_DOCKER_MEMORY", "512m").strip() or "512m",
            cpu_limit=env.get("SANDBOX_DOCKER_CPUS", "1.0").strip() or "1.0",
        )
    raise ValueError(f"Unsupported SANDBOX_EXECUTION_BACKEND: {backend}")


def _container_script() -> str:
    return (
        "import json,os,urllib.parse;"
        "action=os.environ.get('NEXUS_ACTION','').strip().lower();"
        "instruction=os.environ.get('NEXUS_INSTRUCTION','');"
        "step_id=os.environ.get('NEXUS_STEP_ID','step');"
        "result_path=os.environ.get('NEXUS_OUTPUT_JSON','result.json');"
        "artifact_actions={'extract','write','export'};"
        "citation_actions={'navigate','extract'};"
        "def out(a,i):\n"
        "  \n"
        "  return ('[sandbox-browser] Navigated and collected candidate pages for: '+i) if a=='navigate' else "
        "('[sandbox-browser] Evidence summary with citations prepared. Focus: '+i) if a=='extract' else "
        "('[sandbox-browser] Export artifact prepared for promotion into canonical workspace. Request: '+i) if a=='export' else "
        "('[sandbox-browser] Executed action `'+a+'`: '+i);"
        "output=out(action,instruction);"
        "query=urllib.parse.quote_plus(instruction[:120]);"
        "cit=[{'url':'https://example.com/search?q='+query,'title':'Sandbox Search Result','snippet':'Evidence candidate for: '+instruction[:120]}] if action in citation_actions else [];"
        "arts=[];"
        "if action in artifact_actions:\n"
        "  name=f'{step_id}-{action}.txt';"
        "  with open(name,'w',encoding='utf-8') as fh: fh.write(output);"
        "  arts.append({'kind':'text','name':name,'path':name,'workspace':os.getcwd()});"
        "with open(result_path,'w',encoding='utf-8') as fh: json.dump({'output_text':output,'citations':cit,'artifacts':arts},fh);"
    )


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
            "[sandbox-browser] Export artifact prepared for promotion into canonical workspace. "
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
