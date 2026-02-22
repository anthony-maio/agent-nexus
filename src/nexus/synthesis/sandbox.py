"""
Sandbox runtime for safe execution of untrusted code.

Implements process-based isolation with timeout enforcement.
Ported from synthesis.sandbox.runtime.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from nexus.synthesis.models import ExecutionStatus


class SandboxConfig:
    """Configuration for sandbox execution."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds


class ExecutionResult(BaseModel):
    """Result of executing code in the sandbox."""

    status: ExecutionStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class SandboxRuntime:
    """Executes code in a sandboxed subprocess."""

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()

    async def execute(
        self,
        code: str,
        function_name: str,
        arguments: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed subprocess.

        Args:
            code: Python code containing the function to execute.
            function_name: Name of the function to call.
            arguments: Keyword arguments to pass to the function.

        Returns:
            ExecutionResult with status, output, and timing.
        """
        # Write wrapper script to a temp file
        wrapper = self._create_wrapper_script(code, function_name, arguments)

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8")
        try:
            tmp.write(wrapper)
            tmp.flush()
            tmp.close()
            temp_path = tmp.name

            return await self._run_subprocess(temp_path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    async def _run_subprocess(self, script_path: str) -> ExecutionResult:
        """Run the wrapper script in a subprocess with timeout."""
        cmd = [sys.executable, script_path]
        start = asyncio.get_event_loop().time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                elapsed = (asyncio.get_event_loop().time() - start) * 1000
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error=f"Execution timed out after {self.config.timeout_seconds}s",
                    execution_time_ms=elapsed,
                )

            elapsed = (asyncio.get_event_loop().time() - start) * 1000

            if process.returncode == 0:
                try:
                    payload = json.loads(stdout.decode())
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        output=payload.get("result"),
                        execution_time_ms=elapsed,
                    )
                except json.JSONDecodeError:
                    return ExecutionResult(
                        status=ExecutionStatus.ERROR,
                        error="Invalid output format from subprocess",
                        execution_time_ms=elapsed,
                    )
            else:
                error_msg = stderr.decode().strip() or stdout.decode().strip()
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=error_msg[:1000],
                    execution_time_ms=elapsed,
                )

        except Exception as e:
            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=str(e),
                execution_time_ms=elapsed,
            )

    @staticmethod
    def _create_wrapper_script(
        code: str,
        function_name: str,
        arguments: Dict[str, Any],
    ) -> str:
        """Create the wrapper script that will be executed in the subprocess."""
        escaped_args = json.dumps(arguments)

        return f"""import json
import sys

# User code
{code}

# Execute function and output result
try:
    result = {function_name}(**{escaped_args})
    print(json.dumps({{"success": True, "result": result}}, default=str))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}), file=sys.stderr)
    sys.exit(1)
"""
