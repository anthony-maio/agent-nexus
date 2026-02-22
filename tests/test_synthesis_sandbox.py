import pytest


@pytest.mark.asyncio
async def test_sandbox_executes_simple_function():
    from nexus.synthesis.sandbox import SandboxRuntime

    sandbox = SandboxRuntime()
    result = await sandbox.execute(
        code="def add(a, b):\n    return a + b\n",
        function_name="add",
        arguments={"a": 2, "b": 3},
    )
    assert result.status.value == "success"
    assert result.output == 5


@pytest.mark.asyncio
async def test_sandbox_handles_timeout():
    from nexus.synthesis.sandbox import SandboxConfig, SandboxRuntime

    sandbox = SandboxRuntime(config=SandboxConfig(timeout_seconds=1.0))
    result = await sandbox.execute(
        code="import time\ndef slow():\n    time.sleep(10)\n    return True\n",
        function_name="slow",
        arguments={},
    )
    assert result.status.value == "timeout"


@pytest.mark.asyncio
async def test_sandbox_handles_exception():
    from nexus.synthesis.sandbox import SandboxRuntime

    sandbox = SandboxRuntime()
    result = await sandbox.execute(
        code="def fail():\n    raise ValueError('boom')\n",
        function_name="fail",
        arguments={},
    )
    assert result.status.value in ("failed", "error")
