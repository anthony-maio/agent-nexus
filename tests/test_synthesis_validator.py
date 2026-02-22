import pytest


@pytest.mark.asyncio
async def test_valid_code_passes():
    from nexus.synthesis.validator import CodeValidator

    v = CodeValidator()
    result = await v.validate("def add(a, b):\n    return a + b\n")
    assert result.is_valid


@pytest.mark.asyncio
async def test_os_import_blocked():
    from nexus.synthesis.validator import CodeValidator

    v = CodeValidator()
    result = await v.validate("import os\ndef run():\n    os.system('ls')\n")
    assert not result.is_valid
    assert any("os" in i.message for i in result.issues)


@pytest.mark.asyncio
async def test_eval_blocked():
    from nexus.synthesis.validator import CodeValidator

    v = CodeValidator()
    result = await v.validate("def run(x):\n    return eval(x)\n")
    assert not result.is_valid


@pytest.mark.asyncio
async def test_syntax_error_caught():
    from nexus.synthesis.validator import CodeValidator

    v = CodeValidator()
    result = await v.validate("def broken(\n")
    assert not result.is_valid
