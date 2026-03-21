# Testing

Python test runs in this repo collect coverage for the app-first runtime packages by default:

```powershell
uv sync --extra dev
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests -q
```

Default coverage scope:

- `nexus_api`
- `nexus_core`
- `nexus_sandbox_runner`

Useful variants:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python -m pytest tests/test_nexus_api_app.py -q
.\.venv\Scripts\python -m pytest tests -q --cov-report=html
```

The default terminal report hides fully covered files and shows missing lines for files with gaps, so coverage regressions are visible during normal development instead of only in occasional audits.

The suite currently enforces a minimum total coverage of `80%`. Raise that threshold as the uncovered hotspots in `nexus_core.engine`, `nexus_core.planner`, and `nexus_sandbox_runner.executors` are reduced.
