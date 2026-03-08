# Incremental Autonomy And Delegation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the app-first runtime with a richer single-agent tool surface and then add bounded parent-child delegation without discarding the current safety and audit architecture.

**Architecture:** Phase 1 keeps one operator run and extends the planner and sandbox with workspace and code tools. Phase 2 adds explicit parent-child runs and delegation records so the primary operator can spawn bounded child runs, merge their results, and render that activity in the transcript.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Pydantic, React/Vite, pytest, Vitest

---

### Task 1: Expand Tool Contract And Risk Policy

**Files:**
- Modify: `src/nexus_core/policy.py`
- Modify: `src/nexus_sandbox_runner/app.py`
- Modify: `tests/test_sandbox_runner_service.py`

**Step 1: Write the failing test**

Add sandbox-runner API tests for `list_files`, `read_file`, `write_file`, `edit_file`, and `execute_code`, plus policy assertions that read tools are low risk and mutation/code tools are high risk.

**Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_sandbox_runner_service.py tests/test_nexus_core_policy.py -v`
Expected: FAIL because the action set and risk policy do not yet cover the new tools.

**Step 3: Write minimal implementation**

Add the new action names to the sandbox service contract and classify them in the policy layer.

**Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_sandbox_runner_service.py tests/test_nexus_core_policy.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus_core/policy.py src/nexus_sandbox_runner/app.py tests/test_sandbox_runner_service.py tests/test_nexus_core_policy.py
git commit -m "feat: add tool contract and policy for workspace actions"
```

### Task 2: Implement Workspace And Code Tools In The Sandbox

**Files:**
- Modify: `src/nexus_sandbox_runner/executors.py`
- Modify: `tests/test_sandbox_executors.py`
- Modify: `tests/test_sandbox_runner_service.py`

**Step 1: Write the failing test**

Add executor tests for grounded workspace reads, file mutation, and sandboxed code execution with run-scoped persistence and artifact metadata.

**Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_sandbox_executors.py tests/test_sandbox_runner_service.py -v`
Expected: FAIL because the executor does not yet support the new tool families.

**Step 3: Write minimal implementation**

Add `list_files`, `read_file`, `write_file`, `edit_file`, and `execute_code` branches to the local and Docker-backed executors. Persist touched files, outputs, and artifacts in step metadata.

**Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_sandbox_executors.py tests/test_sandbox_runner_service.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus_sandbox_runner/executors.py tests/test_sandbox_executors.py tests/test_sandbox_runner_service.py
git commit -m "feat: implement workspace and code tools in sandbox"
```

### Task 3: Teach The Planner To Choose New Tools

**Files:**
- Modify: `src/nexus_core/planner.py`
- Modify: `src/nexus_api/adaptive_planner.py`
- Modify: `src/nexus_api/app.py`
- Modify: `tests/test_nexus_api_app.py`

**Step 1: Write the failing test**

Add API and engine tests that show the planner choosing workspace or code tools after grounded results instead of forcing browser actions for every next step.

**Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_nexus_api_app.py -k "tool_loop or workspace or execute_code" -v`
Expected: FAIL because the planner still prefers the current narrower action surface.

**Step 3: Write minimal implementation**

Extend planner heuristics and model prompts to include the new actions and preserve approval-safe insertion rules.

**Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_nexus_api_app.py -k "tool_loop or workspace or execute_code" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus_core/planner.py src/nexus_api/adaptive_planner.py src/nexus_api/app.py tests/test_nexus_api_app.py
git commit -m "feat: let planner choose workspace and code tools"
```

### Task 4: Improve Transcript Detail For Tool Arguments And Outputs

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.e2e.test.jsx`

**Step 1: Write the failing test**

Add frontend coverage that expects the transcript to show richer tool detail such as file paths, code execution summaries, and mutation status.

**Step 2: Run test to verify it fails**

Run: `npm test -- --run src/App.e2e.test.jsx`
Expected: FAIL because the transcript does not yet surface structured tool detail.

**Step 3: Write minimal implementation**

Render tool metadata in the transcript without changing the overall app control flow.

**Step 4: Run test to verify it passes**

Run: `npm test -- --run src/App.e2e.test.jsx`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/App.jsx frontend/src/styles.css frontend/src/App.e2e.test.jsx
git commit -m "feat: enrich transcript with tool details"
```

### Task 5: Add Parent-Child Run Persistence

**Files:**
- Create: `alembic/versions/202603080001_add_run_delegation_tables.py`
- Modify: `src/nexus_api/models.py`
- Modify: `src/nexus_api/repository.py`
- Modify: `src/nexus_api/schemas.py`
- Test: `tests/test_nexus_api_app.py`

**Step 1: Write the failing test**

Add API-level tests for child run creation, parent-child listing, and delegation summary visibility in run detail.

**Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_nexus_api_app.py -k delegation -v`
Expected: FAIL because parent-child persistence does not exist yet.

**Step 3: Write minimal implementation**

Add `parent_run_id` and delegation records to the database and expose repository methods for child run creation and retrieval.

**Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_nexus_api_app.py -k delegation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add alembic/versions/202603080001_add_run_delegation_tables.py src/nexus_api/models.py src/nexus_api/repository.py src/nexus_api/schemas.py tests/test_nexus_api_app.py
git commit -m "feat: persist parent-child delegation state"
```

### Task 6: Add Delegation Lifecycle To The Engine

**Files:**
- Modify: `src/nexus_core/models.py`
- Modify: `src/nexus_core/engine.py`
- Modify: `src/nexus_core/planner.py`
- Modify: `src/nexus_api/adaptive_planner.py`
- Test: `tests/test_nexus_api_app.py`

**Step 1: Write the failing test**

Add engine tests for `delegate.started`, child run completion, result merge, and child failure recovery.

**Step 2: Run test to verify it fails**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_nexus_api_app.py -k "delegate_started or child_run or merge" -v`
Expected: FAIL because the engine cannot yet create or reconcile child runs.

**Step 3: Write minimal implementation**

Add a bounded delegation action path where the parent engine creates a child run, waits for completion, merges structured output, and replans from the merged result.

**Step 4: Run test to verify it passes**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_nexus_api_app.py -k "delegate_started or child_run or merge" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/nexus_core/models.py src/nexus_core/engine.py src/nexus_core/planner.py src/nexus_api/adaptive_planner.py tests/test_nexus_api_app.py
git commit -m "feat: add bounded delegation lifecycle"
```

### Task 7: Render Nested Delegation Activity In The App

**Files:**
- Modify: `frontend/src/App.jsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/src/App.e2e.test.jsx`

**Step 1: Write the failing test**

Add UI coverage that expects delegated child activity to appear inline in the parent transcript with role and status markers.

**Step 2: Run test to verify it fails**

Run: `npm test -- --run src/App.e2e.test.jsx`
Expected: FAIL because the app cannot yet render parent-child transcript relationships.

**Step 3: Write minimal implementation**

Render delegated child activity as nested transcript entries and preserve the current approval controls.

**Step 4: Run test to verify it passes**

Run: `npm test -- --run src/App.e2e.test.jsx`
Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/App.jsx frontend/src/styles.css frontend/src/App.e2e.test.jsx
git commit -m "feat: show delegated child runs in transcript"
```

### Task 8: Verify The Integrated Runtime

**Files:**
- Modify: `tests/test_nexus_api_app.py`
- Modify: `tests/test_sandbox_executors.py`
- Modify: `frontend/src/App.e2e.test.jsx`

**Step 1: Run targeted backend tests**

Run: `$env:PYTHONPATH='D:/Development/agent-nexus/src'; D:/Development/agent-nexus/.venv/Scripts/python.exe -m pytest tests/test_nexus_api_app.py tests/test_sandbox_executors.py tests/test_sandbox_runner_service.py -q`
Expected: PASS

**Step 2: Run targeted frontend tests**

Run: `npm test -- --run src/App.e2e.test.jsx`
Expected: PASS

**Step 3: Run frontend build**

Run: `npm run build`
Expected: PASS

**Step 4: Commit**

```bash
git add .
git commit -m "feat: add incremental autonomy and bounded delegation"
```
