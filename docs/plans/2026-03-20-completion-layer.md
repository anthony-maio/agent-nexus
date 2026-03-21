# Completion Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit run completion layer so the runtime decides `verified`, `provisional`, or `blocked` from grounded outcomes instead of marking runs complete whenever no pending steps remain.

**Architecture:** Introduce a small verification module in `nexus_core` that evaluates a run's completed steps, citations, artifacts, and objective strategy. Integrate it into the engine's terminal path so completion is a verified kernel decision, persist the result in run metadata, surface it in the timeline, and fail clearly when a run stops without meeting minimum success criteria.

**Tech Stack:** Python, FastAPI runtime models, SQLAlchemy-backed repository, pytest integration tests

---

### Task 1: Add Failing Integration Tests For Run-Level Verification

**Files:**
- Modify: `D:\Development\agent-nexus\tests\test_nexus_api_app.py`

**Step 1: Write the failing test**

Add one integration test showing a research-style run that stops after grounded extraction without any terminal artifact step is not considered complete. Add one integration test showing a run with citations plus a terminal artifact step records a successful run verification result.

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src .\.venv\Scripts\python -m pytest tests/test_nexus_api_app.py -q --no-cov`

Expected: failure because the runtime currently marks the incomplete run as `completed` with no explicit run verification metadata.

**Step 3: Commit**

```bash
git add tests/test_nexus_api_app.py
git commit -m "test: add failing completion layer scenarios"
```

### Task 2: Implement Run Completion Verification

**Files:**
- Create: `D:\Development\agent-nexus\src\nexus_core\verification.py`
- Modify: `D:\Development\agent-nexus\src\nexus_core\engine.py`
- Modify: `D:\Development\agent-nexus\src\nexus_core\models.py`

**Step 1: Write the minimal verification model**

Create a small runtime model representing run verification output:
- strategy
- result (`verified`, `provisional`, `blocked`)
- reason
- signals

**Step 2: Implement evaluation logic**

Start with pragmatic strategy-aware criteria:
- research/API/reporting tasks require grounded citations plus a terminal synthesis/export artifact to be `verified`
- workflow tasks require a terminal completion action such as `submit`
- artifact tasks require a completed artifact-producing action with at least one artifact
- all other strategies may return `provisional` until stronger criteria exist

**Step 3: Integrate into engine terminal path**

Replace the unconditional `RunStatus.COMPLETED` path when `next_step is None` with:
- evaluate run completion
- persist `run_verification` in run metadata
- publish a `run.verification` event payload
- mark run `completed` for `verified`/`provisional`
- mark run `failed` for `blocked`

**Step 4: Run targeted tests**

Run: `PYTHONPATH=src .\.venv\Scripts\python -m pytest tests/test_nexus_api_app.py -q --no-cov`

Expected: new completion-layer tests pass.

**Step 5: Commit**

```bash
git add src/nexus_core/verification.py src/nexus_core/engine.py src/nexus_core/models.py tests/test_nexus_api_app.py
git commit -m "feat: add run completion verification"
```

### Task 3: Surface Completion Verification In Timeline And Responses

**Files:**
- Modify: `D:\Development\agent-nexus\src\nexus_api\repository.py`
- Modify: `D:\Development\agent-nexus\tests\test_nexus_api_app.py`

**Step 1: Add timeline emission**

Expose a `run.verification` event from persisted run metadata so the transcript/timeline can explain why a run completed or failed.

**Step 2: Add assertions**

Assert the event and metadata shape in the integration tests from Task 1.

**Step 3: Run targeted tests**

Run: `PYTHONPATH=src .\.venv\Scripts\python -m pytest tests/test_nexus_api_app.py -q --no-cov`

Expected: timeline assertions pass.

**Step 4: Commit**

```bash
git add src/nexus_api/repository.py tests/test_nexus_api_app.py
git commit -m "feat: surface run verification events"
```

### Task 4: Full Validation

**Files:**
- Verify only

**Step 1: Run full suite**

Run: `PYTHONPATH=src .\.venv\Scripts\python -m pytest tests -q`

Expected: full suite passes and the coverage gate remains green.

**Step 2: Check runtime metadata manually if needed**

Inspect one completed run response and one failed verification response in tests to confirm `run.metadata.run_verification` and timeline entries are stable.

**Step 3: Commit**

```bash
git add .
git commit -m "test: validate completion layer end-to-end"
```
