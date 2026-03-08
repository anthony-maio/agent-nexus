# Incremental Autonomy And Delegation Design

## Context

The app-first runtime now has the right base shape: it can bootstrap a run from a small set of initial actions, persist run-scoped sandbox state, replan after grounded results, and expose the selected run as a visible transcript in the app. The remaining gaps are narrower, but still important. The tool surface is still smaller than the clone, and delegation is still absent from the app-first path.

The next phase should not replace the current runtime. It should deepen it. The existing safety model, approval gates, timeline events, and promotion flow are already stronger than the clone. The design should keep those properties and extend them.

## Recommendation

Take the work in two phases.

Phase 1 widens the single-agent tool surface. The operator remains one run, one planner, one execution loop, and one approval system. The planner gains a richer set of tool choices so it can use the browser when needed, but switch to workspace and code tools when that is the better move.

Phase 2 adds bounded delegation. The primary run stays in charge, but it can create a child run for a narrow task, wait for the result, then continue. The child uses the same engine, sandbox, approval rules, and transcript model as the parent.

This sequence keeps the architecture coherent and reduces risk. It also produces user-visible progress after each phase.

## Phase 1: Richer Single-Agent Tools

Phase 1 adds four tool families.

- Research tools: `search_web`, `fetch_url`
- Workspace read tools: `list_files`, `read_file`
- Workspace write tools: `write_file`, `edit_file`
- Sandbox code tool: `execute_code`

The current `StepDefinition` contract can stay simple. Each step still carries `action_type` and `instruction`. Execution returns richer metadata so the planner and transcript can carry precise tool context without redesigning the whole schema. Useful metadata includes file paths touched, command summaries, current workspace path, top search result, active URL, and artifacts produced.

Risk handling should follow tool family rather than implementation detail. Search and read actions are low risk. File mutation, code execution, and submission remain high risk in supervised mode. That keeps the existing approval flow intact and predictable.

This phase gives the planner better options, reduces browser overuse, and lays the vocabulary delegation will need later.

## Phase 2: Bounded Delegation

Delegation should start with a parent-child run model, not a free-form swarm runtime.

The parent run can create a child run with:

- a role
- a narrow objective
- inherited context
- an expected output

Inherited context should include the parent objective, selected citations, active sandbox session summary, relevant artifacts, and any workspace paths the child may read. It should not include live mutable browser ownership in the first version. Shared live state makes failure handling and audit trails much harder. A snapshot handoff is enough to start.

The first roles should stay narrow.

- `researcher`: search, fetch, read, summarize
- `operator`: browser and workspace actions for an explicit subtask

That gives useful delegation without creating a scheduling problem the runtime is not ready to solve.

## Runtime Flow

The control flow should stay event-driven.

1. Parent run starts.
2. Planner selects a tool or a delegation action.
3. If it selects a tool, the engine executes it and stores the result.
4. If it selects delegation, the engine creates a child run and emits `delegate.started`.
5. Child run executes with inherited context and writes its own citations, artifacts, and transcript events.
6. When the child finishes, the engine emits `delegate.completed`, merges the child result back into the parent context, and asks the planner for the next parent action.

The app should render this as one continuous transcript with nested worker activity, not as a separate control surface the user has to reconcile manually.

## Error Handling

Child failures should not crash the parent by default. The parent should receive a structured result with:

- failure summary
- retryable flag
- partial outputs, if any

The parent planner can then retry, continue without the child branch, or escalate for approval.

The same rule applies to Phase 1 tools. Code execution and file mutation should produce precise metadata about what changed so failures and approvals remain legible.

## Data Model

Phase 1 can stay mostly within the current schema.

Phase 2 should add explicit delegation records and parent-child run relationships. At minimum:

- `runs.parent_run_id`
- delegation record with `parent_run_id`, `child_run_id`, `role`, `objective`, `status`, and summary fields

This should be persisted first-class in the database, not hidden inside step metadata. That makes retrieval, UI rendering, and audit history simpler.

## Testing Strategy

Testing should follow the rollout.

Phase 1:

- executor tests for each new tool family
- policy tests for tool risk tiers
- API and engine tests for planner selection and approval gates
- transcript/UI tests for visible tool detail

Phase 2:

- repository tests for parent-child persistence
- engine tests for delegation start, completion, failure, and merge
- API tests for child run visibility in run detail and timeline
- UI tests for nested transcript rendering

## Rollout Checkpoints

Phase 1 is complete when the planner can choose browser, workspace, and code tools safely within the existing engine.

Phase 2 is complete when one parent run can delegate one bounded task to one child run and merge the result back into the parent transcript reliably.

Parallel delegation, specialist teams, and broader swarm behavior should wait until that path is stable.
