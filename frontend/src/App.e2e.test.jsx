import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import App from "./App";

class MockWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    MockWebSocket.instances.push(this);
    queueMicrotask(() => {
      this.readyState = 1;
      if (typeof this.onopen === "function") {
        this.onopen();
      }
    });
  }

  close() {
    this.readyState = 3;
    if (typeof this.onclose === "function") {
      this.onclose();
    }
  }

  emitClose() {
    this.readyState = 3;
    if (typeof this.onclose === "function") {
      this.onclose();
    }
  }
}

function createRunState() {
  const runs = [
    {
      id: "run-failed",
      objective: "failed export retry",
      mode: "manual",
      status: "failed",
      created_at: "2026-03-06T12:00:00+00:00",
      updated_at: "2026-03-06T12:02:00+00:00",
      step_count: 2,
      pending_approval_count: 0,
      failed_count: 1
    },
    {
      id: "run-pending",
      objective: "pending approval workflow",
      mode: "supervised",
      status: "pending_approval",
      created_at: "2026-03-06T12:05:00+00:00",
      updated_at: "2026-03-06T12:06:00+00:00",
      step_count: 3,
      pending_approval_count: 1,
      failed_count: 0
    },
    {
      id: "run-delegate",
      objective: "delegated approval handoff",
      mode: "supervised",
      status: "pending_approval",
      created_at: "2026-03-06T12:08:00+00:00",
      updated_at: "2026-03-06T12:09:00+00:00",
      step_count: 2,
      pending_approval_count: 1,
      failed_count: 0
    }
  ];
  const runDetails = {
    "run-failed": {
      id: "run-failed",
      objective: "failed export retry",
      mode: "manual",
      status: "failed",
      steps: [
        {
          id: "s1",
          action_type: "read_file",
          status: "completed",
          instruction: "{\"path\":\"notes/brief.txt\"}",
          output_text: "Draft brief",
          metadata: {
            file_path: "notes/brief.txt",
            bytes_read: 11,
            planner_source: "rule",
            planner_phase: "initial"
          }
        },
        {
          id: "s2",
          action_type: "write_file",
          status: "completed",
          instruction: "{\"path\":\"reports/draft.txt\",\"content\":\"Final report\"}",
          output_text: "[sandbox-workspace] Wrote reports/draft.txt",
          metadata: { file_path: "reports/draft.txt", changed: true, bytes_written: 12 }
        },
        {
          id: "s2b",
          action_type: "click",
          status: "completed",
          instruction: "Click the grounded `Continue` control to continue for: failed export retry",
          output_text: "[sandbox-browser-real] Clicked the requested control on https://docs.example.org/start",
          metadata: {
            target_hint: "Continue",
            target_selector: "button:has-text(\"Continue\")"
          }
        },
        {
          id: "s3",
          action_type: "execute_code",
          status: "failed",
          instruction: "{\"command\":[\"python\",\"-c\",\"print(1)\"]}",
          error_text: "command exited with status 1",
          metadata: {
            command: "python -c print(1)",
            touched_files: ["reports/generated.txt"],
            exit_code: 1
          }
        }
      ],
      citations: [],
      artifacts: [],
      approvals: [],
      child_runs: [
        {
          id: "child-1",
          objective: "Collect competitor docs",
          mode: "manual",
          status: "completed",
          parent_run_id: "run-failed",
          delegation_role: "researcher",
          delegation_status: "completed",
          delegation_summary: "Collected 3 relevant docs",
          delegation_context: {
            parent_objective: "failed export retry",
            handoff_note: "Start from prior research",
            workspace_paths: ["reports/summary.md"],
            citations: [{ url: "https://example.com/research", title: "Research brief" }],
            artifacts: [
              {
                kind: "text",
                name: "research-notes.txt",
                rel_path: "run-failed/research-notes.txt"
              }
            ]
          },
          steps: [
            {
              id: "child-step-1",
              action_type: "search_web",
              status: "completed",
              instruction: "collect competitor docs"
            },
            {
              id: "child-step-2",
              action_type: "read_file",
              status: "completed",
              instruction: "{\"path\":\"notes/competitors.md\"}"
            }
          ]
        }
      ]
    },
    "run-pending": {
      id: "run-pending",
      objective: "pending approval workflow",
      mode: "supervised",
      status: "pending_approval",
      steps: [
        { id: "p1", action_type: "navigate", status: "completed", instruction: "open page" },
        { id: "p2", action_type: "type", status: "pending_approval", instruction: "enter draft" }
      ],
      citations: [],
      artifacts: [],
      approvals: []
    },
    "run-delegate": {
      id: "run-delegate",
      objective: "delegated approval handoff",
      mode: "supervised",
      status: "pending_approval",
      steps: [
        {
          id: "d1",
          action_type: "delegate",
          status: "running",
          instruction:
            "{\"role\":\"operator\",\"objective\":\"Collect references via replanning\"}"
        },
        {
          id: "d2",
          action_type: "navigate",
          status: "pending",
          instruction: "open follow-up page"
        }
      ],
      citations: [],
      artifacts: [],
      approvals: [],
      child_runs: [
        {
          id: "child-pending",
          objective: "Collect references via replanning",
          mode: "supervised",
          status: "pending_approval",
          parent_run_id: "run-delegate",
          delegation_role: "operator",
          delegation_status: "pending_approval",
          delegation_summary: "operator awaiting delegated approval: Collect references via replanning",
          delegation_context: {
            parent_objective: "delegated approval handoff",
            handoff_note: "Only update reports within the handed-off path",
            workspace_paths: ["reports/summary.md"],
            citations: [{ url: "https://example.com/source", title: "Source page" }]
          },
          steps: [
            {
              id: "child-p1",
              action_type: "navigate",
              status: "completed",
              instruction: "open delegated workspace context"
            },
            {
              id: "child-p2",
              action_type: "write_file",
              status: "pending_approval",
              instruction: "{\"path\":\"reports/summary.md\",\"content\":\"delegated summary\"}"
            }
          ]
        }
      ]
    }
  };
  return {
    runs,
    runDetails,
    timelines: {
      "run-failed": [],
      "run-pending": [],
      "run-delegate": [
        {
          type: "delegate.started",
          timestamp: "2026-03-06T12:08:00+00:00",
          child_run_id: "child-pending",
          role: "operator",
          objective: "Collect references via replanning"
        },
        {
          type: "step.pending_approval",
          timestamp: "2026-03-06T12:08:35+00:00",
          step_id: "child-p2",
          action_type: "write_file",
          instruction: "{\"path\":\"reports/summary.md\",\"content\":\"delegated summary\"}",
          planner_source: "model",
          planner_phase: "follow_up"
        }
      ]
    },
    citations: {
      "run-failed": [],
      "run-pending": [],
      "run-delegate": []
    },
    pending: [
      {
        run_id: "run-pending",
        step_id: "p2",
        action_type: "type",
        instruction: "enter draft",
        risk_tier: "high"
      },
      {
        run_id: "child-pending",
        step_id: "child-p2",
        action_type: "write_file",
        instruction: "{\"path\":\"reports/summary.md\",\"content\":\"delegated summary\"}",
        risk_tier: "high"
      }
    ]
  };
}

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async text() {
      return JSON.stringify(payload);
    },
    async json() {
      return payload;
    }
  };
}

function createFetchMock(state) {
  const findRun = (runId) => {
    if (state.runDetails[runId]) {
      return state.runDetails[runId];
    }
    for (const run of Object.values(state.runDetails)) {
      const childRuns = Array.isArray(run.child_runs) ? run.child_runs : [];
      const child = childRuns.find((item) => item.id === runId);
      if (child) {
        return child;
      }
    }
    return null;
  };

  return vi.fn(async (input, init = {}) => {
    const method = (init.method || "GET").toUpperCase();
    const rawUrl = typeof input === "string" ? input : input.url;
    const url = new URL(rawUrl, "http://localhost");
    const path = url.pathname;

    if (path === "/api/bootstrap/status" && method === "GET") {
      return jsonResponse({
        setup_required: false,
        configured: true,
        config_path: "/tmp/config/.env",
        uses_default_admin_password: false
      });
    }
    if (path === "/api/sessions" && method === "POST") {
      return jsonResponse({
        session_id: "session-1",
        token: "token-1",
        username: "admin",
        expires_at: "2030-01-01T00:00:00+00:00"
      });
    }
    if (path === "/api/approvals/pending" && method === "GET") {
      return jsonResponse({ items: state.pending });
    }
    if (path === "/api/runs" && method === "GET") {
      const status = (url.searchParams.get("status") || "").trim();
      const mode = (url.searchParams.get("mode") || "").trim();
      const search = (url.searchParams.get("search") || "").trim().toLowerCase();
      const limit = Number(url.searchParams.get("limit") || "25");
      const offset = Number(url.searchParams.get("offset") || "0");
      let items = [...state.runs];
      if (status) {
        items = items.filter((item) => item.status === status);
      }
      if (mode) {
        items = items.filter((item) => item.mode === mode);
      }
      if (search) {
        items = items.filter((item) => item.objective.toLowerCase().includes(search));
      }
      const total = items.length;
      const sliced = items.slice(offset, offset + limit);
      return jsonResponse({ items: sliced, total, limit, offset });
    }

    const runMatch = path.match(/^\/api\/runs\/([^/]+)$/);
    if (runMatch && method === "GET") {
      const runId = runMatch[1];
      const run = findRun(runId);
      return jsonResponse(run || {}, run ? 200 : 404);
    }
    const timelineMatch = path.match(/^\/api\/runs\/([^/]+)\/timeline$/);
    if (timelineMatch && method === "GET") {
      const runId = timelineMatch[1];
      return jsonResponse({ run_id: runId, timeline: state.timelines[runId] || [] });
    }
    const citationsMatch = path.match(/^\/api\/runs\/([^/]+)\/citations$/);
    if (citationsMatch && method === "GET") {
      const runId = citationsMatch[1];
      return jsonResponse({ run_id: runId, citations: state.citations[runId] || [] });
    }
    const retryMatch = path.match(/^\/api\/runs\/([^/]+)\/retry$/);
    if (retryMatch && method === "POST") {
      const runId = retryMatch[1];
      const run = state.runDetails[runId];
      if (!run) {
        return jsonResponse({ detail: "Run not found" }, 404);
      }
      run.status = "completed";
      run.steps = run.steps.map((step) =>
        step.status === "failed" ? { ...step, status: "completed" } : step
      );
      state.runs = state.runs.map((item) =>
        item.id === runId ? { ...item, status: "completed", failed_count: 0 } : item
      );
      return jsonResponse(run);
    }
    const resumeMatch = path.match(/^\/api\/runs\/([^/]+)\/resume$/);
    if (resumeMatch && method === "POST") {
      const runId = resumeMatch[1];
      const run = findRun(runId);
      if (!run) {
        return jsonResponse({ detail: "Run not found" }, 404);
      }
      return jsonResponse(run);
    }
    const approvalMatch = path.match(/^\/api\/runs\/([^/]+)\/approvals\/([^/]+)$/);
    if (approvalMatch && method === "POST") {
      const runId = approvalMatch[1];
      const stepId = approvalMatch[2];
      const run = findRun(runId);
      if (!run) {
        return jsonResponse({ detail: "Run not found" }, 404);
      }
      const step = Array.isArray(run.steps) ? run.steps.find((item) => item.id === stepId) : null;
      if (!step) {
        return jsonResponse({ detail: "Step not found" }, 404);
      }
      step.status = "completed";
      run.status = "completed";
      state.pending = state.pending.filter((item) => item.step_id !== stepId);

      if (runId === "child-pending") {
        const parentRun = state.runDetails["run-delegate"];
        if (parentRun) {
          parentRun.status = "completed";
          parentRun.steps = parentRun.steps.map((item) =>
            item.id === "d1" || item.id === "d2" ? { ...item, status: "completed" } : item
          );
          parentRun.child_runs = parentRun.child_runs.map((childRun) =>
            childRun.id === runId
              ? {
                  ...childRun,
                  status: "completed",
                  delegation_status: "completed",
                  delegation_summary: "operator completed delegated write"
                }
              : childRun
          );
        }
        state.runs = state.runs.map((item) =>
          item.id === "run-delegate"
            ? { ...item, status: "completed", pending_approval_count: 0 }
            : item
        );
      }

      return jsonResponse(run);
    }

    return jsonResponse({ detail: `Unhandled ${method} ${path}` }, 404);
  });
}

async function login() {
  await waitFor(() => {
    expect(screen.getByText("Admin Login")).toBeInTheDocument();
  });
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: "secret" } });
  fireEvent.click(screen.getByText("Create Session"));
  await waitFor(() => {
    expect(screen.getByText("Run Inbox")).toBeInTheDocument();
  });
}

describe("App run inbox e2e", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    global.WebSocket = MockWebSocket;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    cleanup();
  });

  it("supports run inbox filter + open + retry flow", async () => {
    const state = createRunState();
    const fetchMock = createFetchMock(state);
    global.fetch = fetchMock;

    render(<App />);
    await login();

    await waitFor(() => {
      expect(screen.getByText("failed export retry")).toBeInTheDocument();
      expect(screen.getByText("pending approval workflow")).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText("Search objective"), {
      target: { value: "failed" }
    });
    fireEvent.click(screen.getByText("Apply filters"));
    await waitFor(() => {
      expect(screen.getByText("Showing 1 of 1")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("failed export retry"));
    await waitFor(() => {
      expect(screen.getByText("run-failed")).toBeInTheDocument();
    });
    expect(screen.getByText("Run Transcript")).toBeInTheDocument();
    expect(screen.getByText("notes/brief.txt")).toBeInTheDocument();
    expect(screen.getByText("reports/draft.txt")).toBeInTheDocument();
    expect(screen.getByText("python -c print(1)")).toBeInTheDocument();
    expect(screen.getByText("reports/generated.txt")).toBeInTheDocument();
    expect(screen.getByText("updated")).toBeInTheDocument();
    expect(screen.getByText("rule initial")).toBeInTheDocument();
    expect(screen.getByText("button:has-text(\"Continue\")")).toBeInTheDocument();
    expect(screen.getByText("Delegated researcher")).toBeInTheDocument();
    expect(screen.getByText("Collected 3 relevant docs")).toBeInTheDocument();
    expect(screen.getByText("Start from prior research")).toBeInTheDocument();
    expect(screen.getByText("reports/summary.md")).toBeInTheDocument();
    expect(screen.getByText("1 inherited citation")).toBeInTheDocument();
    expect(screen.getByText("research-notes.txt")).toBeInTheDocument();
    expect(screen.getByText("search_web")).toBeInTheDocument();
    expect(screen.getByText("collect competitor docs")).toBeInTheDocument();
    expect(screen.getByText("notes/competitors.md")).toBeInTheDocument();

    fireEvent.click(screen.getByText("Retry Failed Steps"));
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(([url]) => String(url).includes("/api/runs/run-failed/retry"))
      ).toBe(true);
    });
  });

  it("approves delegated child steps inline from the parent transcript", async () => {
    const state = createRunState();
    const fetchMock = createFetchMock(state);
    global.fetch = fetchMock;

    render(<App />);
    await login();

    fireEvent.click(screen.getByText("delegated approval handoff"));
    await waitFor(() => {
      expect(screen.getByText("run-delegate")).toBeInTheDocument();
    });

    expect(screen.getAllByText("Delegated operator").length).toBeGreaterThan(0);
    expect(
      screen.getByText("operator awaiting delegated approval: Collect references via replanning")
    ).toBeInTheDocument();
    expect(screen.getByText("Only update reports within the handed-off path")).toBeInTheDocument();
    expect(screen.getAllByText("reports/summary.md").length).toBeGreaterThan(0);

    const transcript = screen.getByText("Run Transcript").closest("section");
    expect(transcript).not.toBeNull();
    const transcriptScope = within(transcript);
    expect(transcriptScope.getByText("Delegation started")).toBeInTheDocument();
    expect(transcriptScope.getByText("Awaiting approval")).toBeInTheDocument();
    expect(transcriptScope.getAllByText("Runtime event").length).toBeGreaterThan(0);
    expect(transcriptScope.getByText("Approval needed")).toBeInTheDocument();
    expect(transcriptScope.getByText("Planned by model (follow-up).")).toBeInTheDocument();
    fireEvent.click(transcriptScope.getByRole("button", { name: "Approve" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([url]) => String(url).includes("/api/runs/child-pending/approvals/child-p2")
        )
      ).toBe(true);
    });
    await waitFor(() => {
      expect(screen.getByText("operator completed delegated write")).toBeInTheDocument();
    });
  });

  it("renders chat-first composer controls for new objectives", async () => {
    const state = createRunState();
    global.fetch = createFetchMock(state);

    render(<App />);
    await login();

    expect(screen.getByText("Operator Session")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send objective" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Research and summarize sources" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Research and summarize sources" }));
    const messageBox = screen.getByLabelText("Message");
    expect(messageBox.value).toContain("Research and summarize");
  });

  it("renders readable lifecycle labels in action timeline", async () => {
    const state = createRunState();
    global.fetch = createFetchMock(state);

    render(<App />);
    await login();

    fireEvent.click(screen.getByText("delegated approval handoff"));
    await waitFor(() => {
      expect(screen.getByText("run-delegate")).toBeInTheDocument();
    });

    const timelineCard = screen.getByText("Action Timeline").closest("section");
    expect(timelineCard).not.toBeNull();
    const timeline = within(timelineCard);
    expect(timeline.getByText("Delegation started")).toBeInTheDocument();
    expect(timeline.getByText("Awaiting approval")).toBeInTheDocument();
    expect(timeline.getByText("Planned by model (follow-up).")).toBeInTheDocument();
  });

  it("reconnects stream after websocket close", async () => {
    const state = createRunState();
    global.fetch = createFetchMock(state);

    render(<App />);
    await login();

    fireEvent.click(screen.getByText("pending approval workflow"));
    await waitFor(() => {
      expect(screen.getAllByText("connected").length).toBeGreaterThan(0);
    });

    expect(MockWebSocket.instances.length).toBeGreaterThan(0);
    MockWebSocket.instances[0].emitClose();

    await waitFor(() => {
      expect(screen.getAllByText("reconnecting").length).toBeGreaterThan(0);
    });
    await waitFor(
      () => {
        expect(MockWebSocket.instances.length).toBeGreaterThan(1);
      },
      { timeout: 4000 }
    );
    await waitFor(() => {
      expect(screen.getAllByText("connected").length).toBeGreaterThan(0);
    });
  });
});
