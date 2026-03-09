import { useEffect, useMemo, useRef, useState } from "react";

async function api(path, method = "GET", token = "", body = null) {
  const response = await fetch(`/api${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: body ? JSON.stringify(body) : undefined
  });
  if (!response.ok) {
    throw new Error(`${response.status}: ${await response.text()}`);
  }
  return response.json();
}

const initialBootstrapForm = {
  admin_username: "admin",
  admin_password: "",
  sandbox_backend: "docker",
  browser_mode: "auto",
  openrouter_api_key: "",
  discord_token: "",
  discord_bridge_channel: "human",
  public_host: "",
  acme_email: ""
};

function parseInstructionPayload(instruction) {
  if (!instruction) {
    return null;
  }
  try {
    const parsed = JSON.parse(instruction);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function stringifyCommand(command) {
  if (Array.isArray(command)) {
    return command.join(" ");
  }
  return typeof command === "string" ? command : "";
}

function buildTranscriptDetails(step) {
  const metadata = step?.metadata && typeof step.metadata === "object" ? step.metadata : {};
  const payload = parseInstructionPayload(step?.instruction);
  const details = [];
  const filePath =
    typeof metadata.file_path === "string" && metadata.file_path
      ? metadata.file_path
      : typeof payload?.path === "string"
        ? payload.path
        : "";
  const command =
    typeof metadata.command === "string" && metadata.command
      ? metadata.command
      : stringifyCommand(payload?.command);
  const touchedFiles = Array.isArray(metadata.touched_files)
    ? metadata.touched_files.filter(Boolean)
    : [];
  const files = Array.isArray(metadata.files) ? metadata.files.filter(Boolean) : [];

  if (filePath) {
    details.push({ label: "File", value: filePath });
  }
  if (files.length) {
    details.push({ label: "Files", value: files.join(", ") });
  }
  if (typeof metadata.changed === "boolean") {
    details.push({ label: "Mutation", value: metadata.changed ? "updated" : "unchanged" });
  }
  if (command) {
    details.push({ label: "Command", value: command });
  }
  if (touchedFiles.length) {
    details.push({ label: "Touched", value: touchedFiles.join(", ") });
  }
  if (typeof metadata.exit_code === "number") {
    details.push({ label: "Exit code", value: String(metadata.exit_code) });
  }

  return details;
}

function formatInheritedCount(count, label) {
  return `${count} inherited ${label}${count === 1 ? "" : "s"}`;
}

function buildDelegationContextDetails(context) {
  if (!context || typeof context !== "object") {
    return [];
  }

  const details = [];
  const workspacePaths = Array.isArray(context.workspace_paths)
    ? context.workspace_paths.filter(Boolean)
    : [];
  const citations = Array.isArray(context.citations) ? context.citations.filter(Boolean) : [];
  const artifacts = Array.isArray(context.artifacts) ? context.artifacts.filter(Boolean) : [];
  const artifactNames = artifacts
    .map((artifact) => (typeof artifact?.name === "string" ? artifact.name : ""))
    .filter(Boolean);

  if (typeof context.handoff_note === "string" && context.handoff_note) {
    details.push({ label: "Handoff", value: context.handoff_note });
  }
  if (typeof context.parent_objective === "string" && context.parent_objective) {
    details.push({ label: "Parent", value: context.parent_objective });
  }
  if (workspacePaths.length) {
    details.push({ label: "Workspace", value: workspacePaths.join(", ") });
  }
  if (citations.length) {
    details.push({ label: "Evidence", value: formatInheritedCount(citations.length, "citation") });
  }
  if (artifactNames.length) {
    details.push({ label: "Artifacts", value: artifactNames.join(", ") });
  }

  return details;
}

function TranscriptStep({ step, variant = "assistant", approval = null, onDecide = null }) {
  const details = buildTranscriptDetails(step);

  return (
    <article className={`transcript-bubble ${variant} ${step.status}`}>
      <div className="transcript-meta">
        <strong>{step.action_type}</strong>
        <span className={`run-status ${step.status}`}>{step.status}</span>
      </div>
      <p>{step.instruction}</p>
      {details.length ? (
        <dl className="transcript-details">
          {details.map((detail) => (
            <div
              key={`${step.id}-${detail.label}-${detail.value}`}
              className="transcript-detail"
            >
              <dt>{detail.label}</dt>
              <dd>{detail.value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {step.output_text ? <p className="transcript-output">{step.output_text}</p> : null}
      {step.error_text ? <p className="transcript-error">{step.error_text}</p> : null}
      {approval && onDecide ? (
        <div className="approval inline-approval">
          <p className="approval-heading">
            <strong>Approval needed</strong>
            {approval.risk_tier ? (
              <span className="approval-risk">{`${approval.risk_tier} risk`}</span>
            ) : null}
          </p>
          <p>{approval.instruction}</p>
          <div className="row">
            <button onClick={() => onDecide(approval.run_id, approval.step_id, "approve")}>
              Approve
            </button>
            <button
              className="danger"
              onClick={() => onDecide(approval.run_id, approval.step_id, "reject")}
            >
              Reject
            </button>
          </div>
        </div>
      ) : null}
    </article>
  );
}

function App() {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [session, setSession] = useState(null);
  const [objective, setObjective] = useState("");
  const [runId, setRunId] = useState("");
  const [run, setRun] = useState(null);
  const [runs, setRuns] = useState([]);
  const [runsTotal, setRunsTotal] = useState(0);
  const [runSearch, setRunSearch] = useState("");
  const [runStatusFilter, setRunStatusFilter] = useState("");
  const [runModeFilter, setRunModeFilter] = useState("");
  const runListLimit = 12;
  const [timeline, setTimeline] = useState([]);
  const [citations, setCitations] = useState([]);
  const [pending, setPending] = useState([]);
  const [traceOpen, setTraceOpen] = useState(true);
  const [error, setError] = useState("");
  const [streamState, setStreamState] = useState("disconnected");
  const [streamRetryNonce, setStreamRetryNonce] = useState(0);
  const [bootstrapLoading, setBootstrapLoading] = useState(true);
  const [bootstrapStatus, setBootstrapStatus] = useState(null);
  const [bootstrapForm, setBootstrapForm] = useState(initialBootstrapForm);
  const [bootstrapSaving, setBootstrapSaving] = useState(false);
  const [bootstrapRestarting, setBootstrapRestarting] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const token = session?.token || "";
  const approvalScopeIds = useMemo(() => {
    const ids = [];
    if (run?.id) {
      ids.push(run.id);
    } else if (runId) {
      ids.push(runId);
    }
    const childRuns = Array.isArray(run?.child_runs) ? run.child_runs : [];
    for (const childRun of childRuns) {
      if (childRun?.id) {
        ids.push(childRun.id);
      }
    }
    return ids;
  }, [run, runId]);
  const pendingForCurrent = useMemo(
    () => {
      if (!approvalScopeIds.length) {
        return pending;
      }
      const scope = new Set(approvalScopeIds);
      return pending.filter((item) => scope.has(item.run_id));
    },
    [approvalScopeIds, pending]
  );
  const pendingByStep = useMemo(() => {
    const entries = pendingForCurrent.map((item) => [`${item.run_id}:${item.step_id}`, item]);
    return new Map(entries);
  }, [pendingForCurrent]);
  const childRunsById = useMemo(() => {
    const entries = Array.isArray(run?.child_runs)
      ? run.child_runs
          .filter((childRun) => childRun?.id)
          .map((childRun) => [childRun.id, childRun])
      : [];
    return new Map(entries);
  }, [run]);
  const hasMoreRuns = runs.length < runsTotal;
  const showBootstrap = bootstrapLoading || bootstrapRestarting || bootstrapStatus?.setup_required;

  useEffect(() => {
    loadBootstrapStatus();
  }, []);

  useEffect(() => {
    if (!bootstrapRestarting) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      loadBootstrapStatus({ silent: true });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [bootstrapRestarting]);

  useEffect(() => {
    if (!runId || !token) {
      setStreamState("disconnected");
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      return undefined;
    }

    let closedByCleanup = false;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${protocol}://${window.location.host}/api/runs/${runId}/stream?token=${encodeURIComponent(token)}`;
    const socket = new WebSocket(wsUrl);
    wsRef.current = socket;
    setStreamState(streamRetryNonce > 0 ? "reconnecting" : "connecting");

    socket.onopen = () => {
      setStreamState("connected");
    };
    socket.onmessage = async (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.event_type !== "stream.ready") {
          await refreshRun(runId);
          await refreshPending();
        }
      } catch (err) {
        setError(String(err));
      }
    };
    socket.onerror = () => {
      setStreamState("error");
    };
    socket.onclose = () => {
      if (closedByCleanup) {
        setStreamState("disconnected");
        return;
      }
      setStreamState("reconnecting");
      reconnectTimerRef.current = window.setTimeout(() => {
        setStreamRetryNonce((value) => value + 1);
      }, 1500);
    };

    return () => {
      closedByCleanup = true;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
      }
      socket.close();
      wsRef.current = null;
    };
  }, [runId, token, streamRetryNonce]);

  async function loadBootstrapStatus(options = {}) {
    const { silent = false } = options;
    if (!silent) {
      setBootstrapLoading(true);
    }
    try {
      const data = await api("/bootstrap/status");
      setBootstrapStatus(data);
      if (!data.setup_required) {
        setBootstrapRestarting(false);
        setUsername((current) => current || bootstrapForm.admin_username || "admin");
      }
    } catch (err) {
      if (!silent) {
        setError(String(err));
      }
    } finally {
      if (!silent) {
        setBootstrapLoading(false);
      }
    }
  }

  async function submitBootstrap() {
    setBootstrapSaving(true);
    setError("");
    try {
      const data = await api("/bootstrap/configure", "POST", "", bootstrapForm);
      setBootstrapStatus({
        configured: true,
        setup_required: false,
        config_path: data.config_path,
        uses_default_admin_password: false
      });
      setBootstrapRestarting(Boolean(data.restart_required));
      setUsername(bootstrapForm.admin_username);
      setPassword(bootstrapForm.admin_password);
    } catch (err) {
      setError(String(err));
    } finally {
      setBootstrapSaving(false);
    }
  }

  async function login() {
    setError("");
    try {
      const data = await api("/sessions", "POST", "", { username, password });
      setSession(data);
      await refreshPending(data.token);
      await refreshRuns(data.token);
    } catch (err) {
      setError(String(err));
    }
  }

  async function createRun() {
    if (!objective.trim()) {
      return;
    }
    setError("");
    try {
      const data = await api("/runs", "POST", token, {
        objective,
        mode: "supervised"
      });
      setStreamRetryNonce(0);
      setRun(data);
      setRunId(data.id);
      await refreshRun(data.id);
      await refreshPending();
      await refreshRuns();
    } catch (err) {
      setError(String(err));
    }
  }

  async function refreshRun(id = runId) {
    if (!id) {
      return;
    }
    setError("");
    try {
      const [runData, timelineData, citationsData] = await Promise.all([
        api(`/runs/${id}`, "GET", token),
        api(`/runs/${id}/timeline`, "GET", token),
        api(`/runs/${id}/citations`, "GET", token)
      ]);
      setRun(runData);
      setTimeline(timelineData.timeline || []);
      setCitations(citationsData.citations || []);
      await refreshRuns();
    } catch (err) {
      setError(String(err));
    }
  }

  async function refreshPending(currentToken = token) {
    if (!currentToken) {
      return;
    }
    try {
      const data = await api("/approvals/pending", "GET", currentToken);
      setPending(data.items || []);
    } catch (err) {
      setError(String(err));
    }
  }

  function buildRunsPath(offset = 0, filters = null) {
    const search = filters?.search ?? runSearch;
    const status = filters?.status ?? runStatusFilter;
    const mode = filters?.mode ?? runModeFilter;
    const params = new URLSearchParams();
    params.set("limit", String(runListLimit));
    params.set("offset", String(offset));
    if (search.trim()) {
      params.set("search", search.trim());
    }
    if (status) {
      params.set("status", status);
    }
    if (mode) {
      params.set("mode", mode);
    }
    return `/runs?${params.toString()}`;
  }

  async function refreshRuns(currentToken = token, options = {}) {
    if (!currentToken) {
      return;
    }
    const { offset = 0, append = false, filters = null } = options;
    try {
      const data = await api(buildRunsPath(offset, filters), "GET", currentToken);
      const nextItems = data.items || [];
      setRuns((current) => (append ? [...current, ...nextItems] : nextItems));
      setRunsTotal(data.total || 0);
    } catch (err) {
      setError(String(err));
    }
  }

  async function applyRunFilters() {
    await refreshRuns(token, { offset: 0, append: false });
  }

  async function clearRunFilters() {
    setRunSearch("");
    setRunStatusFilter("");
    setRunModeFilter("");
    await refreshRuns(token, {
      offset: 0,
      append: false,
      filters: { search: "", status: "", mode: "" }
    });
  }

  async function loadMoreRuns() {
    await refreshRuns(token, { offset: runs.length, append: true });
  }

  async function openRun(id) {
    setStreamRetryNonce(0);
    setRunId(id);
    await refreshRun(id);
    await refreshPending();
  }

  async function decide(targetRunId, stepId, decision) {
    if (!targetRunId) {
      return;
    }
    setError("");
    try {
      await api(`/runs/${targetRunId}/approvals/${stepId}`, "POST", token, {
        decision,
        reason: `Decision from web app: ${decision}`
      });
      await refreshRun(runId || targetRunId);
      await refreshPending();
      await refreshRuns();
    } catch (err) {
      setError(String(err));
    }
  }

  async function promote(artifactId) {
    if (!runId) {
      return;
    }
    setError("");
    try {
      await api(`/runs/${runId}/artifacts/${artifactId}/promote`, "POST", token, {
        promoted_by: username
      });
      await refreshRun(runId);
      await refreshRuns();
    } catch (err) {
      setError(String(err));
    }
  }

  async function resumeCurrentRun() {
    if (!runId) {
      return;
    }
    setError("");
    try {
      const data = await api(`/runs/${runId}/resume`, "POST", token);
      setRun(data);
      await refreshRun(runId);
      await refreshPending();
      await refreshRuns();
    } catch (err) {
      setError(String(err));
    }
  }

  async function retryCurrentRun() {
    if (!runId) {
      return;
    }
    setError("");
    try {
      const data = await api(`/runs/${runId}/retry`, "POST", token);
      setRun(data);
      await refreshRun(runId);
      await refreshPending();
      await refreshRuns();
    } catch (err) {
      setError(String(err));
    }
  }

  function updateBootstrapField(field, value) {
    setBootstrapForm((current) => ({ ...current, [field]: value }));
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <span className="eyebrow">Agent Nexus</span>
          <h1>Supervised browser automation, app-first.</h1>
          <p>
            Primary web control plane with approvals, citations, sandboxed execution,
            and an optional trace surface.
          </p>
        </div>
        <div className="hero-panel">
          <div className="metric">
            <span>Bootstrap</span>
            <strong>
              {bootstrapLoading ? "checking" : bootstrapStatus?.setup_required ? "required" : "ready"}
            </strong>
          </div>
          <div className="metric">
            <span>Stream</span>
            <strong>{streamState}</strong>
          </div>
        </div>
      </header>

      {showBootstrap ? (
        <section className="card setup-card">
          <div className="setup-header">
            <div>
              <h2>{bootstrapRestarting ? "Restarting control plane" : "First-run setup"}</h2>
              <p>
                {bootstrapRestarting
                  ? "Configuration was written. The API will come back up with the new settings."
                  : "This replaces the old Discord-first setup path. Configure the app runtime first, then optionally add the Discord bridge."}
              </p>
            </div>
            {bootstrapStatus?.config_path ? (
              <code>{bootstrapStatus.config_path}</code>
            ) : null}
          </div>

          {bootstrapRestarting ? (
            <div className="callout">
              <p>Waiting for the API to restart and report healthy bootstrap status.</p>
              <button onClick={() => loadBootstrapStatus()}>Check readiness</button>
            </div>
          ) : (
            <div className="setup-grid">
              <label>
                Admin Username
                <input
                  value={bootstrapForm.admin_username}
                  onChange={(event) => updateBootstrapField("admin_username", event.target.value)}
                />
              </label>
              <label>
                Admin Password
                <input
                  type="password"
                  value={bootstrapForm.admin_password}
                  onChange={(event) => updateBootstrapField("admin_password", event.target.value)}
                />
              </label>
              <label>
                Sandbox Backend
                <select
                  value={bootstrapForm.sandbox_backend}
                  onChange={(event) => updateBootstrapField("sandbox_backend", event.target.value)}
                >
                  <option value="docker">Docker sandbox</option>
                  <option value="docker-host">Host socket</option>
                  <option value="local">Local process</option>
                </select>
              </label>
              <label>
                Browser Mode
                <select
                  value={bootstrapForm.browser_mode}
                  onChange={(event) => updateBootstrapField("browser_mode", event.target.value)}
                >
                  <option value="auto">Auto fallback</option>
                  <option value="real">Require real browser</option>
                  <option value="simulated">Simulated only</option>
                </select>
              </label>
              <label className="span-2">
                OpenRouter API Key
                <input
                  type="password"
                  value={bootstrapForm.openrouter_api_key}
                  onChange={(event) => updateBootstrapField("openrouter_api_key", event.target.value)}
                  placeholder="Optional for bootstrap, recommended for model-backed runs"
                />
              </label>
              <label>
                Public Host
                <input
                  value={bootstrapForm.public_host}
                  onChange={(event) => updateBootstrapField("public_host", event.target.value)}
                  placeholder="Optional domain for Caddy/TLS"
                />
              </label>
              <label>
                ACME Email
                <input
                  value={bootstrapForm.acme_email}
                  onChange={(event) => updateBootstrapField("acme_email", event.target.value)}
                  placeholder="Optional certificate contact"
                />
              </label>
              <label className="span-2">
                Discord Token
                <input
                  type="password"
                  value={bootstrapForm.discord_token}
                  onChange={(event) => updateBootstrapField("discord_token", event.target.value)}
                  placeholder="Optional, only for the secondary bridge"
                />
              </label>
              <label className="span-2">
                Discord Bridge Channel
                <input
                  value={bootstrapForm.discord_bridge_channel}
                  onChange={(event) => updateBootstrapField("discord_bridge_channel", event.target.value)}
                />
              </label>
            </div>
          )}

          {!bootstrapRestarting ? (
            <div className="row">
              <button disabled={bootstrapSaving} onClick={submitBootstrap}>
                {bootstrapSaving ? "Saving setup..." : "Write config and restart"}
              </button>
              <button className="ghost" onClick={() => loadBootstrapStatus()}>
                Refresh status
              </button>
            </div>
          ) : null}
        </section>
      ) : !session ? (
        <section className="card">
          <h2>Admin Login</h2>
          <p className="subtle">Bootstrap is complete. Sign in to start or supervise runs.</p>
          <label>
            Username
            <input value={username} onChange={(event) => setUsername(event.target.value)} />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          <div className="row">
            <button onClick={login}>Create Session</button>
            <button className="ghost" onClick={() => loadBootstrapStatus()}>
              Re-check setup
            </button>
          </div>
        </section>
      ) : (
        <main className="layout">
          <section className="card primary">
            <h2>Assistant</h2>
            <label>
              Objective
              <textarea
                rows={5}
                value={objective}
                onChange={(event) => setObjective(event.target.value)}
                placeholder="Research and execute a browser-first task..."
              />
            </label>
            <div className="row">
              <button onClick={createRun}>Create Run</button>
              <button className="ghost" onClick={() => refreshRun()}>
                Refresh Run
              </button>
              <button className="ghost" onClick={() => refreshRuns()}>
                Refresh Inbox
              </button>
              <button className="ghost" onClick={() => setTraceOpen((open) => !open)}>
                {traceOpen ? "Hide" : "Show"} Trace
              </button>
            </div>
            {run ? (
              <div className="result">
                <p>
                  <strong>Run:</strong> {run.id}
                </p>
                <p>
                  <strong>Status:</strong> {run.status}
                </p>
                <p>
                  <strong>Mode:</strong> {run.mode}
                </p>
                <p>
                  <strong>Live Stream:</strong> {streamState}
                </p>
                <div className="row">
                  <button
                    className="ghost"
                    onClick={resumeCurrentRun}
                    disabled={!["paused", "running", "pending_approval"].includes(run.status)}
                  >
                    Resume Run
                  </button>
                  <button
                    className="ghost"
                    onClick={retryCurrentRun}
                    disabled={run.status !== "failed"}
                  >
                    Retry Failed Steps
                  </button>
                </div>
              </div>
            ) : (
              <p className="subtle">
                Create a run to produce citations, artifacts, and approval-gated actions.
              </p>
            )}
            {run ? (
              <section className="transcript-panel">
                <div className="transcript-header">
                  <h3>Run Transcript</h3>
                  <span className={`run-status ${run.status}`}>{run.status}</span>
                </div>
                <div className="transcript">
                  <article className="transcript-bubble user">
                    <span className="transcript-role">Objective</span>
                    <p>{run.objective}</p>
                  </article>
                  {(run.steps || []).map((step) => (
                    <TranscriptStep
                      key={step.id}
                      step={step}
                      approval={pendingByStep.get(`${run.id}:${step.id}`) || null}
                      onDecide={decide}
                    />
                  ))}
                  {(run.child_runs || []).map((childRun) => {
                    const contextDetails = buildDelegationContextDetails(childRun.delegation_context);
                    return (
                      <article key={childRun.id} className="transcript-bubble delegate">
                        <span className="transcript-role">
                          Delegated {childRun.delegation_role || "worker"}
                        </span>
                        <div className="transcript-meta">
                          <strong>{childRun.objective}</strong>
                          <span
                            className={`run-status ${
                              childRun.delegation_status || childRun.status || "completed"
                            }`}
                          >
                            {childRun.delegation_status || childRun.status}
                          </span>
                        </div>
                        <p>{childRun.delegation_summary || childRun.delegation_objective}</p>
                        {contextDetails.length ? (
                          <dl className="transcript-details delegate-context">
                            {contextDetails.map((detail) => (
                              <div
                                key={`${childRun.id}-${detail.label}-${detail.value}`}
                                className="transcript-detail"
                              >
                                <dt>{detail.label}</dt>
                                <dd>{detail.value}</dd>
                              </div>
                            ))}
                          </dl>
                        ) : null}
                        {childRun.steps?.length ? (
                          <div className="delegate-steps">
                            {childRun.steps.map((step) => (
                              <TranscriptStep
                                key={step.id}
                                step={step}
                                variant="child-step"
                                approval={pendingByStep.get(`${childRun.id}:${step.id}`) || null}
                                onDecide={decide}
                              />
                            ))}
                          </div>
                        ) : null}
                      </article>
                    );
                  })}
                </div>
              </section>
            ) : null}
          </section>

          <aside className="card sidebar">
            <h3>Run Inbox</h3>
            <button className="ghost" onClick={() => refreshRuns()}>
              Refresh runs
            </button>
            <label>
              Search objective
              <input
                value={runSearch}
                onChange={(event) => setRunSearch(event.target.value)}
                placeholder="keyword"
              />
            </label>
            <label>
              Status
              <select
                value={runStatusFilter}
                onChange={(event) => setRunStatusFilter(event.target.value)}
              >
                <option value="">All statuses</option>
                <option value="running">Running</option>
                <option value="pending_approval">Pending approval</option>
                <option value="paused">Paused</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
            </label>
            <label>
              Mode
              <select
                value={runModeFilter}
                onChange={(event) => setRunModeFilter(event.target.value)}
              >
                <option value="">All modes</option>
                <option value="supervised">Supervised</option>
                <option value="manual">Manual</option>
                <option value="autopilot">Autopilot</option>
              </select>
            </label>
            <div className="row">
              <button className="ghost" onClick={applyRunFilters}>
                Apply filters
              </button>
              <button className="ghost" onClick={clearRunFilters}>
                Clear filters
              </button>
            </div>
            {runs.length === 0 ? (
              <p className="subtle">No runs yet.</p>
            ) : (
              <>
                <p className="subtle">
                  Showing {runs.length} of {runsTotal}
                </p>
                <ul className="stack-list run-list">
                  {runs.map((item) => (
                    <li key={item.id}>
                      <button
                        className={`run-item ${item.id === runId ? "active" : ""}`}
                        onClick={() => openRun(item.id)}
                      >
                        <span className="run-main">{item.objective}</span>
                        <span className="run-meta">
                          <span className={`run-status ${item.status}`}>{item.status}</span>
                          <span>{item.mode}</span>
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
                {hasMoreRuns ? (
                  <button className="ghost" onClick={loadMoreRuns}>
                    Load more
                  </button>
                ) : null}
              </>
            )}

            <h3 className="sidebar-heading">Pending Approvals</h3>
            <button className="ghost" onClick={() => refreshPending()}>
              Refresh approvals
            </button>
            {pendingForCurrent.length === 0 ? (
              <p className="subtle">No pending approvals.</p>
            ) : (
              pendingForCurrent.map((item) => (
                <div key={item.step_id} className="approval">
                  <p>
                    <strong>{item.action_type}</strong> ({item.step_id})
                    {item.run_id !== run?.id && childRunsById.has(item.run_id) ? (
                      <span className="approval-scope">
                        {`Delegated ${childRunsById.get(item.run_id)?.delegation_role || "worker"}`}
                      </span>
                    ) : null}
                  </p>
                  <p>{item.instruction}</p>
                  <div className="row">
                    <button onClick={() => decide(item.run_id, item.step_id, "approve")}>
                      Approve
                    </button>
                    <button
                      className="danger"
                      onClick={() => decide(item.run_id, item.step_id, "reject")}
                    >
                      Reject
                    </button>
                  </div>
                </div>
              ))
            )}
          </aside>
        </main>
      )}

      {traceOpen && run ? (
        <section className="card">
          <h2>Action Timeline</h2>
          {timeline.length === 0 ? (
            <p className="subtle">No events yet.</p>
          ) : (
            <ul className="timeline">
              {timeline.map((item, index) => (
                <li key={`${item.type}-${index}`}>
                  <code>{item.timestamp || "n/a"}</code>
                  <strong>{item.type}</strong>
                  {item.action_type ? <span>{item.action_type}</span> : null}
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : null}

      {run ? (
        <section className="card split-card">
          <div>
            <h2>Citations</h2>
            {citations.length === 0 ? (
              <p className="subtle">No citations yet.</p>
            ) : (
              <ul className="stack-list">
                {citations.map((citation) => (
                  <li key={citation.id}>
                    <a href={citation.url} target="_blank" rel="noreferrer">
                      {citation.title || citation.url}
                    </a>
                    <p>{citation.snippet}</p>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <h2>Artifacts</h2>
            {run.artifacts?.length ? (
              <ul className="stack-list">
                {run.artifacts.map((artifact) => (
                  <li key={artifact.id}>
                    <p>
                      <strong>{artifact.name}</strong> [{artifact.kind}]{" "}
                      {artifact.promoted ? "(promoted)" : ""}
                    </p>
                    {!artifact.promoted ? (
                      <button onClick={() => promote(artifact.id)}>Promote</button>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="subtle">No artifacts yet.</p>
            )}
          </div>
        </section>
      ) : null}

      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}

export default App;
