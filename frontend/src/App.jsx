import { useMemo, useState } from "react";

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

function App() {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("change-me-now");
  const [session, setSession] = useState(null);
  const [objective, setObjective] = useState("");
  const [runId, setRunId] = useState("");
  const [run, setRun] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [citations, setCitations] = useState([]);
  const [pending, setPending] = useState([]);
  const [traceOpen, setTraceOpen] = useState(true);
  const [error, setError] = useState("");

  const token = session?.token || "";

  const pendingForCurrent = useMemo(
    () => pending.filter((i) => !runId || i.run_id === runId),
    [pending, runId]
  );

  async function login() {
    setError("");
    try {
      const data = await api("/sessions", "POST", "", { username, password });
      setSession(data);
    } catch (err) {
      setError(String(err));
    }
  }

  async function createRun() {
    if (!objective.trim()) return;
    setError("");
    try {
      const data = await api("/runs", "POST", token, {
        objective,
        mode: "supervised"
      });
      setRun(data);
      setRunId(data.id);
      await refreshRun(data.id);
      await refreshPending();
    } catch (err) {
      setError(String(err));
    }
  }

  async function refreshRun(id = runId) {
    if (!id) return;
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
    } catch (err) {
      setError(String(err));
    }
  }

  async function refreshPending() {
    if (!token) return;
    try {
      const data = await api("/approvals/pending", "GET", token);
      setPending(data.items || []);
    } catch (err) {
      setError(String(err));
    }
  }

  async function decide(stepId, decision) {
    if (!runId) return;
    setError("");
    try {
      await api(`/runs/${runId}/approvals/${stepId}`, "POST", token, {
        decision,
        reason: `Decision from web app: ${decision}`
      });
      await refreshRun(runId);
      await refreshPending();
    } catch (err) {
      setError(String(err));
    }
  }

  async function promote(artifactId) {
    if (!runId) return;
    setError("");
    try {
      await api(`/runs/${runId}/artifacts/${artifactId}/promote`, "POST", token, {
        promoted_by: username
      });
      await refreshRun(runId);
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <div className="app-shell">
      <header>
        <h1>Agent Nexus</h1>
        <p>Single-assistant control with optional trace, citations, and approvals.</p>
      </header>

      {!session ? (
        <section className="card">
          <h2>Admin Login</h2>
          <label>
            Username
            <input value={username} onChange={(e) => setUsername(e.target.value)} />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <button onClick={login}>Create Session</button>
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
                onChange={(e) => setObjective(e.target.value)}
                placeholder="Research and execute a browser-first task..."
              />
            </label>
            <div className="row">
              <button onClick={createRun}>Create Run</button>
              <button onClick={() => refreshRun()}>Refresh Run</button>
              <button onClick={() => setTraceOpen((x) => !x)}>
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
              </div>
            ) : null}
          </section>

          <aside className="card sidebar">
            <h3>Pending Approvals</h3>
            <button onClick={refreshPending}>Refresh</button>
            {pendingForCurrent.length === 0 ? (
              <p>No pending approvals.</p>
            ) : (
              pendingForCurrent.map((item) => (
                <div key={item.step_id} className="approval">
                  <p>
                    <strong>{item.action_type}</strong> ({item.step_id})
                  </p>
                  <p>{item.instruction}</p>
                  <div className="row">
                    <button onClick={() => decide(item.step_id, "approve")}>Approve</button>
                    <button className="danger" onClick={() => decide(item.step_id, "reject")}>
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
            <p>No events yet.</p>
          ) : (
            <ul>
              {timeline.map((item, idx) => (
                <li key={`${item.type}-${idx}`}>
                  <code>{item.timestamp || "n/a"}</code> - <strong>{item.type}</strong>{" "}
                  {item.action_type ? `(${item.action_type})` : ""}
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : null}

      {run ? (
        <section className="card">
          <h2>Citations</h2>
          {citations.length === 0 ? (
            <p>No citations yet.</p>
          ) : (
            <ul>
              {citations.map((c) => (
                <li key={c.id}>
                  <a href={c.url} target="_blank" rel="noreferrer">
                    {c.title || c.url}
                  </a>
                  <p>{c.snippet}</p>
                </li>
              ))}
            </ul>
          )}
          <h2>Artifacts</h2>
          {run.artifacts?.length ? (
            <ul>
              {run.artifacts.map((a) => (
                <li key={a.id}>
                  {a.name} [{a.kind}] {a.promoted ? "(promoted)" : ""}
                  {!a.promoted ? (
                    <button onClick={() => promote(a.id)}>Promote</button>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : (
            <p>No artifacts yet.</p>
          )}
        </section>
      ) : null}

      {error ? <div className="error">{error}</div> : null}
    </div>
  );
}

export default App;
