"""Web-based setup wizard for first-run configuration.

When Agent Nexus starts without a valid ``config/.env``, this module launches
a lightweight HTTP server on port 8090.  The user fills in the setup form in
their browser, the server writes ``config/.env``, and the process exits so the
container (``restart: unless-stopped``) automatically restarts with the new
configuration.

No external templates or static files — everything is inlined for zero
dependencies beyond *aiohttp* (already required by the bot).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

from aiohttp import web

log = logging.getLogger(__name__)

SETUP_PORT = 8090
ENV_DIR = Path("config")
ENV_FILE = ENV_DIR / ".env"

# ---------------------------------------------------------------------------
# Model / embedding / autonomy choices (mirrors setup.py)
# ---------------------------------------------------------------------------

SWARM_MODELS: list[dict[str, Any]] = [
    {"id": "minimax/minimax-m2.5", "name": "MiniMax M2.5", "hint": "Free - Programming #1", "default": True},
    {"id": "z-ai/glm-5", "name": "Z.ai GLM-5", "hint": "$0.30/$2.55 per M", "default": True},
    {"id": "moonshotai/kimi-k2.5", "name": "Kimi K2.5", "hint": "$0.23/$3.00 per M", "default": True},
    {"id": "qwen/qwen3-coder-next", "name": "Qwen3 Coder Next", "hint": "$0.12/$0.75 per M", "default": True},
    {"id": "anthropic/claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "hint": "$3/$15 per M - Expensive", "default": False},
    {"id": "google/gemini-3-flash", "name": "Gemini 3 Flash", "hint": "$0.10/$0.40 per M", "default": False},
    {"id": "openai/chatgpt-5.2", "name": "ChatGPT 5.2", "hint": "$2.50/$10 per M - Expensive", "default": False},
]

EMBEDDING_MODELS: list[dict[str, Any]] = [
    {"id": "qwen/qwen3-embedding-8b", "name": "Qwen3 Embedding 8B", "hint": "$0.01/M - Recommended"},
    {"id": "openai/text-embedding-3-small", "name": "OpenAI Embedding 3 Small", "hint": "$0.02/M"},
    {"id": "openai/text-embedding-3-large", "name": "OpenAI Embedding 3 Large", "hint": "$0.13/M"},
]

AUTONOMY_MODES: list[dict[str, str]] = [
    {"id": "escalate", "name": "Escalate (Recommended)", "hint": "Auto low-risk, ask for high-risk"},
    {"id": "observe", "name": "Observe", "hint": "Always ask before acting"},
    {"id": "autopilot", "name": "Autopilot", "hint": "Auto-execute everything"},
]


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _build_html() -> str:
    """Return the setup wizard HTML page."""

    def _model_checkboxes() -> str:
        items = []
        for m in SWARM_MODELS:
            checked = "checked" if m["default"] else ""
            items.append(
                f'<label class="model-option">'
                f'<input type="checkbox" name="swarm_models" value="{m["id"]}" {checked}>'
                f'<span class="model-name">{m["name"]}</span>'
                f'<span class="model-hint">{m["hint"]}</span>'
                f'</label>'
            )
        return "\n".join(items)

    def _embedding_radios() -> str:
        items = []
        for i, m in enumerate(EMBEDDING_MODELS):
            checked = "checked" if i == 0 else ""
            items.append(
                f'<label class="model-option">'
                f'<input type="radio" name="embedding_model" value="{m["id"]}" {checked}>'
                f'<span class="model-name">{m["name"]}</span>'
                f'<span class="model-hint">{m["hint"]}</span>'
                f'</label>'
            )
        return "\n".join(items)

    def _autonomy_radios() -> str:
        items = []
        for i, m in enumerate(AUTONOMY_MODES):
            checked = "checked" if i == 0 else ""
            items.append(
                f'<label class="model-option">'
                f'<input type="radio" name="autonomy_mode" value="{m["id"]}" {checked}>'
                f'<span class="model-name">{m["name"]}</span>'
                f'<span class="model-hint">{m["hint"]}</span>'
                f'</label>'
            )
        return "\n".join(items)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Nexus Setup</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; justify-content: center;
    padding: 2rem 1rem;
  }}
  .container {{ max-width: 640px; width: 100%; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
  .step {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;
  }}
  .step-header {{
    display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;
  }}
  .step-num {{
    background: var(--accent); color: var(--bg); width: 28px; height: 28px;
    border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 0.85rem; flex-shrink: 0;
  }}
  .step-title {{ font-weight: 600; font-size: 1.1rem; }}
  .step-hint {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 0.75rem; }}
  input[type="text"], input[type="password"] {{
    width: 100%; padding: 0.6rem 0.75rem; background: var(--bg);
    border: 1px solid var(--border); border-radius: 6px; color: var(--text);
    font-size: 0.95rem; font-family: monospace;
  }}
  input[type="text"]:focus, input[type="password"]:focus {{
    outline: none; border-color: var(--accent);
  }}
  .model-option {{
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 0; cursor: pointer;
  }}
  .model-name {{ font-weight: 500; }}
  .model-hint {{ color: var(--muted); font-size: 0.85rem; }}
  .optional {{ color: var(--muted); font-size: 0.8rem; font-weight: normal; }}
  .btn {{
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.75rem 2rem; background: var(--accent); color: var(--bg);
    border: none; border-radius: 6px; font-size: 1rem; font-weight: 600;
    cursor: pointer; margin-top: 1rem; width: 100%; justify-content: center;
  }}
  .btn:hover {{ opacity: 0.9; }}
  .btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  .test-btn {{
    background: var(--surface); color: var(--text); border: 1px solid var(--border);
    padding: 0.4rem 1rem; font-size: 0.85rem; margin-top: 0.5rem; width: auto;
  }}
  .status {{ font-size: 0.85rem; margin-top: 0.5rem; }}
  .status.ok {{ color: var(--green); }}
  .status.err {{ color: var(--red); }}
  .status.warn {{ color: var(--yellow); }}
  .result {{
    background: var(--surface); border: 1px solid var(--green);
    border-radius: 8px; padding: 1.5rem; margin-top: 1rem; display: none;
  }}
  .result h2 {{ color: var(--green); margin-bottom: 0.5rem; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="container">
  <h1>Agent Nexus Setup</h1>
  <p class="subtitle">First-time configuration. Fill in the fields below and click Save.</p>

  <form id="setup-form" method="POST" action="/save">

    <div class="step">
      <div class="step-header">
        <span class="step-num">1</span>
        <span class="step-title">Discord Bot Token</span>
      </div>
      <p class="step-hint">
        Create a bot at <a href="https://discord.com/developers/applications" target="_blank">Discord Developer Portal</a>
        &rarr; Bot &rarr; Token &rarr; Copy
      </p>
      <input type="password" name="discord_token" id="discord_token" placeholder="MTQ0OTA..." required>
    </div>

    <div class="step">
      <div class="step-header">
        <span class="step-num">2</span>
        <span class="step-title">OpenRouter API Key</span>
      </div>
      <p class="step-hint">
        Sign up free at <a href="https://openrouter.ai" target="_blank">openrouter.ai</a>
        &rarr; Keys &rarr; Create Key
      </p>
      <input type="password" name="openrouter_key" id="openrouter_key" placeholder="sk-or-v1-..." required>
      <button type="button" class="btn test-btn" onclick="testOpenRouter()">Test Connection</button>
      <div id="or-status" class="status"></div>
    </div>

    <div class="step">
      <div class="step-header">
        <span class="step-num">3</span>
        <span class="step-title">Guild ID <span class="optional">(optional)</span></span>
      </div>
      <p class="step-hint">Leave blank to auto-detect. Only needed if the bot is in multiple servers.</p>
      <input type="text" name="guild_id" placeholder="e.g. 1393618841522802850">
    </div>

    <div class="step">
      <div class="step-header">
        <span class="step-num">4</span>
        <span class="step-title">Swarm Models</span>
      </div>
      <p class="step-hint">Pick 2-4 models that will converse and collaborate in Discord.</p>
      {_model_checkboxes()}
    </div>

    <div class="step">
      <div class="step-header">
        <span class="step-num">5</span>
        <span class="step-title">Embedding Model</span>
      </div>
      <p class="step-hint" style="color: var(--red);">
        Cannot be changed after first run — changing invalidates all stored vectors.
      </p>
      {_embedding_radios()}
    </div>

    <div class="step">
      <div class="step-header">
        <span class="step-num">6</span>
        <span class="step-title">Autonomy Mode</span>
      </div>
      <p class="step-hint">Controls how the orchestrator handles task dispatch.</p>
      {_autonomy_radios()}
    </div>

    <div class="step">
      <div class="step-header">
        <span class="step-num">7</span>
        <span class="step-title">Infrastructure <span class="optional">(optional)</span></span>
      </div>
      <p class="step-hint">Defaults work with the bundled docker-compose. Override if using external services.</p>
      <label style="display:block;margin-bottom:0.5rem;color:var(--muted);font-size:0.85rem;">Qdrant URL</label>
      <input type="text" name="qdrant_url" placeholder="http://nexus-qdrant:6333">
      <label style="display:block;margin:0.75rem 0 0.5rem;color:var(--muted);font-size:0.85rem;">Redis URL</label>
      <input type="text" name="redis_url" placeholder="redis://nexus-redis:6379/0">
      <label style="display:block;margin:0.75rem 0 0.5rem;color:var(--muted);font-size:0.85rem;">Ollama URL</label>
      <input type="text" name="ollama_url" placeholder="http://host.docker.internal:11434">
    </div>

    <button type="submit" class="btn" id="save-btn">Save Configuration &amp; Restart</button>
  </form>

  <div id="result" class="result">
    <h2>Setup Complete</h2>
    <p>Configuration saved to <code>config/.env</code>.</p>
    <p style="margin-top:0.5rem;">The bot will restart automatically in a few seconds.</p>
    <p style="margin-top:0.5rem;color:var(--muted);">
      If running without Docker, restart manually: <code>python -m nexus</code>
    </p>
  </div>
</div>

<script>
async function testOpenRouter() {{
  const key = document.getElementById('openrouter_key').value;
  const el = document.getElementById('or-status');
  if (!key) {{ el.textContent = 'Enter an API key first.'; el.className = 'status warn'; return; }}
  el.textContent = 'Testing...'; el.className = 'status';
  try {{
    const r = await fetch('/test-openrouter', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{key}})
    }});
    const data = await r.json();
    if (data.ok) {{
      el.textContent = 'Connected (' + data.models + ' models available)';
      el.className = 'status ok';
    }} else {{
      el.textContent = 'Failed: ' + data.error;
      el.className = 'status err';
    }}
  }} catch(e) {{
    el.textContent = 'Connection error: ' + e.message;
    el.className = 'status err';
  }}
}}

document.getElementById('setup-form').addEventListener('submit', async function(e) {{
  e.preventDefault();
  const btn = document.getElementById('save-btn');
  btn.disabled = true; btn.textContent = 'Saving...';
  try {{
    const fd = new FormData(this);
    const r = await fetch('/save', {{
      method: 'POST',
      body: fd
    }});
    const data = await r.json();
    if (data.ok) {{
      document.getElementById('setup-form').style.display = 'none';
      document.getElementById('result').style.display = 'block';
      // Server shuts down after save — container auto-restarts.
    }} else {{
      alert('Error: ' + data.error);
      btn.disabled = false; btn.textContent = 'Save Configuration & Restart';
    }}
  }} catch(e) {{
    // Expected: server shuts down on save, fetch fails.
    document.getElementById('setup-form').style.display = 'none';
    document.getElementById('result').style.display = 'block';
  }}
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=_build_html(), content_type="text/html")


async def handle_test_openrouter(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        key = body.get("key", "")
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            count = len(data.get("data", []))
            return web.json_response({"ok": True, "models": count})
    except Exception as e:
        log.warning("OpenRouter test failed: %s", e)
        return web.json_response({"ok": False, "error": "Connection test failed. Check your API key."})


async def handle_save(request: web.Request) -> web.Response:
    try:
        data = await request.post()

        discord_token = data.get("discord_token", "").strip()
        openrouter_key = data.get("openrouter_key", "").strip()

        if not discord_token or not openrouter_key:
            return web.json_response(
                {"ok": False, "error": "Discord token and OpenRouter key are required."}
            )

        swarm_models = data.getall("swarm_models", [])
        if not swarm_models:
            return web.json_response(
                {"ok": False, "error": "Select at least one swarm model."}
            )

        embedding_model = data.get("embedding_model", "qwen/qwen3-embedding-8b")
        autonomy_mode = data.get("autonomy_mode", "escalate")
        guild_id = data.get("guild_id", "").strip()
        qdrant_url = data.get("qdrant_url", "").strip()
        redis_url = data.get("redis_url", "").strip()
        ollama_url = data.get("ollama_url", "").strip()

        # Build .env content
        lines = [
            "# Agent Nexus Configuration",
            "# Generated by web setup wizard",
            "",
            "# === REQUIRED ===",
            f"DISCORD_TOKEN={discord_token}",
            f"OPENROUTER_API_KEY={openrouter_key}",
            "",
        ]

        if guild_id:
            lines.append(f"DISCORD_GUILD_ID={guild_id}")
            lines.append("")

        lines.extend([
            "# === MODELS ===",
            f"SWARM_MODELS={','.join(swarm_models)}",
            f"EMBEDDING_MODEL={embedding_model}",
            "",
            "# === ORCHESTRATOR ===",
            f"AUTONOMY_MODE={autonomy_mode}",
            "",
        ])

        if qdrant_url:
            lines.append(f"QDRANT_URL={qdrant_url}")
        if redis_url:
            lines.append(f"REDIS_URL={redis_url}")
        if ollama_url:
            lines.append(f"OLLAMA_BASE_URL={ollama_url}")

        if qdrant_url or redis_url or ollama_url:
            lines.append("")

        # Ensure config directory exists (for fresh Docker volumes)
        ENV_DIR.mkdir(parents=True, exist_ok=True)
        ENV_FILE.write_text("\n".join(lines), encoding="utf-8")
        # Restrict permissions on .env file (Unix only)
        if sys.platform != "win32":
            os.chmod(ENV_FILE, 0o600)
        log.info("Configuration written to %s", ENV_FILE)

        # Respond before shutting down
        resp = web.json_response({"ok": True})
        await resp.prepare(request)
        await resp.write_eof()

        # Shut down the setup server — container will auto-restart with new config.
        log.info("Setup complete. Shutting down setup server...")
        raise web.GracefulExit()

    except web.GracefulExit:
        raise
    except Exception as e:
        log.exception("Setup save failed.")
        return web.json_response({"ok": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_setup_server(port: int = SETUP_PORT) -> None:
    """Start the setup wizard HTTP server (blocking)."""
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_post("/test-openrouter", handle_test_openrouter)
    app.router.add_post("/save", handle_save)

    log.info("Setup wizard available at http://127.0.0.1:%d", port)
    web.run_app(app, host="127.0.0.1", port=port, print=None)
