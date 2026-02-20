"""Entry point for `python -m nexus`."""

from __future__ import annotations

import json
import logging
import os
import sys

from dotenv import load_dotenv


def _preprocess_env() -> None:
    """Convert comma-separated env values to JSON arrays.

    pydantic-settings v2 tries ``json.loads()`` on environment values for
    ``list`` fields.  A plain comma-separated string like
    ``model-a,model-b`` is not valid JSON and causes a parse error in some
    versions.  Pre-converting to ``["model-a", "model-b"]`` makes it safe
    across all pydantic-settings releases.
    """
    for key in ("SWARM_MODELS",):
        val = os.environ.get(key, "")
        if val and not val.startswith("["):
            os.environ[key] = json.dumps(
                [m.strip() for m in val.split(",") if m.strip()]
            )


def main() -> None:
    # Load .env from canonical locations before anything else.
    load_dotenv("config/.env")  # Primary (Docker + local)
    load_dotenv()               # Fallback (CWD/.env)

    # Normalise list-type env vars for pydantic-settings compatibility.
    _preprocess_env()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("nexus")

    # Check if config exists; if not, offer the web setup wizard.
    from nexus.config import has_config

    if not has_config():
        log.warning("No configuration found (missing DISCORD_TOKEN / OPENROUTER_API_KEY).")
        log.info("Starting web setup wizard on http://127.0.0.1:8090 ...")
        log.info("Open your browser to complete first-time setup.")
        try:
            from nexus.setup_web import run_setup_server
            run_setup_server()
        except Exception:
            log.exception("Web setup wizard failed.")
            log.error(
                "Manual setup: copy config/.env.example to config/.env and fill in "
                "DISCORD_TOKEN and OPENROUTER_API_KEY, then restart."
            )
        sys.exit(0)

    # Validate config early
    try:
        from nexus.config import get_settings

        settings = get_settings()
    except Exception as e:
        log.error("Configuration error: %s", e)
        log.error("")
        log.error("  How to fix:")
        log.error("  1. Edit config/.env (or set environment variables)")
        log.error("  2. Ensure DISCORD_TOKEN and OPENROUTER_API_KEY are set")
        log.error("  3. SWARM_MODELS should be comma-separated model IDs")
        log.error("     e.g. SWARM_MODELS=minimax/minimax-m2.5,z-ai/glm-5")
        log.error("")
        log.error("  Or run: python setup/setup.py")
        sys.exit(1)

    log.info("Starting Agent Nexus...")
    log.info("Swarm models: %s", settings.SWARM_MODELS)
    log.info("Embedding model: %s", settings.EMBEDDING_MODEL)

    # Create and run the bot
    from nexus.bot import NexusBot

    bot = NexusBot()
    bot.run(settings.DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
