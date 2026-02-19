"""Entry point for `python -m nexus`."""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv


def main() -> None:
    # Load .env before anything else
    load_dotenv()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("nexus")

    # Validate config early
    try:
        from nexus.config import get_settings

        settings = get_settings()
    except Exception as e:
        log.error(f"Configuration error: {e}")
        log.error("Run `python setup/setup.py` to configure Agent Nexus.")
        sys.exit(1)

    log.info("Starting Agent Nexus...")
    log.info(f"Swarm models: {settings.SWARM_MODELS}")
    log.info(f"Embedding model: {settings.EMBEDDING_MODEL}")

    # Create and run the bot
    from nexus.bot import NexusBot

    bot = NexusBot()
    bot.run(settings.DISCORD_TOKEN, log_handler=None)


if __name__ == "__main__":
    main()
