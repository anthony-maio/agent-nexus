"""Legacy Discord runtime entrypoint (removed)."""

from __future__ import annotations

import logging
import sys


def main() -> None:
    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("nexus")
    log.error("The legacy `python -m nexus` runtime has been removed.")
    log.error("Use the app-first stack (`nexus_api` + frontend) instead.")
    log.error("For remote approvals/status, use `python -m nexus_discord_bridge`.")
    sys.exit(1)


if __name__ == "__main__":
    main()
