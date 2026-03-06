"""Run Nexus API with `python -m nexus_api`."""

from __future__ import annotations

import uvicorn

from nexus_api.config import ApiSettings


def main() -> None:
    settings = ApiSettings()
    uvicorn.run(
        "nexus_api.app:create_app",
        factory=True,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
