"""Run sandbox runner with `python -m nexus_sandbox_runner`."""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("nexus_sandbox_runner.app:create_app", factory=True, host="0.0.0.0", port=8020)


if __name__ == "__main__":
    main()
