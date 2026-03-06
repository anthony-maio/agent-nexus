"""Programmatic Alembic migration runner for nexus_api."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config

from alembic import command


def run_migrations(database_url: str) -> None:
    """Upgrade control-plane schema to the latest Alembic head."""
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")
