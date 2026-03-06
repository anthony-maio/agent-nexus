"""Create app control-plane tables.

Revision ID: 202603060001
Revises:
Create Date: 2026-03-06 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "202603060001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "admin_users",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("username", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
    )

    op.create_table(
        "sessions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("token", sa.String(length=96), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.ForeignKeyConstraint(["user_id"], ["admin_users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token"),
    )
    op.create_index(op.f("ix_sessions_token"), "sessions", ["token"], unique=True)

    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "run_steps",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("instruction", sa.Text(), nullable=False),
        sa.Column("risk_tier", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("error_text", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_run_steps_run_id"), "run_steps", ["run_id"], unique=False)

    op.create_table(
        "approvals",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("step_id", sa.String(length=32), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("decided_by", sa.String(length=120), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.ForeignKeyConstraint(["step_id"], ["run_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_approvals_run_id"), "approvals", ["run_id"], unique=False)
    op.create_index(op.f("ix_approvals_step_id"), "approvals", ["step_id"], unique=False)

    op.create_table(
        "citations",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("step_id", sa.String(length=32), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("snippet", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.ForeignKeyConstraint(["step_id"], ["run_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_citations_run_id"), "citations", ["run_id"], unique=False)
    op.create_index(op.f("ix_citations_step_id"), "citations", ["step_id"], unique=False)

    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("step_id", sa.String(length=32), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False, server_default="text"),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("rel_path", sa.Text(), nullable=False),
        sa.Column("sandbox_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("promoted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.ForeignKeyConstraint(["step_id"], ["run_steps.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_artifacts_run_id"), "artifacts", ["run_id"], unique=False)

    op.create_table(
        "promotions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("artifact_id", sa.String(length=32), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("target_path", sa.Text(), nullable=False),
        sa.Column("promoted_by", sa.String(length=120), nullable=False),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["artifact_id"], ["artifacts.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_promotions_artifact_id"), "promotions", ["artifact_id"], unique=False)
    op.create_index(op.f("ix_promotions_run_id"), "promotions", ["run_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_promotions_run_id"), table_name="promotions")
    op.drop_index(op.f("ix_promotions_artifact_id"), table_name="promotions")
    op.drop_table("promotions")

    op.drop_index(op.f("ix_artifacts_run_id"), table_name="artifacts")
    op.drop_table("artifacts")

    op.drop_index(op.f("ix_citations_step_id"), table_name="citations")
    op.drop_index(op.f("ix_citations_run_id"), table_name="citations")
    op.drop_table("citations")

    op.drop_index(op.f("ix_approvals_step_id"), table_name="approvals")
    op.drop_index(op.f("ix_approvals_run_id"), table_name="approvals")
    op.drop_table("approvals")

    op.drop_index(op.f("ix_run_steps_run_id"), table_name="run_steps")
    op.drop_table("run_steps")

    op.drop_table("runs")

    op.drop_index(op.f("ix_sessions_token"), table_name="sessions")
    op.drop_table("sessions")

    op.drop_table("admin_users")
