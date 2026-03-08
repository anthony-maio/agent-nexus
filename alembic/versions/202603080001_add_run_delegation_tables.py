"""Add parent-child run delegation tables.

Revision ID: 202603080001
Revises: 202603060001
Create Date: 2026-03-08 00:01:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "202603080001"
down_revision = "202603060001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(sa.Column("parent_run_id", sa.String(length=32), nullable=True))
        batch_op.create_index(op.f("ix_runs_parent_run_id"), ["parent_run_id"], unique=False)
        batch_op.create_foreign_key(
            "fk_runs_parent_run_id_runs",
            "runs",
            ["parent_run_id"],
            ["id"],
        )

    op.create_table(
        "run_delegations",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("parent_run_id", sa.String(length=32), nullable=False),
        sa.Column("child_run_id", sa.String(length=32), nullable=False),
        sa.Column("role", sa.String(length=64), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["parent_run_id"], ["runs.id"]),
        sa.ForeignKeyConstraint(["child_run_id"], ["runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_run_delegations_parent_run_id"),
        "run_delegations",
        ["parent_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_run_delegations_child_run_id"),
        "run_delegations",
        ["child_run_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_run_delegations_child_run_id"), table_name="run_delegations")
    op.drop_index(op.f("ix_run_delegations_parent_run_id"), table_name="run_delegations")
    op.drop_table("run_delegations")

    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_constraint("fk_runs_parent_run_id_runs", type_="foreignkey")
        batch_op.drop_index(op.f("ix_runs_parent_run_id"))
        batch_op.drop_column("parent_run_id")
