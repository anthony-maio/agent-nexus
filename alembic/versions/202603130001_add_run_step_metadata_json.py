"""Add persistent step metadata snapshots.

Revision ID: 202603130001
Revises: 202603080002
Create Date: 2026-03-13 10:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "202603130001"
down_revision = "202603080002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("run_steps") as batch_op:
        batch_op.add_column(
            sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}")
        )


def downgrade() -> None:
    with op.batch_alter_table("run_steps") as batch_op:
        batch_op.drop_column("metadata_json")
