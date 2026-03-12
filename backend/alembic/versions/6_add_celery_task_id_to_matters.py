"""Add celery_task_id to matters for cancel processing

Revision ID: 6
Revises: 4
Create Date: 2026-02-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "6"
down_revision: Union[str, None] = "4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "matters",
        sa.Column("celery_task_id", sa.String(255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("matters", "celery_task_id")
