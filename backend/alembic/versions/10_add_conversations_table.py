"""Add conversations table and conversation_id to queries

Revision ID: 10
Revises: 9
Create Date: 2026-03-21

Adds:
- conversations table — conversation threads per matter
- conversation_id column on queries (nullable, for backwards compat)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "10"
down_revision: Union[str, None] = "9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # conversations table
    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False, index=True),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("is_deleted", sa.Boolean, nullable=False, server_default=sa.false()),
    )

    # Add conversation_id to queries (nullable so existing rows are unaffected)
    op.add_column(
        "queries",
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id"),
            nullable=True,
            index=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("queries", "conversation_id")
    op.drop_table("conversations")
