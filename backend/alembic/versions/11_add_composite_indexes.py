"""Add composite indexes for conversations and queries

Revision ID: 11
Revises: 10
Create Date: 2026-03-21
"""
from typing import Sequence, Union
from alembic import op

revision: str = "11"
down_revision: Union[str, None] = "10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("idx_conv_matter_active", "conversations", ["matter_id", "is_deleted", "updated_at"])
    op.create_index("idx_query_conversation_created", "queries", ["conversation_id", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_query_conversation_created", table_name="queries")
    op.drop_index("idx_conv_matter_active", table_name="conversations")
