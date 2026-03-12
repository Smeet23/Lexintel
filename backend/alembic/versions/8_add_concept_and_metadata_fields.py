"""Add concept extraction and document metadata fields

Revision ID: 8
Revises: 7
Create Date: 2026-03-12

Adds:
- chunks.concepts (JSON) — YAKE-extracted keywords per chunk
- documents.summary (Text) — Gemini-generated document summary
- documents.document_type (String) — statute/contract/judgment/etc.
- documents.jurisdiction (String) — US/UK/EU/etc.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8"
down_revision: Union[str, None] = "7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Chunk: add concepts JSON column
    op.add_column("chunks", sa.Column("concepts", sa.JSON, nullable=True))

    # Document: add summary, document_type, jurisdiction
    op.add_column("documents", sa.Column("summary", sa.Text, nullable=True))
    op.add_column("documents", sa.Column("document_type", sa.String(100), nullable=True))
    op.add_column("documents", sa.Column("jurisdiction", sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "concepts")
    op.drop_column("documents", "summary")
    op.drop_column("documents", "document_type")
    op.drop_column("documents", "jurisdiction")
