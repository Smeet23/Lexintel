"""Add temporal awareness columns and amendment_chains table

Revision ID: 12
Revises: 11
Create Date: 2026-03-23
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "12"
down_revision: Union[str, None] = "11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create amendment_chains table first (documents will FK to it)
    op.create_table(
        "amendment_chains",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("canonical_document_id", sa.String(255), nullable=False),
        sa.Column("canonical_name", sa.Text, nullable=False),
        sa.Column("jurisdiction", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "idx_chain_matter_canonical",
        "amendment_chains",
        ["matter_id", "canonical_document_id"],
    )

    # 2. Add temporal columns to documents
    op.add_column("documents", sa.Column("effective_date", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("superseded_date", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("version_number", sa.String(20), nullable=True))
    op.add_column(
        "documents",
        sa.Column("document_status", sa.String(50), server_default="current", nullable=False),
    )
    op.add_column(
        "documents",
        sa.Column(
            "amendment_chain_id",
            UUID(as_uuid=True),
            sa.ForeignKey("amendment_chains.id"),
            nullable=True,
        ),
    )

    # 3. Indexes on new columns
    op.create_index("idx_doc_effective_date", "documents", ["effective_date"])
    op.create_index("idx_doc_superseded_date", "documents", ["superseded_date"])
    op.create_index("idx_doc_status", "documents", ["document_status"])
    op.create_index("idx_doc_amendment_chain", "documents", ["amendment_chain_id"])

    # 4. Backfill: set all existing documents to status="unknown"
    #    (they have no temporal metadata yet)
    op.execute("UPDATE documents SET document_status = 'unknown' WHERE document_status = 'current'")

    # 5. Correct the server_default to 'unknown' so raw inserts omitting the column
    #    also receive 'unknown' rather than the stale 'current' default set above.
    op.alter_column("documents", "document_status", server_default="unknown")


def downgrade() -> None:
    op.drop_index("idx_doc_amendment_chain", table_name="documents")
    op.drop_index("idx_doc_status", table_name="documents")
    op.drop_index("idx_doc_superseded_date", table_name="documents")
    op.drop_index("idx_doc_effective_date", table_name="documents")

    op.drop_column("documents", "amendment_chain_id")
    op.drop_column("documents", "document_status")
    op.drop_column("documents", "version_number")
    op.drop_column("documents", "superseded_date")
    op.drop_column("documents", "effective_date")

    op.drop_index("idx_chain_matter_canonical", table_name="amendment_chains")
    op.drop_table("amendment_chains")
