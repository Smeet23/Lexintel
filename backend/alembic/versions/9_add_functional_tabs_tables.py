"""Add functional tabs tables: contract_reviews, drafts, audit_logs, saved_precedents

Revision ID: 9
Revises: 8
Create Date: 2026-03-14

Adds:
- contract_reviews table — contract risk analysis results
- drafts table — AI-generated legal document drafts
- audit_logs table — activity audit trail for matters
- saved_precedents table — user-saved precedent research results
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "9"
down_revision: Union[str, None] = "8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # contract_reviews
    op.create_table(
        "contract_reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False, index=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id"), nullable=False, index=True),
        sa.Column("risks", sa.JSON, nullable=True),
        sa.Column("summary", sa.JSON, nullable=True),
        sa.Column("missing_clauses", sa.JSON, nullable=True),
        sa.Column("overall_score", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # drafts
    op.create_table(
        "drafts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False, index=True),
        sa.Column("document_type", sa.String(100), nullable=False),
        sa.Column("instructions", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("sources", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    # audit_logs
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False, index=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("user", sa.String(255), nullable=False, server_default="System"),
        sa.Column("details", sa.Text, nullable=True),
        sa.Column("sources", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now(), index=True),
    )
    op.create_index("idx_audit_matter_created", "audit_logs", ["matter_id", "created_at"])

    # saved_precedents
    op.create_table(
        "saved_precedents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("document_name", sa.String(255), nullable=True),
        sa.Column("matter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("chunk_content", sa.Text, nullable=True),
        sa.Column("page_num", sa.Integer, nullable=True),
        sa.Column("section_name", sa.String(255), nullable=True),
        sa.Column("relevance_score", sa.String(10), nullable=True),
        sa.Column("tags", sa.JSON, nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("saved_precedents")
    op.drop_index("idx_audit_matter_created", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_table("drafts")
    op.drop_table("contract_reviews")
