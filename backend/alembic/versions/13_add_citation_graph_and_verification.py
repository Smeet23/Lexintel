"""Add citation graph tables, chunk authority metadata, and query verification columns

Revision ID: 13
Revises: 12
Create Date: 2026-05-30

Adds:
- chunks.authority_metadata (JSON) — authority hierarchy data from authority_detector
- queries.citation_verification (JSON) — citation verification results
- queries.claim_verification (JSON) — claim verification (NLI) results
- queries.conflict_analysis (JSON) — conflict detection results
- citation_nodes table — unique legal citations in the knowledge graph
- citation_edges table — directed treatment relationships between citations
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "13"
down_revision: Union[str, None] = "12"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Chunk: authority metadata written during ingestion (authority_detector)
    op.add_column("chunks", sa.Column("authority_metadata", sa.JSON, nullable=True))

    # 2. Query: verification result columns written on every answered query
    op.add_column("queries", sa.Column("citation_verification", sa.JSON, nullable=True))
    op.add_column("queries", sa.Column("claim_verification", sa.JSON, nullable=True))
    op.add_column("queries", sa.Column("conflict_analysis", sa.JSON, nullable=True))

    # 3. Citation knowledge graph — nodes (unique citations)
    op.create_table(
        "citation_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("node_type", sa.String(20), nullable=False),
        sa.Column("citation_text", sa.String(500), nullable=False),
        sa.Column("name", sa.String(500), nullable=True),
        sa.Column("court", sa.String(200), nullable=True),
        sa.Column("year", sa.Integer, nullable=True),
        sa.Column("jurisdiction", sa.String(20), nullable=True),
        sa.Column("authority_score", sa.Text, nullable=True, server_default="0.0"),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id"),
            nullable=True,
        ),
        sa.Column(
            "matter_id",
            UUID(as_uuid=True),
            sa.ForeignKey("matters.id"),
            nullable=True,
        ),
        sa.Column("courtlistener_id", sa.String(100), nullable=True),
        sa.Column("source_url", sa.String(500), nullable=True),
        sa.Column("is_verified", sa.Boolean, server_default=sa.false(), nullable=False),
        sa.Column("is_good_law", sa.Boolean, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_citation_node_text", "citation_nodes", ["citation_text"])
    op.create_index(
        "idx_citation_node_type_jurisdiction",
        "citation_nodes",
        ["node_type", "jurisdiction"],
    )
    op.create_index("idx_citation_node_matter_id", "citation_nodes", ["matter_id"])
    op.create_index("idx_citation_node_year", "citation_nodes", ["year"])

    # 4. Citation knowledge graph — edges (directed treatment relationships)
    op.create_table(
        "citation_edges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_id",
            UUID(as_uuid=True),
            sa.ForeignKey("citation_nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_id",
            UUID(as_uuid=True),
            sa.ForeignKey("citation_nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("treatment", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Text, nullable=True),
        sa.Column("context_excerpt", sa.Text, nullable=True),
        sa.Column(
            "matter_id",
            UUID(as_uuid=True),
            sa.ForeignKey("matters.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_citation_edge_source", "citation_edges", ["source_id"])
    op.create_index("idx_citation_edge_target", "citation_edges", ["target_id"])
    op.create_index("idx_citation_edge_source_target", "citation_edges", ["source_id", "target_id"])
    op.create_index("idx_citation_edge_treatment", "citation_edges", ["treatment"])
    op.create_index("idx_citation_edge_matter_id", "citation_edges", ["matter_id"])


def downgrade() -> None:
    op.drop_index("idx_citation_edge_matter_id", table_name="citation_edges")
    op.drop_index("idx_citation_edge_treatment", table_name="citation_edges")
    op.drop_index("idx_citation_edge_source_target", table_name="citation_edges")
    op.drop_index("idx_citation_edge_target", table_name="citation_edges")
    op.drop_index("idx_citation_edge_source", table_name="citation_edges")
    op.drop_table("citation_edges")

    op.drop_index("idx_citation_node_year", table_name="citation_nodes")
    op.drop_index("idx_citation_node_matter_id", table_name="citation_nodes")
    op.drop_index("idx_citation_node_type_jurisdiction", table_name="citation_nodes")
    op.drop_index("idx_citation_node_text", table_name="citation_nodes")
    op.drop_table("citation_nodes")

    op.drop_column("queries", "conflict_analysis")
    op.drop_column("queries", "claim_verification")
    op.drop_column("queries", "citation_verification")

    op.drop_column("chunks", "authority_metadata")
