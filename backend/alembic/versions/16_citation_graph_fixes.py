"""Fix citation graph numeric columns and add edge_kind

Revision ID: 16_citation_graph_fixes
Revises: 15
Create Date: 2026-05-30

Converts citation graph numeric columns from text to float to fix
lexicographic sorting bugs, and adds an edge_kind discriminator column to
citation_edges:

- citation_nodes.authority_score: Text -> Float (empty strings -> NULL).
- citation_edges.confidence: Text -> Float (empty strings -> NULL).
- citation_edges.edge_kind (String(20), NOT NULL, default 'document_cites'):
  'document_cites' (doc->case provenance) | 'case_cites' (real case->case).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "16"
down_revision: Union[str, None] = "15"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # MAINTENANCE WINDOW NOTE: ALTER COLUMN ... TYPE acquires an AccessExclusiveLock
    # on citation_nodes and citation_edges for the duration of the type conversion.
    # On small-to-medium tables this lock is brief (< 1 s), but run during a
    # low-traffic window to be safe.  lock_timeout aborts rather than queueing
    # behind long-running queries.
    op.execute("SET lock_timeout = '5s'")

    # The old text columns may carry a text default (e.g. '0.0') that Postgres
    # cannot auto-cast to double precision during the type change, so drop the
    # default first, convert the type, then re-apply a numeric default.
    op.execute("ALTER TABLE citation_nodes ALTER COLUMN authority_score DROP DEFAULT")
    op.alter_column(
        "citation_nodes",
        "authority_score",
        type_=sa.Float(),
        postgresql_using="NULLIF(authority_score, '')::double precision",
        existing_nullable=True,
    )
    op.execute("ALTER TABLE citation_nodes ALTER COLUMN authority_score SET DEFAULT 0.0")

    op.execute("ALTER TABLE citation_edges ALTER COLUMN confidence DROP DEFAULT")
    op.alter_column(
        "citation_edges",
        "confidence",
        type_=sa.Float(),
        postgresql_using="NULLIF(confidence, '')::double precision",
        existing_nullable=True,
    )
    op.add_column(
        "citation_edges",
        sa.Column(
            "edge_kind",
            sa.String(length=20),
            nullable=False,
            server_default="document_cites",
        ),
    )
    op.create_index("idx_citation_edge_kind", "citation_edges", ["edge_kind"])


def downgrade() -> None:
    op.drop_index("idx_citation_edge_kind", table_name="citation_edges")
    op.drop_column("citation_edges", "edge_kind")
    op.alter_column(
        "citation_edges",
        "confidence",
        type_=sa.Text(),
        postgresql_using="confidence::text",
        existing_nullable=True,
    )
    op.alter_column(
        "citation_nodes",
        "authority_score",
        type_=sa.Text(),
        postgresql_using="authority_score::text",
        existing_nullable=True,
    )
    # Restore the server_default that migration 13 originally set on this column.
    op.execute("ALTER TABLE citation_nodes ALTER COLUMN authority_score SET DEFAULT '0.0'")
