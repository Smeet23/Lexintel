"""Add document_id index on citation_nodes and ON DELETE SET NULL to matter/document FKs

Revision ID: 17
Revises: 16
Create Date: 2026-06-02

Changes:
- citation_nodes(document_id): add idx_citation_node_document_id index (missing FK index).
- citation_nodes.matter_id FK: rebuild with ON DELETE SET NULL (column is nullable).
- citation_nodes.document_id FK: rebuild with ON DELETE SET NULL (column is nullable).
- citation_edges.matter_id FK: rebuild with ON DELETE SET NULL (column is nullable).
- amendment_chains.matter_id FK: DROP NOT NULL then rebuild with ON DELETE SET NULL.
  amendment_chains.matter_id was created NOT NULL in migration 12; SET NULL requires
  the column to be nullable, so we alter it to nullable before adding the FK.
- documents.document_status: correct server_default to 'unknown' (was 'current' in
  migration 12 after backfill; applied here as a safety net for already-migrated DBs).

The FK constraint names are the PostgreSQL auto-generated names produced when the
columns were created without an explicit constraint name in migrations 12 and 13:
  {tablename}_{colname}_fkey
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

revision: str = "17"
down_revision: Union[str, None] = "16"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# ---------------------------------------------------------------------------
# Auto-generated FK constraint names (PostgreSQL convention: table_col_fkey).
# These were created inline via sa.ForeignKey() without an explicit name, so
# Postgres assigned these names automatically.
# ---------------------------------------------------------------------------
_CN_MATTER_FK = "citation_nodes_matter_id_fkey"
_CN_DOCUMENT_FK = "citation_nodes_document_id_fkey"
_CE_MATTER_FK = "citation_edges_matter_id_fkey"
_AC_MATTER_FK = "amendment_chains_matter_id_fkey"


def upgrade() -> None:
    # 1. Add missing index on citation_nodes.document_id (FK column, always index).
    op.create_index("idx_citation_node_document_id", "citation_nodes", ["document_id"])

    # 2. citation_nodes.matter_id → matters.id with ON DELETE SET NULL.
    op.drop_constraint(_CN_MATTER_FK, "citation_nodes", type_="foreignkey")
    op.create_foreign_key(
        _CN_MATTER_FK,
        "citation_nodes",
        "matters",
        ["matter_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 3. citation_nodes.document_id → documents.id with ON DELETE SET NULL.
    op.drop_constraint(_CN_DOCUMENT_FK, "citation_nodes", type_="foreignkey")
    op.create_foreign_key(
        _CN_DOCUMENT_FK,
        "citation_nodes",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 4. citation_edges.matter_id → matters.id with ON DELETE SET NULL.
    op.drop_constraint(_CE_MATTER_FK, "citation_edges", type_="foreignkey")
    op.create_foreign_key(
        _CE_MATTER_FK,
        "citation_edges",
        "matters",
        ["matter_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 5. amendment_chains.matter_id → matters.id with ON DELETE SET NULL.
    #    This column was created NOT NULL in migration 12. SET NULL on a NOT NULL
    #    column causes every matter delete to fail with a constraint violation, so we
    #    must drop the NOT NULL constraint before adding the SET NULL FK.
    op.alter_column(
        "amendment_chains",
        "matter_id",
        existing_type=PG_UUID(as_uuid=True),
        nullable=True,
    )
    op.drop_constraint(_AC_MATTER_FK, "amendment_chains", type_="foreignkey")
    op.create_foreign_key(
        _AC_MATTER_FK,
        "amendment_chains",
        "matters",
        ["matter_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # 6. Safety net: correct documents.document_status server_default to 'unknown'.
    #    Migration 12 created the column with server_default='current' then backfilled
    #    existing rows to 'unknown' but never altered the column default, so raw inserts
    #    omitting the column would receive 'current' (an invalid status value). Applying
    #    this here ensures already-migrated DBs are also corrected.
    op.execute("ALTER TABLE documents ALTER COLUMN document_status SET DEFAULT 'unknown'")


def downgrade() -> None:
    # Reverse order: restore plain FKs (no ondelete action) then drop the index.

    # 5. amendment_chains.matter_id — restore plain FK and reinstate NOT NULL.
    #    upgrade() dropped NOT NULL to allow SET NULL; downgrade() puts it back.
    op.drop_constraint(_AC_MATTER_FK, "amendment_chains", type_="foreignkey")
    op.create_foreign_key(
        _AC_MATTER_FK,
        "amendment_chains",
        "matters",
        ["matter_id"],
        ["id"],
    )
    op.alter_column(
        "amendment_chains",
        "matter_id",
        existing_type=PG_UUID(as_uuid=True),
        nullable=False,
    )

    # 4. citation_edges.matter_id — restore plain FK.
    op.drop_constraint(_CE_MATTER_FK, "citation_edges", type_="foreignkey")
    op.create_foreign_key(
        _CE_MATTER_FK,
        "citation_edges",
        "matters",
        ["matter_id"],
        ["id"],
    )

    # 3. citation_nodes.document_id — restore plain FK.
    op.drop_constraint(_CN_DOCUMENT_FK, "citation_nodes", type_="foreignkey")
    op.create_foreign_key(
        _CN_DOCUMENT_FK,
        "citation_nodes",
        "documents",
        ["document_id"],
        ["id"],
    )

    # 2. citation_nodes.matter_id — restore plain FK.
    op.drop_constraint(_CN_MATTER_FK, "citation_nodes", type_="foreignkey")
    op.create_foreign_key(
        _CN_MATTER_FK,
        "citation_nodes",
        "matters",
        ["matter_id"],
        ["id"],
    )

    # 1. Drop the document_id index added in upgrade.
    op.drop_index("idx_citation_node_document_id", table_name="citation_nodes")
