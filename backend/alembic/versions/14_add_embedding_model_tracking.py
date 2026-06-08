"""Add embedding_model tracking to matters and chunks

Revision ID: 14
Revises: 13
Create Date: 2026-05-30

Adds:
- matters.embedding_model (String(64), nullable, indexed) — authoritative
  per-matter vector space; drives admin status + all-matters reindex filter.
- chunks.embedding_model (String(64), nullable) — per-chunk provenance for
  partial-failure resume during a provider re-index.

Backfill: existing rows are stamped with 'embed-english-v3.0'. Voyage was only
just wired in, so no Voyage vectors predate this migration — every pre-existing
matter/chunk was embedded with Cohere embed-english-v3.0.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "14"
down_revision: Union[str, None] = "13"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Pre-Voyage vectors were all produced by Cohere embed-english-v3.0.
_LEGACY_MODEL = "embed-english-v3.0"


def upgrade() -> None:
    # 1. Matter: authoritative per-matter vector space.
    op.add_column("matters", sa.Column("embedding_model", sa.String(64), nullable=True))
    op.create_index("idx_matters_embedding_model", "matters", ["embedding_model"])

    # 2. Chunk: per-chunk provenance for resumable re-index.
    op.add_column("chunks", sa.Column("embedding_model", sa.String(64), nullable=True))

    # 3. Backfill existing rows — all pre-existing vectors are Cohere.
    op.execute(
        sa.text(
            "UPDATE matters SET embedding_model = :model WHERE embedding_model IS NULL"
        ).bindparams(model=_LEGACY_MODEL)
    )
    op.execute(
        sa.text(
            "UPDATE chunks SET embedding_model = :model WHERE embedding_model IS NULL"
        ).bindparams(model=_LEGACY_MODEL)
    )


def downgrade() -> None:
    op.drop_index("idx_matters_embedding_model", table_name="matters")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("matters", "embedding_model")
