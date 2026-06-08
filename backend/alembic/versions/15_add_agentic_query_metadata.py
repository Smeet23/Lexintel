"""Add agentic RAG observability columns to queries

Revision ID: 15
Revises: 14
Create Date: 2026-05-30

Adds nullable observability columns to the queries table, populated only when
the agentic RAG pipeline (settings.agentic_rag_enabled) handles a query.
Non-agentic queries leave all of these NULL, so the migration is purely
additive and backwards-compatible.

Columns:
- queries.routing_decision (String(50)) — router complexity / fast-path label.
- queries.agent_iterations (Integer) — rewrite_count + 1 (graph passes).
- queries.agent_latency_ms (Integer) — total agentic wall-clock in ms.
- queries.agentic_metadata (JSON) — full agentic_metadata blob (sub-questions,
  strategy log, sufficiency signals, node timings).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "15"
down_revision: Union[str, None] = "14"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("queries", sa.Column("routing_decision", sa.String(50), nullable=True))
    op.add_column("queries", sa.Column("agent_iterations", sa.Integer(), nullable=True))
    op.add_column("queries", sa.Column("agent_latency_ms", sa.Integer(), nullable=True))
    op.add_column("queries", sa.Column("agentic_metadata", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("queries", "agentic_metadata")
    op.drop_column("queries", "agent_latency_ms")
    op.drop_column("queries", "agent_iterations")
    op.drop_column("queries", "routing_decision")
