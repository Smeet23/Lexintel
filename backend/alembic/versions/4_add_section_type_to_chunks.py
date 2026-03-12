"""Add section_type column to chunks table

Revision ID: 4
Revises: f794e7d74f24
Create Date: 2026-02-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4'
down_revision: Union[str, None] = 'f794e7d74f24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('chunks', sa.Column('section_type', sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column('chunks', 'section_type')
