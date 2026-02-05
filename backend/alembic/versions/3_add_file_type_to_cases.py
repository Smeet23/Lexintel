"""Add file_type column to cases table

Revision ID: 3
Revises: 2_seed_demo_user
Create Date: 2026-01-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3'
down_revision: Union[str, None] = 'seed_demo_user'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add file_type column to cases table with default value 'pdf'
    op.add_column('cases', sa.Column('file_type', sa.String(10), nullable=False, server_default='pdf'))


def downgrade() -> None:
    # Remove file_type column
    op.drop_column('cases', 'file_type')
