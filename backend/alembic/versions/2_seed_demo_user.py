"""Seed demo user for testing

Revision ID: seed_demo_user
Revises: f794e7d74f24
Create Date: 2026-01-26 11:35:00.000000

"""
from typing import Sequence, Union
from datetime import datetime

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'seed_demo_user'
down_revision: Union[str, None] = 'f794e7d74f24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Insert demo user for testing and development
    # Password: demo123 (hashed with bcrypt)
    # This user is used for local development only
    op.execute(
        sa.text("""
            INSERT INTO users (id, email, password_hash, is_deleted, created_at, updated_at)
            VALUES ('00000000-0000-0000-0000-000000000001', 'demo@lexintel.dev', '$2b$12$R9h/cIPz0gi.URNNX3kh2OPST9/PgBkqquzi.Ss7KIUgO2t0jWMUe', false, NOW(), NOW())
            ON CONFLICT DO NOTHING;
        """)
    )


def downgrade() -> None:
    # Remove demo user
    op.execute(
        sa.text("""
            DELETE FROM users WHERE id = '00000000-0000-0000-0000-000000000001';
        """)
    )
