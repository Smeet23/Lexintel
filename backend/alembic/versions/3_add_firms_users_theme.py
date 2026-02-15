"""Add firms table, user columns (name, role, firm_id), and firm_id to matters

Revision ID: 3_add_firms_users_theme
Revises: seed_demo_user
Create Date: 2026-02-15
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = '3_add_firms_users_theme'
down_revision = 'seed_demo_user'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create firms table with JSONB (not JSON)
    op.create_table(
        'firms',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(255), nullable=False, unique=True),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('theme_config', JSONB, nullable=True, server_default=sa.text("'{}'::jsonb")),
        sa.Column('is_deleted', sa.Boolean, default=False, nullable=False),
        sa.Column('created_at', sa.DateTime, nullable=False),
        sa.Column('updated_at', sa.DateTime, nullable=False),
    )
    op.create_index('ix_firms_slug', 'firms', ['slug'], unique=True)

    # ADD columns to existing users table (NOT create — table exists from f794e7d74f24)
    op.add_column('users', sa.Column('name', sa.String(255), nullable=True))
    op.add_column('users', sa.Column('role', sa.String(50), server_default='associate', nullable=False))
    op.add_column('users', sa.Column('firm_id', UUID(as_uuid=True), sa.ForeignKey('firms.id'), nullable=True))
    op.create_index('ix_users_firm_id', 'users', ['firm_id'])

    # Add firm_id to matters table
    op.add_column('matters', sa.Column('firm_id', UUID(as_uuid=True), sa.ForeignKey('firms.id'), nullable=True))
    op.create_index('ix_matters_firm_id', 'matters', ['firm_id'])


def downgrade() -> None:
    op.drop_index('ix_matters_firm_id', 'matters')
    op.drop_column('matters', 'firm_id')
    op.drop_index('ix_users_firm_id', 'users')
    op.drop_column('users', 'firm_id')
    op.drop_column('users', 'role')
    op.drop_column('users', 'name')
    op.drop_table('firms')
