"""Initial schema: users, matters, chunks, queries, processing_jobs

Revision ID: f794e7d74f24
Revises:
Create Date: 2026-01-25 15:17:15.079379

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f794e7d74f24'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table('users',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('password_hash', sa.String(length=255), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_is_deleted'), 'users', ['is_deleted'], unique=False)

    # Matters table (legal matters / cases)
    op.create_table('matters',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('blob_storage_path', sa.String(length=500), nullable=False),
        sa.Column('file_type', sa.String(length=10), nullable=False, server_default='pdf'),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('is_deleted', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_matters_created_at'), 'matters', ['created_at'], unique=False)
    op.create_index(op.f('ix_matters_is_deleted'), 'matters', ['is_deleted'], unique=False)
    op.create_index(op.f('ix_matters_status'), 'matters', ['status'], unique=False)

    # Chunks table
    op.create_table('chunks',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('matter_id', sa.UUID(), nullable=False),
        sa.Column('page_num', sa.String(length=50), nullable=True),
        sa.Column('section_name', sa.String(length=255), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding_hash', sa.String(length=255), nullable=True),
        sa.Column('chunk_sequence', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['matter_id'], ['matters.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_matter_id_sequence', 'chunks', ['matter_id', 'chunk_sequence'], unique=False)
    op.create_index(op.f('ix_chunks_matter_id'), 'chunks', ['matter_id'], unique=False)

    # Queries table
    op.create_table('queries',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('matter_id', sa.UUID(), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('answer', sa.Text(), nullable=False),
        sa.Column('citations', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['matter_id'], ['matters.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_matter_id_created', 'queries', ['matter_id', 'created_at'], unique=False)
    op.create_index(op.f('ix_queries_matter_id'), 'queries', ['matter_id'], unique=False)
    op.create_index(op.f('ix_queries_created_at'), 'queries', ['created_at'], unique=False)

    # Processing jobs table
    op.create_table('processing_jobs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('matter_id', sa.UUID(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('error_message', sa.String(length=500), nullable=True),
        sa.Column('attempts', sa.Integer(), nullable=True),
        sa.Column('max_attempts', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('next_retry_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['matter_id'], ['matters.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_processing_jobs_matter_id'), 'processing_jobs', ['matter_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_processing_jobs_matter_id'), table_name='processing_jobs')
    op.drop_table('processing_jobs')
    op.drop_index(op.f('ix_queries_created_at'), table_name='queries')
    op.drop_index(op.f('ix_queries_matter_id'), table_name='queries')
    op.drop_index('idx_matter_id_created', table_name='queries')
    op.drop_table('queries')
    op.drop_index(op.f('ix_chunks_matter_id'), table_name='chunks')
    op.drop_index('idx_matter_id_sequence', table_name='chunks')
    op.drop_table('chunks')
    op.drop_index(op.f('ix_matters_status'), table_name='matters')
    op.drop_index(op.f('ix_matters_is_deleted'), table_name='matters')
    op.drop_index(op.f('ix_matters_created_at'), table_name='matters')
    op.drop_table('matters')
    op.drop_index(op.f('ix_users_is_deleted'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
