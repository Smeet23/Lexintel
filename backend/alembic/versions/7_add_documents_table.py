"""Add documents table and document_id FK on chunks

Revision ID: 7
Revises: 6
Create Date: 2026-02-27

This migration handles existing data by:
1. Creating the documents table
2. Adding document_id as NULLABLE on chunks
3. Creating a synthetic Document for each matter that has existing chunks
4. Backfilling chunks.document_id from those synthetic documents
5. Setting document_id to NOT NULL after backfill

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "7"
down_revision: Union[str, None] = "6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create documents table
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("matter_id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("blob_storage_path", sa.String(length=500), nullable=False),
        sa.Column("file_type", sa.String(length=10), nullable=False, server_default="pdf"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="processing"),
        sa.Column("celery_task_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["matter_id"], ["matters.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_documents_matter_id"), "documents", ["matter_id"], unique=False)
    op.create_index(op.f("ix_documents_status"), "documents", ["status"], unique=False)

    # 2. Add document_id as NULLABLE first (safe for existing rows)
    op.add_column("chunks", sa.Column("document_id", sa.UUID(), nullable=True))

    # 3. Backfill: create a synthetic Document for each matter that has chunks,
    #    then assign those chunks to the new document
    conn = op.get_bind()
    matters_with_chunks = conn.execute(sa.text(
        "SELECT DISTINCT c.matter_id FROM chunks c WHERE c.document_id IS NULL"
    )).fetchall()

    for (matter_id,) in matters_with_chunks:
        # Read matter metadata for the synthetic document
        row = conn.execute(sa.text(
            "SELECT name, blob_storage_path, file_type, status, created_at, updated_at "
            "FROM matters WHERE id = :mid"
        ), {"mid": matter_id}).fetchone()
        if not row:
            continue

        # Create a synthetic document record
        doc_id = conn.execute(sa.text(
            "INSERT INTO documents (id, matter_id, name, blob_storage_path, file_type, status, created_at, updated_at) "
            "VALUES (gen_random_uuid(), :mid, :name, :path, :ft, :status, :created, :updated) "
            "RETURNING id"
        ), {
            "mid": matter_id,
            "name": row[0],
            "path": row[1],
            "ft": row[2],
            "status": row[3],
            "created": row[4],
            "updated": row[5],
        }).fetchone()

        if doc_id:
            # Assign all orphaned chunks for this matter to the new document
            conn.execute(sa.text(
                "UPDATE chunks SET document_id = :did WHERE matter_id = :mid AND document_id IS NULL"
            ), {"did": doc_id[0], "mid": matter_id})

    # 4. Now enforce NOT NULL after all existing rows are backfilled
    op.alter_column("chunks", "document_id", nullable=False)

    # 5. Add FK constraint and index
    op.create_foreign_key(
        "fk_chunks_document_id",
        "chunks",
        "documents",
        ["document_id"],
        ["id"],
    )
    op.create_index(op.f("ix_chunks_document_id"), "chunks", ["document_id"], unique=False)


def downgrade() -> None:
    # WARNING: Irreversible data loss — all document records and chunk->document
    # associations will be permanently deleted. Take a DB snapshot before running.
    op.drop_index(op.f("ix_chunks_document_id"), table_name="chunks")
    op.drop_constraint("fk_chunks_document_id", "chunks", type_="foreignkey")
    op.drop_column("chunks", "document_id")
    op.drop_index(op.f("ix_documents_status"), table_name="documents")
    op.drop_index(op.f("ix_documents_matter_id"), table_name="documents")
    op.drop_table("documents")
