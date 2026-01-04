#!/usr/bin/env python3
"""
Database migration: Rename file_path column to blob_url

This script safely migrates the documents table from storing local file paths
to storing Azure Blob Storage presigned URLs (blob_url).

Usage:
    python migrate_file_path_to_blob_url.py
"""

import asyncio
import logging
from sqlalchemy import text
from app.database import engine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_column_exists(conn, table: str, column: str) -> bool:
    """Check if a column exists in a table"""
    result = await conn.execute(
        text("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = :table AND column_name = :column
            )
        """),
        {"table": table, "column": column}
    )
    return result.scalar()


async def migrate():
    """Run the migration"""
    logger.info("Starting migration: file_path → blob_url")

    try:
        async with engine.begin() as conn:
            # Check if file_path exists
            has_file_path = await check_column_exists(conn, "documents", "file_path")
            has_blob_url = await check_column_exists(conn, "documents", "blob_url")

            if has_blob_url and not has_file_path:
                logger.info("✅ Migration already applied! blob_url column exists.")
                return

            if not has_file_path:
                logger.error("❌ file_path column not found. Nothing to migrate.")
                return

            logger.info("Found file_path column. Starting migration...")

            # Step 1: Rename column
            logger.info("Step 1/2: Renaming file_path → blob_url...")
            await conn.execute(
                text("ALTER TABLE documents RENAME COLUMN file_path TO blob_url;")
            )
            logger.info("✅ Column renamed successfully")

            # Step 2: Verify
            logger.info("Step 2/2: Verifying migration...")
            result = await conn.execute(
                text("SELECT id, blob_url FROM documents LIMIT 1;")
            )
            row = result.first()

            if row:
                logger.info(f"✅ Sample record: id={row[0]}, blob_url={row[1][:50]}...")
            else:
                logger.info("✅ No documents yet (table is empty)")

            logger.info("=" * 60)
            logger.info("✅ MIGRATION COMPLETE!")
            logger.info("=" * 60)
            logger.info("\nChanges:")
            logger.info("  • documents.file_path → documents.blob_url")
            logger.info("\nWhat this means:")
            logger.info("  • Documents now store presigned URLs instead of local paths")
            logger.info("  • Files are stored in Azure Blob Storage")
            logger.info("  • Presigned URLs expire after 24 hours")
            logger.info("  • Workers automatically download from blob URLs")
            logger.info("\nNext steps:")
            logger.info("  1. Rebuild services: just build")
            logger.info("  2. Start services: just up")
            logger.info("  3. Test extraction: Upload a document and check logs")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(migrate())
