import os
import shutil
from pathlib import Path
from typing import Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class StorageService:
    """Handle file uploads and storage"""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, document_id: str, filename: str, file_content: bytes) -> str:
        """
        Save uploaded file to disk
        Returns: file path
        """
        try:
            # Create document directory
            doc_dir = self.upload_dir / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Save file
            file_path = doc_dir / filename
            file_path.write_bytes(file_content)

            logger.info(f"[storage] Saved file: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"[storage] Failed to save file: {e}")
            raise

    def get_file(self, file_path: str) -> bytes:
        """Read file from disk"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            return path.read_bytes()
        except Exception as e:
            logger.error(f"[storage] Failed to read file: {e}")
            raise

    def delete_file(self, file_path: str) -> None:
        """Delete file from disk"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"[storage] Deleted file: {file_path}")
        except Exception as e:
            logger.error(f"[storage] Failed to delete file: {e}")
            raise

    def delete_document_files(self, document_id: str) -> None:
        """Delete all files for a document"""
        try:
            doc_dir = self.upload_dir / document_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
                logger.info(f"[storage] Deleted document directory: {doc_dir}")
        except Exception as e:
            logger.error(f"[storage] Failed to delete document directory: {e}")
            raise

    def validate_file(self, filename: str, file_size: int) -> bool:
        """Validate file before upload"""
        # Check file extension
        allowed = settings.ALLOWED_EXTENSIONS.split(",")
        ext = Path(filename).suffix.lower()
        if ext not in allowed:
            logger.warning(f"[storage] Invalid file extension: {ext}")
            return False

        # Check file size
        if file_size > settings.MAX_UPLOAD_SIZE:
            logger.warning(f"[storage] File too large: {file_size} bytes")
            return False

        return True

storage_service = StorageService()
