from datetime import datetime, timedelta, timezone
from app.config import settings
import logging
from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    generate_blob_sas,
)
from pathlib import Path

__all__ = ["StorageService", "storage_service"]

logger = logging.getLogger(__name__)


class StorageService:
    """Handle file uploads and storage using Azure Blob Storage with presigned URLs (SAS tokens)"""

    def __init__(self):
        """Initialize Azure Blob Storage client"""
        if not settings.AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING not configured. "
                "Please set it in your .env file."
            )

        self.blob_service_client = BlobServiceClient.from_connection_string(
            settings.AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME

        # Ensure container exists
        self._ensure_container_exists()

    def _ensure_container_exists(self) -> None:
        """Create container if it doesn't exist"""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            # Try to get properties to check if it exists
            container_client.get_container_properties()
            logger.info(f"[storage] Container '{self.container_name}' exists")
        except Exception:
            # Container doesn't exist, create it
            try:
                self.blob_service_client.create_container(self.container_name)
                logger.info(f"[storage] Created container '{self.container_name}'")
            except Exception as e:
                logger.error(f"[storage] Failed to create container: {e}")
                raise

    def _parse_connection_string(self) -> dict[str, str]:
        """Parse connection string to extract account name and key for SAS token generation"""
        connection_string = settings.AZURE_STORAGE_CONNECTION_STRING
        parts = connection_string.split(";")
        account_name = ""
        account_key = ""

        for part in parts:
            if part.startswith("AccountName="):
                account_name = part[len("AccountName=") :]
            elif part.startswith("AccountKey="):
                account_key = part[len("AccountKey=") :]

        if not account_name or not account_key:
            raise ValueError(
                "Invalid connection string: missing AccountName or AccountKey"
            )

        return {"account_name": account_name, "account_key": account_key}

    def _get_blob_url(self, blob_name: str) -> str:
        """Get the base URL for a blob"""
        container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.url

    def save_file(self, document_id: str, filename: str, file_content: bytes) -> str:
        """
        Save uploaded file to Azure Blob Storage
        Returns: Presigned URL (SAS URL) valid for 24 hours
        """
        try:
            # Create blob name: documents/{document_id}/{filename}
            blob_name = f"documents/{document_id}/{filename}"

            # Upload to Azure
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(file_content, overwrite=True)

            logger.info(f"[storage] Saved blob: {blob_name}")

            # Generate presigned URL (SAS token) with read permission
            sas_url = self.generate_read_sas_url(blob_name, expiry_hours=24)
            return sas_url
        except Exception as e:
            logger.error(f"[storage] Failed to save file: {e}")
            raise

    def generate_read_sas_url(self, blob_name: str, expiry_hours: int = 24) -> str:
        """
        Generate a presigned URL (SAS URL) with read permissions

        Args:
            blob_name: The name/path of the blob
            expiry_hours: Hours until the SAS token expires (default: 24)

        Returns:
            Presigned URL with read permissions
        """
        try:
            credentials = self._parse_connection_string()

            # Generate SAS token with read permissions
            sas_token = generate_blob_sas(
                account_name=credentials["account_name"],
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=credentials["account_key"],
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
            )

            # Get the base blob URL and append SAS token
            blob_url = self._get_blob_url(blob_name)
            sas_url = f"{blob_url}?{sas_token}"

            logger.info(f"[storage] Generated SAS URL for: {blob_name}")
            return sas_url
        except Exception as e:
            logger.error(f"[storage] Failed to generate SAS URL: {e}")
            raise

    def get_file(self, blob_name: str) -> bytes:
        """Read file from Azure Blob Storage"""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            blob_client = container_client.get_blob_client(blob_name)

            # Check if blob exists
            if not blob_client.exists():
                raise FileNotFoundError(f"Blob not found: {blob_name}")

            # Download blob content
            download_stream = blob_client.download_blob()
            return download_stream.readall()
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"[storage] Failed to read file: {e}")
            raise

    def delete_file(self, blob_name: str) -> None:
        """Delete file from Azure Blob Storage"""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            blob_client = container_client.get_blob_client(blob_name)

            if blob_client.exists():
                blob_client.delete_blob()
                logger.info(f"[storage] Deleted blob: {blob_name}")
        except Exception as e:
            logger.error(f"[storage] Failed to delete file: {e}")
            raise

    def delete_document_files(self, document_id: str) -> None:
        """Delete all files for a document"""
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )

            # List all blobs for this document
            blob_list = container_client.list_blobs(
                name_starts_with=f"documents/{document_id}/"
            )

            # Delete each blob
            deleted_count = 0
            for blob in blob_list:
                blob_client = container_client.get_blob_client(blob.name)
                blob_client.delete_blob()
                deleted_count += 1

            logger.info(f"[storage] Deleted {deleted_count} blobs for document {document_id}")
        except Exception as e:
            logger.error(f"[storage] Failed to delete document files: {e}")
            raise

    def extract_blob_name_from_url(self, blob_url: str) -> str:
        """
        Extract blob name from a blob URL (removes SAS token)

        Args:
            blob_url: Full blob URL (may include SAS token)

        Returns:
            Blob name (path in container)
        """
        # Remove query parameters (SAS token) if present
        if "?" in blob_url:
            blob_url = blob_url.split("?")[0]

        # Extract blob name after container name
        # Format: .../container_name/blob_name
        if f"/{self.container_name}/" in blob_url:
            parts = blob_url.split(f"/{self.container_name}/", 1)
            if len(parts) == 2:
                return parts[1]

        raise ValueError(f"Could not extract blob name from URL: {blob_url}")

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
