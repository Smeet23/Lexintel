"""Azure Blob Storage operations"""
import logging
import os
import asyncio
import sys
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)

# Local storage fallback
LOCAL_STORAGE_PATH = Path("/tmp/lexintel_storage")

def get_settings():
    """Get settings - lazy import to avoid circular dependencies"""
    try:
        from config import get_settings as gs
        return gs()
    except ImportError:
        try:
            sys.path.insert(0, '/app')
            from config import get_settings as gs
            return gs()
        except Exception:
            # Return a mock settings object for testing
            class MockSettings:
                azure_storage_connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING', '')
            return MockSettings()

# PDF magic bytes for validation
PDF_MAGIC_BYTES = b"%PDF"


def validate_pdf(file_content: bytes) -> bool:
    """
    Validate that file content is actually a PDF by checking magic bytes.

    Args:
        file_content: Raw file bytes

    Returns:
        True if file starts with PDF magic bytes
    """
    return file_content.startswith(PDF_MAGIC_BYTES)


def get_blob_client():
    """Get Azure Blob Storage client"""
    settings = get_settings()
    return BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )


async def upload_pdf_to_blob(file_content: bytes, case_id: str, filename: str) -> str:
    """
    Upload PDF to Azure Blob Storage and return blob path.
    Falls back to local storage if Azure is unavailable.

    Args:
        file_content: Raw PDF bytes
        case_id: UUID of the case
        filename: Original filename from upload

    Returns:
        Blob path (e.g., "case-uuid/filename.pdf")
    """
    try:
        blob_client = get_blob_client()
        container_client = blob_client.get_container_client("cases")

        # Create container if it doesn't exist
        try:
            container_client.get_container_properties()
        except AzureError:
            logger.info("Creating 'cases' container")
            container_client = blob_client.create_container("cases")

        # Upload blob with case_id directory structure
        blob_name = f"{case_id}/{filename}"
        blob_client_ref = container_client.get_blob_client(blob_name)

        logger.info(f"Uploading blob: {blob_name}")
        blob_client_ref.upload_blob(file_content, overwrite=True)

        return blob_name

    except (AzureError, Exception) as e:
        # Fallback to local storage for testing/development
        logger.warning(f"Azure storage failed: {str(e)}, falling back to local storage")

        try:
            # Create local storage directory
            LOCAL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
            case_dir = LOCAL_STORAGE_PATH / str(case_id)
            case_dir.mkdir(parents=True, exist_ok=True)

            # Write file locally
            blob_name = f"{case_id}/{filename}"
            local_path = case_dir / filename

            logger.info(f"Storing locally: {local_path}")
            local_path.write_bytes(file_content)

            return blob_name
        except Exception as local_err:
            logger.error(f"Local storage also failed: {str(local_err)}")
            raise


def download_pdf_from_blob(blob_path: str) -> bytes:
    """
    Download PDF from Azure Blob Storage or local storage.

    Args:
        blob_path: Path to blob (e.g., "case-uuid/filename.pdf")

    Returns:
        Raw PDF bytes
    """
    # Try local storage first if it exists
    local_path = LOCAL_STORAGE_PATH / blob_path
    if local_path.exists():
        logger.info(f"Reading from local storage: {local_path}")
        return local_path.read_bytes()

    # Fall back to Azure
    try:
        blob_client = get_blob_client()
        blob_client_ref = blob_client.get_blob_client("cases", blob_path)

        logger.info(f"Downloading blob: {blob_path}")
        download_stream = blob_client_ref.download_blob()

        return download_stream.readall()

    except (AzureError, Exception) as e:
        logger.error(f"Error downloading from blob storage: {str(e)}")
        raise


def delete_blob(blob_path: str) -> bool:
    """
    Delete blob from Azure Blob Storage.

    Args:
        blob_path: Path to blob

    Returns:
        True if successful

    Raises:
        AzureError: If deletion fails
    """
    try:
        blob_client = get_blob_client()
        blob_client_ref = blob_client.get_blob_client("cases", blob_path)

        logger.info(f"Deleting blob: {blob_path}")
        blob_client_ref.delete_blob()

        return True

    except AzureError as e:
        logger.error(f"Azure storage error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting blob: {str(e)}")
        raise
