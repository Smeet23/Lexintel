"""Azure Blob Storage operations"""
import logging
import os
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)

try:
    from backend.exceptions import BlobUploadException, BlobDownloadException, BlobDeleteException
except ImportError:
    try:
        from exceptions import BlobUploadException, BlobDownloadException, BlobDeleteException
    except ImportError:
        from ..exceptions import BlobUploadException, BlobDownloadException, BlobDeleteException

def get_settings():
    """Get settings - lazy import to avoid circular dependencies"""
    try:
        from backend.config import get_settings as gs
        return gs()
    except ImportError:
        try:
            from config import get_settings as gs
            return gs()
        except ImportError:
            # Return a mock settings object for testing
            class MockSettings:
                azure_storage_connection_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING', '')
            return MockSettings()

# Magic bytes for file format validation
PDF_MAGIC_BYTES = b"%PDF"
DOCX_MAGIC_BYTES = b"PK\x03\x04"  # ZIP signature (DOCX is a ZIP archive)


def validate_file_format(file_content: bytes, file_type: str) -> bool:
    """
    Validate that file content matches the declared file type by checking magic bytes.

    Args:
        file_content: Raw file bytes
        file_type: Declared file type ('pdf', 'docx', or 'txt')

    Returns:
        True if file content matches the declared type
    """
    if file_type == "pdf":
        return file_content.startswith(PDF_MAGIC_BYTES)
    elif file_type == "docx":
        return file_content.startswith(DOCX_MAGIC_BYTES)
    elif file_type == "txt":
        try:
            file_content.decode('utf-8')
            return True
        except UnicodeDecodeError:
            return False
    return False


def get_blob_client():
    """Get Azure Blob Storage client"""
    settings = get_settings()
    return BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )


async def upload_document_to_blob(file_content: bytes, case_id: str, filename: str) -> str:
    """
    Upload document (PDF, DOCX, or TXT) to Azure Blob Storage and return blob path.

    Note: This function uses fail-fast behavior. If Azure storage is not configured
    or unavailable, it raises BlobUploadException. There is no fallback to local storage.

    Args:
        file_content: Raw document bytes
        case_id: UUID of the case
        filename: Original filename from upload

    Returns:
        Blob path (e.g., "case-uuid/filename.pdf")

    Raises:
        BlobUploadException: If upload fails
    """
    try:
        blob_client = get_blob_client()
        container_client = blob_client.get_container_client("cases")

        # Create container if it doesn't exist
        try:
            container_client.get_container_properties()
        except AzureError as e:
            logger.info("Creating 'cases' container")
            try:
                container_client = blob_client.create_container("cases")
            except AzureError as create_err:
                raise BlobUploadException(
                    "Failed to create blob storage container",
                    detail=f"Container creation failed: {str(create_err)}"
                ) from create_err

        # Upload blob with case_id directory structure
        blob_name = f"{case_id}/{filename}"
        blob_client_ref = container_client.get_blob_client(blob_name)

        logger.info(f"Uploading blob: {blob_name}")
        blob_client_ref.upload_blob(file_content, overwrite=True)

        return blob_name

    except BlobUploadException:
        raise
    except AzureError as e:
        logger.error(f"Azure storage upload failed: {str(e)}")
        raise BlobUploadException(
            "Failed to upload file to blob storage",
            detail=f"Azure error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error uploading blob: {str(e)}")
        raise BlobUploadException(
            "Unexpected error during file upload",
            detail=str(e)
        ) from e


def download_document_from_blob(blob_path: str) -> bytes:
    """
    Download document (PDF, DOCX, or TXT) from Azure Blob Storage.

    Args:
        blob_path: Path to blob (e.g., "case-uuid/filename.pdf")

    Returns:
        Raw document bytes

    Raises:
        BlobDownloadException: If download fails
    """
    try:
        blob_client = get_blob_client()
        blob_client_ref = blob_client.get_blob_client("cases", blob_path)

        logger.info(f"Downloading blob: {blob_path}")
        download_stream = blob_client_ref.download_blob()

        return download_stream.readall()

    except AzureError as e:
        logger.error(f"Azure storage download failed: {str(e)}")
        raise BlobDownloadException(
            "Failed to download file from blob storage",
            detail=f"Azure error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error downloading blob: {str(e)}")
        raise BlobDownloadException(
            "Unexpected error during file download",
            detail=str(e)
        ) from e


def delete_blob(blob_path: str) -> bool:
    """
    Delete blob from Azure Blob Storage.

    Args:
        blob_path: Path to blob

    Returns:
        True if successful

    Raises:
        BlobDeleteException: If deletion fails
    """
    try:
        blob_client = get_blob_client()
        blob_client_ref = blob_client.get_blob_client("cases", blob_path)

        logger.info(f"Deleting blob: {blob_path}")
        blob_client_ref.delete_blob()

        return True

    except AzureError as e:
        logger.error(f"Azure storage deletion failed: {str(e)}")
        raise BlobDeleteException(
            "Failed to delete file from blob storage",
            detail=f"Azure error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error deleting blob: {str(e)}")
        raise BlobDeleteException(
            "Unexpected error during blob deletion",
            detail=str(e)
        ) from e
