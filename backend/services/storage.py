"""Azure Blob Storage operations"""
import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

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
    return BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )


async def upload_pdf_to_blob(file_content: bytes, case_id: str, filename: str) -> str:
    """
    Upload PDF to Azure Blob Storage and return blob path.

    Args:
        file_content: Raw PDF bytes
        case_id: UUID of the case
        filename: Original filename from upload

    Returns:
        Blob path (e.g., "case-uuid/filename.pdf")

    Raises:
        AzureError: If upload fails
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

    except AzureError as e:
        logger.error(f"Azure storage error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading to blob storage: {str(e)}")
        raise


async def download_pdf_from_blob(blob_path: str) -> bytes:
    """
    Download PDF from Azure Blob Storage.

    Args:
        blob_path: Path to blob (e.g., "case-uuid/filename.pdf")

    Returns:
        Raw PDF bytes

    Raises:
        AzureError: If download fails
    """
    try:
        blob_client = get_blob_client()
        blob_client_ref = blob_client.get_blob_client("cases", blob_path)

        logger.info(f"Downloading blob: {blob_path}")
        download_stream = blob_client_ref.download_blob()

        return download_stream.readall()

    except AzureError as e:
        logger.error(f"Azure storage error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading from blob storage: {str(e)}")
        raise


async def delete_blob(blob_path: str) -> bool:
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
