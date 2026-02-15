"""Qdrant vector store service for semantic search and similarity matching"""
import logging
import hashlib
import requests
from functools import lru_cache
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

try:
    from backend.config import get_settings
except ImportError:
    from config import get_settings

# Exception handling
try:
    from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
except ImportError:
    # Fallback for different Qdrant versions
    try:
        from qdrant_client.http import UnexpectedResponse, ResponseHandlingException
    except ImportError:
        # If specific exceptions not available, create dummy ones
        class UnexpectedResponse(Exception):
            pass
        class ResponseHandlingException(Exception):
            pass

try:
    from backend.exceptions import VectorStoreException
except ImportError:
    try:
        from exceptions import VectorStoreException
    except ImportError:
        from .exceptions import VectorStoreException

logger = logging.getLogger(__name__)
settings = get_settings()

# Vector store configuration
VECTOR_SIZE = 3072  # Matches text-embedding-3-large output dimensions
DISTANCE_METRIC = "Cosine"  # Cosine similarity for semantic search


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Get or create Qdrant client with configured connection.

    Returns:
        QdrantClient instance connected to Qdrant server

    Raises:
        ValueError: If Qdrant URL is not configured
    """
    if not settings.qdrant_url:
        logger.error("QDRANT_URL environment variable not set")
        raise ValueError("Vector store is not properly configured")

    logger.debug(f"Creating Qdrant client for URL: {settings.qdrant_url}")

    return QdrantClient(
        url=settings.qdrant_url,
        timeout=30
    )


def _get_collection_name(matter_id: str) -> str:
    """
    Generate collection name from matter ID.

    Args:
        matter_id: Unique matter identifier

    Returns:
        Formatted collection name
    """
    return f"matter_{matter_id}"


def _generate_point_id(chunk_id: str, matter_id: str) -> int:
    """
    Generate a deterministic point ID from chunk_id and matter_id.
    Ensures the same chunk always gets the same ID (for idempotency).

    Args:
        chunk_id: Unique chunk identifier
        matter_id: Unique matter identifier

    Returns:
        Integer point ID (positive)
    """
    # Create deterministic hash from chunk_id and matter_id
    combined = f"{matter_id}:{chunk_id}"
    # MD5 used here only for deterministic ID generation, not for security
    hash_value = hashlib.md5(combined.encode()).digest()
    # Convert first 8 bytes to unsigned int (Qdrant requires positive IDs)
    point_id = int.from_bytes(hash_value[:8], byteorder='big', signed=False)
    return point_id


def create_collection(matter_id: str) -> bool:
    """
    Create a new collection for storing vectors of a matter.

    Args:
        matter_id: Unique matter identifier

    Returns:
        True if collection created successfully

    Raises:
        VectorStoreException: If collection creation fails
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        logger.info(f"Creating collection for matter: {matter_id}")

        # Recreate collection (drops if exists, creates new)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        logger.info(f"Successfully created collection: {collection_name}")
        return True

    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"Qdrant API error creating collection for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Failed to create vector collection",
            detail=f"Qdrant error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating collection for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Unexpected error during collection creation",
            detail=str(e)
        ) from e


def upsert_vectors(
    matter_id: str,
    chunks: List[Dict],
    embeddings: List[List[float]]
) -> int:
    """
    Insert or update vectors with metadata for a matter's chunks.

    Args:
        matter_id: Unique matter identifier
        chunks: List of chunk dicts with keys: id, content, page_num, section_name
        embeddings: List of 3072-dimensional vectors from embeddings service

    Returns:
        Number of vectors successfully upserted

    Raises:
        ValueError: If inputs are invalid or mismatched
        Exception: If upsert operation fails
    """
    if not chunks or not embeddings:
        raise ValueError("Chunks and embeddings lists cannot be empty")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks and embeddings count mismatch: "
            f"expected {len(chunks)} embeddings, got {len(embeddings)}"
        )

    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        logger.info(f"Upserting {len(chunks)} vectors for matter: {matter_id}")

        # Prepare points for upsert
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Validate embedding dimension
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"expected {VECTOR_SIZE}, got {len(embedding)}"
                )

            chunk_id = str(chunk.get("id", "unknown"))

            # Create metadata from chunk
            metadata = {
                "chunk_id": chunk_id,
                "page_num": str(chunk.get("page_num", "")),
                "section_name": str(chunk.get("section_name", "")),
                "content_preview": chunk.get("content", "")[:200],  # Store first 200 chars
            }

            # Generate deterministic point ID for idempotency
            point_id = _generate_point_id(chunk_id, matter_id)

            # Create point with vector and metadata
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            )
            points.append(point)

        # Upsert all points
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info(f"Successfully upserted {len(points)} vectors")
        return len(points)

    except ValueError:
        raise
    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"Qdrant API error upserting vectors for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Failed to upsert vectors",
            detail=f"Qdrant error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error upserting vectors for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Unexpected error during vector upsert",
            detail=str(e)
        ) from e


def search_vectors(
    matter_id: str,
    query_embedding: List[float],
    limit: int = 5
) -> List[Dict]:
    """
    Search for semantically similar chunks using vector similarity.

    Args:
        matter_id: Unique matter identifier
        query_embedding: 3072-dimensional query vector from embeddings service
        limit: Maximum number of results to return (default: 5)

    Returns:
        List of dicts with keys: chunk_id, page_num, content_preview, section_name, score

    Raises:
        ValueError: If query embedding dimension is incorrect
        Exception: If search operation fails
    """
    if len(query_embedding) != VECTOR_SIZE:
        raise ValueError(
            f"Query embedding dimension mismatch: "
            f"expected {VECTOR_SIZE}, got {len(query_embedding)}"
        )

    try:
        collection_name = _get_collection_name(matter_id)

        logger.debug(f"Searching vectors in collection: {collection_name}, limit: {limit}")

        # Use REST API directly for compatibility with Qdrant 1.7.0
        # (query_points API is not available in older versions)
        qdrant_url = settings.qdrant_url.rstrip('/')
        search_url = f"{qdrant_url}/collections/{collection_name}/points/search"

        payload = {
            "vector": query_embedding,
            "limit": limit,
            "with_payload": True
        }

        response = requests.post(search_url, json=payload, timeout=30)
        response.raise_for_status()

        search_data = response.json()
        if not search_data.get("result"):
            return []

        # Convert results to dict format
        results = []
        for hit in search_data["result"]:
            result_dict = {
                "score": hit.get("score", 0),
                **hit.get("payload", {}),  # Unpack metadata (chunk_id, page_num, content_preview, section_name)
                "content": hit.get("payload", {}).get("content_preview", "")  # Map content_preview to content for RAG
            }
            results.append(result_dict)

        logger.debug(f"Found {len(results)} similar vectors")
        return results

    except ValueError:
        raise
    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"Qdrant API error searching vectors in matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Failed to search vectors",
            detail=f"Qdrant error: {str(e)}"
        ) from e
    except requests.RequestException as e:
        logger.error(f"HTTP request error searching vectors for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Failed to reach vector store service",
            detail=f"HTTP error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error searching vectors in matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Unexpected error during vector search",
            detail=str(e)
        ) from e


def delete_collection(matter_id: str) -> bool:
    """
    Delete a collection and all its vectors for a matter.

    Args:
        matter_id: Unique matter identifier

    Returns:
        True if collection deleted successfully

    Raises:
        VectorStoreException: If deletion fails
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        logger.info(f"Deleting collection for matter: {matter_id}")

        client.delete_collection(collection_name=collection_name)

        logger.info(f"Successfully deleted collection: {collection_name}")
        return True

    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"Qdrant API error deleting collection for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Failed to delete vector collection",
            detail=f"Qdrant error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error deleting collection for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Unexpected error during collection deletion",
            detail=str(e)
        ) from e
