"""Qdrant vector store service for semantic search and similarity matching"""
import logging
import hashlib
from functools import lru_cache
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from backend.config import get_settings

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


def _get_collection_name(case_id: str) -> str:
    """
    Generate collection name from case ID.

    Args:
        case_id: Unique case identifier

    Returns:
        Formatted collection name
    """
    return f"case_{case_id}"


def _generate_point_id(chunk_id: str, case_id: str) -> int:
    """
    Generate a deterministic point ID from chunk_id and case_id.
    Ensures the same chunk always gets the same ID (for idempotency).

    Args:
        chunk_id: Unique chunk identifier
        case_id: Unique case identifier

    Returns:
        Integer point ID (positive)
    """
    # Create deterministic hash from chunk_id and case_id
    combined = f"{case_id}:{chunk_id}"
    # MD5 used here only for deterministic ID generation, not for security
    hash_value = hashlib.md5(combined.encode()).digest()
    # Convert first 8 bytes to unsigned int (Qdrant requires positive IDs)
    point_id = int.from_bytes(hash_value[:8], byteorder='big', signed=False)
    return point_id


def create_collection(case_id: str) -> bool:
    """
    Create a new collection for storing vectors of a case.

    Args:
        case_id: Unique case identifier

    Returns:
        True if collection created successfully

    Raises:
        Exception: If collection creation fails
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(case_id)

        logger.info(f"Creating collection for case: {case_id}")

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

    except Exception as e:
        logger.error(f"Failed to create collection for case {case_id}: {str(e)}")
        raise


def upsert_vectors(
    case_id: str,
    chunks: List[Dict],
    embeddings: List[List[float]]
) -> int:
    """
    Insert or update vectors with metadata for a case's chunks.

    Args:
        case_id: Unique case identifier
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
        collection_name = _get_collection_name(case_id)

        logger.info(f"Upserting {len(chunks)} vectors for case: {case_id}")

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
            point_id = _generate_point_id(chunk_id, case_id)

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

    except Exception as e:
        logger.error(f"Failed to upsert vectors for case {case_id}: {str(e)}")
        raise


def search_vectors(
    case_id: str,
    query_embedding: List[float],
    limit: int = 5
) -> List[Dict]:
    """
    Search for semantically similar chunks using vector similarity.

    Args:
        case_id: Unique case identifier
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
        client = get_qdrant_client()
        collection_name = _get_collection_name(case_id)

        logger.debug(f"Searching vectors in collection: {collection_name}, limit: {limit}")

        # Perform similarity search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit
        )

        # Convert results to dict format
        results = []
        for hit in search_results:
            result_dict = {
                "score": hit.score,
                **hit.payload  # Unpack metadata (chunk_id, page_num, content_preview, section_name)
            }
            results.append(result_dict)

        logger.debug(f"Found {len(results)} similar vectors")
        return results

    except Exception as e:
        logger.error(f"Failed to search vectors in case {case_id}: {str(e)}")
        raise


def delete_collection(case_id: str) -> bool:
    """
    Delete a collection and all its vectors for a case.

    Args:
        case_id: Unique case identifier

    Returns:
        True if collection deleted successfully

    Raises:
        Exception: If deletion fails
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(case_id)

        logger.info(f"Deleting collection for case: {case_id}")

        client.delete_collection(collection_name=collection_name)

        logger.info(f"Successfully deleted collection: {collection_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete collection for case {case_id}: {str(e)}")
        raise
