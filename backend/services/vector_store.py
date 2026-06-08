"""Qdrant vector store service for semantic search and similarity matching"""
import logging
import hashlib
from functools import lru_cache
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    HnswConfigDiff, PayloadSchemaType,
    Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseVector, Modifier,
    DatetimeRange, IsEmptyCondition, PayloadField,
)

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

# Exception handling
try:
    from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
except ImportError:
    try:
        from qdrant_client.http import UnexpectedResponse, ResponseHandlingException
    except ImportError:
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
        from ..exceptions import VectorStoreException

logger = logging.getLogger(__name__)
settings = get_settings()

# Vector store configuration
VECTOR_SIZE = 1024  # Cohere embed-english-v3.0 output dimensions
UPSERT_BATCH_SIZE = 100  # Qdrant recommended batch size

# Module-level cache: collection_name -> bool (True = named vectors / hybrid mode).
# Schema is immutable post-create so no invalidation is needed; create_collection
# always sets the entry. search_vectors populates on first query if the entry is
# absent (e.g. collection created before this cache was introduced).
_named_vectors_cache: dict = {}


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
        timeout=10,
        check_compatibility=False
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


def _ensure_payload_indexes(client, collection_name: str):
    """Create payload indexes if they don't exist. Safe to call on existing collections.

    Qdrant's create_payload_index is idempotent — calling it on an
    existing index is a no-op.
    """
    index_fields = {
        "page_num": PayloadSchemaType.KEYWORD,
        "section_name": PayloadSchemaType.KEYWORD,
        "document_type": PayloadSchemaType.KEYWORD,
        "jurisdiction": PayloadSchemaType.KEYWORD,
        # Authority hierarchy metadata (from authority_detector)
        "court_level": PayloadSchemaType.KEYWORD,
        "jurisdiction_code": PayloadSchemaType.KEYWORD,
        "source_type": PayloadSchemaType.KEYWORD,
        "binding_authority": PayloadSchemaType.BOOL,
        "authority_score": PayloadSchemaType.FLOAT,
        # Temporal metadata (from temporal_extractor) — DATETIME enables as-of range filters
        "document_status": PayloadSchemaType.KEYWORD,
        "effective_date": PayloadSchemaType.DATETIME,
        "superseded_date": PayloadSchemaType.DATETIME,
    }
    for field, schema in index_fields.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # Index already exists or field not populated yet


# ----------------------------------------------------------------------
# Temporal ("as-of-date") filtering
# ----------------------------------------------------------------------

# Sentinel keys merged into the plain query_filter dict. They are popped by
# _build_qdrant_filter and never sent to Qdrant as literal field matches.
_TEMPORAL_AS_OF_KEY = "_temporal_as_of"          # ISO 8601 string or None
_TEMPORAL_CURRENT_ONLY_KEY = "_temporal_current_only"  # bool


def build_temporal_filter(
    as_of_date=None,
    exclude_superseded: bool = True,
    base_filter: Optional[Dict] = None,
) -> Dict:
    """Build a plain query_filter dict carrying temporal-filtering sentinels.

    Returns a NEW dict merging ``base_filter`` (jurisdiction, document_type, …)
    with temporal sentinel keys that :func:`_build_qdrant_filter` translates into
    Qdrant ``DatetimeRange`` / ``IsEmptyCondition`` clauses. Inputs are never
    mutated (immutable-style).

    Semantics:
      - ``as_of_date`` set  → include docs whose ``effective_date <= as_of`` (or
        undated, behind the ``temporal_include_undated`` flag) and exclude docs
        whose ``superseded_date <= as_of``. ``exclude_superseded`` is ignored in
        this mode (the date range is authoritative).
      - ``as_of_date`` None and ``exclude_superseded`` True → current-law only
        (``document_status == "current"``).
      - ``as_of_date`` None and ``exclude_superseded`` False → no temporal
        constraint (historical / all-versions intent).

    Args:
        as_of_date: ``datetime`` (or ISO string) to evaluate the law as-of, or None.
        exclude_superseded: When no as-of date, whether to restrict to current docs.
        base_filter: Optional non-temporal field filters to merge.

    Returns:
        Plain dict suitable to pass as ``query_filter`` to search functions.
    """
    merged: Dict = dict(base_filter) if base_filter else {}

    as_of_iso: Optional[str] = None
    if as_of_date is not None:
        as_of_iso = (
            as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)
        )

    merged[_TEMPORAL_AS_OF_KEY] = as_of_iso
    # current-only is only meaningful when there is no explicit as-of date
    merged[_TEMPORAL_CURRENT_ONLY_KEY] = bool(exclude_superseded) and as_of_iso is None
    return merged


def _build_qdrant_filter(query_filter: Optional[Dict]) -> Optional[Filter]:
    """Translate a plain query_filter dict into a Qdrant ``Filter``.

    Shared by BOTH :func:`search_vectors` and :func:`search_sparse_vectors` so the
    dense and sparse retrieval paths apply identical (temporal + field) filtering.

    Pops the temporal sentinels (:data:`_TEMPORAL_AS_OF_KEY`,
    :data:`_TEMPORAL_CURRENT_ONLY_KEY`); all remaining keys become exact
    ``FieldCondition`` matches.

    Null-safety:
      - ``effective_date <= as_of`` is wrapped in a ``should`` with an
        ``IsEmptyCondition`` so undated docs are included (gated on
        ``temporal_include_undated``). A bare ``must`` range would silently drop
        them (Qdrant range conditions never match missing fields).
      - ``superseded_date <= as_of`` goes in ``must_not``; points missing the key
        (never superseded, or ingested before the field existed) stay in —
        desired, slightly permissive.
    """
    if not query_filter:
        return None

    # Copy so we never mutate the caller's dict.
    qf = dict(query_filter)
    temporal_as_of = qf.pop(_TEMPORAL_AS_OF_KEY, None)
    current_only = qf.pop(_TEMPORAL_CURRENT_ONLY_KEY, False)

    must = [
        FieldCondition(key=field, match=MatchValue(value=value))
        for field, value in qf.items()
    ]
    must_not = []

    if temporal_as_of:
        include_undated = getattr(settings, "temporal_include_undated", True)
        effective_in_force = FieldCondition(
            key="effective_date",
            range=DatetimeRange(lte=temporal_as_of),
        )
        if include_undated:
            # (effective_date <= as_of) OR (effective_date is empty)
            must.append(
                Filter(
                    should=[
                        effective_in_force,
                        IsEmptyCondition(is_empty=PayloadField(key="effective_date")),
                    ]
                )
            )
        else:
            must.append(effective_in_force)
        # Exclude anything superseded on/before the as-of date. Null superseded_date
        # rows are left in (range never matches missing fields) — never-superseded.
        must_not.append(
            FieldCondition(
                key="superseded_date",
                range=DatetimeRange(lte=temporal_as_of),
            )
        )
    elif current_only:
        # Exclude superseded/repealed, but KEEP current/unknown/draft. Requiring
        # =="current" is too strict — chunks are stamped 'unknown' at ingest and
        # supersession is resolved authoritatively from the DB post-filter. A hard
        # 'current' match here would drop everything and force a relax every time.
        must_not.append(
            FieldCondition(key="document_status", match=MatchValue(value="superseded"))
        )
        must_not.append(
            FieldCondition(key="document_status", match=MatchValue(value="repealed"))
        )

    if not must and not must_not:
        return None

    return Filter(must=must or None, must_not=must_not or None)


def create_collection(matter_id: str) -> bool:
    """
    Create a collection for storing vectors of a matter (safe — won't drop existing).

    Uses HNSW config optimized for high-dimensional vectors.
    Recreates collection if existing dimension mismatches (e.g. embedding model change).
    Creates payload indexes on page_num and section_name for filtered search.

    Args:
        matter_id: Unique matter identifier

    Returns:
        True if collection created or already exists

    Raises:
        VectorStoreException: If collection creation fails
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        logger.info(f"Ensuring collection exists for matter: {matter_id}")

        # Check if collection already exists
        if client.collection_exists(collection_name):
            # Verify vector dimensions match; recreate if stale
            info = client.get_collection(collection_name)
            existing_size = info.config.params.vectors.size if hasattr(info.config.params.vectors, "size") else None
            if existing_size and existing_size != VECTOR_SIZE:
                logger.warning(
                    f"Collection {collection_name} has dimension {existing_size}, "
                    f"expected {VECTOR_SIZE}. Recreating collection."
                )
                client.delete_collection(collection_name=collection_name)
                # Cache entry will be (re)set after the new collection is created below.
                _named_vectors_cache.pop(collection_name, None)
            else:
                # Ensure new payload indexes exist on pre-existing collections
                _ensure_payload_indexes(client, collection_name)
                # Populate the named-vectors cache from the live collection config.
                existing_vecs_cfg = info.config.params.vectors
                _named_vectors_cache[collection_name] = (
                    isinstance(existing_vecs_cfg, dict) and "dense" in existing_vecs_cfg
                )
                logger.info(f"Collection already exists: {collection_name}")
                return True

        # Create new collection with HNSW config for high-dim vectors
        # Include sparse vector config for BM25 hybrid search (backward-compatible)
        sparse_config = {}
        try:
            sparse_config = {
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            }
        except Exception as e:
            logger.warning(f"Sparse vector config creation failed (dense-only mode): {e}")

        # Use named vectors ("dense") when sparse vectors are available,
        # so hybrid upserts can reference both "dense" and "sparse" by name.
        # Fall back to unnamed vectors if sparse config fails.
        if sparse_config:
            vectors_cfg = {
                "dense": VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                )
            }
        else:
            vectors_cfg = VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            )

        create_kwargs = dict(
            collection_name=collection_name,
            vectors_config=vectors_cfg,
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=200
            ),
        )
        if sparse_config:
            create_kwargs["sparse_vectors_config"] = sparse_config

        client.create_collection(**create_kwargs)

        # Cache whether this collection uses named vectors (hybrid mode).
        _named_vectors_cache[collection_name] = bool(sparse_config)

        # Create payload indexes for filtered search
        _ensure_payload_indexes(client, collection_name)

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
    embeddings: List[List[float]],
    sparse_vectors: Optional[List[Optional[Dict]]] = None
) -> int:
    """
    Insert or update vectors with metadata for a matter's chunks.
    Uses batched upserts (batch_size=100) for reliability.

    Args:
        matter_id: Unique matter identifier
        chunks: List of chunk dicts with keys: id, content, page_num, section_name, chunk_sequence
        embeddings: List of embedding vectors from embeddings service
        sparse_vectors: Optional list of sparse vector dicts with 'indices' and 'values' keys.
            When provided, creates named vectors (dense + sparse) for hybrid search.
            None entries in the list are treated as missing sparse vectors for that chunk.

    Returns:
        Number of vectors successfully upserted

    Raises:
        ValueError: If inputs are invalid or mismatched
        VectorStoreException: If upsert operation fails
    """
    if not chunks or not embeddings:
        raise ValueError("Chunks and embeddings lists cannot be empty")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunks and embeddings count mismatch: "
            f"expected {len(chunks)} embeddings, got {len(embeddings)}"
        )

    # Validate sparse_vectors length if provided
    if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
        logger.warning(
            f"Sparse vectors count mismatch ({len(sparse_vectors)} vs {len(chunks)} chunks). "
            "Falling back to dense-only mode."
        )
        sparse_vectors = None

    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        logger.info(f"Upserting {len(chunks)} vectors for matter: {matter_id}")

        # Determine if we should use named vectors (hybrid mode).
        # Guard: only enable if the collection was actually created with named
        # vectors ("dense"/"sparse"). If the collection is dense-only but sparse
        # vectors were produced, fall back to dense-only to avoid a schema crash.
        wants_named = (
            sparse_vectors is not None
            and any(sv is not None for sv in sparse_vectors)
        )
        collection_has_named = _named_vectors_cache.get(collection_name)
        if collection_has_named is None:
            # Cache miss — populate from live collection config.
            try:
                _info = client.get_collection(collection_name)
                _vecs_cfg = _info.config.params.vectors
                collection_has_named = isinstance(_vecs_cfg, dict) and "dense" in _vecs_cfg
                _named_vectors_cache[collection_name] = collection_has_named
            except Exception:
                collection_has_named = False

        use_named_vectors = wants_named and collection_has_named
        if wants_named and not collection_has_named:
            logger.warning(
                f"Collection {collection_name} is dense-only but sparse vectors were produced; "
                "falling back to dense-only upsert to avoid schema mismatch."
            )
            sparse_vectors = None
        if use_named_vectors:
            logger.info(f"Using named vectors (dense + sparse) for hybrid search")

        # Prepare all points
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Validate embedding dimension
            if len(embedding) != VECTOR_SIZE:
                raise ValueError(
                    f"Embedding dimension mismatch: "
                    f"expected {VECTOR_SIZE}, got {len(embedding)}"
                )

            chunk_id = str(chunk.get("id", "unknown"))

            # Create metadata from chunk — store full content for RAG quality
            metadata = {
                "chunk_id": chunk_id,
                "chunk_sequence": chunk.get("chunk_sequence", 0),
                "page_num": str(chunk.get("page_num", "")),
                "section_name": str(chunk.get("section_name", "")),
                "content": chunk.get("content", ""),
                "document_id": str(chunk.get("document_id", "")),
                "document_name": str(chunk.get("document_name", "")),
                "concepts": chunk.get("concepts", []),
                "document_type": str(chunk.get("document_type", "")),
                "jurisdiction": str(chunk.get("jurisdiction", "")),
                # Authority hierarchy metadata (from authority_detector)
                "court_level": str(chunk.get("court_level", "unknown")),
                "jurisdiction_code": str(chunk.get("jurisdiction_code", "unknown")),
                "authority_score": float(chunk.get("authority_score", 0.5)),
                "binding_authority": bool(chunk.get("binding_authority", False)),
                "source_type": str(chunk.get("source_type", "other")),
                # Temporal metadata (from temporal_extractor)
                "effective_date": chunk.get("effective_date"),  # ISO string or None
                "superseded_date": chunk.get("superseded_date"),  # ISO string or None
                "document_status": str(chunk.get("document_status", "unknown")),
            }

            # Generate deterministic point ID for idempotency
            point_id = _generate_point_id(chunk_id, matter_id)

            # Build vector payload: named vectors for hybrid, plain for dense-only
            if use_named_vectors:
                sv = sparse_vectors[idx] if sparse_vectors else None
                vector_payload = {"dense": embedding}
                if sv is not None and sv.get("indices") and sv.get("values"):
                    vector_payload["sparse"] = SparseVector(
                        indices=sv["indices"],
                        values=sv["values"]
                    )
                point = PointStruct(
                    id=point_id,
                    vector=vector_payload,
                    payload=metadata
                )
            else:
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=metadata
                )
            points.append(point)

        # Batch upsert
        total_upserted = 0
        for i in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[i:i + UPSERT_BATCH_SIZE]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            total_upserted += len(batch)
            logger.debug(f"Upserted batch {i // UPSERT_BATCH_SIZE + 1}: {len(batch)} points")

        logger.info(f"Successfully upserted {total_upserted} vectors")
        return total_upserted

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
    limit: int = 5,
    query_filter: Dict = None
) -> List[Dict]:
    """
    Search for semantically similar chunks using the Qdrant Python client.

    Args:
        matter_id: Unique matter identifier
        query_embedding: Query vector from embeddings service
        limit: Maximum number of results to return (default: 5)
        query_filter: Optional dict of payload field conditions
            e.g. {"jurisdiction": "UK"} or {"document_type": "statute"}

    Returns:
        List of dicts with keys: chunk_id, chunk_sequence, page_num,
        section_name, score, content

    Raises:
        ValueError: If query embedding dimension is incorrect
        VectorStoreException: If search operation fails
    """
    if len(query_embedding) != VECTOR_SIZE:
        raise ValueError(
            f"Query embedding dimension mismatch: "
            f"expected {VECTOR_SIZE}, got {len(query_embedding)}"
        )

    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        # Build Qdrant filter (field + temporal) via shared translation helper
        qdrant_filter = _build_qdrant_filter(query_filter)
        if qdrant_filter is not None:
            logger.debug(f"Applying filter: {query_filter}")

        logger.debug(f"Searching vectors in collection: {collection_name}, limit: {limit}")

        # Detect if collection uses named vectors (hybrid mode) and specify "dense".
        # Use module-level cache to avoid a get_collection RPC on every query.
        query_kwargs = dict(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        is_named = _named_vectors_cache.get(collection_name)
        if is_named is None:
            # Cache miss — populate once from the live collection config.
            try:
                _info = client.get_collection(collection_name)
                _vecs_cfg = _info.config.params.vectors
                is_named = isinstance(_vecs_cfg, dict) and "dense" in _vecs_cfg
                _named_vectors_cache[collection_name] = is_named
            except Exception:
                is_named = False
        if is_named:
            query_kwargs["using"] = "dense"

        response = client.query_points(**query_kwargs)
        hits = response.points

        if not hits:
            return []

        # Convert results to dict format
        results = []
        for hit in hits:
            payload = hit.payload or {}
            # Full content stored in payload; fall back to legacy content_preview field
            full_content = payload.get("content", "") or payload.get("content_preview", "")
            result_dict = {
                "score": hit.score,
                "chunk_id": payload.get("chunk_id", ""),
                "chunk_sequence": payload.get("chunk_sequence", 0),
                "page_num": payload.get("page_num", ""),
                "section_name": payload.get("section_name", ""),
                "content": full_content,
                "document_id": payload.get("document_id", ""),
                "document_name": payload.get("document_name", ""),
                "concepts": payload.get("concepts", []),
                "document_type": payload.get("document_type", ""),
                "jurisdiction": payload.get("jurisdiction", ""),
                # Authority hierarchy metadata
                "court_level": payload.get("court_level", "unknown"),
                "jurisdiction_code": payload.get("jurisdiction_code", "unknown"),
                "authority_score": payload.get("authority_score", 0.5),
                "binding_authority": payload.get("binding_authority", False),
                "source_type": payload.get("source_type", "document"),
                # Temporal metadata
                "effective_date": payload.get("effective_date"),
                "superseded_date": payload.get("superseded_date"),
                "document_status": payload.get("document_status", "unknown"),
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
    except Exception as e:
        logger.error(f"Unexpected error searching vectors in matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Unexpected error during vector search",
            detail=str(e)
        ) from e


def search_sparse_vectors(
    matter_id: str,
    sparse_vector: Dict,
    limit: int = 5,
    query_filter: Dict = None
) -> List[Dict]:
    """
    Search for chunks using BM25 sparse vectors (keyword search).

    Args:
        matter_id: Unique matter identifier
        sparse_vector: Dict with 'indices' (list of ints) and 'values' (list of floats)
        limit: Maximum number of results to return
        query_filter: Optional dict of payload field conditions

    Returns:
        List of dicts with keys: chunk_id, chunk_sequence, page_num,
        section_name, score, content

    Raises:
        VectorStoreException: If search operation fails
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        # Build Qdrant filter (field + temporal) via shared translation helper
        # so the sparse path applies identical temporal filtering as the dense path.
        qdrant_filter = _build_qdrant_filter(query_filter)

        logger.debug(f"Sparse search in collection: {collection_name}, limit: {limit}")

        sparse_query = SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"]
        )

        response = client.query_points(
            collection_name=collection_name,
            query=sparse_query,
            using="sparse",
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        hits = response.points

        if not hits:
            return []

        # Convert results to dict format (same as search_vectors)
        results = []
        for hit in hits:
            payload = hit.payload or {}
            full_content = payload.get("content", "") or payload.get("content_preview", "")
            result_dict = {
                "score": hit.score,
                "chunk_id": payload.get("chunk_id", ""),
                "chunk_sequence": payload.get("chunk_sequence", 0),
                "page_num": payload.get("page_num", ""),
                "section_name": payload.get("section_name", ""),
                "content": full_content,
                "document_id": payload.get("document_id", ""),
                "document_name": payload.get("document_name", ""),
                "concepts": payload.get("concepts", []),
                "document_type": payload.get("document_type", ""),
                "jurisdiction": payload.get("jurisdiction", ""),
                # Authority hierarchy metadata
                "court_level": payload.get("court_level", "unknown"),
                "jurisdiction_code": payload.get("jurisdiction_code", "unknown"),
                "authority_score": payload.get("authority_score", 0.5),
                "binding_authority": payload.get("binding_authority", False),
                "source_type": payload.get("source_type", "document"),
                # Temporal metadata
                "effective_date": payload.get("effective_date"),
                "superseded_date": payload.get("superseded_date"),
                "document_status": payload.get("document_status", "unknown"),
            }
            results.append(result_dict)

        logger.debug(f"Sparse search found {len(results)} results")
        return results

    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"Qdrant API error in sparse search for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Failed to perform sparse vector search",
            detail=f"Qdrant error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error in sparse search for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Unexpected error during sparse vector search",
            detail=str(e)
        ) from e


DELETE_BATCH_SIZE = 500  # Batch size for Qdrant deletions


def delete_vectors_by_document(matter_id: str, chunk_ids: List[str]) -> int:
    """
    Delete specific vectors from a matter's collection by chunk IDs.
    Handles missing collections gracefully (returns 0).
    Batches deletions to avoid timeout on large documents.

    Args:
        matter_id: Unique matter identifier
        chunk_ids: List of chunk ID strings to remove

    Returns:
        Number of vectors deleted (0 if collection doesn't exist)

    Raises:
        VectorStoreException: If deletion fails for reasons other than missing collection
    """
    try:
        client = get_qdrant_client()
        collection_name = _get_collection_name(matter_id)

        # Check if collection exists — skip silently if not
        if not client.collection_exists(collection_name):
            logger.info(f"Collection {collection_name} does not exist, skipping vector deletion")
            return 0

        point_ids = [_generate_point_id(cid, matter_id) for cid in chunk_ids]

        from qdrant_client.models import PointIdsList

        # Batch deletions to avoid timeout on large documents
        for i in range(0, len(point_ids), DELETE_BATCH_SIZE):
            batch = point_ids[i:i + DELETE_BATCH_SIZE]
            client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=batch)
            )

        logger.info(f"Deleted {len(point_ids)} vectors for document chunks in matter {matter_id}")
        return len(point_ids)

    except (UnexpectedResponse, ResponseHandlingException) as e:
        logger.error(f"Qdrant API error deleting vectors for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Failed to delete document vectors",
            detail=f"Qdrant error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error deleting vectors for matter {matter_id}: {str(e)}")
        raise VectorStoreException(
            "Unexpected error during vector deletion",
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
