"""Search API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.services.search import search_service
from app.schemas.search import (
    FullTextSearchRequest,
    SemanticSearchRequest,
    HybridSearchRequest,
    SearchResponse,
)
from shared import generate_embeddings_batch
import logging
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("/full-text", response_model=SearchResponse)
async def full_text_search(
    req: FullTextSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search documents using full-text search (PostgreSQL tsvector)

    Returns chunks ranked by text relevance.
    """
    try:
        start_time = time.time()

        results = await search_service.full_text_search(
            db,
            case_id=req.case_id,
            query=req.query,
            limit=req.limit,
        )

        execution_time = (time.time() - start_time) * 1000

        return SearchResponse(
            results=[r.to_dict() for r in results],
            total=len(results),
            search_type="full_text",
            execution_time_ms=execution_time,
        )

    except Exception as e:
        logger.error(f"[search] Full-text search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    req: SemanticSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Search documents using semantic search (pgvector similarity)

    Generates embedding for query and finds similar document chunks.
    """
    try:
        start_time = time.time()

        # Generate embedding for query
        query_embeddings = await generate_embeddings_batch([req.query])
        query_embedding = query_embeddings[0]

        results = await search_service.semantic_search(
            db,
            case_id=req.case_id,
            query_embedding=query_embedding,
            limit=req.limit,
            threshold=req.threshold,
        )

        execution_time = (time.time() - start_time) * 1000

        return SearchResponse(
            results=[r.to_dict() for r in results],
            total=len(results),
            search_type="semantic",
            execution_time_ms=execution_time,
        )

    except Exception as e:
        logger.error(f"[search] Semantic search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.post("/hybrid", response_model=SearchResponse)
async def hybrid_search(
    req: HybridSearchRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Hybrid search combining full-text and semantic search

    Uses weighted combination of both methods:
    - Full-text (tsvector) for keyword matching
    - Semantic (pgvector) for meaning-based matching

    Weights:
    - ft_weight: weight for full-text results (default 0.5)
    - semantic_weight: weight for semantic results (default 0.5)
    """
    try:
        start_time = time.time()

        # Generate embedding for query
        query_embeddings = await generate_embeddings_batch([req.query])
        query_embedding = query_embeddings[0]

        results = await search_service.hybrid_search(
            db,
            case_id=req.case_id,
            query=req.query,
            query_embedding=query_embedding,
            limit=req.limit,
            threshold=req.threshold,
            ft_weight=req.ft_weight,
            semantic_weight=req.semantic_weight,
        )

        execution_time = (time.time() - start_time) * 1000

        return SearchResponse(
            results=[r.to_dict() for r in results],
            total=len(results),
            search_type="hybrid",
            execution_time_ms=execution_time,
        )

    except Exception as e:
        logger.error(f"[search] Hybrid search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )
