"""
Document search service with full-text and semantic search capabilities.

Implements:
- Full-text search using PostgreSQL tsvector
- Semantic search using pgvector similarity
- Hybrid search combining both methods
"""

import logging
import json
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func
from sqlalchemy.sql import desc
from shared.models import DocumentChunk, Document

logger = logging.getLogger(__name__)


class SearchResult:
    """Search result with metadata"""

    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        document_title: str,
        chunk_text: str,
        chunk_index: int,
        relevance_score: float,
        search_type: str,
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.chunk_text = chunk_text
        self.chunk_index = chunk_index
        self.relevance_score = relevance_score
        self.search_type = search_type

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "chunk_text": self.chunk_text,
            "chunk_index": self.chunk_index,
            "relevance_score": float(self.relevance_score),
            "search_type": self.search_type,
        }


class SearchService:
    """Document search service"""

    @staticmethod
    async def full_text_search(
        session: AsyncSession,
        case_id: str,
        query: str,
        limit: int = 10,
    ) -> List[SearchResult]:
        """
        Full-text search using PostgreSQL tsvector

        Args:
            session: AsyncSession
            case_id: Filter by case
            query: Search query
            limit: Max results to return

        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"[search] Full-text search for: {query}")

            # Use raw SQL for tsvector search
            sql = text("""
                SELECT
                    dc.id,
                    dc.document_id,
                    d.title,
                    dc.chunk_text,
                    dc.chunk_index,
                    ts_rank(dc.search_vector, plainto_tsquery('english', :query)) as relevance
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.case_id = :case_id
                AND dc.search_vector @@ plainto_tsquery('english', :query)
                ORDER BY relevance DESC
                LIMIT :limit
            """)

            result = await session.execute(
                sql,
                {
                    "case_id": case_id,
                    "query": query,
                    "limit": limit,
                },
            )

            results = []
            for row in result:
                results.append(
                    SearchResult(
                        chunk_id=row[0],
                        document_id=row[1],
                        document_title=row[2],
                        chunk_text=row[3],
                        chunk_index=row[4],
                        relevance_score=row[5] or 0.0,
                        search_type="full_text",
                    )
                )

            logger.info(f"[search] Found {len(results)} full-text results")
            return results

        except Exception as e:
            logger.error(f"[search] Full-text search error: {e}")
            raise

    @staticmethod
    async def semantic_search(
        session: AsyncSession,
        case_id: str,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.5,
    ) -> List[SearchResult]:
        """
        Semantic search using pgvector cosine similarity

        Args:
            session: AsyncSession
            case_id: Filter by case
            query_embedding: Query embedding vector
            limit: Max results to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"[search] Semantic search with threshold: {threshold}")

            # Convert embedding list to pgvector string format
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            # Use pgvector cosine distance (closer to 0 = more similar)
            # Convert to similarity score (0-1, 1=most similar)
            sql = text(f"""
                SELECT
                    dc.id,
                    dc.document_id,
                    d.title,
                    dc.chunk_text,
                    dc.chunk_index,
                    (1 - (dc.embedding <=> '{embedding_str}'::vector)) as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE d.case_id = :case_id
                AND dc.embedding IS NOT NULL
                AND (1 - (dc.embedding <=> '{embedding_str}'::vector)) >= :threshold
                ORDER BY similarity DESC
                LIMIT :limit
            """)

            result = await session.execute(
                sql,
                {
                    "case_id": case_id,
                    "threshold": threshold,
                    "limit": limit,
                },
            )

            results = []
            for row in result:
                results.append(
                    SearchResult(
                        chunk_id=row[0],
                        document_id=row[1],
                        document_title=row[2],
                        chunk_text=row[3],
                        chunk_index=row[4],
                        relevance_score=row[5] or 0.0,
                        search_type="semantic",
                    )
                )

            logger.info(f"[search] Found {len(results)} semantic results")
            return results

        except Exception as e:
            logger.error(f"[search] Semantic search error: {e}")
            raise

    @staticmethod
    async def hybrid_search(
        session: AsyncSession,
        case_id: str,
        query: str,
        query_embedding: List[float],
        limit: int = 10,
        threshold: float = 0.5,
        ft_weight: float = 0.5,  # Full-text weight
        semantic_weight: float = 0.5,  # Semantic weight
    ) -> List[SearchResult]:
        """
        Hybrid search combining full-text and semantic search

        Uses weighted combination:
        hybrid_score = (ft_weight * ft_score) + (semantic_weight * semantic_score)

        Args:
            session: AsyncSession
            case_id: Filter by case
            query: Search query text
            query_embedding: Query embedding vector
            limit: Max results to return
            threshold: Minimum similarity threshold for semantic
            ft_weight: Weight for full-text score (0-1)
            semantic_weight: Weight for semantic score (0-1)

        Returns:
            List of SearchResult objects sorted by combined score
        """
        try:
            logger.info(
                f"[search] Hybrid search (ft_weight={ft_weight}, semantic_weight={semantic_weight})"
            )

            # Convert embedding list to pgvector string format
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            # Combine both search methods with weighted scoring
            sql = text(f"""
                WITH ft_results AS (
                    SELECT
                        dc.id,
                        dc.document_id,
                        d.title,
                        dc.chunk_text,
                        dc.chunk_index,
                        ts_rank(dc.search_vector, plainto_tsquery('english', :query)) as ft_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.case_id = :case_id
                    AND dc.search_vector @@ plainto_tsquery('english', :query)
                ),
                semantic_results AS (
                    SELECT
                        dc.id,
                        dc.document_id,
                        d.title,
                        dc.chunk_text,
                        dc.chunk_index,
                        (1 - (dc.embedding <=> '{embedding_str}'::vector)) as semantic_score
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.case_id = :case_id
                    AND dc.embedding IS NOT NULL
                    AND (1 - (dc.embedding <=> '{embedding_str}'::vector)) >= :threshold
                )
                SELECT
                    COALESCE(ft.id, sem.id) as id,
                    COALESCE(ft.document_id, sem.document_id) as document_id,
                    COALESCE(ft.title, sem.title) as title,
                    COALESCE(ft.chunk_text, sem.chunk_text) as chunk_text,
                    COALESCE(ft.chunk_index, sem.chunk_index) as chunk_index,
                    (
                        COALESCE(ft.ft_score, 0) * :ft_weight +
                        COALESCE(sem.semantic_score, 0) * :semantic_weight
                    ) as hybrid_score
                FROM ft_results ft
                FULL OUTER JOIN semantic_results sem ON ft.id = sem.id
                ORDER BY hybrid_score DESC
                LIMIT :limit
            """)

            result = await session.execute(
                sql,
                {
                    "case_id": case_id,
                    "query": query,
                    "threshold": threshold,
                    "ft_weight": ft_weight,
                    "semantic_weight": semantic_weight,
                    "limit": limit,
                },
            )

            results = []
            for row in result:
                results.append(
                    SearchResult(
                        chunk_id=row[0],
                        document_id=row[1],
                        document_title=row[2],
                        chunk_text=row[3],
                        chunk_index=row[4],
                        relevance_score=row[5] or 0.0,
                        search_type="hybrid",
                    )
                )

            logger.info(f"[search] Found {len(results)} hybrid results")
            return results

        except Exception as e:
            logger.error(f"[search] Hybrid search error: {e}")
            raise


search_service = SearchService()
