"""
RAG (Retrieval-Augmented Generation) service

Orchestrates:
- Document context retrieval
- Prompt building with context
- LLM response generation
"""

import logging
from typing import List, AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.search import search_service
from shared import generate_embeddings_batch
from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """RAG orchestration service"""

    @staticmethod
    async def search_context(
        session: AsyncSession,
        query: str,
        case_id: str,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[dict]:
        """
        Search for relevant document chunks to use as context

        Args:
            session: AsyncSession
            query: User question/query
            case_id: Case to search within
            top_k: Number of chunks to retrieve
            threshold: Minimum similarity threshold

        Returns:
            List of relevant chunks with metadata
        """
        try:
            logger.info(f"[rag] Searching context for: {query}")

            # Generate embedding for query
            query_embeddings = await generate_embeddings_batch([query])
            query_embedding = query_embeddings[0]

            # Use semantic search to find relevant chunks
            results = await search_service.semantic_search(
                session,
                case_id=case_id,
                query_embedding=query_embedding,
                limit=top_k,
                threshold=threshold,
            )

            # Convert results to dict format
            context_chunks = [r.to_dict() for r in results]
            logger.info(f"[rag] Retrieved {len(context_chunks)} context chunks")

            return context_chunks

        except Exception as e:
            logger.error(f"[rag] Context search error: {e}")
            raise

    @staticmethod
    def build_system_prompt(
        query: str,
        context_chunks: List[dict],
    ) -> str:
        """
        Build system prompt with context from documents

        Args:
            query: User question
            context_chunks: Retrieved document chunks

        Returns:
            Complete system prompt for LLM
        """
        # Build context section
        context_text = ""
        if context_chunks:
            context_text = "Based on the following document excerpts:\n\n"
            for i, chunk in enumerate(context_chunks, 1):
                context_text += f"[{i}] From '{chunk['document_title']}' (chunk {chunk['chunk_index']}):\n"
                context_text += f"{chunk['chunk_text']}\n\n"
        else:
            context_text = "No relevant documents found.\n\n"

        # Build full prompt
        prompt = f"""You are a legal research assistant. Answer the user's question based on the provided document excerpts.

{context_text}

User Question: {query}

Instructions:
- Base your answer on the provided documents
- If the documents don't contain relevant information, say "Information not found in documents"
- Cite the specific document sections you're referencing
- Be clear and professional in your response"""

        return prompt

    @staticmethod
    async def stream_response(
        prompt: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream response from OpenAI

        Args:
            prompt: Complete prompt for LLM

        Yields:
            Token chunks from LLM response
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

            logger.info("[rag] Calling OpenAI API for streaming response")

            stream = await client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL or "gpt-4o",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"[rag] LLM streaming error: {e}")
            raise

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Estimate token count (simplified)

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4


rag_service = RAGService()
