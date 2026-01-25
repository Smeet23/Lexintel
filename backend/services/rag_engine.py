"""RAG Query Engine for legal document analysis with comprehensive error handling"""
import logging
import asyncio
import tiktoken
from typing import List, Dict, Tuple, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from backend.services.embeddings import embed_text
from backend.services.vector_store import search_vectors
from backend.models import Case, Query
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Configuration
CONTEXT_TOKEN_BUDGET = 12_800
LEGAL_SYSTEM_PROMPT = """You are an expert legal assistant specialized in analyzing court documents, case law, and legal statutes. Your role is to: 1. Answer questions ONLY based on the provided document excerpts 2. Provide precise, factually accurate responses 3. Always cite the exact page in square brackets [Page X] 4. Distinguish between facts, arguments, and judgments 5. Flag any ambiguities or gaps in the source material 6. Never speculate beyond what the documents state. For each claim, include the page number [Page X] and reference the specific section when available."""

MIN_QUERY_LENGTH = 3
MIN_CONFIDENCE_SCORE = 0.7
RETRIEVAL_TOP_K = 10
FINAL_CHUNK_COUNT = 4


def count_tokens_gpt4o(text: str) -> int:
    """
    Count tokens in text using tiktoken for GPT-4o model.

    Args:
        text: Text to count tokens for

    Returns:
        Integer number of tokens

    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    try:
        encoder = tiktoken.encoding_for_model("gpt-4o")
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"Failed to count tokens: {str(e)}")
        raise


def validate_token_budget(token_count: int, budget: int) -> bool:
    """
    Validate if token count is within budget.

    Args:
        token_count: Current token count
        budget: Token budget limit

    Returns:
        True if within budget, False otherwise
    """
    return token_count <= budget


def format_legal_context(chunks: List[Dict], case_name: str) -> str:
    """
    Format retrieved chunks into structured legal context with metadata.

    Args:
        chunks: List of chunk dicts with content, page_num, section_name, score
        case_name: Name of the case

    Returns:
        Formatted context string with metadata

    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    # Sort by score (highest first)
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    context_parts = [f"Case: {case_name}\n", "=" * 60, "\n"]

    for i, chunk in enumerate(sorted_chunks, 1):
        page_num = chunk.get("page_num", "Unknown")
        section = chunk.get("section_name", "")
        score = chunk.get("score", 0)
        content = chunk.get("content", "")

        # Format excerpt header with metadata
        header = f"--- EXCERPT {i} (Page {page_num}"
        if section:
            header += f", Section: {section}"
        header += f", Score: {score:.2f}) ---\n"

        context_parts.append(header)
        context_parts.append(content)
        context_parts.append("\n\n")

    return "".join(context_parts)


def embed_query(query: str) -> List[float]:
    """
    Embed user query into vector space.

    Args:
        query: User query string

    Returns:
        3072-dimensional embedding vector

    Raises:
        ValueError: If query is empty
        Exception: If embedding fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    return embed_text(query)


def retrieve_chunks(case_id: str, query_embedding: List[float], top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
    """
    Retrieve similar chunks from vector store.

    Args:
        case_id: Case identifier
        query_embedding: Query embedding vector
        top_k: Number of top results to retrieve

    Returns:
        List of chunk dicts with score, page_num, content, etc.

    Raises:
        Exception: If vector search fails
    """
    try:
        results = search_vectors(case_id, query_embedding, limit=top_k)
        return results
    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {str(e)}")
        raise


def extract_citations(answer: str, chunks: List[Dict]) -> List[Dict]:
    """
    Extract citations from answer and match to retrieved chunks.

    Args:
        answer: Generated answer with [Page X] citations
        chunks: List of retrieved chunks

    Returns:
        List of citation dicts with chunk_id, page_num, relevance_score

    Raises:
        None - gracefully handles citation mismatches
    """
    import re

    citations = []
    page_pattern = r'\[Page\s+(\d+)\]'
    matches = re.finditer(page_pattern, answer)

    # Create page number to chunk mapping
    page_to_chunks = {}
    for chunk in chunks:
        page_num = str(chunk.get("page_num", ""))
        if page_num not in page_to_chunks:
            page_to_chunks[page_num] = chunk

    # Extract citations and match to chunks
    for match in matches:
        page_num = match.group(1)
        if page_num in page_to_chunks:
            chunk = page_to_chunks[page_num]
            citation = {
                "chunk_id": chunk.get("chunk_id", ""),
                "page_num": page_num,
                "relevance_score": chunk.get("score", 0)
            }
            if citation not in citations:  # Avoid duplicates
                citations.append(citation)

    return citations


async def generate_answer(
    query: str,
    context: str,
    temperature: float = 0.2
) -> Tuple[str, int]:
    """
    Generate answer using OpenAI ChatCompletion API.

    Args:
        query: User query
        context: Formatted context from retrieval
        temperature: Temperature parameter (0.2 for legal precision)

    Returns:
        Tuple of (answer, tokens_used)

    Raises:
        Exception: If API call fails
    """
    try:
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        messages = [
            {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            max_tokens=2000
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return answer, tokens_used

    except Exception as e:
        logger.error(f"Failed to generate answer: {str(e)}")
        raise


async def query_case(
    case_id: str,
    query: str,
    db: Session,
    top_k: int = FINAL_CHUNK_COUNT,
    temperature: float = 0.2
) -> Dict:
    """
    Main RAG query orchestration function.

    Implements full RAG pipeline:
    1. Validate query
    2. Embed query
    3. Retrieve similar chunks
    4. Filter by confidence
    5. Format context
    6. Generate answer with LLM
    7. Extract citations
    8. Return structured response

    Args:
        case_id: Case identifier (UUID string)
        query: User query string
        db: Database session
        top_k: Number of chunks to include in context
        temperature: LLM temperature for response

    Returns:
        Dict with keys:
        - answer: Generated answer string (or None if error)
        - sources: List of source dicts with citation info
        - case_id: Case identifier
        - query: Original query
        - model: Model name used
        - tokens_used: Total tokens consumed
        - confidence: "high|medium|low|none"
        - error: Error message (or None)
    """
    error_response = {
        "answer": None,
        "sources": [],
        "case_id": case_id,
        "query": query,
        "model": "gpt-4o",
        "tokens_used": 0,
        "confidence": "none",
        "error": None
    }

    # 1. Validate query
    if not query or len(query.strip()) < MIN_QUERY_LENGTH:
        error_response["error"] = f"Query must be at least {MIN_QUERY_LENGTH} characters"
        return error_response

    try:
        # Get case from database
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            error_response["error"] = f"Case not found: {case_id}"
            return error_response

        # 2. Embed query
        try:
            query_embedding = embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            error_response["error"] = f"Failed to process query: {str(e)}"
            return error_response

        # 3. Retrieve chunks
        try:
            retrieved_chunks = retrieve_chunks(case_id, query_embedding, top_k=RETRIEVAL_TOP_K)
        except Exception as e:
            logger.error(f"Chunk retrieval failed: {str(e)}")
            error_response["error"] = f"No chunks found for case"
            return error_response

        # Check for empty retrieval
        if not retrieved_chunks:
            error_response["error"] = "No relevant documents found"
            return error_response

        # 4. Filter by confidence and select top k
        high_confidence_chunks = [c for c in retrieved_chunks if c.get("score", 0) >= MIN_CONFIDENCE_SCORE]

        if not high_confidence_chunks:
            # All chunks are low confidence
            error_response["confidence"] = "low"
            error_response["error"] = "Retrieved documents have low relevance"
            # Include scores in response
            scores = [c.get("score", 0) for c in retrieved_chunks]
            if scores:
                avg_score = sum(scores) / len(scores)
                logger.warning(f"Low confidence retrieval - average score: {avg_score:.2f}")
            return error_response

        # Use top k chunks (sorted by score)
        final_chunks = sorted(high_confidence_chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

        # 5. Format context with token budgeting
        try:
            formatted_context = format_legal_context(final_chunks, case.name)

            # Count tokens in context
            context_tokens = count_tokens_gpt4o(formatted_context)
            query_tokens = count_tokens_gpt4o(query)

            # Check token budget (reserve buffer for response)
            estimated_total = context_tokens + query_tokens + 500  # 500 token buffer
            if estimated_total > CONTEXT_TOKEN_BUDGET:
                # Trim context to fewer chunks
                final_chunks = final_chunks[:2]
                formatted_context = format_legal_context(final_chunks, case.name)
                context_tokens = count_tokens_gpt4o(formatted_context)
                estimated_total = context_tokens + query_tokens + 500

                if estimated_total > CONTEXT_TOKEN_BUDGET:
                    error_response["error"] = "Context too large for processing"
                    return error_response

        except Exception as e:
            logger.error(f"Context formatting failed: {str(e)}")
            error_response["error"] = "Failed to format context"
            return error_response

        # 6. Generate answer
        try:
            answer, tokens_used = await generate_answer(query, formatted_context, temperature)
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            error_response["error"] = f"Failed to generate answer: API error"
            return error_response

        # 7. Extract citations
        citations = extract_citations(answer, final_chunks)

        # 8. Prepare sources list
        sources = []
        for chunk in final_chunks:
            source = {
                "chunk_id": chunk.get("chunk_id", ""),
                "page_num": chunk.get("page_num", ""),
                "relevance_score": chunk.get("score", 0),
                "content_preview": chunk.get("content_preview", "")[:200] if chunk.get("content_preview") else ""
            }
            sources.append(source)

        # Determine confidence level
        scores = [c.get("score", 0) for c in final_chunks]
        avg_score = sum(scores) / len(scores) if scores else 0

        if avg_score >= 0.9:
            confidence = "high"
        elif avg_score >= 0.8:
            confidence = "medium"
        else:
            confidence = "low"

        # Prepare successful response
        return {
            "answer": answer,
            "sources": sources,
            "case_id": case_id,
            "query": query,
            "model": "gpt-4o",
            "tokens_used": tokens_used,
            "confidence": confidence,
            "error": None
        }

    except Exception as e:
        logger.error(f"Unexpected error in query_case: {str(e)}")
        error_response["error"] = f"Unexpected error: {str(e)}"
        return error_response
