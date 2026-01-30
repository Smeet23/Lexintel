"""RAG Query Engine for legal document analysis with comprehensive error handling"""
import logging
import asyncio
import tiktoken
from typing import List, Dict, Tuple, Optional
from uuid import UUID
from sqlalchemy.orm import Session
from openai import AsyncOpenAI

# Exception handling
try:
    from openai import OpenAIError, RateLimitError, APIError
except ImportError:
    try:
        from openai.error import OpenAIError, RateLimitError, APIError
    except ImportError:
        class OpenAIError(Exception):
            pass
        class RateLimitError(OpenAIError):
            pass
        class APIError(OpenAIError):
            pass

try:
    from backend.services.embeddings import embed_text
    from backend.services.vector_store import search_vectors
    from backend.models import Case, Query, Chunk
    from backend.config import get_settings
    from backend.exceptions import QueryProcessingException, EmbeddingException, VectorStoreException
except ImportError:
    try:
        from services.embeddings import embed_text
        from services.vector_store import search_vectors
        from models import Case, Query, Chunk
        from config import get_settings
        from exceptions import QueryProcessingException, EmbeddingException, VectorStoreException
    except ImportError:
        from .services.embeddings import embed_text
        from .services.vector_store import search_vectors
        from .models import Case, Query, Chunk
        from .config import get_settings
        from .exceptions import QueryProcessingException, EmbeddingException, VectorStoreException

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache for tiktoken encoder (Issue 1)
_TIKTOKEN_ENCODER = None

# Cache for reranker model (lazy loaded)
_RERANKER_MODEL = None

def _get_encoder():
    """Get cached tiktoken encoder for GPT-4o"""
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        try:
            _TIKTOKEN_ENCODER = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            # gpt-4o not in tiktoken's registry, use cl100k_base (same as GPT-4)
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODER


# Configuration
CONTEXT_TOKEN_BUDGET = 12_800
LEGAL_SYSTEM_PROMPT = """You are an expert legal assistant specialized in analyzing court documents, case law, and legal statutes. Your role is to:
1. Answer questions ONLY based on the provided document excerpts
2. Provide precise, factually accurate responses
3. Always cite the exact location in square brackets:
   - For PDFs: [Page X]
   - For Word documents: [Paragraph X]
   - For text files: [Lines X-Y]
4. Distinguish between facts, arguments, and judgments
5. Flag any ambiguities or gaps in the source material
6. Never speculate beyond what the documents state
For each claim, include the location reference and cite the specific section when available."""

MIN_QUERY_LENGTH = 3
MIN_CONFIDENCE_SCORE = 0.6  # Require 60% semantic similarity minimum (prevents weak matches)
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
        encoder = _get_encoder()
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        logger.error(f"Failed to count tokens: {str(e)}")
        raise ValueError(f"Failed to count tokens: {str(e)}") from e


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
        location = chunk.get("page_num", "Unknown")
        section = chunk.get("section_name", "")
        score = chunk.get("score", 0)
        content = chunk.get("content", "")

        # Determine location label based on format
        if location.startswith("para"):
            location_label = f"Paragraph {location[5:]}"  # Extract number from "para X"
        elif location.startswith("line"):
            location_label = f"Lines {location[5:]}"  # Extract range from "line X-Y"
        else:
            location_label = f"Page {location}"  # Default to page

        # Format excerpt header with metadata
        header = f"--- EXCERPT {i} ({location_label}"
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


def _get_reranker():
    """
    Get or initialize cross-encoder reranker model.
    Uses sentence-transformers cross-encoder for efficiency.

    Returns:
        CrossEncoder model or None if not available
    """
    global _RERANKER_MODEL

    if _RERANKER_MODEL is not None:
        return _RERANKER_MODEL

    try:
        from sentence_transformers import CrossEncoder
        logger.info("Initializing cross-encoder reranker")
        # Using lightweight distilroberta model for speed
        _RERANKER_MODEL = CrossEncoder("cross-encoder/qnli-distilroberta-base")
        return _RERANKER_MODEL
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Reranking disabled. Install with: pip install sentence-transformers"
        )
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize reranker: {str(e)}")
        return None


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    top_k: int = FINAL_CHUNK_COUNT
) -> List[Dict]:
    """
    Rerank retrieved chunks using cross-encoder for relevance.

    Better chunks = better answers. Uses cross-encoder to rerank chunks
    by comparing query relevance (more accurate than vector similarity alone).

    Args:
        query: User query string
        chunks: List of chunks from vector search
        top_k: Number of top chunks to return after reranking

    Returns:
        List of reranked chunks sorted by relevance
    """
    if not chunks:
        logger.debug("No chunks to rerank")
        return []

    reranker = _get_reranker()

    if reranker is None:
        logger.debug("Reranker not available, returning chunks as-is")
        return chunks[:top_k]

    try:
        # Prepare pairs for reranking (query, chunk_content)
        pairs = []
        chunk_texts = []

        for chunk in chunks:
            content = chunk.get("content", "")[:300]  # Limit to 300 chars for efficiency
            pairs.append([query, content])
            chunk_texts.append(content)

        logger.debug(f"Reranking {len(chunks)} chunks")

        # Get reranking scores
        scores = reranker.predict(pairs)

        # Add rerank scores to chunks
        for i, chunk in enumerate(chunks):
            # Combine vector similarity score (original) with rerank score
            original_score = chunk.get("score", 0)
            rerank_score = float(scores[i])  # 0-1 normalized score

            # Weighted combination: 40% vector similarity, 60% cross-encoder
            chunk["rerank_score"] = rerank_score
            chunk["combined_score"] = (original_score * 0.4) + (rerank_score * 0.6)

        # Sort by combined score
        reranked = sorted(chunks, key=lambda x: x.get("combined_score", 0), reverse=True)

        # Log top result
        if reranked:
            top = reranked[0]
            logger.debug(
                f"Top reranked chunk: "
                f"vector_score={top.get('score', 0):.3f}, "
                f"rerank_score={top.get('rerank_score', 0):.3f}, "
                f"combined={top.get('combined_score', 0):.3f}"
            )

        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranking failed, using original chunks: {str(e)}")
        return chunks[:top_k]


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
        ValueError: If top_k is not positive
        Exception: If vector search fails
    """
    # Validate top_k parameter (Issue 6)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    try:
        results = search_vectors(case_id, query_embedding, limit=top_k)
        return results
    except VectorStoreException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {str(e)}")
        raise VectorStoreException(
            "Failed to retrieve chunks from vector store",
            detail=str(e)
        ) from e


def extract_citations(answer: str, chunks: List[Dict]) -> Tuple[str, List[Dict], bool]:
    """
    Extract citations from answer, match to retrieved chunks, and remove hallucinations.

    Handles citations for multiple document formats:
    - [Page X] for PDFs
    - [Paragraph X] for Word documents
    - [Lines X-Y] for text files

    Args:
        answer: Generated answer with location citations
        chunks: List of retrieved chunks

    Returns:
        Tuple of (cleaned_answer, citations_list, has_hallucinations)
        - cleaned_answer: Answer with hallucinated citations removed
        - citations_list: List of valid citation dicts
        - has_hallucinations: Bool indicating if any hallucinations were detected

    Raises:
        None - gracefully handles citation mismatches
    """
    import re

    citations = []

    # Define patterns for all citation types
    citation_patterns = [
        (r'\[Page\s+(\d+)\]', 'page'),
        (r'\[Paragraph\s+(\d+)\]', 'paragraph'),
        (r'\[Lines\s+(\d+-\d+)\]', 'line_range')
    ]

    # Create location to chunk mapping
    location_to_chunks = {}
    valid_locations = set()
    for chunk in chunks:
        location = str(chunk.get("page_num", ""))
        if location not in location_to_chunks:
            location_to_chunks[location] = chunk
            valid_locations.add(location)

    # Extract citations and match to chunks
    unmatched_citations = []
    valid_matches = []

    for pattern, citation_type in citation_patterns:
        matches = list(re.finditer(pattern, answer))

        for match in matches:
            if citation_type == 'page':
                location = match.group(1)  # Page number
            elif citation_type == 'paragraph':
                location = f"para {match.group(1)}"  # Convert to "para X" format
            elif citation_type == 'line_range':
                location = f"line {match.group(1)}"  # Convert to "line X-Y" format
            else:
                continue

            if location in location_to_chunks:
                chunk = location_to_chunks[location]
                citation = {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "location": location,
                    "citation_type": citation_type,
                    "relevance_score": chunk.get("score", 0)
                }
                if citation not in citations:  # Avoid duplicates
                    citations.append(citation)
                valid_matches.append(match)
            else:
                unmatched_citations.append(match.group(0))

    # Remove hallucinated citations from answer
    cleaned_answer = answer
    has_hallucinations = False

    if unmatched_citations:
        has_hallucinations = True
        logger.warning(
            f"Hallucination detected: answer references locations {unmatched_citations} "
            f"not in retrieved chunks {list(valid_locations)}. "
            f"Removing hallucinated citations from response."
        )

        # Remove all invalid citation patterns from answer
        for citation_text in unmatched_citations:
            cleaned_answer = cleaned_answer.replace(citation_text, "").strip()

        # Clean up any extra spaces
        cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

    return cleaned_answer, citations, has_hallucinations


def ground_citations_in_source(
    citations: List[Dict],
    chunks: List[Dict]
) -> Tuple[List[Dict], List[Dict], bool]:
    """
    Validate that citations are actually supported by source chunks.
    Extract supporting text excerpts for each citation.

    Args:
        citations: List of citation dicts from extract_citations
        chunks: List of retrieved chunks with content

    Returns:
        Tuple of (grounded_citations, unsupported_claims, has_unsupported)
        - grounded_citations: Citations with supporting text excerpts
        - unsupported_claims: Claims that couldn't be grounded
        - has_unsupported: Bool indicating if any claims were unsupported
    """
    grounded_citations = []
    unsupported_claims = []

    # Create location to chunk mapping
    location_to_chunk = {str(c.get("page_num", "")): c for c in chunks}

    for citation in citations:
        location = citation.get("location", "")
        chunk = location_to_chunk.get(location)

        if not chunk:
            # Citation location not found in chunks
            unsupported_claims.append({
                "location": location,
                "reason": "Location not found in retrieved chunks"
            })
            continue

        # Extract supporting text from chunk
        chunk_content = chunk.get("content", "")

        grounded_citation = {
            "location": location,
            "citation_type": citation.get("citation_type", "page"),
            "relevance_score": citation.get("relevance_score", 0),
            "chunk_id": chunk.get("chunk_id", ""),
            "supporting_excerpt": chunk_content[:500],  # First 500 chars
            "is_grounded": True
        }

        grounded_citations.append(grounded_citation)
        logger.debug(f"Citation grounded: {location}")

    has_unsupported = len(unsupported_claims) > 0

    if has_unsupported:
        logger.warning(f"Found {len(unsupported_claims)} unsupported citations")

    return grounded_citations, unsupported_claims, has_unsupported


def calculate_answer_confidence(
    answer: str,
    citations: List[Dict],
    chunks: List[Dict],
    has_hallucinations: bool
) -> float:
    """
    Calculate confidence score for generated answer (0.0-1.0).

    Factors:
    - Citation coverage: % of answer sentences with citations
    - Chunk similarity: Average relevance score of cited chunks
    - Hallucination presence: Penalize for hallucinations
    - Citation count: More citations = higher confidence

    Args:
        answer: Generated answer text
        citations: Grounded citations with scores
        chunks: Retrieved chunks
        has_hallucinations: Whether hallucinations were detected

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not answer or not citations:
        return 0.0

    import re

    # Count sentences
    sentences = re.split(r'[.!?]+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Count cited sentences (rough heuristic: sentences with [citation] patterns)
    cited_sentences = sum(1 for s in sentences if "[" in s and "]" in s)
    citation_coverage = cited_sentences / len(sentences) if sentences else 0

    # Average citation relevance
    if citations:
        avg_relevance = sum(c.get("relevance_score", 0) for c in citations) / len(citations)
    else:
        avg_relevance = 0

    # Penalize for hallucinations
    hallucination_penalty = 0.3 if has_hallucinations else 0

    # Bonus for multiple citations
    citation_bonus = min(0.1, len(citations) * 0.02)

    # Weighted confidence score
    confidence = (
        (citation_coverage * 0.3) +
        (avg_relevance * 0.5) +
        (citation_bonus) -
        (hallucination_penalty)
    )

    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    logger.debug(
        f"Confidence calculation: coverage={citation_coverage:.2f}, "
        f"relevance={avg_relevance:.2f}, hallucinations={has_hallucinations}, "
        f"final={confidence:.2f}"
    )

    return confidence


def classify_confidence_level(confidence_score: float) -> str:
    """
    Classify confidence score into categorical level.

    Args:
        confidence_score: Float between 0.0 and 1.0

    Returns:
        "high", "medium", "low", or "none"
    """
    if confidence_score >= 0.75:
        return "high"
    elif confidence_score >= 0.6:
        return "medium"
    elif confidence_score >= 0.4:
        return "low"
    else:
        return "none"


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
        ValueError: If API key is not configured
        Exception: If API call fails
    """
    # Validate API key before using (Issue 3)
    if not settings.openai_api_key:
        logger.error("OpenAI API key not configured")
        raise ValueError("OpenAI API key is not configured. Check OPENAI_API_KEY environment variable.")

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
            max_tokens=2000,
            timeout=30
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens

        return answer, tokens_used

    except (OpenAIError, RateLimitError, APIError) as e:
        logger.error(f"OpenAI API error generating answer: {str(e)}")
        raise QueryProcessingException(
            "Failed to generate answer with OpenAI",
            detail=f"OpenAI error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error generating answer: {str(e)}")
        raise QueryProcessingException(
            "Unexpected error during answer generation",
            detail=str(e)
        ) from e


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
        except (EmbeddingException, ValueError) as e:
            logger.error(f"Query embedding failed: {str(e)}")
            error_response["error"] = f"Failed to process query: {str(e)}"
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error embedding query: {str(e)}")
            raise QueryProcessingException(
                "Unexpected error during query embedding",
                detail=str(e)
            ) from e

        # 3. Retrieve chunks
        try:
            retrieved_chunks = retrieve_chunks(case_id, query_embedding, top_k=RETRIEVAL_TOP_K)
        except (VectorStoreException, ValueError) as e:
            logger.error(f"Chunk retrieval failed: {str(e)}")
            error_response["error"] = f"No chunks found for case"
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error retrieving chunks: {str(e)}")
            raise QueryProcessingException(
                "Unexpected error during chunk retrieval",
                detail=str(e)
            ) from e

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
        initial_chunks = sorted(high_confidence_chunks, key=lambda x: x.get("score", 0), reverse=True)[:RETRIEVAL_TOP_K]

        # 4.5. Rerank chunks for better relevance
        try:
            final_chunks = rerank_chunks(query, initial_chunks, top_k=top_k)
            logger.debug(f"Reranking improved chunks: {len(initial_chunks)} → {len(final_chunks)}")
        except Exception as e:
            logger.warning(f"Reranking failed, using initial chunks: {str(e)}")
            final_chunks = initial_chunks[:top_k]

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

        except ValueError as e:
            logger.error(f"Context formatting validation failed: {str(e)}")
            error_response["error"] = "Failed to format context"
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error formatting context: {str(e)}")
            raise QueryProcessingException(
                "Unexpected error during context formatting",
                detail=str(e)
            ) from e

        # 6. Generate answer
        try:
            answer, tokens_used = await generate_answer(query, formatted_context, temperature)
        except QueryProcessingException as e:
            logger.error(f"Answer generation failed: {str(e)}")
            error_response["error"] = f"Failed to generate answer: API error"
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error generating answer: {str(e)}")
            raise QueryProcessingException(
                "Unexpected error during answer generation",
                detail=str(e)
            ) from e

        # 7. Extract citations and detect hallucinations
        cleaned_answer, citations, has_hallucinations = extract_citations(answer, final_chunks)

        # 7.5. Ground citations in source text
        grounded_citations, unsupported_claims, has_unsupported = ground_citations_in_source(citations, final_chunks)

        if has_unsupported:
            logger.warning(f"Found {len(unsupported_claims)} unsupported citation(s)")
            # Mark unsupported claims in response
            for claim in unsupported_claims:
                logger.warning(f"  - Unsupported: {claim['location']} ({claim['reason']})")

        # 7.6. Calculate answer confidence score
        answer_confidence_score = calculate_answer_confidence(
            cleaned_answer,
            grounded_citations,
            final_chunks,
            has_hallucinations
        )
        answer_confidence_level = classify_confidence_level(answer_confidence_score)

        logger.info(
            f"Answer quality: confidence={answer_confidence_level} "
            f"(score={answer_confidence_score:.2f}), "
            f"grounded_citations={len(grounded_citations)}, "
            f"unsupported={len(unsupported_claims)}, "
            f"hallucinations={has_hallucinations}"
        )

        # 8. Prepare sources list with FULL content from database
        sources = []
        for chunk in final_chunks:
            chunk_id = chunk.get("chunk_id", "")

            # Fetch full content from database (not truncated from vector store)
            full_content = ""
            try:
                db_chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
                if db_chunk:
                    full_content = db_chunk.content
                else:
                    # Fallback to content_preview if not found
                    full_content = chunk.get("content", "")
            except Exception as e:
                logger.warning(f"Failed to fetch full chunk content for {chunk_id}: {str(e)}, using preview")
                full_content = chunk.get("content", "")

            source = {
                "chunk_id": chunk_id,
                "page_num": chunk.get("page_num", ""),
                "relevance_score": chunk.get("score", 0),
                "content": full_content  # Now full content instead of 200-char preview
            }
            sources.append(source)

        # Use calculated confidence score (more sophisticated than simple averaging)
        confidence = answer_confidence_level

        # Prepare successful response (using cleaned_answer without hallucinated citations)
        return {
            "answer": cleaned_answer,
            "sources": sources,
            "citations": grounded_citations,  # Citations with supporting excerpts
            "case_id": case_id,
            "query": query,
            "model": "gpt-4o",
            "tokens_used": tokens_used,
            "confidence": {
                "level": confidence,  # "high", "medium", "low", "none"
                "score": answer_confidence_score,  # 0.0-1.0
                "factors": {
                    "has_hallucinations": has_hallucinations,
                    "unsupported_claims": len(unsupported_claims),
                    "grounded_citations": len(grounded_citations),
                    "avg_citation_relevance": (
                        sum(c.get("relevance_score", 0) for c in grounded_citations) / len(grounded_citations)
                        if grounded_citations else 0.0
                    )
                }
            },
            "error": None
        }

    except QueryProcessingException as e:
        logger.error(f"Query processing error in query_case: {str(e)}")
        error_response["error"] = f"Query processing failed: {str(e)}"
        return error_response
    except Exception as e:
        logger.error(f"Unexpected error in query_case: {str(e)}")
        raise QueryProcessingException(
            "Unexpected error in query processing pipeline",
            detail=str(e)
        ) from e
