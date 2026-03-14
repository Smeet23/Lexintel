"""RAG Query Engine for legal document analysis with comprehensive error handling"""
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from uuid import UUID
from sqlalchemy.orm import Session
import google.generativeai as genai

try:
    from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
except ImportError:
    class GoogleAPIError(Exception):
        pass
    class ResourceExhausted(GoogleAPIError):
        pass

try:
    from backend.services.embeddings import embed_text, embed_query as embed_query_fn
    from backend.services.vector_store import search_vectors
    from backend.services.document_summary import generate_document_summary
    from backend.models import Matter, Document, Query, Chunk
    from backend.config import get_settings
    from backend.exceptions import QueryProcessingException, EmbeddingException, VectorStoreException
except ImportError:
    try:
        from services.embeddings import embed_text, embed_query as embed_query_fn
        from services.vector_store import search_vectors
        from services.document_summary import generate_document_summary
        from models import Matter, Document, Query, Chunk
        from config import get_settings
        from exceptions import QueryProcessingException, EmbeddingException, VectorStoreException
    except ImportError:
        from .embeddings import embed_text, embed_query as embed_query_fn
        from .vector_store import search_vectors
        from .document_summary import generate_document_summary
        from ..models import Matter, Document, Query, Chunk
        from ..config import get_settings
        from ..exceptions import QueryProcessingException, EmbeddingException, VectorStoreException

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache for reranker model (lazy loaded)
_RERANKER_MODEL = None

# Gemini model name
GEMINI_MODEL = settings.gemini_model

# Configuration
CONTEXT_TOKEN_BUDGET = 50_000
LEGAL_SYSTEM_PROMPT = """You are an expert legal assistant specialized in analyzing court documents, case law, and legal statutes. Your role is to:
1. Answer questions ONLY based on the provided document excerpts
2. Provide precise, factually accurate responses
3. Always cite the exact location in square brackets:
   - For PDFs: [Page X]
   - For Word documents: [Paragraph X]
   - For text files: [Lines X-Y]
   - For named sections: [Section "Section Name"]
4. Distinguish between facts, arguments, and judgments
5. Flag any ambiguities or gaps in the source material
6. Never speculate beyond what the documents state
For each claim, include the location reference. When a named section is available in the excerpt metadata, use [Section "Name"] citations for precise referencing.
When conversation history is provided, use it to resolve references like "that", "it", "the above", etc. Answer the current question based on the document excerpts, using conversation context only for disambiguation."""

MIN_QUERY_LENGTH = 3
MIN_CONFIDENCE_SCORE = 0.30  # Require 30% semantic similarity minimum (tuned for Cohere embed-v3)
RETRIEVAL_TOP_K = 15
RETRIEVAL_LIMIT = 30  # Request more from vector store, then take top_k (improves recall)
FINAL_CHUNK_COUNT = 8


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count for text (roughly 4 characters per token).

    Args:
        text: Text to count tokens for

    Returns:
        Estimated number of tokens

    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # Rough estimate: 1 token ~ 4 characters
    return len(text) // 4


def _detect_query_filters(query: str) -> Dict:
    """Detect optional Qdrant filters from query text.

    Only applies explicit jurisdiction mentions. Returns empty dict
    if no strong signal is detected (empty dict = no filters applied).
    """
    filters = {}
    query_lower = query.lower()

    jurisdiction_hints = {
        "UK": ["uk ", "united kingdom", "english law", "british"],
        "US": ["us ", "united states", "american", "federal law"],
        "EU": ["eu ", "european union", "gdpr", "directive"],
        "AU": ["australia", "australian"],
        "CA": ["canada", "canadian"],
        "SG": ["singapore", "singaporean"],
        "IN": ["india", "indian law"],
    }
    for code, hints in jurisdiction_hints.items():
        if any(h in query_lower for h in hints):
            filters["jurisdiction"] = code
            break

    return filters


def format_legal_context(chunks: List[Dict], matter_name: str, doc_summaries: Dict = None) -> str:
    """
    Format retrieved chunks into structured legal context with metadata.

    Args:
        chunks: List of chunk dicts with content, page_num, section_name, score
        matter_name: Name of the matter
        doc_summaries: Optional dict mapping document_id -> {"name": str, "summary": str}

    Returns:
        Formatted context string with metadata

    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    # Sort by score (highest first)
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    context_parts = [f"Matter: {matter_name}\n", "=" * 60, "\n"]

    # Add document summaries as preamble (once per document, not per chunk)
    if doc_summaries:
        context_parts.append("Document Summaries:\n")
        for doc_id, info in doc_summaries.items():
            context_parts.append(f"  - {info['name']}: {info['summary']}\n")
        context_parts.append("\n")

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
        doc_name = chunk.get("document_name", "")
        if doc_name:
            header += f", Document: {doc_name}"
        header += f", Score: {score:.2f}) ---\n"

        context_parts.append(header)
        context_parts.append(content)
        context_parts.append("\n\n")

    return "".join(context_parts)


def embed_query(query: str) -> List[float]:
    """
    Embed user query into vector space (uses Cohere search_query input type).

    Args:
        query: User query string

    Returns:
        1024-dimensional embedding vector
    """
    return embed_query_fn(query)


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
        _RERANKER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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

        for chunk in chunks:
            content = chunk.get("content", "")[:512]  # Cross-encoder handles up to ~512 tokens
            pairs.append([query, content])

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


def retrieve_chunks(matter_id: str, query_embedding: List[float], top_k: int = RETRIEVAL_TOP_K, query_filter: Dict = None) -> List[Dict]:
    """
    Retrieve similar chunks from vector store.

    Args:
        matter_id: Matter identifier
        query_embedding: Query embedding vector
        top_k: Number of top results to retrieve
        query_filter: Optional payload filter dict for Qdrant

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
        results = search_vectors(matter_id, query_embedding, limit=top_k, query_filter=query_filter)
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
        (r'\[Page\s+(\d+)(?:,\s*Section[:\s]+[^\]]+)?\]', 'page'),  # [Page 3] or [Page 3, Section: 4.1]
        (r'\[Paragraph\s+(\d+)\]', 'paragraph'),
        (r'\[Lines?\s+(\d+-\d+)(?:,\s*Section[:\s]+[^\]]+)?\]', 'line_range'),  # [Lines 1-5] or [Lines 1-5, Section: 9]
        (r'\[Section[:\s]+"?([^"\]]+)"?\]', 'section'),
    ]

    # Fallback patterns for less structured LLM citations
    fallback_patterns = [
        (r'(?:on |see |from |per )page\s+(\d+)', 'page'),
        (r'(?:in |see |per )section\s+(\d+(?:\.\d+)*)', 'section'),
        (r'(?:in |see |per )paragraph\s+(\d+)', 'paragraph'),
    ]

    def _line_range_overlaps(cited_range: str, chunk_range: str) -> bool:
        """Check if a cited line range (e.g. '1-5') overlaps with a chunk range (e.g. 'line 1-50')."""
        try:
            # Parse cited range: "1-5" or "101-148"
            cited_parts = cited_range.split("-")
            cited_start, cited_end = int(cited_parts[0]), int(cited_parts[1])
            # Parse chunk range: "line 1-50" or "line 51-100"
            chunk_nums = chunk_range.replace("line ", "").split("-")
            chunk_start, chunk_end = int(chunk_nums[0]), int(chunk_nums[1])
            return cited_start <= chunk_end and cited_end >= chunk_start
        except (ValueError, IndexError):
            return False

    # Create location to chunk mapping
    location_to_chunks = {}
    section_to_chunks = {}
    valid_locations = set()
    line_range_chunks = []  # For overlap matching
    for chunk in chunks:
        location = str(chunk.get("page_num", ""))
        if location not in location_to_chunks:
            location_to_chunks[location] = chunk
            valid_locations.add(location)
        # Track line-range chunks for overlap matching
        if location.startswith("line "):
            line_range_chunks.append((location, chunk))
        # Also map section names for [Section "X"] citations
        section_name = str(chunk.get("section_name", ""))
        if section_name and section_name not in section_to_chunks:
            section_to_chunks[section_name] = chunk

    def _find_chunk_for_citation(location: str, citation_type: str):
        """Find the best matching chunk for a citation, using overlap for line ranges."""
        # Exact match first
        if location in location_to_chunks:
            return location_to_chunks[location]
        # For line ranges, check overlap with chunk ranges
        if citation_type == 'line_range' and location.startswith("line "):
            cited_nums = location.replace("line ", "")
            best_chunk = None
            best_score = 0
            for chunk_loc, chunk in line_range_chunks:
                if _line_range_overlaps(cited_nums, chunk_loc):
                    score = chunk.get("score", 0)
                    if score > best_score:
                        best_score = score
                        best_chunk = chunk
            return best_chunk
        return None

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
            elif citation_type == 'section':
                location = match.group(1).strip()  # Section name
            else:
                continue

            # Match section citations against section_name metadata
            if citation_type == 'section' and location in section_to_chunks:
                chunk = section_to_chunks[location]
                citation = {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "location": location,
                    "citation_type": citation_type,
                    "relevance_score": chunk.get("score", 0)
                }
                if citation not in citations:
                    citations.append(citation)
                valid_matches.append(match)
                continue

            matched_chunk = _find_chunk_for_citation(location, citation_type)
            if matched_chunk:
                citation = {
                    "chunk_id": matched_chunk.get("chunk_id", ""),
                    "location": location,
                    "citation_type": citation_type,
                    "relevance_score": matched_chunk.get("score", 0)
                }
                if citation not in citations:  # Avoid duplicates
                    citations.append(citation)
                valid_matches.append(match)
            else:
                unmatched_citations.append(match.group(0))

    # Try fallback patterns if no citations found via primary patterns
    if not citations:
        for pattern, citation_type in fallback_patterns:
            matches = list(re.finditer(pattern, answer, re.IGNORECASE))
            for match in matches:
                if citation_type == 'page':
                    location = match.group(1)
                elif citation_type == 'paragraph':
                    location = f"para {match.group(1)}"
                elif citation_type == 'section':
                    location = match.group(1)
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
                    if citation not in citations:
                        citations.append(citation)
                    valid_matches.append(match)

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


def _calculate_citation_coverage(answer: str, citations: List[Dict]) -> float:
    """Calculate percentage of answer with citation support (0.0-1.0)"""
    if not answer or not citations:
        return 0.0

    import re
    sentences = re.split(r'[.!?]+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    cited_sentences = 0
    for sentence in sentences:
        for citation in citations:
            location = citation.get("location", "")
            if location and location in sentence:
                cited_sentences += 1
                break

    return min(1.0, cited_sentences / len(sentences))


def _calculate_average_relevance(citations: List[Dict]) -> float:
    """Calculate average relevance score of citations (0.0-1.0)"""
    if not citations:
        return 0.0

    scores = [c.get("relevance_score", 0.0) for c in citations]
    return sum(scores) / len(scores) if scores else 0.0


def _format_relevance_level(score: float) -> str:
    """Convert relevance score to descriptive level"""
    if score >= 0.85:
        return "excellent"
    elif score >= 0.70:
        return "good"
    elif score >= 0.55:
        return "moderate"
    else:
        return "weak"


def _generate_confidence_summary(
    overall_score: float,
    citation_coverage: float,
    avg_relevance: float,
    citation_count: int,
    has_hallucinations: bool
) -> str:
    """Generate human-readable summary of confidence factors"""
    rating = "high" if overall_score >= 0.75 else "medium" if overall_score >= 0.60 else "low"

    factors = []

    if citation_coverage > 0.8:
        factors.append("strong citation coverage")
    elif citation_coverage > 0.5:
        factors.append("partial citation coverage")
    else:
        factors.append("limited citation coverage")

    if avg_relevance >= 0.85:
        factors.append("highly relevant sources")
    elif avg_relevance >= 0.70:
        factors.append("moderately relevant sources")
    else:
        factors.append("weak source relevance")

    if citation_count >= 3:
        factors.append("multiple supporting citations")
    elif citation_count > 0:
        factors.append("single source citation")

    if has_hallucinations:
        factors.append("potential unsupported claims")

    if rating == "high":
        return f"High confidence answer with {', '.join(factors)}."
    elif rating == "medium":
        return f"Medium confidence answer with {', '.join(factors)}."
    else:
        return f"Low confidence answer due to {', '.join(factors)}."


def explain_confidence_score(
    answer: str,
    citations: List[Dict],
    has_hallucinations: bool,
    confidence_score: float
) -> Dict:
    """
    Generate structured explanation for confidence score with factor breakdown.

    Args:
        answer: Generated answer text
        citations: Grounded citations with scores
        has_hallucinations: Whether hallucinations were detected
        confidence_score: Overall confidence score (0.0-1.0)

    Returns:
        Dict with overall_score, rating, factors, and summary
    """
    # Calculate individual factors
    citation_coverage = _calculate_citation_coverage(answer, citations)
    avg_relevance = _calculate_average_relevance(citations)
    citation_count = len(citations)

    # Hallucination risk (1.0 if no hallucinations, lower if hallucinations present)
    hallucination_risk = 0.0 if has_hallucinations else 1.0

    # Citation quantity factor (scales with number of citations)
    citation_quantity_score = min(1.0, citation_count / 3.0)

    # Determine rating
    if confidence_score >= 0.75:
        rating = "high"
    elif confidence_score >= 0.60:
        rating = "medium"
    else:
        rating = "low"

    # Build factor explanations
    factors = {
        "citation_coverage": {
            "score": citation_coverage,
            "explanation": f"{int(citation_coverage * 100)}% of answer is supported by citations"
        },
        "source_relevance": {
            "score": avg_relevance,
            "explanation": f"Source relevance is {_format_relevance_level(avg_relevance)}"
        },
        "hallucination_risk": {
            "score": hallucination_risk,
            "explanation": "Potential unsupported claims detected" if has_hallucinations else "No unsupported claims detected"
        },
        "citation_quantity": {
            "score": citation_quantity_score,
            "explanation": f"{citation_count} supporting citations provided"
        }
    }

    # Generate summary
    summary = _generate_confidence_summary(
        confidence_score,
        citation_coverage,
        avg_relevance,
        citation_count,
        has_hallucinations
    )

    return {
        "overall_score": confidence_score,
        "rating": rating,
        "factors": factors,
        "summary": summary
    }


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

    # Count logical sentences — split on sentence-ending punctuation that is NOT inside brackets
    # Remove bracketed citations first to get clean sentence boundaries, then check originals
    clean_for_splitting = re.sub(r'\[[^\]]*\]', ' [CITE] ', answer.strip())
    sentences = re.split(r'(?<=[.!?])\s+', clean_for_splitting)
    sentences = [s.strip() for s in sentences if s.strip() and s.strip() != '[CITE]']

    if not sentences:
        return 0.0

    # For citation coverage, check how many sentences in the ORIGINAL answer have citations nearby
    # Split original answer the same way and check for bracket patterns
    original_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', answer.strip())
    original_sentences = [s.strip() for s in original_sentences if s.strip()]
    if not original_sentences:
        original_sentences = sentences

    citation_pattern = re.compile(r'\[')  # Any bracket citation in proximity
    cited_sentences = sum(1 for s in original_sentences if citation_pattern.search(s))
    citation_coverage = cited_sentences / len(original_sentences) if original_sentences else 0

    # Average citation relevance
    if citations:
        avg_relevance = sum(c.get("relevance_score", 0) for c in citations) / len(citations)
    else:
        avg_relevance = 0

    # Penalize for hallucinations
    hallucination_penalty = 0.2 if has_hallucinations else 0

    # Bonus for multiple citations (up to 0.15)
    citation_bonus = min(0.15, len(citations) * 0.03)

    # Recalibrated weights for realistic distribution
    # avg_relevance typically 0.6-0.8, citation_coverage typically 0.1-0.5
    confidence = (
        (citation_coverage * 0.25) +
        (avg_relevance * 0.45) +
        (citation_bonus) +
        0.15  # Base score for having any grounded response
    ) - hallucination_penalty

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


def _get_gemini_model():
    """Configure and return Gemini model instance."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    genai.configure(api_key=settings.google_api_key)

    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=LEGAL_SYSTEM_PROMPT,
    )


async def generate_answer(
    query: str,
    context: str,
    temperature: float = 0.2,
    conversation_history: list = None
) -> Tuple[str, int]:
    """
    Generate answer using Google Gemini API.

    Args:
        query: User query
        context: Formatted context from retrieval
        temperature: Temperature parameter (0.2 for legal precision)
        conversation_history: Optional list of previous Q&A turns for follow-up support

    Returns:
        Tuple of (answer, tokens_used)

    Raises:
        ValueError: If API key is not configured
        QueryProcessingException: If API call fails
    """
    try:
        model = _get_gemini_model()

        # Build prompt with conversation history for follow-up support
        prompt_parts = [f"Context:\n{context}\n"]

        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for turn in conversation_history[-5:]:  # Last 5 exchanges max
                prompt_parts.append(f"User: {turn['question']}")
                answer_preview = turn['answer'][:500] if turn.get('answer') else ''
                prompt_parts.append(f"Assistant: {answer_preview}")
            prompt_parts.append("")

        prompt_parts.append(f"Question: {query}")
        prompt = "\n".join(prompt_parts)

        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2000,
            ),
        )

        answer = response.text
        tokens_used = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)

        return answer, tokens_used

    except ValueError:
        raise
    except (ResourceExhausted, GoogleAPIError) as e:
        logger.error(f"Google AI API error generating answer: {str(e)}")
        raise QueryProcessingException(
            "Failed to generate answer with Google AI",
            detail=f"Google AI error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error generating answer: {str(e)}")
        raise QueryProcessingException(
            "Unexpected error during answer generation",
            detail=str(e)
        ) from e


async def query_matter(
    matter_id: str,
    query: str,
    db: Session,
    top_k: int = FINAL_CHUNK_COUNT,
    temperature: float = 0.2,
    conversation_history: list = None
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
        matter_id: Matter identifier (UUID string)
        query: User query string
        db: Database session
        top_k: Number of chunks to include in context
        temperature: LLM temperature for response

    Returns:
        Dict with keys:
        - answer: Generated answer string (or None if error)
        - sources: List of source dicts with citation info
        - matter_id: Matter identifier
        - query: Original query
        - model: Model name used
        - tokens_used: Total tokens consumed
        - confidence: "high|medium|low|none"
        - error: Error message (or None)
    """
    error_response = {
        "answer": None,
        "sources": [],
        "matter_id": matter_id,
        "query": query,
        "model": GEMINI_MODEL,
        "tokens_used": 0,
        "confidence": {"level": "none", "score": 0.0, "factors": {}},
        "error": None
    }

    # 1. Validate query
    if not query or len(query.strip()) < MIN_QUERY_LENGTH:
        error_response["error"] = f"Query must be at least {MIN_QUERY_LENGTH} characters"
        return error_response

    try:
        # Get matter from database
        matter = db.query(Matter).filter(Matter.id == UUID(matter_id)).first()
        if not matter:
            error_response["error"] = f"Matter not found: {matter_id}"
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

        # 3. Retrieve chunks (request more than top_k to improve recall, then take best)
        # Detect optional filters from query text (jurisdiction, doc type)
        query_filters = _detect_query_filters(query)
        try:
            retrieved_chunks = retrieve_chunks(
                matter_id, query_embedding,
                top_k=RETRIEVAL_LIMIT,
                query_filter=query_filters if query_filters else None
            )
            # Fallback: if filtered search returns too few results, retry unfiltered
            if query_filters and len(retrieved_chunks) < 3:
                logger.info(
                    f"Filtered search returned {len(retrieved_chunks)} results "
                    f"(filter: {query_filters}), retrying unfiltered"
                )
                retrieved_chunks = retrieve_chunks(matter_id, query_embedding, top_k=RETRIEVAL_LIMIT)
        except (VectorStoreException, ValueError) as e:
            logger.error(f"Chunk retrieval failed: {str(e)}")
            error_response["error"] = f"No chunks found for matter"
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

        # Log scores for debugging
        scores = [c.get("score", 0) for c in retrieved_chunks]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks, scores: {[f'{s:.4f}' for s in scores]}")

        # 4. Filter by confidence and select top k; fallback to best available if all below threshold
        high_confidence_chunks = [c for c in retrieved_chunks if c.get("score", 0) >= MIN_CONFIDENCE_SCORE]
        low_confidence_fallback = False
        retrieval_avg_score = sum(scores) / len(scores) if scores else 0.0

        if not high_confidence_chunks:
            # Use best available chunks and mark as low confidence (still answer the question)
            low_confidence_fallback = True
            logger.warning(
                f"Low confidence retrieval - average score: {retrieval_avg_score:.2f}; "
                "using top chunks and marking answer as low confidence"
            )
            initial_chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0), reverse=True)[:RETRIEVAL_TOP_K]
        else:
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
            # Fetch document summaries for retrieved chunks
            doc_ids = {UUID(c["document_id"]) for c in final_chunks if c.get("document_id")}
            doc_summaries = {}
            if doc_ids:
                docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
                doc_summaries = {
                    str(d.id): {"name": d.name, "summary": d.summary}
                    for d in docs if d.summary
                }

            formatted_context = format_legal_context(final_chunks, matter.name, doc_summaries)

            # Count tokens in context (estimate)
            context_tokens = count_tokens_estimate(formatted_context)
            query_tokens = count_tokens_estimate(query)

            # Check token budget (reserve buffer for response)
            estimated_total = context_tokens + query_tokens + 500  # 500 token buffer
            if estimated_total > CONTEXT_TOKEN_BUDGET:
                # Trim context to fewer chunks
                final_chunks = final_chunks[:2]
                formatted_context = format_legal_context(final_chunks, matter.name, doc_summaries)
                context_tokens = count_tokens_estimate(formatted_context)
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
            answer, tokens_used = await generate_answer(query, formatted_context, temperature, conversation_history)
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

        # 7.6. Calculate answer confidence score (or use retrieval-based score when fallback)
        if low_confidence_fallback:
            answer_confidence_score = max(0.0, min(1.0, retrieval_avg_score))
            answer_confidence_level = "low"
        else:
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

        # 8. Prepare sources list with FULL content from database (batch fetch)
        sources = []
        # Batch-fetch all chunk content in a single query
        chunk_uuids = []
        for chunk in final_chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id:
                try:
                    chunk_uuids.append(UUID(chunk_id))
                except (ValueError, AttributeError):
                    pass

        db_chunks_map = {}
        if chunk_uuids:
            db_chunks = db.query(Chunk).filter(Chunk.id.in_(chunk_uuids)).all()
            db_chunks_map = {str(c.id): c for c in db_chunks}

        for chunk in final_chunks:
            chunk_id = chunk.get("chunk_id", "")
            db_chunk = db_chunks_map.get(chunk_id)
            full_content = db_chunk.content if db_chunk else chunk.get("content", "")

            source = {
                "chunk_id": chunk_id,
                "page_num": chunk.get("page_num", ""),
                "section_name": chunk.get("section_name", ""),
                "relevance_score": chunk.get("score", 0),
                "content": full_content,
                "document_id": chunk.get("document_id", ""),
                "document_name": chunk.get("document_name", ""),
            }
            sources.append(source)

        # Use calculated confidence score (more sophisticated than simple averaging)
        confidence = answer_confidence_level

        # Generate confidence explanation (NEW)
        confidence_explanation = explain_confidence_score(
            answer=cleaned_answer,
            citations=grounded_citations,
            has_hallucinations=has_hallucinations,
            confidence_score=answer_confidence_score
        )

        # Generate document summary (NEW)
        doc_summary = generate_document_summary(matter)

        # Prepare successful response (using cleaned_answer without hallucinated citations)
        return {
            "answer": cleaned_answer,
            "sources": sources,
            "citations": grounded_citations,  # Citations with supporting excerpts
            "matter_id": matter_id,
            "query": query,
            "model": GEMINI_MODEL,
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
            "confidence_explanation": confidence_explanation,
            "source_document": doc_summary,  # NEW FIELD
            "error": None
        }

    except QueryProcessingException as e:
        logger.error(f"Query processing error in query_matter: {str(e)}")
        error_response["error"] = f"Query processing failed: {str(e)}"
        return error_response
    except Exception as e:
        logger.error(f"Unexpected error in query_matter: {str(e)}")
        raise QueryProcessingException(
            "Unexpected error in query processing pipeline",
            detail=str(e)
        ) from e
