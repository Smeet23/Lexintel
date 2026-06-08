from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # Database
    database_url: str

    # Google AI (Gemini) - used for RAG LLM
    google_api_key: str

    # Cohere - used for embeddings (fallback when Voyage not available)
    cohere_api_key: str

    # Voyage AI - legal-specific embeddings (preferred when key is set)
    # Get free key at https://voyageai.com (50M free tokens)
    voyage_api_key: str = ""

    # Embedding re-index target model: the canonical model name re-index tooling
    # migrates matters/chunks toward. Stored on Matter/Chunk.embedding_model and
    # reported by /admin/embedding-status. Not hardcoded as a literal in tasks.
    embedding_model: str = "voyage-law-2"

    # Gemini LLM model
    gemini_model: str = "gemini-2.5-flash-lite"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Azure Blob Storage
    azure_storage_connection_string: str

    # CORS Configuration
    allowed_origins: str = "http://localhost:3000"

    # Environment
    debug: bool = False

    # Query Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours

    # CourtListener API (for on-demand legal research)
    courtlistener_api_token: str = ""

    # Citation Verification
    citation_verification_enabled: bool = True

    # Claim Verification (grounding)
    claim_verification_enabled: bool = True

    # Conflict Detection
    conflict_detection_enabled: bool = True

    # Hybrid Search (BM25 + Dense)
    hybrid_search_enabled: bool = True
    bm25_weight: float = 0.7
    dense_weight: float = 0.3
    rrf_k: int = 60
    bm25_top_k: int = 30   # candidates pulled from the sparse (BM25) path before fusion
    dense_top_k: int = 30  # candidates pulled from the dense path before fusion

    # Authority Hierarchy Scoring
    authority_scoring_enabled: bool = True

    # Agentic RAG (LangGraph multi-agent pipeline)
    agentic_rag_enabled: bool = False  # Disabled by default until tested
    groq_api_key: str = ""  # Groq API for fast LLM routing (optional)
    # Planning node: derive a research plan from clarification before retrieval.
    agentic_planning_enabled: bool = True
    # Gated CourtListener external-research step inside the agentic graph.
    agentic_external_research_enabled: bool = True
    # Hard cap on planner-derived iterations (sub-question fan-out + 1).
    agentic_max_iterations: int = 4
    # Sufficiency / strategy-selection thresholds (tunable without code change).
    agentic_sufficiency_high: float = 0.70   # avg relevance considered sufficient
    agentic_coverage_high: float = 0.80      # sub-question coverage considered sufficient
    agentic_relevance_low: float = 0.45      # below this -> reformulate query
    agentic_coverage_low: float = 0.50       # below this -> targeted sub-question search

    # Citation Knowledge Graph
    citation_graph_enabled: bool = True
    citation_graph_enrich_enabled: bool = True  # Best-effort CourtListener node enrichment at ingestion
    citation_graph_rag_injection_enabled: bool = True  # Inject good/bad-law validation block into the RAG prompt
    # Robustness guards (research-backed; see services/citation_graph.py)
    citation_graph_chronology_guard_enabled: bool = True  # Gap 1: drop negative case_cites edges that violate decision-date order
    treatment_verification_enabled: bool = True  # Gap 2: independent Groq second-pass verify of NEGATIVE treatments (CoVe)
    citation_graph_max_windows: int = 6  # Gap 2: cap on sliding windows for relationship mining (bounds LLM cost)
    citation_graph_max_concurrency: int = 8  # Max concurrent external (LLM/CourtListener) calls during ingest indexing
    llm_answer_provider: str = "gemini"  # Primary LLM for RAG answer generation: "gemini" or "groq" (Groq = fast/free, Gemini = default). Other provider is the fallback.
    # Agentic-loop context management / conversation summarisation
    agentic_context_summarize_after: int = 6   # Summarise once conversation exceeds this many turns
    agentic_context_keep_recent: int = 4       # Keep this many most-recent turns verbatim
    agentic_context_summary_max_chars: int = 1500  # Cap on the rolling summary length

    # Temporal Awareness
    temporal_extraction_enabled: bool = True
    temporal_filter_default: bool = True  # Exclude superseded by default in queries
    temporal_llm_fallback: bool = True    # Use Gemini when regex fails
    # As-of-date temporal queries ("what was the law as of <date>")
    temporal_as_of_enabled: bool = True   # Enable server-side as-of date filtering
    temporal_include_undated: bool = True  # Include docs with no effective_date in as-of results

    # Problem Formulation (Issue Spotting)
    problem_formulation_enabled: bool = True

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_allowed_origins_list(self) -> List[str]:
        """Parse comma-separated allowed origins"""
        origins = [origin.strip() for origin in self.allowed_origins.split(",")]
        return origins

@lru_cache()
def get_settings():
    return Settings()
