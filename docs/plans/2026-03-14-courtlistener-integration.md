# CourtListener On-Demand Legal Research — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "Include Legal Research" toggle to the chat UI that queries CourtListener's API for real US case law and merges results into the existing RAG pipeline.

**Architecture:** When the toggle is ON, the `/ask` endpoint calls CourtListener's search API in parallel with the existing Qdrant vector search. Results are merged, reranked together, and fed to Gemini with a modified prompt that distinguishes "Your Documents" from "Case Law". No new DB tables, no background jobs.

**Tech Stack:** Python httpx (async HTTP client), CourtListener REST API v4, React state + Lucide icons for the toggle UI.

---

### Task 1: Add CourtListener config

**Files:**
- Modify: `backend/config.py:6-49`

**Step 1: Write the failing test**

Create `tests/test_courtlistener_config.py`:

```python
"""Tests for CourtListener configuration"""
import pytest
import os


def test_settings_has_courtlistener_token():
    """Settings class should have courtlistener_api_token field"""
    from backend.config import Settings

    # Should have the field with a default empty string
    fields = Settings.model_fields
    assert "courtlistener_api_token" in fields


def test_settings_courtlistener_token_default():
    """CourtListener token should default to empty string (optional)"""
    from backend.config import Settings

    fields = Settings.model_fields
    assert fields["courtlistener_api_token"].default == ""
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_courtlistener_config.py -v`
Expected: FAIL with `KeyError: 'courtlistener_api_token'`

**Step 3: Write minimal implementation**

In `backend/config.py`, add inside the `Settings` class after the `cache_ttl_seconds` field (line 38):

```python
    # CourtListener API (for on-demand legal research)
    courtlistener_api_token: str = ""
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_courtlistener_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/config.py tests/test_courtlistener_config.py
git commit -m "feat: add CourtListener API token to config"
```

---

### Task 2: Create CourtListener service — search_cases()

**Files:**
- Create: `backend/services/legal_research.py`
- Test: `tests/test_legal_research.py`

**Step 1: Write the failing test**

Create `tests/test_legal_research.py`:

```python
"""Tests for CourtListener legal research service"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json


# ---- search_cases tests ----

@pytest.mark.asyncio
async def test_search_cases_returns_list():
    """search_cases should return a list of case dicts"""
    mock_response_data = {
        "results": [
            {
                "caseName": "Smith v. Jones",
                "court": "Supreme Court of the United States",
                "dateFiled": "2020-01-15",
                "snippet": "The court held that...",
                "absolute_url": "/opinion/12345/smith-v-jones/",
                "citation": ["531 U.S. 98"],
                "status": "Published",
                "id": 12345,
            }
        ]
    }

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_response_data
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        from backend.services.legal_research import search_cases

        results = await search_cases("section 230 immunity")

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["case_name"] == "Smith v. Jones"
    assert results[0]["court"] == "Supreme Court of the United States"
    assert results[0]["date_filed"] == "2020-01-15"
    assert results[0]["snippet"] == "The court held that..."
    assert results[0]["url"] == "https://www.courtlistener.com/opinion/12345/smith-v-jones/"
    assert results[0]["citation"] == "531 U.S. 98"


@pytest.mark.asyncio
async def test_search_cases_empty_results():
    """search_cases should return empty list when no results"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        from backend.services.legal_research import search_cases

        results = await search_cases("xyznonexistentquery123")

    assert results == []


@pytest.mark.asyncio
async def test_search_cases_handles_api_error():
    """search_cases should return empty list on API failure"""
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = Exception("Server error")

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        from backend.services.legal_research import search_cases

        results = await search_cases("test query")

    assert results == []


@pytest.mark.asyncio
async def test_search_cases_handles_timeout():
    """search_cases should return empty list on timeout"""
    import httpx

    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("timeout")):
        from backend.services.legal_research import search_cases

        results = await search_cases("test query")

    assert results == []
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.services.legal_research'`

**Step 3: Write minimal implementation**

Create `backend/services/legal_research.py`:

```python
"""CourtListener API integration for on-demand legal research"""
import logging
from typing import List, Dict, Optional

import httpx

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

logger = logging.getLogger(__name__)

COURTLISTENER_BASE_URL = "https://www.courtlistener.com"
COURTLISTENER_SEARCH_URL = f"{COURTLISTENER_BASE_URL}/api/rest/v4/search/"
REQUEST_TIMEOUT = 5.0  # seconds


def _get_headers() -> Dict[str, str]:
    """Build request headers with optional auth token."""
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if settings.courtlistener_api_token:
        headers["Authorization"] = f"Token {settings.courtlistener_api_token}"
    return headers


def _normalize_result(result: Dict) -> Dict:
    """Normalize a single CourtListener search result into a standard format."""
    # citation can be a list or string
    citation_raw = result.get("citation", [])
    if isinstance(citation_raw, list):
        citation = citation_raw[0] if citation_raw else ""
    else:
        citation = str(citation_raw)

    absolute_url = result.get("absolute_url", "")
    full_url = f"{COURTLISTENER_BASE_URL}{absolute_url}" if absolute_url else ""

    return {
        "case_name": result.get("caseName", "Unknown Case"),
        "court": result.get("court", ""),
        "date_filed": result.get("dateFiled", ""),
        "snippet": result.get("snippet", ""),
        "url": full_url,
        "citation": citation,
        "status": result.get("status", ""),
        "courtlistener_id": result.get("id", ""),
    }


async def search_cases(
    query: str,
    max_results: int = 5,
) -> List[Dict]:
    """
    Search CourtListener for relevant case law.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (1-10)

    Returns:
        List of normalized case dicts. Empty list on any error.
    """
    max_results = min(max(1, max_results), 10)

    params = {
        "q": query,
        "type": "o",  # opinions only
        "order_by": "score",
        "page_size": max_results,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                COURTLISTENER_SEARCH_URL,
                params=params,
                headers=_get_headers(),
            )
            response.raise_for_status()

        data = response.json()
        raw_results = data.get("results", [])
        return [_normalize_result(r) for r in raw_results]

    except httpx.TimeoutException:
        logger.warning("CourtListener API timed out")
        return []
    except Exception as e:
        logger.warning(f"CourtListener API error: {e}")
        return []
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add backend/services/legal_research.py tests/test_legal_research.py
git commit -m "feat: add CourtListener search service"
```

---

### Task 3: Add format_as_context() to convert CourtListener results to RAG chunks

**Files:**
- Modify: `backend/services/legal_research.py`
- Modify: `tests/test_legal_research.py`

**Step 1: Write the failing test**

Append to `tests/test_legal_research.py`:

```python
# ---- format_as_context tests ----

def test_format_as_context_converts_cases_to_chunks():
    """format_as_context should convert case results to chunk-like dicts"""
    from backend.services.legal_research import format_as_context

    cases = [
        {
            "case_name": "Smith v. Jones",
            "court": "Supreme Court of the United States",
            "date_filed": "2020-01-15",
            "snippet": "The court held that the defendant was liable.",
            "url": "https://www.courtlistener.com/opinion/12345/smith-v-jones/",
            "citation": "531 U.S. 98",
            "status": "Published",
            "courtlistener_id": 12345,
        }
    ]

    chunks = format_as_context(cases)

    assert len(chunks) == 1
    chunk = chunks[0]
    # Must have fields that rag_engine expects
    assert "content" in chunk
    assert "page_num" in chunk
    assert "section_name" in chunk
    assert "score" in chunk
    assert "document_name" in chunk
    assert "source_type" in chunk
    assert chunk["source_type"] == "case_law"
    assert "Smith v. Jones" in chunk["content"]
    assert "531 U.S. 98" in chunk["content"]
    assert chunk["document_name"] == "Smith v. Jones"
    assert chunk["url"] == "https://www.courtlistener.com/opinion/12345/smith-v-jones/"


def test_format_as_context_empty_input():
    """format_as_context should return empty list for empty input"""
    from backend.services.legal_research import format_as_context

    assert format_as_context([]) == []


def test_format_as_context_assigns_descending_scores():
    """Earlier results (higher ranked) should get higher scores"""
    from backend.services.legal_research import format_as_context

    cases = [
        {"case_name": "First", "court": "C", "date_filed": "2020-01-01",
         "snippet": "s1", "url": "u1", "citation": "c1", "status": "Published",
         "courtlistener_id": 1},
        {"case_name": "Second", "court": "C", "date_filed": "2020-01-01",
         "snippet": "s2", "url": "u2", "citation": "c2", "status": "Published",
         "courtlistener_id": 2},
    ]

    chunks = format_as_context(cases)
    assert chunks[0]["score"] > chunks[1]["score"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research.py::test_format_as_context_converts_cases_to_chunks -v`
Expected: FAIL with `ImportError: cannot import name 'format_as_context'`

**Step 3: Write minimal implementation**

Add to `backend/services/legal_research.py`:

```python
def format_as_context(cases: List[Dict]) -> List[Dict]:
    """
    Convert CourtListener results into chunk-like dicts compatible with the RAG pipeline.

    Each case becomes a "chunk" with content, metadata, and a synthetic score
    based on its search rank position.

    Args:
        cases: List of normalized case dicts from search_cases()

    Returns:
        List of chunk-like dicts with keys: content, page_num, section_name,
        score, document_name, document_id, chunk_id, source_type, url
    """
    if not cases:
        return []

    chunks = []
    for i, case in enumerate(cases):
        # Build readable content from case metadata
        parts = []
        if case.get("citation"):
            parts.append(f"Citation: {case['citation']}")
        parts.append(f"Case: {case.get('case_name', 'Unknown')}")
        if case.get("court"):
            parts.append(f"Court: {case['court']}")
        if case.get("date_filed"):
            parts.append(f"Date: {case['date_filed']}")
        if case.get("snippet"):
            # Strip HTML tags from CourtListener snippets
            import re
            clean_snippet = re.sub(r"<[^>]+>", "", case["snippet"])
            parts.append(f"\n{clean_snippet}")

        content = "\n".join(parts)

        # Assign descending score based on rank (1st result = 0.90, last = lower)
        score = max(0.50, 0.90 - (i * 0.05))

        chunk = {
            "content": content,
            "page_num": case.get("citation", "Case Law"),
            "section_name": case.get("court", ""),
            "score": score,
            "document_name": case.get("case_name", "Unknown Case"),
            "document_id": f"courtlistener-{case.get('courtlistener_id', '')}",
            "chunk_id": f"cl-{case.get('courtlistener_id', '')}",
            "source_type": "case_law",
            "url": case.get("url", ""),
        }
        chunks.append(chunk)

    return chunks
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add backend/services/legal_research.py tests/test_legal_research.py
git commit -m "feat: add format_as_context to convert case law to RAG chunks"
```

---

### Task 4: Add httpx dependency

**Files:**
- Modify: `requirements.txt`

**Step 1: Add httpx to requirements**

Add `httpx>=0.27.0` to `requirements.txt` (the root one).

**Step 2: Install**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && pip install httpx>=0.27.0`

**Step 3: Verify import works**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -c "import httpx; print(httpx.__version__)"`
Expected: prints version number

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add httpx dependency for CourtListener API"
```

Also add `pytest-asyncio` if not already present (needed for async tests):

Run: `pip install pytest-asyncio`

---

### Task 5: Wire legal research into the RAG engine

**Files:**
- Modify: `backend/services/rag_engine.py:917-924` (query_matter signature)
- Modify: `backend/services/rag_engine.py:993-1018` (retrieval section)
- Modify: `backend/services/rag_engine.py:1053-1065` (context formatting)

**Step 1: Write the failing test**

Create `tests/test_legal_research_integration.py`:

```python
"""Tests for legal research integration into RAG pipeline"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from uuid import uuid4


@pytest.mark.asyncio
async def test_query_matter_accepts_include_legal_research_param():
    """query_matter should accept include_legal_research parameter"""
    import inspect
    from backend.services.rag_engine import query_matter

    sig = inspect.signature(query_matter)
    assert "include_legal_research" in sig.parameters


@pytest.mark.asyncio
async def test_query_matter_calls_courtlistener_when_enabled(db):
    """When include_legal_research=True, should call search_cases"""
    from backend.models import Matter
    import uuid

    # Create a test matter
    matter = Matter(
        id=uuid.uuid4(),
        name="Test Matter",
        status="ready",
        file_type="pdf",
        blob_storage_path="test/path",
    )
    db.add(matter)
    db.commit()

    mock_cases = [
        {
            "case_name": "Test v. Case",
            "court": "Test Court",
            "date_filed": "2024-01-01",
            "snippet": "Test snippet",
            "url": "https://example.com",
            "citation": "123 U.S. 456",
            "status": "Published",
            "courtlistener_id": 99999,
        }
    ]

    with patch("backend.services.rag_engine.search_cases", new_callable=AsyncMock, return_value=mock_cases) as mock_search, \
         patch("backend.services.rag_engine.embed_query", return_value=[0.1] * 1024), \
         patch("backend.services.rag_engine.retrieve_chunks", return_value=[
             {"content": "test content", "page_num": "1", "section_name": "s1",
              "score": 0.8, "chunk_id": str(uuid.uuid4()), "document_id": str(uuid.uuid4()),
              "document_name": "doc.pdf"}
         ]), \
         patch("backend.services.rag_engine.rerank_chunks", side_effect=lambda q, c, top_k=8: c[:top_k]), \
         patch("backend.services.rag_engine.generate_answer", new_callable=AsyncMock, return_value=("Test answer [Page 1]", 100)), \
         patch("backend.services.rag_engine.generate_document_summary", return_value=None):

        from backend.services.rag_engine import query_matter

        result = await query_matter(
            str(matter.id), "test query", db,
            include_legal_research=True
        )

        mock_search.assert_called_once()
        assert result["answer"] is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research_integration.py::test_query_matter_accepts_include_legal_research_param -v`
Expected: FAIL with `TypeError` (parameter doesn't exist yet)

**Step 3: Write minimal implementation**

Modify `backend/services/rag_engine.py`:

**3a.** Add import at top (after other imports, around line 38):

```python
from backend.services.legal_research import search_cases, format_as_context
```

(Add this inside the existing try/except import block pattern.)

**3b.** Update `query_matter` signature (line 917-924) to add the new parameter:

```python
async def query_matter(
    matter_id: str,
    query: str,
    db: Session,
    top_k: int = FINAL_CHUNK_COUNT,
    temperature: float = 0.2,
    conversation_history: list = None,
    include_legal_research: bool = False,
) -> Dict:
```

**3c.** After chunk retrieval (after line ~1018, after the existing `retrieve_chunks` calls), add legal research merging:

```python
        # 3.5 On-demand legal research (CourtListener)
        external_chunks = []
        if include_legal_research:
            try:
                case_results = await search_cases(query, max_results=5)
                if case_results:
                    external_chunks = format_as_context(case_results)
                    logger.info(f"CourtListener returned {len(external_chunks)} case law results")
            except Exception as e:
                logger.warning(f"Legal research failed (non-blocking): {e}")

        # Merge external chunks with retrieved chunks
        if external_chunks:
            retrieved_chunks = retrieved_chunks + external_chunks
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research_integration.py -v`
Expected: PASS

Also run existing RAG tests to ensure no regression:

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_rag_engine.py -v`
Expected: PASS (existing tests unaffected since `include_legal_research` defaults to False)

**Step 5: Commit**

```bash
git add backend/services/rag_engine.py tests/test_legal_research_integration.py
git commit -m "feat: wire CourtListener into RAG pipeline with include_legal_research flag"
```

---

### Task 6: Update the system prompt to distinguish case law from user documents

**Files:**
- Modify: `backend/services/rag_engine.py:118-175` (format_legal_context function)

**Step 1: Write the failing test**

Append to `tests/test_legal_research_integration.py`:

```python
def test_format_legal_context_labels_case_law_chunks():
    """Case law chunks should be labeled differently from user doc chunks"""
    from backend.services.rag_engine import format_legal_context

    chunks = [
        {"content": "User doc content", "page_num": "1", "section_name": "Intro",
         "score": 0.8, "document_name": "contract.pdf"},
        {"content": "Court held that...", "page_num": "531 U.S. 98",
         "section_name": "Supreme Court", "score": 0.75,
         "document_name": "Smith v. Jones", "source_type": "case_law"},
    ]

    context = format_legal_context(chunks, "Test Matter")

    assert "YOUR DOCUMENTS" in context or "EXCERPT" in context
    assert "CASE LAW" in context
    assert "Smith v. Jones" in context
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research_integration.py::test_format_legal_context_labels_case_law_chunks -v`
Expected: FAIL — "CASE LAW" not in context

**Step 3: Write minimal implementation**

In `backend/services/rag_engine.py`, modify `format_legal_context` (around line 148-173). Replace the chunk formatting loop with one that checks `source_type`:

```python
    for i, chunk in enumerate(sorted_chunks, 1):
        location = chunk.get("page_num", "Unknown")
        section = chunk.get("section_name", "")
        score = chunk.get("score", 0)
        content = chunk.get("content", "")
        source_type = chunk.get("source_type", "document")

        if source_type == "case_law":
            # Label case law chunks distinctly
            header = f"--- CASE LAW {i} (Case: {chunk.get('document_name', 'Unknown')}"
            if section:
                header += f", Court: {section}"
            header += f", Score: {score:.2f}) ---\n"
        else:
            # Determine location label based on format
            if location.startswith("para"):
                location_label = f"Paragraph {location[5:]}"
            elif location.startswith("line"):
                location_label = f"Lines {location[5:]}"
            else:
                location_label = f"Page {location}"

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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research_integration.py -v`
Expected: PASS

Run existing tests: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_rag_engine.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/services/rag_engine.py tests/test_legal_research_integration.py
git commit -m "feat: label case law chunks distinctly in RAG context"
```

---

### Task 7: Update the API endpoint to accept include_legal_research

**Files:**
- Modify: `backend/main.py:729-809` (ask_question endpoint)
- Modify: `backend/schemas.py`

**Step 1: Update the schema**

In `backend/schemas.py`, modify `QueryCreate`:

```python
class QueryCreate(BaseModel):
    """Query request (ask question)"""
    question: str = Field(..., min_length=1, max_length=1000)
    include_legal_research: bool = Field(False, description="Include CourtListener case law in results")
```

**Step 2: Update the endpoint**

In `backend/main.py`, modify the `ask_question` endpoint (line 729-733):

Change:
```python
@app.post("/matters/{matter_id}/ask", response_model=dict)
async def ask_question(
    matter_id: str,
    question: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
```

To:
```python
@app.post("/matters/{matter_id}/ask", response_model=dict)
async def ask_question(
    matter_id: str,
    body: QueryCreate = Body(...),
    db: Session = Depends(get_db)
):
```

Then update the function body:
- Change `validate_question(question)` to `validate_question(body.question)`
- Change all references to `question` to `body.question`
- Pass `include_legal_research=body.include_legal_research` to `query_matter()`

The `query_matter` call (line ~787) becomes:

```python
        rag_result = await query_matter(
            str(matter_uuid), body.question, db,
            conversation_history=conversation_history,
            include_legal_research=body.include_legal_research,
        )
```

And the query storage (line ~791):
```python
        if rag_result.get("answer"):
            db_query = Query(
                id=uuid.uuid4(),
                matter_id=matter_uuid,
                question=body.question,
                ...
            )
```

And the audit log (line ~801):
```python
            log_activity(db, str(matter_uuid), "query", details=body.question)
```

**Step 3: Run existing tests**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/ -v --ignore=tests/test_legal_research_integration.py`
Expected: PASS

**Step 4: Commit**

```bash
git add backend/main.py backend/schemas.py
git commit -m "feat: accept include_legal_research in /ask endpoint"
```

---

### Task 8: Update sources response to include case law metadata

**Files:**
- Modify: `backend/services/rag_engine.py:1142-1173` (sources building section)

**Step 1: Write the failing test**

Append to `tests/test_legal_research_integration.py`:

```python
def test_source_includes_case_law_fields():
    """Sources from case law should include source_type and url"""
    # Simulate what query_matter builds for sources
    chunk = {
        "chunk_id": "cl-12345",
        "page_num": "531 U.S. 98",
        "section_name": "Supreme Court",
        "score": 0.85,
        "content": "The court held...",
        "document_id": "courtlistener-12345",
        "document_name": "Smith v. Jones",
        "source_type": "case_law",
        "url": "https://www.courtlistener.com/opinion/12345/smith-v-jones/",
    }

    # Build source dict the same way rag_engine does
    source = {
        "chunk_id": chunk.get("chunk_id", ""),
        "page_num": chunk.get("page_num", ""),
        "section_name": chunk.get("section_name", ""),
        "relevance_score": chunk.get("score", 0),
        "content": chunk.get("content", ""),
        "document_id": chunk.get("document_id", ""),
        "document_name": chunk.get("document_name", ""),
        "source_type": chunk.get("source_type", "document"),
        "url": chunk.get("url", ""),
    }

    assert source["source_type"] == "case_law"
    assert source["url"] == "https://www.courtlistener.com/opinion/12345/smith-v-jones/"
```

**Step 2: Modify the source building in rag_engine.py**

In `backend/services/rag_engine.py`, in the sources building loop (around line 1159-1173), add `source_type` and `url` to each source dict:

```python
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
                "source_type": chunk.get("source_type", "document"),
                "url": chunk.get("url", ""),
            }
            sources.append(source)
```

**Step 3: Run tests**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/test_legal_research_integration.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add backend/services/rag_engine.py tests/test_legal_research_integration.py
git commit -m "feat: include source_type and url in RAG sources response"
```

---

### Task 9: Frontend — Update types and API service

**Files:**
- Modify: `frontend/lib/types.ts:1-9`
- Modify: `frontend/lib/api-services.ts:42-74,127-130`

**Step 1: Update Citation type**

In `frontend/lib/types.ts`, add `sourceType` and `url` to the `Citation` interface:

```typescript
export interface Citation {
  documentName: string
  pageNumber: number
  section?: string
  excerpt: string
  relevanceScore: number
  /** Full chunk content for click-to-view */
  content?: string
  /** "document" or "case_law" */
  sourceType?: "document" | "case_law"
  /** CourtListener URL for case law */
  url?: string
}
```

**Step 2: Update AskResponse type**

In `frontend/lib/api-services.ts`, update the `sources` type inside `AskResponse` (around line 42-52):

```typescript
  sources: {
    chunk_id: string
    page_num: string
    section_name: string
    relevance_score: number
    content: string
    document_id: string
    document_name: string
    source_type?: "document" | "case_law"
    url?: string
  }[]
```

**Step 3: Update askQuestion function**

In `frontend/lib/api-services.ts`, update the `askQuestion` function (line 127-130):

```typescript
export async function askQuestion(
  matterId: string,
  question: string,
  includeLegalResearch: boolean = false,
): Promise<AskResponse> {
  const { data } = await api.post<AskResponse>(`/matters/${matterId}/ask`, {
    question,
    include_legal_research: includeLegalResearch,
  })
  return data
}
```

**Step 4: Commit**

```bash
git add frontend/lib/types.ts frontend/lib/api-services.ts
git commit -m "feat: update frontend types and API for legal research toggle"
```

---

### Task 10: Frontend — Update hooks to pass includeLegalResearch

**Files:**
- Modify: `frontend/hooks/use-matters.ts:98-108`

**Step 1: Update useAskQuestion hook**

In `frontend/hooks/use-matters.ts`, change the `useAskQuestion` mutation to accept an object:

```typescript
export function useAskQuestion(matterId: string) {
  const queryClient = useQueryClient()

  return useMutation<AskResponse, Error, { question: string; includeLegalResearch?: boolean }>({
    mutationFn: ({ question, includeLegalResearch }) =>
      askQuestion(matterId, question, includeLegalResearch ?? false),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["matters", matterId] })
      queryClient.invalidateQueries({ queryKey: ["matters", matterId, "queries"] })
    },
  })
}
```

**Step 2: Commit**

```bash
git add frontend/hooks/use-matters.ts
git commit -m "feat: update useAskQuestion hook to support legal research toggle"
```

---

### Task 11: Frontend — Add toggle to ChatPanel

**Files:**
- Modify: `frontend/components/ChatPanel.tsx`

**Step 1: Update ChatPanel props and add toggle state**

Update the `ChatPanel` component props to include `onSend` accepting a second argument, and add the toggle UI:

In `ChatPanel.tsx`, update the component:

```typescript
export default function ChatPanel({
  messages,
  onSend,
  isLoading,
  onSelectCitation,
  onCitationClick,
}: {
  messages: QueryMessage[]
  onSend: (message: string, includeLegalResearch: boolean) => void
  isLoading?: boolean
  onSelectCitation?: (citations: Citation[]) => void
  onCitationClick?: (citation: Citation) => void
}) {
  const [input, setInput] = useState("")
  const [includeLegalResearch, setIncludeLegalResearch] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
```

Update `handleSubmit` to pass the toggle:

```typescript
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSend(input.trim(), includeLegalResearch)
      setInput("")
    }
  }
```

Add the toggle UI in the form area, between the `</div>` closing the Textarea wrapper and the `<p>` hint text (around line 226-228). Replace the existing hint paragraph with:

```tsx
        <div className="flex items-center justify-between mt-2">
          <p className="text-[11px] text-muted-foreground">
            Enter to send &middot; Shift+Enter for new line
          </p>
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <span className="text-[11px] text-muted-foreground">Include Legal Research</span>
            <button
              type="button"
              role="switch"
              aria-checked={includeLegalResearch}
              onClick={() => setIncludeLegalResearch(!includeLegalResearch)}
              className={cn(
                "relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent transition-colors",
                includeLegalResearch ? "bg-primary" : "bg-muted/30"
              )}
            >
              <span
                className={cn(
                  "pointer-events-none block h-4 w-4 rounded-full bg-white shadow-sm transition-transform",
                  includeLegalResearch ? "translate-x-4" : "translate-x-0"
                )}
              />
            </button>
          </label>
        </div>
```

**Step 2: Commit**

```bash
git add frontend/components/ChatPanel.tsx
git commit -m "feat: add Include Legal Research toggle to ChatPanel"
```

---

### Task 12: Frontend — Wire toggle through Matter workspace page

**Files:**
- Modify: `frontend/app/matters/[id]/page.tsx:97-153` (handleSendMessage)

**Step 1: Update handleSendMessage**

In `frontend/app/matters/[id]/page.tsx`, update `handleSendMessage` to accept and pass the toggle:

```typescript
  const handleSendMessage = useCallback((content: string, includeLegalResearch: boolean) => {
    const userMsg: QueryMessage = {
      id: `msg-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])

    askQuestion.mutate({ question: content, includeLegalResearch }, {
      onSuccess: (result) => {
        if (result.answer) {
          const citations: Citation[] = (result.sources || []).map((s) => ({
            documentName: s.document_name || matter?.name || "Document",
            pageNumber: parseInt(s.page_num) || 0,
            section: s.section_name || "",
            excerpt: s.content?.slice(0, 200) || "",
            relevanceScore: s.relevance_score || 0,
            content: s.content ?? undefined,
            sourceType: (s.source_type as "document" | "case_law") || "document",
            url: s.url || undefined,
          }))

          const confidenceScore = typeof result.confidence === "object"
            ? Math.round((result.confidence.score || 0) * 100)
            : 0

          const aiMsg: QueryMessage = {
            id: `msg-${Date.now() + 1}`,
            role: "assistant",
            content: result.answer,
            citations,
            confidenceScore,
            timestamp: new Date().toISOString(),
          }
          setMessages((prev) => [...prev, aiMsg])
          setSelectedCitations(citations)
        } else {
          const errorMsg: QueryMessage = {
            id: `msg-${Date.now() + 1}`,
            role: "assistant",
            content: result.error || "Sorry, I couldn't generate an answer. Please try rephrasing your question.",
            timestamp: new Date().toISOString(),
          }
          setMessages((prev) => [...prev, errorMsg])
        }
      },
      onError: () => {
        const errorMsg: QueryMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "assistant",
          content: "An error occurred while processing your question. Please try again.",
          timestamp: new Date().toISOString(),
        }
        setMessages((prev) => [...prev, errorMsg])
      },
    })
  }, [askQuestion, matter])
```

**Step 2: Commit**

```bash
git add frontend/app/matters/[id]/page.tsx
git commit -m "feat: pass legal research toggle from workspace to ask mutation"
```

---

### Task 13: Frontend — Update CitationPanel for case law badges

**Files:**
- Modify: `frontend/components/CitationPanel.tsx`

**Step 1: Update CitationPanel to show case law badges**

In `frontend/components/CitationPanel.tsx`, update the citation card (around line 41-79) to distinguish case law from user docs:

Replace the icon and add a badge. In the existing citation map (line 42-79), update:

```tsx
        {citations.map((citation, idx) => (
          <button
            type="button"
            key={idx}
            onClick={() => onCitationClick?.(citation)}
            className="w-full text-left rounded-xl border border-border bg-white p-3 hover:shadow-elevated hover:border-border-strong transition-all duration-200 cursor-pointer group"
          >
            <div className="flex items-start gap-2.5">
              <div className={cn(
                "flex h-7 w-7 shrink-0 items-center justify-center rounded-lg transition-colors",
                citation.sourceType === "case_law"
                  ? "bg-blue-50 text-blue-600 group-hover:bg-blue-100"
                  : "bg-surface text-muted group-hover:bg-surface-hover"
              )}>
                <FileText className="h-3.5 w-3.5" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <p className="text-[13px] font-medium text-foreground truncate">{citation.documentName}</p>
                  {citation.sourceType === "case_law" ? (
                    <a
                      href={citation.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="shrink-0"
                    >
                      <ExternalLink className="h-3 w-3 text-blue-500 hover:text-blue-700 transition-colors" />
                    </a>
                  ) : (
                    <ExternalLink className="h-3 w-3 text-muted opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                  )}
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  {citation.sourceType === "case_law" ? (
                    <span className="inline-flex items-center rounded-md bg-blue-50 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 border border-blue-200/60">
                      Case Law
                    </span>
                  ) : (
                    <span className="text-[11px] text-muted">
                      Page {citation.pageNumber}
                    </span>
                  )}
                  {citation.section && <span className="text-[11px] text-muted">&middot; {citation.section}</span>}
                </div>
                {citation.excerpt && (
                  <p className="text-[11px] text-muted/70 mt-1.5 line-clamp-2 italic leading-relaxed">
                    &ldquo;{citation.excerpt}&rdquo;
                  </p>
                )}
              </div>
              <span className={cn(
                "shrink-0 font-mono text-[11px] font-medium px-1.5 py-0.5 rounded-md border",
                citation.relevanceScore >= 0.8
                  ? "bg-emerald-50 text-emerald-700 border-emerald-200/60"
                  : citation.relevanceScore >= 0.6
                    ? "bg-amber-50 text-amber-700 border-amber-200/60"
                    : "bg-surface text-muted border-border"
              )}>
                {Math.round(citation.relevanceScore * 100)}%
              </span>
            </div>
          </button>
        ))}
```

**Step 2: Commit**

```bash
git add frontend/components/CitationPanel.tsx
git commit -m "feat: show Case Law badge and link for case law citations"
```

---

### Task 14: Verify end-to-end

**Step 1: Run all backend tests**

Run: `cd /Users/smeet/Documents/GitHub/Lexintel && python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Start the backend and frontend**

Run backend: `cd /Users/smeet/Documents/GitHub/Lexintel/backend && uvicorn main:app --reload`
Run frontend: `cd /Users/smeet/Documents/GitHub/Lexintel/frontend && npm run dev`

**Step 3: Manual test**

1. Open http://localhost:3000/matters/{any-matter-id}
2. Verify the "Include Legal Research" toggle appears next to the chat input
3. Toggle it ON
4. Ask: "What does Section 230 of the Communications Decency Act say?"
5. Verify response includes case law citations with blue "Case Law" badges
6. Verify case law citations have clickable links to CourtListener
7. Toggle it OFF, ask another question, verify only user doc citations appear

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete CourtListener on-demand legal research integration"
```
