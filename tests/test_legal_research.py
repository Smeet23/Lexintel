"""Tests for CourtListener legal research service"""
import pytest
import httpx
import respx


# ---- Helper: mock CourtListener response ----

def _mock_courtlistener_response(results=None):
    """Build a realistic CourtListener API v4 response."""
    if results is None:
        results = [
            {
                "caseName": "Smith v. Jones",
                "court": "Supreme Court of the United States",
                "court_id": "scotus",
                "dateFiled": "2020-01-15",
                "absolute_url": "/opinion/12345/smith-v-jones/",
                "citation": ["531 U.S. 98"],
                "status": "Published",
                "cluster_id": 12345,
                "citeCount": 42,
                "opinions": [
                    {
                        "id": 99999,
                        "type": "lead-opinion",
                        "snippet": "The court <mark>held</mark> that the defendant was liable.",
                    }
                ],
                "meta": {"score": {"bm25": 15.234}},
            }
        ]
    return {"count": len(results), "next": None, "previous": None, "results": results}


# ---- search_cases tests ----

@pytest.mark.asyncio
@respx.mock
async def test_search_cases_returns_normalized_results():
    """search_cases should return a list of normalized case dicts"""
    respx.get(
        "https://www.courtlistener.com/api/rest/v4/search/"
    ).mock(return_value=httpx.Response(200, json=_mock_courtlistener_response()))

    from backend.services.legal_research import search_cases

    results = await search_cases("section 230 immunity")

    assert isinstance(results, list)
    assert len(results) == 1
    r = results[0]
    assert r["case_name"] == "Smith v. Jones"
    assert r["court"] == "Supreme Court of the United States"
    assert r["date_filed"] == "2020-01-15"
    # Snippet should have HTML stripped
    assert r["snippet"] == "The court held that the defendant was liable."
    assert "<mark>" not in r["snippet"]
    assert r["url"] == "https://www.courtlistener.com/opinion/12345/smith-v-jones/"
    assert r["citation"] == "531 U.S. 98"
    assert r["cite_count"] == 42
    assert r["courtlistener_id"] == 12345


@pytest.mark.asyncio
@respx.mock
async def test_search_cases_empty_results():
    """search_cases should return empty list when no results"""
    respx.get(
        "https://www.courtlistener.com/api/rest/v4/search/"
    ).mock(return_value=httpx.Response(200, json=_mock_courtlistener_response(results=[])))

    from backend.services.legal_research import search_cases

    results = await search_cases("xyznonexistentquery123")
    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_search_cases_handles_api_error():
    """search_cases should return empty list on API failure"""
    respx.get(
        "https://www.courtlistener.com/api/rest/v4/search/"
    ).mock(return_value=httpx.Response(500))

    from backend.services.legal_research import search_cases

    results = await search_cases("test query")
    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_search_cases_handles_timeout():
    """search_cases should return empty list on timeout"""
    respx.get(
        "https://www.courtlistener.com/api/rest/v4/search/"
    ).mock(side_effect=httpx.TimeoutException("timeout"))

    from backend.services.legal_research import search_cases

    results = await search_cases("test query")
    assert results == []


@pytest.mark.asyncio
@respx.mock
async def test_search_cases_multiple_citations():
    """Should pick the first citation when multiple exist"""
    result_with_multi_cite = {
        "caseName": "Multi v. Cite",
        "court": "Test Court",
        "court_id": "test",
        "dateFiled": "2024-01-01",
        "absolute_url": "/opinion/111/multi-v-cite/",
        "citation": ["123 F.3d 456", "2024 WL 789"],
        "status": "Published",
        "cluster_id": 111,
        "citeCount": 5,
        "opinions": [{"snippet": "Test snippet"}],
    }
    respx.get(
        "https://www.courtlistener.com/api/rest/v4/search/"
    ).mock(return_value=httpx.Response(200, json=_mock_courtlistener_response(results=[result_with_multi_cite])))

    from backend.services.legal_research import search_cases

    results = await search_cases("test")
    assert results[0]["citation"] == "123 F.3d 456"


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
            "cite_count": 42,
        }
    ]

    chunks = format_as_context(cases)

    assert len(chunks) == 1
    chunk = chunks[0]
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
         "courtlistener_id": 1, "cite_count": 0},
        {"case_name": "Second", "court": "C", "date_filed": "2020-01-01",
         "snippet": "s2", "url": "u2", "citation": "c2", "status": "Published",
         "courtlistener_id": 2, "cite_count": 0},
    ]

    chunks = format_as_context(cases)
    assert chunks[0]["score"] > chunks[1]["score"]


def test_strip_html():
    """_strip_html should remove HTML tags"""
    from backend.services.legal_research import _strip_html

    assert _strip_html("The <mark>court</mark> held") == "The court held"
    assert _strip_html("No tags here") == "No tags here"
    assert _strip_html("") == ""
