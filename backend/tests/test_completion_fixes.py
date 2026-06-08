"""Unit tests for the May 2026 completion-pass fixes.

Covers logic that is unit-testable without live infra:
- amendment_chain_manager.recalculate_chain_status (supersession marking)
- citation_graph._year_from_date
- embeddings._cache_key (provider/role scoping)
"""
import os
import sys
import types
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def _doc(status="unknown", effective=None, created=None):
    """Lightweight Document stand-in (only the attributes the logic touches)."""
    return types.SimpleNamespace(
        document_status=status,
        effective_date=effective,
        created_at=created,
        superseded_date=None,
    )


class _FakeDB:
    def add(self, _obj):
        pass

    def flush(self):
        pass


class TestRecalculateChainStatus:
    def test_marks_older_versions_superseded_and_latest_current(self, monkeypatch):
        from backend.services import amendment_chain_manager as acm

        d1 = _doc(effective=datetime(2018, 1, 1, tzinfo=timezone.utc))
        d2 = _doc(effective=datetime(2020, 6, 1, tzinfo=timezone.utc))
        d3 = _doc(effective=datetime(2023, 3, 1, tzinfo=timezone.utc))
        # get_version_chain returns oldest -> newest
        monkeypatch.setattr(acm, "get_version_chain", lambda cid, db: [d1, d2, d3])

        acm.recalculate_chain_status("chain-id", _FakeDB())

        assert d1.document_status == "superseded"
        assert d2.document_status == "superseded"
        assert d3.document_status == "current"
        # superseded_date == successor's effective_date
        assert d1.superseded_date == d2.effective_date
        assert d2.superseded_date == d3.effective_date
        assert d3.superseded_date is None

    def test_single_document_is_unchanged(self, monkeypatch):
        from backend.services import amendment_chain_manager as acm

        only = _doc(status="unknown")
        monkeypatch.setattr(acm, "get_version_chain", lambda cid, db: [only])

        acm.recalculate_chain_status("chain-id", _FakeDB())

        assert only.document_status == "unknown"  # untouched
        assert only.superseded_date is None

    def test_explicit_repealed_status_preserved(self, monkeypatch):
        from backend.services import amendment_chain_manager as acm

        older_repealed = _doc(status="repealed", effective=datetime(2010, 1, 1, tzinfo=timezone.utc))
        newer = _doc(status="unknown", effective=datetime(2015, 1, 1, tzinfo=timezone.utc))
        monkeypatch.setattr(acm, "get_version_chain", lambda cid, db: [older_repealed, newer])

        acm.recalculate_chain_status("chain-id", _FakeDB())

        # repealed is more specific than superseded — keep it
        assert older_repealed.document_status == "repealed"
        assert newer.document_status == "current"

    def test_falls_back_to_created_at_when_no_effective_date(self, monkeypatch):
        from backend.services import amendment_chain_manager as acm

        d1 = _doc(created=datetime(2019, 1, 1, tzinfo=timezone.utc))
        d2 = _doc(created=datetime(2021, 1, 1, tzinfo=timezone.utc))
        monkeypatch.setattr(acm, "get_version_chain", lambda cid, db: [d1, d2])

        acm.recalculate_chain_status("chain-id", _FakeDB())

        assert d1.document_status == "superseded"
        assert d1.superseded_date == d2.created_at  # successor effective None -> created_at


class TestYearFromDate:
    def test_parses_iso_date(self):
        from backend.services.citation_graph import _year_from_date

        assert _year_from_date("1954-05-17") == 1954
        assert _year_from_date("2020") == 2020

    def test_handles_missing_or_bad_input(self):
        from backend.services.citation_graph import _year_from_date

        assert _year_from_date(None) is None
        assert _year_from_date("") is None
        assert _year_from_date("not-a-date") is None
        assert _year_from_date("n/a 2020") is None  # year not at start


class TestEmbeddingCacheKey:
    def test_key_is_provider_and_role_scoped(self):
        from backend.services.embeddings import _cache_key, _detect_provider

        provider = _detect_provider()
        doc_key = _cache_key("hello world", kind="document")
        query_key = _cache_key("hello world", kind="query")

        assert doc_key.startswith(f"{provider}:document:")
        assert query_key.startswith(f"{provider}:query:")
        # Same text, different role -> different cache key (asymmetric embeddings)
        assert doc_key != query_key

    def test_distinct_text_distinct_key(self):
        from backend.services.embeddings import _cache_key

        assert _cache_key("alpha") != _cache_key("beta")
