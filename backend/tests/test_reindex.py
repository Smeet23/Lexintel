"""Unit tests for embedding re-index tooling (May 2026).

Pure-logic coverage, no live DB / no live embedding API:
- embeddings._cache_key provider override (cross-provider key distinctness)
- embeddings.embed_chunks_with_provider routing (voyage vs cohere branch)
- embeddings.embed_chunks remains a thin wrapper over the explicit-provider path
- tasks._provider_for_model mapping
- tasks._chunk_to_payload_dict hydration completeness
"""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ─────────────────────────────────────────────────────────────
# _cache_key provider override
# ─────────────────────────────────────────────────────────────
class TestCacheKeyProviderOverride:
    def test_explicit_provider_prefixes_key(self):
        from backend.services.embeddings import _cache_key

        assert _cache_key("hello", provider="voyage").startswith("voyage:document:")
        assert _cache_key("hello", provider="cohere").startswith("cohere:document:")

    def test_cross_provider_keys_differ_for_same_text(self):
        from backend.services.embeddings import _cache_key

        v = _cache_key("same text", provider="voyage")
        c = _cache_key("same text", provider="cohere")
        assert v != c  # prevents cross-provider cache poisoning

    def test_override_falls_back_to_detect_when_absent(self):
        from backend.services.embeddings import _cache_key, _detect_provider

        detected = _detect_provider()
        assert _cache_key("x").startswith(f"{detected}:document:")

    def test_kind_still_scoped_under_explicit_provider(self):
        from backend.services.embeddings import _cache_key

        doc = _cache_key("t", kind="document", provider="voyage")
        qry = _cache_key("t", kind="query", provider="voyage")
        assert doc.startswith("voyage:document:")
        assert qry.startswith("voyage:query:")
        assert doc != qry


# ─────────────────────────────────────────────────────────────
# embed_chunks_with_provider routing
# ─────────────────────────────────────────────────────────────
class _FakeCache:
    """In-memory embedding cache stand-in (get/put)."""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        self.store[key] = value


class TestEmbedChunksWithProviderRouting:
    def test_routes_to_voyage_branch(self, monkeypatch):
        from backend.services import embeddings

        calls = {}

        def fake_voyage(texts, input_type="document"):
            calls["voyage"] = list(texts)
            return [[0.1] * 1024 for _ in texts]

        def fake_cohere_client():
            raise AssertionError("cohere branch must not be called for provider='voyage'")

        monkeypatch.setattr(embeddings, "_voyage_embed_batch_with_retry", fake_voyage)
        monkeypatch.setattr(embeddings, "get_cohere_client", fake_cohere_client)
        monkeypatch.setattr(embeddings, "get_embedding_cache", lambda: _FakeCache())

        out = embeddings.embed_chunks_with_provider(["a", "b"], "voyage")
        assert calls["voyage"] == ["a", "b"]
        assert len(out) == 2 and len(out[0]) == 1024

    def test_routes_to_cohere_branch(self, monkeypatch):
        from backend.services import embeddings

        calls = {}

        def fake_voyage(texts, input_type="document"):
            raise AssertionError("voyage branch must not be called for provider='cohere'")

        def fake_cohere_client():
            return object()

        def fake_cohere_batch(client, texts, input_type="search_document"):
            calls["cohere"] = list(texts)
            return [[0.2] * 1024 for _ in texts]

        monkeypatch.setattr(embeddings, "_voyage_embed_batch_with_retry", fake_voyage)
        monkeypatch.setattr(embeddings, "get_cohere_client", fake_cohere_client)
        monkeypatch.setattr(embeddings, "_cohere_embed_batch_with_retry", fake_cohere_batch)
        monkeypatch.setattr(embeddings, "get_embedding_cache", lambda: _FakeCache())

        out = embeddings.embed_chunks_with_provider(["x", "y", "z"], "cohere")
        assert calls["cohere"] == ["x", "y", "z"]
        assert len(out) == 3

    def test_rejects_unknown_provider(self):
        from backend.services import embeddings

        with pytest.raises(ValueError):
            embeddings.embed_chunks_with_provider(["a"], "openai")

    def test_rejects_empty_chunks(self):
        from backend.services import embeddings

        with pytest.raises(ValueError):
            embeddings.embed_chunks_with_provider([], "voyage")

    def test_caches_under_explicit_provider_key(self, monkeypatch):
        from backend.services import embeddings

        cache = _FakeCache()
        monkeypatch.setattr(embeddings, "get_embedding_cache", lambda: cache)
        monkeypatch.setattr(
            embeddings, "_voyage_embed_batch_with_retry",
            lambda texts, input_type="document": [[0.3] * 1024 for _ in texts],
        )
        embeddings.embed_chunks_with_provider(["hello"], "voyage")
        assert any(k.startswith("voyage:document:") for k in cache.store)

    def test_embed_chunks_delegates_to_detected_provider(self, monkeypatch):
        from backend.services import embeddings

        seen = {}

        def fake_with_provider(chunks, provider):
            seen["provider"] = provider
            return [[0.0] * 1024 for _ in chunks]

        monkeypatch.setattr(embeddings, "_detect_provider", lambda: "cohere")
        monkeypatch.setattr(embeddings, "embed_chunks_with_provider", fake_with_provider)
        embeddings.embed_chunks(["a"])
        assert seen["provider"] == "cohere"


# ─────────────────────────────────────────────────────────────
# tasks._provider_for_model
# ─────────────────────────────────────────────────────────────
class TestProviderForModel:
    def test_voyage_models_route_voyage(self):
        from backend.tasks import _provider_for_model

        assert _provider_for_model("voyage-law-2") == "voyage"
        assert _provider_for_model("VOYAGE-LAW-2") == "voyage"

    def test_cohere_and_unknown_route_cohere(self):
        from backend.tasks import _provider_for_model

        assert _provider_for_model("embed-english-v3.0") == "cohere"
        assert _provider_for_model("some-random-model") == "cohere"
        assert _provider_for_model("") == "cohere"


# ─────────────────────────────────────────────────────────────
# tasks._chunk_to_payload_dict
# ─────────────────────────────────────────────────────────────
def _chunk(**kw):
    base = dict(
        id="11111111-1111-1111-1111-111111111111",
        content="chunk body",
        page_num="3",
        section_name="Section 2",
        section_type="article",
        chunk_sequence=4,
        concepts=["a", "b"],
        document_id="22222222-2222-2222-2222-222222222222",
        authority_metadata={
            "source_type": "case_law",
            "court_level": "supreme",
            "jurisdiction_code": "US",
            "authority_score": 0.9,
            "binding_authority": True,
        },
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _document(**kw):
    base = dict(
        name="Contract.pdf",
        document_type="contract",
        jurisdiction="US",
        summary="A short summary.",
        effective_date=None,
        superseded_date=None,
        document_status="current",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


class TestChunkToPayloadDict:
    def test_hydrates_document_name_and_authority(self):
        from backend.tasks import _chunk_to_payload_dict

        payload = _chunk_to_payload_dict(_chunk(), _document())
        assert payload["document_name"] == "Contract.pdf"
        assert payload["document_type"] == "contract"
        assert payload["jurisdiction"] == "US"
        assert payload["source_type"] == "case_law"
        assert payload["court_level"] == "supreme"
        assert payload["jurisdiction_code"] == "US"
        assert payload["authority_score"] == 0.9
        assert payload["binding_authority"] is True
        assert payload["content"] == "chunk body"
        assert payload["chunk_sequence"] == 4

    def test_does_not_mutate_chunk_authority_metadata(self):
        from backend.tasks import _chunk_to_payload_dict

        chunk = _chunk()
        original = dict(chunk.authority_metadata)
        _chunk_to_payload_dict(chunk, _document())
        assert chunk.authority_metadata == original  # immutable: input untouched

    def test_handles_missing_document_gracefully(self):
        from backend.tasks import _chunk_to_payload_dict

        payload = _chunk_to_payload_dict(_chunk(), None)
        assert payload["document_name"] == ""
        assert payload["jurisdiction"] == ""
        assert payload["document_status"] == "unknown"

    def test_defaults_when_authority_metadata_none(self):
        from backend.tasks import _chunk_to_payload_dict

        payload = _chunk_to_payload_dict(_chunk(authority_metadata=None), _document())
        assert payload["source_type"] == "other"
        assert payload["authority_score"] == 0.5
        assert payload["binding_authority"] is False
