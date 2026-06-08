"""Tests for the agentic-loop context-management / summarisation node.

context_manager_node summarises OLDER conversation turns once history exceeds a
threshold, keeping recent turns verbatim, so the LangGraph loop's prompt context
stays bounded. The LLM call is mocked (offline, deterministic).
"""
import asyncio

import pytest

from backend.services import agentic_rag as ar
from backend.services import llm as llmmod


def _history(n):
    return [{"question": f"Q{i}", "answer": f"A{i} " * 50} for i in range(n)]


def _run(state):
    return asyncio.run(ar.context_manager_node(state))


def test_short_history_is_noop(monkeypatch):
    called = {"n": 0}

    async def fake_agen(*a, **k):
        called["n"] += 1
        return "SUMMARY"

    monkeypatch.setattr(llmmod, "agenerate", fake_agen)
    out = _run({"conversation_history": _history(3)})
    assert out == {}              # below summarize_after (6) → no change
    assert called["n"] == 0       # LLM not invoked


def test_long_history_summarised(monkeypatch):
    async def fake_agen(prompt, **k):
        assert "CONVERSATION:" in prompt
        return "Earlier: parties A & B; payment $50k; governing law Delaware."

    monkeypatch.setattr(llmmod, "agenerate", fake_agen)
    out = _run({"conversation_history": _history(10)})
    managed = out["conversation_history"]
    # summary turn + keep_recent(4) recent turns
    assert managed[0]["question"] == "[Summary of earlier conversation]"
    assert "Delaware" in managed[0]["answer"]
    assert len(managed) == 1 + 4
    # the 4 verbatim turns are the MOST RECENT ones (Q6..Q9)
    assert [t["question"] for t in managed[1:]] == ["Q6", "Q7", "Q8", "Q9"]
    assert out["conversation_summary"].startswith("Earlier:")


def test_llm_failure_is_noop(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("groq down")

    monkeypatch.setattr(llmmod, "agenerate", boom)
    out = _run({"conversation_history": _history(10)})
    assert out == {}              # best-effort: original history left untouched


def test_empty_summary_is_noop(monkeypatch):
    async def empty(*a, **k):
        return "   "

    monkeypatch.setattr(llmmod, "agenerate", empty)
    out = _run({"conversation_history": _history(10)})
    assert out == {}
