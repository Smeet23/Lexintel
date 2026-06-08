"""Unit tests for services/llm.py — the unified Gemini/Groq client.

The real google-generativeai / groq SDKs are replaced with in-memory fakes via
sys.modules so these run offline and deterministically. Async entry points are
driven with asyncio.run (project convention; no pytest-asyncio dependency).
"""
import asyncio
import sys
import types
import json as _json

import pytest

from backend.services import llm as llmmod


# ── Fakes ─────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, text="", candidates=("c",)):
        self._text = text
        self.candidates = list(candidates)

    @property
    def text(self):
        return self._text


def _make_fake_genai(*, text="ok", candidates=("c",), raise_exc=None):
    mod = types.ModuleType("google.generativeai")

    class GenerationConfig:
        def __init__(self, **kw):
            self.kw = kw

    class GenerativeModel:
        def __init__(self, model_name=None):
            self.model_name = model_name

        def generate_content(self, prompt, generation_config=None):
            if raise_exc:
                raise raise_exc
            return _Resp(text=text, candidates=candidates)

        async def generate_content_async(self, prompt, generation_config=None):
            if raise_exc:
                raise raise_exc
            return _Resp(text=text, candidates=candidates)

    mod.GenerationConfig = GenerationConfig
    mod.GenerativeModel = GenerativeModel
    mod.configure = lambda **kw: None
    return mod


def _make_fake_groq(*, content="groq-ok", choices=True, raise_exc=None):
    mod = types.ModuleType("groq")

    class _Msg:
        def __init__(self, c):
            self.message = types.SimpleNamespace(content=c)

    class _Client:
        def __init__(self, api_key=None):
            pass

        class _Chat:
            class _Comp:
                @staticmethod
                def create(**kw):
                    if raise_exc:
                        raise raise_exc
                    return types.SimpleNamespace(
                        choices=[_Msg(content)] if choices else []
                    )
            completions = _Comp()
        chat = _Chat()

    mod.Groq = _Client
    return mod


@pytest.fixture
def settings(monkeypatch):
    s = types.SimpleNamespace(
        google_api_key="g-key", gemini_model="gemini-test", groq_api_key="q-key"
    )
    monkeypatch.setattr(llmmod, "get_settings", lambda: s)
    return s


def _install(monkeypatch, *, genai=None, groq=None):
    if genai is not None:
        monkeypatch.setitem(sys.modules, "google.generativeai", genai)
        # `import google.generativeai as g` resolves via getattr(google, 'generativeai'),
        # so once the REAL submodule is imported elsewhere in the suite the sys.modules
        # swap alone is bypassed — also rebind the attribute on the google package.
        import google  # noqa: WPS433
        monkeypatch.setattr(google, "generativeai", genai, raising=False)
    if groq is not None:
        monkeypatch.setitem(sys.modules, "groq", groq)


# ── strip_json_fences / parse_json ─────────────────────────────────────────────

def test_strip_fences_plain():
    assert llmmod.strip_json_fences('{"a":1}') == '{"a":1}'


def test_strip_fences_json_block():
    assert llmmod.strip_json_fences('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_fences_bare_block():
    assert llmmod.strip_json_fences('```\n{"a":1}\n```') == '{"a":1}'


def test_parse_json_direct():
    assert llmmod.parse_json('{"a":1}') == {"a": 1}


def test_parse_json_fenced():
    assert llmmod.parse_json('```json\n{"a":1}\n```') == {"a": 1}


def test_parse_json_raises_on_garbage():
    with pytest.raises(_json.JSONDecodeError):
        llmmod.parse_json("not json at all")


# ── agenerate: Gemini happy paths ──────────────────────────────────────────────

def test_agenerate_gemini_text(settings, monkeypatch):
    _install(monkeypatch, genai=_make_fake_genai(text="hello"))
    assert asyncio.run(llmmod.agenerate("hi")) == "hello"


def test_agenerate_gemini_json(settings, monkeypatch):
    _install(monkeypatch, genai=_make_fake_genai(text='```json\n{"k":2}\n```'))
    assert asyncio.run(llmmod.agenerate("hi", json=True)) == {"k": 2}


# ── agenerate: blocked-response guard ──────────────────────────────────────────

def test_agenerate_blocked_raises(settings, monkeypatch):
    # No candidates → centralised safety-block guard raises; with Groq also
    # failing it surfaces as LLMError.
    _install(
        monkeypatch,
        genai=_make_fake_genai(candidates=()),
        groq=_make_fake_groq(raise_exc=RuntimeError("no groq")),
    )
    with pytest.raises(llmmod.LLMError):
        asyncio.run(llmmod.agenerate("hi"))


# ── agenerate: Gemini→Groq fallback ────────────────────────────────────────────

def test_agenerate_falls_back_to_groq(settings, monkeypatch):
    _install(
        monkeypatch,
        genai=_make_fake_genai(raise_exc=RuntimeError("gemini down")),
        groq=_make_fake_groq(content="from-groq"),
    )
    assert asyncio.run(llmmod.agenerate("hi", provider="gemini", fallback=True)) == "from-groq"


def test_agenerate_no_fallback_raises(settings, monkeypatch):
    _install(monkeypatch, genai=_make_fake_genai(raise_exc=RuntimeError("gemini down")))
    with pytest.raises(llmmod.LLMError):
        asyncio.run(llmmod.agenerate("hi", fallback=False))


# ── agenerate: provider="groq" primary ─────────────────────────────────────────

def test_agenerate_groq_primary(settings, monkeypatch):
    _install(
        monkeypatch,
        genai=_make_fake_genai(text="gemini-text"),
        groq=_make_fake_groq(content="groq-primary"),
    )
    assert asyncio.run(llmmod.agenerate("hi", provider="groq")) == "groq-primary"


# ── generate (sync) ─────────────────────────────────────────────────────────

def test_generate_sync_text(settings, monkeypatch):
    _install(monkeypatch, genai=_make_fake_genai(text="sync-hello"))
    assert llmmod.generate("hi") == "sync-hello"


def test_generate_sync_json(settings, monkeypatch):
    _install(monkeypatch, genai=_make_fake_genai(text='{"v":9}'))
    assert llmmod.generate("hi", json=True) == {"v": 9}


def test_generate_missing_gemini_key_uses_groq(monkeypatch):
    s = types.SimpleNamespace(google_api_key="", gemini_model="m", groq_api_key="q")
    monkeypatch.setattr(llmmod, "get_settings", lambda: s)
    _install(
        monkeypatch,
        genai=_make_fake_genai(text="x"),
        groq=_make_fake_groq(content="groq-no-gemini"),
    )
    assert llmmod.generate("hi") == "groq-no-gemini"


def test_system_prompt_passed_to_groq(settings, monkeypatch):
    captured = {}
    mod = _make_fake_groq(content="ok")
    # wrap create to capture messages
    orig_create = mod.Groq._Chat._Comp.create
    def cap(**kw):
        captured["messages"] = kw.get("messages")
        return orig_create(**kw)
    mod.Groq._Chat._Comp.create = staticmethod(cap)
    _install(monkeypatch, groq=mod)
    out = asyncio.run(llmmod.agenerate("hi", system="You are a judge.", provider="groq", fallback=False))
    assert out == "ok"
    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["system", "user"]
    assert captured["messages"][0]["content"] == "You are a judge."


def test_system_prompt_sets_gemini_system_instruction(settings, monkeypatch):
    captured = {}
    mod = _make_fake_genai(text="ok")
    OrigModel = mod.GenerativeModel
    class CapModel(OrigModel):
        def __init__(self, model_name=None, system_instruction=None):
            captured["system_instruction"] = system_instruction
            super().__init__(model_name=model_name)
    mod.GenerativeModel = CapModel
    _install(monkeypatch, genai=mod)
    out = asyncio.run(llmmod.agenerate("hi", system="SYS", fallback=False))
    assert out == "ok"
    assert captured["system_instruction"] == "SYS"
