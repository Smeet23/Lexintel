"""Unified LLM client — the single place that owns Gemini/Groq access.

Every service used to hand-roll the same boilerplate: ``genai.configure`` +
``GenerativeModel`` + ``generate_content[_async]`` + a bespoke ``response.text``
guard + ad-hoc ```` ```json ```` fence stripping + an inconsistent Groq fallback.
That meant a single safety-block bug (empty ``candidates`` → ``response.text``
raising) had to be fixed in ~13 places, and model swaps touched every file.

This module centralises all of it behind two calls — :func:`agenerate` (async)
and :func:`generate` (sync) — plus the :func:`parse_json` / :func:`strip_json_fences`
helpers. Behaviour contract:

* On a safety-blocked / empty Gemini response it raises :class:`LLMBlocked`
  (callers already wrap LLM calls in ``try/except`` and degrade to a default —
  raising preserves that behaviour rather than silently returning empties).
* When every configured provider fails it raises :class:`LLMError`.
* ``json=True`` requests JSON mode from the provider and returns the parsed
  object (tolerant of code-fence wrapping); otherwise returns the raw text.
* ``provider`` selects the primary backend and ``fallback=True`` tries the other
  backend when the primary fails (Gemini⇄Groq), matching the two orderings the
  codebase already relied on.
"""
from __future__ import annotations

import asyncio
import json as _json
import logging
import re
from typing import Any, Optional

try:  # tolerate either package layout until the import-fallback cleanup lands
    from backend.config import get_settings
except ImportError:  # pragma: no cover - layout fallback
    try:
        from config import get_settings
    except ImportError:  # pragma: no cover - layout fallback
        from ..config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class LLMError(Exception):
    """No configured LLM provider could satisfy the request."""


class LLMBlocked(LLMError):
    """The provider returned an empty / safety-blocked response (no candidates)."""


# ── JSON helpers ──────────────────────────────────────────────────────────────

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n?", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$", re.MULTILINE)


def strip_json_fences(text: str) -> str:
    """Remove a leading ```` ```json ```` / ```` ``` ```` fence and trailing ```` ``` ````."""
    cleaned = _FENCE_OPEN.sub("", (text or "").strip())
    cleaned = _FENCE_CLOSE.sub("", cleaned.strip())
    return cleaned.strip()


def parse_json(text: str) -> Any:
    """Parse JSON from a model response, tolerating markdown code fences.

    Tries a direct parse first (the common case when JSON mode is honoured),
    then falls back to fence-stripping. Raises ``json.JSONDecodeError`` if both
    fail so the caller can degrade exactly as before.
    """
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        return _json.loads(strip_json_fences(text))


# ── Provider calls ──────────────────────────────────────────────────────────

def _gemini_text(
    prompt: str, *, json: bool, temperature: float,
    max_output_tokens: Optional[int], model: Optional[str], sync: bool,
    system: Optional[str] = None,
):
    """Build the Gemini request. Returns (coroutine_or_response, model_obj).

    Kept tiny so both sync and async paths share config building. The blocked
    guard is applied by the callers on the resolved response.
    """
    import google.generativeai as genai

    settings = get_settings()
    if not settings.google_api_key:
        raise LLMError("Gemini unavailable: no google_api_key configured")
    genai.configure(api_key=settings.google_api_key)
    model_kwargs = {"model_name": model or settings.gemini_model}
    if system:
        model_kwargs["system_instruction"] = system
    gmodel = genai.GenerativeModel(**model_kwargs)

    cfg_kwargs: dict = {"temperature": temperature}
    if max_output_tokens is not None:
        cfg_kwargs["max_output_tokens"] = max_output_tokens
    if json:
        cfg_kwargs["response_mime_type"] = "application/json"
    gen_config = genai.GenerationConfig(**cfg_kwargs)

    if sync:
        return gmodel.generate_content(prompt, generation_config=gen_config)
    return gmodel.generate_content_async(prompt, generation_config=gen_config)


def _gemini_response_text(response) -> str:
    """Extract text from a Gemini response, guarding the empty/blocked case."""
    if not getattr(response, "candidates", None):
        raise LLMBlocked("Empty/blocked Gemini response (no candidates)")
    text = (response.text or "").strip()
    if not text:
        raise LLMBlocked("Empty Gemini response text")
    return text


def _groq_text_sync(
    prompt: str, *, json: bool, temperature: float,
    max_output_tokens: Optional[int], groq_model: str,
    system: Optional[str] = None,
) -> str:
    """Call Groq synchronously and return the message text (guards empty choices)."""
    from groq import Groq

    settings = get_settings()
    groq_key = getattr(settings, "groq_api_key", "") or ""
    if not groq_key:
        raise LLMError("Groq unavailable: no groq_api_key configured")
    client = Groq(api_key=groq_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    kwargs: dict = {
        "model": groq_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    if json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    if not getattr(resp, "choices", None):
        raise LLMBlocked("Empty Groq response (no choices)")
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise LLMBlocked("Empty Groq response content")
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def _finish(text: str, json: bool) -> Any:
    return parse_json(text) if json else text


async def agenerate(
    prompt: str,
    *,
    system: Optional[str] = None,
    json: bool = False,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    provider: str = "gemini",
    fallback: bool = True,
    groq_model: str = _DEFAULT_GROQ_MODEL,
) -> Any:
    """Generate text/JSON, async. ``provider`` is the primary backend; when it
    fails and ``fallback`` is set the other backend is tried. ``system`` is the
    system prompt (Gemini ``system_instruction`` / Groq system message). Raises
    :class:`LLMError` if all providers fail.
    """
    order = ["gemini", "groq"] if provider == "gemini" else ["groq", "gemini"]
    if not fallback:
        order = order[:1]

    last_exc: Optional[Exception] = None
    for backend in order:
        try:
            if backend == "gemini":
                coro = _gemini_text(
                    prompt, json=json, temperature=temperature,
                    max_output_tokens=max_output_tokens, model=model, sync=False,
                    system=system,
                )
                response = await (asyncio.wait_for(coro, timeout) if timeout else coro)
                return _finish(_gemini_response_text(response), json)
            else:  # groq is sync SDK → offload to a thread
                text = await asyncio.to_thread(
                    _groq_text_sync, prompt, json=json, temperature=temperature,
                    max_output_tokens=max_output_tokens, groq_model=groq_model,
                    system=system,
                )
                return _finish(text, json)
        except asyncio.TimeoutError:
            raise  # caller decides how to treat a timeout (often a distinct path)
        except Exception as exc:  # try the next backend
            last_exc = exc
            logger.debug(f"LLM provider {backend!r} failed: {exc}")
            continue
    raise LLMError(f"All LLM providers failed: {last_exc}")


def generate(
    prompt: str,
    *,
    system: Optional[str] = None,
    json: bool = False,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
    provider: str = "gemini",
    fallback: bool = True,
    groq_model: str = _DEFAULT_GROQ_MODEL,
) -> Any:
    """Generate text/JSON, synchronous (for non-async call sites)."""
    order = ["gemini", "groq"] if provider == "gemini" else ["groq", "gemini"]
    if not fallback:
        order = order[:1]

    last_exc: Optional[Exception] = None
    for backend in order:
        try:
            if backend == "gemini":
                response = _gemini_text(
                    prompt, json=json, temperature=temperature,
                    max_output_tokens=max_output_tokens, model=model, sync=True,
                    system=system,
                )
                return _finish(_gemini_response_text(response), json)
            else:
                text = _groq_text_sync(
                    prompt, json=json, temperature=temperature,
                    max_output_tokens=max_output_tokens, groq_model=groq_model,
                    system=system,
                )
                return _finish(text, json)
        except Exception as exc:
            last_exc = exc
            logger.debug(f"LLM provider {backend!r} failed: {exc}")
            continue
    raise LLMError(f"All LLM providers failed: {last_exc}")
