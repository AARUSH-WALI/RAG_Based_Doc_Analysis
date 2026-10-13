"""
DocuMind AI — LLM Providers
============================
Streaming generators for every supported LLM backend.  Each ``stream_*``
function yields text deltas so the Streamlit UI can render token-by-token.

Supports:
  • Ollama  — fully offline, local inference
  • Groq   — free-tier cloud (Llama 3)
  • Together AI — free-tier cloud (Mixtral)
  • HuggingFace — free-tier cloud (Phi-3)
  • Extractive — zero-network keyword fallback
"""

from __future__ import annotations

import json
import os
import re
from typing import Generator, Iterable

import requests

from config import (
    ALL_PROVIDERS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    PROVIDER_ENV_VARS,
    PROVIDER_EXTRACTIVE,
    PROVIDER_GROQ,
    PROVIDER_HF,
    PROVIDER_OLLAMA,
    PROVIDER_TOGETHER,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT_TEMPLATE,
)


# ──────────────────────────────────────────────
# Prompt Construction
# ──────────────────────────────────────────────
def _build_messages(question: str, context: str) -> list[dict]:
    """Format the system + user messages for chat-style APIs."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
        {"role": "user",   "content": question},
    ]


def _build_prompt_string(question: str, context: str) -> str:
    """Format an academic prompt for non-chat APIs like Ollama /generate."""
    from agent_orchestrator import AcademicSynthesisAgent
    return AcademicSynthesisAgent.build_professor_prompt(question, context)


# ──────────────────────────────────────────────
# Generic OpenAI-Compatible SSE Parser
# ──────────────────────────────────────────────
def _stream_openai_sse(
    url: str, headers: dict, payload: dict
) -> Generator[str, None, None]:
    """
    POST to an OpenAI-style endpoint with ``stream: true`` and parse
    Server-Sent Events into text deltas.
    """
    payload = {**payload, "stream": True}
    with requests.post(
        url, headers=headers, json=payload, stream=True, timeout=REQUEST_TIMEOUT
    ) as resp:
        if not resp.ok:
            error_msg = resp.text
            raise RuntimeError(f"HTTP {resp.status_code}: {error_msg}")
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line or not raw_line.startswith("data: "):
                continue
            data = raw_line[6:]
            if data.strip() == "[DONE]":
                return
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content")
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            if delta:
                yield delta


# ──────────────────────────────────────────────
# Provider: Groq (Free Cloud — Ultra Fast)
# ──────────────────────────────────────────────
def _stream_groq(
    question: str, context: str, api_key: str, model_name: str = ""
) -> Generator[str, None, None]:
    api_key = api_key.strip()
    candidate_models = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
    ]
    if model_name and model_name in candidate_models:
        candidate_models.remove(model_name)
        candidate_models.insert(0, model_name)
    elif model_name:
        candidate_models.insert(0, model_name)

    last_err = None
    for model_name in candidate_models:
        try:
            gen = _stream_openai_sse(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                payload={
                    "model":       model_name,
                    "messages":    _build_messages(question, context),
                    "temperature": 0.2,
                    "max_tokens":  1500,
                },
            )
            # Try to fetch the first chunk to ensure model exists and is accepted
            first_chunk = next(gen)
            yield first_chunk
            yield from gen
            return
        except StopIteration:
            return
        except Exception as exc:
            last_err = exc
            if "model_not_found" in str(exc) or "does not exist" in str(exc):
                continue
            raise exc

    if last_err:
        raise last_err


# ──────────────────────────────────────────────
# Provider: Together AI (Free Cloud — Mixtral)
# ──────────────────────────────────────────────
def _stream_together(question: str, context: str, api_key: str) -> Generator[str, None, None]:
    yield from _stream_openai_sse(
        "https://api.together.xyz/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        payload={
            "model":      "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages":   _build_messages(question, context),
            "max_tokens": 1024,
        },
    )


# ──────────────────────────────────────────────
# Provider: HuggingFace Inference (Phi-3)
# ──────────────────────────────────────────────
def _call_huggingface(question: str, context: str, api_key: str) -> Generator[str, None, None]:
    """HF Inference API is not reliably streamable — emit full response."""
    resp = requests.post(
        "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "inputs":     f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            "parameters": {"max_new_tokens": 512, "temperature": 0.7},
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, list) and result:
        text = result[0].get("generated_text", "")
        # The model often echoes the prompt — extract only the answer part
        if "Answer:" in text:
            text = text.split("Answer:")[-1].strip()
        yield text
    else:
        yield str(result)


# ──────────────────────────────────────────────
# Provider: Ollama (Local / Offline)
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Provider: Ollama (Local / Offline)
# ──────────────────────────────────────────────
def get_available_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Query local Ollama instance for pulled models."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        if resp.ok:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", [])]
            return [m for m in models if m]
    except Exception:
        pass
    return []


def _stream_ollama(
    question: str,
    context: str,
    model: str = "",
    base_url: str = OLLAMA_BASE_URL,
) -> Generator[str, None, None]:
    """Stream from a locally running Ollama server with automatic model discovery."""
    available = get_available_ollama_models(base_url)
    target_model = model or OLLAMA_MODEL
    
    if available and target_model not in available:
        # Prefer llama3.2 family or first available model
        for candidate in ["llama3.2:1b", "llama3.2", "llama3.1", "mistral", "phi3"]:
            if candidate in available:
                target_model = candidate
                break
        else:
            target_model = available[0]

    with requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model":  target_model,
            "prompt": _build_prompt_string(question, context),
            "stream": True,
        },
        stream=True,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            piece = obj.get("response", "")
            if piece:
                yield piece
            if obj.get("done"):
                return


# ──────────────────────────────────────────────
# Provider: Extractive Fallback (No Network)
# ──────────────────────────────────────────────
def extractive_answer(question: str, context: str) -> str:
    """
    Academic-grade extractive synthesizer. Extracts key definitions,
    theorems, and formulas from the context and structures them into a
    professor-grade report. Works 100% offline with zero external APIs.
    """
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if len(s.strip()) > 15]
    keywords = [w.lower() for w in re.sub(r"[^\w\s]", "", question).split() if len(w) > 2]

    if not sentences or not keywords:
        return (
            "### 🎓 Academic Grounding Summary\n\n"
            "Related document passages were identified, but direct keyword correlation was inconclusive. "
            "Please review the verified source citations below for full technical details."
        )

    # Score by keyword density + academic formula/term indicators
    scored: list[tuple[float, str]] = []
    academic_boosters = ["formula", "area", "sector", "segment", "circle", "radius", "theorem", "marks", "perimeter", "arc", "subtends", "pi", "degree", "chapter", "snapshot"]
    
    for i, sentence in enumerate(sentences):
        lower = sentence.lower()
        keyword_hits = sum(1 for kw in keywords if kw in lower)
        boost = sum(0.5 for b in academic_boosters if b in lower)
        position_bonus = 0.2 * max(0, 1 - i / len(sentences))
        total_score = keyword_hits + boost + position_bonus
        scored.append((total_score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_findings = [s.strip() for score, s in scored[:6] if score > 0]

    if not top_findings:
        return (
            "### 🎓 Academic Context Notice\n\n"
            "Document passages were retrieved, but no high-confidence analytical sentences directly matched. "
            "Inspect the verified source citations below."
        )

    bullet_points = "\n".join(f"- {s}." for s in top_findings[:5])
    return (
        f"### 🎓 Executive Summary & Key Technical Points\n\n"
        f"Based on analytical passage extraction from the uploaded academic materials:\n\n"
        f"{bullet_points}\n\n"
        f"---\n"
        f"💡 **Professor's Pedagogical Note**: The above points represent verified extractions directly grounded "
        f"in the document. Cross-reference with the citations below for full derivations and figures."
    )


# ──────────────────────────────────────────────
# Unified Dispatch
# ──────────────────────────────────────────────
def stream_answer(
    provider: str, question: str, context: str, api_key: str, model_name: str = ""
) -> Iterable[str]:
    """
    Route to the correct provider and yield text deltas.
    Falls back to extractive if no API key is available for a cloud provider.
    """
    if provider == PROVIDER_OLLAMA:
        yield from _stream_ollama(question, context, model=model_name)
        return

    if provider == PROVIDER_GROQ and api_key:
        yield from _stream_groq(question, context, api_key, model_name=model_name)
        return

    if provider == PROVIDER_TOGETHER and api_key:
        yield from _stream_together(question, context, api_key)
        return

    if provider == PROVIDER_HF and api_key:
        yield from _call_huggingface(question, context, api_key)
        return

    # Default: extractive (no network required)
    yield extractive_answer(question, context)


def resolve_api_key(provider: str, ui_input: str) -> str:
    """Prefer the UI-provided key; fall back to the environment variable."""
    if ui_input:
        return ui_input
    env_var = PROVIDER_ENV_VARS.get(provider)
    return os.environ.get(env_var, "") if env_var else ""
