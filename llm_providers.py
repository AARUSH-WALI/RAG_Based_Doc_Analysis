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
from table_analyzer import analyze_tables_for_query


# ──────────────────────────────────────────────
# Prompt Construction
# ──────────────────────────────────────────────
def _build_messages(question: str, context: str) -> list[dict]:
    """Format the system + user messages for chat-style APIs.

    Uses ``.replace()`` instead of ``.format()`` because the template
    contains LaTeX curly braces like ``{Area}``, ``{\\theta}`` that would
    cause ``KeyError`` with Python's ``str.format()``.
    """
    system = SYSTEM_PROMPT_TEMPLATE.replace("{context}", context)
    return [
        {"role": "system", "content": system},
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
    """Stream from a locally running Ollama server using the /api/chat endpoint.
    
    Uses the chat-style API (system + user messages) instead of /api/generate
    for significantly better instruction following with small models.
    """
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

    # Build the system prompt with pre-computed table analysis injected.
    # NOTE: We use .replace() instead of .format() because the template
    # contains LaTeX curly braces like {Area}, {\theta} that would cause
    # KeyError with Python's str.format().
    system_content = SYSTEM_PROMPT_TEMPLATE.replace("{context}", context)

    # Pre-compute table analysis and inject into context
    try:
        table_result = analyze_tables_for_query(context, question, for_llm_prompt=True)
        if table_result:
            system_content += (
                "\n\n--- PRE-COMPUTED TABLE ANALYSIS (use these EXACT numbers) ---\n"
                + table_result
                + "\n--- END PRE-COMPUTED ANALYSIS ---"
            )
    except Exception:
        pass

    with requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model":  target_model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user",   "content": question},
            ],
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
            },
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
            # /api/chat returns {"message": {"content": "..."}}
            msg = obj.get("message", {})
            piece = msg.get("content", "")
            if piece:
                yield piece
            if obj.get("done"):
                return


# ──────────────────────────────────────────────
# Provider: Extractive Fallback (No Network)
# ──────────────────────────────────────────────
def _decompose_into_academic_units(clean_body: str) -> list[str]:
    """
    Deconstructs dense revision sheets, mindmaps, and paragraphs into
    atomic mathematical formulas, definitions, and rules.
    """
    from utils import clean_markdown_math

    # Normalize dividers: bullets, pipes, em-dashes
    text = clean_body.replace("•", "\n").replace("·", "\n").replace("|", "\n").replace("—", " - ")

    # Insert linebreaks before major formula keywords if glued together
    formula_anchors = [
        r"(?=Circumference\b)",
        r"(?=Area of circle\b)",
        r"(?=Area of semicircle\b)",
        r"(?=Area of quadrant\b)",
        r"(?=Area of sector\b)",
        r"(?=Area of major sector\b)",
        r"(?=Area of minor segment\b)",
        r"(?=Area of major segment\b)",
        r"(?=Area of ring\b)",
        r"(?=Length of arc\b)",
        r"(?=Perimeter of sector\b)",
        r"(?=Perimeter =)",
        r"(?=Distance in n revolutions\b)",
        r"(?=Angle swept by\b)",
        r"(?=Scaling r →\b)",
        r"(?=MCQ \(1 mark\))",
        r"(?=Very short\b)",
        r"(?=Short answer\b)",
        r"(?=Long answer\b)",
        r"(?=Case-based\b)",
        r"(?=Common Errors\b)",
        r"(?=Quick tip:)",
        r"(?=Minor Segment\b)",
        r"(?=Major Segment\b)",
        r"(?=Special Angles\b)",
        r"(?=Fifteen one-line facts\b)",
    ]
    for anchor in formula_anchors:
        text = re.sub(anchor, "\n", text, flags=re.IGNORECASE)

    raw_units = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Split further on sentence ends if very long
        if len(line) > 120 and "." in line:
            sub = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if len(s.strip()) > 15]
            raw_units.extend(sub)
        else:
            raw_units.append(line)

    cleaned_units = []
    seen = set()
    for u in raw_units:
        # Strip trailing punctuation / noise
        u_clean = re.sub(r"^[-*•\d\.\s]+\s*", "", u).strip()
        u_clean = re.sub(r"^\[Source\s+\d+[^\]]*\]\s*", "", u_clean).strip()
        u_clean = clean_markdown_math(u_clean)
        
        # Filter out generic noise or titles
        if len(u_clean) < 8 or u_clean.startswith("CBSE Class 10") or u_clean.startswith("Revision Page"):
            continue
        if u_clean.lower() not in seen:
            seen.add(u_clean.lower())
            cleaned_units.append(u_clean)

    return cleaned_units


def extractive_answer(question: str, context: str) -> str:
    """
    **Query-aware** extractive synthesizer for zero-network offline operation.

    Pipeline:
    1. Check if context contains Markdown tables → delegate to Table
       Pre-Computation Engine for programmatic numeric analysis.
    2. If question is a comparison/extremum query → produce ranked table.
    3. Otherwise, filter extracted units by relevance to the question
       and present categorised results.
    """
    # ── Step 1: Try table-based programmatic answer ──
    try:
        table_answer = analyze_tables_for_query(context, question, for_llm_prompt=False)
        if table_answer:
            return (
                f"### 🎓 DocuMind AI — Offline Intelligent Analysis\n\n"
                f"{table_answer}\n\n"
                f"---\n"
                f"💡 **Note**: This answer was computed **programmatically** by scanning every row "
                f"in the document table. No LLM was used — results are mathematically exact.\n\n"
                f"For a natural-language explanation, switch **AI Provider** to **`Ollama (Local LLM)`**."
            )
    except Exception:
        pass

    # ── Step 2: Clean context and decompose ──
    clean_lines = []
    for line in context.split("\n"):
        line_s = line.strip()
        if not line_s or line_s.startswith("[Source ") or "Relevance:" in line_s:
            continue
        clean_lines.append(line_s)
    clean_body = " ".join(clean_lines)

    units = _decompose_into_academic_units(clean_body)

    if not units:
        return (
            "### 🎓 Academic Grounding Summary\n\n"
            "Document passages were identified, but no atomic analytical units could be extracted. "
            "Please review the verified source citations below for full details."
        )

    # ── Step 3: Query-aware relevance filtering ──
    q_words = set(re.sub(r"[^\w\s]", "", question.lower()).split())
    q_words -= {"a", "an", "the", "is", "are", "was", "were", "what", "which",
                "how", "who", "where", "when", "why", "do", "does", "did",
                "in", "of", "to", "for", "and", "or", "it", "my", "your",
                "this", "that", "be", "me", "can", "you", "all"}

    scored_units: list[tuple[float, str]] = []
    for u in units:
        u_lower = u.lower()
        overlap = sum(1 for w in q_words if w in u_lower and len(w) > 2)
        scored_units.append((overlap, u))

    # Sort by relevance, then by original order
    scored_units.sort(key=lambda x: x[0], reverse=True)

    # If there's a clear relevance signal, use top-scoring units
    # Otherwise, fall back to all units (original behavior)
    if scored_units and scored_units[0][0] > 0:
        relevant_units = [u for score, u in scored_units if score > 0][:12]
        remaining_units = [u for score, u in scored_units if score == 0][:4]
        all_display = relevant_units + remaining_units
    else:
        all_display = [u for _, u in scored_units][:12]

    # ── Step 4: Categorize into academic sections ──
    formulas: list[str] = []
    exam_strategy: list[str] = []
    pitfalls: list[str] = []
    concepts: list[str] = []

    for u in all_display:
        u_lower = u.lower()
        if any(term in u_lower for term in ["error", "wrong", "mistake", "tip:", "quick tip", "checklist"]):
            pitfalls.append(u)
        elif any(term in u_lower for term in ["mcq", "very short", "long answer"]):
            exam_strategy.append(u)
        elif any(sym in u for sym in ["=", r"\pi", r"\theta", "r^2", "R^2", r"\frac", r"\sqrt", "area", "perimeter", "circumference", "arc", "formula", "scaling"]):
            formulas.append(u)
        else:
            concepts.append(u)

    # Build structured response
    output_sections = []

    if formulas:
        f_list = "\n".join(f"  - **{f}**" if "=" in f else f"  - {f}" for f in formulas[:10])
        output_sections.append(f"### 📐 Verified Mathematical Formulas & Sheet\n{f_list}")

    if concepts:
        c_list = "\n".join(f"  - {c}" for c in concepts[:8])
        output_sections.append(f"### 🧠 Key Findings & Information\n{c_list}")

    if exam_strategy:
        e_list = "\n".join(f"  - {e}" for e in exam_strategy[:5])
        output_sections.append(f"### ⏱️ Examination Strategy & Marking Breakdown\n{e_list}")

    if pitfalls:
        p_list = "\n".join(f"  - ⚠️ {p}" for p in pitfalls[:4])
        output_sections.append(f"### 💡 Professor's Critical Caveats & Common Errors\n{p_list}")

    if not output_sections:
        fallback_list = "\n".join(f"- {u}" for u in all_display[:8])
        output_sections.append(f"### 🎓 Executive Document Findings\n{fallback_list}")

    joined_report = "\n\n---\n\n".join(output_sections)

    return (
        f"### 🎓 Academic Synthesis — Query: *{question[:80]}{'...' if len(question) > 80 else ''}*\n\n"
        f"The following verified data has been extracted and filtered for relevance to your question:\n\n"
        f"{joined_report}\n\n"
        f"---\n"
        f"💡 **Note on Local Modes**: This response was generated by the zero-network **Extractive Synthesizer**. "
        f"To get a fluid, natural conversational explanation, switch **AI Provider** in the sidebar to **`Ollama (Local LLM)`**."
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
