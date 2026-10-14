"""
DocuMind AI — Configuration & Constants
========================================
Centralised settings for the entire RAG pipeline.  Every tuneable parameter
lives here so the rest of the codebase stays free of magic numbers.
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = os.environ.get(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# ──────────────────────────────────────────────
# Chunking Parameters
# ──────────────────────────────────────────────
CHUNK_MAX_WORDS: int = int(os.environ.get("CHUNK_MAX_WORDS", 400))
CHUNK_OVERLAP_SENTENCES: int = int(os.environ.get("CHUNK_OVERLAP_SENTENCES", 2))

# ──────────────────────────────────────────────
# Retrieval Defaults
# ──────────────────────────────────────────────
DEFAULT_TOP_K: int = 10
DEFAULT_MIN_SCORE: float = 0.15
DEFAULT_BM25_WEIGHT: float = 0.4
RRF_K_CONSTANT: int = 60  # Reciprocal Rank Fusion constant

# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────
DATA_DIR: Path = Path(os.environ.get("DATA_DIR", "data"))

# ──────────────────────────────────────────────
# LLM Provider Identifiers
# ──────────────────────────────────────────────
PROVIDER_EXTRACTIVE = "Extractive (Offline — No API)"
PROVIDER_OLLAMA     = "Ollama (Local LLM)"
PROVIDER_GROQ       = "Groq (Free Cloud)"
PROVIDER_HF         = "HuggingFace (Free Cloud)"
PROVIDER_TOGETHER   = "Together AI (Free Cloud)"

ALL_PROVIDERS: list[str] = [
    PROVIDER_EXTRACTIVE,
    PROVIDER_OLLAMA,
    PROVIDER_GROQ,
    PROVIDER_HF,
    PROVIDER_TOGETHER,
]

PROVIDER_ENV_VARS: dict[str, str] = {
    PROVIDER_GROQ:     "GROQ_API_KEY",
    PROVIDER_HF:       "HUGGINGFACE_API_KEY",
    PROVIDER_TOGETHER:  "TOGETHER_API_KEY",
}

PROVIDER_SIGNUP_URLS: dict[str, str] = {
    PROVIDER_GROQ:     "https://console.groq.com",
    PROVIDER_HF:       "https://huggingface.co/settings/tokens",
    PROVIDER_TOGETHER:  "https://api.together.xyz",
    PROVIDER_OLLAMA:    "https://ollama.com",
}

# ──────────────────────────────────────────────
# Ollama Settings
# ──────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.environ.get("OLLAMA_MODEL", "llama3.2")

# ──────────────────────────────────────────────
# Network
# ──────────────────────────────────────────────
REQUEST_TIMEOUT: int = int(os.environ.get("REQUEST_TIMEOUT", 60))

# ──────────────────────────────────────────────
# LLM System Prompt Template (Professor Grade)
# ──────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE: str = """You are DocuMind AI, an elite academic research assistant and university professor.
Synthesize an authoritative, rigorously structured, and comprehensive response to the user's inquiry based strictly on the verified document passages below.

--- ACADEMIC PRESENTATION & FORMATTING GUIDELINES ---
1. Executive Summary:
   - Begin with a direct, clear 1–2 sentence answer or formal definition.
2. Mathematical Formulations & Rigor:
   - Standalone formulas: MUST use LaTeX notation wrapped in double dollar signs:
     $$ \\text{Area} = \\frac{\\theta}{360^\\circ} \\times \\pi r^2 $$
   - Inline variables/symbols: MUST use single dollar signs: e.g. $r$, $\\theta$, $\\pi$, $cm^2$.
   - STRICT PROHIBITION: NEVER wrap formulas in raw brackets like [ \\boxed{...} ] or [ ... ].
3. Structured Technical Exposition:
   - Present principles, prerequisites, theorems, and proofs with clean bullet points or numbered steps.
   - Use Markdown comparison tables when classifying question types, marks, formulas, or conditions.
4. Source Grounding & Precise Page Citations:
   - Rigorously cite source page numbers at the point of assertion, e.g. *(Source: Document Name, Page X)*.
5. Tabular & Numerical Extremum Analysis:
   - For questions comparing prices, quantities, costs, or maximum/minimum values (e.g. "which is most expensive", "cheapest", "highest"):
     * Systematically check EVERY row in the table before declaring a top item.
     * Differentiate explicitly between "Price per Piece / Unit Price" and "Total Price".
     * Cite the exact numbers from the table. Never invent or hallucinate digits.
     * Provide a clear ranked comparison of the top items.
6. Professor's Pedagogical Commentary:
   - Conclude with an educational insight, typical student pitfalls, or examination tips based on the text.
7. Zero Mojibake:
   - Never output corrupted symbols (e.g. â€”, â€¢, Â).

--- VERIFIED CONTEXT PASSAGES ---
{context}
--- END CONTEXT ---"""
