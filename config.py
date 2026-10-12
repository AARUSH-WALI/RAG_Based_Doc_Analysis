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
# LLM System Prompt Template
# ──────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE: str = """You are DocuMind AI, an intelligent, professional document assistant.
Answer the user's question accurately using the provided context passages.

Formatting and quality rules:
1. Provide a well-structured, clear, and comprehensive response.
2. For mathematical formulas, equations, or variables:
   - Use standard LaTeX notation enclosed in double dollar signs for standalone formulas:
     $$A = \\pi r^2 = \\frac{\\pi d^2}{4}$$
   - Use single dollar signs for inline variables, e.g. $r$, $d = 2r$.
   - Do NOT wrap formulas in raw brackets like [ \\boxed{{...}} ] or [ ... ].
3. Present formulas and lists with clean bullet points or organized sections.
4. Always cite specific page numbers clearly, e.g. *(Source: Page 5)*.
5. Never output corrupted encoding characters (such as â€” or broken symbols).
6. Base the answer strictly on the document context provided below.

--- CONTEXT PASSAGES ---
{context}
--- END CONTEXT ---"""
