"""
DocuMind AI — Retrieval Engine
===============================
Implements hybrid search combining dense (FAISS) and sparse (BM25) signals
via Reciprocal Rank Fusion (RRF).  Includes confidence scoring, query
pre-processing, and near-duplicate filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import DEFAULT_BM25_WEIGHT, DEFAULT_MIN_SCORE, DEFAULT_TOP_K, RRF_K_CONSTANT
from embedding_engine import DocumentStore


# ──────────────────────────────────────────────
# Search Result Dataclass
# ──────────────────────────────────────────────
@dataclass
class SearchResult:
    """A single retrieval hit with all contextual information."""
    chunk_id: int
    text: str
    doc_name: str
    pages: List[int]
    score: float
    page_label: str

    @property
    def confidence_pct(self) -> str:
        return f"{self.score:.0%}"


# ──────────────────────────────────────────────
# Query Pre-processing
# ──────────────────────────────────────────────
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "by", "from", "as", "into", "about",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "it", "its", "and", "or", "but", "not", "no", "so", "if",
    "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "how", "when", "where", "why", "tell", "explain", "describe",
})


def preprocess_query(query: str) -> str:
    """
    Lightly clean a user query for better retrieval.
    - Strips excess whitespace
    - The full query is still used for embedding (semantics matter),
      but BM25 benefits from this normalisation.
    """
    return " ".join(query.strip().split())


def _bm25_tokens(text: str) -> List[str]:
    """Tokenise for BM25 matching, removing stop-words."""
    return [w for w in text.lower().split() if w not in _STOP_WORDS and len(w) > 1]


# ──────────────────────────────────────────────
# Near-Duplicate Detection
# ──────────────────────────────────────────────
def _is_near_duplicate(new_text: str, existing_texts: List[str], threshold: float = 0.85) -> bool:
    """Quick Jaccard check — reject chunks that are >threshold similar."""
    new_tokens = set(new_text.lower().split())
    for existing in existing_texts:
        existing_tokens = set(existing.lower().split())
        if not new_tokens or not existing_tokens:
            continue
        intersection = len(new_tokens & existing_tokens)
        union = len(new_tokens | existing_tokens)
        if union > 0 and intersection / union >= threshold:
            return True
    return False


# ──────────────────────────────────────────────
# Hybrid Search with Reciprocal Rank Fusion
# ──────────────────────────────────────────────
def search(
    store: DocumentStore,
    model: SentenceTransformer,
    query: str,
    *,
    k: int = DEFAULT_TOP_K,
    hybrid: bool = False,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    doc_filter: Optional[List[str]] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    deduplicate: bool = True,
) -> List[SearchResult]:
    """
    Retrieve the top-k chunks most relevant to ``query``.

    Parameters
    ----------
    store : DocumentStore
        Must have ``faiss_index`` built.
    model : SentenceTransformer
        Same model used to build the index.
    query : str
        Natural-language question.
    k : int
        Maximum results to return.
    hybrid : bool
        If True, fuse dense + BM25 via Reciprocal Rank Fusion.
    bm25_weight : float
        Weight given to BM25 in the fusion (0–1).  Ignored if *hybrid*
        is False.
    doc_filter : list[str] | None
        Only return results from these documents.
    min_score : float
        Discard results below this score.
    deduplicate : bool
        Remove near-duplicate chunks from results.

    Returns
    -------
    list[SearchResult]
    """
    if not store.chunks or store.faiss_index is None:
        return []

    query = preprocess_query(query)
    n = len(store.chunks)

    # ── Dense retrieval ──────────────────────
    q_emb = np.asarray(model.encode([query]), dtype="float32")
    faiss.normalize_L2(q_emb)
    dense_scores, dense_ids = store.faiss_index.search(q_emb, n)

    # Build rank map: chunk_index → rank (0-based)
    dense_rank: dict[int, int] = {}
    for rank, idx in enumerate(dense_ids[0]):
        dense_rank[int(idx)] = rank

    dense_score_map: dict[int, float] = {
        int(i): float(s) for i, s in zip(dense_ids[0], dense_scores[0])
    }

    if hybrid and store.bm25_index is not None:
        # ── Sparse retrieval ─────────────────
        bm25_raw = np.asarray(
            store.bm25_index.get_scores(query.lower().split())
        )
        bm25_order = np.argsort(-bm25_raw)  # descending
        bm25_rank: dict[int, int] = {int(idx): rank for rank, idx in enumerate(bm25_order)}

        # ── Reciprocal Rank Fusion ───────────
        rrf_scores: dict[int, float] = {}
        for i in range(n):
            dr = dense_rank.get(i, n)
            br = bm25_rank.get(i, n)
            dense_contrib = (1 - bm25_weight) / (RRF_K_CONSTANT + dr)
            bm25_contrib  = bm25_weight / (RRF_K_CONSTANT + br)
            rrf_scores[i] = dense_contrib + bm25_contrib

        # Normalise to 0-1 range
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        scored = [(i, rrf_scores[i] / max_rrf if max_rrf > 0 else 0.0) for i in range(n)]
    else:
        scored = [(i, dense_score_map.get(i, 0.0)) for i in range(n)]

    scored.sort(key=lambda x: x[1], reverse=True)

    # ── Filter & collect results ─────────────
    results: List[SearchResult] = []
    seen_texts: List[str] = []

    for idx, score in scored:
        if score < min_score:
            continue
        chunk = store.chunks[idx]

        if doc_filter and chunk.doc_name not in doc_filter:
            continue

        if deduplicate and _is_near_duplicate(chunk.text, seen_texts):
            continue

        results.append(SearchResult(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            doc_name=chunk.doc_name,
            pages=chunk.pages,
            score=score,
            page_label=chunk.page_label(),
        ))
        seen_texts.append(chunk.text)

        if len(results) >= k:
            break

    return results
