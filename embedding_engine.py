"""
DocuMind AI — Embedding Engine
===============================
Manages the dual-index system (FAISS for dense vectors, BM25 for sparse
keyword matching) and on-disk persistence.  The ``DocumentStore`` is the
central data structure shared across the pipeline.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, EMBEDDING_MODEL_NAME
from document_processor import ChunkRecord, DocumentMeta


# ──────────────────────────────────────────────
# Document Store
# ──────────────────────────────────────────────
@dataclass
class DocumentStore:
    """
    Central in-memory store holding every ingested chunk together with
    its FAISS dense index and BM25 sparse index.
    """
    chunks: List[ChunkRecord] = field(default_factory=list)
    doc_metadata: Dict[str, dict] = field(default_factory=dict)
    faiss_index: Optional[faiss.Index] = None
    bm25_index: Optional[BM25Okapi] = None

    # ── convenience helpers ──────────────────
    @property
    def document_names(self) -> List[str]:
        """Sorted list of unique document names."""
        return sorted(set(c.doc_name for c in self.chunks))

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def has_document(self, name: str) -> bool:
        return name in self.doc_metadata

    def add_document(self, chunks: List[ChunkRecord], meta: DocumentMeta) -> None:
        """Append chunks and metadata for a newly ingested document."""
        # Re-number chunk IDs globally
        offset = len(self.chunks)
        for i, c in enumerate(chunks):
            c.chunk_id = offset + i
        self.chunks.extend(chunks)
        self.doc_metadata[meta.doc_name] = {
            "total_pages":  meta.total_pages,
            "total_chars":  meta.total_chars,
            "total_words":  meta.total_words,
            "num_chunks":   meta.num_chunks,
            "ingested_at":  meta.ingested_at,
        }


# ──────────────────────────────────────────────
# Embedding Model (cached at module level)
# ──────────────────────────────────────────────
def load_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model.
    Prioritises the local offline cache first to guarantee zero-network
    operation when Wi-Fi is disconnected.
    """
    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ──────────────────────────────────────────────
# Index Construction
# ──────────────────────────────────────────────
def build_indexes(store: DocumentStore, model: SentenceTransformer) -> None:
    """
    (Re)build both the dense FAISS index and the sparse BM25 index
    from whatever chunks are currently in the store.
    """
    if not store.chunks:
        store.faiss_index = None
        store.bm25_index = None
        return

    texts = [c.text for c in store.chunks]

    # ── Dense (FAISS) ────────────────────────
    embeddings = np.asarray(
        model.encode(texts, show_progress_bar=False, batch_size=32),
        dtype="float32",
    )
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    store.faiss_index = index

    # ── Sparse (BM25) ────────────────────────
    tokenised = [t.lower().split() for t in texts]
    store.bm25_index = BM25Okapi(tokenised)


# ──────────────────────────────────────────────
# Persistence — Save / Load / List / Delete
# ──────────────────────────────────────────────
def save_store(store: DocumentStore, name: str) -> Path:
    """Write the FAISS index + chunk data + metadata to ``data/<name>/``."""
    if store.faiss_index is None:
        raise ValueError("Cannot save an empty store — build indexes first.")

    folder = DATA_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    faiss.write_index(store.faiss_index, str(folder / "faiss.index"))

    payload = {
        "chunks": [
            {
                "chunk_id":   c.chunk_id,
                "text":       c.text,
                "doc_name":   c.doc_name,
                "pages":      c.pages,
                "word_count": c.word_count,
            }
            for c in store.chunks
        ],
        "doc_metadata": store.doc_metadata,
    }
    (folder / "store.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return folder


def load_store(name: str) -> DocumentStore:
    """Restore a previously saved store.  BM25 is rebuilt (it's cheap)."""
    folder = DATA_DIR / name
    raw = json.loads((folder / "store.json").read_text(encoding="utf-8"))

    chunks = [
        ChunkRecord(
            chunk_id=c["chunk_id"],
            text=c["text"],
            doc_name=c["doc_name"],
            pages=c["pages"],
            word_count=c["word_count"],
        )
        for c in raw["chunks"]
    ]
    store = DocumentStore(
        chunks=chunks,
        doc_metadata=raw["doc_metadata"],
    )
    store.faiss_index = faiss.read_index(str(folder / "faiss.index"))

    # Rebuild BM25 from text (small cost, avoids pickling issues)
    if chunks:
        tokenised = [c.text.lower().split() for c in chunks]
        store.bm25_index = BM25Okapi(tokenised)

    return store


def list_saved_stores() -> List[str]:
    """Return names of stores that have been persisted to disk."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_dir() and (p / "faiss.index").exists()
    )


def delete_store(name: str) -> None:
    """Remove a persisted store from disk."""
    folder = DATA_DIR / name
    if folder.exists():
        shutil.rmtree(folder)
