"""
DocuMind AI — Document Processor
=================================
Handles PDF text extraction via PyMuPDF (fitz) and sentence-aware chunking
that preserves semantic boundaries.  Every chunk records which page(s) it
originates from so downstream citations can point users to the right place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

import fitz  # PyMuPDF

from config import CHUNK_MAX_WORDS, CHUNK_OVERLAP_SENTENCES


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────
@dataclass
class PageContent:
    """Raw text extracted from a single PDF page."""
    page_number: int          # 1-indexed
    text: str


@dataclass
class ChunkRecord:
    """A text chunk with provenance metadata."""
    chunk_id: int
    text: str
    doc_name: str
    pages: List[int]          # which page(s) this chunk spans
    word_count: int

    def page_label(self) -> str:
        """Human-readable page reference, e.g. 'p.3' or 'pp.3-5'."""
        if not self.pages:
            return ""
        if len(self.pages) == 1:
            return f"p.{self.pages[0]}"
        return f"pp.{self.pages[0]}-{self.pages[-1]}"


@dataclass
class DocumentMeta:
    """Metadata collected during ingestion of a single PDF."""
    doc_name: str
    total_pages: int
    total_chars: int
    total_words: int
    num_chunks: int
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ──────────────────────────────────────────────
# PDF Extraction (PyMuPDF)
# ──────────────────────────────────────────────
def extract_pages(pdf_bytes: bytes) -> List[PageContent]:
    """
    Open a PDF from raw bytes and return page-level text.
    Extracts structured Markdown tables using PyMuPDF's table finder
    alongside continuous body text, preserving column-row alignments.
    """
    pages: List[PageContent] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]

            # 1. Extract structured tables if present
            table_markdowns: list[str] = []
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    md = tab.to_markdown()
                    if md and len(md.strip()) > 10:
                        table_markdowns.append(md.strip())
            except Exception:
                pass

            # 2. Extract standard text
            raw_text = page.get_text("text")
            cleaned = _clean_text(raw_text)

            # 3. If structured tables were found, integrate them cleanly
            if table_markdowns:
                table_block = "\n\n### Extracted Table Data:\n" + "\n\n".join(table_markdowns)
                cleaned = f"{cleaned}\n{table_block}"

            if cleaned:
                pages.append(PageContent(page_number=page_idx + 1, text=cleaned))
    finally:
        doc.close()
    return pages


def _clean_text(text: str) -> str:
    """Normalise whitespace, fix mojibake artifacts, and strip control characters."""
    # Fix common UTF-8 encoding misinterpretations (mojibake)
    mojibake_map = {
        "â": "—",
        "â": "–",
        "â": "'",
        "â": "'",
        "â": '"',
        "â": '"',
        "â¢": "•",
        "â": "→",
        "â": "⇒",
        "â": "≈",
        "â ": "≠",
        "â": "°",
        "â¦": "…",
        "â": "✓",
        "Â": "",
        "â": "",
    }
    for bad, good in mojibake_map.items():
        if bad in text:
            text = text.replace(bad, good)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)   # control chars
    text = re.sub(r"[ \t]+", " ", text)                           # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                        # excess newlines
    return text.strip()


# ──────────────────────────────────────────────
# Sentence-Aware Chunking
# ──────────────────────────────────────────────
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'   # split after sentence-ending punctuation
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristics."""
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def chunk_document(
    pages: List[PageContent],
    doc_name: str,
    max_words: int = CHUNK_MAX_WORDS,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> tuple[List[ChunkRecord], DocumentMeta]:
    """
    Build sentence-aware chunks from extracted pages.

    Unlike naive word-count splitting, this groups *complete sentences*
    until the word budget is reached, then starts a new chunk.  The last
    ``overlap_sentences`` sentences carry over into the next chunk to
    preserve context across boundaries.

    Returns
    -------
    chunks : list[ChunkRecord]
    meta   : DocumentMeta
    """
    # First, build a list of (sentence, page_number) pairs
    sentence_page_pairs: List[tuple[str, int]] = []
    total_chars = 0
    total_words = 0

    for page in pages:
        total_chars += len(page.text)
        total_words += len(page.text.split())
        for sentence in _split_sentences(page.text):
            sentence_page_pairs.append((sentence, page.page_number))

    if not sentence_page_pairs:
        meta = DocumentMeta(
            doc_name=doc_name,
            total_pages=len(pages),
            total_chars=total_chars,
            total_words=total_words,
            num_chunks=0,
        )
        return [], meta

    # Greedy grouping: add sentences until word budget is exceeded
    chunks: List[ChunkRecord] = []
    chunk_id = 0
    idx = 0
    n = len(sentence_page_pairs)

    while idx < n:
        current_sentences: List[str] = []
        current_pages: List[int] = []
        current_word_count = 0

        while idx < n:
            sentence, page_num = sentence_page_pairs[idx]
            sentence_words = len(sentence.split())
            if current_word_count + sentence_words > max_words and current_sentences:
                break  # budget exceeded — start new chunk
            current_sentences.append(sentence)
            if page_num not in current_pages:
                current_pages.append(page_num)
            current_word_count += sentence_words
            idx += 1

        chunk_text = " ".join(current_sentences)
        chunks.append(ChunkRecord(
            chunk_id=chunk_id,
            text=chunk_text,
            doc_name=doc_name,
            pages=sorted(current_pages),
            word_count=current_word_count,
        ))
        chunk_id += 1

        # Overlap: rewind by overlap_sentences so context carries over
        if idx < n and overlap_sentences > 0:
            rewind = min(overlap_sentences, len(current_sentences) - 1)
            idx -= rewind

    meta = DocumentMeta(
        doc_name=doc_name,
        total_pages=len(pages),
        total_chars=total_chars,
        total_words=total_words,
        num_chunks=len(chunks),
    )
    return chunks, meta
