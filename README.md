# 🧠 DocuMind AI — Intelligent Document Question Answering

> A Retrieval-Augmented Generation (RAG) system that lets you upload PDF documents and ask questions in natural language. Built with sentence-aware chunking, hybrid retrieval (FAISS + BM25), page-level citations, and full offline support via Ollama.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

DocuMind AI is an AI-powered document assistant that extracts text from your PDFs using PyMuPDF, splits it into semantically meaningful chunks at sentence boundaries, and indexes them with both dense embeddings (FAISS) and sparse keyword matching (BM25). At query time, it fuses both retrieval signals using Reciprocal Rank Fusion (RRF) and streams answers from your chosen LLM — complete with page-level source citations.

The system works **fully offline** using Ollama for local LLM inference, or connects to free cloud providers (Groq, Together AI, HuggingFace) when an API key is provided. A zero-dependency extractive fallback is also available.

---

## Features

- **Multi-PDF ingestion** with PyMuPDF (fast C-based extraction)
- **Sentence-aware chunking** that preserves semantic boundaries
- **Page-level tracking** — every chunk knows which page(s) it comes from
- **Hybrid retrieval** via Reciprocal Rank Fusion (FAISS dense + BM25 sparse)
- **Near-duplicate filtering** to reduce redundant results
- **Five LLM backends:**
  - **Ollama** (local, fully offline) — recommended for private documents
  - **Groq** (free cloud, Llama 3 inference)
  - **HuggingFace** (free cloud, Phi-3-mini)
  - **Together AI** (free cloud, Mixtral-8x7B)
  - **Extractive fallback** (no network, no API key)
- **Streaming responses** for Groq, Together AI, and Ollama
- **Confidence scoring** with visual quality indicators
- **Source citations** with document name, page number, and relevance score
- **Persistent vector store** — save/load indexes to disk
- **Chat export** to Markdown or PDF
- **Premium dark UI** with glassmorphism and aurora animations

---

## Architecture

```
                      ┌─────────────────────────────┐
       PDF Upload ───►│     PyMuPDF (fitz)          │
                      │  Page-level text extraction  │
                      └───────────┬─────────────────┘
                                  │
                                  ▼
                      ┌─────────────────────────────┐
                      │  Sentence-Aware Chunking     │
                      │  (preserves page references) │
                      └───────────┬─────────────────┘
                                  │
                      ┌───────────┴─────────────┐
                      ▼                         ▼
             ┌──────────────────┐     ┌──────────────────┐
             │   Dense Index    │     │  Sparse Index     │
             │  (FAISS FlatIP)  │     │  (BM25 Okapi)     │
             └────────┬─────────┘     └────────┬─────────┘
                      │                        │
                      └───────────┬────────────┘
                                  ▼
                      ┌─────────────────────────────┐
                      │  Reciprocal Rank Fusion      │
                      │  + Deduplication + Filtering  │
                      └───────────┬─────────────────┘
                                  │
                                  ▼
                      ┌─────────────────────────────┐
                      │  LLM Answer Generation       │
                      │  (Ollama / Groq / HF / etc.) │
                      └───────────┬─────────────────┘
                                  │
                                  ▼
                      ┌─────────────────────────────┐
                      │  Streamlit UI                │
                      │  Answer + Page Citations     │
                      └─────────────────────────────┘
```

---

## Project Structure

```
RAG/
├── app.py                    # Streamlit web application
├── config.py                 # Centralised configuration & constants
├── document_processor.py     # PDF extraction (PyMuPDF) + sentence chunking
├── embedding_engine.py       # FAISS/BM25 indexing + DocumentStore + persistence
├── retrieval_engine.py       # Hybrid search with RRF + confidence scoring
├── llm_providers.py          # LLM backends (Ollama, Groq, HF, Together, extractive)
├── utils.py                  # Export helpers + quality indicators
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore
└── data/                     # Persisted indexes (auto-created)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo-url>
cd RAG
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
copy .env.example .env
# Edit .env to add API keys or change defaults
```

### 3. Run

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

### 4. For offline mode (Ollama)

```bash
# Install Ollama from https://ollama.com
ollama serve          # start the server
ollama pull llama3.2  # download a model
```

Then select **"Ollama (Local LLM)"** in the sidebar.

---

## Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Extraction | PyMuPDF (fitz) | Fast, C-based text extraction with page tracking |
| Embeddings | Sentence Transformers (MiniLM-L6-v2) | Local text embeddings (~80MB model) |
| Dense Index | FAISS (IndexFlatIP) | Semantic similarity search |
| Sparse Index | BM25Okapi (rank-bm25) | Keyword-based retrieval |
| Fusion | Reciprocal Rank Fusion | Merging dense + sparse rankings |
| Local LLM | Ollama | Fully offline inference (Llama 3.2, etc.) |
| Cloud LLMs | Groq, Together AI, HuggingFace | Free-tier API inference |
| Web UI | Streamlit | Interactive web application |
| Persistence | FAISS + JSON | Save/load indexes to disk |

---

## How It Works

1. **Upload** — PDFs are processed page-by-page using PyMuPDF
2. **Chunk** — Text is split at sentence boundaries (not arbitrary word counts)
3. **Index** — Chunks are embedded with MiniLM-L6-v2 and indexed in FAISS; tokens are also indexed with BM25
4. **Search** — User queries are matched against both indexes; results are fused via Reciprocal Rank Fusion
5. **Answer** — The top-k chunks are fed to the selected LLM with page references in the context
6. **Cite** — The UI displays source passages with document name, page number, and confidence bars

---

## License

MIT License — see [LICENSE](LICENSE) for details.
