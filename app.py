"""
DocuMind AI — Main Application
================================
Streamlit-based web interface for the intelligent document QA system.
Features a premium dark glassmorphism design with aurora effects.
"""

import os
from datetime import datetime
from typing import List

import streamlit as st

from config import (
    ALL_PROVIDERS,
    PROVIDER_ENV_VARS,
    PROVIDER_EXTRACTIVE,
    PROVIDER_GROQ,
    PROVIDER_OLLAMA,
    PROVIDER_SIGNUP_URLS,
)
from agent_orchestrator import (
    AcademicSynthesisAgent,
    AgentInspectionTrace,
    IntentAgent,
    PassageInspectorAgent,
    QueryExpansionAgent,
    QueryIntent,
)
from document_processor import chunk_document, extract_pages
from embedding_engine import (
    DocumentStore,
    build_indexes,
    delete_store,
    list_saved_stores,
    load_embedding_model,
    load_store,
    save_store,
)
from llm_providers import (
    extractive_answer,
    get_available_ollama_models,
    resolve_api_key,
    stream_answer,
)
from retrieval_engine import search
from utils import (
    answer_quality_label,
    build_context_string,
    chat_to_markdown,
    chat_to_pdf_bytes,
    clean_markdown_math,
)

# ═══════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="DocuMind AI — Intelligent Document QA",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
# PREMIUM DARK GLASSMORPHISM CSS
# ═══════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ─────────────────────── */
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #141229 30%, #1a1040 60%, #0d1b2a 100%);
    }

    /* ── Sidebar ──────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMetric label,
    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
    }

    /* ── Aurora Header ────────────────────── */
    .aurora-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 30%, #f093fb 60%, #667eea 100%);
        background-size: 300% 300%;
        animation: aurora-shift 8s ease infinite;
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.35);
    }
    .aurora-header::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
        animation: aurora-glow 6s ease-in-out infinite alternate;
    }
    .aurora-header h1 {
        margin: 0; font-size: 2.5rem; font-weight: 800;
        position: relative; z-index: 2;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .aurora-header p {
        margin: 0.5rem 0 0 0; font-size: 1.05rem; opacity: 0.9;
        position: relative; z-index: 2; font-weight: 400;
    }
    @keyframes aurora-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes aurora-glow {
        0% { transform: translate(-10%, -10%) rotate(0deg); }
        100% { transform: translate(10%, 10%) rotate(180deg); }
    }

    /* ── Glass Cards ──────────────────────── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(139, 92, 246, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(139, 92, 246, 0.15);
    }

    /* ── Stat Cards ───────────────────────── */
    .stat-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stat-card:hover {
        background: rgba(255, 255, 255, 0.07);
        border-color: rgba(139, 92, 246, 0.4);
        transform: translateY(-5px);
        box-shadow: 0 10px 35px rgba(139, 92, 246, 0.2);
    }
    .stat-card .stat-icon { font-size: 1.75rem; margin-bottom: 0.3rem; }
    .stat-card .stat-value {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #a78bfa, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-card .stat-label {
        font-size: 0.8rem; color: #94a3b8; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem;
    }

    /* ── Section Titles ───────────────────── */
    .section-title {
        color: #e2e8f0;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Citation Cards ───────────────────── */
    .citation-card {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.3s ease;
    }
    .citation-card:hover {
        border-color: rgba(139, 92, 246, 0.45);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.12);
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .citation-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .citation-page {
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .citation-score {
        color: #94a3b8;
        font-size: 0.75rem;
    }
    .citation-text {
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* ── Confidence Bar ───────────────────── */
    .confidence-bar-container {
        background: rgba(255,255,255,0.06);
        border-radius: 8px;
        height: 6px;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #667eea, #a78bfa, #f093fb);
        transition: width 0.6s ease;
    }

    /* ── Info / Success / Warning Boxes ──── */
    .info-glass {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #c7d2fe;
    }
    .success-glass {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        color: #a7f3d0;
    }

    /* ── Upload Area ──────────────────────── */
    .upload-zone {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #94a3b8;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(139, 92, 246, 0.6);
        background: rgba(139, 92, 246, 0.05);
    }

    /* ── Buttons ──────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 1.8rem !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.45) !important;
    }

    /* ── Chat Styling ─────────────────────── */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
    }

    /* ── Global Text Color ────────────────── */
    .stMarkdown, .stText, p, li, span {
        color: #e2e8f0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }

    /* ── Footer ───────────────────────────── */
    .footer-glass {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 3rem;
    }
    .footer-glass p {
        margin: 0.2rem 0;
    }
    .footer-brand {
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .footer-sub {
        font-size: 0.8rem; color: #64748b;
    }

    /* ── Quality Badge ────────────────────── */
    .quality-badge {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .quality-green  { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
    .quality-orange { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
    .quality-red    { background: rgba(239,68,68,0.15);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }

    /* ── Agent Inspection Badges & Cards ─── */
    .agent-trace-box {
        background: rgba(30, 27, 75, 0.45);
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-radius: 14px;
        padding: 0.9rem 1.25rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .agent-trace-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #c4b5fd;
        margin-bottom: 0.45rem;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem;
    }
    .agent-step-pill {
        display: inline-block;
        background: rgba(139, 92, 246, 0.2);
        color: #ddd6fe;
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 0.2rem 0.65rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
    }
    .agent-stat-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.2rem 0.65rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════
if "store" not in st.session_state:
    st.session_state.store = DocumentStore()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0


# ═══════════════════════════════════════════════
# CACHED RESOURCES
# ═══════════════════════════════════════════════
@st.cache_resource
def get_embedding_model():
    """Load the sentence-transformer once and cache it."""
    return load_embedding_model()


# ═══════════════════════════════════════════════
# HELPER: INGEST PDF
# ═══════════════════════════════════════════════
def ingest_pdf(file_name: str, file_bytes: bytes, store: DocumentStore) -> int:
    """Extract → chunk → store. Returns the number of new chunks added."""
    pages = extract_pages(file_bytes)
    if not pages:
        st.warning(f"⚠️ No text could be extracted from **{file_name}**")
        return 0

    chunks, meta = chunk_document(pages, file_name)
    if not chunks:
        st.warning(f"⚠️ No chunks generated from **{file_name}**")
        return 0

    store.add_document(chunks, meta)
    return meta.num_chunks


# ═══════════════════════════════════════════════
# AURORA HEADER
# ═══════════════════════════════════════════════
st.markdown("""
<div class="aurora-header">
    <h1>🧠 DocuMind AI</h1>
    <p>Intelligent Document Question Answering — Upload PDFs, ask questions, get cited answers</p>
</div>
""", unsafe_allow_html=True)

store: DocumentStore = st.session_state.store

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🤖 AI Provider")
    ai_provider = st.selectbox(
        "Choose LLM backend",
        ALL_PROVIDERS,
        help="Select how answers are generated. Ollama and Extractive work fully offline.",
        label_visibility="collapsed",
    )

    api_key_input = ""
    selected_model_name = ""
    if ai_provider in PROVIDER_ENV_VARS:
        env_var = PROVIDER_ENV_VARS[ai_provider]
        env_loaded = bool(os.environ.get(env_var))
        api_key_input = st.text_input(
            f"API Key ({ai_provider.split('(')[0].strip()})",
            type="password",
            help=(
                f"Leave empty to use {env_var} from environment"
                if env_loaded
                else "Enter your API key to enable this provider"
            ),
        )
        if ai_provider == PROVIDER_GROQ:
            selected_model_name = st.selectbox(
                "Groq Model",
                [
                    "llama-3.3-70b-versatile",
                    "openai/gpt-oss-20b",
                    "qwen/qwen3-32b",
                ],
                index=0,
                help="Select Groq model",
            )
        if env_loaded and not api_key_input:
            st.success(f"🔐 Using `{env_var}` from environment")
        elif ai_provider in PROVIDER_SIGNUP_URLS:
            st.info(f"🔗 Free key: {PROVIDER_SIGNUP_URLS[ai_provider]}")
    elif ai_provider == PROVIDER_OLLAMA:
        ollama_models = get_available_ollama_models()
        if ollama_models:
            st.success(f"🟢 Ollama active ({len(ollama_models)} models discovered)")
            selected_model_name = st.selectbox(
                "Ollama Model",
                ollama_models,
                index=0,
                help="Select local model for offline inference",
            )
        else:
            st.info("🦙 Ollama active. Local model is initializing...")
            selected_model_name = st.text_input(
                "Ollama Model",
                value="llama3.2:1b",
                help="Model name to run in local Ollama",
            )

    st.markdown("---")
    st.markdown("### 🎓 Academic Multi-Agent Settings")
    multi_agent_inspection = st.checkbox(
        "Multi-Agent Deep Inspection",
        value=True,
        help="Runs Intent Agent, Query Expansion Agent, Passage Inspector, and KaTeX Math Guard.",
    )
    st.markdown("---")
    st.markdown("### 🎯 Retrieval Settings")
    use_hybrid = st.checkbox(
        "Hybrid search (BM25 + Dense)",
        value=True,
        help="Combine keyword and semantic search via Reciprocal Rank Fusion.",
    )
    similarity_threshold = st.slider("Minimum similarity", 0.0, 1.0, 0.15, 0.05)
    num_results = st.slider("Max results", 1, 20, 8)

    doc_filter: List[str] = []
    if store.document_names:
        doc_filter = st.multiselect(
            "Filter by document",
            options=store.document_names,
            default=store.document_names,
            help="Restrict search to selected documents.",
        )

    st.markdown("---")
    st.markdown("### 💾 Saved Indexes")
    saved = list_saved_stores()
    selected_saved = st.selectbox("Load an index", ["—"] + saved) if saved else "—"

    col_load, col_del = st.columns(2)
    with col_load:
        if st.button("📂 Load", disabled=(selected_saved == "—"), use_container_width=True):
            st.session_state.store = load_store(selected_saved)
            st.success(f"✅ Loaded '{selected_saved}'")
            st.rerun()
    with col_del:
        if st.button("🗑️ Delete", disabled=(selected_saved == "—"), use_container_width=True):
            delete_store(selected_saved)
            st.success(f"Deleted '{selected_saved}'")
            st.rerun()

    save_name = st.text_input("Save current index as", placeholder="my-research")
    if st.button("💾 Save Index", disabled=(not save_name or store.faiss_index is None), use_container_width=True):
        try:
            save_store(store, save_name.strip())
            st.success(f"Saved to `data/{save_name}`")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    st.markdown("---")
    st.markdown("### 📊 Pipeline Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Chunks", store.total_chunks)
    with c2:
        st.metric("Documents", len(store.document_names))
    st.metric("Queries", st.session_state.query_count)

    if store.doc_metadata:
        with st.expander("📄 Document Details", expanded=True):
            for name, meta in list(store.doc_metadata.items()):
                col_name, col_del = st.columns([4, 1])
                with col_name:
                    st.markdown(
                        f"**{name}**  \n"
                        f"{meta['num_chunks']} chunks · {meta['total_pages']} pages"
                    )
                with col_del:
                    if st.button("❌", key=f"del_doc_{name}", help=f"Remove '{name}' from index"):
                        store.remove_document(name)
                        if store.chunks:
                            build_indexes(store, get_embedding_model())
                        st.rerun()

            if st.button("🗑️ Clear All Documents", key="btn_clear_all_side", use_container_width=True):
                store.clear()
                st.session_state.chat_history.clear()
                st.rerun()

# ═══════════════════════════════════════════════
# STAT CARDS
# ═══════════════════════════════════════════════
stat_data = [
    ("📑", f"{store.total_chunks:,}", "Text Chunks"),
    ("📚", str(len(store.document_names)), "Documents"),
    ("💬", str(len(st.session_state.chat_history)), "Messages"),
    ("🔍", str(st.session_state.query_count), "Searches"),
]

cols = st.columns(4)
for col, (icon, value, label) in zip(cols, stat_data):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PDF UPLOAD
# ═══════════════════════════════════════════════
col_u_title, col_u_clear = st.columns([3, 1])
with col_u_title:
    st.markdown('<div class="section-title">📤 Upload Documents</div>', unsafe_allow_html=True)
with col_u_clear:
    if store.chunks:
        if st.button("🗑️ Clear All Docs", key="btn_clear_all_main", help="Unload all loaded documents and chat history", use_container_width=True):
            store.clear()
            st.session_state.chat_history.clear()
            st.rerun()

uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    added_any = False
    progress_bar = st.progress(0, text="Processing documents...")

    for file_idx, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(
            (file_idx) / len(uploaded_files),
            text=f"Processing {uploaded_file.name}...",
        )
        if store.has_document(uploaded_file.name):
            st.info(f"⏭️ Already loaded: **{uploaded_file.name}**")
            continue
        n = ingest_pdf(uploaded_file.name, uploaded_file.getvalue(), store)
        if n:
            st.markdown(
                f'<div class="success-glass">✅ Extracted <strong>{n} chunks</strong> from <strong>{uploaded_file.name}</strong></div>',
                unsafe_allow_html=True,
            )
            added_any = True

    progress_bar.progress(1.0, text="Processing complete!")

    if added_any:
        with st.spinner("🔧 Building search indexes (FAISS + BM25)..."):
            build_indexes(store, get_embedding_model())
        st.markdown(
            f'<div class="success-glass">🎉 <strong>{store.total_chunks} chunks</strong> indexed across '
            f'<strong>{len(store.document_names)} document(s)</strong>. Ready to chat!</div>',
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════
# CHAT INTERFACE
# ═══════════════════════════════════════════════
if store.faiss_index is not None and store.chunks:
    st.markdown('<div class="section-title">💬 Ask Your Documents</div>', unsafe_allow_html=True)

    # Render existing chat history
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # New user query
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.query_count += 1
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # ── Agent 1: Intent Analysis ─────
            intent, intent_reasoning, direct_reply = IntentAgent.analyze_intent(
                prompt, has_documents=bool(store.chunks)
            )

            if intent == QueryIntent.GREETING_OR_CHITCHAT and direct_reply:
                st.markdown(
                    '<span class="quality-badge quality-green">🧠 Intent: Conversational Interaction</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(direct_reply)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": direct_reply}
                )
            else:
                # ── Agent 2: Query Decomposition & Academic Keyword Expansion ─────
                search_query, expanded_terms = QueryExpansionAgent.expand_query(prompt)

                # ── Agent 3: Hybrid Retrieval (FAISS + BM25) ─────
                with st.spinner("🔬 Multi-Agent Research: Ingesting & searching academic passages..."):
                    try:
                        raw_results = search(
                            store,
                            get_embedding_model(),
                            search_query,
                            k=num_results,
                            hybrid=use_hybrid,
                            doc_filter=doc_filter or None,
                            min_score=similarity_threshold,
                        )
                    except Exception as exc:
                        st.error(f"❌ Retrieval error: {exc}")
                        raw_results = []

                # ── Agent 4: Passage Inspector & Fact-Verification ─────
                verified_results, trace = PassageInspectorAgent.inspect_and_verify(raw_results, prompt)

                if not verified_results:
                    no_result_msg = (
                        f"### ⚠️ Academic Grounding Notice\n\n"
                        f"No verified document passages exceeded the minimum similarity threshold (**{similarity_threshold:.0%}**). "
                        "Try lowering the similarity threshold in the sidebar or rephrasing your research inquiry."
                    )
                    st.warning(no_result_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": no_result_msg}
                    )
                else:
                    # ── Render Agent Inspection Trace Card ─────
                    pages_str = ", ".join(f"Page {p}" for p in trace.verified_pages) if trace.verified_pages else "Full Context"
                    expansion_str = f" · Expanded: {len(expanded_terms)} concepts" if expanded_terms else ""

                    st.markdown(f"""
                    <div class="agent-trace-box">
                        <div class="agent-trace-title">
                            🎓 Multi-Agent Academic Inspection Trace
                            <span class="agent-stat-pill">Grounding Confidence: {trace.confidence_pct}</span>
                            <span class="agent-stat-pill">Verified: {len(verified_results)} chunks</span>
                            <span class="agent-stat-pill">{pages_str}</span>
                        </div>
                        <div style="margin-top: 0.35rem;">
                            <span class="agent-step-pill">🧠 Intent: {trace.intent_label}</span>
                            <span class="agent-step-pill">🔍 Query Expansion: {len(expanded_terms)} terms</span>
                            <span class="agent-step-pill">🛡️ Fact Verification: Grounded</span>
                            <span class="agent-step-pill">📐 KaTeX LaTeX Math Guard: Active</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Agent 5: Professor-Grade Synthesis & Formatting ─────
                    context = build_context_string(verified_results)
                    api_key = resolve_api_key(ai_provider, api_key_input)
                    if ai_provider in PROVIDER_ENV_VARS and not api_key:
                        st.warning("⚠️ No API key — running Professor-Grade Extractive Synthesizer offline")

                    try:
                        answer = st.write_stream(
                            stream_answer(
                                ai_provider,
                                prompt,
                                context,
                                api_key,
                                model_name=selected_model_name,
                            )
                        )
                    except Exception as exc:
                        st.error(f"❌ LLM error: {exc} — synthesizing offline academic report.")
                        answer = AcademicSynthesisAgent.synthesize_offline_academic_report(
                            prompt, verified_results, trace
                        )
                        st.write(answer)

                    # Clean LaTeX math before storing
                    answer = clean_markdown_math(answer)

                    # ── Verified Source Citations ──────────
                    with st.expander(f"📌 Verified Citations & Passage Evidence ({len(verified_results)} passages)"):
                        for i, r in enumerate(verified_results, 1):
                            page_info = f" · {r.page_label}" if r.page_label else ""
                            score_pct = r.score * 100
                            st.markdown(f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span class="citation-badge">Source {i}</span>
                                    <span class="citation-page">{r.doc_name}{page_info}</span>
                                    <span class="citation-score">Relevance: {r.confidence_pct}</span>
                                </div>
                                <div class="citation-text">{clean_markdown_math(r.text[:350])}{'...' if len(r.text) > 350 else ''}</div>
                                <div class="confidence-bar-container">
                                    <div class="confidence-bar-fill" style="width: {score_pct:.0f}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

    # ── Chat Actions ─────────────────────────
    st.markdown("---")
    action_cols = st.columns([1, 1, 1, 3])
    with action_cols[0]:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    with action_cols[1]:
        st.download_button(
            "⬇️ Markdown",
            data=chat_to_markdown(st.session_state.chat_history),
            file_name=f"documind-chat-{datetime.now():%Y%m%d-%H%M%S}.md",
            mime="text/markdown",
            disabled=not st.session_state.chat_history,
        )
    with action_cols[2]:
        pdf_bytes = (
            chat_to_pdf_bytes(st.session_state.chat_history)
            if st.session_state.chat_history
            else None
        )
        st.download_button(
            "⬇️ PDF",
            data=pdf_bytes if pdf_bytes is not None else b"",
            file_name=f"documind-chat-{datetime.now():%Y%m%d-%H%M%S}.pdf",
            mime="application/pdf",
            disabled=(pdf_bytes is None),
            help=(
                "Install fpdf2 to enable PDF export"
                if pdf_bytes is None
                else "Download chat as PDF"
            ),
        )

else:
    # ── Empty state ──────────────────────────
    st.markdown("""
    <div class="info-glass" style="text-align: center; padding: 2.5rem;">
        <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</p>
        <p style="font-size: 1.15rem; font-weight: 600; color: #c7d2fe;">
            Upload PDF documents or load a saved index to get started
        </p>
        <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 0.5rem;">
            DocuMind AI will extract text, build semantic + keyword indexes,<br>
            and let you ask questions with page-level citations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer-glass">
    <p class="footer-brand">🧠 DocuMind AI</p>
    <p class="footer-sub">Intelligent Document Question Answering · RAG Pipeline · Offline Capable</p>
</div>
""", unsafe_allow_html=True)
