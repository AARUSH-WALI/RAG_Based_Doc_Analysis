"""
DocuMind AI — Utility Functions
================================
Chat export (Markdown / PDF), session helpers, and answer-quality scoring.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from retrieval_engine import SearchResult


# ──────────────────────────────────────────────
# Chat Export — Markdown
# ──────────────────────────────────────────────
def chat_to_markdown(history: List[dict]) -> str:
    """Convert the chat history list into a downloadable Markdown string."""
    lines = [
        f"# DocuMind AI — Chat Transcript\n",
        f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "---\n",
    ]
    for msg in history:
        role = msg["role"].capitalize()
        lines.append(f"## {role}\n\n{msg['content']}\n")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Chat Export — PDF (optional, requires fpdf2)
# ──────────────────────────────────────────────
def chat_to_pdf_bytes(history: List[dict]) -> bytes | None:
    """
    Generate a PDF of the chat transcript.

    Returns None if ``fpdf2`` is not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "DocuMind AI - Chat Transcript", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0, 8,
        f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(6)

    for msg in history:
        # Role heading
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, msg["role"].capitalize(), new_x="LMARGIN", new_y="NEXT")
        # Body text (gracefully degrade non-latin-1 characters)
        pdf.set_font("Helvetica", "", 10)
        safe_text = msg["content"].encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_text)
        pdf.ln(3)

    return bytes(pdf.output(dest="S"))


def clean_markdown_math(text: str) -> str:
    """
    Clean raw math notations into standard KaTeX markdown syntax:
    - Converts [ \\boxed{...} ] or [ ... ] math blocks into $$ ... $$
    - Converts unicode fractions (½, ¼, ⅓, ⅙) and superscripts (r², R², d²) into LaTeX
    - Converts Greek symbols (π, θ) and operators (×, ·, −) into standard math syntax
    - Fixes leftover unicode/mojibake artifacts
    """
    import re
    # Convert [ \boxed{...} ] or [ ... ] with LaTeX formulas into $$ ... $$
    text = re.sub(r'\[\s*\\boxed\{([^}]+)\}\s*\]', r'$$\1$$', text)
    text = re.sub(r'\[\s*(\\frac|\\pi|\w+\s*=)[^\]]*\]', lambda m: f"$${m.group(0)[1:-1].strip()}$$", text)
    
    # Common math symbols to LaTeX KaTeX equivalents
    math_symbols = {
        "π": r"\pi",
        "θ": r"\theta",
        "r²": r"r^2",
        "R²": r"R^2",
        "d²": r"d^2",
        "k²": r"k^2",
        "½": r"\frac{1}{2}",
        "¼": r"\frac{1}{4}",
        "⅓": r"\frac{1}{3}",
        "⅙": r"\frac{1}{6}",
        "√3": r"\sqrt{3}",
        "Δ OAB": r"\Delta OAB",
        "Δ": r"\Delta",
        "−": "-",
        "·": r"\cdot ",
    }
    for sym, latex in math_symbols.items():
        if sym in text:
            text = text.replace(sym, latex)

    # Fix mojibake
    mojibake = {
        "â€”": "—", "â€“": "–", "â€™": "'", "â€˜": "'",
        "â€œ": '"', "â€ ": '"', "â€¢": "•", "â†’": "→",
        "â‡’": "⇒", "â‰ˆ": "≈", "â‰ ": "≠", "âˆ˜": "°",
        "Â": "", "â": "",
    }
    for bad, good in mojibake.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


# ──────────────────────────────────────────────
# Context Builder for LLM Prompt
# ──────────────────────────────────────────────
def build_context_string(results: List[SearchResult]) -> str:
    """
    Combine search results into a single context block for the LLM,
    including page references so the model can cite them.
    """
    parts: List[str] = []
    for i, r in enumerate(results, 1):
        header = f"[Source {i} — {r.doc_name}"
        if r.page_label:
            header += f", {r.page_label}"
        header += f" — Relevance: {r.confidence_pct}]"
        cleaned_text = clean_markdown_math(r.text)
        parts.append(f"{header}\n{cleaned_text}")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# Answer Quality Indicator
# ──────────────────────────────────────────────
def answer_quality_label(results: List[SearchResult]) -> tuple[str, str]:
    """
    Return a (label, colour) pair summarising how well the retrieved
    context supports an answer.

    Uses the top-result score and the number of results as signals.
    """
    if not results:
        return ("⚠️ No context found", "red")

    top_score = results[0].score
    count = len(results)

    if top_score >= 0.70 and count >= 3:
        return ("🟢 High confidence", "green")
    if top_score >= 0.45 and count >= 2:
        return ("🟡 Moderate confidence", "orange")
    return ("🔴 Low confidence", "red")
