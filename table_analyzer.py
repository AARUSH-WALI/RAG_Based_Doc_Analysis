"""
DocuMind AI — Table Analysis Engine
=====================================
Programmatic analysis of Markdown tables extracted from PDFs.
Handles numeric comparison, extremum detection, aggregation,
and structured answer generation — offloading arithmetic
from small LLMs that cannot reliably compare dozens of numbers.

This module ensures that questions like "which is the most expensive"
get an exact, correct answer by scanning every row computationally,
rather than hoping a 1-billion-parameter model can do arithmetic.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Markdown Table Parser
# ──────────────────────────────────────────────
def parse_markdown_tables(text: str) -> List[List[Dict[str, str]]]:
    """
    Parse all Markdown tables from text into structured data.

    Returns a list of tables, where each table is a list of row dicts
    mapping column-header → cell-value.
    """
    tables: list[list[dict[str, str]]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Detect a table header row (must contain at least 2 pipes)
        if line.count("|") >= 2 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # Check for separator row (---|---|...)
            if re.match(r"^[\|\s\-:]+$", next_line) and "-" in next_line:
                # Parse header cells
                headers = [h.strip() for h in line.split("|")]
                headers = [h for h in headers if h]  # remove empty from leading/trailing |
                if not headers:
                    i += 1
                    continue

                # Parse data rows
                rows: list[dict[str, str]] = []
                j = i + 2
                while j < len(lines) and "|" in lines[j]:
                    row_line = lines[j].strip()
                    # Skip additional separator rows
                    if re.match(r"^[\|\s\-:]+$", row_line):
                        j += 1
                        continue
                    # Skip empty lines
                    if not row_line:
                        j += 1
                        continue
                    cells = [c.strip() for c in row_line.split("|")]
                    cells = [c for c in cells if c or cells.index(c) > 0]  # handle leading empty
                    # Remove empty strings from leading/trailing pipes
                    while cells and cells[0] == "":
                        cells.pop(0)
                    while cells and cells[-1] == "":
                        cells.pop()

                    if cells:
                        row: dict[str, str] = {}
                        for k in range(min(len(headers), len(cells))):
                            row[headers[k]] = cells[k]
                        if row:
                            rows.append(row)
                    j += 1

                if rows:
                    tables.append(rows)
                i = j
                continue
        i += 1
    return tables


# ──────────────────────────────────────────────
# Numeric Extraction
# ──────────────────────────────────────────────
def _extract_number(value: str) -> Optional[float]:
    """
    Extract a numeric value from a cell string.
    Handles currency symbols (₹$€£¥), commas, whitespace, and trailing text.
    """
    if not value:
        return None
    # Remove currency symbols, commas, whitespace
    cleaned = re.sub(r"[₹$€£¥,\s]", "", value.strip())
    # Remove trailing non-numeric text (e.g. "78.10 per piece")
    cleaned = re.sub(r"[a-zA-Z%]+.*$", "", cleaned).strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────
# Column Classification
# ──────────────────────────────────────────────
def _detect_numeric_columns(table: List[Dict[str, str]]) -> List[str]:
    """Identify columns where ≥ 50 % of cells are parseable as numbers."""
    if not table:
        return []
    numeric_cols = []
    for col in table[0].keys():
        numeric_count = sum(
            1 for row in table if _extract_number(row.get(col, "")) is not None
        )
        if numeric_count >= len(table) * 0.5:
            numeric_cols.append(col)
    return numeric_cols


def _detect_name_column(table: List[Dict[str, str]]) -> Optional[str]:
    """
    Detect the label/name column (predominantly non-numeric).
    Falls back to the first column if all are numeric.
    """
    if not table:
        return None
    for col in table[0].keys():
        numeric_count = sum(
            1 for row in table if _extract_number(row.get(col, "")) is not None
        )
        if numeric_count < len(table) * 0.3:
            return col
    # Fallback: first column
    return list(table[0].keys())[0]


# ──────────────────────────────────────────────
# Query Intent Detection (Extremum / Aggregation)
# ──────────────────────────────────────────────
_MAX_KEYWORDS = [
    "most expensive", "highest price", "maximum price", "costliest",
    "highest total", "highest cost", "highest", "maximum", "most costly",
    "expensive of all", "expensive", "largest", "biggest", "greatest",
    "top", "most", "max",
]
_MIN_KEYWORDS = [
    "cheapest", "least expensive", "lowest price", "minimum price",
    "lowest cost", "lowest", "minimum", "least costly", "smallest",
    "least", "fewest", "min",
]
_SUM_KEYWORDS = [
    "total", "sum", "aggregate", "grand total", "overall", "combined",
    "how much in total",
]
_COUNT_KEYWORDS = [
    "how many", "count", "number of", "total items", "total products",
]


def _classify_query(query: str) -> Tuple[str, str, Optional[str]]:
    """
    Classify a query into (query_type, direction, metric_hint).

    query_type: 'extremum' | 'aggregation' | 'count' | 'general'
    direction:  'max' | 'min' | 'sum' | 'count' | ''
    metric_hint: 'unit_price' | 'total_price' | 'quantity' | None
    """
    q = query.lower()

    # Extremum detection
    direction = ""
    for kw in _MAX_KEYWORDS:
        if kw in q:
            direction = "max"
            break
    if not direction:
        for kw in _MIN_KEYWORDS:
            if kw in q:
                direction = "min"
                break

    if direction:
        # Metric hint detection
        metric_hint = None
        if any(t in q for t in ["unit price", "price per piece", "per piece", "per unit", "rate per"]):
            metric_hint = "unit_price"
        elif any(t in q for t in ["total price", "total cost", "total", "batch cost"]):
            metric_hint = "total_price"
        elif any(t in q for t in ["quantity", "count", "units", "pieces", "stock"]):
            metric_hint = "quantity"
        return "extremum", direction, metric_hint

    # Aggregation
    for kw in _SUM_KEYWORDS:
        if kw in q:
            return "aggregation", "sum", None

    # Count
    for kw in _COUNT_KEYWORDS:
        if kw in q:
            return "count", "count", None

    return "general", "", None


def _best_column_for_hint(
    numeric_cols: List[str], hint: Optional[str]
) -> Optional[str]:
    """Map a metric_hint to the best matching numeric column name."""
    if not numeric_cols:
        return None
    if hint is None:
        return None

    for col in numeric_cols:
        cl = col.lower()
        if hint == "unit_price" and any(
            t in cl for t in ["price per", "unit price", "rate", "per piece"]
        ):
            return col
        if hint == "total_price" and "total" in cl:
            return col
        if hint == "quantity" and any(
            t in cl for t in ["quantity", "qty", "count", "units", "pieces"]
        ):
            return col
    return None


# ──────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────
def analyze_tables_for_query(context: str, query: str) -> Optional[str]:
    """
    Given RAG context (which may contain Markdown tables) and a user query,
    perform **programmatic** numeric analysis and return a pre-computed
    answer string that can be injected into the LLM prompt.

    Returns ``None`` when no table-based analysis is applicable.
    """
    tables = parse_markdown_tables(context)
    if not tables:
        return None

    query_type, direction, metric_hint = _classify_query(query)

    if query_type == "extremum":
        return _analyze_extremum(tables, direction, metric_hint, query)
    if query_type == "aggregation":
        return _analyze_aggregation(tables)
    if query_type == "count":
        return _analyze_count(tables)

    # General: provide a concise structural summary so the LLM has orientation
    return _generate_table_summary(tables)


# ──────────────────────────────────────────────
# Extremum Analysis
# ──────────────────────────────────────────────
def _analyze_extremum(
    tables: List[List[Dict[str, str]]],
    direction: str,
    metric_hint: Optional[str],
    query: str,
) -> str:
    sections: list[str] = []

    for t_idx, table in enumerate(tables):
        name_col = _detect_name_column(table)
        numeric_cols = _detect_numeric_columns(table)
        if not numeric_cols:
            continue

        # Determine which columns to rank
        target_cols: list[str] = []
        if metric_hint:
            best = _best_column_for_hint(numeric_cols, metric_hint)
            if best:
                target_cols = [best]
        if not target_cols:
            target_cols = numeric_cols  # analyse every numeric column

        for col in target_cols:
            ranked: list[tuple[str, float, dict[str, str]]] = []
            for row in table:
                val = _extract_number(row.get(col, ""))
                name = row.get(name_col, "Unknown") if name_col else "Unknown"
                if val is not None:
                    ranked.append((name, val, row))

            if not ranked:
                continue

            ranked.sort(key=lambda x: x[1], reverse=(direction == "max"))
            top5 = ranked[:5]
            dir_label = "Highest → Lowest" if direction == "max" else "Lowest → Highest"

            lines = [
                f"\n**Column: {col}** ({dir_label}):\n",
                f"| Rank | Item | {col} | Other Details |",
                f"|------|------|-------|---------------|",
            ]
            for rank, (name, val, row) in enumerate(top5, 1):
                extras = []
                for oc in numeric_cols:
                    if oc != col:
                        ov = row.get(oc, "")
                        extras.append(f"{oc}: {ov}")
                extra_str = ", ".join(extras) if extras else "—"
                lines.append(f"| {rank} | {name} | {val:,.2f} | {extra_str} |")

            # Highlight the winner
            winner_name, winner_val, _ = top5[0]
            lines.append(
                f"\n**Winner**: **{winner_name}** with **{col} = {winner_val:,.2f}**"
            )
            sections.append("\n".join(lines))

    if not sections:
        return _generate_table_summary(tables)

    total_rows = sum(len(t) for t in tables)
    dir_word = "most expensive / highest" if direction == "max" else "cheapest / lowest"

    return (
        f"### 📊 PRE-COMPUTED TABLE ANALYSIS (Programmatic — Zero Hallucination)\n\n"
        f"**Query Type**: {dir_word.title()} value detection\n"
        f"**Tables Analyzed**: {len(tables)} table(s), **{total_rows} rows** scanned exhaustively\n"
        + "\n".join(sections)
        + "\n\n> ⚡ These rankings were computed by scanning **every single row** "
        "programmatically. Numbers are **exact** — no LLM arithmetic involved."
    )


# ──────────────────────────────────────────────
# Aggregation Analysis
# ──────────────────────────────────────────────
def _analyze_aggregation(tables: List[List[Dict[str, str]]]) -> str:
    sections: list[str] = []
    for table in tables:
        numeric_cols = _detect_numeric_columns(table)
        for col in numeric_cols:
            values = [
                _extract_number(row.get(col, ""))
                for row in table
                if _extract_number(row.get(col, "")) is not None
            ]
            if values:
                sections.append(
                    f"- **{col}**: Sum = {sum(values):,.2f}, "
                    f"Average = {sum(values)/len(values):,.2f}, "
                    f"Min = {min(values):,.2f}, Max = {max(values):,.2f}, "
                    f"Count = {len(values)}"
                )
    if not sections:
        return _generate_table_summary(tables)
    return (
        "### 📊 Aggregation Summary (Computed Programmatically)\n\n"
        + "\n".join(sections)
    )


# ──────────────────────────────────────────────
# Count Analysis
# ──────────────────────────────────────────────
def _analyze_count(tables: List[List[Dict[str, str]]]) -> str:
    total_rows = sum(len(t) for t in tables)
    return (
        f"### 📊 Count Analysis\n\n"
        f"- **Total tables found**: {len(tables)}\n"
        f"- **Total data rows**: {total_rows}\n"
    )


# ──────────────────────────────────────────────
# Table Summary (fallback for non-analytic queries)
# ──────────────────────────────────────────────
def _generate_table_summary(tables: List[List[Dict[str, str]]]) -> Optional[str]:
    if not tables:
        return None
    lines = ["### 📊 Detected Table Structure\n"]
    for i, table in enumerate(tables, 1):
        if not table:
            continue
        cols = list(table[0].keys())
        numeric_cols = _detect_numeric_columns(table)
        lines.append(f"**Table {i}**: {len(table)} rows × {len(cols)} columns")
        lines.append(f"- Columns: {', '.join(cols)}")
        if numeric_cols:
            lines.append(f"- Numeric columns: {', '.join(numeric_cols)}")
    return "\n".join(lines)
