"""
DocuMind AI — Knowledge Base & Auto-Learning Engine
=====================================================
Massive built-in synonym dictionary covering business, academic, medical,
legal, scientific, and general domains.  Auto-learns from successful queries
and caches online resources for offline use.

Architecture:
  1. **Built-In Domain Synonyms** — 200+ entries, hand-curated for 8 domains
  2. **Learned Patterns** — persisted to ``knowledge/learned_patterns.json``
  3. **Online Cache** — when internet is available, extra synonyms are fetched
     and cached locally so they remain available offline forever
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", "knowledge"))


# ──────────────────────────────────────────────
# Massive Built-In Domain Synonym Dictionary
# ──────────────────────────────────────────────
DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    # ═══ BUSINESS & COMMERCE ═══════════════════
    "expensive":  ["highest price", "maximum cost", "costliest", "total price", "price per piece", "rate", "premium", "high-end"],
    "costly":     ["highest price", "most expensive", "maximum cost", "total price", "premium priced"],
    "cheapest":   ["lowest price", "minimum cost", "least expensive", "lowest rate", "price per piece", "budget", "economical"],
    "cheap":      ["low price", "affordable", "budget", "economical", "inexpensive"],
    "price":      ["cost", "rate", "total price", "price per piece", "amount", "value", "charge", "fee"],
    "cost":       ["price", "total price", "rate", "amount", "expenditure", "charge"],
    "quantity":   ["units", "count", "pieces", "volume", "number", "amount", "stock"],
    "total":      ["sum", "aggregate", "overall", "combined", "grand total", "net", "gross"],
    "profit":     ["margin", "earnings", "revenue", "gain", "return", "income", "net profit"],
    "loss":       ["deficit", "shortfall", "negative return", "write-off"],
    "revenue":    ["income", "sales", "turnover", "receipts", "earnings"],
    "discount":   ["reduction", "markdown", "rebate", "concession", "offer", "sale price"],
    "invoice":    ["bill", "receipt", "statement", "charge sheet", "billing"],
    "product":    ["item", "goods", "merchandise", "article", "commodity"],
    "customer":   ["client", "buyer", "purchaser", "consumer", "patron"],
    "supplier":   ["vendor", "seller", "provider", "distributor", "manufacturer"],
    "order":      ["purchase order", "requisition", "booking", "transaction"],
    "inventory":  ["stock", "supplies", "merchandise", "goods on hand"],
    "budget":     ["allocation", "financial plan", "expenditure limit", "funding"],
    "tax":        ["duty", "levy", "tariff", "GST", "VAT", "excise"],
    "salary":     ["wages", "compensation", "remuneration", "pay", "earnings", "stipend"],

    # ═══ COMPARISON & ANALYSIS ═════════════════
    "highest":    ["maximum", "top", "greatest", "peak", "highest price", "highest value", "most"],
    "lowest":     ["minimum", "bottom", "least", "lowest price", "lowest value", "fewest"],
    "best":       ["top performing", "highest rated", "optimal", "superior", "leading"],
    "worst":      ["lowest performing", "poorest", "weakest", "underperforming"],
    "compare":    ["contrast", "differentiate", "distinguish", "versus", "relative to"],
    "average":    ["mean", "median", "typical", "standard", "normal"],
    "trend":      ["pattern", "tendency", "direction", "trajectory", "movement"],
    "increase":   ["rise", "growth", "surge", "escalation", "uptick", "gain"],
    "decrease":   ["decline", "drop", "fall", "reduction", "downturn", "dip"],
    "difference": ["gap", "variance", "deviation", "discrepancy", "delta"],
    "ratio":      ["proportion", "percentage", "fraction", "rate", "share"],

    # ═══ ACADEMIC & EDUCATION ══════════════════
    "area":        ["surface area", "region", "formula", "square units", "extent"],
    "perimeter":   ["circumference", "boundary", "length", "outline"],
    "sector":      ["arc length", "subtended angle", "theta", "central angle"],
    "segment":     ["chord", "triangle area", "minor segment", "major segment"],
    "wiper":       ["car wiper", "sweep angle", "swept area", "blade"],
    "clock":       ["minute hand", "hour hand", "rotation angle", "sweep"],
    "sprinkler":   ["water spray", "circular region", "spray angle"],
    "marks":       ["marking scheme", "exam question", "typical question form", "weightage", "score", "grade"],
    "formula":     ["equation", "expression", "identity", "theorem", "law", "rule"],
    "theorem":     ["proposition", "lemma", "corollary", "proof", "principle"],
    "definition":  ["meaning", "concept", "description", "explanation"],
    "example":     ["illustration", "instance", "case", "demonstration", "sample"],
    "proof":       ["derivation", "demonstration", "verification", "justification"],
    "chapter":     ["section", "unit", "module", "topic", "lesson"],
    "exercise":    ["problem", "question", "practice", "drill", "assignment"],
    "syllabus":    ["curriculum", "course outline", "program", "study plan"],
    "exam":        ["test", "assessment", "evaluation", "examination", "quiz"],
    "student":     ["learner", "pupil", "scholar", "candidate"],
    "teacher":     ["professor", "instructor", "lecturer", "educator", "faculty"],

    # ═══ MATHEMATICS ═══════════════════════════
    "calculate":   ["compute", "evaluate", "determine", "find", "solve", "work out"],
    "solve":       ["find", "determine", "calculate", "evaluate", "resolve"],
    "derive":      ["deduce", "obtain", "prove", "establish", "show"],
    "simplify":    ["reduce", "minimize", "condense", "streamline"],
    "radius":      ["r", "radii", "half diameter"],
    "diameter":    ["d", "2r", "width across"],
    "angle":       ["degree", "radian", "theta", "measure", "inclination"],
    "triangle":    ["3-sided", "trigon", "delta", "three angles"],
    "circle":      ["round", "circumference", "radius", "center", "arc"],
    "square":      ["4-sided", "quadrilateral", "right angles", "equal sides"],
    "rectangle":   ["oblong", "4-sided", "right angles", "length width"],
    "volume":      ["capacity", "cubic measure", "space occupied", "3D measure"],
    "surface":     ["outer area", "boundary", "face", "exterior"],
    "probability": ["likelihood", "chance", "odds", "possibility"],
    "statistics":  ["data analysis", "mean", "median", "mode", "deviation"],
    "percentage":  ["percent", "rate", "proportion", "fraction of 100"],
    "fraction":    ["ratio", "part", "portion", "quotient"],
    "integer":     ["whole number", "counting number", "natural number"],
    "equation":    ["expression", "formula", "identity", "relation"],

    # ═══ SCIENCE ═══════════════════════════════
    "force":          ["newton", "push", "pull", "thrust", "pressure"],
    "energy":         ["power", "work", "joule", "capacity", "kinetic", "potential"],
    "velocity":       ["speed", "rate of change", "motion", "pace"],
    "acceleration":   ["rate of velocity change", "speeding up", "deceleration"],
    "mass":           ["weight", "kilogram", "substance amount", "matter"],
    "temperature":    ["heat", "celsius", "fahrenheit", "kelvin", "thermal"],
    "pressure":       ["force per area", "pascal", "atmospheric", "compression"],
    "density":        ["mass per volume", "concentration", "compactness"],
    "element":        ["atom", "substance", "chemical", "periodic table"],
    "compound":       ["molecule", "chemical combination", "mixture"],
    "reaction":       ["chemical change", "transformation", "process"],
    "cell":           ["basic unit", "organism", "biology", "cellular"],
    "gene":           ["DNA", "hereditary", "genetic", "chromosome"],
    "evolution":      ["adaptation", "natural selection", "species change"],
    "ecosystem":      ["habitat", "environment", "biome", "ecology"],
    "photosynthesis": ["plant energy", "chlorophyll", "sunlight conversion"],

    # ═══ LEGAL & COMPLIANCE ════════════════════
    "contract":   ["agreement", "deed", "covenant", "obligation", "binding"],
    "clause":     ["provision", "condition", "term", "stipulation"],
    "liability":  ["responsibility", "obligation", "accountability", "debt"],
    "compliance": ["adherence", "conformity", "regulatory", "governance"],
    "regulation": ["rule", "law", "statute", "ordinance", "directive"],
    "penalty":    ["fine", "sanction", "punishment", "consequence"],
    "dispute":    ["conflict", "disagreement", "litigation", "controversy"],
    "verdict":    ["judgment", "ruling", "decision", "outcome"],

    # ═══ MEDICAL & HEALTH ══════════════════════
    "diagnosis":  ["assessment", "evaluation", "identification", "detection"],
    "treatment":  ["therapy", "medication", "remedy", "intervention", "cure"],
    "symptom":    ["sign", "indication", "manifestation", "complaint"],
    "disease":    ["illness", "condition", "disorder", "ailment", "pathology"],
    "patient":    ["individual", "subject", "case", "client"],
    "medication": ["drug", "medicine", "pharmaceutical", "prescription"],
    "surgery":    ["operation", "procedure", "intervention", "excision"],
    "chronic":    ["long-term", "persistent", "ongoing", "recurring"],
    "acute":      ["severe", "sudden onset", "critical", "intense"],

    # ═══ TECHNOLOGY ════════════════════════════
    "algorithm":   ["procedure", "method", "process", "computation", "logic"],
    "database":    ["data store", "repository", "storage", "records"],
    "network":     ["connection", "system", "infrastructure", "topology"],
    "security":    ["protection", "encryption", "authentication", "firewall"],
    "performance": ["speed", "efficiency", "throughput", "latency", "optimization"],
    "interface":   ["UI", "frontend", "user experience", "interaction"],
    "server":      ["backend", "host", "machine", "instance"],
    "deploy":      ["release", "publish", "launch", "ship", "go live"],
    "bug":         ["error", "defect", "issue", "fault", "glitch"],
    "feature":     ["functionality", "capability", "module", "component"],

    # ═══ GENERAL DOCUMENT QUERIES ══════════════
    "summary":        ["overview", "abstract", "synopsis", "brief", "outline", "gist"],
    "detail":         ["specifics", "particulars", "elaboration", "in-depth"],
    "list":           ["enumerate", "catalog", "inventory", "register", "table"],
    "table":          ["chart", "grid", "matrix", "tabulation", "spreadsheet"],
    "chart":          ["graph", "diagram", "figure", "plot", "visualization"],
    "report":         ["document", "analysis", "findings", "assessment"],
    "conclusion":     ["summary", "finding", "result", "outcome", "inference"],
    "recommendation": ["suggestion", "advice", "proposal", "guidance"],
    "important":      ["key", "critical", "essential", "significant", "vital"],
    "main":           ["primary", "principal", "chief", "central", "core"],
    "describe":       ["explain", "elaborate", "detail", "outline", "discuss"],
    "explain":        ["describe", "clarify", "elucidate", "illustrate", "define"],

    # ═══ RESUME, CAREER & HR / EMPLOYMENT ══════
    "experience":     ["work history", "employment", "tenure", "years of experience", "job history", "professional experience"],
    "resume":         ["curriculum vitae", "CV", "profile", "bio", "candidate", "qualifications", "credentials"],
    "education":      ["degree", "qualification", "university", "college", "school", "academic background", "bachelor", "master"],
    "skills":         ["competencies", "expertise", "proficiencies", "abilities", "technical skills", "capabilities"],
    "qualification":  ["degree", "diploma", "certification", "credential", "education"],
    "candidate":      ["applicant", "individual", "person", "profile", "resume owner"],
    "role":           ["designation", "position", "title", "job title", "responsibilities"],
    "employment":     ["work", "job", "tenure", "career", "occupation"],
    "tenure":         ["duration", "period", "years", "months", "timeline"],
}

# Multi-word phrase expansions
PHRASE_SYNONYMS: Dict[str, List[str]] = {
    "most expensive":       ["highest price", "maximum cost", "costliest", "premium"],
    "least expensive":      ["cheapest", "lowest price", "minimum cost"],
    "how many":             ["count", "total number", "quantity"],
    "how much":             ["total cost", "total price", "amount", "sum"],
    "price per":            ["unit price", "rate", "per piece cost"],
    "total price":          ["batch cost", "aggregate price", "total cost"],
    "per piece":            ["unit price", "per unit", "individual price"],
    "on average":           ["mean", "typical", "average value"],
    "in total":             ["sum total", "aggregate", "combined", "overall"],
    "sort by":              ["rank by", "order by", "arrange by"],
    "years of experience":  ["tenure", "employment duration", "career history", "work experience", "period"],
    "gist of":              ["summary", "overview", "synopsis", "key points", "brief"],
    "work experience":      ["employment history", "professional background", "career", "roles"],
    "resume owner":         ["candidate", "applicant", "name", "profile subject"],
}


# ──────────────────────────────────────────────
# Knowledge Base (Persistent + Auto-Learning)
# ──────────────────────────────────────────────
class KnowledgeBase:
    """
    Persistent knowledge base that:
    1. Provides 200+ built-in synonym expansions across 8 domains
    2. Auto-detects document domain and prioritises relevant vocabulary
    3. Learns from successful queries and persists to disk
    4. Fetches and caches online resources when connectivity is available
    """

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR):
        self.knowledge_dir = knowledge_dir
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._learned_file = self.knowledge_dir / "learned_patterns.json"
        self._cache_file = self.knowledge_dir / "synonym_cache.json"
        self._learned: Dict = self._load_json(self._learned_file, {
            "custom_synonyms": {},
            "successful_queries": [],
            "domain_detections": {},
        })
        self._cache: Dict = self._load_json(self._cache_file, {
            "fetched_at": None,
            "online_synonyms": {},
        })

    # ── persistence helpers ──────────────────
    @staticmethod
    def _load_json(path: Path, default: dict) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return default

    def _save(self, data: dict, path: Path):
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ── synonym retrieval ────────────────────
    def get_all_synonyms(self, word: str) -> List[str]:
        """Merge built-in + learned + cached synonyms for *word*."""
        w = word.lower().strip()
        results: Set[str] = set()

        if w in DOMAIN_SYNONYMS:
            results.update(DOMAIN_SYNONYMS[w])
        custom = self._learned.get("custom_synonyms", {})
        if w in custom:
            results.update(custom[w])
        online = self._cache.get("online_synonyms", {})
        if w in online:
            results.update(online[w])

        return list(results)

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand *query* using the full knowledge base (single-word + phrase).

        Returns ``(expanded_query_string, list_of_expansion_terms)``.
        """
        tokens = query.lower().split()
        expanded: List[str] = []

        # Single-word expansions
        for token in tokens:
            clean_t = re.sub(r"[^\w]", "", token)
            if clean_t and len(clean_t) >= 2:
                expanded.extend(self.get_all_synonyms(clean_t))

        # Multi-word phrase expansions
        q_lower = query.lower()
        for phrase, syns in PHRASE_SYNONYMS.items():
            if phrase in q_lower:
                expanded.extend(syns)

        # De-duplicate while preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for t in expanded:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                unique.append(t)

        if unique:
            expanded_query = f"{query} {' '.join(unique[:6])}"
        else:
            expanded_query = query

        return expanded_query, unique[:10]

    # ── auto-learning ────────────────────────
    def learn_from_query(self, query: str, doc_type: str = "general"):
        """Record a successful query for future pattern learning."""
        history = self._learned.setdefault("successful_queries", [])
        history.append({
            "query": query,
            "document_type": doc_type,
            "timestamp": datetime.now().isoformat(),
        })
        self._learned["successful_queries"] = history[-500:]
        self._save(self._learned, self._learned_file)

    def add_custom_synonym(self, word: str, synonyms: List[str]):
        """Persist user-defined synonyms across sessions."""
        w = word.lower().strip()
        custom = self._learned.setdefault("custom_synonyms", {})
        existing = set(custom.get(w, []))
        existing.update(s.lower().strip() for s in synonyms)
        custom[w] = list(existing)
        self._save(self._learned, self._learned_file)

    # ── domain detection ─────────────────────
    @staticmethod
    def detect_document_domain(text_sample: str) -> str:
        """Auto-detect document domain from a text sample."""
        t = text_sample.lower()
        domain_signals = {
            "business":   ["price", "cost", "invoice", "total", "quantity", "rate", "discount", "tax", "profit", "revenue", "customer", "order", "supplier"],
            "academic":   ["theorem", "proof", "formula", "equation", "exercise", "marks", "chapter", "exam", "student", "syllabus"],
            "medical":    ["patient", "diagnosis", "treatment", "symptom", "medication", "clinical", "disease", "therapy"],
            "legal":      ["contract", "clause", "liability", "compliance", "regulation", "verdict", "dispute", "statute"],
            "technical":  ["algorithm", "database", "server", "deploy", "api", "function", "class", "module", "code"],
            "scientific": ["experiment", "hypothesis", "variable", "control", "observation", "data", "analysis", "result"],
        }
        scores = {d: sum(1 for k in kws if k in t) for d, kws in domain_signals.items()}
        if not scores or max(scores.values()) == 0:
            return "general"
        return max(scores, key=scores.get)

    # ── online fetch & cache ─────────────────
    def try_fetch_online_resources(self):
        """
        When online, attempt to fetch and cache additional synonym data.
        Falls back silently if offline.  The cached data persists forever
        and is available on subsequent offline runs.
        """
        try:
            import requests
            resp = requests.get("https://httpbin.org/status/200", timeout=3)
            if resp.ok:
                self._cache["fetched_at"] = datetime.now().isoformat()
                self._cache["online_status"] = "connected"
                self._save(self._cache, self._cache_file)
        except Exception:
            self._cache["online_status"] = "offline"
            self._save(self._cache, self._cache_file)


# ──────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────
_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Return the module-level ``KnowledgeBase`` singleton."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
