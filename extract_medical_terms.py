"""
Multilingual medical term extractor with three modes:
- hybrid: regex candidates + LLM categorization
- category-guided: dictionary matching only (no API calls)
- llm-only: LLM extracts and categorizes directly
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from openpyxl import load_workbook
from dotenv import load_dotenv
from PyPDF2 import PdfReader

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None  # type: ignore


# -------------------------- Constants & defaults --------------------------- #

MAXQDA_FIELDS: Sequence[str] = (
    "Категория",
    "Комментарий",
    "Ключевое слово",
    "Целое слово",
    "Различать заглавные",
    "Начало слова",
    "Категория активирована",
    "Ключевое слово активировано",
)

MAX_CATEGORY_DEPTH = 3  # keep at most 3 levels of hierarchy
# Separator used for category paths (no surrounding spaces)
CATEGORY_SEPARATOR = "\\"

# Conservative multilingual regex: supports Russian and English words/phrases.
MULTILINGUAL_PATTERN = re.compile(
    r"[а-яА-ЯёЁa-zA-Z]{3,}(?:\s+[а-яА-ЯёЁa-zA-Z]{3,})*", re.UNICODE
)

BASE_STOPWORDS_RU: Set[str] = {
    "и",
    "или",
    "в",
    "на",
    "по",
    "к",
    "от",
    "для",
    "с",
    "о",
    "об",
    "у",
    "из",
    "над",
    "под",
    "между",
    "через",
    "без",
    "при",
    "перед",
    "за",
    "но",
    "а",
    "же",
    "ли",
    "не",
    "ни",
    "бы",
    "это",
    "то",
    "все",
    "всё",
    "каждый",
    "какой",
    "который",
    "может",
    "можно",
    "нельзя",
    "нужно",
    "надо",
    "следует",
    "также",
    "однако",
    "таким образом",
    "например",
}

BASE_STOPWORDS_EN: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "nor",
    "so",
    "yet",
    "in",
    "on",
    "at",
    "to",
    "from",
    "by",
    "with",
    "of",
    "for",
    "as",
    "about",
    "above",
    "across",
    "after",
    "before",
    "is",
    "are",
    "was",
    "were",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "this",
    "that",
    "these",
    "those",
    "which",
    "who",
    "what",
    "also",
    "however",
    "thus",
    "therefore",
    "such",
    "like",
}

# A lightweight general-purpose category set to help prompts and local matching.
DEFAULT_CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Диагностика / Diagnostics": {
        "keywords": ["diagnosis", "diagnostic", "МРТ", "MRI", "УЗИ", "CT", "рентген"],
        "comment": "Методы диагностики",
    },
    "Операции / Surgical": {
        "keywords": [
            "arthroscopy",
            "arthroscopic",
            "surgical repair",
            "операция",
            "хирургическое",
            "decompression",
            "reconstruction",
        ],
        "comment": "Хирургическое лечение",
    },
    "Консервативная терапия / Conservative care": {
        "keywords": [
            "physical therapy",
            "реабилитация",
            "rehabilitation",
            "immobilization",
            "brace",
            "NSAID",
            "анальгезия",
            "pain management",
        ],
        "comment": "Безоперационное лечение",
    },
    "Биологические методы / Biologics": {
        "keywords": [
            "PRP",
            "platelet-rich plasma",
            "stem cells",
            "growth factor",
            "биологическая терапия",
            "плазма",
        ],
        "comment": "PRP и другие биологические методы",
    },
    "Осложнения / Complications": {
        "keywords": [
            "infection",
            "retear",
            "stiffness",
            "nonunion",
            "complication",
            "осложнение",
            "повторный",
        ],
        "comment": "Послеоперационные осложнения",
    },
}


@dataclass
class TermRecord:
    category: str
    keyword: str
    comment: str = ""
    whole_word: bool = False
    match_case: bool = False
    category_enabled: bool = True
    keyword_enabled: bool = True


# ------------------------------ IO helpers -------------------------------- #


def load_env() -> None:
    """Load .env if present to populate OpenAI credentials."""
    load_dotenv()


def ensure_output_path(path: Path) -> None:
    """Create parent folders for the given output file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_txt(path: Path) -> str:
    """Read UTF-8 text content from a TXT file."""
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def read_pdf(path: Path) -> str:
    """Extract text from a PDF using PyPDF2."""
    text: List[str] = []
    with path.open("rb") as handle:
        reader = PdfReader(handle)
        for page in reader.pages:
            content = page.extract_text() or ""
            text.append(content)
    return "\n".join(text)


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text from a PDF or UTF-8 TXT file.

    Returns an empty string if parsing fails, keeping the pipeline resilient.
    """
    try:
        if file_path.suffix.lower() == ".pdf":
            text = read_pdf(file_path)
        else:
            text = read_txt(file_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"❌ Failed to read {file_path}: {exc}")
        return ""
    cleaned = clean_text(text)
    return cleaned


def load_documents(input_path: Path) -> List[Tuple[str, str]]:
    """
    Read all PDF/TXT files under input_path.

    Returns:
        List of tuples (filename, text_content)
    """
    files: List[Path] = []
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in {".pdf", ".txt"}
        )

    docs: List[Tuple[str, str]] = []
    for path in files:
        text = extract_text_from_file(path)
        if text:
            docs.append((path.name, text))
        else:
            print(f"⚠️ Skipping empty or unreadable file: {path}")
    return docs


# --------------------------- Text processing ------------------------------ #


def normalize_term(term: str) -> str:
    term = term.strip()
    term = re.sub(r"\s+", " ", term)
    return term


def normalize_phrase_for_match(text: str) -> str:
    """Lowercase + remove punctuation for matching to codebook entries."""
    cleaned = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def build_category_name(parts: List[str]) -> str:
    """Join category parts, capped to MAX_CATEGORY_DEPTH, fallback to General."""
    if not parts:
        return "General"
    capped = [p.strip() for p in parts[:MAX_CATEGORY_DEPTH] if p.strip()]
    return CATEGORY_SEPARATOR.join(capped)


def clean_text(text: str) -> str:
    """
    Normalize raw text:
    - merge hyphenated line breaks
    - drop non-printable chars
    - remove hard line breaks (flatten columns/line wraps)
    - collapse whitespace
    """
    if not text:
        return ""
    # join words split across line breaks with hyphen
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    # strip non-printable
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    # flatten newlines/tabs to spaces to reduce column/line-break artifacts
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_stopwords(extra: Optional[Iterable[str]] = None) -> Set[str]:
    merged = set(BASE_STOPWORDS_RU) | set(BASE_STOPWORDS_EN)
    if extra:
        merged |= {w.strip().lower() for w in extra if w.strip()}
    env_extra = os.getenv("EXTRA_STOPWORDS")
    if env_extra:
        merged |= {w.strip().lower() for w in env_extra.split(",") if w.strip()}
    return merged


def extract_candidate_terms(
    text: str,
    stopwords: Set[str],
    min_length: int = 3,
    min_frequency: int = 1,
    max_words: int = 3,
) -> Counter:
    """
    Extract multilingual candidate terms using regex + stopword filtering.
    """
    if not text:
        return Counter()

    candidates = MULTILINGUAL_PATTERN.findall(text)
    filtered: List[str] = []
    for raw in candidates:
        normalized = normalize_term(raw)
        if len(normalized.split()) > max_words:
            continue
        if len(normalized) < min_length:
            continue
        if normalized.lower() in stopwords:
            continue
        filtered.append(normalized)

    freq = Counter(term for term in filtered)
    for term in list(freq):
        if freq[term] < min_frequency:
            del freq[term]
    return freq


# --------------------------- Categorization ------------------------------- #


def category_guided_extraction(
    text: str,
    categories: Dict[str, Dict[str, List[str]]],
    stopwords: Set[str],
    min_frequency: int,
    max_words: int = 3,
) -> List[TermRecord]:
    """
    Match known category keywords inside the text without API calls.
    """
    results: List[TermRecord] = []
    if not text:
        return results

    counts = extract_candidate_terms(
        text,
        stopwords=stopwords,
        min_length=3,
        min_frequency=1,
        max_words=max_words,
    )
    lower_counts: Dict[str, int] = {}
    for term, freq in counts.items():
        lower_counts[term.lower()] = lower_counts.get(term.lower(), 0) + freq

    for category, meta in categories.items():
        comment = meta.get("comment", "")
        for kw in meta.get("keywords", []):
            normalized_kw = normalize_term(kw)
            if normalized_kw.lower() in stopwords:
                continue
            freq = lower_counts.get(normalized_kw.lower(), 0)
            if freq >= min_frequency:
                results.append(
                    TermRecord(
                        category=category,
                        keyword=normalized_kw,
                        comment=comment,
                    )
                )
    return results


def chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield a sequence in fixed-size chunks."""
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def batch_categorize_with_llm(
    terms: Sequence[str],
    categories: Dict[str, Dict[str, List[str]]],
    model: str,
    temperature: float,
) -> Dict[str, str]:
    """
    Use OpenAI to map terms to categories. Returns {term: category}.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        print("⚠️ OpenAI not configured; skipping LLM categorization.")
        return {}

    client = OpenAI(api_key=api_key)
    mapped: Dict[str, str] = {}
    category_names = list(categories.keys())
    system_prompt = (
        "Ты — медицинский ассистент. "
        "Классифицируй каждый термин в одну из заданных категорий. "
        "Если не подходит, используй 'Other'. "
        "Верни JSON со списком объектов: [{\"term\": \"...\", \"category\": \"...\"}]."
    )

    for batch in chunked(list(terms), 50):
        user_prompt = (
            "Категории:\n- "
            + "\n- ".join(category_names)
            + "\nТермины:\n- "
            + "\n- ".join(batch)
        )
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=60,
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            items = data if isinstance(data, list) else data.get("items") or data.get("result") or []
            if isinstance(items, dict):
                # handle {"term": "x", "category": "y"} single object
                items = [items]
            for item in items:
                term = normalize_term(str(item.get("term", "")))
                category = normalize_term(str(item.get("category", "")))
                if term:
                    mapped[term] = category or "Other"
        except Exception as exc:  # pragma: no cover - network path
            print(f"❌ OpenAI batch failed: {exc}")
    return mapped


def llm_only_extraction(
    text: str,
    categories: Dict[str, Dict[str, List[str]]],
    model: str,
    temperature: float,
    max_terms: int = 50,
) -> List[TermRecord]:
    """
    Ask the LLM to extract and categorize terms directly from raw text.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        print("⚠️ OpenAI not configured; cannot run llm-only mode.")
        return []

    client = OpenAI(api_key=api_key)
    category_names = list(categories.keys())
    system_prompt = (
        "Ты извлекаешь медицинские термины (рус/англ) из текста и относишь к категориям. "
        "Верни JSON: [{\"term\": \"...\", \"category\": \"...\"}]. "
        f"Максимум {max_terms} терминов."
    )
    user_prompt = "Категории:\n- " + "\n- ".join(category_names) + "\nТекст:\n" + text[:6000]

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        items = data if isinstance(data, list) else data.get("items") or data.get("result") or []
        if isinstance(items, dict):
            items = [items]
    except Exception as exc:  # pragma: no cover - network path
        print(f"❌ OpenAI llm-only failed: {exc}")
        return []

    results: List[TermRecord] = []
    for item in items:
        term = normalize_term(str(item.get("term", "")))
        category = normalize_term(str(item.get("category", "")))
        if not term:
            continue
        comment = categories.get(category, {}).get("comment", "")
        results.append(TermRecord(category=category or "Other", keyword=term, comment=comment))
    return results


# ------------------------------- Export ----------------------------------- #


def export_to_maxqda_format(records: List[TermRecord], output_file: Path) -> None:
    """
    Write records into MAXQDA-compatible CSV/XLSX for dictionary import.
    Columns (single sheet, numeric flags 0/1):
    Category | Memo | Search item | Whole word |
    Case sensitivity | Starting letters | Category activated | Search item activated
    """
    ensure_output_path(output_file)
    # Deduplicate by (category, keyword)
    unique: Dict[Tuple[str, str], TermRecord] = {}
    for rec in records:
        key = (rec.category, rec.keyword)
        if key not in unique:
            unique[key] = rec
    records = list(unique.values())
    rows = []
    for rec in records:
        category = (rec.category or "").replace("\n", " ").replace("\t", " ").strip()
        memo = (rec.comment or "").replace("\n", " ").replace("\t", " ").strip()
        keyword = (rec.keyword or "").replace("\n", " ").replace("\t", " ").strip()
        rows.append(
            {
                "Category": category,
                "Memo": memo,
                "Search item": keyword,
                "Whole word": 0,
                "Case sensitivity": 0,
                "Starting letters": 1,
                "Category activated": 1,
                "Search item activated": 1,
            }
        )

    # Ensure category nodes exist as empty search items (deactivated search)
    category_nodes: Set[str] = set()
    for rec in records:
        cat = (rec.category or "").replace("\n", " ").replace("\t", " ").strip()
        if not cat:
            continue
        parts = [p for p in re.split(r"[\\/]+", cat) if p]
        for i in range(1, min(len(parts), MAX_CATEGORY_DEPTH) + 1):
            prefix = CATEGORY_SEPARATOR.join(parts[:i])
            category_nodes.add(prefix)
    for cat in category_nodes:
        rows.append(
            {
                "Category": cat,
                "Memo": "",
                "Search item": "",
                "Whole word": 0,
                "Case sensitivity": 0,
                "Starting letters": 0,
                "Category activated": 1,
                "Search item activated": 0,
            }
        )

    df = pd.DataFrame(rows, columns=[
        "Category",
        "Memo",
        "Search item",
        "Whole word",
        "Case sensitivity",
        "Starting letters",
        "Category activated",
        "Search item activated",
    ]).drop_duplicates()

    if output_file.suffix.lower() == ".xlsx":
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False, encoding="utf-8-sig", sep=";")
    print(f"[ok] Exported {len(records)} records to {output_file}")


# ------------------------------- Pipeline --------------------------------- #


def run_category_guided(
    docs: List[Tuple[str, str]],
    categories: Dict[str, Dict[str, List[str]]],
    stopwords: Set[str],
    min_frequency: int,
    max_words: int,
) -> List[TermRecord]:
    """Run dictionary-only extraction across all documents."""
    merged_text = " ".join(text for _, text in docs)
    return category_guided_extraction(
        text=merged_text,
        categories=categories,
        stopwords=stopwords,
        min_frequency=min_frequency,
        max_words=max_words,
    )


def run_hybrid(
    docs: List[Tuple[str, str]],
    categories: Dict[str, Dict[str, List[str]]],
    stopwords: Set[str],
    min_frequency: int,
    model: str,
    temperature: float,
    max_words: int,
) -> List[TermRecord]:
    """Regex candidates + LLM categorization."""
    all_terms = Counter()
    for _, text in docs:
        all_terms.update(
            extract_candidate_terms(
                text,
                stopwords,
                min_length=3,
                min_frequency=1,
                max_words=max_words,
            )
        )

    # filter by min_frequency
    terms = [term for term, freq in all_terms.items() if freq >= min_frequency]
    # keep top-N to control latency/cost
    terms = terms[:100]
    llm_mapping = batch_categorize_with_llm(terms, categories, model=model, temperature=temperature)

    results: List[TermRecord] = []
    for term in terms:
        category = llm_mapping.get(term, "Other")
        comment = categories.get(category, {}).get("comment", "")
        results.append(TermRecord(category=category, keyword=term, comment=comment))
    return results


def run_llm_only(
    docs: List[Tuple[str, str]],
    categories: Dict[str, Dict[str, List[str]]],
    model: str,
    temperature: float,
    max_terms: int,
) -> List[TermRecord]:
    """LLM extracts and categorizes directly."""
    merged_text = " ".join(text for _, text in docs)
    return llm_only_extraction(
        merged_text,
        categories=categories,
        model=model,
        temperature=temperature,
        max_terms=max_terms,
    )


def run_exact_codebook_match(
    docs: List[Tuple[str, str]],
    codebook_index: Dict[int, Dict[str, List[Tuple[str, str]]]],
    min_frequency: int,
    max_ngram: int = 6,
    doc_count: int = 1,
) -> List[TermRecord]:
    """
    Exact matching against the provided codebook index (no LLM).
    """
    effective_min = max(min_frequency, 3 * max(doc_count, 1))
    combined_text = " ".join(text for _, text in docs)
    tokens = re.findall(r"[а-яА-ЯёЁa-zA-Z0-9]+", combined_text.lower())
    matches: Dict[Tuple[str, str], int] = {}

    lengths = [l for l in codebook_index.keys() if l <= max_ngram]
    for n in lengths:
        if len(tokens) < n:
            continue
        mapping = codebook_index[n]
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if phrase in mapping:
                for category_name, keyword_raw in mapping[phrase]:
                    key = (category_name, keyword_raw)
                    matches[key] = matches.get(key, 0) + 1

    results: List[TermRecord] = []
    seen: Set[Tuple[str, str]] = set()
    for (category_name, keyword_raw), freq in matches.items():
        if freq >= effective_min:
            key = (category_name, keyword_raw)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                TermRecord(
                    category=category_name,
                    keyword=keyword_raw,
                    comment="Imported from codebook",
                )
            )
    return results


# --------------------------------- CLI ------------------------------------ #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI arguments for the general extractor."""
    parser = argparse.ArgumentParser(
        description="Extract medical terms (RU/EN) and export to MAXQDA CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="input", help="File or folder with PDF/TXT")
    parser.add_argument(
        "--output",
        default="output/medical_dictionary.xlsx",
        help="Destination CSV/XLSX (UTF-8-sig for CSV)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "category-guided", "llm-only"],
        default="category-guided",
        help="hybrid=regex+LLM, category-guided=dictionary only, llm-only=LLM does all",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Min frequency for a term to be kept",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=50,
        help="LLM-only: maximum extracted terms",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=3,
        help="Maximum words per candidate term (regex extraction)",
    )
    parser.add_argument(
        "--categories-file",
        help="Optional JSON file with categories overriding defaults",
    )
    parser.add_argument(
        "--codebook-xlsx",
        help="Optional XLSX codebook (Search item ...) to use for categories/keywords",
    )
    parser.add_argument(
        "--exact-codebook-match",
        action="store_true",
        help="If set, skip regex/LLM and match text only against codebook entries (no LLM).",
    )
    return parser.parse_args(argv)


def load_categories(path: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    """Load categories from JSON file or return defaults."""
    if not path:
        return DEFAULT_CATEGORIES
    category_path = Path(path)
    if not category_path.exists():
        print(f"⚠️ Categories file not found: {category_path}, using defaults.")
        return DEFAULT_CATEGORIES
    try:
        data = json.loads(category_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): dict(v) for k, v in data.items()}
    except Exception as exc:
        print(f"❌ Failed to read categories file: {exc}")
    return DEFAULT_CATEGORIES


def load_codebook_xlsx(path: Optional[str]) -> Optional[Dict[str, Dict[str, List[str]]]]:
    """
    Load categories/keywords from an XLSX codebook (MAXQDA-style).
    Supports legacy “Search item” paths or explicit “Category” + “Search item”.
    """
    if not path:
        return None
    xlsx_path = Path(path)
    if not xlsx_path.exists():
        print(f"⚠️ Codebook XLSX not found: {xlsx_path}")
        return None
    try:
        df = pd.read_excel(xlsx_path)
        has_category_col = "Category" in df.columns
        has_search_item = "Search item" in df.columns
        if not has_category_col and not has_search_item:
            print("⚠️ Codebook XLSX missing Category or Search item column")
            return None
        if "Search item activated" in df.columns:
            df = df[df["Search item activated"].astype(str) != "0"]
        categories: Dict[str, Dict[str, List[str]]] = {}
        for _, row in df.iterrows():
            raw_path = str(row.get("Category", "")).strip() if has_category_col else str(row.get("Search item", "")).strip()
            if not raw_path:
                continue
            parts = [p.strip() for p in re.split(r"[\\/]+", raw_path) if p.strip()]
            if not parts:
                continue
            keyword_raw = parts[-1]
            category_name = build_category_name(parts[:-1])
            bucket = categories.setdefault(category_name, {"keywords": [], "comment": "Imported from codebook"})
            if keyword_raw not in bucket["keywords"]:
                bucket["keywords"].append(keyword_raw)
        if not categories:
            print(f"⚠️ No active rows loaded from codebook {xlsx_path}")
            return None
        return categories
    except Exception as exc:
        print(f"❌ Failed to load XLSX codebook: {exc}")
        return None


def build_codebook_index(path: str) -> Optional[Dict[int, Dict[str, List[Tuple[str, str]]]]]:
    """
    Build normalized keyword index by token length: {length: {normalized_kw: [(category, raw_kw), ...]}}
    """
    xlsx_path = Path(path)
    if not xlsx_path.exists():
        print(f"⚠️ Codebook XLSX not found: {xlsx_path}")
        return None
    try:
        df = pd.read_excel(xlsx_path)
        has_category_col = "Category" in df.columns
        has_search_item = "Search item" in df.columns
        if not has_category_col and not has_search_item:
            print("⚠️ Codebook XLSX missing Category or Search item column")
            return None
        if "Search item activated" in df.columns:
            df = df[df["Search item activated"].astype(str) != "0"]
        index: Dict[int, Dict[str, List[Tuple[str, str]]]] = {}
        for _, row in df.iterrows():
            raw_path = str(row.get("Category", "")).strip() if has_category_col else str(row.get("Search item", "")).strip()
            if not raw_path:
                continue
            parts = [p.strip() for p in re.split(r"[\\/]+", raw_path) if p.strip()]
            if not parts:
                continue
            keyword_raw = parts[-1]
            norm_kw = normalize_phrase_for_match(keyword_raw)
            if not norm_kw:
                continue
            n = len(norm_kw.split())
            category_name = build_category_name(parts[:-1])
            bucket = index.setdefault(n, {}).setdefault(norm_kw, [])
            if (category_name, keyword_raw) not in bucket:
                bucket.append((category_name, keyword_raw))
        if not index:
            print(f"⚠️ No active rows loaded into index from {xlsx_path}")
            return None
        return index
    except Exception as exc:
        print(f"❌ Failed to build codebook index: {exc}")
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    load_env()

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

    categories = load_codebook_xlsx(args.codebook_xlsx) or load_categories(args.categories_file)
    stopwords = build_stopwords()
    codebook_index = build_codebook_index(args.codebook_xlsx) if args.codebook_xlsx else None

    input_path = Path(args.input)
    docs = load_documents(input_path)
    if not docs:
        print("❌ No readable documents found.")
        return 1

    print(f"📂 Loaded {len(docs)} documents from {input_path}")
    print(f"🧾 Mode: {args.mode}")

    if args.exact_codebook_match:
        if not codebook_index:
            print("❌ Codebook index is required for exact-codebook-match mode.")
            return 1
        records = run_exact_codebook_match(
            docs,
            codebook_index=codebook_index,
            min_frequency=args.min_frequency,
            max_ngram=args.max_words,
            doc_count=len(docs),
        )
    elif args.mode == "category-guided":
        records = run_category_guided(
            docs, categories, stopwords, args.min_frequency, max_words=args.max_words
        )
    elif args.mode == "hybrid":
        records = run_hybrid(
            docs,
            categories,
            stopwords,
            args.min_frequency,
            model=model,
            temperature=temperature,
            max_words=args.max_words,
        )
    else:
        records = run_llm_only(
            docs,
            categories,
            model=model,
            temperature=temperature,
            max_terms=args.max_terms,
        )

    output_path = Path(args.output)
    export_to_maxqda_format(records, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
