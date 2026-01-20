from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Default chemistry branch
CHEM_PREFIX_DEFAULT = r"Chemicals and Drugs\\"

# Basic abbreviation heuristic
def looks_like_abbr(tok: str) -> bool:
    tok = tok.strip()
    if not (2 <= len(tok) <= 12):
        return False
    if " " in tok:
        return False
    return any(c.isupper() for c in tok) or any(c.isdigit() for c in tok)


def normalize_path(cat: str, sep: str = "\\") -> str:
    # unify slashes, remove extra separators/spaces, cap empty -> ""
    cat = cat.replace("/", "\\")
    cat = re.sub(r"[\\\\]+", "\\\\", cat)
    parts = [p.strip() for p in cat.split("\\") if p.strip()]
    return sep.join(parts)


def normalize_text(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def read_pdf_tokens(path: Path, max_pages: int = 10, max_tokens: int = 20000) -> List[str]:
    if PdfReader is None:
        return []
    try:
        reader = PdfReader(str(path))
        chunks: List[str] = []
        for p in reader.pages[:max_pages]:
            t = p.extract_text() or ""
            chunks.append(t)
        text = normalize_text(" ".join(chunks))
        tokens = re.findall(r"[A-Za-z0-9\\-\\']+", text)
        return tokens[:max_tokens]
    except Exception:
        return []


def abbr_variants(abbr: str) -> set[str]:
    abbr = abbr.strip()
    out = set()
    if not abbr:
        return out
    base = abbr.replace(".", "")
    out.add(base)
    out.add(base.replace("-", ""))
    out.add(base.replace("-", " "))
    out.add(re.sub(r"\\s*-\\s*", "-", base))
    out.add(re.sub(r"\\s*-\\s*", "", base))
    out.add(re.sub(r"\\s*-\\s*", " ", base))
    for x in list(out):
        if re.search(r"\\d+\\s+[A-Za-z]{1,5}", x):
            out.add(x.replace(" ", ""))
    out |= {x.upper() for x in out}
    out = {x for x in out if 2 <= len(x) <= 16 and " " not in x}
    return out


def collect_abbr_contexts(tokens: List[str], window: int = 8) -> Dict[str, Counter]:
    contexts: Dict[str, Counter] = defaultdict(Counter)
    for i, tok in enumerate(tokens):
        if not looks_like_abbr(tok):
            continue
        context_tokens = tokens[max(0, i - window) : i + window + 1]
        contexts[tok].update(context_tokens)
    return contexts


def build_category_profiles(df: pd.DataFrame, chem_prefix: str) -> Dict[str, str]:
    df["Category"] = df["Category"].astype(str).apply(normalize_path)
    chem_prefix_norm = normalize_path(chem_prefix)
    df_chem = df[df["Category"].str.startswith(chem_prefix_norm)].copy()
    profiles: Dict[str, List[str]] = defaultdict(list)
    for _, r in df_chem.iterrows():
        cat = r["Category"]
        term = str(r["Search item"]).strip()
        parts = [p for p in cat.split("\\") if p]
        profiles[cat].extend(parts[:3])
        profiles[cat].append(term)
    return {cat: " ".join(vals) for cat, vals in profiles.items()}


def select_category_for_abbr(
    abbr_ctx: Counter,
    cats: List[str],
    cat_vecs,
    tfidf: TfidfVectorizer,
    min_score: float,
) -> Tuple[str, float]:
    if not abbr_ctx or cat_vecs is None or not cats:
        return "", 0.0
    ctx_tokens = []
    for tok, freq in abbr_ctx.items():
        ctx_tokens.extend([tok] * freq)
    ctx_doc = " ".join(ctx_tokens)
    ctx_vec = tfidf.transform([ctx_doc])
    sims = cosine_similarity(ctx_vec, cat_vecs).flatten()
    if sims.size == 0:
        return "", 0.0
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    return (cats[best_idx], float(best_score)) if best_score >= min_score else ("", float(best_score))


def main() -> None:
    parser = argparse.ArgumentParser(description="Context-based abbreviation classifier (patched).")
    parser.add_argument("--pdf-dir", default="primery")
    parser.add_argument("--codebook-in", default="codebook/common.cleaned.with_kb_abbrev.xlsx")
    parser.add_argument("--codebook-out", default="codebook/common.cleaned.with_context_abbrev.xlsx")
    parser.add_argument("--predictions-csv", default="codebook/context_abbrev_predictions.csv")
    parser.add_argument("--chem-prefix", default=CHEM_PREFIX_DEFAULT)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-abbr-count", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--min-score", type=float, default=0.06)
    parser.add_argument("--max-pdf", type=int, default=15, help="Limit number of PDFs to scan for speed")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    df = pd.read_excel(args.codebook_in)
    if "Category" not in df.columns or "Search item" not in df.columns:
        raise ValueError("codebook must have Category and Search item columns")

    chem_prefix_norm = normalize_path(args.chem_prefix)
    cat_profiles = build_category_profiles(df, chem_prefix_norm)
    if not cat_profiles:
        print("⚠️ No chemistry categories found, skipping.")
        return

    all_contexts: Dict[str, Counter] = defaultdict(Counter)
    for idx, pdf in enumerate(pdf_dir.glob("*.pdf")):
        if idx >= args.max_pdf:
            break
        tokens = read_pdf_tokens(pdf)
        ctxs = collect_abbr_contexts(tokens, window=args.window)
        for ab, ctx in ctxs.items():
            all_contexts[ab].update(ctx)

    abbr_counts = {abbr: sum(ctx.values()) for abbr, ctx in all_contexts.items()}
    abbr_filtered = {
        abbr: ctx for abbr, ctx in all_contexts.items() if abbr_counts.get(abbr, 0) >= args.min_abbr_count
    }

    tfidf = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z0-9\-]+",
        lowercase=True,
        min_df=1,
    )
    cats = list(cat_profiles.keys())
    cat_docs = list(cat_profiles.values())
    cat_vecs = tfidf.fit_transform(cat_docs) if cat_docs else None

    predictions = []
    # process most frequent abbreviations first, cap to 500 to control runtime
    abbr_order = sorted(abbr_filtered.items(), key=lambda x: abbr_counts.get(x[0], 0), reverse=True)[:500]
    for abbr, ctx in abbr_order:
        cat, score = select_category_for_abbr(ctx, cats, cat_vecs, tfidf, args.min_score)
        if cat:
            confidence = score
            if confidence >= args.min_confidence:
                predictions.append(
                    {
                        "abbr": abbr,
                        "count": abbr_counts.get(abbr, 0),
                        "predicted_category": normalize_path(cat),
                        "top_score": round(score, 4),
                        "confidence": round(confidence, 4),
                    }
                )

    pred_df = pd.DataFrame(predictions)
    if not pred_df.empty:
        pred_df = pred_df.sort_values(["confidence", "count"], ascending=[False, False])
    else:
        pred_df = pd.DataFrame(columns=["abbr", "count", "predicted_category", "top_score", "confidence"])
    Path(args.predictions_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.predictions_csv, index=False, encoding="utf-8-sig")

    flag_cols = [c for c in df.columns if c not in ("Category", "Search item")]
    existing = set(
        (normalize_path(cat), str(term).strip())
        for cat, term in zip(df["Category"].astype(str), df["Search item"].astype(str))
    )

    rows_add = []
    added = 0
    for _, r in pred_df.iterrows():
        cat = normalize_path(str(r["predicted_category"]))
        abbr = str(r["abbr"]).strip()
        if not cat or not abbr:
            continue
        if not cat.startswith(chem_prefix_norm):
            continue
        for v in abbr_variants(abbr):
            key = (cat, v)
            if key in existing:
                continue
            existing.add(key)
            row = {"Category": cat, "Search item": v}
            for c in flag_cols:
                row[c] = 1 if "activated" in c.lower() else 0
            rows_add.append(row)
            added += 1

    out = pd.concat([df, pd.DataFrame(rows_add)], ignore_index=True)
    if "Search item activated" in out.columns:
        out["Search item activated"] = 1
    if "Category activated" in out.columns:
        out["Category activated"] = 1

    Path(args.codebook_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(args.codebook_out, index=False)

    print(f"[ok] Predictions saved: {args.predictions_csv} | rows: {len(pred_df)}")
    print(f"[ok] Codebook saved: {args.codebook_out} | added abbrev rows: {added}")


if __name__ == "__main__":
    main()
