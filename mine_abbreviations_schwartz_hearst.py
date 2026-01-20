from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Patterns for LF (SF) and SF (LF)
PAT_LF_SF = re.compile(
    r"\b([A-Za-z][A-Za-z0-9 \-/,]{8,})\s*\(([A-Za-z0-9][A-Za-z0-9.\-]{2,12})\)"
)
PAT_SF_LF = re.compile(
    r"\b([A-Za-z0-9][A-Za-z0-9.\-]{2,12})\s*\(([A-Za-z][A-Za-z0-9 \-/,]{8,})\)"
)

GREEK_MAP = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "μ": "micro",
    "κ": "kappa",
    "λ": "lambda",
    "ω": "omega",
}

BAN_ABBR = {"HR", "BP", "CI", "OR", "RR", "SD"}  # extend if needed


def normalize_text(s: str) -> str:
    s = s.replace("\u00A0", " ")
    for g, latin in GREEK_MAP.items():
        s = s.replace(g, f" {latin} ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_pdf_text(path: Path) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(str(path))
        chunks: List[str] = []
        for p in reader.pages[:60]:
            chunks.append(p.extract_text() or "")
        return normalize_text("\n".join(chunks))
    except Exception:
        return ""


def looks_good_abbr(abbr: str) -> bool:
    abbr = abbr.strip()
    if not (2 <= len(abbr) <= 12):
        return False
    if " " in abbr:
        return False
    if abbr.upper() in BAN_ABBR:
        return False
    return any(c.isupper() for c in abbr) or any(c.isdigit() for c in abbr)


def abbr_variants(abbr: str) -> set[str]:
    abbr = normalize_text(abbr)
    variants: set[str] = set()
    if not abbr:
        return variants

    v = abbr.replace(".", "")
    variants.add(v)
    variants.add(v.replace("-", ""))
    variants.add(v.replace("-", " "))
    variants = {re.sub(r"\s+", " ", x).strip() for x in variants}
    for x in list(variants):
        if re.search(r"\d+\s+[A-Za-z]{1,4}", x):
            variants.add(x.replace(" ", ""))
    variants |= {x.upper() for x in variants}
    variants = {x for x in variants if 2 <= len(x) <= 16}
    return variants


def clean_long_form(lf: str) -> str:
    lf = normalize_text(lf)
    lf = lf.strip(" ,;/")
    lf = re.sub(r"\s+", " ", lf)
    return lf


def schwartz_hearst_no_parens(text: str) -> Iterable[Tuple[str, str]]:
    """
    Simplified Schwartz & Hearst heuristic:
    - find SF (2-10 chars, uppercase/digit)
    - look left for candidate LF words whose initials match SF
    """
    tokens = text.split()
    results = []
    for i, tok in enumerate(tokens):
        sf = tok.strip()
        if not looks_good_abbr(sf):
            continue
        # skip if already matched via parens patterns
        if "(" in sf or ")" in sf:
            continue
        sf_clean = re.sub(r"[^\w]", "", sf)
        if not (2 <= len(sf_clean) <= 12):
            continue
        # look back up to 20 words
        window = tokens[max(0, i - 20) : i]
        initials = "".join(w[0].upper() for w in window if w and w[0].isalpha())
        if not initials:
            continue
        # find shortest suffix of initials that matches sf_clean
        sfu = sf_clean.upper()
        for start in range(len(initials)):
            if initials[start:].startswith(sfu):
                lf = " ".join(window[start:])
                lf = clean_long_form(lf)
                if len(lf.split()) >= 2:
                    results.append((lf, sf))
                break
    return results


def extract_abbr_pairs(text: str) -> Counter:
    pairs: Counter[tuple[str, str, str]] = Counter()

    # LF (SF)
    for m in PAT_LF_SF.finditer(text):
        lf = clean_long_form(m.group(1))
        sf = m.group(2).strip()
        if looks_good_abbr(sf) and len(lf.split()) >= 2:
            for v in abbr_variants(sf):
                pairs[(lf, v, "LF(SF)")] += 1

    # SF (LF)
    for m in PAT_SF_LF.finditer(text):
        sf = m.group(1).strip()
        lf = clean_long_form(m.group(2))
        if looks_good_abbr(sf) and len(lf.split()) >= 2:
            for v in abbr_variants(sf):
                pairs[(lf, v, "SF(LF)")] += 1

    # NO_PARENS heuristic
    for lf, sf in schwartz_hearst_no_parens(text):
        if looks_good_abbr(sf):
            for v in abbr_variants(sf):
                pairs[(lf, v, "NO_PARENS")] += 1

    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract abbreviations from PDF corpus.")
    parser.add_argument("--pdf-dir", default="primery", help="Folder with PDF files")
    parser.add_argument(
        "--out", default="codebook/abbrev_from_corpus.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--min-count", type=int, default=2, help="Minimum frequency to keep a pair"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs: Counter[tuple[str, str, str]] = Counter()
    for pdf in pdf_dir.glob("*.pdf"):
        text = read_pdf_text(pdf)
        if not text:
            continue
        pairs.update(extract_abbr_pairs(text))

    rows = []
    for (lf, abbr, pat), cnt in pairs.items():
        if cnt < args.min_count:
            continue
        rows.append({"long_form": lf, "abbr": abbr, "count": cnt, "pattern": pat})

    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] saved: {out_csv} | rows: {len(df)}")


if __name__ == "__main__":
    main()
