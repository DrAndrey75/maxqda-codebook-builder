from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


CHEM_PREFIX_DEFAULT = r"Chemicals and Drugs\\"

BANNED_ABBR = {
    "HR",
    "BP",
    "RR",
    "SD",
    "SE",
    "CI",
    "mg",
    "kg",
    "cm",
    "mm",
    "vs",
    "et",
    "al",
}

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


def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def normalize_key(s: str) -> str:
    """
    Normalize for matching long_form with Search item:
    - lower
    - remove punctuation
    - collapse spaces
    """
    s = str(s).lower()
    for g, latin in GREEK_MAP.items():
        s = s.replace(g, f" {latin} ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_good_abbr(abbr: str, min_len: int = 3, max_len: int = 12) -> bool:
    abbr = norm_spaces(abbr)
    if not (min_len <= len(abbr) <= max_len):
        return False
    if " " in abbr:
        return False
    abbr2 = abbr.replace(".", "").upper()
    if abbr2 in BANNED_ABBR:
        return False
    return any(c.isupper() for c in abbr) or any(c.isdigit() for c in abbr)


def abbr_variants(abbr: str) -> set[str]:
    """
    Glue forms:
    5-HT -> 5HT
    IL-6 -> IL6
    H.T.P. -> HTP
    """
    abbr = norm_spaces(abbr)
    if not abbr:
        return set()

    out: set[str] = set()

    a = abbr.replace(".", "")
    out.add(a)
    out.add(a.replace("-", ""))
    out.add(a.replace("-", " "))
    out.add(re.sub(r"\s*-\s*", "-", a))
    out.add(re.sub(r"\s*-\s*", "", a))
    out.add(re.sub(r"\s*-\s*", " ", a))

    for x in list(out):
        if re.search(r"\d+\s+[A-Za-z]{1,5}", x):
            out.add(x.replace(" ", ""))

    out |= {x.upper() for x in out}
    out = {x for x in out if 2 <= len(x) <= 16 and " " not in x}
    return out


def build_abbrev_kb(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    pairs_df columns: long_form, abbr, count, pattern
    Output: KB by abbr with best_long_form, counts, top_share.
    """
    agg = defaultdict(int)
    patterns = defaultdict(set)

    for _, r in pairs_df.iterrows():
        lf = norm_spaces(r["long_form"])
        ab = norm_spaces(r["abbr"])
        cnt = int(r.get("count", 1))
        pat = str(r.get("pattern", "")).strip()

        if not lf or not ab:
            continue

        agg[(ab, lf)] += cnt
        if pat:
            patterns[(ab, lf)].add(pat)

    by_abbr = defaultdict(list)
    for (ab, lf), cnt in agg.items():
        by_abbr[ab].append((lf, cnt, sorted(patterns[(ab, lf)])))

    rows = []
    for ab, lst in by_abbr.items():
        total = sum(x[1] for x in lst)
        lst_sorted = sorted(lst, key=lambda x: x[1], reverse=True)
        best_lf, best_cnt, best_pats = lst_sorted[0]
        top_share = best_cnt / total if total else 0.0

        rows.append(
            {
                "abbr": ab,
                "best_long_form": best_lf,
                "best_count": best_cnt,
                "total_count": total,
                "top_share": round(top_share, 4),
                "patterns_best": ",".join(best_pats),
                "n_candidates": len(lst_sorted),
            }
        )

    kb = pd.DataFrame(rows).sort_values(
        ["total_count", "top_share"], ascending=[False, False]
    )
    return kb


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--codebook-in", type=str, default="codebook/common.cleaned.xlsx")
    ap.add_argument("--pairs-csv", type=str, default="codebook/abbrev_from_corpus.csv")
    ap.add_argument(
        "--codebook-out",
        type=str,
        default="codebook/common.cleaned.with_kb_abbrev.xlsx",
    )
    ap.add_argument("--kb-out", type=str, default="codebook/abbrev_kb.csv")
    ap.add_argument("--chem-prefix", type=str, default=CHEM_PREFIX_DEFAULT)
    ap.add_argument(
        "--min-total",
        type=int,
        default=3,
        help="минимальная общая частота abbr в корпусе",
    )
    ap.add_argument(
        "--min-share",
        type=float,
        default=0.60,
        help="доля лучшей расшифровки (уверенность)",
    )
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=12)

    args = ap.parse_args()

    codebook_in = Path(args.codebook_in)
    pairs_csv = Path(args.pairs_csv)
    codebook_out = Path(args.codebook_out)
    kb_out = Path(args.kb_out)

    df = pd.read_excel(codebook_in)
    pairs = pd.read_csv(pairs_csv)

    if "Category" not in df.columns or "Search item" not in df.columns:
        raise ValueError("codebook должен содержать колонки: Category и Search item")

    kb = build_abbrev_kb(pairs)

    kb_f = kb[
        (kb["total_count"] >= args.min_total) & (kb["top_share"] >= args.min_share)
    ].copy()
    kb_f = kb_f[
        kb_f["abbr"].apply(lambda x: looks_good_abbr(x, args.min_len, args.max_len))
    ]

    kb_out.parent.mkdir(parents=True, exist_ok=True)
    kb.to_csv(kb_out, index=False, encoding="utf-8-sig")

    chem_prefix = args.chem_prefix
    df_chem = df[df["Category"].astype(str).str.startswith(chem_prefix)].copy()

    term_to_cat: dict[str, str] = {}
    for _, r in df_chem.iterrows():
        term = normalize_key(r["Search item"])
        cat = str(r["Category"]).strip()
        if term and cat:
            term_to_cat[term] = cat

    flag_cols = [c for c in df.columns if c not in ("Category", "Search item")]
    existing = set(zip(df["Category"].astype(str), df["Search item"].astype(str)))

    rows_add = []
    added = 0

    for _, r in kb_f.iterrows():
        ab = str(r["abbr"]).strip()
        lf = str(r["best_long_form"]).strip()
        lf_key = normalize_key(lf)
        cat = term_to_cat.get(lf_key)
        if not cat:
            for tkey, tcat in term_to_cat.items():
                if lf_key in tkey or tkey in lf_key:
                    cat = tcat
                    break
        if not cat or not cat.startswith(chem_prefix):
            continue

        for v in abbr_variants(ab):
            key = (cat, v)
            if key in existing:
                continue
            existing.add(key)
            row = {"Category": cat, "Search item": v}
            for c in flag_cols:
                if "activated" in c.lower():
                    row[c] = 1
                else:
                    row[c] = 0
            rows_add.append(row)
            added += 1

    out = pd.concat([df, pd.DataFrame(rows_add)], ignore_index=True)
    if "Search item activated" in out.columns:
        out["Search item activated"] = 1
    if "Category activated" in out.columns:
        out["Category activated"] = 1

    codebook_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(codebook_out, index=False)

    print(f"[ok] KB saved: {kb_out}")
    print(f"[ok] Codebook saved: {codebook_out}")
    print(f"[ok] Added abbreviations rows: {added}")


if __name__ == "__main__":
    main()
