from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


CHEM_PREFIX = "Chemicals and Drugs\\"

STOPWORDS_CHEM = {
    "induced",
    "related",
    "associated",
    "derivative",
    "derivatives",
    "analog",
    "analogs",
    "compound",
    "compounds",
    "inhibitor",
    "inhibitors",
    "agonist",
    "antagonist",
    "effect",
    "effects",
}

RE_MULTI_SPACE = re.compile(r"\s+")
RE_SPLIT_LIST = re.compile(r"\s*[,;]\s*")
RE_PARENS = re.compile(r"^(.*?)\s*\((.*?)\)\s*$")

QUALIFIER_TOKENS = {
    "human",
    "humans",
    "animal",
    "animals",
    "mouse",
    "mice",
    "rat",
    "rats",
    "dog",
    "dogs",
    "rabbit",
    "rabbits",
    "male",
    "female",
    "adult",
    "adults",
    "child",
    "children",
}

BANNED_SINGLE_TERMS = {
    "human",
    "humans",
    "patient",
    "patients",
    "study",
    "studies",
    "result",
    "results",
    "method",
    "methods",
    "group",
    "groups",
}


def norm_spaces(s: str) -> str:
    return RE_MULTI_SPACE.sub(" ", s.strip())


def to_clean_text(s: str) -> str:
    s = str(s).strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[/\\]+", " ", s)  # слеши в пробел
    s = re.sub(r"[-]+", " ", s)  # дефисы в пробел
    s = re.sub(r"[^\w\s]", " ", s)  # пунктуация в пробел
    return norm_spaces(s)


def looks_like_abbrev(x: str) -> bool:
    x = x.strip()
    if not (2 <= len(x) <= 10):
        return False
    if " " in x:
        return False
    return any(c.isupper() for c in x) or any(c.isdigit() for c in x)


def split_comma_safely(term: str) -> set[str]:
    term = norm_spaces(term)
    out = {term}

    # semicolon: treat as list
    if ";" in term:
        parts = [norm_spaces(p) for p in term.split(";") if p.strip()]
        out |= set(parts)
        return out

    if "," not in term:
        return out

    parts = [norm_spaces(p) for p in term.split(",") if p.strip()]
    if len(parts) != 2:
        out |= set(parts)
        return out

    a, b = parts[0], parts[1]
    a_words = a.split()
    b_words = b.split()
    b_low = b.lower()

    # MeSH-like qualifier: "Mammary Glands, Human"
    if len(b_words) == 1 and b_low in QUALIFIER_TOKENS:
        out.add(a)
        out.add(norm_spaces(f"{b} {a}"))
        return out

    # Real list: "Indazoles, Benzimidazoles"
    if len(a_words) == 1 and len(b_words) == 1:
        out.add(a)
        out.add(b)
        return out

    # Default inversion: "Arrest, Heart" -> Heart Arrest
    out.add(a)
    out.add(norm_spaces(f"{b} {a}"))
    return out


def variants_global(term: str) -> set[str]:
    term = norm_spaces(term)
    out: set[str] = set()
    if not term or term.lower() == "nan":
        return out

    out.add(term)

    # (1) X (Y) -> X + Y
    m = RE_PARENS.match(term)
    if m:
        left = norm_spaces(m.group(1))
        inside = norm_spaces(m.group(2))
        if left:
            out.add(left)
        if inside and looks_like_abbrev(inside):
            out.add(inside)

    out |= split_comma_safely(term)

    # (3) cleaned punctuation/dashes
    cleaned = to_clean_text(term)
    if cleaned:
        out.add(cleaned)

    # filter noisy singles
    filtered: set[str] = set()
    for v in out:
        v = norm_spaces(v)
        if not v:
            continue
        v_low = v.lower()
        if v_low in BANNED_SINGLE_TERMS:
            continue
        if len(v.split()) == 1 and v_low in QUALIFIER_TOKENS:
            continue
        filtered.add(v)

    return filtered


def variants_chem(term: str) -> set[str]:
    """Агрессивные варианты только для химии."""
    out: set[str] = set()
    cleaned = to_clean_text(term)
    toks = [t for t in cleaned.lower().split() if t]

    core = [t for t in toks if t not in STOPWORDS_CHEM and not t.isdigit()]
    if len(core) >= 2:
        out.add(" ".join(core[:3]))

    if toks:
        longest = max(toks, key=len)
        if len(longest) >= 6:
            out.add(longest)

    return out


def chem_symbol_variants(s: str) -> set[str]:
    """
    Extra variants for chemical/biomedical symbols:
    IL-6 <-> IL6
    5-HT <-> 5HT <-> 5 HT
    TNF-α <-> TNFα <-> TNF alpha
    """
    s = norm_spaces(str(s))
    out: set[str] = set()
    if not s:
        return out

    out.add(s.replace(".", ""))
    out.add(re.sub(r"(?<=\w)-(?=\w)", "", s))
    out.add(re.sub(r"\s*-\s*", "-", s))
    out.add(re.sub(r"\s*-\s*", "", s))
    out.add(re.sub(r"\s*-\s*", " ", s))

    if re.search(r"\d+\s+[A-Za-z]{1,5}", s):
        out.add(s.replace(" ", ""))

    for g, latin in {"α": "alpha", "β": "beta", "γ": "gamma"}.items():
        if g in s:
            out.add(s.replace(g, latin))
            out.add(s.replace(g, ""))

    out = {norm_spaces(x) for x in out if x}
    return out


def main() -> None:
    inp = Path("codebook/common_without_brackets.xlsx")
    out = Path("codebook/common.cleaned.xlsx")

    df = pd.read_excel(inp)

    if "Category" not in df.columns or "Search item" not in df.columns:
        raise ValueError("Нужны колонки: Category и Search item")

    flag_cols = [c for c in df.columns if c not in ("Category", "Search item")]

    rows = []
    seen: set[tuple[str, str]] = set()

    for _, r in df.iterrows():
        cat = str(r["Category"]).strip()
        term = str(r["Search item"]).strip()
        if not cat or not term or term.lower() == "nan":
            continue

        vset = variants_global(term)
        vset |= chem_symbol_variants(term)
        if cat.startswith(CHEM_PREFIX):
            for v in list(vset):
                vset |= variants_chem(v)
                vset |= chem_symbol_variants(v)

        for v in sorted(vset):
            v = norm_spaces(v)
            if len(v) < 3:
                continue
            key = (cat, v)
            if key in seen:
                continue
            seen.add(key)
            row = {"Category": cat, "Search item": v}
            for c in flag_cols:
                row[c] = r[c]
            rows.append(row)

    out_df = pd.DataFrame(rows)

    if "Search item activated" in out_df.columns:
        out_df["Search item activated"] = 1
    if "Category activated" in out_df.columns:
        out_df["Category activated"] = 1

    out_df.to_excel(out, index=False)
    print(f"[ok] saved: {out} | rows: {len(out_df)}")


if __name__ == "__main__":
    main()
