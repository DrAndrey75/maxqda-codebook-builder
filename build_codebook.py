"""
Pipeline: read documents from input folder, match against codebook XLSX,
and export a MAXQDA-ready code table into Maxqda_Codes/output/.
Uses exact codebook matching (no LLM).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from extract_medical_terms import (  # noqa: E402
    export_to_maxqda_format,
    load_documents,
    load_env,
    run_exact_codebook_match,
    build_codebook_index,
)


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build MAXQDA codebook from documents using an XLSX codebook (exact match)."
    )
    parser.add_argument(
        "--input",
        default=str(base / "primery"),
        help="Folder with PDF/TXT documents",
    )
    parser.add_argument(
        "--codebook-xlsx",
        default=str(base / "codebook" / "common_without_brackets.xlsx"),
        help="XLSX codebook (Search item...)",
    )
    parser.add_argument(
        "--output",
        default=str(base / "output" / "maxqda_codebook.xlsx"),
        help="Destination XLSX",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum occurrences of a codebook keyword to keep",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=3,
        help="Maximum n-gram length when matching codebook keywords",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()

    codebook_index = build_codebook_index(args.codebook_xlsx)
    if not codebook_index:
        print(f"❌ Cannot build index from {args.codebook_xlsx}")
        return 1

    docs = load_documents(Path(args.input))
    if not docs:
        print(f"❌ No readable documents in {args.input}")
        return 1

    records = run_exact_codebook_match(
        docs,
        codebook_index=codebook_index,
        min_frequency=args.min_frequency,
        max_ngram=args.max_words,
    )
    export_to_maxqda_format(records, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
