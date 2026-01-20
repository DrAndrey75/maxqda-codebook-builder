# MAXQDA Codebook Builder (LLM-free)

Offline Python pipeline that builds a MAXQDA-ready dictionary/codebook from an XLSX codebook and a local PDF/TXT corpus (abbreviation mining + exact matching).

## Goal and scope
This project automates preparing a **MAXQDA import file** from:
1) a large source codebook (`.xlsx`), and
2) a local document corpus (`.pdf` / `.txt`).

The goal is to produce a **practical, corpus-aware keyword list**: keep only terms (plus spelling variants and abbreviations) that actually occur in your documents, and export them in a MAXQDA-friendly format.

## What the pipeline does
By default, the pipeline is **offline and LLM-free**:
- Normalizes and expands the source codebook (spelling variants, parentheses, hyphens, MeSH-like patterns).
- Mines abbreviations from the PDF corpus (Schwartz & Hearst-like heuristics).
- Optionally enriches the codebook with mined abbreviations (including a chemistry branch, if applicable).
- Performs **exact matching** of codebook terms against document text.
- Exports the final table to `output/maxqda_codebook.xlsx`.

## Why this is useful
- **For MAXQDA users (researchers/analysts):** less manual work and fewer noisy terms in the final dictionary.
- **For developers/methodologists:** a reproducible pipeline that can be tuned (normalization, thresholds, matching strategy).

---

## For MAXQDA users (researchers)

### Features
- Input: `primery/` (documents) + `codebook/common_without_brackets.xlsx` (source codebook).
- Output: `output/maxqda_codebook.xlsx` (ready to import into MAXQDA).
- Supports PDF and TXT.
- Generates common term variants (punctuation, hyphenation, parentheses, qualifiers).
- Saves intermediate artifacts in `codebook/` for review/auditing.

### Quick start (Windows)
1) Put your PDF/TXT files into `primery/`.
2) Ensure your source codebook is at `codebook/common_without_brackets.xlsx` (required columns: `Category`, `Search item`).
3) Run:
   - `run_pipeline.bat`
4) Result:
   - `output/maxqda_codebook.xlsx`

### Inputs
- `primery/`: `.pdf` and/or `.txt` documents (this folder is intentionally not tracked in git).
- `codebook/common_without_brackets.xlsx`: source codebook.

### Outputs
- `output/maxqda_codebook.xlsx`: final MAXQDA import file.
- Intermediate files in `codebook/`:
  - `codebook/common.cleaned.xlsx`
  - `codebook/abbrev_from_corpus.csv`
  - `codebook/abbrev_kb.csv`
  - `codebook/common.cleaned.with_kb_abbrev.xlsx`
  - `codebook/context_abbrev_predictions.csv`
  - `codebook/common.cleaned.with_context_abbrev.xlsx`

### Tuning “strictness”
The most important knobs are in step 5 of `run_pipeline.bat`:
- `--min-frequency`: higher means fewer low-signal matches.
- `--max-words`: maximum phrase length used during matching.

---

## For developers

### Pipeline overview
The pipeline is orchestrated by `run_pipeline.bat` and runs these scripts:
1) `preprocess_codebook.py` - normalize codebook + generate variants.
2) `mine_abbreviations_schwartz_hearst.py` - mine abbreviation pairs from PDFs.
3) `abbrev_knowledge_base.py` - aggregate/filter abbreviations + optionally enrich codebook.
4) `context_abbrev_classifier_patched.py` - TF-IDF + cosine similarity context classifier (may add 0 rows).
5) `build_codebook.py` - exact-match against corpus + export via `extract_medical_terms.py`.

### Environment
- `setup_env.bat` creates `libs/venv` and installs dependencies from `requirements.txt`.
- Main dependencies: `pandas`, `openpyxl`, `PyPDF2`, `scikit-learn`, `python-dotenv`.

### Extension points
- Term normalization/variants: `preprocess_codebook.py`.
- Abbreviation mining and filters: `mine_abbreviations_schwartz_hearst.py`, `abbrev_knowledge_base.py`.
- Context classifier: `context_abbrev_classifier_patched.py`.
- Matching + export: `extract_medical_terms.py` (`build_codebook_index`, `run_exact_codebook_match`, `export_to_maxqda_format`).

### Optional LLM modes (not used by default)
`extract_medical_terms.py` also contains `hybrid` and `llm-only` modes via OpenAI, but the default pipeline is designed to be LLM-free.
To enable LLM modes, set `OPENAI_API_KEY` in `.env` and install the `openai` package.

---

## Repository layout
- `primery/`: input corpus (PDF/TXT) - empty in git, add locally
- `codebook/`: source and generated codebooks/CSVs
- `output/`: generated MAXQDA import files
- `libs/venv/`: local virtual environment (created automatically)
- `run_pipeline.bat`: end-to-end pipeline
- `setup_env.bat`: environment bootstrap

---

## Troubleshooting
- `Multiple definitions in dictionary ... /F13`
  - This is a `PyPDF2` warning for some PDFs (fonts/dictionaries). It is usually harmless.
- Output is empty or too small
  - Check `--min-frequency` and `--max-words`, and verify that your PDFs contain extractable text (scanned PDFs may require OCR).

---

## License
MIT License. See `LICENSE`.

## Contributing
See `CONTRIBUTING.md`.

## Code of Conduct
See `CODE_OF_CONDUCT.md`.

## Security
See `SECURITY.md`.

## Versioning and releases
- Semantic Versioning (SemVer): tags like `vX.Y.Z`
- Changelog: `CHANGELOG.md`

