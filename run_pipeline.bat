@echo off
setlocal
set ROOT=%~dp0
set VENV=%ROOT%libs\venv

if not exist "%VENV%\Scripts\python.exe" (
    echo [bootstrap] Creating venv...
    py -3 -m venv "%VENV%"
)
call "%VENV%\Scripts\activate.bat"

echo [bootstrap] Installing deps...
pip install -r "%ROOT%requirements.txt"

pushd "%ROOT%"

REM 1) Clean base codebook
python preprocess_codebook.py

REM 2) Extract abbreviations from PDF corpus
python mine_abbreviations_schwartz_hearst.py --pdf-dir primery --out codebook/abbrev_from_corpus.csv --min-count 1

REM 3) Build KB and enrich codebook
python abbrev_knowledge_base.py --codebook-in codebook/common.cleaned.xlsx --pairs-csv codebook/abbrev_from_corpus.csv --codebook-out codebook/common.cleaned.with_kb_abbrev.xlsx --kb-out codebook/abbrev_kb.csv --min-total 3 --min-share 0.60

REM 4) Context-based abbr classification (may add 0 rows if no confident matches)
python context_abbrev_classifier_patched.py --pdf-dir primery --codebook-in codebook/common.cleaned.with_kb_abbrev.xlsx --codebook-out codebook/common.cleaned.with_context_abbrev.xlsx --predictions-csv codebook/context_abbrev_predictions.csv --chem-prefix "Chemicals and Drugs\\" --min-abbr-count 5 --min-confidence 0.75 --min-score 0.06 --max-pdf 15

REM 5) Build final MAXQDA dictionary
python build_codebook.py --codebook-xlsx codebook/common.cleaned.with_context_abbrev.xlsx --input primery --output output/maxqda_codebook.xlsx --max-words 4 --min-frequency 2

popd

echo [done] Finished pipeline. Output: output\maxqda_codebook.xlsx
endlocal
