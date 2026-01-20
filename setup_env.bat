@echo off
setlocal
set ROOT=%~dp0
set VENV=%ROOT%libs\venv

if not exist "%VENV%\Scripts\python.exe" (
    echo [setup] Creating venv in %VENV%
    py -3 -m venv "%VENV%"
)

call "%VENV%\Scripts\activate.bat"

echo [setup] Installing dependencies...
pip install -r "%ROOT%requirements.txt"

echo [done] Environment ready. Activate manually with:
echo   call "%VENV%\Scripts\activate.bat"
endlocal
