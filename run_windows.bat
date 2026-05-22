@echo off
REM ─── GLM-OCR Local Runner — Windows Setup ───────────────────────────────────
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   GLM-OCR Local CPU Runner Setup    ║
echo  ╚══════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Create venv if not exists
if not exist ".venv" (
    echo [1/4] Creating virtual environment...
    python -m venv .venv
)

echo [2/4] Activating environment...
call .venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo [4/4] Done!
echo.
echo ─────────────────────────────────────────────
echo  Launching GUI...
echo  (Model will download ~1.8GB on first run)
echo ─────────────────────────────────────────────
echo.
python glm_ocr_local.py

pause
