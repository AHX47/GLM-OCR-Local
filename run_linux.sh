#!/usr/bin/env bash
# ─── GLM-OCR Local Runner — Linux/Mac Setup ─────────────────────────────────
set -e

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║   GLM-OCR Local CPU Runner Setup    ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+ first."
    exit 1
fi

# Create venv
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv .venv
fi

echo "[2/4] Activating environment..."
source .venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/4] Done!"
echo ""
echo "─────────────────────────────────────────────"
echo " Launching GUI..."
echo " (Model downloads ~1.8 GB on first run)"
echo "─────────────────────────────────────────────"
echo ""

python glm_ocr_local.py
