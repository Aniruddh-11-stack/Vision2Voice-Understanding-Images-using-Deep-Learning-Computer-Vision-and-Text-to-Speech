#!/usr/bin/env bash
# =============================================================
# Vision2Voice — Linux / macOS Startup Script
# =============================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "  Vision2Voice AI — Startup Script"
echo "========================================"

# --- Python check ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 is not installed. Please install Python 3.9+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO]  Python version: $PYTHON_VERSION"

# --- Virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[INFO]  Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "[INFO]  Virtual environment activated."

# --- Dependencies ---
echo "[INFO]  Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# --- Model weight check ---
MODELS_DIR="$REPO_ROOT/models"
MODEL_FILE="$MODELS_DIR/modelConcat_1_89.h5"
TOKENIZER_FILE="$MODELS_DIR/caption_train_tokenizer.pkl"

if [ ! -f "$MODEL_FILE" ] || [ ! -f "$TOKENIZER_FILE" ]; then
    echo ""
    echo "⚠️  WARNING: Model weights not found in models/"
    echo "   Expected:"
    echo "     - models/modelConcat_1_89.h5"
    echo "     - models/caption_train_tokenizer.pkl"
    echo "   The app will start but inference will be disabled."
    echo ""
fi

# --- Launch ---
echo "[INFO]  Launching Vision2Voice dashboard..."
PYTHONPATH="$REPO_ROOT/src" streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=localhost
