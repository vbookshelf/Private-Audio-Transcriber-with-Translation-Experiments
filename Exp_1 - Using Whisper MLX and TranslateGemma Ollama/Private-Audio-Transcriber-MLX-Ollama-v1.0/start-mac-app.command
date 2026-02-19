#!/bin/bash
echo "============================================"
echo "   Starting Private Audio Transcriber"
echo "============================================"

# Ensure the script runs from its own directory
cd "$(dirname "$0")"

# --- Check for uv ---
if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] 'uv' is not installed."
    echo "See: https://docs.astral.sh/uv/getting-started/installation"
    exit 1
fi

# --- Create venv if missing ---
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python 3.12 environment via uv..."
    # Note: Updated this to 3.12 to match your pyproject.toml
    uv venv --python 3.12 || { echo "[ERROR] uv failed to create venv"; exit 1; }
fi

# --- Activate venv ---
source .venv/bin/activate

# --- Lock & Sync dependencies ---
echo "[INFO] Updating lockfile..."
uv lock || { echo "[ERROR] uv failed to update lockfile"; exit 1; }

echo "[INFO] Syncing dependencies..."
uv sync || { echo "[ERROR] uv failed to sync dependencies"; exit 1; }

# --- Download HuggingFace Model (only if missing) ---
MODEL_DIR="./models/whisper-turbo-mlx"
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "[INFO] Model already exists at $MODEL_DIR - skipping download"
else
    echo "[INFO] Downloading model: mlx-community/whisper-turbo"
    uvx --from huggingface-hub hf download mlx-community/whisper-turbo --local-dir "$MODEL_DIR" || {
        echo "[ERROR] Failed to download model"
        exit 1
    }
fi

# --- Launch app ---
echo "[INFO] Launching app..."
python app.py