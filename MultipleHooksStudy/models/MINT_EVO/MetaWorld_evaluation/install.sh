#!/usr/bin/env bash
# install.sh — Eval-only dependency setup for MINT_EVO baseline MetaWorld evaluation.
# Targets Colab (Ubuntu, single GPU). Does NOT install training deps or download datasets.
#
# Usage:
#   bash install.sh
#
set -euo pipefail

echo "=================================================="
echo " MINT_EVO MetaWorld eval dependency installer"
echo "=================================================="

# ── 1. System packages (osmesa for headless MuJoCo rendering) ─────────────────
echo ""
echo "[1/5] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    ffmpeg

# ── 2. Core PyTorch stack ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Installing PyTorch stack..."
pip install -q \
    torch==2.5.1 \
    torchvision==0.20.1 \
    accelerate \
    safetensors

# ── 3. Model dependencies ──────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing model dependencies..."
pip install -q \
    transformers==4.39.0 \
    timm \
    einops \
    diffusers \
    Pillow \
    opencv-python

# ── 4. Simulation + WebSocket (client side) ───────────────────────────────────
echo ""
echo "[4/5] Installing simulation and client dependencies..."
pip install -q \
    mujoco \
    metaworld \
    websockets \
    packaging \
    gymnasium

# ── 5. flash-attn (skip if already present) ───────────────────────────────────
echo ""
echo "[5/5] Checking flash-attn..."
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "  flash-attn already installed — skipping"
else
    echo "  Installing flash-attn (this takes a few minutes)..."
    MAX_JOBS=4 pip install -q flash-attn --no-build-isolation
fi

# ── GCS / gcloud check ────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Dependency installation complete."
echo "=================================================="
echo ""
echo " NOTE: Checkpoint pulling requires gcloud authentication."
echo " If running on Colab, authenticate with:"
echo "   from google.colab import auth; auth.authenticate_user()"
echo " Or via service account:"
echo "   gcloud auth activate-service-account --key-file=KEY.json"
echo ""
echo " Verify setup:"
echo "   python -c \"import metaworld, torch, safetensors, websockets; print('OK')\""
