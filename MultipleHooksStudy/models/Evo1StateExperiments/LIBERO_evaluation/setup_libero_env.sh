#!/usr/bin/env bash
# setup_libero_env.sh — One-shot LIBERO conda environment setup for Colab/GCP.
#
# Usage:
#   bash setup_libero_env.sh            # install everything
#   bash setup_libero_env.sh --skip-clone  # skip git clone if LIBERO/ already exists
#
# After this script completes, run the eval launcher with --colab:
#   python Quick_training_launch/LIBERO/launch_eval.py --exp exp1 --step best --colab --no-video
#
set -euo pipefail

SKIP_CLONE=false
for arg in "$@"; do
  [[ "$arg" == "--skip-clone" ]] && SKIP_CLONE=true
done

# ── 1. Install Miniconda if not present ───────────────────────────────────────
CONDA_DIR="/root/miniconda3"
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
  echo "[setup] Installing Miniconda..."
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm /tmp/miniconda.sh
  echo "[setup] Miniconda installed at $CONDA_DIR"
else
  echo "[setup] Miniconda already present at $CONDA_DIR — skipping install"
fi

export PATH="$CONDA_DIR/bin:$PATH"

# ── 2. Create conda env (skip if already exists) ──────────────────────────────
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

if conda env list | grep -q "^libero "; then
  echo "[setup] conda env 'libero' already exists — skipping creation"
else
  echo "[setup] Creating conda env: libero (python=3.8.13)"
  conda create -n libero python=3.8.13 -y
fi

LIBERO_PYTHON="$CONDA_DIR/envs/libero/bin/python"
LIBERO_PIP="$CONDA_DIR/envs/libero/bin/pip"

# ── 3. Clone LIBERO repo ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBERO_REPO="$SCRIPT_DIR/LIBERO"

if [ "$SKIP_CLONE" = true ]; then
  echo "[setup] --skip-clone set — skipping git clone"
elif [ -d "$LIBERO_REPO/.git" ]; then
  echo "[setup] LIBERO repo already cloned at $LIBERO_REPO — skipping"
else
  echo "[setup] Cloning LIBERO..."
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_REPO"
fi

# ── 4. Install LIBERO dependencies ────────────────────────────────────────────
echo "[setup] Installing PyTorch (cu113)..."
"$LIBERO_PIP" install --quiet \
  torch==1.11.0+cu113 \
  torchvision==0.12.0+cu113 \
  torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113

echo "[setup] Installing LIBERO requirements..."
"$LIBERO_PIP" install --quiet -r "$LIBERO_REPO/requirements.txt"

echo "[setup] Installing LIBERO package..."
"$LIBERO_PIP" install --quiet -e "$LIBERO_REPO"

echo "[setup] Installing eval dependencies..."
"$LIBERO_PIP" install --quiet websockets imageio huggingface_hub

# ── 5. Verify ─────────────────────────────────────────────────────────────────
echo "[setup] Verifying installation..."
"$LIBERO_PYTHON" -c "import libero; import websockets; import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}')"

echo ""
echo "============================================================"
echo "  LIBERO env ready."
echo "  Python: $LIBERO_PYTHON"
echo ""
echo "  Run evaluation with:"
echo "    python Quick_training_launch/LIBERO/launch_eval.py \\"
echo "      --exp exp1 --step best --colab --no-video"
echo "============================================================"
