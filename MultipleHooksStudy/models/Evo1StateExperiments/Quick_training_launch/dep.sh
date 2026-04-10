#!/usr/bin/env bash
# dep.sh — Evo-1 environment setup (Colab, GCP, or any Linux host)
#
# Usage:
#   bash dep.sh [BASE_DIR]
#
# BASE_DIR defaults to /content on Colab, $HOME elsewhere.
# Override explicitly:
#   bash dep.sh /workspace
#
set -euo pipefail

# ── Base directory ──────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    BASE_DIR="$1"
elif [ -d "/content" ]; then
    BASE_DIR="/content"        # Google Colab
else
    BASE_DIR="$HOME"           # GCP, local, etc.
fi
echo "[dep] BASE_DIR = $BASE_DIR"

EVO_DIR="$BASE_DIR/Evo_1"
DATA_DIR="$BASE_DIR/Evo1_training_dataset"

# ── 1. Clone repo and set up Evo_1 ─────────────────────────────────────
rm -rf "$BASE_DIR/EmbodiedLLM" "$EVO_DIR"
git clone https://github.com/PrithviTM-glitch/EmbodiedLLM.git "$BASE_DIR/EmbodiedLLM"
cp -r "$BASE_DIR/EmbodiedLLM/MultipleHooksStudy/models/Evo1StateExperiments" "$EVO_DIR"
rm -rf "$BASE_DIR/EmbodiedLLM"

# ── 2. Install Python dependencies ─────────────────────────────────────
pip install -r "$EVO_DIR/Evo_1/requirements.txt"
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation

pip install torch torchvision accelerate transformers
pip install wandb swanlab einops timm
pip install pandas pyarrow av
pip install mujoco metaworld websockets opencv-python packaging huggingface_hub

# ── 3. WandB login ──────────────────────────────────────────────────────
# Set WANDB_API_KEY in your environment to skip the interactive prompt:
export WANDB_API_KEY='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'
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[dep] WANDB_API_KEY not set — running interactive wandb login"
    wandb login
else
    wandb login "$WANDB_API_KEY"
fi

# ── 4. MetaWorld dataset ────────────────────────────────────────────────
mkdir -p "$DATA_DIR"
GIT_LFS_SKIP_SMUDGE=1 git clone \
    https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset \
    "$DATA_DIR/Metaworld"
git -C "$DATA_DIR/Metaworld" lfs pull

# Fix case mismatch in config (MetaWorld → Metaworld)
sed -i "s|${DATA_DIR}/MetaWorld|${DATA_DIR}/Metaworld|g" \
    "$EVO_DIR/Evo_1/dataset/metaworld_config.yaml"

# ── 5. LIBERO datasets ──────────────────────────────────────────────────
LIBERO_DIR="$DATA_DIR/libero"
mkdir -p "$LIBERO_DIR"

for suite in \
    "libero_spatial_no_noops_1.0.0_lerobot libero_spatial" \
    "libero_object_no_noops_1.0.0_lerobot libero_object" \
    "libero_goal_no_noops_1.0.0_lerobot   libero_goal"   \
    "libero_10_no_noops_1.0.0_lerobot     libero_10"
do
    hf_name="${suite%% *}"
    local_name="${suite##* }"
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        "https://huggingface.co/datasets/IPEC-COMMUNITY/${hf_name}" \
        "$LIBERO_DIR/$local_name"
    git -C "$LIBERO_DIR/$local_name" lfs pull
done

# Patch libero_config.yaml with the actual BASE_DIR and short folder names
LIBERO_CFG="$EVO_DIR/Evo_1/dataset/libero_config.yaml"
sed -i \
    -e "s|/content/Evo1_training_dataset|${DATA_DIR}|g" \
    -e 's|libero_spatial_no_noops_1\.0\.0_lerobot|libero_spatial|g' \
    -e 's|libero_object_no_noops_1\.0\.0_lerobot|libero_object|g'   \
    -e 's|libero_goal_no_noops_1\.0\.0_lerobot|libero_goal|g'       \
    -e 's|libero_10_no_noops_1\.0\.0_lerobot|libero_10|g'           \
    "$LIBERO_CFG"

# Also patch metaworld_config.yaml base path if not Colab
if [ "$BASE_DIR" != "/content" ]; then
    sed -i "s|/content/Evo1_training_dataset|${DATA_DIR}|g" \
        "$EVO_DIR/Evo_1/dataset/metaworld_config.yaml"
fi

echo ""
echo "[dep] Setup complete."
echo "  Evo_1   : $EVO_DIR"
echo "  Datasets: $DATA_DIR"
