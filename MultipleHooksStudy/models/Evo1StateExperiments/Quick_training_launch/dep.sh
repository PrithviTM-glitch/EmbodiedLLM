#!/usr/bin/env bash
# dep.sh — Evo-1 environment setup (idempotent)
#
# Usage:
#   bash dep.sh [BASE_DIR]
#
set -euo pipefail

# ── Base directory ──────────────────────────────────────────────────────
if [ -n "${1:-}" ]; then
    BASE_DIR="$1"
elif [ -d "/content" ]; then
    BASE_DIR="/content"
else
    BASE_DIR="$HOME"
fi
echo "[dep] BASE_DIR = $BASE_DIR"

EVO_DIR="$BASE_DIR/Evo_1"
DATA_DIR="$BASE_DIR/Evo1_training_dataset"
# ── git-lfs ─────────────────────────────────────────────────────────────
if ! git lfs version &>/dev/null; then
    echo "[dep] Installing git-lfs..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install -y git-lfs
    git lfs install
else
    echo "[dep] git-lfs already installed — skipping"
fi

# ── 1. Clone repo ───────────────────────────────────────────────────────
if [ ! -f "$EVO_DIR/Evo_1/scripts/train.py" ]; then
    echo "[dep] Cloning repo..."
    TMP="$BASE_DIR/EmbodiedLLM"
    rm -rf "$TMP"
    git clone https://github.com/PrithviTM-glitch/EmbodiedLLM.git "$TMP"
    cp -r "$TMP/MultipleHooksStudy/models/Evo1StateExperiments" "$EVO_DIR"
    rm -rf "$TMP"
else
    echo "[dep] Repo already present — skipping clone"
fi

# ── 2. Python dependencies ──────────────────────────────────────────────
echo "[dep] Installing Python dependencies..."
pip install -q -r "$EVO_DIR/Evo_1/requirements.txt"
pip install -q torch torchvision accelerate transformers
pip install -q wandb swanlab einops timm
pip install -q pandas pyarrow av
pip install -q mujoco metaworld websockets opencv-python packaging huggingface_hub
pip install -q deepspeed

# flash-attn only if not already installed
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo "[dep] Installing flash-attn..."
    MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
else
    echo "[dep] flash-attn already installed — skipping"
fi

# ── 3. WandB login ──────────────────────────────────────────────────────
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" --relogin
else
    echo "[dep] WANDB_API_KEY not set — skipping wandb login"
    echo "      Run: wandb login"
fi

# ── 4. MetaWorld dataset ────────────────────────────────────────────────
METAWORLD_DIR="$DATA_DIR/Metaworld"
if [ ! -d "$METAWORLD_DIR/meta" ]; then
    echo "[dep] Downloading MetaWorld dataset..."
    mkdir -p "$DATA_DIR"
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset \
        "$METAWORLD_DIR"
    git -C "$METAWORLD_DIR" lfs pull
    
else
    echo "[dep] MetaWorld already present — skipping"
fi

# ── 5. LIBERO datasets ──────────────────────────────────────────────────
LIBERO_DIR="$DATA_DIR/libero"
mkdir -p "$LIBERO_DIR"

declare -A LIBERO_SUITES=(
    ["libero_spatial"]="libero_spatial_no_noops_1.0.0_lerobot"
    ["libero_object"]="libero_object_no_noops_1.0.0_lerobot"
    ["libero_goal"]="libero_goal_no_noops_1.0.0_lerobot"
    ["libero_10"]="libero_10_no_noops_1.0.0_lerobot"
)

for local_name in "${!LIBERO_SUITES[@]}"; do
    hf_name="${LIBERO_SUITES[$local_name]}"
    dest="$LIBERO_DIR/$local_name"
    if [ ! -d "$dest/meta" ]; then
        echo "[dep] Downloading LIBERO suite: $local_name..."
        GIT_LFS_SKIP_SMUDGE=1 git clone \
            "https://huggingface.co/datasets/IPEC-COMMUNITY/${hf_name}" \
            "$dest"
        git -C "$dest" lfs pull
    else
        echo "[dep] LIBERO $local_name already present — skipping"
    fi
done

# ── 6. Patch config yaml paths ──────────────────────────────────────────
echo "[dep] Patching dataset config paths..."

METAWORLD_CFG="$EVO_DIR/Evo_1/dataset/metaworld_config.yaml"
LIBERO_CFG="$EVO_DIR/Evo_1/dataset/libero_config.yaml"

# Replace any hardcoded base path with the actual DATA_DIR
sed -i "s|/content/Evo1_training_dataset|${DATA_DIR}|g" "$METAWORLD_CFG"
sed -i "s|\$HOME/Evo1_training_dataset|${DATA_DIR}|g"   "$METAWORLD_CFG"

sed -i "s|/content/Evo1_training_dataset|${DATA_DIR}|g" "$LIBERO_CFG"
sed -i "s|\$HOME/Evo1_training_dataset|${DATA_DIR}|g"   "$LIBERO_CFG"

# Patch long HuggingFace names → short local names in libero config
sed -i \
    -e 's|libero_spatial_no_noops_1\.0\.0_lerobot|libero_spatial|g' \
    -e 's|libero_object_no_noops_1\.0\.0_lerobot|libero_object|g'   \
    -e 's|libero_goal_no_noops_1\.0\.0_lerobot|libero_goal|g'       \
    -e 's|libero_10_no_noops_1\.0\.0_lerobot|libero_10|g'           \
    "$LIBERO_CFG"

echo ""
echo "[dep] Setup complete."
echo "  Evo_1   : $EVO_DIR"
echo "  Datasets: $DATA_DIR"
echo ""
echo "  Verify:"
echo "    grep 'path:' $METAWORLD_CFG"
echo "    grep 'path:' $LIBERO_CFG"