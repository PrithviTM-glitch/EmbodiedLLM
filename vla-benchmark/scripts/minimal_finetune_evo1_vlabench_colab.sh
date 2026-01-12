#!/usr/bin/env bash
set -euo pipefail

# Minimal end-to-end runner for fine-tuning Evo-1 on VLABench in Google Colab.
#
# What it does:
# 1) Clones VLABench and Evo-1 into this repo's recommended folders
# 2) Installs minimal Python deps (incl. Colab-compatible Open3D)
# 3) Patches VLABench's converter to tolerate LeRobot internal layout changes
# 4) Calls this repo's wrapper: vla-benchmark/scripts/finetune_evo1_vlabench.py

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VLABENCH_DIR="$REPO_ROOT/vla-benchmark/benchmark/VLABench"
EVO1_DIR="$REPO_ROOT/vla-benchmark/models/Evo-1"

mkdir -p "$(dirname "$VLABENCH_DIR")" "$(dirname "$EVO1_DIR")"

if [[ ! -d "$VLABENCH_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/OpenMOSS/VLABench.git "$VLABENCH_DIR"
fi

if [[ ! -d "$EVO1_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/MINT-SJTU/Evo-1.git "$EVO1_DIR"
fi

# System deps (Colab usually has these; safe to re-run)
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y libgl1 libglib2.0-0 libegl1
fi

python3 -m pip install -U pip

# Colab wheel compatibility pins
python3 -m pip install "numpy<2.0" "opencv-python-headless<4.12" "open3d==0.19.0"

# LeRobot is needed for VLABench conversion and Evo-1 dataset loading.
python3 -m pip install lerobot

# Install VLABench + Evo-1 python deps
python3 -m pip install -r "$VLABENCH_DIR/requirements.txt"
python3 -m pip install -r "$EVO1_DIR/Evo_1/requirements.txt"

# Download VLABench assets (required before trajectory generation)
( cd "$VLABENCH_DIR" && python3 scripts/download_assets.py )

# Patch VLABench converter to support both LeRobot layouts:
# - older: lerobot.common.datasets.lerobot_dataset
# - newer: lerobot.datasets.lerobot_dataset
python3 - <<'PY'
from __future__ import annotations

from pathlib import Path

converter = Path("vla-benchmark/benchmark/VLABench/scripts/convert_to_lerobot.py")
text = converter.read_text()

needle_1 = "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset"
needle_2 = "from lerobot.datasets.lerobot_dataset import LeRobotDataset"

if "try:\n    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset" in text:
    print(f"[ok] Converter already patched: {converter}")
elif needle_1 in text and needle_2 not in text:
    patched = text.replace(
        needle_1,
        "try:\n    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset\nexcept ModuleNotFoundError:\n    from lerobot.datasets.lerobot_dataset import LeRobotDataset",
    )
    converter.write_text(patched)
    print(f"[patched] Updated LeRobot import in: {converter}")
else:
    print(f"[warn] Unexpected converter import state; no changes made: {converter}")
PY

# Headless MuJoCo rendering on Colab
export MUJOCO_GL=${MUJOCO_GL:-egl}

# Where the LeRobot dataset will be written (HF_HOME/lerobot/<dataset-name>)
export HF_HOME=${HF_HOME:-"$REPO_ROOT/.hf"}

# Edit these two lines for your run
TASKS=(select_toy)
DATASET_NAME=vlabench_minimal_demo

python3 vla-benchmark/scripts/finetune_evo1_vlabench.py \
  --vlabench-root "$VLABENCH_DIR" \
  --evo1-root "$EVO1_DIR" \
  --tasks "${TASKS[@]}" \
  --n-samples-per-task 2 \
  --dataset-name "$DATASET_NAME" \
  --run collect convert stage1

# To also run stage2, add it to --run above ("stage2") once stage1 finishes.
