#!/bin/bash
# ==============================================================================
# SLURM Job Script: Pi0 Complete Evaluation
# Runs MultipleHooksStudy/notebooks/pi0_complete.ipynb on TC1
#
# NOTE: Before submitting, ensure HUGGING_FACE_HUB_TOKEN is set in your
#       environment or hardcode it below. Required to download Pi0 checkpoints.
#
# Submit with: sbatch pi0_complete_job.sh
# ==============================================================================

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=8
#SBATCH --time=360
#SBATCH --job-name=Pi0_Complete
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/logs/error_%x_%j.err

set -e
echo "[$(date)] ====== Pi0 Complete Job Started ======"

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT=/tc1home/FYP/prithvi004/EmbodiedLLM
MHS_DIR=${PROJECT_ROOT}/MultipleHooksStudy
WORKSPACE=${MHS_DIR}/workspace/pi0
RESULTS_DIR=${MHS_DIR}/results/pi0_complete
NOTEBOOK=${MHS_DIR}/notebooks/pi0_complete.ipynb

OPENPI_DIR=${WORKSPACE}/openpi
VLABENCH_DIR=${WORKSPACE}/VLABench
CKPT_DIR=${WORKSPACE}/ckpt_pi0_vlabench
LOGS_DIR=${WORKSPACE}/logs

mkdir -p "${WORKSPACE}" "${RESULTS_DIR}" "${LOGS_DIR}" "${CKPT_DIR}"

# Export for use in the patch script
export TC1_WORKSPACE="${WORKSPACE}"
export TC1_RESULTS_DIR="${RESULTS_DIR}"
export TC1_OPENPI_DIR="${OPENPI_DIR}"
export TC1_VLABENCH_DIR="${VLABENCH_DIR}"
export TC1_CKPT_DIR="${CKPT_DIR}"
export TC1_LOGS_DIR="${LOGS_DIR}"

# ==============================================================================
# Module Loading
# ==============================================================================
module load anaconda
module load cuda/12.4

# ==============================================================================
# Conda Environment Setup
# ==============================================================================
ENV_NAME=pi0_env

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[$(date)] Using existing conda environment: ${ENV_NAME}"
else
    echo "[$(date)] Creating conda environment: ${ENV_NAME} (Python 3.10)"
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

source /tc1apps/anaconda3/bin/activate "${ENV_NAME}"

# ==============================================================================
# Install Dependencies (idempotent)
# ==============================================================================
echo "[$(date)] Installing core dependencies..."
pip install -q --upgrade pip

# PyTorch with CUDA 12.4
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Notebook tools and standard ML libs
pip install -q einops imageio jupyter nbconvert huggingface_hub

# lerobot from HuggingFace (official Pi0 implementation)
echo "[$(date)] Installing lerobot..."
pip install -q "lerobot@git+https://github.com/huggingface/lerobot.git"

# ==============================================================================
# Clone and Install openpi (Physical Intelligence's Pi0 server)
# ==============================================================================
if [ ! -d "${OPENPI_DIR}" ]; then
    echo "[$(date)] Cloning openpi..."
    git clone https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}"
fi
cd "${OPENPI_DIR}"
pip install -q -e ".[pi0]" 2>/dev/null || pip install -q -e .
cd -

# ==============================================================================
# Clone and Install VLABench
# ==============================================================================
if [ ! -d "${VLABENCH_DIR}" ]; then
    echo "[$(date)] Cloning VLABench..."
    git clone https://github.com/TEA-Lab/VLABench.git "${VLABENCH_DIR}"
    cd "${VLABENCH_DIR}"
    pip install -q -e .
    cd -
fi

# ==============================================================================
# Environment Variables for Evaluation
# ==============================================================================
export MUJOCO_GL=egl
export PYTHONPATH="${OPENPI_DIR}/src:${VLABENCH_DIR}:${PYTHONPATH:-}"
unset DISPLAY

# HuggingFace token — set via environment before job submission OR uncomment:
# export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXX

# ==============================================================================
# Convert Notebook → Python Script
# ==============================================================================
echo "[$(date)] Converting notebook to Python script..."
SCRIPT_RAW=/tmp/pi0_complete_raw.py

jupyter nbconvert --to script "${NOTEBOOK}" --output /tmp/pi0_complete_raw

# ==============================================================================
# Patch Script: Remove Colab-specific code, remap /content/ → TC1 paths
# ==============================================================================
echo "[$(date)] Patching script for TC1 environment..."
SCRIPT_PATCHED=/tmp/pi0_complete_tc1.py

python3 << 'PYEOF'
import re, os, sys

script_raw     = '/tmp/pi0_complete_raw.py'
script_patched = '/tmp/pi0_complete_tc1.py'

workspace    = os.environ['TC1_WORKSPACE']
results_dir  = os.environ['TC1_RESULTS_DIR']
openpi_dir   = os.environ['TC1_OPENPI_DIR']
vlabench_dir = os.environ['TC1_VLABENCH_DIR']
ckpt_dir     = os.environ['TC1_CKPT_DIR']
logs_dir     = os.environ['TC1_LOGS_DIR']

with open(script_raw, 'r') as f:
    code = f.read()

# ---- Remove Google Colab / auth imports ----
remove_patterns = [
    r'from google\.colab import[^\n]*\n',
    r'from google import[^\n]*\n',
    r'import google[^\n]*\n',
    r"get_ipython\(\)\.system\('apt-get[^']*'\)\n",
    r'get_ipython\(\)\.system\("apt-get[^"]*"\)\n',
]
for pat in remove_patterns:
    code = re.sub(pat, '# [TC1-PATCH] Removed\n', code)

# ---- Neutralise Colab calls that remain as function calls ----
code = re.sub(r'drive\.mount\([^\)]*\)', '# [TC1-PATCH] Skipped drive.mount', code)
code = re.sub(r'_colab_auth\.\w+\([^\)]*\)', '# [TC1-PATCH] Skipped Colab auth', code)
code = re.sub(r'_hf_login\([^\)]*\)', '# [TC1-PATCH] Skipped HF login (use HUGGING_FACE_HUB_TOKEN env var)', code)

# ---- Remap path strings (longest/most-specific first) ----
path_map = [
    ("/content/drive/MyDrive/pi0_study/Results/gradient", results_dir + "/gradient"),
    ("/content/drive/MyDrive/pi0_study",                  results_dir),
    ("/content/openpi_vlabench",                          openpi_dir),
    ("/content/openpi",                                   openpi_dir),
    ("/content/VLABench",                                 vlabench_dir),
    ("/content/ckpt_pi0_vlabench",                        ckpt_dir),
    ("/content/mw_eval.py",                               workspace + "/mw_eval.py"),
    ("/content/logs",                                     logs_dir),
    ("/content/",                                         workspace + "/"),
    ("/content",                                          workspace),
]
for old, new in path_map:
    code = code.replace(old, new)

# ---- Header: sys.path + directory guards ----
header = f"""# ============================================================
# TC1 PATCHED SCRIPT — auto-generated from pi0_complete.ipynb
# ============================================================
import sys, os
sys.path.insert(0, '{openpi_dir}/src')
sys.path.insert(0, '{vlabench_dir}')
os.makedirs('{logs_dir}', exist_ok=True)
os.makedirs('{results_dir}', exist_ok=True)
os.makedirs('{results_dir}/gradient', exist_ok=True)
os.makedirs('{ckpt_dir}', exist_ok=True)
"""
code = header + code

with open(script_patched, 'w') as f:
    f.write(code)
print(f"[TC1-PATCH] Written patched script to {script_patched}")
PYEOF

# ==============================================================================
# Run Evaluation
# ==============================================================================
echo "[$(date)] Starting Pi0 Complete evaluation..."
cd "${MHS_DIR}/notebooks"
python3 "${SCRIPT_PATCHED}"

echo "[$(date)] ====== Pi0 Complete Job Finished ======"
