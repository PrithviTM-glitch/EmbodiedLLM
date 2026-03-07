#!/bin/bash
# ==============================================================================
# SLURM Job Script: SmolVLA Complete Evaluation
# Runs MultipleHooksStudy/notebooks/smolvla_complete.ipynb on TC1
#
# Three experiments:
#   1. Baseline benchmarking (LIBERO, MetaWorld, VLABench)
#   2. Ablation study (zero state_proj output)
#   3. Gradient analysis (state encoder gradient flow)
#
# NOTE: First run downloads model checkpoints (~2GB) and fine-tunes models.
#       May approach the 6-hour QoS limit. Consider requesting extended QoS
#       via CCDSgpu-tc@ntu.edu.sg for initial run.
#
# Submit with: sbatch smolvla_complete_job.sh
# ==============================================================================

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=8
#SBATCH --time=360
#SBATCH --job-name=SmolVLA_Complete
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/smolvla/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/smolvla/logs/error_%x_%j.err

set -euo pipefail
echo "[$(date)] ====== SmolVLA Complete Job Started ======"
echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Node: $(hostname)"

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT=/tc1home/FYP/prithvi004/EmbodiedLLM
MHS_DIR=${PROJECT_ROOT}/MultipleHooksStudy
WORKSPACE=${MHS_DIR}/workspace/smolvla
RESULTS_DIR=${MHS_DIR}/results/smolvla_complete
NOTEBOOK=${MHS_DIR}/notebooks/smolvla_complete.ipynb

LEROBOT_DIR=${WORKSPACE}/lerobot
LIBERO_DIR=${WORKSPACE}/LIBERO
METAWORLD_DIR=${WORKSPACE}/metaworld
VLABENCH_DIR=${WORKSPACE}/VLABench
CHECKPOINTS_DIR=${WORKSPACE}/checkpoints
LOGS_DIR=${WORKSPACE}/logs
DATA_DIR=${WORKSPACE}/data

mkdir -p "${WORKSPACE}" "${RESULTS_DIR}" "${CHECKPOINTS_DIR}" "${LOGS_DIR}" "${DATA_DIR}"
mkdir -p "${RESULTS_DIR}/baseline" "${RESULTS_DIR}/ablation" "${RESULTS_DIR}/gradient"
mkdir -p "$(dirname "$0")/logs"   # ensure SLURM log dir exists

# Export for use in the patch script
export TC1_WORKSPACE="${WORKSPACE}"
export TC1_RESULTS_DIR="${RESULTS_DIR}"
export TC1_LEROBOT_DIR="${LEROBOT_DIR}"
export TC1_LIBERO_DIR="${LIBERO_DIR}"
export TC1_METAWORLD_DIR="${METAWORLD_DIR}"
export TC1_VLABENCH_DIR="${VLABENCH_DIR}"
export TC1_CHECKPOINTS_DIR="${CHECKPOINTS_DIR}"
export TC1_LOGS_DIR="${LOGS_DIR}"
export TC1_DATA_DIR="${DATA_DIR}"
export TC1_PROJECT_ROOT="${PROJECT_ROOT}"
export TC1_MHS_DIR="${MHS_DIR}"

# ==============================================================================
# Module Loading
# ==============================================================================
module load anaconda
module load cuda/12.4

# ==============================================================================
# Conda Environment Setup
# ==============================================================================
ENV_NAME=smolvla_env

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
pip install -q einops imageio jupyter nbconvert huggingface_hub safetensors
pip install -q matplotlib seaborn scikit-learn numpy Pillow

# ==============================================================================
# Clone and Install LeRobot with SmolVLA extras
# ==============================================================================
if [ ! -d "${LEROBOT_DIR}" ]; then
    echo "[$(date)] Cloning LeRobot..."
    git clone --depth 1 https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
fi

echo "[$(date)] Installing LeRobot with SmolVLA extras..."
pip install -q -e "${LEROBOT_DIR}[smolvla]" 2>&1 | tail -5

# Additional simulation dependencies
pip install -q gymnasium mujoco 2>&1 | tail -3

# ==============================================================================
# Clone and Install LIBERO
# ==============================================================================
if [ ! -d "${LIBERO_DIR}" ]; then
    echo "[$(date)] Cloning LIBERO..."
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
    pip install -q -e "${LIBERO_DIR}" 2>&1 | tail -5
fi

# Install LIBERO extras for lerobot
pip install -q -e "${LEROBOT_DIR}[libero]" 2>&1 | tail -3 || true

# ==============================================================================
# Clone and Install MetaWorld
# ==============================================================================
if [ ! -d "${METAWORLD_DIR}" ]; then
    echo "[$(date)] Cloning MetaWorld..."
    git clone --depth 1 https://github.com/Farama-Foundation/Metaworld.git "${METAWORLD_DIR}"
    pip install -q -e "${METAWORLD_DIR}" 2>&1 | tail -5
fi

# ==============================================================================
# Clone and Install VLABench
# ==============================================================================
if [ ! -d "${VLABENCH_DIR}" ]; then
    echo "[$(date)] Cloning VLABench..."
    git clone --depth 1 https://github.com/OpenDriveLab/VLABench.git "${VLABENCH_DIR}"
    pip install -q -e "${VLABENCH_DIR}" 2>&1 | tail -5 || true
fi

# ==============================================================================
# Ensure EmbodiedLLM is on correct branch
# ==============================================================================
cd "${PROJECT_ROOT}"
git checkout AnalyseMultipleHooks 2>/dev/null || true
cd -

# ==============================================================================
# Environment Variables for Evaluation
# ==============================================================================
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${MHS_DIR}:${LEROBOT_DIR}:${VLABENCH_DIR}:${PYTHONPATH:-}"
unset DISPLAY

# HuggingFace token — set via environment before job submission OR uncomment:
# export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXX

# ==============================================================================
# Convert Notebook → Python Script
# ==============================================================================
echo "[$(date)] Converting notebook to Python script..."
SCRIPT_RAW=/tmp/smolvla_complete_raw.py

jupyter nbconvert --to script "${NOTEBOOK}" --output /tmp/smolvla_complete_raw

# ==============================================================================
# Patch Script: Remove Colab-specific code, remap /content/ → TC1 paths
# ==============================================================================
echo "[$(date)] Patching script for TC1 environment..."
SCRIPT_PATCHED=/tmp/smolvla_complete_tc1.py

python3 << 'PYEOF'
import re, os, sys

script_raw     = '/tmp/smolvla_complete_raw.py'
script_patched = '/tmp/smolvla_complete_tc1.py'

workspace       = os.environ['TC1_WORKSPACE']
results_dir     = os.environ['TC1_RESULTS_DIR']
lerobot_dir     = os.environ['TC1_LEROBOT_DIR']
libero_dir      = os.environ['TC1_LIBERO_DIR']
metaworld_dir   = os.environ['TC1_METAWORLD_DIR']
vlabench_dir    = os.environ['TC1_VLABENCH_DIR']
checkpoints_dir = os.environ['TC1_CHECKPOINTS_DIR']
logs_dir        = os.environ['TC1_LOGS_DIR']
data_dir        = os.environ['TC1_DATA_DIR']
project_root    = os.environ['TC1_PROJECT_ROOT']
mhs_dir         = os.environ['TC1_MHS_DIR']

with open(script_raw, 'r') as f:
    code = f.read()

# ---- Remove Google Colab / auth imports ----
remove_patterns = [
    r'from google\.colab import[^\n]*\n',
    r'from google import[^\n]*\n',
    r'import google[^\n]*\n',
    # apt-get (no sudo on TC1; system deps already present or loaded by modules)
    r"get_ipython\(\)\.system\('apt-get[^']*'\)\n",
    r'get_ipython\(\)\.system\("apt-get[^"]*"\)\n',
    r"get_ipython\(\)\.system\('apt[^']*'\)\n",
    # IPython magic (%%bash cells become get_ipython().run_cell_magic)
    r"get_ipython\(\)\.run_cell_magic\('bash'[^)]*\)\n",
]
for pat in remove_patterns:
    code = re.sub(pat, '# [TC1-PATCH] Removed\n', code)

# ---- Neutralise Colab calls ----
code = re.sub(r'drive\.mount\([^\)]*\)', '# [TC1-PATCH] Skipped drive.mount', code)

# ---- Remap path strings (longest/most-specific first) ----
path_map = [
    # Drive paths → TC1 results
    ("/content/drive/MyDrive/smolvla_study/Results/baseline",  results_dir + "/baseline"),
    ("/content/drive/MyDrive/smolvla_study/Results/ablation",  results_dir + "/ablation"),
    ("/content/drive/MyDrive/smolvla_study/Results/gradient",  results_dir + "/gradient"),
    ("/content/drive/MyDrive/smolvla_study/Results",           results_dir),
    ("/content/drive/MyDrive/smolvla_study",                   results_dir),
    # Repo clones
    ("/content/lerobot",                                        lerobot_dir),
    ("/content/LIBERO",                                         libero_dir),
    ("/content/metaworld",                                      metaworld_dir),
    ("/content/VLABench",                                       vlabench_dir),
    ("/content/EmbodiedLLM/MultipleHooksStudy",                 mhs_dir),
    ("/content/EmbodiedLLM",                                    project_root),
    # Local paths
    ("/content/checkpoints",                                    checkpoints_dir),
    ("/content/logs",                                           logs_dir),
    ("/content/data",                                           data_dir),
    # Generic fallback (must be last)
    ("/content/",                                               workspace + "/"),
    ("/content",                                                workspace),
]
for old, new in path_map:
    code = code.replace(old, new)

# ---- Replace WORKSPACE string literal pointing to Drive ----
# The notebook sets WORKSPACE = '/content/drive/MyDrive/smolvla_study'
# After path substitution above, it becomes the results_dir. Override to workspace root.
code = code.replace(
    f"WORKSPACE = '{results_dir}'",
    f"WORKSPACE = '{workspace}'"
)

# ---- Header: sys.path + directory guards ----
header = f"""# ============================================================
# TC1 PATCHED SCRIPT — auto-generated from smolvla_complete.ipynb
# ============================================================
import sys, os

# Ensure project modules are importable
sys.path.insert(0, '{mhs_dir}')
sys.path.insert(0, '{lerobot_dir}')
sys.path.insert(0, '{vlabench_dir}')

# Create required directories upfront
for _d in [
    '{workspace}', '{results_dir}',
    '{results_dir}/baseline', '{results_dir}/ablation', '{results_dir}/gradient',
    '{checkpoints_dir}', '{logs_dir}', '{data_dir}',
]:
    os.makedirs(_d, exist_ok=True)

# Environment variables
os.environ['MUJOCO_GL'] = 'egl'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
"""
code = header + code

with open(script_patched, 'w') as f:
    f.write(code)
print(f"[TC1-PATCH] Written patched script to {script_patched}")
print(f"[TC1-PATCH] Script size: {{len(code)}} chars")
PYEOF

# ==============================================================================
# Validate Patched Script (syntax check)
# ==============================================================================
echo "[$(date)] Validating patched script syntax..."
python3 -c "
import py_compile, sys
try:
    py_compile.compile('/tmp/smolvla_complete_tc1.py', doraise=True)
    print('[VALIDATE] Syntax OK')
except py_compile.PyCompileError as e:
    print(f'[VALIDATE] Syntax error: {e}')
    sys.exit(1)
"

# ==============================================================================
# Run Evaluation
# ==============================================================================
echo "[$(date)] Starting SmolVLA Complete evaluation..."
echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

cd "${MHS_DIR}/notebooks"
python3 "${SCRIPT_PATCHED}" 2>&1 | tee "${LOGS_DIR}/smolvla_complete_run.log"
EXIT_CODE=${PIPESTATUS[0]}

# ==============================================================================
# Post-run Summary
# ==============================================================================
echo ""
echo "[$(date)] ====== SmolVLA Complete Job Finished ======"
echo "[$(date)] Exit code: ${EXIT_CODE}"
echo "[$(date)] Results directory: ${RESULTS_DIR}"
echo "[$(date)] Files generated:"
find "${RESULTS_DIR}" -type f -name "*.json" -o -name "*.png" | sort

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[$(date)] WARNING: Script exited with non-zero code ${EXIT_CODE}"
    echo "[$(date)] Check error log for details."
fi

exit ${EXIT_CODE}
