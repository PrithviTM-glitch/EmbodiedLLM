#!/bin/bash
# ==============================================================================
# SLURM Job Script: Pi0 Complete Evaluation
# Runs MultipleHooksStudy/notebooks/pi0_complete.ipynb on TC1
#
# Three experiments:
#   1. Baseline benchmarking (LIBERO, MetaWorld, VLABench)
#   2. Ablation study (zero state_proj output)
#   3. Gradient analysis (state encoder gradient flow)
#
# NOTE: Before submitting, ensure HUGGING_FACE_HUB_TOKEN is set in your
#       environment or hardcode it below. Required to download Pi0 checkpoints.
#
# Pi0 uses JAX-based openpi for inference. Dependencies are heavy (~30 min first
# install). The script manages two openpi forks:
#   - Physical-Intelligence/openpi:   official LIBERO/MetaWorld eval
#   - Shiduo-zh/openpi:              VLABench checkpoint
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
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/pi0/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/pi0/logs/error_%x_%j.err

set -euo pipefail
echo "[$(date)] ====== Pi0 Complete Job Started ======"
echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Node: $(hostname)"

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT=/tc1home/FYP/prithvi004/EmbodiedLLM
MHS_DIR=${PROJECT_ROOT}/MultipleHooksStudy
WORKSPACE=${MHS_DIR}/workspace/pi0
RESULTS_DIR=${MHS_DIR}/results/pi0_complete
NOTEBOOK=${MHS_DIR}/notebooks/pi0_complete.ipynb

OPENPI_DIR=${WORKSPACE}/openpi
OPENPI_VLA_DIR=${WORKSPACE}/openpi_vlabench
VLABENCH_DIR=${WORKSPACE}/VLABench
CKPT_DIR=${WORKSPACE}/ckpt_pi0_vlabench
LOGS_DIR=${WORKSPACE}/logs

mkdir -p "${WORKSPACE}" "${RESULTS_DIR}" "${LOGS_DIR}" "${CKPT_DIR}"
mkdir -p "${RESULTS_DIR}/baseline" "${RESULTS_DIR}/ablation" "${RESULTS_DIR}/gradient"
mkdir -p "$(dirname "$0")/logs"   # ensure SLURM log dir exists

# Export for use in the patch script
export TC1_WORKSPACE="${WORKSPACE}"
export TC1_RESULTS_DIR="${RESULTS_DIR}"
export TC1_OPENPI_DIR="${OPENPI_DIR}"
export TC1_OPENPI_VLA_DIR="${OPENPI_VLA_DIR}"
export TC1_VLABENCH_DIR="${VLABENCH_DIR}"
export TC1_CKPT_DIR="${CKPT_DIR}"
export TC1_LOGS_DIR="${LOGS_DIR}"
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
pip install -q einops imageio jupyter nbconvert huggingface_hub safetensors
pip install -q matplotlib seaborn scikit-learn numpy Pillow
pip install -q websockets msgpack uv

# lerobot with Pi0 + MetaWorld extras
echo "[$(date)] Installing lerobot..."
pip install -q "lerobot[metaworld,pi]@git+https://github.com/huggingface/lerobot.git" 2>&1 | tail -5
pip install -q "transformers@git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" 2>&1 | tail -5

# ==============================================================================
# Clone and Install openpi (Physical Intelligence's Pi0 server)
# ==============================================================================
if [ ! -d "${OPENPI_DIR}" ]; then
    echo "[$(date)] Cloning openpi..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "${OPENPI_DIR}"
fi
cd "${OPENPI_DIR}"
pip install -q -e ".[pi0]" 2>/dev/null || pip install -q -e . || true
cd -

# Clone VLABench openpi fork
if [ ! -d "${OPENPI_VLA_DIR}" ]; then
    echo "[$(date)] Cloning openpi VLABench fork..."
    git clone https://github.com/Shiduo-zh/openpi.git "${OPENPI_VLA_DIR}" || true
fi

# ==============================================================================
# Clone and Install VLABench
# ==============================================================================
if [ ! -d "${VLABENCH_DIR}" ]; then
    echo "[$(date)] Cloning VLABench..."
    git clone https://github.com/OpenMOSS/VLABench.git "${VLABENCH_DIR}"
    cd "${VLABENCH_DIR}"
    pip install -q -e . || true
    cd -
fi

# ==============================================================================
# Environment Variables for Evaluation
# ==============================================================================
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH="${OPENPI_DIR}/src:${VLABENCH_DIR}:${MHS_DIR}:${PYTHONPATH:-}"
unset DISPLAY

# HuggingFace token — set via environment before job submission OR uncomment:
# export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXX
# export HF_TOKEN=${HUGGING_FACE_HUB_TOKEN}

# Guard: fail early if no HF token is available (needed for Pi0 checkpoints + VLABench)
if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: No HuggingFace token found."
    echo "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before submitting this job."
    echo "  export HF_TOKEN=hf_XXXXXXXXXXXXXXXX"
    echo "  sbatch pi0_complete_job.sh"
    exit 1
fi
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

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

workspace      = os.environ['TC1_WORKSPACE']
results_dir    = os.environ['TC1_RESULTS_DIR']
openpi_dir     = os.environ['TC1_OPENPI_DIR']
openpi_vla_dir = os.environ['TC1_OPENPI_VLA_DIR']
vlabench_dir   = os.environ['TC1_VLABENCH_DIR']
ckpt_dir       = os.environ['TC1_CKPT_DIR']
logs_dir       = os.environ['TC1_LOGS_DIR']
project_root   = os.environ['TC1_PROJECT_ROOT']
mhs_dir        = os.environ['TC1_MHS_DIR']

with open(script_raw, 'r') as f:
    code = f.read()

# ---- Remove Google Colab / auth imports ----
remove_patterns = [
    r'from google\.colab import[^\n]*\n',
    r'from google import[^\n]*\n',
    r'import google[^\n]*\n',
    r"get_ipython\(\)\.system\('apt-get[^']*'\)\n",
    r'get_ipython\(\)\.system\("apt-get[^"]*"\)\n',
    r"get_ipython\(\)\.run_cell_magic\('bash'[^)]*\)\n",
]
for pat in remove_patterns:
    code = re.sub(pat, '# [TC1-PATCH] Removed\n', code)

# ---- Neutralise Colab calls ----
code = re.sub(r'drive\.mount\([^\)]*\)', '# [TC1-PATCH] Skipped drive.mount', code)
code = re.sub(r'_colab_auth\.\w+\([^\)]*\)', '# [TC1-PATCH] Skipped Colab auth', code)
code = re.sub(r'_hf_login\([^\)]*\)', '# [TC1-PATCH] Skipped HF login', code)

# ---- Remap path strings (longest/most-specific first) ----
path_map = [
    ("/content/drive/MyDrive/pi0_study/Results/gradient", results_dir + "/gradient"),
    ("/content/drive/MyDrive/pi0_study/Results/baseline",  results_dir + "/baseline"),
    ("/content/drive/MyDrive/pi0_study/Results/ablation",  results_dir + "/ablation"),
    ("/content/drive/MyDrive/pi0_study/Results",           results_dir),
    ("/content/drive/MyDrive/pi0_study",                   results_dir),
    ("/content/openpi_vlabench",                           openpi_vla_dir),
    ("/content/openpi",                                    openpi_dir),
    ("/content/VLABench",                                  vlabench_dir),
    ("/content/ckpt_pi0_vlabench",                         ckpt_dir),
    ("/content/EmbodiedLLM/MultipleHooksStudy",            mhs_dir),
    ("/content/EmbodiedLLM",                               project_root),
    ("/content/mw_eval.py",                                workspace + "/mw_eval.py"),
    ("/content/logs",                                      logs_dir),
    ("/content/",                                          workspace + "/"),
    ("/content",                                           workspace),
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
sys.path.insert(0, '{mhs_dir}')
os.makedirs('{logs_dir}', exist_ok=True)
os.makedirs('{results_dir}', exist_ok=True)
os.makedirs('{results_dir}/baseline', exist_ok=True)
os.makedirs('{results_dir}/ablation', exist_ok=True)
os.makedirs('{results_dir}/gradient', exist_ok=True)
os.makedirs('{ckpt_dir}', exist_ok=True)
os.environ['MUJOCO_GL'] = 'egl'
"""
code = header + code

with open(script_patched, 'w') as f:
    f.write(code)
print(f"[TC1-PATCH] Written patched script to {script_patched}")
PYEOF

# ==============================================================================
# Validate Patched Script (syntax check)
# ==============================================================================
echo "[$(date)] Validating patched script syntax..."
python3 -c "
import py_compile, sys
try:
    py_compile.compile('/tmp/pi0_complete_tc1.py', doraise=True)
    print('[VALIDATE] Syntax OK')
except py_compile.PyCompileError as e:
    print(f'[VALIDATE] Syntax error: {e}')
    sys.exit(1)
"

# ==============================================================================
# Run Evaluation
# ==============================================================================
echo "[$(date)] Starting Pi0 Complete evaluation..."
echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

cd "${MHS_DIR}/notebooks"
python3 "${SCRIPT_PATCHED}" 2>&1 | tee "${LOGS_DIR}/pi0_complete_run.log"
EXIT_CODE=${PIPESTATUS[0]}

# ==============================================================================
# Post-run Summary
# ==============================================================================
echo ""
echo "[$(date)] ====== Pi0 Complete Job Finished ======"
echo "[$(date)] Exit code: ${EXIT_CODE}"
echo "[$(date)] Results directory: ${RESULTS_DIR}"
echo "[$(date)] Files generated:"
find "${RESULTS_DIR}" -type f -name "*.json" -o -name "*.png" 2>/dev/null | sort

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[$(date)] WARNING: Script exited with non-zero code ${EXIT_CODE}"
    echo "[$(date)] Check error log for details."
fi

exit ${EXIT_CODE}
