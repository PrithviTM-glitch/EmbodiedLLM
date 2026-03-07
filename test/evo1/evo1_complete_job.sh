#!/bin/bash
# ==============================================================================
# SLURM Job Script: Evo-1 Complete Evaluation
# Runs MultipleHooksStudy/notebooks/evo1_complete.ipynb on TC1
#
# Three experiments:
#   1. Baseline benchmarking (LIBERO, MetaWorld)
#   2. Ablation study (zero state_encoder → normalize_state returns zeros)
#   3. Gradient analysis (state encoder gradient flow via flow matching loss)
#
# ARCHITECTURE NOTE:
#   Evo-1 uses a WebSocket server-client architecture:
#   - Server (evo1_server env): runs model inference (PyTorch 2.5.1+cu121)
#   - LIBERO client (libero_client env): runs LIBERO sim (PyTorch 1.11+cu113)
#   - MetaWorld client (metaworld_client env): runs MetaWorld sim
#   This requires 3 separate conda environments.
#
# IMPORTANT: First run creates 3 conda envs + downloads checkpoints.
#   May exceed 6-hour QoS. Request extended QoS via CCDSgpu-tc@ntu.edu.sg.
#
# Submit with: sbatch evo1_complete_job.sh
# ==============================================================================

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=8
#SBATCH --time=360
#SBATCH --job-name=Evo1_Complete
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/evo1/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/evo1/logs/error_%x_%j.err

set -euo pipefail
echo "[$(date)] ====== Evo-1 Complete Job Started ======"
echo "[$(date)] Job ID: ${SLURM_JOB_ID:-local}"
echo "[$(date)] Node: $(hostname)"

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT=/tc1home/FYP/prithvi004/EmbodiedLLM
MHS_DIR=${PROJECT_ROOT}/MultipleHooksStudy
WORKSPACE=${MHS_DIR}/workspace/evo1
RESULTS_DIR=${MHS_DIR}/results/evo1_complete
NOTEBOOK=${MHS_DIR}/notebooks/evo1_complete.ipynb

EVO1_DIR=${WORKSPACE}/Evo-1
LIBERO_DIR=${WORKSPACE}/LIBERO
CKPT_DIR=${WORKSPACE}/checkpoints
LOGS_DIR=${WORKSPACE}/logs
DATA_DIR=${WORKSPACE}/data

mkdir -p "${WORKSPACE}" "${RESULTS_DIR}" "${CKPT_DIR}" "${LOGS_DIR}" "${DATA_DIR}"
mkdir -p "${RESULTS_DIR}/baseline" "${RESULTS_DIR}/ablation" "${RESULTS_DIR}/gradient"
mkdir -p "$(dirname "$0")/logs"   # ensure SLURM log dir exists

# Export for use in the patch script
export TC1_WORKSPACE="${WORKSPACE}"
export TC1_RESULTS_DIR="${RESULTS_DIR}"
export TC1_EVO1_DIR="${EVO1_DIR}"
export TC1_LIBERO_DIR="${LIBERO_DIR}"
export TC1_CKPT_DIR="${CKPT_DIR}"
export TC1_LOGS_DIR="${LOGS_DIR}"
export TC1_DATA_DIR="${DATA_DIR}"
export TC1_PROJECT_ROOT="${PROJECT_ROOT}"
export TC1_MHS_DIR="${MHS_DIR}"

# ==============================================================================
# Module Loading
# ==============================================================================
module load anaconda
module load cuda/12.1

# Ensure conda is on PATH for subprocess calls inside the notebook
export PATH=/tc1apps/anaconda3/bin:${PATH}

# ==============================================================================
# Environment 1: evo1_server (Python 3.10 + PyTorch 2.5.1+cu121)
# Runs model inference via WebSocket server.
# ==============================================================================
if ! conda env list | grep -q "^evo1_server "; then
    echo "[$(date)] Creating conda env: evo1_server (Python 3.10)..."
    conda create -n evo1_server python=3.10 -y

    echo "[$(date)] Installing PyTorch 2.5.1+cu121 in evo1_server..."
    conda run -n evo1_server pip install -q \
        torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

    echo "[$(date)] Installing evo1_server dependencies..."
    conda run -n evo1_server pip install -q \
        'numpy>=1.26.4,<2.0' 'transformers==4.57.6' accelerate diffusers \
        einops timm pillow opencv-python-headless \
        websockets pyyaml huggingface_hub tqdm h5py safetensors \
        matplotlib seaborn scikit-learn jupyter nbconvert

    # Clone Evo-1 repo
    if [ ! -d "${EVO1_DIR}" ]; then
        echo "[$(date)] Cloning Evo-1..."
        git clone https://github.com/MINT-SJTU/Evo-1.git "${EVO1_DIR}"
    fi

    echo "[$(date)] Installing Evo-1 requirements in evo1_server..."
    conda run -n evo1_server pip install -q \
        -r "${EVO1_DIR}/Evo_1/requirements.txt" 2>&1 | tail -5 || true

    # flash-attn (optional — compilation takes 10-15 min)
    echo "[$(date)] Attempting flash-attn install (optional, ~10 min compile)..."
    conda run -n evo1_server pip install -q flash-attn --no-build-isolation 2>/dev/null \
        || echo "[$(date)] flash-attn skipped (build failed — optional)"

    echo "[$(date)] Installing Evo-1 package..."
    conda run -n evo1_server pip install -q -e "${EVO1_DIR}" 2>&1 | tail -5 || true
else
    echo "[$(date)] evo1_server already exists, skipping creation."
    # Still ensure Evo-1 is cloned
    if [ ! -d "${EVO1_DIR}" ]; then
        echo "[$(date)] Cloning Evo-1..."
        git clone https://github.com/MINT-SJTU/Evo-1.git "${EVO1_DIR}"
    fi
fi

# ==============================================================================
# Environment 2: libero_client (Python 3.8.13 — LIBERO official requirement)
# Runs LIBERO benchmark simulation.
# ==============================================================================
if ! conda env list | grep -q "^libero_client "; then
    echo "[$(date)] Creating conda env: libero_client (Python 3.8.13)..."
    conda create -n libero_client python=3.8.13 -y

    echo "[$(date)] Installing PyTorch 1.11.0+cu113 in libero_client..."
    conda run -n libero_client pip install -q \
        torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
        --extra-index-url https://download.pytorch.org/whl/cu113

    echo "[$(date)] Installing libero_client dependencies..."
    conda run -n libero_client pip install -q \
        'numpy>=1.20,<2.0' robosuite==1.4.1 mujoco==2.3.7 \
        imageio h5py bddl websockets huggingface_hub 'transformers==4.57.6'

    # Clone LIBERO
    if [ ! -d "${LIBERO_DIR}" ]; then
        echo "[$(date)] Cloning LIBERO..."
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
    fi

    conda run -n libero_client pip install -q \
        -r "${LIBERO_DIR}/requirements.txt" 2>&1 | tail -5 || true
    conda run -n libero_client pip install -q -e "${LIBERO_DIR}" 2>&1 | tail -5 || true
else
    echo "[$(date)] libero_client already exists, skipping creation."
    if [ ! -d "${LIBERO_DIR}" ]; then
        echo "[$(date)] Cloning LIBERO..."
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
    fi
fi

# ==============================================================================
# Environment 3: metaworld_client (Python 3.10)
# Runs MetaWorld benchmark simulation.
# ==============================================================================
if ! conda env list | grep -q "^metaworld_client "; then
    echo "[$(date)] Creating conda env: metaworld_client (Python 3.10)..."
    conda create -n metaworld_client python=3.10 -y

    echo "[$(date)] Installing metaworld_client dependencies..."
    conda run -n metaworld_client pip install -q \
        mujoco websockets opencv-python packaging huggingface_hub \
        metaworld gymnasium
else
    echo "[$(date)] metaworld_client already exists, skipping creation."
fi

# ==============================================================================
# Environment Variables
# ==============================================================================
export MUJOCO_GL=egl
unset DISPLAY

# HuggingFace token — set via environment before job submission OR uncomment:
# export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXX

# ==============================================================================
# Convert Notebook → Python Script
# ==============================================================================
echo "[$(date)] Converting notebook to Python script..."

# Use evo1_server env (has jupyter/nbconvert)
source /tc1apps/anaconda3/bin/activate evo1_server
pip install -q jupyter nbconvert 2>/dev/null || true

SCRIPT_RAW=/tmp/evo1_complete_raw.py
SCRIPT_PATCHED=/tmp/evo1_complete_tc1.py

jupyter nbconvert --to script "${NOTEBOOK}" --output /tmp/evo1_complete_raw

# ==============================================================================
# Patch Script: Remove Colab-specific code, remap /content/ → TC1 paths
# ==============================================================================
echo "[$(date)] Patching script for TC1 environment..."

python3 << 'PYEOF'
import re, os

script_raw     = '/tmp/evo1_complete_raw.py'
script_patched = '/tmp/evo1_complete_tc1.py'

workspace    = os.environ['TC1_WORKSPACE']
results_dir  = os.environ['TC1_RESULTS_DIR']
evo1_dir     = os.environ['TC1_EVO1_DIR']
libero_dir   = os.environ['TC1_LIBERO_DIR']
ckpt_dir     = os.environ['TC1_CKPT_DIR']
logs_dir     = os.environ['TC1_LOGS_DIR']
data_dir     = os.environ['TC1_DATA_DIR']
project_root = os.environ['TC1_PROJECT_ROOT']
mhs_dir      = os.environ['TC1_MHS_DIR']

with open(script_raw, 'r') as f:
    code = f.read()

# ---- Remove Google Colab / system-level installs ----
remove_patterns = [
    r'from google\.colab import[^\n]*\n',
    r'from google import[^\n]*\n',
    r'import google[^\n]*\n',
    r"get_ipython\(\)\.system\('apt-get[^']*'\)\n",
    r'get_ipython\(\)\.system\("apt-get[^"]*"\)\n',
    r"get_ipython\(\)\.system\('apt[^']*'\)\n",
    r"get_ipython\(\)\.system\('wget[^']*[Mm]iniconda[^']*'\)\n",
    r"get_ipython\(\)\.system\('bash[^']*[Mm]iniconda[^']*'\)\n",
    r'get_ipython\(\)\.system\("wget[^"]*[Mm]iniconda[^"]*"\)\n',
    r'get_ipython\(\)\.system\("bash[^"]*[Mm]iniconda[^"]*"\)\n',
    r"get_ipython\(\)\.run_cell_magic\('bash'[^)]*\)\n",
]
for pat in remove_patterns:
    code = re.sub(pat, '# [TC1-PATCH] Removed (not needed on TC1)\n', code)

# ---- Neutralise Colab-specific calls ----
code = re.sub(r'drive\.mount\([^\)]*\)', '# [TC1-PATCH] Skipped drive.mount', code)

# ---- Remap /content/ paths (longest match first) ----
path_map = [
    # Drive result paths
    ("/content/drive/MyDrive/Research/URECA/Results/MetaworldAblationPass0", results_dir + "/MetaWorld"),
    ("/content/drive/MyDrive/Research/URECA/Results/LIBEROAblationPass0",    results_dir + "/LIBERO"),
    ("/content/drive/MyDrive/evo1_study/Results/baseline",                    results_dir + "/baseline"),
    ("/content/drive/MyDrive/evo1_study/Results/ablation",                    results_dir + "/ablation"),
    ("/content/drive/MyDrive/evo1_study/Results/gradient",                    results_dir + "/gradient"),
    ("/content/drive/MyDrive/evo1_study/Results",                             results_dir),
    ("/content/drive/MyDrive/evo1_study",                                     results_dir),
    ("/content/drive/MyDrive/evo1_ablation",                                  results_dir),
    # Repo paths
    ("/content/Evo-1",                                                        evo1_dir),
    ("/content/LIBERO_evaluation/LIBERO",                                     libero_dir),
    ("/content/LIBERO_evaluation",                                            os.path.dirname(libero_dir)),
    ("/content/LIBERO",                                                       libero_dir),
    ("/content/EmbodiedLLM/MultipleHooksStudy",                               mhs_dir),
    ("/content/EmbodiedLLM",                                                  project_root),
    # Local paths
    ("/content/checkpoints",                                                  ckpt_dir),
    ("/content/logs",                                                         logs_dir),
    ("/content/data",                                                         data_dir),
    ("/content/evo1_gradient_analysis_results.json",                          results_dir + "/gradient/evo1_gradient_analysis_results.json"),
    # Generic fallback
    ("/content/",                                                             workspace + "/"),
    ("/content",                                                              workspace),
]
for old, new in path_map:
    code = code.replace(old, new)

# ---- Replace conda path references ----
code = code.replace('/opt/conda/bin', '/tc1apps/anaconda3/bin')

# ---- Header ----
header = f"""# ============================================================
# TC1 PATCHED SCRIPT — auto-generated from evo1_complete.ipynb
# ============================================================
import sys, os

# Create required directories upfront
for _d in [
    '{workspace}', '{results_dir}',
    '{results_dir}/baseline', '{results_dir}/ablation', '{results_dir}/gradient',
    '{results_dir}/MetaWorld', '{results_dir}/LIBERO',
    '{ckpt_dir}', '{logs_dir}', '{data_dir}',
]:
    os.makedirs(_d, exist_ok=True)

# Ensure conda is findable in subprocess calls
_conda_bin = '/tc1apps/anaconda3/bin'
if _conda_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _conda_bin + ':' + os.environ.get('PATH', '')

# Add project to sys.path
sys.path.insert(0, '{mhs_dir}')
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
    py_compile.compile('/tmp/evo1_complete_tc1.py', doraise=True)
    print('[VALIDATE] Syntax OK')
except py_compile.PyCompileError as e:
    print(f'[VALIDATE] Syntax error: {e}')
    sys.exit(1)
"

# ==============================================================================
# Run Evaluation
# ==============================================================================
echo "[$(date)] Starting Evo-1 Complete evaluation..."
echo "[$(date)] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

cd "${MHS_DIR}/notebooks"
python3 "${SCRIPT_PATCHED}" 2>&1 | tee "${LOGS_DIR}/evo1_complete_run.log"
EXIT_CODE=${PIPESTATUS[0]}

# ==============================================================================
# Post-run Summary
# ==============================================================================
echo ""
echo "[$(date)] ====== Evo-1 Complete Job Finished ======"
echo "[$(date)] Exit code: ${EXIT_CODE}"
echo "[$(date)] Results directory: ${RESULTS_DIR}"
echo "[$(date)] Files generated:"
find "${RESULTS_DIR}" -type f -name "*.json" -o -name "*.png" 2>/dev/null | sort

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[$(date)] WARNING: Script exited with non-zero code ${EXIT_CODE}"
    echo "[$(date)] Check error log for details."
fi

exit ${EXIT_CODE}
