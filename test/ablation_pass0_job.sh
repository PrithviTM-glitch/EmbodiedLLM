#!/bin/bash
# ==============================================================================
# SLURM Job Script: Evo-1 Ablation — State Encoder Pass 0
# Runs MultipleHooksStudy/notebooks/ablation_state_encoder_pass0.ipynb on TC1
#
# IMPORTANT: This job creates 3 conda environments and downloads model
# checkpoints. First run may take longer than 6 hours. If so, apply for
# extended QoS (up to 48h) via CCDSgpu-tc@ntu.edu.sg.
#
# To avoid recreating envs on reruns, environments persist in ~/.conda/envs/.
#
# Submit with: sbatch ablation_pass0_job.sh
# ==============================================================================

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=8
#SBATCH --time=360
#SBATCH --job-name=Ablation_Pass0
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/logs/error_%x_%j.err

set -euo pipefail
echo "[$(date)] ====== Ablation Pass0 Job Started ======"

# ==============================================================================
# Paths
# ==============================================================================
PROJECT_ROOT=/tc1home/FYP/prithvi004/EmbodiedLLM
MHS_DIR=${PROJECT_ROOT}/MultipleHooksStudy
WORKSPACE=${MHS_DIR}/workspace/evo1_ablation
RESULTS_DIR=${MHS_DIR}/results/ablation_pass0
NOTEBOOK=${MHS_DIR}/notebooks/ablation_state_encoder_pass0.ipynb

EVO1_DIR=${WORKSPACE}/Evo-1
LIBERO_DIR=${WORKSPACE}/LIBERO_evaluation/LIBERO
CKPT_DIR=${WORKSPACE}/checkpoints
LOGS_DIR=${WORKSPACE}/logs

mkdir -p "${WORKSPACE}" "${RESULTS_DIR}" "${CKPT_DIR}" "${LOGS_DIR}"
mkdir -p "${WORKSPACE}/LIBERO_evaluation"

# Export for the patch script
export TC1_WORKSPACE="${WORKSPACE}"
export TC1_RESULTS_DIR="${RESULTS_DIR}"
export TC1_EVO1_DIR="${EVO1_DIR}"
export TC1_LIBERO_DIR="${LIBERO_DIR}"
export TC1_CKPT_DIR="${CKPT_DIR}"
export TC1_LOGS_DIR="${LOGS_DIR}"

# ==============================================================================
# Module Loading
# ==============================================================================
module load anaconda
module load cuda/12.1

# Ensure conda is on PATH for subprocess calls inside the notebook
export PATH=/tc1apps/anaconda3/bin:${PATH}

# ==============================================================================
# Environment 1: evo1_server (Python 3.10 + PyTorch 2.5.1 cu121)
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
        websockets pyyaml huggingface_hub tqdm

    # Clone Evo-1 repo (needed for requirements.txt and scripts)
    if [ ! -d "${EVO1_DIR}" ]; then
        echo "[$(date)] Cloning Evo-1..."
        git clone https://github.com/MINT-SJTU/Evo-1.git "${EVO1_DIR}"
    fi

    echo "[$(date)] Installing Evo-1 requirements in evo1_server..."
    conda run -n evo1_server pip install -q \
        -r "${EVO1_DIR}/Evo_1/requirements.txt"

    # flash-attn is optional; skip if it fails (no sudo for system libs)
    conda run -n evo1_server pip install -q flash-attn --no-build-isolation 2>/dev/null \
        || echo "[$(date)] flash-attn skipped (build failed — optional)"
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
        imageio h5py bddl websockets huggingface_hub

    # Clone LIBERO
    if [ ! -d "${LIBERO_DIR}" ]; then
        echo "[$(date)] Cloning LIBERO..."
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
    fi

    conda run -n libero_client pip install -q \
        -r "${LIBERO_DIR}/requirements.txt"
    conda run -n libero_client pip install -q -e "${LIBERO_DIR}"
else
    echo "[$(date)] libero_client already exists, skipping creation."
    if [ ! -d "${LIBERO_DIR}" ]; then
        echo "[$(date)] Cloning LIBERO..."
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
    fi
fi

# ==============================================================================
# Environment 3: metaworld_client (Python 3.10)
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
# Download Evo-1 Checkpoint via HuggingFace Hub
# (The notebook does this via huggingface_hub.snapshot_download)
# Set HUGGING_FACE_HUB_TOKEN if models are gated.
# ==============================================================================
# export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXXXXXXXXXX

# ==============================================================================
# Environment Variables for MuJoCo / rendering
# ==============================================================================
export MUJOCO_GL=egl
unset DISPLAY

# ==============================================================================
# Convert Notebook → Python Script
# ==============================================================================
echo "[$(date)] Converting notebook to Python script..."

# Use a base conda env that has jupyter/nbconvert
source /tc1apps/anaconda3/bin/activate evo1_server
pip install -q jupyter nbconvert 2>/dev/null || true

SCRIPT_RAW=/tmp/ablation_pass0_raw.py
SCRIPT_PATCHED=/tmp/ablation_pass0_tc1.py

jupyter nbconvert --to script "${NOTEBOOK}" --output /tmp/ablation_pass0_raw

# ==============================================================================
# Patch Script: Remove Colab-specific code, remap /content/ → TC1 paths
# ==============================================================================
echo "[$(date)] Patching script for TC1 environment..."

python3 << 'PYEOF'
import re, os

script_raw     = '/tmp/ablation_pass0_raw.py'
script_patched = '/tmp/ablation_pass0_tc1.py'

workspace    = os.environ['TC1_WORKSPACE']
results_dir  = os.environ['TC1_RESULTS_DIR']
evo1_dir     = os.environ['TC1_EVO1_DIR']
libero_dir   = os.environ['TC1_LIBERO_DIR']
ckpt_dir     = os.environ['TC1_CKPT_DIR']
logs_dir     = os.environ['TC1_LOGS_DIR']

libero_eval_dir = os.path.dirname(libero_dir)  # WORKSPACE/LIBERO_evaluation

with open(script_raw, 'r') as f:
    code = f.read()

# ---- Remove Google Colab / system-level installs not available on TC1 ----
remove_patterns = [
    # Google Colab imports
    r'from google\.colab import[^\n]*\n',
    r'from google import[^\n]*\n',
    r'import google[^\n]*\n',
    # apt-get (no sudo on TC1; system deps already present)
    r"get_ipython\(\)\.system\('apt-get[^']*'\)\n",
    r'get_ipython\(\)\.system\("apt-get[^"]*"\)\n',
    r"get_ipython\(\)\.system\('apt[^']*'\)\n",
    # Miniconda download/install (conda already on TC1 via module load anaconda)
    r"get_ipython\(\)\.system\('wget[^']*[Mm]iniconda[^']*'\)\n",
    r"get_ipython\(\)\.system\('bash[^']*[Mm]iniconda[^']*'\)\n",
    r"get_ipython\(\)\.system\(\"wget[^\"]*[Mm]iniconda[^\"]*\"\)\n",
    r"get_ipython\(\)\.system\(\"bash[^\"]*[Mm]iniconda[^\"]*\"\)\n",
]
for pat in remove_patterns:
    code = re.sub(pat, '# [TC1-PATCH] Removed (not needed on TC1)\n', code)

# ---- Neutralise remaining Colab-specific calls ----
code = re.sub(r'drive\.mount\([^\)]*\)', '# [TC1-PATCH] Skipped drive.mount', code)

# ---- Remap /content/ paths (longest match first) ----
path_map = [
    # Google Drive result paths → TC1 results
    ("/content/drive/MyDrive/Research/URECA/Results/MetaworldAblationPass0", results_dir + "/MetaWorld"),
    ("/content/drive/MyDrive/Research/URECA/Results/LIBEROAblationPass0",    results_dir + "/LIBERO"),
    ("/content/drive/MyDrive/evo1_ablation",                                  results_dir),
    # Repo paths
    ("/content/Evo-1",                                                         evo1_dir),
    ("/content/LIBERO_evaluation/LIBERO",                                      libero_dir),
    ("/content/LIBERO_evaluation",                                             libero_eval_dir),
    # Checkpoint and logs
    ("/content/checkpoints",                                                   ckpt_dir),
    ("/content/logs",                                                          logs_dir),
    # Generic /content/ fallback
    ("/content/",                                                              workspace + "/"),
    ("/content",                                                               workspace),
]
for old, new in path_map:
    code = code.replace(old, new)

# ---- Replace conda path references (Colab uses /opt/conda; TC1 uses /tc1apps/anaconda3) ----
code = code.replace('/opt/conda/bin', '/tc1apps/anaconda3/bin')

# ---- Header: directory creation + env-specific sys.path ----
header = f"""# ============================================================
# TC1 PATCHED SCRIPT — auto-generated from ablation_state_encoder_pass0.ipynb
# ============================================================
import sys, os

# Create required directories upfront
for _d in [
    '{workspace}', '{results_dir}',
    '{results_dir}/MetaWorld', '{results_dir}/LIBERO',
    '{ckpt_dir}', '{logs_dir}',
    '{libero_eval_dir}',
]:
    os.makedirs(_d, exist_ok=True)

# Ensure conda is findable in subprocess calls spawned by the notebook
_conda_bin = '/tc1apps/anaconda3/bin'
if _conda_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _conda_bin + ':' + os.environ.get('PATH', '')
"""
code = header + code

with open(script_patched, 'w') as f:
    f.write(code)
print(f"[TC1-PATCH] Written patched script to {script_patched}")
PYEOF

# ==============================================================================
# Run Evaluation
# ==============================================================================
echo "[$(date)] Starting Evo-1 Ablation Pass0 evaluation..."
cd "${MHS_DIR}/notebooks"
python3 "${SCRIPT_PATCHED}"

echo "[$(date)] ====== Ablation Pass0 Job Finished ======"
