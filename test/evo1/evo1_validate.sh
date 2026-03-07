#!/bin/bash
# ==============================================================================
# Evo-1 Pre-flight Validation Script
# Run this BEFORE submitting the full job to verify environment and dependencies.
#
# Evo-1 requires 3 conda envs: evo1_server, libero_client, metaworld_client
# This script checks all three.
#
# Usage: sbatch evo1_validate.sh
# ==============================================================================

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=4
#SBATCH --time=30
#SBATCH --job-name=Evo1_Validate
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/evo1/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/evo1/logs/error_%x_%j.err

set -euo pipefail
PASS=0
FAIL=0
WARN=0

check_pass() { echo "  [PASS] $1"; ((PASS++)); }
check_fail() { echo "  [FAIL] $1"; ((FAIL++)); }
check_warn() { echo "  [WARN] $1"; ((WARN++)); }

echo "============================================================"
echo "Evo-1 Pre-flight Validation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# --- 1. Module availability ---
echo ""
echo "1. Module Availability"
module load anaconda 2>/dev/null && check_pass "anaconda module" || check_fail "anaconda module"
module load cuda/12.1 2>/dev/null && check_pass "cuda/12.1 module" || check_fail "cuda/12.1 module"

# --- 2. GPU check ---
echo ""
echo "2. GPU Check"
if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    check_pass "GPU available: ${GPU_NAME} (${GPU_MEM})"
else
    check_fail "No GPU detected (nvidia-smi failed)"
fi

# --- 3. Conda environments ---
echo ""
echo "3. Conda Environments (Evo-1 uses 3 envs)"

# evo1_server
if conda env list | grep -q "^evo1_server "; then
    check_pass "evo1_server env exists"
    source /tc1apps/anaconda3/bin/activate evo1_server 2>/dev/null

    PY_VER=$(python3 --version 2>&1)
    check_pass "  Python: ${PY_VER}"

    for pkg in torch transformers accelerate websockets einops timm; do
        if python3 -c "import ${pkg}" 2>/dev/null; then
            ver=$(python3 -c "import ${pkg}; print(getattr(${pkg}, '__version__', 'ok'))" 2>/dev/null)
            check_pass "  ${pkg} (${ver})"
        else
            check_fail "  ${pkg} not installed in evo1_server"
        fi
    done

    # Check transformers version (must be 4.x, NOT 5.x)
    TF_VER=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    if [[ "$TF_VER" == 4.* ]]; then
        check_pass "  transformers version OK (${TF_VER})"
    else
        check_fail "  transformers version ${TF_VER} — must be 4.57.6 (NOT 5.x)"
    fi

    # flash-attn
    if python3 -c "import flash_attn" 2>/dev/null; then
        check_pass "  flash-attn installed"
    else
        check_warn "  flash-attn not installed (optional but recommended)"
    fi

    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        check_pass "  PyTorch CUDA available"
    else
        check_fail "  PyTorch cannot see CUDA in evo1_server"
    fi
    conda deactivate 2>/dev/null || true
else
    check_warn "evo1_server env not found (will be created by job script)"
fi

# libero_client
if conda env list | grep -q "^libero_client "; then
    check_pass "libero_client env exists"
    source /tc1apps/anaconda3/bin/activate libero_client 2>/dev/null
    PY_VER=$(python3 --version 2>&1)
    check_pass "  Python: ${PY_VER}"
    for pkg in torch robosuite mujoco websockets; do
        if python3 -c "import ${pkg}" 2>/dev/null; then
            check_pass "  ${pkg}"
        else
            check_fail "  ${pkg} not installed in libero_client"
        fi
    done
    conda deactivate 2>/dev/null || true
else
    check_warn "libero_client env not found (will be created by job script)"
fi

# metaworld_client
if conda env list | grep -q "^metaworld_client "; then
    check_pass "metaworld_client env exists"
    source /tc1apps/anaconda3/bin/activate metaworld_client 2>/dev/null
    PY_VER=$(python3 --version 2>&1)
    check_pass "  Python: ${PY_VER}"
    for pkg in mujoco metaworld websockets; do
        if python3 -c "import ${pkg}" 2>/dev/null; then
            check_pass "  ${pkg}"
        else
            check_fail "  ${pkg} not installed in metaworld_client"
        fi
    done
    conda deactivate 2>/dev/null || true
else
    check_warn "metaworld_client env not found (will be created by job script)"
fi

# --- 4. Project structure ---
echo ""
echo "4. Project Structure"
PROJECT_ROOT=/tc1home/FYP/prithvi004/EmbodiedLLM
MHS_DIR=${PROJECT_ROOT}/MultipleHooksStudy

for f in \
    "${MHS_DIR}/notebooks/evo1_complete.ipynb" \
    "${MHS_DIR}/hooks/model_specific/evo1_hooks.py" \
    "${MHS_DIR}/hooks/losses/evo1_loss.py" \
    "${MHS_DIR}/hooks/model_specific/__init__.py" \
    "${MHS_DIR}/hooks/gradient_hooks.py" \
    "${MHS_DIR}/hooks/ablation_hooks.py" \
    "${MHS_DIR}/scripts/run_evo1_gradient_analysis.py" \
    "${MHS_DIR}/scripts/data_collectors/libero_collector.py"; do
    if [ -f "$f" ]; then
        check_pass "$(basename $f)"
    else
        check_fail "Missing: $f"
    fi
done

# --- 5. Workspace clones ---
echo ""
echo "5. Workspace Clones (cached)"
WORKSPACE=${MHS_DIR}/workspace/evo1
for d in "${WORKSPACE}/Evo-1" "${WORKSPACE}/LIBERO" "${WORKSPACE}/checkpoints"; do
    if [ -d "$d" ]; then
        check_pass "$(basename $d) exists"
    else
        check_warn "$(basename $d) not yet created (will be by job script)"
    fi
done

# --- 6. Import tests ---
echo ""
echo "6. Import Tests"
if conda env list | grep -q "^evo1_server "; then
    source /tc1apps/anaconda3/bin/activate evo1_server 2>/dev/null
    python3 << 'PYTEST' 2>&1
import sys
sys.path.insert(0, '/tc1home/FYP/prithvi004/EmbodiedLLM/MultipleHooksStudy')

try:
    from hooks.model_specific.evo1_hooks import Evo1Hooks
    print("  [PASS] Evo1Hooks import")
except Exception as e:
    print(f"  [FAIL] Evo1Hooks import: {e}")

try:
    from hooks.losses.evo1_loss import evo1_flow_matching_loss
    print("  [PASS] evo1_flow_matching_loss import")
except Exception as e:
    print(f"  [FAIL] evo1_flow_matching_loss import: {e}")

try:
    import torch
    if torch.cuda.is_available():
        x = torch.randn(2, 2, device='cuda')
        print(f"  [PASS] GPU compute ({torch.cuda.get_device_name(0)})")
    else:
        print("  [FAIL] CUDA not available")
except Exception as e:
    print(f"  [FAIL] GPU test: {e}")
PYTEST
    conda deactivate 2>/dev/null || true
else
    check_warn "Skipping import tests (evo1_server env not found)"
fi

# --- 7. Disk and network ---
echo ""
echo "7. Disk & Network"
USAGE=$(df -h /tc1home 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
if [ -n "$USAGE" ] && [ "$USAGE" -lt 90 ]; then
    check_pass "Disk usage: ${USAGE}%"
else
    check_warn "Disk usage: ${USAGE}% (>90% may cause issues)"
fi

HOME_SIZE=$(du -sh /tc1home/FYP/prithvi004 2>/dev/null | cut -f1)
check_pass "Home directory size: ${HOME_SIZE}"

if python3 -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=10)" 2>/dev/null; then
    check_pass "HuggingFace Hub reachable"
else
    check_warn "HuggingFace Hub unreachable"
fi

# --- 8. Port availability (Evo-1 uses ports 9001-9040) ---
echo ""
echo "8. Port Availability"
PORT_CONFLICT=0
for port in 9001 9011 9021 9031; do
    if netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
        check_warn "Port ${port} in use"
        ((PORT_CONFLICT++))
    fi
done
if [ $PORT_CONFLICT -eq 0 ]; then
    check_pass "Ports 9001-9040 available"
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "VALIDATION SUMMARY"
echo "  Passed: ${PASS}"
echo "  Failed: ${FAIL}"
echo "  Warnings: ${WARN}"
echo "============================================================"

if [ $FAIL -gt 0 ]; then
    echo "Fix failures before submitting the full job."
    exit 1
else
    echo "Ready to submit: sbatch evo1_complete_job.sh"
    exit 0
fi
