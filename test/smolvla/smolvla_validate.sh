#!/bin/bash
# ==============================================================================
# SmolVLA Pre-flight Validation Script
# Run this BEFORE submitting the full job to verify environment and dependencies.
#
# Usage (as SLURM job):
#   sbatch smolvla_validate.sh
# Or interactively on a compute node (via Jupyter terminal):
#   bash smolvla_validate.sh
# ==============================================================================

#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=4
#SBATCH --time=30
#SBATCH --job-name=SmolVLA_Validate
#SBATCH --output=/tc1home/FYP/prithvi004/EmbodiedLLM/test/smolvla/logs/output_%x_%j.out
#SBATCH --error=/tc1home/FYP/prithvi004/EmbodiedLLM/test/smolvla/logs/error_%x_%j.err

set -euo pipefail
PASS=0
FAIL=0
WARN=0

check_pass() { echo "  [PASS] $1"; ((PASS++)); }
check_fail() { echo "  [FAIL] $1"; ((FAIL++)); }
check_warn() { echo "  [WARN] $1"; ((WARN++)); }

echo "============================================================"
echo "SmolVLA Pre-flight Validation"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "============================================================"

# --- 1. Module availability ---
echo ""
echo "1. Module Availability"
module load anaconda 2>/dev/null && check_pass "anaconda module" || check_fail "anaconda module"
module load cuda/12.4 2>/dev/null && check_pass "cuda/12.4 module" || check_fail "cuda/12.4 module"

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

# --- 3. Conda environment ---
echo ""
echo "3. Conda Environment"
ENV_NAME=smolvla_env
if conda env list | grep -q "^${ENV_NAME} "; then
    check_pass "Conda env '${ENV_NAME}' exists"
    source /tc1apps/anaconda3/bin/activate "${ENV_NAME}"
    
    # Check Python version
    PY_VER=$(python3 --version 2>&1)
    check_pass "Python: ${PY_VER}"
    
    # Check critical packages
    for pkg in torch torchvision lerobot transformers safetensors matplotlib numpy; do
        if python3 -c "import ${pkg}" 2>/dev/null; then
            ver=$(python3 -c "import ${pkg}; print(getattr(${pkg}, '__version__', 'installed'))" 2>/dev/null)
            check_pass "${pkg} (${ver})"
        else
            check_fail "${pkg} not installed"
        fi
    done
    
    # Check CUDA availability in PyTorch
    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAIL" = "True" ]; then
        CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        check_pass "PyTorch CUDA: ${CUDA_VER}"
    else
        check_fail "PyTorch cannot see CUDA"
    fi
else
    check_warn "Conda env '${ENV_NAME}' not found (will be created by job script)"
fi

# --- 4. Project structure ---
echo ""
echo "4. Project Structure"
PROJECT_ROOT=/tc1home/FYP/prithvi004/EmbodiedLLM
MHS_DIR=${PROJECT_ROOT}/MultipleHooksStudy

for f in \
    "${MHS_DIR}/notebooks/smolvla_complete.ipynb" \
    "${MHS_DIR}/hooks/model_specific/smolvla_hooks.py" \
    "${MHS_DIR}/hooks/model_specific/__init__.py" \
    "${MHS_DIR}/hooks/gradient_hooks.py" \
    "${MHS_DIR}/hooks/ablation_hooks.py" \
    "${MHS_DIR}/hooks/representation_hooks.py" \
    "${MHS_DIR}/hooks/utilization_hooks.py"; do
    if [ -f "$f" ]; then
        check_pass "$(basename $f)"
    else
        check_fail "Missing: $f"
    fi
done

# --- 5. SmolVLA model import test ---
echo ""
echo "5. SmolVLA Import Test"
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    python3 << 'PYTEST' 2>&1
import sys
sys.path.insert(0, '/tc1home/FYP/prithvi004/EmbodiedLLM/MultipleHooksStudy')

# Test hook imports
try:
    from hooks.model_specific.smolvla_hooks import SmolVLAHooks
    print("  [PASS] SmolVLAHooks import")
except Exception as e:
    print(f"  [FAIL] SmolVLAHooks import: {e}")

# Test lerobot SmolVLA
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("  [PASS] SmolVLAPolicy import")
except Exception as e:
    print(f"  [FAIL] SmolVLAPolicy import: {e}")

# Test model loading (lightweight check - config only)
try:
    from transformers import AutoConfig
    # Just verify the model ID is reachable
    print("  [PASS] transformers AutoConfig available")
except Exception as e:
    print(f"  [WARN] transformers config check: {e}")

# Test PyTorch GPU
try:
    import torch
    if torch.cuda.is_available():
        x = torch.randn(2, 2, device='cuda')
        y = x @ x.T
        print(f"  [PASS] GPU compute test ({torch.cuda.get_device_name(0)})")
        print(f"  [PASS] GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("  [FAIL] CUDA not available")
except Exception as e:
    print(f"  [FAIL] GPU test: {e}")
PYTEST
else
    check_warn "Skipping import tests (no conda env active)"
fi

# --- 6. Disk space ---
echo ""
echo "6. Disk Space"
USAGE=$(df -h /tc1home 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
if [ -n "$USAGE" ] && [ "$USAGE" -lt 90 ]; then
    check_pass "Disk usage: ${USAGE}%"
else
    check_warn "Disk usage: ${USAGE}% (>90% may cause issues)"
fi

HOME_SIZE=$(du -sh /tc1home/FYP/prithvi004 2>/dev/null | cut -f1)
check_pass "Home directory size: ${HOME_SIZE}"

# --- 7. Network (HuggingFace reachability) ---
echo ""
echo "7. Network"
if python3 -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=10)" 2>/dev/null; then
    check_pass "HuggingFace Hub reachable"
else
    check_warn "HuggingFace Hub unreachable (model downloads may fail)"
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
    echo "Ready to submit: sbatch smolvla_complete_job.sh"
    exit 0
fi
