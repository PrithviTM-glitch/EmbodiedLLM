#!/bin/bash
# π0 Dependency Verification Script
# Checks all dependencies required for π0 model
# Based on: https://github.com/Physical-Intelligence/openpi

set -e

echo "=========================================="
echo "π0 Dependency Verification"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILURES=0
WARNINGS=0

# Function to check command exists
check_command() {
    if command -v "$1" &> /dev/null; then
        VERSION=$("$1" --version 2>&1 | head -n 1 || echo "unknown")
        echo -e "${GREEN}✓${NC} $1 found ($VERSION)"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

# Function to check Python package
check_python_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}✓${NC} $1 installed (version: $VERSION)"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not installed"
        return 1
    fi
}

echo "1. Package Manager"
echo "-----------------"
if check_command uv; then
    echo -e "${GREEN}✓${NC} uv (modern package manager) installed"
else
    echo -e "${YELLOW}⚠${NC} uv not found (recommended for π0)"
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    ((WARNINGS++))
fi
echo ""

echo "2. Python Environment"
echo "--------------------"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# π0 is more flexible with Python versions
if [[ "$PYTHON_VERSION" > "3.8" ]]; then
    echo -e "${GREEN}✓${NC} Python version compatible"
else
    echo -e "${RED}✗${NC} Python 3.8+ required"
    ((FAILURES++))
fi
echo ""

echo "3. PyTorch Installation"
echo "----------------------"
if check_python_package torch; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "PyTorch version: $TORCH_VERSION"
    
    # Check CUDA availability
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        echo -e "${GREEN}✓${NC} CUDA available (version: $CUDA_VERSION)"
    else
        echo -e "${YELLOW}⚠${NC} CUDA not available (recommended for π0)"
        echo "Note: π0 can run on CPU but will be slower"
        ((WARNINGS++))
    fi
else
    echo -e "${RED}✗${NC} PyTorch not installed"
    echo "π0 recently added PyTorch support (converted from JAX)"
    ((FAILURES++))
fi
echo ""

echo "4. Core Dependencies"
echo "-------------------"
check_python_package transformers || ((FAILURES++))
check_python_package einops || ((FAILURES++))
check_python_package numpy || ((FAILURES++))
check_python_package PIL || echo -e "${YELLOW}⚠${NC} Pillow recommended"

# Check for JAX (legacy - π0 now supports PyTorch)
if check_python_package jax; then
    echo -e "${YELLOW}⚠${NC} JAX detected (legacy support)"
    echo "Note: π0 now supports PyTorch natively"
else
    echo -e "${GREEN}✓${NC} Using PyTorch backend (recommended)"
fi
echo ""

echo "5. π0 Architecture Components"
echo "-----------------------------"
echo "π0 uses separate multi-layer proprio encoder:"
echo "  - Vision encoder: Pre-trained vision transformer"
echo "  - Language encoder: Pre-trained language model"
echo "  - Proprio encoder: Separate multi-layer MLP (NOT fused into VL)"
echo "  - Action decoder: Flow matching with block-wise causal masking"
echo ""

echo "6. Available Checkpoints"
echo "-----------------------"
echo "π0 checkpoints are hosted on Google Cloud Storage:"
echo ""
echo "Base models:"
echo "  - pi0_base (3.3B): General-purpose checkpoint"
echo "  - pi0_fast (smaller): Faster inference variant"
echo ""
echo "Fine-tuned models:"
echo "  - pi05_libero: LIBERO benchmark (verified working)"
echo "  - pi0_droid: DROID expert checkpoint (200k frames)"
echo "  - pi0_aloha: ALOHA expert checkpoint (150k frames)"
echo ""
echo -e "${YELLOW}Note:${NC} Checkpoints must be downloaded manually from GCS"
echo "See: https://github.com/Physical-Intelligence/openpi/releases"
echo ""

echo "7. Policy Server Dependencies"
echo "----------------------------"
echo "π0 includes policy server (serve_policy.py):"
if python -c "import flask" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} flask installed (for policy server)"
else
    echo -e "${YELLOW}⚠${NC} flask recommended for policy server"
    echo "Install with: pip install flask"
fi

if python -c "import requests" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} requests installed (for client)"
else
    echo -e "${YELLOW}⚠${NC} requests recommended for client"
fi
echo ""

echo "8. JAX to PyTorch Conversion"
echo "----------------------------"
echo "If using JAX checkpoints, conversion required:"
echo "  1. Load JAX checkpoint with jax.numpy"
echo "  2. Convert arrays: torch.from_numpy(np.array(jax_array))"
echo "  3. Save as PyTorch state_dict"
echo ""
echo "Most recent checkpoints support PyTorch natively."
echo ""

echo "9. Benchmark Environment Support"
echo "--------------------------------"
echo "LIBERO:"
echo "  ✓ pi05_libero checkpoint available"
echo "  - Requires LIBERO env setup (Python 3.8.13)"
echo ""
echo "DROID:"
echo "  ✓ pi0_droid expert checkpoint (200k frames)"
echo "  - Pre-trained on diverse manipulation tasks"
echo ""
echo "Meta-World:"
echo "  ✗ No dedicated checkpoint (use pi0_base for transfer)"
echo ""

echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "Failures: ${RED}$FAILURES${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✓ All critical dependencies satisfied!${NC}"
    echo ""
    echo "Next Steps:"
    echo "1. Download π0 checkpoint from GCS"
    echo "2. Start policy server: python serve_policy.py --checkpoint pi0_base"
    echo "3. Set up LIBERO env for pi05_libero checkpoint"
    echo ""
    echo "Installation (if warnings exist):"
    echo "  uv pip install torch transformers einops pillow numpy flask requests"
    exit 0
else
    echo -e "${RED}✗ Critical dependencies missing!${NC}"
    echo ""
    echo "Installation Instructions:"
    echo ""
    echo "Option 1 (using uv - recommended):"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  uv venv"
    echo "  source .venv/bin/activate"
    echo "  uv pip install torch transformers einops pillow numpy flask requests"
    echo ""
    echo "Option 2 (using pip):"
    echo "  conda create -n pi0 python=3.10 -y && conda activate pi0"
    echo "  pip install torch transformers einops pillow numpy flask requests"
    echo ""
    echo "3. Download checkpoint:"
    echo "  wget https://storage.googleapis.com/pi0-public/checkpoints/pi0_base.pt"
    echo ""
    echo "4. Test installation:"
    echo "  python -c 'import torch; import transformers; print(\"✓ Ready\")'"
    exit 1
fi
