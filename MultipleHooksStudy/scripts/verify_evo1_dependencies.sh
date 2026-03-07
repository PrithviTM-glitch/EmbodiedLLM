#!/bin/bash
# Evo-1 Dependency Verification Script
# Checks all dependencies required for Evo-1 model
# Based on: https://github.com/MINT-SJTU/Evo-1

set -e

echo "=========================================="
echo "Evo-1 Dependency Verification"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILURES=0
WARNINGS=0

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

# Function to check HuggingFace model
check_hf_model() {
    echo -n "Checking HuggingFace model: $1... "
    if python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='$1', filename='config.json', local_files_only=True)" 2>/dev/null; then
        echo -e "${GREEN}✓ cached locally${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ not cached (will download on first use)${NC}"
        return 1
    fi
}

echo "1. Python Environment"
echo "--------------------"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_PYTHON="3.10"
echo "Python version: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" == "$REQUIRED_PYTHON"* ]]; then
    echo -e "${GREEN}✓${NC} Python 3.10.x detected"
else
    echo -e "${YELLOW}⚠${NC} Python $REQUIRED_PYTHON required (current: $PYTHON_VERSION)"
    ((WARNINGS++))
fi
echo ""

echo "2. PyTorch Installation"
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
        echo -e "${RED}✗${NC} CUDA not available (required for Evo-1)"
        ((FAILURES++))
    fi
else
    echo -e "${RED}✗${NC} PyTorch not installed"
    ((FAILURES++))
fi
echo ""

echo "3. Flash Attention (CRITICAL)"
echo "----------------------------"
if check_python_package flash_attn; then
    FLASH_VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "unknown")
    if [[ "$FLASH_VERSION" == "2.8.3"* ]]; then
        echo -e "${GREEN}✓${NC} Correct flash-attn version (2.8.3)"
    else
        echo -e "${YELLOW}⚠${NC} flash-attn 2.8.3 recommended (current: $FLASH_VERSION)"
        ((WARNINGS++))
    fi
    echo ""
    echo -e "${YELLOW}IMPORTANT:${NC} flash-attn must be installed with MAX_JOBS=64:"
    echo "  MAX_JOBS=64 pip install flash-attn --no-build-isolation"
else
    echo -e "${RED}✗ CRITICAL: flash-attn not installed${NC}"
    echo ""
    echo -e "${RED}Installation REQUIRED:${NC}"
    echo "  MAX_JOBS=64 pip install flash-attn --no-build-isolation"
    echo ""
    echo "This is a CRITICAL dependency for Evo-1!"
    ((FAILURES++))
fi
echo ""

echo "4. Core Dependencies"
echo "-------------------"
check_python_package transformers || ((FAILURES++))
check_python_package accelerate || ((FAILURES++))
check_python_package timm || ((FAILURES++))
check_python_package einops || ((FAILURES++))
check_python_package PIL || ((FAILURES++))
echo ""

echo "5. Vision-Language Backbone"
echo "---------------------------"
if python -c "import transformers" 2>/dev/null; then
    check_hf_model "OpenGVLab/InternVL3-1B" || ((WARNINGS++))
else
    echo -e "${RED}✗${NC} Cannot check HF models (transformers not installed)"
    ((FAILURES++))
fi
echo ""

echo "6. Evo-1 Model Checkpoints"
echo "-------------------------"
echo "Meta-World checkpoint:"
if check_hf_model "MINT-SJTU/Evo1_MetaWorld"; then
    echo -e "${GREEN}✓${NC} Meta-World checkpoint cached"
else
    echo -e "${YELLOW}⚠${NC} Meta-World checkpoint not cached"
    ((WARNINGS++))
fi

echo "LIBERO checkpoint:"
if check_hf_model "MINT-SJTU/Evo1_LIBERO"; then
    echo -e "${GREEN}✓${NC} LIBERO checkpoint cached"
else
    echo -e "${YELLOW}⚠${NC} LIBERO checkpoint not cached"
    ((WARNINGS++))
fi
echo ""

echo "7. Server Dependencies (for benchmark running)"
echo "----------------------------------------------"
check_python_package websockets || echo -e "${YELLOW}⚠${NC} websockets needed for server-client"
check_python_package opencv-python || echo -e "${YELLOW}⚠${NC} opencv-python needed for image processing"
check_python_package numpy || echo -e "${YELLOW}⚠${NC} numpy needed for array operations"
echo ""

echo "8. Benchmark Environments"
echo "------------------------"
echo "Meta-World setup (separate env required):"
echo "  - Python 3.10"
echo "  - mujoco, metaworld packages"
echo "  - Evo-1 client: mt50_evo1_client_prompt.py"
echo ""
echo "LIBERO setup (separate env required):"
echo "  - Python 3.8.13"
echo "  - LIBERO repository clone"
echo "  - Evo-1 client: libero_client_4tasks.py"
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
    echo "1. Start Evo-1 server: python scripts/Evo1_server.py"
    echo "2. Set up Meta-World env (Python 3.10) for MT50 benchmark"
    echo "3. Set up LIBERO env (Python 3.8.13) for LIBERO benchmark"
    echo ""
    echo "Performance:"
    echo "  - Meta-World MT50: 80.6% success (SOTA)"
    echo "  - LIBERO: 94.8% success (SOTA)"
    exit 0
else
    echo -e "${RED}✗ Critical dependencies missing!${NC}"
    echo ""
    echo "Installation Instructions:"
    echo "1. Create conda environment:"
    echo "   conda create -n Evo1 python=3.10 -y && conda activate Evo1"
    echo ""
    echo "2. Install flash-attn (CRITICAL - note MAX_JOBS=64):"
    echo "   MAX_JOBS=64 pip install flash-attn --no-build-isolation"
    echo ""
    echo "3. Install from requirements.txt:"
    echo "   pip install -r requirements.txt"
    echo ""
    echo "4. Or install packages manually:"
    echo "   pip install transformers accelerate timm einops pillow numpy opencv-python websockets"
    echo ""
    echo "5. Test installation:"
    echo "   python -c 'import torch; import flash_attn; print(f\"CUDA: {torch.cuda.is_available()}\")'"
    exit 1
fi
