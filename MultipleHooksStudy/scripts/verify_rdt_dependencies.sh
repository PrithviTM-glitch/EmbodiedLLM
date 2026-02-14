#!/bin/bash
# RDT-1B Dependency Verification Script
# Checks all dependencies required for RDT-1B model
# Based on: https://github.com/thu-ml/RoboticsDiffusionTransformer

set -e

echo "=========================================="
echo "RDT-1B Dependency Verification"
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
        echo -e "${GREEN}✓${NC} $1 found"
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
    echo -e "${YELLOW}⚠${NC} Python $REQUIRED_PYTHON recommended (current: $PYTHON_VERSION)"
    ((WARNINGS++))
fi
echo ""

echo "2. PyTorch Installation"
echo "----------------------"
if check_python_package torch; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "PyTorch version: $TORCH_VERSION"
    if [[ "$TORCH_VERSION" == "2.1.0"* ]]; then
        echo -e "${GREEN}✓${NC} Correct PyTorch version (2.1.0)"
    else
        echo -e "${YELLOW}⚠${NC} PyTorch 2.1.0 recommended (current: $TORCH_VERSION)"
        ((WARNINGS++))
    fi
    
    # Check CUDA availability
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        echo -e "${GREEN}✓${NC} CUDA available (version: $CUDA_VERSION)"
    else
        echo -e "${RED}✗${NC} CUDA not available (required for RDT-1B)"
        ((FAILURES++))
    fi
else
    echo -e "${RED}✗${NC} PyTorch not installed"
    ((FAILURES++))
fi
echo ""

echo "3. Critical Dependencies"
echo "-----------------------"
# Flash Attention (CRITICAL for performance)
if check_python_package flash_attn; then
    echo -e "${GREEN}✓${NC} flash-attn installed (CRITICAL for performance)"
else
    echo -e "${RED}✗${NC} flash-attn not installed (CRITICAL)"
    echo "Install with: pip install flash-attn --no-build-isolation"
    ((FAILURES++))
fi

# Other critical packages
check_python_package transformers || ((FAILURES++))
check_python_package diffusers || ((FAILURES++))
check_python_package accelerate || ((FAILURES++))
check_python_package packaging || ((FAILURES++))
echo ""

echo "4. Vision & Language Encoders"
echo "-----------------------------"
# Check if transformers is available for HF checks
if python -c "import transformers" 2>/dev/null; then
    check_hf_model "google/t5-v1_1-xxl" || ((WARNINGS++))
    check_hf_model "google/siglip-so400m-patch14-384" || ((WARNINGS++))
else
    echo -e "${RED}✗${NC} Cannot check HF models (transformers not installed)"
    ((FAILURES++))
fi
echo ""

echo "5. Optional Dependencies"
echo "-----------------------"
check_python_package PIL || echo -e "${YELLOW}⚠${NC} Pillow recommended for image processing"
check_python_package numpy || echo -e "${YELLOW}⚠${NC} numpy recommended"
check_python_package einops || echo -e "${YELLOW}⚠${NC} einops recommended"
check_python_package opencv-python || echo -e "${YELLOW}⚠${NC} opencv-python recommended for video"
echo ""

echo "6. RDT Model Checkpoint"
echo "----------------------"
if check_hf_model "robotics-diffusion-transformer/rdt-1b"; then
    echo -e "${GREEN}✓${NC} RDT-1B checkpoint cached"
else
    echo -e "${YELLOW}⚠${NC} RDT-1B checkpoint not cached"
    echo "Will auto-download from: robotics-diffusion-transformer/rdt-1b"
    ((WARNINGS++))
fi
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
    echo "Installation Instructions (if warnings exist):"
    echo "1. Create conda environment: conda create -n rdt python=3.10.0 -y"
    echo "2. Install PyTorch: pip install torch==2.1.0 torchvision==0.16.0"
    echo "3. Install flash-attn: pip install flash-attn --no-build-isolation"
    echo "4. Install other deps: pip install transformers diffusers accelerate packaging pillow numpy einops opencv-python"
    echo "5. Models will auto-download on first use"
    exit 0
else
    echo -e "${RED}✗ Critical dependencies missing!${NC}"
    echo ""
    echo "Installation Instructions:"
    echo "1. Create conda environment: conda create -n rdt python=3.10.0 -y && conda activate rdt"
    echo "2. Install PyTorch: pip install torch==2.1.0 torchvision==0.16.0"
    echo "3. Install flash-attn: pip install flash-attn --no-build-isolation"
    echo "4. Install other deps: pip install transformers diffusers accelerate packaging pillow numpy einops opencv-python"
    echo "5. Test installation: python -c 'import torch; import flash_attn; print(torch.cuda.is_available())'"
    exit 1
fi
