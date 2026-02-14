#!/bin/bash
# Cloud Setup Script for Evo-1
# Run this on cloud GPU instance to set up Evo-1 environment
# Based on: https://github.com/MINT-SJTU/Evo-1

set -e

echo "=========================================="
echo "Evo-1 Cloud Environment Setup"
echo "=========================================="
echo ""
echo "Target: Cloud GPU instance (A100/H100/V100)"
echo "Python: 3.10"
echo "CUDA: Required"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running in cloud (has nvidia-smi)
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Are you running on a GPU instance?"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "1. Creating Evo-1 conda environment..."
echo "--------------------------------------"
conda create -n Evo1 python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Evo1

echo ""
echo "2. Installing PyTorch with CUDA..."
echo "-----------------------------------"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "3. Installing flash-attn (CRITICAL - this takes 5-10 minutes)..."
echo "-----------------------------------------------------------------"
echo "Using MAX_JOBS=64 for faster compilation"
MAX_JOBS=64 pip install flash-attn --no-build-isolation

echo ""
echo "4. Installing core dependencies..."
echo "-----------------------------------"
pip install transformers accelerate timm einops pillow numpy opencv-python websockets

echo ""
echo "5. Downloading Evo-1 repository..."
echo "-----------------------------------"
cd ~
if [ ! -d "Evo-1" ]; then
    git clone https://github.com/MINT-SJTU/Evo-1.git
    cd Evo-1
else
    echo "Evo-1 repository already exists"
    cd Evo-1
    git pull
fi

echo ""
echo "6. Downloading VL backbone (InternVL3-1B)..."
echo "--------------------------------------------"
python - <<EOF
from transformers import AutoModel
print("Downloading InternVL3-1B...")
model = AutoModel.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True)
print("✓ InternVL3-1B cached")
EOF

echo ""
echo "7. Downloading Evo-1 checkpoints..."
echo "------------------------------------"
python - <<EOF
from huggingface_hub import snapshot_download
import os

print("Downloading Meta-World checkpoint...")
snapshot_download("MINT-SJTU/Evo1_MetaWorld", local_dir=os.path.expanduser("~/.cache/evo1/metaworld"))
print("✓ Meta-World checkpoint cached")

print("Downloading LIBERO checkpoint...")
snapshot_download("MINT-SJTU/Evo1_LIBERO", local_dir=os.path.expanduser("~/.cache/evo1/libero"))
print("✓ LIBERO checkpoint cached")
EOF

echo ""
echo "8. Testing installation..."
echo "--------------------------"
python - <<EOF
import torch
import flash_attn
import transformers

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"flash-attn: {flash_attn.__version__}")
print(f"transformers: {transformers.__version__}")
print("✓ All packages working!")
EOF

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Evo-1 Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Environment: Evo1 (conda)"
echo "Repository: ~/Evo-1"
echo "Checkpoints:"
echo "  - Meta-World: ~/.cache/evo1/metaworld"
echo "  - LIBERO: ~/.cache/evo1/libero"
echo ""
echo "Next Steps:"
echo "1. Activate environment: conda activate Evo1"
echo "2. Start server: cd ~/Evo-1 && python scripts/Evo1_server.py"
echo "3. Run Meta-World benchmark (separate env needed)"
echo "4. Run LIBERO benchmark (separate env needed)"
echo ""
echo "Performance expectations:"
echo "  - Meta-World MT50: 80.6% success rate"
echo "  - LIBERO: 94.8% success rate"
