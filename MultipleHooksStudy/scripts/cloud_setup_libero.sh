#!/bin/bash
# Cloud Setup Script for LIBERO Benchmark Environment
# Run this on cloud GPU instance AFTER setting up Evo-1
# Based on: https://github.com/MINT-SJTU/Evo-1

set -e

echo "=========================================="
echo "LIBERO Benchmark Environment Setup"
echo "=========================================="
echo ""
echo "Target: Cloud GPU instance"
echo "Python: 3.8.13 (REQUIRED for LIBERO)"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "1. Creating LIBERO conda environment..."
echo "----------------------------------------"
conda create -n libero python=3.8.13 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate libero

echo ""
echo "2. Installing PyTorch 1.11.0 with CUDA 11.3..."
echo "-----------------------------------------------"
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo ""
echo "3. Cloning LIBERO repository..."
echo "--------------------------------"
cd ~
mkdir -p LIBERO_evaluation
cd LIBERO_evaluation

if [ ! -d "LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
    cd LIBERO
else
    echo "LIBERO repository already exists"
    cd LIBERO
    git pull
fi

echo ""
echo "4. Installing LIBERO dependencies..."
echo "-------------------------------------"
pip install -e .

echo ""
echo "5. Installing client dependencies..."
echo "--------------------------------------"
pip install websockets opencv-python numpy pillow

echo ""
echo "6. Testing LIBERO installation..."
echo "----------------------------------"
python - <<EOF
import torch
import libero
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("✓ LIBERO installed successfully!")
EOF

echo ""
echo "7. Setting up LIBERO benchmark tasks..."
echo "----------------------------------------"
echo "LIBERO includes 4 task suites:"
echo "  - LIBERO-Spatial (10 tasks)"
echo "  - LIBERO-Object (10 tasks)"
echo "  - LIBERO-Goal (10 tasks)"
echo "  - LIBERO-Long (10 tasks)"
echo ""

echo ""
echo -e "${GREEN}=========================================="
echo "✓ LIBERO Environment Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Environment: libero (conda)"
echo "Python: 3.8.13"
echo "Repository: ~/LIBERO_evaluation/LIBERO"
echo ""
echo "Next Steps:"
echo "1. Ensure Evo-1 server is running (in Evo1 env)"
echo "2. Activate LIBERO env: conda activate libero"
echo "3. Run LIBERO client from Evo-1 repo:"
echo "   cd ~/Evo-1"
echo "   python LIBERO_evaluation/libero_client_4tasks.py"
echo ""
echo "Expected Performance:"
echo "  - Evo-1 on LIBERO: 94.8% success rate (SOTA)"
