#!/bin/bash
# Cloud Setup Script for Meta-World Benchmark Environment
# Run this on cloud GPU instance AFTER setting up Evo-1
# Based on: https://github.com/MINT-SJTU/Evo-1

set -e

echo "=========================================="
echo "Meta-World Benchmark Environment Setup"
echo "=========================================="
echo ""
echo "Target: Cloud GPU instance"
echo "Python: 3.10 (compatible with Evo-1)"
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "1. Creating Meta-World conda environment..."
echo "--------------------------------------------"
conda create -n metaworld python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate metaworld

echo ""
echo "2. Installing MuJoCo..."
echo "-----------------------"
pip install mujoco

echo ""
echo "3. Installing Meta-World..."
echo "---------------------------"
pip install metaworld

echo ""
echo "4. Installing client dependencies..."
echo "--------------------------------------"
pip install websockets opencv-python numpy pillow torch

echo ""
echo "5. Testing Meta-World installation..."
echo "--------------------------------------"
python - <<EOF
import metaworld
import numpy as np

print(f"Meta-World version: {metaworld.__version__ if hasattr(metaworld, '__version__') else 'installed'}")

# Test MT50 benchmark
mt50 = metaworld.MT50()
print(f"MT50 tasks: {len(mt50.train_classes)} training tasks")
print("✓ Meta-World installed successfully!")

# List some tasks
print("\nSample tasks:")
for i, task_name in enumerate(list(mt50.train_classes.keys())[:5]):
    print(f"  {i+1}. {task_name}")
EOF

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Meta-World Environment Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Environment: metaworld (conda)"
echo "Python: 3.10"
echo ""
echo "Meta-World MT50 Benchmark:"
echo "  - 50 distinct manipulation tasks"
echo "  - Robotic arm manipulation"
echo "  - MuJoCo physics simulator"
echo ""
echo "Next Steps:"
echo "1. Ensure Evo-1 server is running (in Evo1 env)"
echo "2. Activate Meta-World env: conda activate metaworld"
echo "3. Run Meta-World client from Evo-1 repo:"
echo "   cd ~/Evo-1"
echo "   python Meta-World/mt50_evo1_client_prompt.py"
echo ""
echo "Expected Performance:"
echo "  - Evo-1 on Meta-World MT50: 80.6% success rate (SOTA)"
