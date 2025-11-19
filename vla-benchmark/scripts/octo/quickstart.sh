#!/bin/bash
# Quick start script for OCTO OpenX benchmark
# This script runs a series of tests and a quick benchmark evaluation

set -e  # Exit on error

echo "================================================================================"
echo "OCTO OpenX Benchmark - Quick Start"
echo "================================================================================"
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "octo" ]]; then
    echo "⚠️  Warning: 'octo' conda environment is not activated"
    echo "Please run: conda activate octo"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Change to script directory
cd "$(dirname "$0")"

echo "Step 1: Testing framework installation"
echo "--------------------------------------------------------------------------------"
python test_framework.py
if [ $? -ne 0 ]; then
    echo "❌ Framework test failed. Please check your installation."
    exit 1
fi
echo ""

echo "Step 2: Running quick benchmark (5 episodes)"
echo "--------------------------------------------------------------------------------"
echo "This will:"
echo "  1. Load OCTO-small model from HuggingFace"
echo "  2. Setup OpenX benchmark (Bridge V2 dataset)"
echo "  3. Evaluate on 5 episodes"
echo "  4. Save results to ../../results/openx/"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping benchmark run. You can run manually:"
    echo "  python run_openx_benchmark.py --max-episodes 5"
    exit 0
fi

python run_openx_benchmark.py --max-episodes 5

echo ""
echo "================================================================================"
echo "Quick start completed!"
echo "================================================================================"
echo ""
echo "What's next?"
echo ""
echo "1. Run longer evaluation:"
echo "   python run_openx_benchmark.py --max-episodes 100"
echo ""
echo "2. Try different model:"
echo "   python run_openx_benchmark.py --model hf://rail-berkeley/octo-base-1.5"
echo ""
echo "3. Check results:"
echo "   ls -lh ../../results/openx/"
echo ""
echo "4. Read documentation:"
echo "   cat ../../docs/openx_summary.md"
echo "   cat ../../docs/architecture.md"
echo ""
echo "================================================================================"
