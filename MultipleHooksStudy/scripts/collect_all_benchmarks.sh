#!/bin/bash
# Collect real observations from all benchmarks for gradient analysis

set -e  # Exit on error

# Configuration
OUTPUT_DIR="/content/benchmark_observations"
NUM_SAMPLES=50
SEED=42

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Benchmark Data Collection${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Output directory: $OUTPUT_DIR"
echo "Samples per benchmark: $NUM_SAMPLES"
echo "Random seed: $SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Collect LIBERO observations
echo -e "${GREEN}[1/3] Collecting LIBERO observations...${NC}"
python scripts/data_collectors/libero_collector.py \
    --benchmark libero_90 \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ LIBERO collection complete${NC}"
else
    echo "✗ LIBERO collection failed (skipping)"
fi
echo ""

# Collect MetaWorld observations
echo -e "${GREEN}[2/3] Collecting MetaWorld observations...${NC}"
python scripts/data_collectors/metaworld_collector.py \
    --benchmark ML10 \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ MetaWorld collection complete${NC}"
else
    echo "✗ MetaWorld collection failed (skipping)"
fi
echo ""

# Collect Bridge observations (optional - only for Pi0 and RDT-1B)
echo -e "${GREEN}[3/3] Collecting Bridge observations...${NC}"
python scripts/data_collectors/bridge_collector.py \
    --dataset bridge_dataset \
    --split "train[:1000]" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Bridge collection complete${NC}"
else
    echo "✗ Bridge collection failed (skipping)"
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Collection Summary${NC}"
echo -e "${BLUE}========================================${NC}"
ls -lh "$OUTPUT_DIR"/*.h5 2>/dev/null || echo "No HDF5 files found"
echo ""

# Print file sizes
echo "Total disk usage:"
du -h "$OUTPUT_DIR" 2>/dev/null || echo "0 bytes"
echo ""

echo -e "${GREEN}✓ Data collection complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Run gradient analysis with real data:"
echo "   python scripts/run_evo1_gradient_analysis.py --data-path $OUTPUT_DIR/libero_libero_90_seed42_${NUM_SAMPLES}samples.h5 --num-samples $NUM_SAMPLES"
echo ""
echo "2. Compare with synthetic data:"
echo "   python scripts/run_evo1_gradient_analysis.py"
