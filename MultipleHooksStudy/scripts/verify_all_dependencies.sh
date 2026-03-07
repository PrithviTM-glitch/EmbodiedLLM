#!/bin/bash
# Master Dependency Verification Script
# Runs all individual model verification scripts and provides summary

set -e

echo "=========================================="
echo "VLA Models - Dependency Verification"
echo "=========================================="
echo ""
echo "Verifying dependencies for:"
echo "  1. RDT-1B (1.2B params)"
echo "  2. Evo-1 (0.77B params)"  
echo "  3. π0 (3.3B params)"
echo ""
echo "This will check Python versions, PyTorch, flash-attn,"
echo "HuggingFace models, and benchmark environments."
echo ""
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

MODELS_READY=0
MODELS_FAILED=0

# Run RDT verification
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MODEL 1/3: RDT-1B"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if bash "$SCRIPT_DIR/verify_rdt_dependencies.sh"; then
    echo -e "${GREEN}✓ RDT-1B ready${NC}"
    ((MODELS_READY++))
else
    echo -e "${RED}✗ RDT-1B not ready${NC}"
    ((MODELS_FAILED++))
fi

# Run Evo-1 verification
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MODEL 2/3: Evo-1"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if bash "$SCRIPT_DIR/verify_evo1_dependencies.sh"; then
    echo -e "${GREEN}✓ Evo-1 ready${NC}"
    ((MODELS_READY++))
else
    echo -e "${RED}✗ Evo-1 not ready${NC}"
    ((MODELS_FAILED++))
fi

# Run π0 verification
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MODEL 3/3: π0"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if bash "$SCRIPT_DIR/verify_pi0_dependencies.sh"; then
    echo -e "${GREEN}✓ π0 ready${NC}"
    ((MODELS_READY++))
else
    echo -e "${RED}✗ π0 not ready${NC}"
    ((MODELS_FAILED++))
fi

# Print summary
echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
echo -e "Models Ready:  ${GREEN}$MODELS_READY/3${NC}"
echo -e "Models Failed: ${RED}$MODELS_FAILED/3${NC}"
echo ""

if [ $MODELS_READY -eq 3 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✓ ALL MODELS READY FOR BENCHMARKING!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Benchmark Coverage:"
    echo "  LIBERO:     Evo-1 ✓ (94.8%), π0 ✓"
    echo "  Meta-World: Evo-1 ✓ (80.6%)"
    echo "  ManiSkill:  RDT ✓ (53.6%)"
    echo ""
    echo "Next Steps:"
    echo "  1. Set up LIBERO environment (Python 3.8.13)"
    echo "  2. Set up Meta-World environment (Python 3.10)"
    echo "  3. Start Evo-1 server for benchmark testing"
    echo "  4. Integrate hooks into server code"
    exit 0
elif [ $MODELS_READY -gt 0 ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}⚠ PARTIAL SETUP - $MODELS_READY/$((MODELS_READY + MODELS_FAILED)) models ready${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Re-run individual verification scripts to see details:"
    echo "  ./scripts/verify_rdt_dependencies.sh"
    echo "  ./scripts/verify_evo1_dependencies.sh"
    echo "  ./scripts/verify_pi0_dependencies.sh"
    exit 1
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}✗ NO MODELS READY${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Run individual scripts for installation instructions:"
    echo "  ./scripts/verify_rdt_dependencies.sh"
    echo "  ./scripts/verify_evo1_dependencies.sh"
    echo "  ./scripts/verify_pi0_dependencies.sh"
    exit 1
fi
