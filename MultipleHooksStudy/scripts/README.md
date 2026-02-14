# Dependency Verification Scripts

Automated scripts to verify dependencies for all VLA models before running benchmarks.

## Quick Start

Run all verifications at once:
```bash
./scripts/verify_all_dependencies.sh
```

Or run individual model checks:
```bash
./scripts/verify_rdt_dependencies.sh    # RDT-1B dependencies
./scripts/verify_evo1_dependencies.sh   # Evo-1 dependencies
./scripts/verify_pi0_dependencies.sh    # π0 dependencies
```

## What Gets Checked

### RDT-1B (`verify_rdt_dependencies.sh`)
- ✓ Python 3.10.x environment
- ✓ PyTorch 2.1.0 with CUDA
- ✓ flash-attn (critical for performance)
- ✓ Core packages: transformers, diffusers, accelerate
- ✓ Vision encoder: `google/siglip-so400m-patch14-384`
- ✓ Language encoder: `google/t5-v1_1-xxl`
- ✓ Model checkpoint: `robotics-diffusion-transformer/rdt-1b`

**Critical Notes:**
- Unified action space uses 6D rotation (NOT Euler angles)
- flash-attn must be installed with `--no-build-isolation`
- No server-client (runs standalone)

### Evo-1 (`verify_evo1_dependencies.sh`)
- ✓ Python 3.10 environment
- ✓ PyTorch with CUDA
- ✓ **flash-attn 2.8.3** (CRITICAL - must use `MAX_JOBS=64`)
- ✓ Core packages: transformers, accelerate, timm, einops
- ✓ VL backbone: `OpenGVLab/InternVL3-1B`
- ✓ Meta-World checkpoint: `MINT-SJTU/Evo1_MetaWorld`
- ✓ LIBERO checkpoint: `MINT-SJTU/Evo1_LIBERO`
- ✓ Server dependencies: websockets, opencv-python

**Critical Notes:**
- flash-attn installation: `MAX_JOBS=64 pip install flash-attn --no-build-isolation`
- Has BOTH LIBERO (94.8%) AND Meta-World (80.6%) - only model with both!
- Server-client architecture already implemented

### π0 (`verify_pi0_dependencies.sh`)
- ✓ Python 3.8+ environment
- ✓ uv package manager (recommended)
- ✓ PyTorch (JAX→PyTorch conversion support)
- ✓ Core packages: transformers, einops, numpy
- ✓ Policy server: flask, requests
- ✓ Available checkpoints: pi0_base, pi05_libero, pi0_droid

**Critical Notes:**
- Recently converted from JAX to PyTorch
- Checkpoints hosted on Google Cloud Storage
- Separate multi-layer proprio encoder (NOT fused into VL)
- Policy server already implemented

## Output

Each script returns:
- **Exit code 0**: All critical dependencies satisfied ✓
- **Exit code 1**: Critical dependencies missing ✗

Output includes:
- ✓ Green checkmarks: Dependency installed
- ✗ Red X marks: Missing critical dependency
- ⚠ Yellow warnings: Optional or cached dependencies

## Installation Instructions

If verification fails, each script provides complete installation instructions.

### Example: RDT-1B Setup
```bash
conda create -n rdt python=3.10.0 -y && conda activate rdt
pip install torch==2.1.0 torchvision==0.16.0
pip install flash-attn --no-build-isolation
pip install transformers diffusers accelerate packaging pillow numpy einops opencv-python
```

### Example: Evo-1 Setup
```bash
conda create -n Evo1 python=3.10 -y && conda activate Evo1
MAX_JOBS=64 pip install flash-attn --no-build-isolation  # CRITICAL step!
pip install transformers accelerate timm einops pillow numpy opencv-python websockets
```

### Example: π0 Setup
```bash
# Option 1: Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install torch transformers einops pillow numpy flask requests

# Option 2: Using pip
conda create -n pi0 python=3.10 -y && conda activate pi0
pip install torch transformers einops pillow numpy flask requests
```

## Benchmark Coverage Summary

| Model | LIBERO | Meta-World | ManiSkill | Server-Client |
|-------|--------|------------|-----------|---------------|
| **Evo-1** | ✓ 94.8% | ✓ 80.6% | ✗ | ✓ Implemented |
| **π0** | ✓ Checkpoint | ✗ | ✗ | ✓ Policy server |
| **RDT-1B** | ✗ | ✗ | ✓ 53.6% | ✗ Need to build |

**Key Insight**: Evo-1 is the only model with BOTH LIBERO + Meta-World benchmarks!

## Next Steps After Verification

1. **Set up LIBERO environment** (Python 3.8.13):
   ```bash
   conda create -n libero python=3.8.13 -y
   cd LIBERO_evaluation/
   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
   pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

2. **Set up Meta-World environment** (Python 3.10):
   ```bash
   conda create -n metaworld python=3.10 -y
   pip install mujoco metaworld websockets opencv-python
   ```

3. **Start Evo-1 server** (for LIBERO/Meta-World):
   ```bash
   conda activate Evo1
   python scripts/Evo1_server.py  # Websocket server on port 8000
   ```

4. **Integrate hooks**:
   - Modify server code to inject hook managers
   - Capture gradient flow during benchmark runs
   - Export analysis data for notebooks

## Troubleshooting

### flash-attn installation fails
```bash
# Use MAX_JOBS to limit parallelism
MAX_JOBS=64 pip install flash-attn --no-build-isolation

# If still fails, reduce MAX_JOBS based on available RAM
MAX_JOBS=32 pip install flash-attn --no-build-isolation
```

### CUDA not available
```bash
# Check CUDA installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### HuggingFace models not cached
Models will auto-download on first use. To pre-download:
```python
from transformers import AutoModel
AutoModel.from_pretrained("google/t5-v1_1-xxl")  # For RDT
AutoModel.from_pretrained("OpenGVLab/InternVL3-1B")  # For Evo-1
```

## Related Documentation

- [CODEBASE_ANALYSIS.md](../docs/CODEBASE_ANALYSIS.md) - Detailed model analysis
- [RDT-1B Repository](https://github.com/thu-ml/RoboticsDiffusionTransformer)
- [Evo-1 Repository](https://github.com/MINT-SJTU/Evo-1)
- [π0 Repository](https://github.com/Physical-Intelligence/openpi)
