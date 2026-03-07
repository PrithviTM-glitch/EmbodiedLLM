# Evo-1 — TC1 Cluster Deployment

## Overview

This directory contains everything needed to run the Evo-1 comprehensive study on the CCDS TC1 GPU cluster. The study consists of three experiments:

1. **Baseline Benchmarking**: Evaluate on LIBERO and MetaWorld
2. **Ablation Study (Pass0)**: Zero `state_encoder` output and measure success rate drops
3. **Gradient Analysis**: Measure state encoder gradient contributions using flow matching loss

**Model**: Evo-1 (0.77B parameters)  
**VLM Backbone**: InternVL3-1B (Flash Attention 2)  
**State Encoder**: `action_head.state_encoder` (CategorySpecificMLP, 3-layer MLP)  
**Framework**: Custom (MINT-SJTU/Evo-1)  
**GPU**: NVIDIA Tesla V100 32GB (TC1 standard)

---

## Architecture: Server-Client Design

Evo-1 uses a **WebSocket server-client architecture** requiring **3 separate conda environments**:

| Environment | Python | Purpose | Key Deps |
|-------------|--------|---------|----------|
| `evo1_server` | 3.10 | Model inference server | PyTorch 2.5.1+cu121, transformers==4.57.6 |
| `libero_client` | 3.8.13 | LIBERO simulation client | PyTorch 1.11+cu113, robosuite==1.4.1 |
| `metaworld_client` | 3.10 | MetaWorld simulation client | mujoco, metaworld, gymnasium |

The server runs model inference on GPU; clients run simulation environments and communicate via WebSocket.

**Port assignments:**

| Phase | Benchmark | Ports |
|-------|-----------|-------|
| Baseline | LIBERO | 9001–9010 |
| Baseline | MetaWorld | 9011–9020 |
| Ablation | LIBERO | 9021–9030 |
| Ablation | MetaWorld | 9031–9040 |

---

## Files

| File | Purpose |
|------|---------|
| `evo1_complete_job.sh` | Main SLURM job script — runs full 3-part study |
| `evo1_validate.sh` | Pre-flight validation — checks all 3 envs, GPU, imports |
| `README.md` | This file |

---

## Quick Start

### 1. Validate Environment (recommended first run)

```bash
cd /tc1home/FYP/prithvi004/EmbodiedLLM/test/evo1
mkdir -p logs
module load slurm
sbatch evo1_validate.sh
cat logs/output_Evo1_Validate_*.out
```

### 2. Submit Full Job

```bash
sbatch evo1_complete_job.sh
squeue -u $(whoami)
```

### 3. Check Results

```bash
cat logs/output_Evo1_Complete_*.out
cat logs/error_Evo1_Complete_*.err
ls -la /tc1home/FYP/prithvi004/EmbodiedLLM/MultipleHooksStudy/results/evo1_complete/
```

---

## Resource Requirements

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | 1x V100 32GB | Model + flash-attn |
| Memory | 32 GB | 3 envs + model loading |
| CPUs | 8 | Server + parallel clients |
| Time | 360 min (6 hr) | **Will likely exceed on first run** |
| Disk | ~25 GB | 3 conda envs + checkpoints + repos |

### First Run Warning — CRITICAL

The first run will:
- Create **3 conda environments** (~15 min each)
- Compile flash-attn from source (~10-15 min)
- Clone Evo-1 and LIBERO repos
- Download model checkpoints from HuggingFace (~4 GB)

**This will almost certainly exceed the 6-hour QoS limit on first run.**

**Recommended approach:**
1. **Request extended QoS** (up to 48h) via CCDSgpu-tc@ntu.edu.sg
2. **Or run in phases**: Submit, let it create envs, resubmit for evaluation
   - The script is idempotent — existing envs/repos are reused

### Subsequent Runs

After envs and repos are created, the job focuses on evaluation (~2-3 hours).

---

## Critical Dependency Notes

### transformers Version

**MUST be 4.57.6, NOT 5.x.** Version 5.0.0 causes meta tensor issues with InternVL3 initialization.

```bash
# Verify in evo1_server:
conda run -n evo1_server python -c "import transformers; print(transformers.__version__)"
# Must print 4.57.6
```

### flash-attn

Optional but recommended for performance. Compiles from source (~10-15 min). If compilation fails on TC1, the model falls back to standard attention (slower but functional).

---

## Directory Structure (after run)

```
MultipleHooksStudy/
├── workspace/evo1/               # Ephemeral working directory
│   ├── Evo-1/                    # Evo-1 repo clone
│   ├── LIBERO/                   # LIBERO repo clone
│   ├── checkpoints/              # Model checkpoints (HuggingFace)
│   │   ├── libero/               # MINT-SJTU/Evo1_LIBERO
│   │   └── metaworld/            # MINT-SJTU/Evo1_MetaWorld
│   ├── logs/                     # Server/client runtime logs
│   └── data/                     # Gradient analysis data
├── results/evo1_complete/        # Persistent results
│   ├── baseline/                 # Baseline success rates (JSON)
│   ├── ablation/                 # Ablation results (JSON)
│   ├── gradient/                 # Gradient plots (PNG) + data (JSON)
│   ├── MetaWorld/                # Per-benchmark results
│   └── LIBERO/                   # Per-benchmark results
└── notebooks/
    └── evo1_complete.ipynb       # Source notebook
```

---

## Debugging

### Common Issues

**1. "transformers 5.x meta tensor error"**
```bash
conda run -n evo1_server pip install transformers==4.57.6
```

**2. "flash-attn build failed"**
- Not critical — model falls back to standard attention
- Ensure CUDA 12.1 is loaded: `module load cuda/12.1`
- May need: `conda run -n evo1_server pip install flash-attn --no-build-isolation`

**3. "WebSocket connection refused"**
- Port conflict. Check: `netstat -tlnp | grep 900`
- Kill stale processes: `kill $(lsof -t -i:9001)` (if accessible)
- The notebook uses ports 9001-9040 — ensure no other jobs use them

**4. "state_encoder not found"**
- Evo-1's state encoder lives at `model.action_head.state_encoder`
- It's a `CategorySpecificMLP` (3-layer MLP), not a simple Linear
- Check: `conda run -n evo1_server python -c "from Evo_1.model.action_head.flow_matching import FlowmatchingActionHead; ..."`

**5. "LIBERO/robosuite ImportError"**
- libero_client requires Python 3.8 + old PyTorch — don't mix with evo1_server
- Check: `conda run -n libero_client python -c "import robosuite; print(robosuite.__version__)"`

**6. Job times out (6h limit)**
- Environments are cached — resubmit and it continues from where it left off
- Request extended QoS for the initial setup run

**7. "ModuleNotFoundError: No module named 'hooks'"**
- Ensure EmbodiedLLM is on correct branch: `cd /tc1home/FYP/prithvi004/EmbodiedLLM && git checkout AnalyseMultipleHooks`
- Check sys.path includes MHS_DIR in the patched script

### Log Files

| Log | Location |
|-----|----------|
| SLURM stdout | `test/evo1/logs/output_Evo1_Complete_<jobid>.out` |
| SLURM stderr | `test/evo1/logs/error_Evo1_Complete_<jobid>.err` |
| Runtime log | `MultipleHooksStudy/workspace/evo1/logs/evo1_complete_run.log` |

### Useful SLURM Commands

```bash
module load slurm
squeue -u $(whoami)              # Check job status
scancel <jobid>                  # Cancel job
seff <jobid>                     # Resource efficiency report
sacct -u $(whoami)               # Job history
```

### Manual Environment Cleanup

If you need to recreate environments:

```bash
module load anaconda
conda env remove -n evo1_server -y
conda env remove -n libero_client -y
conda env remove -n metaworld_client -y
# Then resubmit the job
```

---

## HuggingFace Checkpoints

The following checkpoints are downloaded automatically:
- `MINT-SJTU/Evo1_LIBERO` → `workspace/evo1/checkpoints/libero/`
- `MINT-SJTU/Evo1_MetaWorld` → `workspace/evo1/checkpoints/metaworld/`

If gated, set token before submission:
```bash
export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXX
sbatch evo1_complete_job.sh
```
