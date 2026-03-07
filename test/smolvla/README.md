# SmolVLA — TC1 Cluster Deployment

## Overview

This directory contains everything needed to run the SmolVLA comprehensive study on the CCDS TC1 GPU cluster. The study consists of three experiments:

1. **Baseline Benchmarking**: Fine-tune + evaluate on LIBERO, MetaWorld, and VLABench
2. **Ablation Study (Pass0)**: Zero `state_proj` output and measure success rate drops
3. **Gradient Analysis**: Measure state encoder gradient contributions using flow matching loss

**Model**: `lerobot/smolvla_base` (450M parameters)  
**Framework**: LeRobot (HuggingFace)  
**GPU**: NVIDIA Tesla V100 32GB (TC1 standard)

---

## Files

| File | Purpose |
|------|---------|
| `smolvla_complete_job.sh` | Main SLURM job script — runs full 3-part study |
| `smolvla_validate.sh` | Pre-flight validation — checks env, GPU, imports |
| `README.md` | This file |

---

## Quick Start

### 1. Validate Environment (recommended first run)

```bash
# SSH into TC1 head node
ssh -l <username> 10.96.189.11

# Navigate to this directory
cd /tc1home/FYP/prithvi004/EmbodiedLLM/test/smolvla

# Submit validation job (takes ~5 min)
module load slurm
sbatch smolvla_validate.sh

# Check results
cat logs/output_SmolVLA_Validate_*.out
```

### 2. Submit Full Job

```bash
# Ensure logs directory exists
mkdir -p logs

# Submit the main job
sbatch smolvla_complete_job.sh

# Monitor
squeue -u $(whoami)
```

### 3. Check Results

```bash
# View SLURM output
cat logs/output_SmolVLA_Complete_*.out

# View error log (if any)
cat logs/error_SmolVLA_Complete_*.err

# Results are saved to:
ls -la /tc1home/FYP/prithvi004/EmbodiedLLM/MultipleHooksStudy/results/smolvla_complete/
```

---

## Resource Requirements

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | 1x V100 32GB | Required for model + fine-tuning |
| Memory | 32 GB | Model + dataset loading |
| CPUs | 8 | Data loading parallelism |
| Time | 360 min (6 hr) | Fine-tuning is compute-intensive |
| Disk | ~15 GB | Checkpoints + datasets + results |

### First Run Warning

The first run will:
- Create conda environment `smolvla_env` (~5 min)
- Clone LeRobot, LIBERO, MetaWorld, VLABench repos (~5 min)
- Install all dependencies (~10 min)
- Download `lerobot/smolvla_base` model (~2 GB)
- Fine-tune on LIBERO (100k steps) and MetaWorld (100k steps)

This may approach or exceed the 6-hour QoS limit. Options:
1. **Request extended QoS** (up to 48h) via CCDSgpu-tc@ntu.edu.sg
2. **Split into phases** — the script uses checkpoint caching, so resubmitting resumes where it left off

### Subsequent Runs

After the first run, conda env and repos persist. Fine-tuned checkpoints are cached. A rerun only evaluates (much faster).

---

## Environment Details

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | Latest (cu124) |
| CUDA | 12.4 |
| LeRobot | Latest (GitHub main) |
| Conda env name | `smolvla_env` |

---

## Directory Structure (after run)

```
MultipleHooksStudy/
├── workspace/smolvla/            # Ephemeral working directory
│   ├── lerobot/                  # LeRobot clone
│   ├── LIBERO/                   # LIBERO clone
│   ├── metaworld/                # MetaWorld clone
│   ├── VLABench/                 # VLABench clone
│   ├── checkpoints/              # Fine-tuned model checkpoints
│   ├── logs/                     # Runtime logs
│   └── data/                     # Gradient analysis data
├── results/smolvla_complete/     # Persistent results
│   ├── baseline/                 # Baseline success rates (JSON)
│   ├── ablation/                 # Ablation results (JSON)
│   ├── gradient/                 # Gradient plots (PNG) + data (JSON)
│   └── smolvla_complete_results.json  # Combined results
└── notebooks/
    └── smolvla_complete.ipynb    # Source notebook
```

---

## Debugging

### Common Issues

**1. "ModuleNotFoundError: No module named 'lerobot'"**
```bash
# Ensure conda env is activated in the job script
source /tc1apps/anaconda3/bin/activate smolvla_env
pip install -e /path/to/lerobot[smolvla]
```

**2. "CUDA out of memory"**
- Reduce batch size: change `LIBERO_BATCH = 4` → `2` in the notebook
- Enable gradient checkpointing (already enabled by default)

**3. Job times out (6h limit)**
- Checkpoints are cached — resubmit and it resumes
- Request extended QoS for initial run

**4. "state_proj not found" in hook attachment**
- The hook scanner searches multiple attribute names
- If SmolVLA API changed, check: `python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; p = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base'); print([n for n, _ in p.named_modules() if 'state' in n.lower()])"`

**5. VLABench import errors**
- VLABench is optional — baseline/ablation will still produce LIBERO and MetaWorld results
- Check: `pip install -e /path/to/VLABench`

### Log Files

| Log | Location |
|-----|----------|
| SLURM stdout | `test/smolvla/logs/output_SmolVLA_Complete_<jobid>.out` |
| SLURM stderr | `test/smolvla/logs/error_SmolVLA_Complete_<jobid>.err` |
| Runtime log | `MultipleHooksStudy/workspace/smolvla/logs/smolvla_complete_run.log` |

### Useful SLURM Commands

```bash
module load slurm

squeue -u $(whoami)              # Check job status
scancel <jobid>                  # Cancel job
seff <jobid>                     # Resource efficiency report
sacct -u $(whoami)               # Job history
tail -f logs/error_SmolVLA_Complete_*.err  # Live error log
```

---

## HuggingFace Token

If the model is gated, set your HF token before submitting:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXX
sbatch smolvla_complete_job.sh
```

Or uncomment the token line in `smolvla_complete_job.sh`.
