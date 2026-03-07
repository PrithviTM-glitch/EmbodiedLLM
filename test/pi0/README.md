# Pi0 — TC1 Cluster Deployment

## Overview

This directory contains everything needed to run the Pi0 comprehensive study on the CCDS TC1 GPU cluster. The study consists of three experiments:

1. **Baseline Benchmarking**: Evaluate on LIBERO, MetaWorld, VLABench
2. **Ablation Study (Pass0)**: Zero `state_proj` output and measure success rate drops
3. **Gradient Analysis**: Measure state encoder gradient contributions using flow matching loss

**Model**: Pi0 / Pi0.5-DROID (3.3B parameters)  
**Framework**: openpi (Physical Intelligence) + LeRobot  
**Architecture**: PaliGemma VLM + Expert Gemma + Flow Matching  
**GPU**: NVIDIA Tesla V100 32GB (TC1 standard)

---

## Files

| File | Purpose |
|------|---------|
| `pi0_complete_job.sh` | Main SLURM job script — runs full 3-part study |
| `pi0_validate.sh` | Pre-flight validation — checks env, GPU, imports |
| `README.md` | This file |

---

## Quick Start

### 1. Validate Environment (recommended first run)

```bash
cd /tc1home/FYP/prithvi004/EmbodiedLLM/test/pi0
mkdir -p logs
module load slurm
sbatch pi0_validate.sh
cat logs/output_Pi0_Validate_*.out
```

### 2. Set HuggingFace Token (if model is gated)

```bash
export HUGGING_FACE_HUB_TOKEN=hf_XXXXXXXX
```

Or edit `pi0_complete_job.sh` and uncomment the token line.

### 3. Submit Full Job

```bash
sbatch pi0_complete_job.sh
squeue -u $(whoami)
```

### 4. Check Results

```bash
cat logs/output_Pi0_Complete_*.out
cat logs/error_Pi0_Complete_*.err
ls -la /tc1home/FYP/prithvi004/EmbodiedLLM/MultipleHooksStudy/results/pi0_complete/
```

---

## Resource Requirements

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | 1x V100 32GB | Pi0 is 3.3B params — fills most of V100 |
| Memory | 32 GB | Model + JAX runtime |
| CPUs | 8 | Data loading + openpi server |
| Time | 360 min (6 hr) | May need extended QoS for first run |
| Disk | ~20 GB | Checkpoints + openpi + repos |

### Pi0-Specific Dependencies

Pi0 uses a unique server-client architecture:
- **Server**: JAX-based inference server (openpi) — handles model forward passes
- **Client**: Python clients for LIBERO/MetaWorld/VLABench — handles simulation
- **Communication**: WebSocket-based request/response

This means the job installs both PyTorch and JAX ecosystems.

### First Run Warning

The first run will:
- Create conda environment `pi0_env` (~5 min)
- Clone openpi, VLABench repos (~5 min)
- Install heavy JAX + openpi dependencies (~15 min)
- Download Pi0 model checkpoint (~6 GB)

Consider requesting extended QoS (up to 48h) for the initial run.

---

## Environment Details

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | Latest (cu124) |
| CUDA | 12.4 |
| JAX | Latest (for openpi server) |
| LeRobot | Latest with Pi0 extras |
| Conda env name | `pi0_env` |

---

## Directory Structure (after run)

```
MultipleHooksStudy/
├── workspace/pi0/                # Ephemeral working directory
│   ├── openpi/                   # Physical-Intelligence/openpi clone
│   ├── openpi_vlabench/          # Shiduo-zh/openpi fork (VLABench ckpt)
│   ├── VLABench/                 # VLABench clone
│   ├── ckpt_pi0_vlabench/        # VLABench checkpoint
│   └── logs/                     # Server/client runtime logs
├── results/pi0_complete/         # Persistent results
│   ├── baseline/                 # Baseline success rates (JSON)
│   ├── ablation/                 # Ablation results (JSON)
│   └── gradient/                 # Gradient plots (PNG) + data (JSON)
└── notebooks/
    └── pi0_complete.ipynb        # Source notebook
```

---

## Debugging

### Common Issues

**1. "JAX out of memory" or "XLA compilation failed"**
- V100 has 32GB; Pi0 (3.3B) is tight. The job sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`
- If still failing, try reducing batch size

**2. "openpi server failed to start"**
- Check `workspace/pi0/logs/` for server logs
- Ensure `uv` is installed: `pip install uv`
- Try manual: `cd openpi && uv sync && uv run openpi.server ...`

**3. "state_proj not found"**
- Pi0 uses `state_proj` (Linear layer) in the model root
- If API changed: `python -c "from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy; ..."`

**4. "HuggingFace 401 Unauthorized"**
- Set `export HUGGING_FACE_HUB_TOKEN=hf_XXXX` before `sbatch`
- Or `huggingface-cli login` in the conda env

**5. WebSocket connection refused**
- Server/client port conflicts. Check that ports 9001-9040 are free
- `netstat -tp | grep 900`

### Log Files

| Log | Location |
|-----|----------|
| SLURM stdout | `test/pi0/logs/output_Pi0_Complete_<jobid>.out` |
| SLURM stderr | `test/pi0/logs/error_Pi0_Complete_<jobid>.err` |
| Runtime log | `MultipleHooksStudy/workspace/pi0/logs/pi0_complete_run.log` |

### Useful SLURM Commands

```bash
module load slurm
squeue -u $(whoami)              # Check job status
scancel <jobid>                  # Cancel job
seff <jobid>                     # Resource efficiency report
sacct -u $(whoami)               # Job history
```
