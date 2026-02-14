# Cloud Deployment Guide for VLA Benchmark Experiments

Complete guide for running VLA model benchmarks on cloud GPU instances.

## 🎯 Quick Start

### Prerequisites
- Cloud GPU instance (A100/H100/V100 recommended)
- CUDA 12.1+
- ~100GB disk space
- ~20GB GPU memory

### 1-Minute Setup
```bash
# Clone repository
git clone https://github.com/PrithviTM-glitch/EmbodiedLLM.git
cd EmbodiedLLM/MultipleHooksStudy

# Run Evo-1 setup (includes all dependencies)
bash scripts/cloud_setup_evo1.sh

# Setup benchmarks
bash scripts/cloud_setup_libero.sh
bash scripts/cloud_setup_metaworld.sh
```

## 📊 Complete Workflow

### Option 1: Using Notebooks (Recommended for Beginners)

**Step 1: Evo-1 Setup and Analysis**
```bash
jupyter notebook notebooks/evo1_complete.ipynb
```
- Part 1: Model analysis with hooks
- Part 2: Cloud environment setup (Python 3.10)
- Part 3: LIBERO + Meta-World benchmarks (integrated)
- Downloads checkpoints (~10GB)
- **All-in-one notebook** (no separate server needed)

**Expected Results**:
- LIBERO: **94.8% success rate** (4 task suites)
- Meta-World MT50: **80.6% success rate** (50 tasks)
- Hook data collected for all episodes
- Runs on Cloud GPU or Google Colab

**Alternative Models**:
```bash
# π0 (3.3B) - All benchmarks integrated
jupyter notebook notebooks/pi0_complete.ipynb

# RDT-1B (1.2B) - Includes ManiSkill + RoboTwin
jupyter notebook notebooks/rdt_1b_complete.ipynb
```
- Expected: **80.6% success rate**

### Option 2: Using Scripts (For Cloud Automation)

**Terminal 1: Start Evo-1 Server**
```bash
conda activate Evo1
python evo1_server_with_hooks.py --benchmark metaworld --port 8000
```

**Terminal 2: Run LIBERO Benchmark**
```bash
conda activate libero
python libero_client_with_hooks.py
```

**Terminal 3: Run Meta-World Benchmark**
```bash
conda activate metaworld
python metaworld_client_with_hooks.py
```

## 🧪 Model Coverage & Benchmarks

| Model | Size | LIBERO | Meta-World | ManiSkill | Server-Client | Status |
|-------|------|--------|------------|-----------|---------------|--------|
| **Evo-1** | 0.77B | ✅ 94.8% | ✅ 80.6% | ❌ | ✅ Implemented | **Priority** |
| **π0** | 3.3B | ✅ Checkpoint | ❌ | ❌ | ✅ Policy server | Ready |
| **RDT-1B** | 1.2B | ❌ | ❌ | ✅ 53.6% | ❌ Need to build | Planned |

**Strategic Decision**: Focus on Evo-1 first (only model with BOTH LIBERO + Meta-World)

## 🔧 Environment Details

### Evo-1 Server Environment
```yaml
Name: Evo1
Python: 3.10
Key Dependencies:
  - PyTorch (latest with CUDA 12.1)
  - flash-attn==2.8.3 (CRITICAL - use MAX_JOBS=64)
  - InternVL3-1B (VL backbone)
  - transformers, accelerate, timm, einops
Checkpoints:
  - MINT-SJTU/Evo1_MetaWorld (~5GB)
  - MINT-SJTU/Evo1_LIBERO (~5GB)
```

### LIBERO Benchmark Environment
```yaml
Name: libero
Python: 3.8.13 (REQUIRED - newer versions incompatible)
Key Dependencies:
  - PyTorch 1.11.0+cu113
  - LIBERO repository (clone from GitHub)
  - websockets, opencv-python
Tasks: 4 suites × 10 tasks = 40 tasks
Time: ~2-3 hours (50 episodes per task)
```

### Meta-World Benchmark Environment
```yaml
Name: metaworld
Python: 3.10
Key Dependencies:
  - mujoco, metaworld
  - websockets, opencv-python
  - torch (for client processing)
Tasks: 50 manipulation tasks (MT50)
Time: ~3-4 hours (5 episodes per task)
```

## 🚀 Performance Expectations

### Evo-1 (Our Primary Model)
- **LIBERO**: 94.8% success (SOTA among VLAs)
- **Meta-World MT50**: 80.6% success (SOTA among VLAs)
- **Advantage**: Two-stage training preserves semantic attention
- **Unique**: Integration module aligns VL + proprio without semantic drift

### π0 (Alternative, Larger Model)
- **LIBERO**: Checkpoint available (pi05_libero)
- **Advantage**: Separate multi-layer proprio encoder
- **Challenge**: JAX→PyTorch conversion, GCS checkpoints

### RDT-1B (Alternative, Diffusion-Based)
- **ManiSkill**: 53.6% success
- **RoboTwin**: 2nd place on leaderboard
- **Advantage**: Unified action space, 6D rotation
- **Challenge**: No LIBERO/Meta-World benchmarks

## 📁 Output Structure

After running benchmarks, you'll have:

```
MultipleHooksStudy/
├── libero_results/
│   ├── libero_spatial_results.json      # Task-level results
│   ├── libero_object_results.json
│   ├── libero_goal_results.json
│   ├── libero_long_results.json
│   ├── hook_data_all.pkl                # Hook data for analysis
│   └── success_rates.png                # Visualization
│
├── metaworld_results/
│   ├── mt50_results.json                # All 50 tasks
│   ├── task_success_rates.json          # Per-task breakdown
│   ├── hook_data_all.pkl                # Hook data for analysis
│   ├── mt50_analysis.png                # Distribution + top/bottom tasks
│   └── benchmark_comparison.png         # LIBERO vs Meta-World
│
└── notebooks/
    ├── evo1_complete.ipynb              # Evo-1: Analysis + Cloud + Benchmarks
    ├── pi0_complete.ipynb               # π0: Analysis + Cloud + Benchmarks
    └── rdt_1b_complete.ipynb            # RDT-1B: Analysis + Cloud + Benchmarks
```

**Note**: All notebooks are **self-contained** - benchmarks run in-process without separate servers. Perfect for Colab!

## 🔬 Hook Data Collection

All benchmarks automatically collect:
- **Integration module outputs**: Aligned VL + proprio representations
- **Action gradients**: Gradient flow to action decoder
- **Attention patterns**: Semantic preservation analysis
- **Episode statistics**: Success/failure, step count, rewards

Use analysis notebooks to visualize:
```python
import pickle
with open('libero_results/hook_data_all.pkl', 'rb') as f:
    hook_data = pickle.load(f)

# Analyze integration module
for episode in hook_data:
    for step in episode:
        integration_output = step['integration_module_output']
        action_gradients = step['action_gradients']
        # ... analyze
```

## 🐛 Troubleshooting

### flash-attn Installation Fails
```bash
# Try with reduced parallelism
MAX_JOBS=32 pip install flash-attn --no-build-isolation

# If still fails, check CUDA version
nvcc --version  # Must be 11.3+
```

### Server Connection Refused
```bash
# Check server is running
ps aux | grep evo1_server

# Check port is open
netstat -tuln | grep 8000

# Restart server with explicit port
python evo1_server_with_hooks.py --port 8000
```

### LIBERO Environment Errors
```bash
# Must use Python 3.8.13 (not 3.9+)
conda create -n libero python=3.8.13 -y

# Use specific PyTorch version
pip install torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### GPU Out of Memory
```bash
# Use gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Reduce batch size (if applicable)
# Use PyTorch 2.0+ for memory efficiency
```

## 📈 Expected Timeline

Full benchmark run on A100 GPU:

| Task | Time | Output |
|------|------|--------|
| Evo-1 setup | 15 min | Model + checkpoints cached |
| LIBERO setup | 10 min | Environment ready |
| Meta-World setup | 5 min | Environment ready |
| **LIBERO benchmark** | **2-3 hours** | 40 tasks × 50 episodes |
| **Meta-World benchmark** | **3-4 hours** | 50 tasks × 5 episodes |
| Analysis notebooks | 30 min | Hook visualizations |
| **Total** | **~7 hours** | Complete results |

## 🎓 Analysis Workflow

1. **Run benchmarks** (collect data)
   - LIBERO: 94.8% expected
   - Meta-World: 80.6% expected

2. **Analyze hook data** (understand behavior)
   - Integration module: How VL + proprio fuse
   - Semantic preservation: Attention patterns
   - Failure modes: Low success tasks

3. **Compare models** (if running multiple)
   - Evo-1 vs π0 vs RDT-1B
   - Architecture differences
   - Trade-offs (size vs performance)

4. **Publish results** (document findings)
   - Success rates vs expected
   - Hook analysis insights
   - Failure case studies

## 🔗 Related Documentation

- [CODEBASE_ANALYSIS.md](../docs/CODEBASE_ANALYSIS.md) - Detailed model analysis
- [scripts/README.md](../scripts/README.md) - Dependency verification guide
- [Evo-1 Repository](https://github.com/MINT-SJTU/Evo-1)
- [LIBERO Repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Meta-World Repository](https://github.com/Farama-Foundation/Metaworld)

## 💡 Tips for Cloud Deployment

### Cost Optimization
- Use spot/preemptible instances (70% cheaper)
- Download checkpoints to persistent disk (reuse across instances)
- Run benchmarks in batches (can pause/resume)

### Instance Selection
- **Minimum**: 1× V100 (16GB) - Can run but slower
- **Recommended**: 1× A100 (40GB) - Optimal price/performance
- **Overkill**: 8× H100 - Unnecessary for VLA inference

### Storage Planning
- System: 20GB
- Evo-1 checkpoints: 10GB
- InternVL3-1B: 5GB
- Results + hook data: 10GB
- **Total**: ~50GB (provision 100GB for safety)

## 📞 Support

Issues? Check:
1. Dependency verification: `./scripts/verify_all_dependencies.sh`
2. Server logs: `tail -f evo1_server.log`
3. GPU status: `nvidia-smi`
4. Environment: `conda env list`

Still stuck? Open an issue with:
- GPU type and CUDA version
- Error logs (full traceback)
- Environment output (`conda list`)
