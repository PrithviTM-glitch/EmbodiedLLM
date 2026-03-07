# VLA Model Codebase Analysis

This document contains findings from **actually reading** the real model repositories.

## 📊 Model Comparison

| Model | Size | HF Model | Benchmarks Supported | Server-Client | Status |
|-------|------|----------|---------------------|---------------|---------|
| **RDT-1B** | 1.2B | `robotics-diffusion-transformer/rdt-1b` | ManiSkill, RoboTwin | ❌ No | ✅ Ready |
| **Evo-1** | 0.77B | `MINT-SJTU/Evo1_MetaWorld`, `MINT-SJTU/Evo1_LIBERO` | Meta-World, LIBERO | ✅ Yes | ✅ Ready |
| **π0** | 3.3B | TBD (convert from JAX) | LIBERO, DROID, ALOHA | ✅ Yes (policy server) | ⚠️ Needs conversion |
| OpenVLA | 7B | `openvla/openvla-7b` | ManiSkill | ❌ No | ❌ **NO PROPRIO - SKIP** |

---

## 🔬 RDT-1B Deep Dive

### Repository
- **GitHub**: https://github.com/thu-ml/RoboticsDiffusionTransformer
- **HuggingFace**: `robotics-diffusion-transformer/rdt-1b` (1M steps), `rdt-170m` (500K steps)

### Dependencies
```bash
# Core
python==3.10.0
torch==2.1.0
torchvision==0.16.0
flash-attn (no-build-isolation)
packaging==24.0

# Required Encoders
google/t5-v1_1-xxl  # Language encoder
google/siglip-so400m-patch14-384  # Vision encoder
```

### Installation
```bash
conda create -n rdt python=3.10.0
conda activate rdt

# PyTorch (check your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Flash attention
MAX_JOBS=64 pip install flash-attn --no-build-isolation

# Other requirements
pip install -r requirements.txt
```

### Unified Action Space
**Critical**: RDT uses a unified action vector:
- **Joint positions**: 10 slots (fill first N for N-DOF robot)
- **EEF pose**: 6D rotation representation (NOT Euler angles!)
- **Gripper width**: Normalized to [0, 1]
- **No normalization** for physical quantities (except gripper)
- Use **International System of Units**

**Important Notes**:
1. **Single-arm robots**: Fill actions into **right-arm portion** of vector
2. **EEF rotation**: Must convert to 6D representation (see `docs/test_6drot.py`)
3. **Language embeddings**: Precompute if GPU VRAM < 24GB (see `scripts/encode_lang.py`)

### Benchmarks

#### 1. ManiSkill (5 tasks)
- **Tasks**: PegInsertionSide, PickCube, StackCube, PlugCharger, PushCube
- **Results**: 53.6% avg success (best among baselines)
- **Script**: `eval_sim/eval_rdt_maniskill.py`
- **Data**: [HuggingFace maniskill-model](https://huggingface.co/robotics-diffusion-transformer/maniskill-model)

#### 2. RoboTwin (50 dual-arm tasks)
- **Leaderboard**: 2nd place (after π0)
- **Environment**: Sapien-based simulator
- **Docs**: https://robotwin-platform.github.io/doc/usage/RDT.html

#### ⚠️ LIBERO / Meta-World
**NOT mentioned in RDT repo!** Would need custom implementation.

### Fine-tuning
```bash
# Download pre-trained checkpoint
huggingface-cli download robotics-diffusion-transformer/rdt-1b

# Fine-tune (150K+ steps recommended)
source finetune.sh
```

### Memory Optimization
- Use RDT-170M (smaller model)
- ZeRO-3 with offload
- 8-bit Adam (`use_8bit_adam=True`)
- Gradient accumulation (`--gradient_accumulation_steps`)

---

## 🔬 Evo-1 Deep Dive

### Repository
- **GitHub**: https://github.com/MINT-SJTU/Evo-1
- **HuggingFace Meta-World**: `MINT-SJTU/Evo1_MetaWorld`
- **HuggingFace LIBERO**: `MINT-SJTU/Evo1_LIBERO`
- **Dataset**: `MINT-SJTU/Evo1_MetaWorld_Dataset`

### Dependencies
```bash
# Core
python==3.10
flash-attn==2.8.3+cu12torch2.7cxx11abiTRUE  # CRITICAL - must install correctly!
transformers
accelerate
timm

# VL Backbone
OpenGVLab/InternVL3-1B  # 1B vision-language model
```

### Installation
```bash
conda create -n Evo1 python=3.10 -y
conda activate Evo1

cd Evo_1/
pip install -r requirements.txt

# CRITICAL: FlashAttention (reduce MAX_JOBS if needed)
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
```

### Benchmarks

#### 1. Meta-World (MT50)
- **Performance**: 80.6% success rate (SOTA!)
- **Setup**: Separate conda environment
```bash
conda create -n metaworld python=3.10 -y
conda activate metaworld
pip install mujoco metaworld websockets opencv-python packaging huggingface_hub
```

- **Server-Client Architecture** ✅:
```bash
# Terminal 1: Start model server
conda activate Evo1
cd Evo_1
python scripts/Evo1_server.py

# Terminal 2: Run Meta-World client
conda activate metaworld
cd MetaWorld_evaluation
python mt50_evo1_client_prompt.py
```

#### 2. LIBERO (4 tasks)
- **Performance**: 94.8% success rate (SOTA!)
- **Setup**: Requires LIBERO repo
```bash
conda create -n libero python=3.8.13 -y
conda activate libero

cd LIBERO_evaluation/
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
pip install websockets huggingface_hub
```

- **Server-Client Architecture** ✅:
```bash
# Terminal 1: Model server (same as Meta-World)
conda activate Evo1
cd Evo_1
python scripts/Evo1_server.py

# Terminal 2: LIBERO client
conda activate libero
cd LIBERO_evaluation
python libero_client_4tasks.py
```

### Training (Two-Stage)

**Stage 1**: Train integration module + action expert only
```bash
accelerate launch --num_processes 1 scripts/train.py \
  --run_name Evo1_metaworld_stage1 \
  --action_head flowmatching \
  --lr 1e-5 --batch_size 16 --max_steps 5000 \
  --finetune_action_head \
  --vlm_name OpenGVLab/InternVL3-1B \
  --save_dir /path/to/checkpoints/stage1
```

**Stage 2**: Full-scale training (VLM + action expert)
```bash
accelerate launch --num_processes 1 scripts/train.py \
  --run_name Evo1_metaworld_stage2 \
  --finetune_vlm --finetune_action_head \
  --max_steps 80000 \
  --resume --resume_pretrain \
  --resume_path /path/to/checkpoints/stage1/step_5000 \
  --save_dir /path/to/checkpoints/stage2
```

### Key Innovations
1. **Integration Module**: Aligns VL features + proprioceptive state
2. **Two-Stage Training**: Preserves semantic attention (avoids drift)
3. **Flow Matching**: Continuous action generation
4. **LeRobot v2.1 Format**: Uses standardized robot dataset format

---

## 🔬 π0 Deep Dive

### Repository
- **GitHub**: https://github.com/Physical-Intelligence/openpi
- **Framework**: JAX (original), PyTorch (recently added!)

### Dependencies
```bash
# Managed via uv (modern Python package manager)
uv sync
uv pip install -e .
```

### PyTorch Support (NEW!)
**Note**: Must convert JAX checkpoints to PyTorch:
```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /path/to/jax/checkpoint \
  --config_name <config> \
  --output_path /path/to/pytorch/checkpoint
```

### Available Checkpoints
| Checkpoint | Type | HF Path | Use Case |
|------------|------|---------|----------|
| `pi0_base` | Base | `gs://openpi-assets/checkpoints/pi0_base` | Fine-tuning |
| `pi0_fast_base` | Base | `gs://openpi-assets/checkpoints/pi0_fast_base` | Autoregressive |
| `pi05_base` | Base | `gs://openpi-assets/checkpoints/pi05_base` | Latest (knowledge insulation) |
| `pi0_fast_droid` | Expert | `gs://openpi-assets/checkpoints/pi0_fast_droid` | DROID robot |
| `pi05_libero` | Expert | `gs://openpi-assets/checkpoints/pi05_libero` | LIBERO benchmark |

### Benchmarks

#### 1. LIBERO
- **Checkpoint**: `pi05_libero`
- **Docker workflow** provided for evaluation
- **See**: `examples/libero/README.md`

#### 2. DROID
- **Checkpoints**: `pi0_fast_droid`, `pi0_droid`, `pi05_droid`
- **Training**: Uses DROID dataset (full or with idle filter)
- **See**: `examples/droid/README_train.md`

#### 3. ALOHA
- **Checkpoints**: `pi0_aloha_towel`, `pi0_aloha_tupperware`, `pi0_aloha_pen_uncap`
- **Tasks**: Towel folding, tupperware unpacking, pen uncapping
- **See**: `examples/aloha_sim/`, `examples/aloha_real/`

### Policy Server-Client ✅
```bash
# Server: Load model
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_libero \
  --policy.dir=checkpoints/pi05_libero/my_experiment/20000

# Client: Robot runtime queries server via websockets
# See: docs/remote_inference.md
```

### Key Features
- **Flow Matching**: Continuous action generation
- **Block-wise Causal Masking**: Asymmetric conditioning
- **Separate Proprio Encoder**: Multi-layer dedicated state encoder
- **10k+ hours pre-training**: Real robot data

---

## 🎯 Benchmark Status

| Benchmark | RDT-1B | Evo-1 | π0 | OpenVLA |
|-----------|---------|-------|-----|---------|
| **LIBERO** | ❌ Not tested | ✅ 94.8% | ✅ Checkpoint available | ✅ Tested (low perf) |
| **Meta-World** | ❌ Not tested | ✅ 80.6% | ❌ Not tested | ❌ Not tested |
| **ManiSkill** | ✅ 53.6% | ❌ Not tested | ❌ Not tested | ✅ 4.8% |
| **DROID** | ❌ Not tested | ❌ Not tested | ✅ Expert checkpoint | ❌ Not tested |
| **RoboTwin** | ✅ 2nd place | ⏳ Script coming | ❌ Not tested | ❌ Not tested |

### 🚨 Critical Findings

1. **NO model has both LIBERO + Meta-World**:
   - Evo-1: ✅ LIBERO + ✅ Meta-World (WINNER!)
   - RDT: ✅ ManiSkill + ✅ RoboTwin (no LIBERO/Meta-World)
   - π0: ✅ LIBERO + ✅ DROID (no Meta-World)

2. **OpenVLA has NO proprioceptive state** → No value for embodied robotics research

3. **Server-client already exists**:
   - Evo-1: Full implementation (websockets)
   - π0: Policy server with remote inference
   - RDT: No server-client (runs standalone)

---

## 🎨 Recommended Architecture

### For Evo-1 (Already Implemented!)
```
Server (Evo1_server.py):
├── Load model weights
├── Listen on websocket (default: ws://localhost:8000)
└── Receive obs → Return action chunk

Client (benchmark-specific):
├── mt50_evo1_client_prompt.py (Meta-World)
├── libero_client_4tasks.py (LIBERO)
└── Custom client for other benchmarks
```

### For RDT (Need to Build)
```
Server:
├── Load RDT model from HF
├── Load encoders (T5-XXL, SigLIP)
├── Listen on port
└── Process obs → Return unified action vector

Client:
├── Convert benchmark obs → RDT format
├── Send to server
├── Convert unified action → robot-specific action
└── Execute in environment
```

### For π0 (Use Existing)
```
Policy Server (existing):
├── serve_policy.py
├── Load converted PyTorch checkpoint
└── HTTP/WebSocket interface

Client:
├── Use existing examples (LIBERO, DROID, ALOHA)
└── Adapt for Meta-World (if needed)
```

---

## 🛠️ Next Steps

1. **Verify Dependencies**:
   - [ ] Create conda environments for each model
   - [ ] Test HuggingFace model loading
   - [ ] Verify flash-attn installation

2. **Benchmark Infrastructure**:
   - [ ] Set up LIBERO environment (Python 3.8.13)
   - [ ] Set up Meta-World environment (Python 3.10)
   - [ ] Create unified evaluation framework

3. **Server-Client Implementation**:
   - [ ] Use Evo-1's existing server-client for LIBERO + Meta-World
   - [ ] Build RDT server-client for ManiSkill
   - [ ] Adapt π0 policy server for new benchmarks

4. **Hook Integration**:
   - [ ] Inject hooks into Evo-1 server before inference
   - [ ] Inject hooks into RDT server
   - [ ] Inject hooks into π0 policy server
   - [ ] Collect gradient/representation data during benchmark runs

---

**Last Updated**: Based on repository reads (Feb 2026)
**Status**: OpenVLA removed from pipeline (no proprioceptive state)
