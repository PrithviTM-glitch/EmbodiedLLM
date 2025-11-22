# Evo-1 Model Setup

## Overview

**Evo-1** is a lightweight Vision-Language-Action (VLA) model that achieves state-of-the-art performance with only 0.77 billion parameters. It achieves a **94.8% success rate on LIBERO-90** benchmark, making it one of the top-performing models in its size category.

### Key Features
- **Parameters**: 0.77B (770M)
- **LIBERO-90 Success Rate**: 94.8%
- **Architecture**: Built on InternVL3-1B with cross-modulated diffusion transformer
- **Training**: Two-stage training paradigm without robot data pretraining
- **Efficiency**: High inference frequency with low memory overhead

### Paper and Resources
- **Paper**: [arXiv:2511.04555](https://arxiv.org/abs/2511.04555)
- **GitHub**: [MINT-SJTU/Evo-1](https://github.com/MINT-SJTU/Evo-1)
- **HuggingFace Models**: 
  - [Meta-World Checkpoint](https://huggingface.co/MINT-SJTU/Evo1_MetaWorld)
  - [LIBERO Checkpoint](https://huggingface.co/MINT-SJTU/Evo1_LIBERO)
- **Website**: [https://mint-sjtu.github.io/Evo-1.io/](https://mint-sjtu.github.io/Evo-1.io/)

---

## Installation

### Prerequisites
- Python 3.10
- CUDA 11.3 or higher
- Conda package manager

### Step 1: Clone Evo-1 Repository

```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models
git clone https://github.com/MINT-SJTU/Evo-1.git
cd Evo-1
```

### Step 2: Create Conda Environment

```bash
# Create dedicated environment for Evo-1
conda create -n Evo1 python=3.10 -y
conda activate Evo1
```

### Step 3: Install Core Dependencies

```bash
cd Evo_1
pip install -r requirements.txt

# Install Flash Attention (critical for performance)
# Note: Adjust MAX_JOBS based on your system (default: 64)
# Skipping this may cause lower success rate or unstable robot motion
MAX_JOBS=64 pip install -v flash-attn --no-build-isolation
```

> **⚠️ IMPORTANT**: Installing Flash Attention is critical. Skipping this step may result in lower success rates or unstable robot motion during evaluation.

---

## LIBERO Benchmark Setup

### Step 1: Prepare LIBERO Environment

```bash
# Create separate environment for LIBERO
conda create -n libero python=3.8.13 -y
conda activate libero

# Navigate to LIBERO evaluation directory
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/LIBERO_evaluation/

# Clone LIBERO repository
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

# Install LIBERO dependencies
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .

# Install additional dependencies for Evo-1 evaluation
pip install websockets
pip install huggingface_hub
```

### Step 2: Download LIBERO Model Weights

```bash
# Download pre-trained LIBERO checkpoint
huggingface-cli download MINT-SJTU/Evo1_LIBERO --local-dir /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/checkpoints/libero/
```

> **Note**: If `huggingface-cli` is not available, install it with:
> ```bash
> pip install huggingface_hub[cli]
> ```

### Step 3: Configure Checkpoint Paths

After downloading the model weights, you need to modify the configuration files to point to the correct checkpoint locations:

**File 1: `Evo_1/scripts/Evo1_server.py`** (Line 149)
```python
# Modify checkpoint directory
checkpoint_dir = "/Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/checkpoints/libero/"
```

**File 2: `LIBERO_evaluation/libero_client_4tasks.py`** (Line 24)
```python
# Modify checkpoint name to match downloaded checkpoint
ckpt_name = "your_checkpoint_name_here"  # Update after download
```

**Optional: Modify Server/Client Ports**
- Server port: `Evo_1/scripts/Evo1_server.py` (Line 152)
- Client port: `LIBERO_evaluation/libero_client_4tasks.py` (Line 23)

---

## Running LIBERO Evaluation

Evo-1 uses a client-server architecture for evaluation. You need to run two terminals simultaneously:

### Terminal 1: Start Evo-1 Server

```bash
conda activate Evo1
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/Evo_1
python scripts/Evo1_server.py
```

### Terminal 2: Run LIBERO Client

```bash
conda activate libero
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/models/Evo-1/LIBERO_evaluation
python libero_client_4tasks.py
```

The evaluation will run the LIBERO benchmark tasks and output results.

---

## Expected Results

Based on the original paper, Evo-1 should achieve:
- **LIBERO-90 Success Rate**: 94.8%
- **Inference**: High frequency with low memory overhead
- **Real-world Success Rate**: 78% (for reference)

---

## Troubleshooting

### Issue: Lower than expected success rate
- **Solution**: Ensure Flash Attention is properly installed with `MAX_JOBS=64 pip install -v flash-attn --no-build-isolation`

### Issue: CUDA out of memory
- **Solution**: Reduce batch size or use a GPU with more memory (recommended: 16GB+ VRAM)

### Issue: Connection refused between server and client
- **Solution**: Check that both server and client are using the same port numbers in their respective configuration files

### Issue: Checkpoint not found
- **Solution**: Verify the checkpoint path in `Evo1_server.py` matches your actual download location

---

## Next Steps

After successful setup and evaluation:
1. Results will be saved in the evaluation directory
2. Document results in `/Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/docs/evo-1/results.md`
3. Compare achieved results with paper benchmarks
4. Log any discrepancies or issues encountered

---

## Citation

If using Evo-1 in your research, please cite:

```bibtex
@article{lin2025evo,
  title={Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment},
  author={Lin, Tao and Zhong, Yilei and Du, Yuxin and Zhang, Jingjing and Liu, Jiting and Chen, Yinxinyu and Gu, Encheng and Liu, Ziyan and Cai, Hongyi and Zou, Yanwen and others},
  journal={arXiv preprint arXiv:2511.04555},
  year={2025}
}
```
