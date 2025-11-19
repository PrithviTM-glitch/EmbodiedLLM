# OCTO Trial Scripts

This directory contains scripts for testing and benchmarking OCTO models.

## Dependencies

To run these scripts, you need to install OCTO and its dependencies. Due to dependency conflicts, we recommend the following installation approach:

### Option 1: Install in OCTO's Environment (Recommended)

```bash
# Navigate to OCTO directory
cd ../../models/octo

# Create a conda environment (recommended)
conda create -n octo python=3.10
conda activate octo

# Install OCTO
pip install -e .
pip install -r requirements.txt

# For GPU support (CUDA 11)
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Test installation
python examples/01_inference_pretrained.ipynb  # or convert to .py first
```

### Option 2: Minimal Installation for Trial Script

If you just want to run the trial script without full OCTO dependencies:

```bash
# Activate your venv
cd ../..
source venv/bin/activate

# Install minimal dependencies
pip install numpy pillow matplotlib requests

# Install JAX (CPU version)
pip install jax==0.4.20 jaxlib

# Install Flax and transformers
pip install flax transformers huggingface-hub

# Add OCTO to path (the script does this automatically)
```

## Scripts

### `trial_inference.py`

A standalone script that:
1. Loads the OCTO-small model from HuggingFace
2. Downloads a sample image from the Bridge V2 dataset
3. Runs inference with multiple language instructions
4. Saves results to `../../results/octo/trial/`

**Usage:**
```bash
# Make sure OCTO is installed first!
python trial_inference.py
```

**Expected Output:**
- Model loads successfully from HuggingFace
- Sample image is downloaded or created
- Inference runs on 3 language instructions
- Results saved as JSON with timestamp
- Summary statistics printed

**What it tests:**
- ✓ OCTO model loading
- ✓ Image preprocessing
- ✓ Language instruction encoding
- ✓ Action prediction
- ✓ Model inference speed

## Troubleshooting

### Import Errors
If you see import errors, make sure you:
1. Installed OCTO dependencies: `cd ../../models/octo && pip install -r requirements.txt`
2. The OCTO package is installed: `pip install -e ../../models/octo`

### Network Timeouts
If pip install times out:
1. Increase timeout: `pip install --timeout=1000 <package>`
2. Try a different time or network
3. Download packages manually and install from local files

### JAX/CUDA Issues
- For CPU-only: `pip install jax==0.4.20`
- For CUDA 11: `pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- For CUDA 12: `pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

### Model Download Issues
The script downloads the model from HuggingFace on first run. If this fails:
1. Check your internet connection
2. Manually download from: https://huggingface.co/rail-berkeley/octo-small-1.5
3. Load from local path: modify `model_name` in the script

## Benchmark Scripts

### `run_openx_benchmark.py`

**NEW!** Complete benchmark evaluation on Open X-Embodiment datasets.

**Features:**
- Loads OCTO model via OctoAdapter
- Evaluates on Bridge V2 dataset (offline action prediction)
- Computes metrics: MSE, MAE, cosine similarity
- Saves detailed results with timestamps
- Supports language-conditioned tasks

**Usage:**
```bash
# Quick test (5 episodes)
python run_openx_benchmark.py --max-episodes 5

# Standard evaluation (100 episodes, default)
python run_openx_benchmark.py

# Use base model instead of small
python run_openx_benchmark.py --model hf://rail-berkeley/octo-base-1.5

# Custom output directory
python run_openx_benchmark.py --output-dir results/my_test
```

**Options:**
- `--model`: OCTO model checkpoint (default: octo-small-1.5)
- `--max-episodes`: Number of episodes to evaluate (default: 10)
- `--dataset`: Dataset name (default: bridge_dataset)
- `--output-dir`: Results directory (default: results/openx)
- `--use-language`: Use language instructions (default: True)

**Expected Output:**
```
[1/4] Initializing OCTO adapter...
✅ OCTO model loaded: hf://rail-berkeley/octo-small-1.5
   - Action dimension: 7
   - Action space: {'low': [-1, -1, ...], 'high': [1, 1, ...]}

[2/4] Initializing OpenX benchmark...
✅ Benchmark initialized
   - Dataset: bridge_dataset
   - Episodes: 10
   - Using language: True

[3/4] Running evaluation...
Evaluating task: openx_eval |██████████| 10/10 episodes

[4/4] Results Summary
================================================================================
📊 Overall Metrics:
   Success Rate: 45.00%
   Total Episodes: 10
   Successful Episodes: 4
   Failed Episodes: 6

📈 Action Prediction Metrics:
   MSE (Mean): 0.085432 ± 0.042156
   MAE (Mean): 0.234567 ± 0.098234
   Cosine Similarity: 0.8234 ± 0.1234

⏱️  Performance:
   Avg Time per Episode: 2.34s
   Total Evaluation Time: 23.45s

💾 Full results saved to: results/openx/OpenX_20240101_120000.json
```

**What it benchmarks:**
- ✓ Action prediction accuracy (offline)
- ✓ Language instruction following
- ✓ Multi-step trajectory prediction
- ✓ Inference speed and efficiency

## Next Steps

Once the benchmark scripts work:
1. ✅ Create benchmark runners for Open X-Embodiment (DONE!)
2. ⏳ Create LIBERO-90 benchmark implementation
3. Run full evaluations on both benchmarks
4. Compare OCTO-small vs OCTO-base results
5. Compare against published baseline metrics

## Results Location

**Trial Results:**
```
../../results/octo/trial/trial_results_YYYYMMDD_HHMMSS.json
```

**Benchmark Results:**
```
../../results/openx/OpenX_YYYYMMDD_HHMMSS.json
../../results/libero/LIBERO90_YYYYMMDD_HHMMSS.json
```

Each result file contains:
- Timestamp and configuration
- Model information
- Per-episode results with predictions
- Aggregated metrics and statistics
- Task success rates

````
