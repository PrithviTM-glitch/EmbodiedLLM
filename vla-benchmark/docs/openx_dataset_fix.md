# Open X-Embodiment Dataset Access

## Issue: Dataset Not Found

The Open X-Embodiment datasets are not part of standard TensorFlow Datasets. They're hosted separately and accessed through OCTO's data utilities.

## Solution

### 1. **Use OCTO's Data Loading** (Implemented ✅)

The benchmark now uses `octo.data.oxe.make_oxe_dataset_kwargs` to properly load OXE datasets.

**Available Datasets:** 58 total datasets
- Run `python scripts/octo/list_oxe_datasets.py` to see all

**Recommended for Testing:**
1. `fractal20220817_data` - Smaller dataset, faster to load
2. `kuka` - Standard manipulation tasks  
3. `austin_buds_dataset_converted_externally_to_rlds` - Very small, good for quick tests

### 2. **Dataset Access Methods**

#### Option A: Google Cloud Storage (Default)
```bash
python run_openx_benchmark.py \
    --dataset fractal20220817_data \
    --data-path gs://gresearch/robotics \
    --max-episodes 10
```

**Requirements:**
- Google Cloud authentication (for remote access)
- Internet connection
- May have download costs

**Setup Google Cloud Auth:**
```bash
# Install gcloud CLI
pip install google-cloud-storage

# Authenticate
gcloud auth application-default login
```

#### Option B: Local Dataset (Faster, No Cloud Needed)
If you have downloaded the dataset locally:

```bash
python run_openx_benchmark.py \
    --dataset fractal20220817_data \
    --data-path /path/to/local/oxe/data \
    --max-episodes 10
```

**Download datasets from:**
- Full OXE: `gs://gresearch/robotics` (requires gsutil)
- Bridge V2: https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/

### 3. **Quick Test Command**

For immediate testing without cloud setup:

```bash
# Use small dataset with few episodes
python run_openx_benchmark.py \
    --dataset austin_buds_dataset_converted_externally_to_rlds \
    --max-episodes 5
```

This will attempt to stream from cloud storage (no local download needed for small tests).

## Updated Benchmark Features

✅ **Fixed Issues:**
- Changed from `tfds.builder()` to `make_oxe_dataset_kwargs()`
- Updated default dataset from `bridge_dataset` to `fractal20220817_data`
- Added proper error messages for dataset access
- Support for 58 different OXE datasets

✅ **New Capabilities:**
- Proper OCTO data format handling
- Language instruction extraction
- Multi-camera support (primary, wrist, secondary)
- Proprioceptive state handling
- Configurable data paths (cloud or local)

## Dataset Structure

OCTO's data format:
```python
trajectory = {
    'observation': {
        'image_primary': (T, H, W, 3),  # Primary camera
        'image_wrist': (T, H, W, 3),     # Wrist camera (optional)
        'proprio': (T, proprio_dim),      # Robot state (optional)
    },
    'action': (T, action_dim),            # Robot actions
    'task': {
        'language_instruction': str,      # Task description
    }
}
```

## Troubleshooting

### Error: "Dataset not found"
- Check dataset name with `python list_oxe_datasets.py`
- Verify data path is correct
- For cloud access, authenticate with gcloud

### Error: "Permission denied" (Cloud Storage)
```bash
# Authenticate
gcloud auth application-default login

# Or set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### Error: "No module named 'octo.data.oxe'"
```bash
# Install OCTO
cd models/octo
pip install -e .
```

### Slow Loading
- Use smaller datasets (austin_buds, fractal)
- Reduce `--max-episodes`
- Download dataset locally for repeated use

## Examples

### Quick Test (5 episodes, small dataset)
```bash
python run_openx_benchmark.py \
    --dataset austin_buds_dataset_converted_externally_to_rlds \
    --max-episodes 5
```

### Standard Evaluation (100 episodes)
```bash
python run_openx_benchmark.py \
    --dataset fractal20220817_data \
    --max-episodes 100
```

### Using Local Dataset
```bash
python run_openx_benchmark.py \
    --dataset bridge_dataset \
    --data-path /Users/tmprithvi/datasets/oxe \
    --max-episodes 50
```

### Compare Multiple Datasets
```bash
for dataset in fractal20220817_data kuka austin_buds_dataset_converted_externally_to_rlds; do
    python run_openx_benchmark.py \
        --dataset $dataset \
        --max-episodes 20 \
        --output-dir results/openx/$dataset
done
```

## Next Steps

1. **Test the fix:**
   ```bash
   cd scripts/octo
   python list_oxe_datasets.py  # List all available datasets
   python run_openx_benchmark.py --max-episodes 5  # Quick test
   ```

2. **Run benchmark:**
   ```bash
   python run_openx_benchmark.py --max-episodes 100
   ```

3. **Try different datasets:**
   ```bash
   python run_openx_benchmark.py --dataset kuka --max-episodes 50
   ```
