# Benchmark Data Collectors

This directory contains data collectors for gathering real robot observations from benchmark environments. These collectors provide standardized outputs that can be used for gradient analysis across different models (Evo-1, Pi0, RDT-1B).

## Architecture

All collectors inherit from `BenchmarkDataCollector` (base_collector.py) and implement:
- `setup_environment()`: Initialize the environment/dataset
- `collect_observation()`: Collect one observation
- `get_observation_spec()`: Document the observation format

## Output Format

All collectors produce HDF5 files with:
```python
{
    'image': np.ndarray (N, H, W, 3) uint8,
    'robot_state': np.ndarray (N, state_dim) float32,
    'task_description': list of strings,
    'metadata': {
        'benchmark_name': str,
        'num_samples': int,
        'seed': int,
        'observation_spec': dict
    }
}
```

## Available Collectors

### 1. LIBERO Collector (`libero_collector.py`)

Collects from LIBERO benchmark tasks.

**Usage:**
```bash
# Collect 100 samples from LIBERO-90
python libero_collector.py \
    --benchmark libero_90 \
    --num-samples 100 \
    --output-dir /content/benchmark_observations

# Available benchmarks: libero_spatial, libero_object, libero_goal, libero_10, libero_90
```

**Observation Details:**
- Image: (224, 224, 3) RGB from agentview camera
- State: Variable dimension [gripper_states + joint_states + ee_states]
- Task: Natural language from BDDL files

### 2. MetaWorld Collector (`metaworld_collector.py`)

Collects from MetaWorld manipulation tasks.

**Usage:**
```bash
# Collect 100 samples from ML10 benchmark
python metaworld_collector.py \
    --benchmark ML10 \
    --num-samples 100 \
    --output-dir /content/benchmark_observations

# Available benchmarks: MT50 (50 tasks), ML10 (10 tasks), ML1 (single task)
```

**Observation Details:**
- Image: (224, 224, 3) RGB from corner camera
- State: (39,) [end_effector_pos, end_effector_quat, gripper_state, obj_pos, obj_quat, goal_pos, ...]
- Task: Task name (e.g., "reach-v3", "push-v3", "pick-place-v3")

### 3. Bridge/OpenX Collector (`bridge_collector.py`)

Collects from Bridge V2 and other OpenX datasets.

**Usage:**
```bash
# Collect 100 samples from Bridge dataset
python bridge_collector.py \
    --dataset bridge_dataset \
    --split train[:1000] \
    --num-samples 100 \
    --output-dir /content/benchmark_observations

# Available datasets: bridge_dataset, fractal20220817_data, etc.
# See TFDS catalog: https://www.tensorflow.org/datasets/catalog/overview
```

**Observation Details:**
- Image: (224, 224, 3) RGB from robot camera
- State: Variable dimension (depends on robot)
- Task: Natural language instruction

## Integration with Gradient Analysis

### Running Analysis with Real Data

```bash
# 1. Collect LIBERO observations
python scripts/data_collectors/libero_collector.py \
    --benchmark libero_90 \
    --num-samples 50 \
    --output-dir /content/benchmark_observations

# 2. Run Evo-1 gradient analysis with real data
python scripts/run_evo1_gradient_analysis.py \
    --checkpoint metaworld \
    --data-path /content/benchmark_observations/libero_libero_90_seed42_50samples.h5 \
    --num-samples 50 \
    --output /content/evo1_libero_results.json
```

### Comparing Real vs Synthetic Data

```bash
# Run with synthetic data (baseline)
python scripts/run_evo1_gradient_analysis.py \
    --checkpoint metaworld \
    --output /content/evo1_synthetic_results.json

# Run with real LIBERO data
python scripts/run_evo1_gradient_analysis.py \
    --checkpoint metaworld \
    --data-path /content/benchmark_observations/libero_libero_90_seed42_50samples.h5 \
    --num-samples 50 \
    --output /content/evo1_libero_results.json

# Run with real MetaWorld data
python scripts/run_evo1_gradient_analysis.py \
    --checkpoint metaworld \
    --data-path /content/benchmark_observations/metaworld_ml10_seed42_50samples.h5 \
    --num-samples 50 \
    --output /content/evo1_metaworld_results.json
```

## Model-Specific Data Requirements

### Evo-1
- **Benchmarks:** LIBERO, MetaWorld (no VLA-Bench support)
- **Image size:** 224x224
- **State dim:** 7 (end-effector pose + gripper)

### Pi0
- **Benchmarks:** LIBERO, MetaWorld, Bridge/OpenX
- **Image size:** 224x224
- **State dim:** Variable (7-39 depending on robot)

### RDT-1B
- **Benchmarks:** LIBERO, MetaWorld, Bridge/OpenX
- **Image size:** 224x224
- **State dim:** Variable

## Batch Data Collection Script

For convenience, collect from all benchmarks at once:

```bash
#!/bin/bash
# collect_all_benchmarks.sh

OUTPUT_DIR="/content/benchmark_observations"
NUM_SAMPLES=50
SEED=42

echo "Collecting LIBERO observations..."
python scripts/data_collectors/libero_collector.py \
    --benchmark libero_90 \
    --num-samples $NUM_SAMPLES \
    --output-dir $OUTPUT_DIR \
    --seed $SEED

echo "Collecting MetaWorld observations..."
python scripts/data_collectors/metaworld_collector.py \
    --benchmark ML10 \
    --num-samples $NUM_SAMPLES \
    --output-dir $OUTPUT_DIR \
    --seed $SEED

echo "Collecting Bridge observations..."
python scripts/data_collectors/bridge_collector.py \
    --dataset bridge_dataset \
    --split train[:1000] \
    --num-samples $NUM_SAMPLES \
    --output-dir $OUTPUT_DIR \
    --seed $SEED

echo "All observations collected to: $OUTPUT_DIR"
ls -lh $OUTPUT_DIR
```

## Troubleshooting

### LIBERO Issues
- **Problem:** `ModuleNotFoundError: No module named 'libero'`
  - **Solution:** Install LIBERO: `pip install libero-embodied`

- **Problem:** Missing BDDL files
  - **Solution:** Clone LIBERO repo and set `LIBERO_DIR` environment variable

### MetaWorld Issues
- **Problem:** `rendering fails` or `GLEW initialization error`
  - **Solution:** Set `export MUJOCO_GL=osmesa` or `export MUJOCO_GL=egl`

- **Problem:** Camera images all black
  - **Solution:** Try different camera names: `'corner'`, `'corner2'`, `'corner3'`

### Bridge/OpenX Issues
- **Problem:** Dataset download too slow
  - **Solution:** Use smaller split: `train[:100]` instead of `train`

- **Problem:** TFDS dataset not found
  - **Solution:** Check available datasets: `tfds.list_builders()`

## Data Format Validation

Verify collected data:

```python
import h5py
import numpy as np

# Load dataset
with h5py.File('libero_libero_90_seed42_50samples.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Image shape:", f['image'].shape)
    print("State shape:", f['robot_state'].shape)
    print("Num samples:", f['metadata'].attrs['num_samples'])
    print("Benchmark:", f['metadata'].attrs['benchmark_name'])
```

## Citation

If you use these data collectors, please cite the original benchmarks:

**LIBERO:**
```
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Carl and Feng, Zeren and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

**MetaWorld:**
```
@inproceedings{yu2020meta,
  title={Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning},
  author={Yu, Tianhe and Quillen, Deirdre and He, Zhanpeng and Julian, Ryan and Hausman, Karol and Finn, Chelsea and Levine, Sergey},
  booktitle={Conference on Robot Learning},
  year={2020}
}
```

**Bridge:**
```
@article{walke2023bridgedata,
  title={BridgeData V2: A Dataset for Robot Learning at Scale},
  author={Walke, Homer and Black, Kevin and Lee, Ken and Kim, Moo Jin and Du, Max and Zheng, Chongyi and Zhao, Tony and Hansen-Estruch, Philippe and Vuong, Quan and He, Andre and Myers, Vivek and Fang, Kuan and Finn, Chelsea and Levine, Sergey},
  journal={arXiv preprint arXiv:2308.12952},
  year={2023}
}
```
