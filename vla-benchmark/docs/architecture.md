# OCTO Benchmark Architecture

## Overview

This document describes the complete benchmark evaluation architecture for OCTO and other VLA models.

## Architecture Pattern

We use an **Adapter Pattern** to decouple model implementations from benchmark evaluations:

```
┌──────────────────────────────────────────────────────────────┐
│                     Benchmark Runner                          │
│                 (run_openx_benchmark.py)                      │
└───────────────────┬──────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────┐       ┌──────────────────┐
│  OctoAdapter  │       │ OpenXBenchmark   │
│  (Model)      │       │  (Dataset)       │
└───────┬───────┘       └────────┬─────────┘
        │                        │
        │ implements             │ implements
        ▼                        ▼
┌───────────────┐       ┌──────────────────┐
│  BaseAdapter  │       │ BaseBenchmark    │
│  (Interface)  │       │  (Interface)     │
└───────────────┘       └──────────────────┘
```

## Components

### 1. Base Classes (Interfaces)

#### `adapters/base_adapter.py` (212 lines)
Abstract base class defining standard interface for all VLA models.

**Key Methods:**
- `load_model()`: Load model checkpoint
- `preprocess_observation(obs)`: Standardize input
- `predict_action(obs, task)`: Core inference
- `postprocess_action(action)`: Denormalize/clip
- `get_action(obs, task)`: End-to-end pipeline

**Properties:**
- `action_dim`: Action space dimensionality
- `action_space`: Action bounds
- `is_loaded`: Model status

#### `benchmarks/base_benchmark.py` (305 lines)
Abstract base class for dataset evaluation.

**Key Methods:**
- `setup()`: Initialize dataset
- `get_task_list()`: List evaluation tasks
- `load_episode(task_id, ep_idx)`: Load data
- `evaluate_episode(adapter, episode)`: Run inference
- `compute_metrics(pred, gt)`: Calculate metrics
- `run_evaluation(adapter)`: Main evaluation loop

### 2. Concrete Implementations

#### `adapters/octo_adapter.py` (289 lines)
OCTO model implementation.

**Features:**
- HuggingFace model loading
- Multi-camera observation handling
- Language/goal conditioning support
- Action normalization using model statistics
- 7-DoF action space (6-DoF + gripper)

**Usage:**
```python
from adapters import OctoAdapter

adapter = OctoAdapter(model_path='hf://rail-berkeley/octo-small-1.5')
adapter.load_model()

obs = {'image_primary': image}  # (H, W, 3)
action = adapter.get_action(obs, task_description="pick up the cube")
# Returns: (7,) array normalized to [-1, 1]
```

#### `benchmarks/openx_benchmark.py` (273 lines)
Open X-Embodiment offline evaluation.

**Features:**
- Loads Bridge V2 dataset from TensorFlow Datasets
- Offline action prediction evaluation
- Metrics: MSE, MAE, cosine similarity
- Language instruction support
- Configurable episode limits

**Usage:**
```python
from benchmarks import OpenXBenchmark

benchmark = OpenXBenchmark(
    dataset_name='bridge_dataset',
    config={'max_episodes_per_task': 100}
)
benchmark.setup()
results = benchmark.run_evaluation(adapter)
```

### 3. Benchmark Runner Scripts

#### `scripts/octo/run_openx_benchmark.py` (152 lines)
End-to-end evaluation script.

**Workflow:**
1. Parse command-line arguments
2. Initialize OCTO adapter
3. Load OpenX benchmark
4. Run evaluation with progress tracking
5. Display and save results

**Command:**
```bash
python scripts/octo/run_openx_benchmark.py \
    --model hf://rail-berkeley/octo-small-1.5 \
    --max-episodes 100 \
    --output-dir results/openx
```

### 4. Configuration Files

#### `config/octo_config.yaml`
Model-specific configuration:
- Model checkpoints (small, base)
- Action space settings
- Inference parameters
- Task conditioning options

#### `config/benchmark_config.yaml`
Benchmark settings:
- Dataset configurations
- Evaluation modes (quick/standard/comprehensive)
- Metric thresholds
- Output settings

## Data Flow

### Evaluation Pipeline

```
1. Load Model
   ├─ OctoAdapter.load_model()
   └─ Download checkpoint from HuggingFace

2. Setup Benchmark
   ├─ OpenXBenchmark.setup()
   └─ Load dataset from TensorFlow Datasets

3. For each episode:
   ├─ Load episode: benchmark.load_episode(task, idx)
   │  └─ Returns: {observations, actions, task_description}
   │
   ├─ For each timestep:
   │  ├─ Preprocess: adapter.preprocess_observation(obs)
   │  ├─ Predict: adapter.predict_action(obs, task)
   │  └─ Postprocess: adapter.postprocess_action(action)
   │
   ├─ Compute metrics: benchmark.compute_metrics(pred, gt)
   │  └─ Returns: {action_mse, action_mae, cosine_similarity}
   │
   └─ Determine success: mse < threshold

4. Aggregate Results
   ├─ Compute overall statistics
   ├─ Calculate success rate
   └─ Save to JSON file
```

### Observation Format

**Input (from dataset):**
```python
{
    'image_primary': np.ndarray,  # (H, W, 3) uint8 [0, 255]
    'state': np.ndarray,          # (state_dim,) optional
}
```

**Processed (for model):**
```python
{
    'image_primary': np.ndarray,  # (H, W, 3) float32 [0, 1]
    'image_wrist': np.ndarray,    # Optional second camera
    'task': {
        'language_instruction': str,  # or None
        'goal_image': np.ndarray,     # or None
    }
}
```

### Action Format

**Model Output:**
```python
action = np.ndarray  # (7,) float32 normalized [-1, 1]
# [x, y, z, roll, pitch, yaw, gripper]
```

**Denormalized (for execution):**
```python
action_real = action * action_std + action_mean
# Values in actual robot units (meters, radians, etc.)
```

## Metrics

### Action Prediction Metrics (Offline)

1. **Mean Squared Error (MSE)**
   - Measures L2 distance between predicted and ground truth actions
   - Lower is better
   - Success threshold: typically < 0.1

2. **Mean Absolute Error (MAE)**
   - Measures L1 distance
   - More robust to outliers than MSE
   - Lower is better

3. **Cosine Similarity**
   - Measures directional alignment
   - Range: [-1, 1], higher is better
   - Useful for understanding action direction accuracy

4. **Success Rate**
   - Percentage of episodes with MSE < threshold
   - Primary metric for overall performance

## Extending the Framework

### Adding a New Model

1. Create adapter: `adapters/my_model_adapter.py`
2. Inherit from `BaseAdapter`
3. Implement required methods:
   - `load_model()`
   - `predict_action()`
   - `preprocess_observation()` (optional override)
4. Register in `adapters/__init__.py`

**Example:**
```python
from adapters.base_adapter import BaseAdapter

class MyModelAdapter(BaseAdapter):
    def load_model(self):
        self.model = load_my_model(self.model_path)
        self._is_loaded = True
    
    def predict_action(self, obs, task_description=None, **kwargs):
        return self.model.predict(obs)
```

### Adding a New Benchmark

1. Create benchmark: `benchmarks/my_benchmark.py`
2. Inherit from `BaseBenchmark`
3. Implement required methods:
   - `setup()`
   - `get_task_list()`
   - `load_episode()`
   - `compute_metrics()` (optional override)
4. Register in `benchmarks/__init__.py`

**Example:**
```python
from benchmarks.base_benchmark import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    def setup(self):
        self.data = load_my_dataset(self.data_path)
        self._is_initialized = True
    
    def get_task_list(self):
        return [{'task_id': 'task1', 'task_name': 'My Task'}]
    
    def load_episode(self, task_id, episode_idx):
        return self.data[episode_idx]
```

## File Structure

```
vla-benchmark/
├── adapters/
│   ├── __init__.py
│   ├── base_adapter.py        (212 lines) - Abstract base
│   └── octo_adapter.py        (289 lines) - OCTO implementation
│
├── benchmarks/
│   ├── __init__.py
│   ├── base_benchmark.py      (305 lines) - Abstract base
│   └── openx_benchmark.py     (273 lines) - OpenX implementation
│
├── config/
│   ├── octo_config.yaml       - Model settings
│   └── benchmark_config.yaml  - Benchmark settings
│
├── scripts/octo/
│   ├── README.md              - Documentation
│   ├── trial_inference.py     (198 lines) - Quick test
│   └── run_openx_benchmark.py (152 lines) - Full benchmark
│
└── results/
    ├── openx/                 - OpenX results
    └── libero/                - LIBERO results (coming soon)
```

## Dependencies

**Core:**
- Python 3.10 (required for OCTO)
- JAX 0.4.20 + jaxlib 0.4.20
- Flax 0.7.5
- TensorFlow 2.15.0 (for datasets)

**OCTO:**
- OCTO package (from models/octo/)
- NumPy, PIL, transformers

**Benchmarks:**
- tensorflow_datasets (for OpenX)
- Additional packages per benchmark

## Running Benchmarks

### Quick Test (5 episodes)
```bash
conda activate octo
cd scripts/octo
python run_openx_benchmark.py --max-episodes 5
```

### Standard Evaluation (100 episodes)
```bash
python run_openx_benchmark.py --max-episodes 100
```

### Different Model
```bash
python run_openx_benchmark.py \
    --model hf://rail-berkeley/octo-base-1.5 \
    --max-episodes 100
```

## Next Steps

1. ✅ OpenX benchmark implementation complete
2. ⏳ Create LIBERO-90 benchmark
3. Add visualization tools for results
4. Implement multi-model comparison
5. Add real-time robot evaluation support

## References

- OCTO Paper: https://octo-models.github.io/
- Open X-Embodiment: https://robotics-transformer-x.github.io/
- LIBERO: https://lifelong-robot-learning.github.io/LIBERO/
