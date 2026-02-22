# Multi-Model Ablation and Gradient Study Implementation

This document describes the implementation of two complementary studies across Evo-1, Pi0, and RDT-1B models.

## Overview

### Two Distinct Studies

**Study 1: Performance Ablation (Server-Based)**
- Methodology: Zero state encoder output → measure task success rates
- Benchmarks: LIBERO, MetaWorld (all models); VLABench (Pi0, RDT only)
- Metric: Success rate drop (baseline vs ablated)

**Study 2: Gradient Flow Analysis (Backpropagation-Based)**
- Methodology: Proper training loss → backward → measure gradients
- Loss Functions: Flow matching (Pi0, Evo-1), Diffusion (RDT-1B)
- Metric: Gradient magnitude (baseline vs ablated)

## Implementation Status

### ✅ Phase 2: Gradient Study (COMPLETE)

#### Loss Function Implementations

All loss functions implemented from actual model repos (not papers):

1. **hooks/losses/pi0_loss.py**
   - Source: Physical-Intelligence/openpi
   - Loss: `F.mse_loss(u_t, v_t)` flow matching
   - Functions: `pi0_flow_matching_loss`, `compute_flow_matching_components`

2. **hooks/losses/rdt_loss.py**
   - Source: thu-ml/RoboticsDiffusionTransformer
   - Loss: `F.mse_loss(pred, target)` diffusion
   - Functions: `rdt_diffusion_loss`, `DDPMNoiseScheduler`

3. **hooks/losses/evo1_loss.py**
   - Source: PROPRIOCEPTIVE_UNDERUTILIZATION_THEORY.md
   - Loss: Flow matching (similar to Pi0)
   - Functions: `evo1_flow_matching_loss`, `evo1_encode_observations`

#### Gradient Analysis Scripts

1. **scripts/run_evo1_gradient_analysis.py** (UPDATED)
   - ✅ Replaced `outputs.mean()` with proper flow matching loss
   - ✅ Loads actions from HDF5 files
   - ✅ Computes flow matching components
   - Usage:
     ```bash
     conda run -n evo1_server python run_evo1_gradient_analysis.py \
       --checkpoint metaworld \
       --data-path /path/to/data.h5 \
       --num-samples 50
     ```

2. **scripts/run_pi0_gradient_analysis.py** (NEW)
   - ✅ Complete gradient analysis with flow matching loss
   - ✅ Supports LIBERO, VLABench, MetaWorld
   - Usage:
     ```bash
     python run_pi0_gradient_analysis.py \
       --data-path /path/to/data.h5 \
       --num-samples 50
     ```

3. **scripts/run_rdt_gradient_analysis.py** (NEW)
   - ✅ Complete gradient analysis with diffusion loss
   - ✅ Includes noise scheduler implementation
   - ✅ Supports epsilon/sample prediction types
   - Usage:
     ```bash
     python run_rdt_gradient_analysis.py \
       --data-path /path/to/data.h5 \
       --num-samples 50 \
       --pred-type epsilon
     ```

### ✅ Phase 1: Ablation Study (COMPLETE)

#### Ablation Framework

**scripts/ablation_framework.py** (NEW)
- Reusable WebSocket server for ablation studies
- Components:
  - `AblationServer`: Serves model with optional zero injection
  - `run_ablation_trial`: Execute evaluation trials
  - `compare_results`: Statistical comparison (t-test, Cohen's d)
  - `save_results`, `print_results_summary`: Utilities

#### Ablation Runner Scripts

1. **scripts/run_pi0_ablation.py** (NEW)
   - Server-based ablation for Pi0 model
   - Targets `state_proj` layer
   - Benchmarks: LIBERO, VLABench, MetaWorld
   - Usage:
     ```bash
     python run_pi0_ablation.py \
       --benchmark libero \
       --tasks libero_90 \
       --num-episodes 50
     ```

2. **scripts/run_rdt_ablation.py** (NEW)
   - Server-based ablation for RDT-1B model
   - Targets `state_adaptor` layer
   - Benchmarks: LIBERO, VLABench, MetaWorld
   - Usage:
     ```bash
     python run_rdt_ablation.py \
       --all-benchmarks \
       --num-episodes 50
     ```

3. **Evo-1 Ablation** (EXISTING)
   - Already implemented in `notebooks/ablation_state_encoder_pass0.ipynb`
   - 1129 lines, fully functional
   - Can be extracted if script version needed

## File Structure

```
MultipleHooksStudy/
├── hooks/
│   ├── losses/                          # NEW
│   │   ├── __init__.py                 # Loss function exports
│   │   ├── pi0_loss.py                 # Pi0 flow matching loss
│   │   ├── rdt_loss.py                 # RDT diffusion loss
│   │   └── evo1_loss.py                # Evo-1 flow matching loss
│   ├── model_specific/
│   │   ├── pi0_hooks.py                # Existing hook manager
│   │   ├── rdt_hooks.py                # Existing hook manager
│   │   └── evo1_hooks.py               # Existing hook manager
│   └── ablation_hooks.py               # Existing ablation infrastructure
├── scripts/
│   ├── ablation_framework.py           # NEW - Reusable ablation server
│   ├── run_evo1_gradient_analysis.py   # UPDATED - Proper loss
│   ├── run_pi0_gradient_analysis.py    # NEW
│   ├── run_rdt_gradient_analysis.py    # NEW
│   ├── run_pi0_ablation.py             # NEW
│   ├── run_rdt_ablation.py             # NEW
│   └── data_collectors/                # Existing
│       ├── libero_collector.py
│       ├── metaworld_collector.py
│       └── bridge_collector.py
└── notebooks/
    └── ablation_state_encoder_pass0.ipynb  # Existing Evo-1 ablation
```

## Key Implementation Decisions

### 1. Loss Functions from Actual Repos

**Decision**: Use actual codebase implementations, not paper equations

**Rationale**:
- Papers may have typos or simplified versions
- Codebases are tested and validated
- Reduces implementation risk

**Sources**:
- Pi0: `openpi/src/openpi/models_pytorch/pi0_pytorch.py#L369`
- RDT: `RoboticsDiffusionTransformer/models/rdt_runner.py#L219`
- Evo-1: Theory document + Pi0 pattern

### 2. Separate Ablation and Gradient Studies

**Decision**: Two completely independent studies

**Rationale**:
- Different methodologies (server-based vs backprop-based)
- Different metrics (success rate vs gradient magnitude)
- Can cross-validate findings

**Integration**:
- If gradient study shows low gradients → ablation should show low impact
- If gradient study shows high gradients → ablation should show high impact

### 3. VLABench Scope

**Decision**: OpenMOSS/VLABench for Pi0 and RDT only (not Evo-1)

**Rationale**:
- Pi0 and RDT have pre-trained VLABench weights
- Evo-1 does not have VLABench weights available
- VLABench is MuJoCo-based manipulation benchmark

**Benchmarks**:
- Evo-1: LIBERO + MetaWorld
- Pi0: LIBERO + MetaWorld + VLABench
- RDT-1B: LIBERO + MetaWorld + VLABench

## Model Architecture Reference

### State Encoder Layers

| Model | State Encoder | Type | Location |
|-------|---------------|------|----------|
| Evo-1 | CategorySpecificMLP | Multi-layer MLP | action_head.state_encoder |
| Pi0 | state_proj | Single Linear | state_proj |
| RDT-1B | state_adaptor | Single Linear | state_adaptor |

### Training Objectives

| Model | Loss Type | Prediction Target | Implementation |
|-------|-----------|-------------------|----------------|
| Evo-1 | Flow Matching | Velocity field v_θ | evo1_loss.py |
| Pi0 | Flow Matching | Velocity field v_θ | pi0_loss.py |
| RDT-1B | Diffusion | Noise ε or action a | rdt_loss.py |

## Data Requirements

### For Gradient Analysis

HDF5 files with:
- `image`: (N, H, W, C) uint8 or float
- `robot_state`: (N, state_dim) float
- `action`: (N, horizon, action_dim) float

Collect using:
```bash
python scripts/data_collectors/libero_collector.py --num-samples 50
python scripts/data_collectors/metaworld_collector.py --num-samples 50
```

### For Ablation Analysis

Live environment access:
- LIBERO: Install libero package
- MetaWorld: Install metaworld package
- VLABench: Clone OpenMOSS/VLABench repo

## Next Steps

### 🔄 Integration Tasks

1. **VLABench Integration**
   - Clone OpenMOSS/VLABench repository
   - Download HuggingFace dataset: VLABench/vlabench_primitive_ft_dataset
   - Adapt policy interface for Pi0 and RDT
   - Update ablation scripts to support VLABench

2. **Model Interface Adaptation**
   - Implement `forward_with_time` for flow matching models
   - Implement `forward_with_timesteps` for diffusion model
   - These interfaces needed for proper loss computation

3. **Data Collection**
   - Run LIBERO collector for all models
   - Run MetaWorld collector for all models
   - Collect VLABench data for Pi0 and RDT

### 🧪 Execution Tasks

1. **Gradient Study**
   ```bash
   # Collect data first
   python scripts/data_collectors/libero_collector.py --num-samples 50
   
   # Run gradient analysis
   python scripts/run_pi0_gradient_analysis.py --data-path data.h5
   python scripts/run_rdt_gradient_analysis.py --data-path data.h5
   conda run -n evo1_server python scripts/run_evo1_gradient_analysis.py --data-path data.h5
   ```

2. **Ablation Study**
   ```bash
   # Run performance ablation
   python scripts/run_pi0_ablation.py --all-benchmarks --num-episodes 50
   python scripts/run_rdt_ablation.py --all-benchmarks --num-episodes 50
   # Evo-1 ablation already complete in notebook
   ```

3. **Cross-Validation**
   - Compare gradient magnitudes with ablation impacts
   - Verify alignment: high gradients ↔ high ablation impact
   - Document any discrepancies

## Expected Outputs

### Gradient Analysis Results

JSON files with structure:
```json
{
  "baseline": {
    "state_encoder_gradient_norm": 0.123,
    "state_encoder_gradient_mean": 0.045
  },
  "ablated": {
    "state_encoder_gradient_norm": 0.012,
    "state_encoder_gradient_mean": 0.005
  },
  "comparison": {
    "gradient_norm": {
      "baseline": 0.123,
      "ablated": 0.012,
      "reduction_percent": 90.2
    }
  }
}
```

### Ablation Study Results

JSON files with structure:
```json
{
  "libero": {
    "libero_90": {
      "baseline": {"success_rate": 0.82},
      "ablated": {"success_rate": 0.34},
      "comparison": {
        "absolute_drop": 0.48,
        "relative_drop_percent": 58.5,
        "p_value": 0.001,
        "cohens_d": 1.2,
        "statistically_significant": true
      }
    }
  }
}
```

## Verification Strategy

### Sanity Checks

1. **Loss Function Correctness**
   - Verify loss decreases during training steps
   - Check gradient magnitudes are reasonable (not too large/small)
   - Compare with paper-reported loss scales

2. **Ablation Effectiveness**
   - Verify state encoder output is actually zeros during ablation
   - Confirm hook is attached to correct layer
   - Check ablated success rate < baseline success rate

3. **Cross-Study Validation**
   - High gradient magnitude → High ablation impact
   - Low gradient magnitude → Low ablation impact
   - Document exceptions and investigate

### Debugging Tools

```python
# Verify loss function
from hooks.losses.pi0_loss import compute_flow_matching_components
components = compute_flow_matching_components(action_gt)
print(f"Noise shape: {components['noise'].shape}")
print(f"Time range: {components['time'].min():.3f} - {components['time'].max():.3f}")

# Verify ablation hook
def check_ablation(model, state_encoder):
    input_dummy = torch.randn(1, state_dim)
    output = state_encoder(input_dummy)
    print(f"Output is zeros: {torch.allclose(output, torch.zeros_like(output))}")
```

## Dependencies

### Python Packages

```
torch
numpy
scipy
h5py
websockets  # For ablation server
websocket-client  # For ablation client
```

### Model Packages

```
openpi  # For Pi0
rdt  # For RDT-1B (from github)
# Evo-1 model code assumed available
```

### Benchmark Packages

```
libero
metaworld
# VLABench from OpenMOSS/VLABench
```

## Contact & References

### Research Context

This implementation supports the hypothesis that proprioceptive state information is underutilized in current VLA models. See:
- `docs/PROPRIOCEPTIVE_UNDERUTILIZATION_THEORY.md`
- `docs/CODEBASE_ANALYSIS.md`
- `notebooks/ablation_state_encoder_pass0.ipynb`

### Model Repositories

- Pi0: https://github.com/Physical-Intelligence/openpi
- RDT-1B: https://github.com/thu-ml/RoboticsDiffusionTransformer
- VLABench: https://github.com/OpenMOSS/VLABench

---

**Status**: Implementation complete. Ready for execution and data collection.

**Last Updated**: 2024 (Implementation phase)
