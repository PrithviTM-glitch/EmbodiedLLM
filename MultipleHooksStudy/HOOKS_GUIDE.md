# Diagnostic Hooks Infrastructure - User Guide

Complete guide to using the diagnostic hooks system for analyzing proprioceptive encoder utilization in VLA models.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Hook Categories](#hook-categories)
- [Model-Specific Adapters](#model-specific-adapters)
- [Running Experiments](#running-experiments)
- [Analyzing Results](#analyzing-results)
- [API Reference](#api-reference)

## Overview

This infrastructure provides comprehensive diagnostic tools for analyzing why proprioceptive state encoders are underutilized in VLA models.

**What it does**:
- ✅ Measures gradient flow through encoders
- ✅ Computes representation quality (CKA, effective rank)
- ✅ Tracks downstream utilization (attention, mutual information)
- ✅ Performs systematic ablation studies
- ✅ Generates publication-ready visualizations

**Models supported**:
- OpenVLA (7B) - No proprio encoder
- Octo (93M) - Linear projection
- RDT-1B (1.2B) - MLP + Fourier features
- π0 (3.3B) - Separate encoder + causal masking

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/EmbodiedLLM.git
cd EmbodiedLLM/MultipleHooksStudy

# Install dependencies
pip install torch numpy matplotlib seaborn scipy transformers

# For Colab
!git clone https://github.com/yourusername/EmbodiedLLM.git
import sys
sys.path.append('/content/EmbodiedLLM/MultipleHooksStudy')
```

## Quick Start

**5-Minute Example: Analyze Gradient Flow**

```python
from transformers import AutoModel
from hooks.model_specific import OctoHooks

# Load model
model = AutoModel.from_pretrained("octo-base")

# Create hooks
hooks = OctoHooks(model)
hooks.attach_gradient_hooks()

# Run training step
output = model(**batch)
loss = output.mean()
loss.backward()

# Get results
report = hooks.gradient_analyzer.get_comprehensive_report()
print(f"Proprio/Vision Ratio: {report['encoder_stats']['proprio_encoder']['proprio_vision_ratio']:.4f}")

# Cleanup
hooks.cleanup()
```

**10-Minute Example: Full Diagnostic**

```python
from analysis import ExperimentCoordinator

# Setup
coordinator = ExperimentCoordinator(output_dir="./results")
coordinator.register_model("octo", octo_model)

# Run diagnostics
results = coordinator.run_full_diagnostic(
    data_loader=dataloader,
    eval_fn=lambda m, b: eval_metric(m, b),
    gradient_batches=10,
    representation_batches=50,
    ablation_batches=20,
    utilization_batches=30
)

# Visualize
from analysis.result_analyzer import ResultAnalyzer
analyzer = ResultAnalyzer("./results")
analyzer.load_all_results()
analyzer.plot_all(output_dir="./results/plots")
```

## Hook Categories

### 1. Gradient Hooks

**Purpose**: Measure gradient magnitudes flowing to each encoder

**Classes**:
- `EncoderGradientTracker` - Track encoder-level gradients
- `LayerWiseGradientProfiler` - Profile each layer
- `GradientFlowAnalyzer` - High-level coordinator

**Usage**:
```python
hooks.attach_gradient_hooks()

# Run backward pass
loss.backward()

# Get results
report = hooks.gradient_analyzer.get_comprehensive_report()
# Returns: encoder stats, layer profiles, vanishing points, decay rates
```

**Key Metrics**:
- `mean_norm`: Average gradient magnitude
- `proprio_vision_ratio`: Ratio of proprio/vision gradients
- `vanishing_point`: First layer where gradients < 1e-4
- `gradient_decay`: Percentage decrease from first to last layer

### 2. Representation Hooks

**Purpose**: Assess information content and redundancy

**Classes**:
- `FeatureExtractor` - Capture intermediate features
- `CKASimilarityAnalyzer` - Compute CKA between encoders
- `EffectiveRankCalculator` - Measure dimensionality
- `RepresentationQualityAnalyzer` - Coordinator

**Usage**:
```python
hooks.attach_representation_hooks()

# Run forward passes (no backward needed)
for batch in dataloader:
    model(**batch)

# Compute CKA
similarity_matrix = hooks.representation_analyzer.cka_analyzer.get_similarity_matrix()

# Compute effective rank
for name, calc in hooks.representation_analyzer.rank_calculators.items():
    rank, utilization = calc.get_summary()
    print(f"{name}: Rank={rank:.1f}, Utilization={utilization:.1f}%")
```

**Key Metrics**:
- `cka_similarity`: Centered kernel alignment (0-1)
- `effective_rank`: Participation ratio of eigenvalues
- `utilization_percent`: effective_rank / total_dimensions * 100
- `high_redundancy_pairs`: Encoder pairs with CKA > 0.7

### 3. Ablation Hooks

**Purpose**: Measure encoder importance via removal

**Classes**:
- `ZeroOutAblationHook` - Zero encoder outputs
- `NoiseInjectionHook` - Add Gaussian noise
- `ModalityAblationManager` - Coordinate ablations
- `AblationStudyCoordinator` - Run standard suite

**Usage**:
```python
hooks.attach_ablation_hooks()

# Run ablation suite
def eval_fn(model, batch):
    return compute_performance_metric(model, batch)

ablation_results = hooks.ablation_coordinator.run_standard_ablations()

# Evaluate each configuration
for config_name, config in ablation_results.items():
    # Apply ablation
    hooks.ablation_coordinator.ablation_manager.ablate_only(config["ablated_encoders"])
    
    # Measure performance
    performance = evaluate(model, dataloader, eval_fn)
    
    # Reset
    hooks.ablation_coordinator.ablation_manager.reset_all()

# Rank encoder importance
deltas = hooks.ablation_coordinator.compute_ablation_deltas(results)
ranking = hooks.ablation_coordinator.get_encoder_importance_ranking(deltas)
```

**Key Metrics**:
- `ablation_delta`: Performance drop when encoder removed
- `encoder_importance_ranking`: Sorted by abs(delta)

### 4. Utilization Hooks

**Purpose**: Track how downstream layers use encoder features

**Classes**:
- `AttentionWeightTracker` - Attention mass per modality
- `FeatureSimilarityTracker` - Layer-to-layer similarity
- `MutualInformationEstimator` - MI(features, actions)
- `DownstreamUtilizationAnalyzer` - Coordinator

**Usage**:
```python
hooks.attach_utilization_hooks()

# Set modality token ranges
modality_ranges = hooks.get_modality_token_ranges()
hooks.utilization_analyzer.attention_tracker.set_modality_ranges(modality_ranges)

# Run forward passes
for batch in dataloader:
    model(**batch)

# Compute attention distribution
attention_dist = hooks.utilization_analyzer.attention_tracker.compute_modality_attention()
# Returns: {"vision": 0.65, "proprio": 0.15, "language": 0.20}

# Find stagnant layers
stagnant = hooks.utilization_analyzer.similarity_tracker.find_stagnant_layers(threshold=0.95)
```

**Key Metrics**:
- `modality_attention`: Percentage of attention to each modality
- `stagnant_layers`: Layers with >95% similarity to previous
- `mutual_information`: MI(features, actions) in nats

## Model-Specific Adapters

Each model requires a custom adapter to locate encoders and attach hooks correctly.

### OpenVLA
```python
from hooks.model_specific import OpenVLAHooks

hooks = OpenVLAHooks(model)
hooks.discover_model_structure()  # Find vision + language encoders
hooks.print_model_info()

# Note: OpenVLA has no proprio encoder
# Only vision and language hooks available
```

**Architecture**:
- Vision: SigLIP (400M)
- Language: Llama-2-7B
- Fusion: Vision-as-prefix

### Octo
```python
from hooks.model_specific import OctoHooks

hooks = OctoHooks(model)
hooks.discover_model_structure()  # Find vision, proprio, language

# Proprio encoder is linear projection
# Single layer - no layer-wise profiling needed
```

**Architecture**:
- Vision: ResNet-34 or ViT
- Proprio: Linear + position embeddings
- Fusion: Concatenation
- Conditioning: Diffusion cross-attention

### RDT-1B
```python
from hooks.model_specific import RDTHooks

hooks = RDTHooks(model)
hooks.discover_model_structure()

# MLP encoder - layer-wise profiling available
# Fourier features - separate analysis
```

**Architecture**:
- Vision: ViT or CNN
- Proprio: Fourier → MLP (2-3 layers)
- Fusion: Concatenation

**Special Features**:
- Layer-wise gradient profiling through MLP
- Separate Fourier feature analysis
- Can ablate Fourier vs full MLP

### π0
```python
from hooks.model_specific import Pi0Hooks

hooks = Pi0Hooks(model)
hooks.discover_model_structure()

# Separate encoder - full profiling
# Causal attention layers - special tracking
```

**Architecture**:
- Vision: Pre-trained ViT
- Proprio: Separate multi-layer encoder
- Fusion: Block-wise causal masking
- Action: Flow matching

**Special Features**:
- Layer-wise profiling of separate encoder
- Causal attention tracking
- Mutual information with flow matching

## Running Experiments

### Experiment Coordinator

The `ExperimentCoordinator` runs consistent experiments across models:

```python
from analysis import ExperimentCoordinator

coordinator = ExperimentCoordinator(output_dir="./results")

# Register models
coordinator.register_model("octo", octo_model)
coordinator.register_model("openvla", openvla_model)
coordinator.register_model("rdt", rdt_model)
coordinator.register_model("pi0", pi0_model)

# Run individual analyses
grad_results = coordinator.run_gradient_analysis(dataloader, num_batches=10)
repr_results = coordinator.run_representation_analysis(dataloader, num_batches=50)
abl_results = coordinator.run_ablation_study(dataloader, eval_fn, num_batches=20)
util_results = coordinator.run_utilization_analysis(dataloader, num_batches=30)

# Or run full suite
all_results = coordinator.run_full_diagnostic(
    data_loader=dataloader,
    eval_fn=eval_fn,
    gradient_batches=10,
    representation_batches=50,
    ablation_batches=20,
    utilization_batches=30,
    models=["octo", "rdt"]  # Optional: specify subset
)
```

### Data Requirements

**Gradient Analysis**:
- Requires: Training batches with labels/targets
- Batches: ~10 (sufficient for gradient statistics)

**Representation Analysis**:
- Requires: Any input data
- Batches: ~50 (need enough samples for CKA)

**Ablation Study**:
- Requires: Evaluation function
- Batches: ~20 per configuration

**Utilization Analysis**:
- Requires: Any input data
- Batches: ~30 (attention statistics)

### Evaluation Function

For ablation studies, provide an evaluation function:

```python
def eval_fn(model, batch):
    """
    Compute performance metric on a single batch.
    
    Args:
        model: VLA model
        batch: Data batch (images, proprio, task, actions)
    
    Returns:
        float: Performance metric (lower is better for MSE)
    """
    images, proprio, task, actions = batch
    
    with torch.no_grad():
        outputs = model(images=images, proprio_state=proprio, task=task)
        predictions = outputs.logits  # or outputs directly
        
        # Action prediction error
        mse = F.mse_loss(predictions, actions)
        return mse.item()
```

## Analyzing Results

### Result Analyzer

```python
from analysis.result_analyzer import ResultAnalyzer

analyzer = ResultAnalyzer("./results")
analyzer.load_all_results()

# Generate plots
analyzer.plot_gradient_comparison()
analyzer.plot_representation_comparison()
analyzer.plot_ablation_results()
analyzer.plot_attention_distribution()

# Or all at once
analyzer.plot_all(output_dir="./results/plots")

# Text summary
summary = analyzer.generate_summary_report()
print(summary)
analyzer.save_summary_report("./results/summary.txt")
```

### Visualizations Generated

1. **Gradient Comparison** (2x2 grid):
   - Encoder gradient norms
   - Proprio/vision ratios
   - Vanishing point locations
   - Gradient decay rates

2. **Representation Quality** (2x2 grid):
   - Effective rank by encoder
   - Utilization percentages
   - CKA similarity matrix
   - Redundancy pair counts

3. **Ablation Results** (1x2 grid):
   - Performance drops by encoder
   - Importance rankings

4. **Attention Distribution**:
   - Stacked bar chart of modality percentages

## API Reference

### Base Hooks

```python
class BaseHook:
    def attach(self, module, name)  # Attach to module
    def remove()                     # Remove hook
    def enable()                     # Enable hook
    def disable()                    # Disable hook
    def get_results()                # Get collected results
    def reset()                      # Clear results
```

### Gradient Analysis

```python
class GradientFlowAnalyzer:
    def setup_encoder_tracking(vision, language, proprio)
    def setup_layer_profiling(name, layers, layer_names)
    def get_comprehensive_report() -> Dict
    def print_summary()
    def remove_all()
```

### Representation Analysis

```python
class RepresentationQualityAnalyzer:
    def setup(vision, language, proprio)
    def compute_all_similarities() -> Dict
    def compute_all_ranks() -> Dict
    def find_redundant_pairs(threshold=0.7) -> List
    def get_comprehensive_report() -> Dict
    def remove_hooks()
```

### Ablation Study


```python
class AblationStudyCoordinator:
    def setup(vision, language, proprio, ablation_type="zero")
    def run_standard_ablations() -> Dict[str, Dict]
    def compute_ablation_deltas(results, baseline_key) -> Dict
    def get_encoder_importance_ranking(deltas) -> List[Tuple]
    def cleanup()
```

### Utilization Analysis

```python
class DownstreamUtilizationAnalyzer:
    def __init__()
    attention_tracker: AttentionWeightTracker
    similarity_tracker: FeatureSimilarityTracker
    mi_estimator: MutualInformationEstimator
    
    def cleanup()
```

---

**Status**: Infrastructure complete ✅  
**Next Steps**: Test on real models with actual datasets

For examples, see `example_usage.py`
