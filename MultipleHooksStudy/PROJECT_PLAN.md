# VLA Proprioceptive Encoder Study - Project Plan

## Executive Summary

**Core Hypothesis**: Proprioceptive state encoders are underutilized in VLA models because vision encoders already capture spatial state information, making dedicated state encoders redundant regardless of their complexity.

**Test Strategy**: Compare 4 models with different state encoding approaches:
1. **OpenVLA** - No state encoder (baseline: is proprio even needed?)
2. **Octo** - Simple linear projection (minimal encoding)
3. **RDT-1B** - MLP with Fourier features (sophisticated encoding)
4. **π0** - Separate encoder with causal masking (advanced encoding)

**Key Question**: Does encoding complexity (None → Linear → MLP → Advanced) improve state encoder utilization, or does vision dominance persist across all approaches?

**Expected Finding**: All models show similar underutilization (~5-15% ablation impact) regardless of encoding sophistication, proving the issue is architectural/optimization, not lack of encoding capacity.

---

## Execution Environment: Google Colab

**All code runs in Colab notebooks. No local execution.**

### Colab Setup Requirements
- Runtime: GPU (T4 minimum, A100 preferred)
- RAM: High-RAM runtime (minimum 25GB, 40GB preferred)
- Storage: Mount Google Drive for checkpoints (~50GB required)
- Python: 3.10+

---

## Models: Exact Specifications

**Selection Rationale**: Models chosen to maximize diversity in proprioceptive state encoding approaches while maintaining manageable scope. Covers spectrum from no encoding to sophisticated MLP-based methods.

### Model 1: OpenVLA (7B) - **NO STATE ENCODER**
**Repository**: `https://github.com/openvla/openvla`  
**Checkpoint**: HuggingFace `openvla/openvla-7b`  
**Architecture**: Vision-as-prefix (SigLIP-400M + Llama-2-7B)  
**Proprioception Input**: **None** - Does not use proprioceptive information  
**State Encoding**: **No encoder** - Pure vision + language, no robot state  
**Fusion**: Concatenation at token level (vision tokens prepended to language)  
**Action Prediction**: Autoregressive token generation  
**Why This Model**: Baseline for "no state" approach. Tests if proprioception is even necessary. Well-documented, reproducible.

**Colab Loading**:
```python
from transformers import AutoModelForVision2Seq, AutoProcessor
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b")
```

---

### Model 2: Octo-Base (93M) - **LINEAR PROJECTION**
**Repository**: `https://github.com/octo-models/octo`  
**Checkpoint**: HuggingFace `rail-berkeley/octo-base`  
**Architecture**: Diffusion transformer with observation tokenizer  
**Proprioception Input**: 8-dim vector (7 joints + gripper state)  
**State Encoding**: **Linear projection** then added to position embeddings, concatenated into Transformer input  
**Fusion**: Cross-attention conditioning in diffusion process  
**Action Prediction**: Diffusion-based action generation  
**Why This Model**: Simplest form of state encoding. Different paradigm (diffusion). Tests if minimal encoding is sufficient.

**Colab Loading**:
```python
from octo.model.octo_model import OctoModel
model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
```

---

### Model 3: RDT-1B (1.2B) - **MLP WITH FOURIER FEATURES**
**Repository**: `https://github.com/thu-ml/RoboticsDiffusionTransformer`  
**Checkpoint**: HuggingFace `robotics-diffusion-transformer/rdt-1b`  
**Architecture**: Diffusion transformer with physically interpretable action space  
**Proprioception Input**: Variable dim (depends on embodiment, typically 7-15 dims)  
**State Encoding**: **MLP encoder with Fourier features** - Sophisticated encoding for physical state  
**Fusion**: Encoded state concatenated with vision/language features before diffusion  
**Action Prediction**: Diffusion in unified physically interpretable action space  
**Why This Model**: Represents sophisticated state encoding (MLP + Fourier). Tests if complex encoding improves utilization.

**Colab Loading**:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "robotics-diffusion-transformer/rdt-1b",
    trust_remote_code=True
)
```

---

### Model 4: π0 (Pi-Zero) (3.3B) - **SEPARATE ENCODER WITH CAUSAL MASKING**
**Repository**: `https://github.com/Physical-Intelligence/openpi`  
**Paper**: `https://arxiv.org/abs/2410.24164`  
**Architecture**: Flow matching with pre-trained VLM backbone  
**Proprioception Input**: Variable dim (embodiment-specific)  
**State Encoding**: **Proprioceptive state encoded separately with block-wise causal masking** - Advanced temporal encoding  
**Fusion**: Block-wise causal attention allows state to condition action prediction asymmetrically  
**Action Prediction**: Flow matching (continuous normalizing flows)  
**Why This Model**: Most advanced state encoding approach. Tests if architectural sophistication improves state utilization.

**Colab Loading**:
```python
# Note: π0 may require custom loading - check repo for latest instructions
from openpi.models import PiZeroModel
model = PiZeroModel.load_pretrained("physical-intelligence/pi0-3.3b")
```

**Fallback if π0 unavailable**: **OTTER** - Embodiment-specific MLP encoder, concatenated with language/vision features

---

## Model Coverage Summary

| Model | Size | State Encoding | Action Prediction | State Dims | Key Test |
|-------|------|----------------|-------------------|------------|----------|
| OpenVLA | 7B | **None** | Autoregressive | 0 | Is proprio needed? |
| Octo | 93M | **Linear projection** | Diffusion | 8 | Is simple encoding enough? |
| RDT-1B | 1.2B | **MLP + Fourier** | Diffusion | 7-15 | Does complex encoding help? |
| π0 | 3.3B | **Separate + causal** | Flow matching | Variable | Does advanced encoding win? |

**Architecture Diversity**:
- Autoregressive: OpenVLA
- Diffusion: Octo, RDT-1B
- Flow matching: π0

**State Encoding Spectrum**:
- No encoding → Linear → MLP → Advanced separate encoding

**Size Range**: 93M to 7B (allows scaling analysis)

---

## Hooks: Measurement & Intervention Methods

### Hook Category 1: Gradient Flow Analysis

#### Hook 1.1: Encoder Gradient Magnitude Tracker
**Implementation**: PyTorch backward hook on encoder output  
**Location**: Attach to:
- Vision encoder final layer
- Language encoder final layer  
- Proprioceptive encoder final layer

**Code Pattern (Colab cell)**:
```python
class GradientMagnitudeHook:
    def __init__(self):
        self.gradients = {}
    
    def hook_fn(self, module, grad_input, grad_output):
        if grad_output[0] is not None:
            self.gradients[module] = grad_output[0].norm().item()
        return None

# Usage
grad_hook = GradientMagnitudeHook()
model.vision_encoder.register_full_backward_hook(grad_hook.hook_fn)
model.proprio_encoder.register_full_backward_hook(grad_hook.hook_fn)
```

**Expected Result**:
- Vision encoder: grad norm = 10^-2 to 10^-1
- Proprio encoder: grad norm = 10^-4 to 10^-3
- Ratio (proprio/vision): **0.01 - 0.1** (confirms underutilization)

**Success Metric**: Reproduce <10% gradient ratio across models with state encoders (Octo, RDT-1B, π0). OpenVLA expected to show N/A (no state encoder).

---

#### Hook 1.2: Layer-wise Gradient Flow Profiler
**Implementation**: Hook at every layer of proprio encoder  
**Purpose**: Identify where gradients vanish (which layer)

**Code Pattern**:
```python
def attach_gradient_profiler(model, encoder_name='proprio_encoder'):
    grad_norms = {}
    encoder = getattr(model, encoder_name)
    
    for i, layer in enumerate(encoder.layers):
        def make_hook(layer_idx):
            def hook(module, grad_in, grad_out):
                if grad_out[0] is not None:
                    grad_norms[f'layer_{layer_idx}'] = grad_out[0].norm().item()
            return hook
        layer.register_full_backward_hook(make_hook(i))
    
    return grad_norms
```

**Expected Result**:
- Early layers (0-2): normal gradient flow
- Middle layers (3-5): gradient magnitude drops 10x
- Final layer: minimal gradient (indicating fusion bottleneck)

**Success Metric**: Identify exact layer where gradient drops below 10^-4.

---

### Hook Category 2: Representation Analysis

#### Hook 2.1: Feature Extractor Hook
**Implementation**: Forward hook to capture intermediate representations  
**Purpose**: Extract vision, language, proprio features before fusion

**Code Pattern**:
```python
class FeatureExtractorHook:
    def __init__(self):
        self.features = {}
    
    def get_features(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook

# Attach
feature_hook = FeatureExtractorHook()
model.vision_encoder.register_forward_hook(feature_hook.get_features('vision'))
model.proprio_encoder.register_forward_hook(feature_hook.get_features('proprio'))
```

**Expected Result**:
- Vision features: (batch, seq_len, 768-1024) - high dimensional
- Proprio features: (batch, 64-128) - low dimensional
- Feature statistics will show vision dominates by 10-15x in dimensionality

---

#### Hook 2.2: CKA Similarity Analyzer
**Implementation**: Post-inference computation between cached features  
**Purpose**: Measure redundancy between vision and proprio encoders

**Code Pattern**:
```python
def linear_CKA(X, Y):
    # X, Y: (n_samples, n_features)
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    return np.linalg.norm(X.T @ Y, 'fro')**2 / (
        np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro')
    )

# Usage after collecting features over 100+ samples
cka_score = linear_CKA(vision_features_stacked, proprio_features_stacked)
```

**Expected Result**:
- CKA(vision, proprio): **0.6 - 0.8** (high redundancy)
- CKA(language, proprio): 0.3 - 0.5 (moderate redundancy)
- Interpretation: Vision encoder already captures spatial state

**Success Metric**: CKA > 0.6 confirms redundancy hypothesis.

---

#### Hook 2.3: Effective Rank Calculator
**Implementation**: SVD on covariance matrix of encoder outputs  
**Purpose**: Measure intrinsic dimensionality (how many dimensions actually used)

**Code Pattern**:
```python
def effective_rank(features):
    # features: (n_samples, n_features)
    cov = np.cov(features.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Positive only
    
    # Effective rank via participation ratio
    return (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

# Usage
vision_rank = effective_rank(vision_features)
proprio_rank = effective_rank(proprio_features)
```

**Expected Result**:
- Vision encoder: effective rank = 200-400 out of 768 dims (50% utilization)
- Proprio encoder: effective rank = 5-15 out of 64 dims (**15-25% utilization**)
- Interpretation: Proprio encoder severely underutilized

**Success Metric**: Proprio effective rank < 30% of total dimensions.

---

### Hook Category 3: Ablation Mechanisms

#### Hook 3.1: Zero-Out Ablation Hook
**Implementation**: Forward hook that zeros encoder output  
**Purpose**: Measure performance drop when removing modality

**Code Pattern**:
```python
class AblationHook:
    def __init__(self, ablate=False):
        self.ablate = ablate
    
    def hook_fn(self, module, input, output):
        if self.ablate:
            return torch.zeros_like(output)
        return output

# Usage
ablation_hook = AblationHook(ablate=True)
handle = model.proprio_encoder.register_forward_hook(ablation_hook.hook_fn)

# Run eval with ablation
success_rate_ablated = evaluate_model(model, tasks)
handle.remove()  # Remove hook

# Run eval without ablation
success_rate_full = evaluate_model(model, tasks)

delta = success_rate_full - success_rate_ablated
```

**Expected Result**:
- Ablate vision: Δ = **-40% to -60%** (critical modality)
- Ablate language: Δ = -25% to -40% (important for conditioning)
- Ablate proprio: Δ = **-5% to -15%** (confirms underutilization)

**Success Metric**: Proprio ablation causes <15% drop on spatial precision tasks.

---

#### Hook 3.2: Noise Injection Evaluator
**Implementation**: Add Gaussian noise to encoder output  
**Purpose**: Test robustness (well-utilized encoders should be noise-sensitive)

**Code Pattern**:
```python
class NoiseInjectionHook:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
    
    def hook_fn(self, module, input, output):
        noise = torch.randn_like(output) * self.noise_std
        return output + noise

# Test at different noise levels
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
results = {}
for noise_std in noise_levels:
    hook = NoiseInjectionHook(noise_std)
    handle = model.proprio_encoder.register_forward_hook(hook.hook_fn)
    results[noise_std] = evaluate_model(model, tasks)
    handle.remove()
```

**Expected Result**:
- Vision encoder: performance degrades smoothly with noise (utilized)
- Proprio encoder: **minimal degradation even at high noise** (not utilized)
- At 0.2 std noise: vision Δ = -20%, proprio Δ = **-2%**

**Success Metric**: Proprio encoder shows <5% degradation at 0.2 noise level.

---

### Hook Category 4: Information Flow Measurement

#### Hook 4.1: Mutual Information Estimator
**Implementation**: MINE (Mutual Information Neural Estimation) between proprio features and actions  
**Purpose**: Quantify how much action-relevant information is in proprio encoding

**Code Pattern**:
```python
# Use existing MINE implementation
from mine import MINE  # https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation

def estimate_MI(proprio_features, actions, epochs=100):
    mine = MINE(input_dim=proprio_features.shape[1], 
                hidden_dim=64, 
                output_dim=actions.shape[1])
    
    # Train MINE network
    for epoch in range(epochs):
        mi_estimate = mine.train_step(proprio_features, actions)
    
    return mi_estimate

# Compare across modalities
mi_proprio_action = estimate_MI(proprio_features, actions)
mi_vision_action = estimate_MI(vision_features, actions)
```

**Expected Result**:
- MI(vision, action): **2.5 - 4.0 nats**
- MI(proprio, action): **0.3 - 0.8 nats** (much lower)
- MI ratio: proprio/vision = **0.1 - 0.2**

**Success Metric**: MI(proprio, action) < 1.0 nats, ratio < 0.25.

---

#### Hook 4.2: Attention Weight Tracker (Cross-Attention Models Only)
**Implementation**: Forward hook on attention layer outputs  
**Purpose**: Measure what fraction of attention goes to proprio tokens

**Code Pattern**:
```python
class AttentionWeightHook:
    def __init__(self):
        self.attention_weights = []
    
    def hook_fn(self, module, input, output):
        # output typically includes attention weights
        # Format varies by architecture
        attn_weights = output[1]  # Adjust based on model
        self.attention_weights.append(attn_weights.detach().cpu())
        return output

# For models with cross-attention or causal masking (Octo, π0)
attn_hook = AttentionWeightHook()
model.cross_attention_layer.register_forward_hook(attn_hook.hook_fn)  # For Octo
# OR
model.causal_attention_layer.register_forward_hook(attn_hook.hook_fn)  # For π0

# Analyze: what % of attention mass goes to proprio tokens/features?
```

**Expected Result** (for attention-based models):
| Model | Vision Attention | Language Attention | Proprio Attention | Mechanism |
|-------|-----------------|-------------------|------------------|------------|
| Octo | 65-75% | 20-30% | **3-7%** | Cross-attention |
| π0 | 60-70% | 25-35% | **5-10%** | Causal masking |
| RDT-1B | 70-80% | 15-25% | **2-5%** | Concatenation |

**Note**: OpenVLA has no proprio input. π0 expected to show slightly higher proprio attention due to separate encoding with causal masking, but still underweighted.

**Success Metric**: Proprio receives <12% of attention mass across all models. π0 may reach 10-12% (best case) but still far below vision.

---

## Week 1 Experiments: Diagnostic Phase

### Experiment 1: Gradient Flow Characterization
**Models**: OpenVLA, Octo, RDT-1B, π0  
**Hooks Used**: 1.1 (Encoder Gradient Tracker), 1.2 (Layer-wise Profiler)  
**Procedure**:
1. Load model in Colab
2. Load 100 samples from LIBERO benchmark
3. Attach gradient hooks to all encoders (where applicable)
4. Run forward + backward pass (loss from action prediction)
5. Record gradient norms at each encoder

**Note**: OpenVLA has no state encoder, so gradient analysis focuses on vision/language only. Other models tested for state encoder gradient flow.

**Colab Notebook Structure**:
```
# Cell 1: Setup
!git clone https://github.com/openvla/openvla
!pip install -e openvla
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Load model + data
model = load_openvla_model()
dataset = load_libero_data(n_samples=100)

# Cell 3: Attach hooks
grad_hook = attach_gradient_hooks(model)

# Cell 4: Run gradient analysis
gradient_results = run_gradient_analysis(model, dataset, grad_hook)

# Cell 5: Visualize
plot_gradient_ratios(gradient_results)
```

**Expected Output** (per model):
| Model | Vision Grad | Proprio Grad | Ratio | Layer Where Vanishing | State Encoding |
|-------|-------------|--------------|-------|----------------------|----------------|
| OpenVLA | 0.08 | N/A (no encoder) | N/A | N/A | None |
| Octo | 0.12 | 0.008 | 0.067 | Cross-attn layer | Linear projection |
| RDT-1B | 0.09 | 0.006 | 0.067 | MLP layer 3/4 | MLP + Fourier |
| π0 | 0.10 | 0.012 | 0.12 | Causal mask layer | Separate + causal |

**Success Criteria**: Models with state encoders show ratio < 0.15. π0 may show slightly higher ratio due to advanced encoding, but still underutilized relative to vision.

---

### Experiment 2: Representation Redundancy Analysis
**Models**: Octo, RDT-1B, π0 (OpenVLA excluded - no state encoder)  
**Hooks Used**: 2.1 (Feature Extractor), 2.2 (CKA Similarity), 2.3 (Effective Rank)  
**Procedure**:
1. Extract features from all encoders on 500 samples (larger dataset for statistics)
2. Compute CKA between vision-proprio, language-proprio
3. Compute effective rank for each encoder
4. Compute correlation between proprio features and visual observations

**Note**: OpenVLA serves as control - no state features to analyze. Focus on three models with different encoding complexities.

**Dataset**: Bridge V2 (diverse tasks, available via HuggingFace)
```python
from datasets import load_dataset
dataset = load_dataset("rail-berkeley/bridge_orig", split="train[:500]")
```

**Expected Output**:
| Model | CKA(vision, proprio) | Effective Rank (proprio) | Proprio Dims Used | Encoding Type |
|-------|---------------------|-------------------------|------------------|---------------|
| Octo | 0.68 | 6 / 8 | 75% | Linear (simple) |
| RDT-1B | 0.65 | 8 / 15 | 53% | MLP + Fourier |
| π0 | 0.58 | 12 / 20 | 60% | Separate + causal |

**Interpretation**: 
- Even with sophisticated encoding (π0), CKA remains high (>0.55) - vision still captures most spatial state
- More complex encoders (RDT, π0) show better effective rank utilization than simple linear (Octo)
- But still majority of dimensions underutilized even with MLP/Fourier features

**Success Criteria**: CKA > 0.55 for all models. Effective rank improves with encoding complexity: Linear < MLP < Advanced.

---

### Experiment 3: Ablation Study Across Task Types
**Models**: All 4 models (OpenVLA, Octo, RDT-1B, π0)  
**Hooks Used**: 3.1 (Zero-Out Ablation)  
**Procedure**:
1. Define task categories:
   - **Precision tasks**: peg insertion, small object pick, stacking (10 tasks)
   - **Coarse tasks**: pushing, toppling, reaching (10 tasks)
   - **Dynamic tasks**: throwing, sliding, wiping (10 tasks)
2. For each task category, run:
   - Full model (baseline)
   - Ablate vision
   - Ablate proprio
   - Ablate language
3. Measure success rate (50 episodes per task)

**Colab Execution**:
```python
# Use MetaWorld for quick iteration
import metaworld
ml1 = metaworld.ML1('pick-place-v2')

task_categories = {
    'precision': ['peg-insert-side-v2', 'pick-place-v2', 'stack-v2'],
    'coarse': ['push-v2', 'reach-v2', 'door-open-v2'],
    'dynamic': ['sweep-into-v2', 'basketball-v2']
}

results = {}
for category, tasks in task_categories.items():
    for task in tasks:
        results[task] = {
            'full': evaluate_task(model, task, ablate=None),
            'ablate_vision': evaluate_task(model, task, ablate='vision'),
            'ablate_proprio': evaluate_task(model, task, ablate='proprio'),
            'ablate_language': evaluate_task(model, task, ablate='language')
        }
```

**Expected Output** (averaged across models with state encoders):
| Task Type | Full | -Vision | -Proprio | -Language | Notes |
|-----------|------|---------|----------|-----------|-------|
| Precision | 78% | 32% | **70%** (Δ=-8%) | 45% | OpenVLA (no proprio) baseline: 76% |
| Coarse | 85% | 38% | **80%** (Δ=-5%) | 52% | OpenVLA baseline: 84% |
| Dynamic | 72% | 28% | **68%** (Δ=-4%) | 48% | OpenVLA baseline: 70% |

**Model-Specific Findings**:
- **OpenVLA**: No proprio encoder yet performs within 2-3% of models with state - confirms hypothesis
- **Octo** (linear): -6% on precision tasks
- **RDT-1B** (MLP): -9% on precision tasks (slightly better)
- **π0** (advanced): -11% on precision tasks (best, but still minimal)

**Key Finding**: Even sophisticated state encoding (π0) shows <12% ablation impact. OpenVLA performs comparably without any state encoder.

**Success Criteria**: Proprio ablation Δ < 15% on precision tasks across all models. OpenVLA baseline within 5% of models with state.

---

### Experiment 4: Information Content Measurement
**Models**: Octo, RDT-1B, π0 (OpenVLA excluded - no state encoder)  
**Hooks Used**: 4.1 (MI Estimator), 2.1 (Feature Extractor)  
**Procedure**:
1. Collect (proprio_features, action) pairs over 1000 episodes
2. Train MINE network to estimate MI(proprio, action)
3. Compare with MI(vision, action) and MI(language, action)
4. Compute conditional MI: MI(proprio, action | vision)

**Expected Output** (averaged across 3 models with state):
| Modality | MI with Action (nats) | Δ When Conditioned on Vision | Best Model |
|----------|----------------------|------------------------------|------------|
| Vision | 3.2 | - | - |
| Language | 2.1 | - | - |
| Proprio (Octo - linear) | **0.5** | **0.08** (near-zero!) | - |
| Proprio (RDT - MLP) | **0.7** | **0.12** | - |
| Proprio (π0 - advanced) | **0.9** | **0.18** | π0 |

**Interpretation**: 
- Even best state encoder (π0) provides only 0.9 nats vs 3.2 for vision
- Conditional MI nearly zero for all: vision already captures spatial state
- More sophisticated encoding (π0) does provide more unique information (0.18 vs 0.08) but still minimal

**Success Criteria**: MI(proprio, action | vision) < 0.25 nats for all models. Advanced encoders show higher MI than simple, but all remain low.

---

## Week 2 Experiments: Improvement Phase

### State Encoding Variant 1: Hierarchical Spatial-Temporal Encoding

**Implementation Hook**: Replace proprio encoder with custom two-stage encoder

**Architecture**:
```python
class HierarchicalProprioEncoder(nn.Module):
    def __init__(self, input_dim=8, spatial_dim=32, temporal_dim=32):
        super().__init__()
        # Stage 1: Spatial encoding (positions)
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, spatial_dim)
        )
        
        # Stage 2: Temporal encoding (velocities via SSM)
        self.temporal_encoder = Mamba(
            d_model=temporal_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # Fusion
        self.fusion = nn.Linear(spatial_dim + temporal_dim, 64)
    
    def forward(self, proprio_state, proprio_history):
        # proprio_state: current positions
        # proprio_history: last 10 timesteps for velocities
        
        spatial_features = self.spatial_encoder(proprio_state)
        temporal_features = self.temporal_encoder(proprio_history)
        
        return self.fusion(torch.cat([spatial_features, temporal_features], dim=-1))
```

**Colab Integration**:
```python
# Cell: Replace encoder
original_encoder = model.proprio_encoder
new_encoder = HierarchicalProprioEncoder()

# Residual integration (don't disrupt pretrained weights)
class ResidualEncoderWrapper(nn.Module):
    def __init__(self, original, new, alpha=0.2):
        super().__init__()
        self.original = original
        self.new = new
        self.alpha = alpha
    
    def forward(self, x, history):
        return (1 - self.alpha) * self.original(x) + self.alpha * self.new(x, history)

model.proprio_encoder = ResidualEncoderWrapper(original_encoder, new_encoder)
```

**Experiment**:
1. Test on dynamic tasks (throwing, sliding) where velocity matters
2. Measure gradient flow to new encoder
3. Ablate new encoder vs original

**Expected Results**:
| Metric | Original | Hierarchical | Δ |
|--------|----------|-------------|---|
| Dynamic Task Success | 72% | **81%** | +9% |
| Gradient Magnitude | 0.004 | **0.015** | 3.75x |
| Ablation Impact | -4% | **-12%** | 3x more critical |

**Success Criteria**: Gradient magnitude increases >2x, dynamic task improvement >5%.

---

### State Encoding Variant 2: Task-Conditioned Proprioception

**Implementation Hook**: Cross-attention between task embedding and proprio state

**Architecture**:
```python
class TaskConditionedProprioEncoder(nn.Module):
    def __init__(self, proprio_dim=8, task_embed_dim=512, output_dim=64):
        super().__init__()
        self.proprio_projection = nn.Linear(proprio_dim, 128)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            kdim=task_embed_dim,
            vdim=task_embed_dim
        )
        self.output_projection = nn.Linear(128, output_dim)
    
    def forward(self, proprio_state, task_embedding):
        # Task embedding comes from language encoder
        proprio_query = self.proprio_projection(proprio_state).unsqueeze(1)
        task_kv = task_embedding.unsqueeze(1)
        
        attended, _ = self.cross_attention(
            query=proprio_query,
            key=task_kv,
            value=task_kv
        )
        
        return self.output_projection(attended.squeeze(1))
```

**Experiment**:
1. Test on multi-task benchmark (10 diverse tasks)
2. Visualize which joints get attended to for each task
3. Measure CKA(vision, proprio) - should decrease
4. Compare against baseline models (especially π0 which already has advanced encoding)

**Expected Results**:
| Metric | Baseline | Task-Conditioned | Δ |
|--------|----------|-----------------|---|
| CKA(vision, proprio) | 0.72 | **0.48** | -33% redundancy |
| Precision Task Success | 78% | **86%** | +8% |
| Attention to Proprio | 5% | **18%** | 3.6x |

**Success Criteria**: CKA drops below 0.55, attention to proprio >15%.

---

### State Encoding Variant 3: Compressed Multi-Resolution Encoding

**Implementation Hook**: Vector quantization bottleneck + multi-scale encoding

**Architecture**:
```python
class CompressedMultiResolutionEncoder(nn.Module):
    def __init__(self, proprio_dim=8, codebook_size=256, output_dim=64):
        super().__init__()
        # Fine-grained: all joints
        self.fine_encoder = nn.Linear(proprio_dim, 32)
        
        # Coarse: end-effector pose (derived from joints)
        self.coarse_encoder = nn.Linear(6, 16)  # xyz + rpy
        
        # Vector quantization bottleneck
        self.vq = VectorQuantize(
            dim=48,  # 32 + 16
            codebook_size=codebook_size,
            commitment_weight=0.25
        )
        
        self.output = nn.Linear(48, output_dim)
    
    def forward(self, proprio_state):
        fine_features = self.fine_encoder(proprio_state)
        
        # Compute end-effector pose from joint positions (forward kinematics)
        ee_pose = self.compute_ee_pose(proprio_state)
        coarse_features = self.coarse_encoder(ee_pose)
        
        multi_res = torch.cat([fine_features, coarse_features], dim=-1)
        
        quantized, vq_loss = self.vq(multi_res)
        
        return self.output(quantized), vq_loss
```

**Experiment**:
1. Measure effective rank before and after compression
2. Test if 16-32 dim compressed encoding matches 64 dim baseline
3. Analyze codebook usage (are all codes used?)

**Expected Results**:
| Metric | 64-dim Baseline | 32-dim Compressed | 16-dim Compressed |
|--------|----------------|-------------------|-------------------|
| Task Success | 78% | **77%** (-1%) | 74% (-4%) |
| Effective Rank | 12 / 64 (19%) | **14 / 32 (44%)** | 10 / 16 (63%) |
| Gradient Magnitude | 0.004 | **0.011** | 0.009 |

**Interpretation**: 32-dim compressed encoding achieves similar performance but 2.3x higher dimensionality utilization (forced compression increases information density).

**Success Criteria**: 32-dim encoding achieves within 2% of 64-dim performance with >40% effective rank.

---

### State Encoding Variant 4: Differential State Encoding

**Implementation Hook**: Encode deltas instead of absolute positions

**Architecture**:
```python
class DifferentialProprioEncoder(nn.Module):
    def __init__(self, proprio_dim=8, output_dim=64):
        super().__init__()
        # Reference pose (learned or set to neutral)
        self.register_buffer('reference_pose', torch.zeros(proprio_dim))
        
        self.delta_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Also encode velocity (temporal derivative)
        self.velocity_encoder = nn.Linear(proprio_dim, output_dim)
        
        self.fusion = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, proprio_state, prev_proprio_state):
        # Spatial delta from reference
        delta_from_ref = proprio_state - self.reference_pose
        position_features = self.delta_encoder(delta_from_ref)
        
        # Temporal delta (velocity)
        velocity = proprio_state - prev_proprio_state
        velocity_features = self.velocity_encoder(velocity)
        
        return self.fusion(torch.cat([position_features, velocity_features], dim=-1))
```

**Experiment**:
1. Test on tasks with significant motion (not static manipulation)
2. Measure CKA(vision, proprio) - should decrease since vision shows absolute pose
3. Compare on dynamic vs static tasks

**Expected Results**:
| Task Type | Absolute Encoding | Differential Encoding | Δ |
|-----------|------------------|----------------------|---|
| Static (pick-place) | 78% | 76% | -2% (slightly worse) |
| Dynamic (throwing) | 72% | **79%** | +7% (better) |
| CKA with Vision | 0.72 | **0.52** | -28% |

**Interpretation**: Differential encoding reduces redundancy with vision (which captures absolute pose). Better for dynamic tasks.

**Success Criteria**: CKA < 0.6, dynamic task improvement > 5%.

---

## Fusion Mechanism Experiments

### Fusion Variant 1: Gated Modality Fusion

**Implementation**:
```python
class GatedFusion(nn.Module):
    def __init__(self, vision_dim=768, language_dim=512, proprio_dim=64, output_dim=512):
        super().__init__()
        # Learned gates per modality
        self.vision_gate = nn.Sequential(
            nn.Linear(vision_dim + language_dim + proprio_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.language_gate = nn.Sequential(...)  # Same structure
        self.proprio_gate = nn.Sequential(...)    # Same structure
        
        self.fusion = nn.Linear(vision_dim + language_dim + proprio_dim, output_dim)
    
    def forward(self, vision_feat, language_feat, proprio_feat):
        combined = torch.cat([vision_feat, language_feat, proprio_feat], dim=-1)
        
        gate_v = self.vision_gate(combined)
        gate_l = self.language_gate(combined)
        gate_p = self.proprio_gate(combined)
        
        # Normalize gates
        gate_sum = gate_v + gate_l + gate_p + 1e-8
        gate_v, gate_l, gate_p = gate_v / gate_sum, gate_l / gate_sum, gate_p / gate_sum
        
        weighted = torch.cat([
            gate_v * vision_feat,
            gate_l * language_feat,
            gate_p * proprio_feat
        ], dim=-1)
        
        return self.fusion(weighted), (gate_v, gate_l, gate_p)
```

**Experiment**:
1. Track gate values across different tasks
2. Measure if proprio gate increases (better balance)
3. Compare gradient flow

**Expected Results**:
| Metric | Concatenation | Gated Fusion | Δ |
|--------|--------------|--------------|---|
| Proprio Gate Value | N/A (implicit ~5%) | **15-20%** | Explicit balancing |
| Gradient to Proprio | 0.004 | **0.009** | 2.25x |
| Task Success | 78% | **82%** | +4% |

**Success Criteria**: Proprio gate > 12%, gradient increase > 2x.

---

### Fusion Variant 2: Hierarchical Staged Fusion

**Implementation**:
```python
class HierarchicalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1: Fuse vision + proprio (spatial modalities)
        self.spatial_fusion = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8
        )
        
        # Stage 2: Fuse spatial result with language (task conditioning)
        self.task_fusion = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8
        )
    
    def forward(self, vision_feat, proprio_feat, language_feat):
        # Stage 1: Spatial fusion
        spatial_query = vision_feat.unsqueeze(1)
        proprio_kv = proprio_feat.unsqueeze(1)
        
        spatial_fused, spatial_attn = self.spatial_fusion(
            query=spatial_query,
            key=proprio_kv,
            value=proprio_kv
        )
        
        # Stage 2: Task conditioning
        task_query = spatial_fused
        language_kv = language_feat.unsqueeze(1)
        
        final_fused, task_attn = self.task_fusion(
            query=task_query,
            key=language_kv,
            value=language_kv
        )
        
        return final_fused, (spatial_attn, task_attn)
```

**Experiment**:
1. Test on spatial precision tasks
2. Measure attention to proprio in stage 1 (should be higher than baseline)
3. Compare with flat fusion

**Expected Results**:
| Task Type | Flat Fusion | Hierarchical | Δ |
|-----------|------------|-------------|---|
| Precision Spatial | 78% | **85%** | +7% |
| Proprio Attention | 5% | **22%** | 4.4x |
| Gradient to Proprio | 0.004 | **0.013** | 3.25x |

**Success Criteria**: Spatial task improvement > 5%, attention > 18%.

---

## Colab Notebook Structure

### Setup Notebook (`1_setup_environment.ipynb`)
**Purpose**: One-time setup for all experiments

**Cells**:
1. Mount Google Drive
2. Clone repositories (OpenVLA, Octo, RDT, π0)
3. Install dependencies (PyTorch, Transformers, Octo, RDT, OpenPI)
4. Download model checkpoints to Drive (openvla-7b, octo-base, rdt-1b, pi0-3.3b)
5. Download benchmark datasets (LIBERO, MetaWorld, Bridge)
6. Verify GPU availability (A100 required for 7B model)

**Output**: Checkpoint paths, data paths saved to config file.

---

### Diagnostic Notebook (`2_diagnostic_analysis.ipynb`)
**Purpose**: Run all Week 1 experiments

**Cells**:
1. Load models (OpenVLA, Octo, RDT-1B, π0)
2. **Experiment 1**: Gradient flow analysis
   - Output: Gradient ratio table, layer-wise plots
3. **Experiment 2**: Representation redundancy
   - Output: CKA matrix, effective rank table
4. **Experiment 3**: Ablation study
   - Output: Performance table by task type
5. **Experiment 4**: Information content
   - Output: MI estimates, conditional MI
6. Summary visualization: Combined plots

**Runtime**: ~6 hours on A100

---

### Encoding Variants Notebook (`3_state_encoding_variants.ipynb`)
**Purpose**: Test 4 encoding variants

**Cells**:
1. Load baseline model (OpenVLA)
2. **Variant 1**: Hierarchical encoding
   - Train/evaluate, output results table
3. **Variant 2**: Task-conditioned encoding
   - Train/evaluate, output results table
4. **Variant 3**: Compressed encoding
   - Train/evaluate, output results table
5. **Variant 4**: Differential encoding
   - Train/evaluate, output results table
6. Comparison: All variants vs baseline
   - Output: Combined performance table, gradient flow comparison

**Runtime**: ~10 hours (2.5 hours per variant)

---

### Fusion Mechanisms Notebook (`4_fusion_mechanisms.ipynb`)
**Purpose**: Test fusion strategies

**Cells**:
1. Load baseline model (start with Octo - simplest state encoder)
2. **Fusion 1**: Gated fusion
3. **Fusion 2**: Hierarchical fusion
4. Comparison across models (Octo, RDT-1B, π0) - test if fusion improvements generalize
5. OpenVLA analysis: Can we add state encoder + fusion and see improvement?
6. Visualization: Attention patterns, gate values

**Runtime**: ~8 hours

---

### Final Analysis Notebook (`5_final_analysis.ipynb`)
**Purpose**: Aggregate all results, create publication-ready plots

**Cells**:
1. Load all experiment results
2. Cross-model comparison
3. Best configuration identification
4. Generalization analysis
5. Export final tables and figures

**Runtime**: ~1 hour

---

## Success Criteria Summary

### Week 1 (Diagnostic)
- [ ] **Gradient ratio** (proprio/vision) < 0.15 across models with state encoders (Octo, RDT-1B, π0)
- [ ] **CKA similarity** (vision, proprio) > 0.55 across models with state encoders
- [ ] **Effective rank** (proprio) increases with encoding complexity: Linear (Octo) < MLP (RDT) < Advanced (π0), but all still <70% utilization
- [ ] **Proprio ablation impact** < 15% on precision tasks across all models with state
- [ ] **OpenVLA baseline**: Performs within 5% of models with state encoders (proving state encoder underutilization)
- [ ] **MI(proprio, action | vision)** < 0.25 nats, with π0 showing highest (~0.18) but still minimal

### Week 2 (Improvements)
- [ ] At least 1 encoding variant achieves >2x gradient increase over baseline
- [ ] At least 1 encoding variant achieves >5% task improvement over baseline
- [ ] At least 1 encoding variant reduces CKA to < 0.5 (lower redundancy with vision)
- [ ] At least 1 fusion mechanism increases proprio attention to >15%
- [ ] Best configuration generalizes across 2+ models (different encoding types)
- [ ] **Critical test**: Can improved state encoding make OpenVLA perform better when added?

### Overall
- [ ] Identify root cause of proprio underutilization across encoding types (documented with evidence)
- [ ] Quantify relationship between encoding complexity and utilization (Linear vs MLP vs Advanced)
- [ ] Provide actionable recommendations for VLA architecture design
- [ ] Demonstrate measurable improvement in proprio encoder utilization that transfers across architectures

---

## File Structure in Colab

```
/content/
├── drive/MyDrive/EmbodiedLLM/
│   ├── checkpoints/
│   │   ├── openvla-7b/
│   │   ├── octo-base/
│   │   ├── rdt-1b/
│   │   └── pi0-3.3b/
│   ├── datasets/
│   │   ├── libero/
│   │   ├── metaworld/
│   │   └── bridge/
│   ├── results/
│   │   ├── week1_diagnostics/
│   │   │   ├── gradient_flow.pkl
│   │   │   ├── cka_results.pkl
│   │   │   └── ablation_results.pkl
│   │   └── week2_improvements/
│   │       ├── variant1_results.pkl
│   │       ├── fusion_results.pkl
│   │       └── final_comparison.pkl
│   └── notebooks/
│       ├── 1_setup_environment.ipynb
│       ├── 2_diagnostic_analysis.ipynb
│       ├── 3_state_encoding_variants.ipynb
│       ├── 4_fusion_mechanisms.ipynb
│       └── 5_final_analysis.ipynb
├── openvla/  (cloned repo)
├── octo/     (cloned repo)
├── RoboticsDiffusionTransformer/  (cloned repo)
├── openpi/   (cloned repo)
└── MultipleHooksStudy/  (cloned from your repo)
    └── hooks/
        ├── gradient_hooks.py
        ├── feature_hooks.py
        ├── ablation_hooks.py
        └── fusion_modules.py
```

**All code modularized in `MultipleHooksStudy/hooks/` directory, imported into notebooks.**

---

## Data Requirements

### LIBERO Benchmark
- Size: ~15GB
- Download: `wget https://utexas.box.com/shared/static/libero_data.zip`
- Tasks: 90 long-horizon manipulation tasks

### MetaWorld
- Size: ~500MB (environments only, generated on-the-fly)
- Install: `pip install metaworld`
- Tasks: 50 single-task environments

### Bridge V2
- Size: ~50GB (use subset for analysis)
- Download: HuggingFace `datasets.load_dataset("rail-berkeley/bridge_orig")`
- Purpose: Real-world data for CKA/MI analysis

---

## Compute Budget Estimate

### Week 1 (Diagnostic)
- Model loading: 4 models × 30 min = 2 hours
- Experiment 1 (gradient flow): 3 models with state × 100 samples = 2 hours
- Experiment 2 (CKA analysis): 3 models × 500 samples = 4 hours
- Experiment 3 (ablation): 4 models × 30 tasks × 50 episodes = 12 hours
- Experiment 4 (MI estimation): 3 models × 1000 episodes = 6 hours
**Total**: ~26 hours GPU time

### Week 2 (Improvements)
- Variant development × 4: 10 hours
- Fusion experiments × 2: 8 hours
- Cross-model validation: 6 hours
**Total**: ~24 hours GPU time

### Grand Total: ~50 GPU hours
**Recommended**: Colab Pro+ (A100, 40GB RAM) = ~$50 for 100 hours
**Note**: 7B model (OpenVLA) and 3.3B model (π0) require A100. Can use T4/V100 for Octo (93M) and RDT-1B if needed.

---

## Risk Mitigation

### Risk 1: Model Checkpoint Unavailable
**Mitigation**: 
- OpenVLA, Octo, RDT-1B all have verified public HuggingFace checkpoints
- π0: Check Physical Intelligence GitHub for latest checkpoint access
- **Fallback hierarchy**: π0 unavailable → use OTTER (embodiment-specific MLP) or DreamVLA (MLP to token space)
- If needed, can substitute with multiple Octo variants (Small-27M, Base-93M) for scaling analysis

### Risk 2: OOM on Colab
**Mitigation**: 
- Use gradient checkpointing
- Batch size = 1 for inference
- Offload encoders to CPU when not in use
- Use Octo-Base (93M) instead of Octo-Large if needed

### Risk 3: Benchmark Environment Issues
**Mitigation**:
- Primary: MetaWorld (lightweight, well-maintained)
- Fallback: Simpler gym-robotics environments
- Last resort: Evaluation on held-out Bridge subset (no simulator needed)

### Risk 4: Hook Integration Fails
**Mitigation**:
- Test hooks on minimal examples first (single forward/backward pass)
- Use try-except wrappers to catch hook errors
- Fallback: Modify source code directly if hooks don't work

---

## Deliverables Checklist

- [ ] 5 Colab notebooks (setup, diagnostics, variants, fusion, analysis)
- [ ] `hooks/` Python module with all hook implementations
- [ ] Results pickle files for each experiment
- [ ] **Model comparison tables**: 
  - [ ] Gradient flow across 4 models (OpenVLA, Octo, RDT-1B, π0)
  - [ ] Representation quality (CKA, effective rank) across 3 models with state
  - [ ] Ablation impact across all 4 models
  - [ ] Encoding complexity vs utilization analysis
- [ ] Visualization plots (gradient flow, CKA matrices, attention heatmaps, performance bars)
- [ ] **Critical analysis**: 
  - [ ] Why does OpenVLA (no state) perform comparably to models with state encoders?
  - [ ] Does encoding complexity (Linear → MLP → Advanced) improve utilization?
  - [ ] What information is vision encoder implicitly capturing?
- [ ] Actionable recommendations for future VLA architectures
- [ ] Code on GitHub (MultipleHooksStudy folder updated)

---

## Key Research Questions Answered

1. **Is proprioceptive state even necessary?**
   - OpenVLA (no state) vs models with state encoders
   - Expected: <5% performance difference on most tasks

2. **Does encoding complexity matter?**
   - Linear (Octo) vs MLP+Fourier (RDT) vs Separate+Causal (π0)
   - Expected: Better encoding → higher gradient flow, but still underutilized

3. **Why is vision sufficient?**
   - CKA analysis: vision features already contain spatial state information
   - MI analysis: conditional MI nearly zero - vision captures what proprio would provide

4. **Can we design better state encoders?**
   - Test 4 variants in Week 2
   - Best variant should: reduce CKA, increase gradient flow, improve precision tasks

---

## Next Steps

1. **Validate this plan**: Review all hooks, expected results, success criteria
2. **Create setup notebook**: Test model loading on Colab
3. **Implement Hook 1.1**: Gradient magnitude tracker (simplest hook)
4. **Run mini-experiment**: Test gradient tracking on 10 samples OpenVLA
5. **If successful**: Proceed with full Week 1 experiments
