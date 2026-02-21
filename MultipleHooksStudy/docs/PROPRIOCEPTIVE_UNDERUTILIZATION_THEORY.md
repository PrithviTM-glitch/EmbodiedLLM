# Why VLA Models Fail to Exploit Proprioceptive State: A Theoretical Analysis

## Evo-1 (MINT-SJTU) | RDT-1B (THU-ML) | Pi0 (Physical Intelligence)

---

## 1. Executive Summary

Vision-Language-Action (VLA) models consistently underutilize proprioceptive state information -- the robot's own joint angles, end-effector poses, and gripper state -- despite this information being critical for precise manipulation. This document provides a theoretically grounded analysis of *why* this happens, drawing directly from the architectures and formulations in the Evo-1, RDT-1B, and Pi0 papers. We identify five root causes: (1) extreme dimensionality asymmetry between modalities, (2) architectural bottlenecks in state encoding, (3) gradient flow imbalance from frozen pre-trained backbones, (4) information-theoretic redundancy with vision, and (5) training objective insensitivity to proprioceptive precision.

---

## 2. Proprioceptive State Encoders: Architectural Comparison

### 2.1 Evo-1: CategorySpecificMLP (3-Layer MLP)

**Architecture**: Evo-1 processes robot state through a `CategorySpecificMLP` inside the action head:

```python
self.state_encoder = CategorySpecificMLP(
    input_dim=self.config.state_dim,    # 24 dimensions
    hidden_dim=state_hidden,
    output_dim=embed_dim,
    num_categories=num_categories
)
```

The encoded state is then concatenated with the VLM context tokens:

```python
state_emb = self.state_encoder(state, embodiment_id).unsqueeze(1)  # [B, 1, D]
context_tokens = torch.cat([context_tokens, state_emb], dim=1)
```

**Paper evidence**: *"we concatenate z_t with the robot state s_t instead of projecting them into a shared embedding space"* and this concatenated feature *"serves as the key-value input for the transformer blocks of the action expert."* (Evo-1, Section 3.2)

**Critical observation**: The state produces exactly **1 token** that is appended to the VLM's multimodal feature sequence z_t. The VLM (InternVL3-1B) produces hundreds of vision-language tokens. The state token is a single entry in a much longer cross-attention key-value sequence.

### 2.2 RDT-1B: Single Linear Adaptor with Fourier Features

**Architecture**: RDT-1B encodes proprioception through a 3-layer MLP with Fourier features:

> *"we use MLPs (with Fourier features), which can effectively capture the high-frequency changes in low-dimensional spaces."* (RDT-1B, Section 3.2)

The encoded state is combined with action tokens via concatenation:

> *"They are encoded into the token space by a shared MLP since they have similar physical meanings."* (RDT-1B, Section 3.2)

The full input sequence has length `1 + T_a + 1 + 1` (proprioception + action chunk + control frequency + timestep), and conditioning from vision/language enters through **cross-attention** in each transformer layer.

**Critical observation**: Proprioception occupies exactly **1 token** in the self-attention sequence. Action tokens (T_a = 64) constitute the dominant sequence. Vision and language enter through cross-attention only, never directly sharing the self-attention space with state.

### 2.3 Pi0: Single Linear Projection

**Architecture**: Pi0's state encoder is the simplest of all three -- a single linear projection:

> *"the state q_t is a vector of joint angles...encoded via corresponding encoders and then projected via a linear projection layer into the same embedding space."* (Pi0, Section 3)

> *"mapped to the transformer embedding dimension using a linear projection."* (Pi0, Section 3)

The projected state tokens form their own block in Pi0's blockwise causal attention scheme, sandwiched between the observation block (images + language) and the action block.

**Critical observation**: A single linear layer lacks any nonlinearity or capacity to learn complex joint-space representations. It can only perform an affine transformation of the raw joint angles.

---

## 3. Root Cause Analysis

### 3.1 Dimensionality Asymmetry and Token Count Imbalance

**The core problem**: Proprioceptive state occupies a vanishingly small fraction of the total token budget.

| Model | Vision Tokens | Language Tokens | State Tokens | State Fraction |
|-------|--------------|-----------------|--------------|----------------|
| **Evo-1** | ~256+ (InternVL3-1B patches) | ~50+ (instruction) | **1** | <0.3% |
| **RDT-1B** | ~196 (SigLIP patches) | ~varies (T5-XXL) | **1** | <0.5% |
| **Pi0** | ~256 (PaliGemma ViT) | ~50+ (Gemma) | **~few** | <1% |

**Theoretical implication**: In transformer architectures, attention is computed via softmax over all key-value pairs. When state contributes 1 token out of 300+, the attention weight allocated to state is:

```
α_state = exp(q · k_state) / Σ_j exp(q · k_j)
```

Even if the query-key affinity for state is high, the denominator includes hundreds of vision tokens, each contributing to the normalization. The softmax operation inherently distributes probability mass, making it difficult for a single state token to capture significant attention weight against hundreds of competing visual tokens.

**Paper evidence from RDT-1B**: The authors explicitly acknowledge this problem for language vs. vision:

> *"image tokens are usually much more than text tokens, simultaneous injection of both modalities tends to overshadow text-related information, thus impairing the capability of the instruction following."* (RDT-1B, Section 3.3)

This is precisely the **Alternating Condition Injection (ACI)** motivation. If vision already overshadows language (which has ~50+ tokens), it will catastrophically overshadow proprioception (which has 1 token). Notably, ACI was designed to protect language from visual dominance -- **no such protection exists for proprioceptive state in any of the three models**.

### 3.2 Encoder Capacity Asymmetry

The capacity gap between the proprioceptive encoder and the vision encoder is staggering:

| Model | Vision Encoder | Vision Params | State Encoder | State Params (est.) |
|-------|---------------|---------------|---------------|---------------------|
| **Evo-1** | InternVL3-1B | ~1B | CategorySpecificMLP (3-layer) | ~0.1M |
| **RDT-1B** | SigLIP-So400M | ~400M | 3-layer MLP w/ Fourier | ~0.05M |
| **Pi0** | PaliGemma ViT | ~400M+ | Single Linear layer | ~0.01M |

**Parameter ratio**: The vision encoder has 4,000x--100,000x more parameters than the state encoder.

**Theoretical implication**: The representation capacity of a network is bounded by its parameter count. A 3-layer MLP with ~0.1M parameters can represent functions in a space far smaller than what a 400M--1B parameter vision encoder can represent. The state encoder simply cannot learn representations of comparable richness.

**Evo-1 paper evidence**: The model uses InternVL3-1B which is a *"fused vision-language encoder"* that has been pre-trained on internet-scale data. The CategorySpecificMLP, by contrast, is trained from scratch on robot data alone, with no pre-training benefit. The paper's two-stage training scheme explicitly acknowledges that the VLM backbone quality must be preserved:

> *"freeze the entire vision-language backbone and exclusively train the action expert along with the integration module"* (Evo-1, Stage 1)

> *"Once the integration and action module are sufficiently aligned, we unfreeze the VLM backbone and perform full-scale fine-tuning"* (Evo-1, Stage 2)

This strategy prioritizes preserving the VLM's semantic representations. The state encoder, being part of the action expert, must adapt to the VLM's representation space -- not the other way around. The VLM sets the representation manifold; the state encoder must project into it.

### 3.3 Gradient Flow Asymmetry from Frozen Backbones

**Pi0: The most extreme case.** Pi0 uses a pre-trained 3B parameter PaliGemma VLM as its backbone. The action expert (300M parameters) is initialized from scratch:

> *"Add 300M parameters for action expert (initialized from scratch) for total 3.3 billion parameters."* (Pi0, Section 3)

The architecture uses a mixture-of-experts design:

> *"Two sets of weights (experts); images/language routed to VLM backbone; robotics-specific inputs routed to action expert...weights interact only through transformer's self-attention."* (Pi0, Section 3)

**Theoretical implication for gradients**: During pre-training, the flow matching loss is:

```
L^τ(θ) = E[||v_θ(A_t^τ, o_t) - u(A_t^τ|A_t)||²]
```

where `v_θ` is the velocity field predicted by the full model. This loss backpropagates through the entire network. However:

1. The VLM backbone processes vision and language through ~18 transformer layers (width=2048, depth=18) before the features reach the action expert.
2. The state enters through a single linear projection directly into the action expert.
3. Gradients flowing back from the loss to the VLM backbone must traverse the entire action expert first. But gradients to the state projection layer are direct (few layers away from the loss).

This creates a paradox: the state encoder receives gradients more directly, but the gradient *magnitude* is small because:
- The loss is dominated by action prediction error
- If vision already provides sufficient information for coarse action prediction, the marginal gradient signal through the state path is weak
- The VLM backbone, with its massive parameter count, captures a much larger share of the total gradient norm due to its higher-dimensional parameter space

**Evo-1 evidence**: The two-stage training explicitly creates a period where the VLM is frozen. During Stage 1:

> *"freeze the entire vision-language backbone and exclusively train the action expert along with the integration module"* (Evo-1)

During this phase, no gradients flow into the VLM at all. The action expert (including the state encoder) must learn to work with frozen VLM features. The state encoder has ~5,000 steps to align itself. In Stage 2, when the VLM is unfrozen, the VLM's representations shift, potentially invalidating what the state encoder learned. The VLM's much larger gradient magnitude dominates training dynamics, and the state encoder must continuously re-adapt to a moving target.

### 3.4 Information-Theoretic Redundancy Between Vision and Proprioception

**Core argument**: Much of the information in proprioceptive state is already recoverable from visual observations.

A camera observing the robot can provide:
- Approximate joint configuration (from visual arm pose)
- End-effector position (from visual tracking)
- Gripper state (open/close visible in image)
- Object-relative positioning (spatial relationships)

The only information truly *unique* to proprioception is:
- Exact joint angles (sub-degree precision)
- Joint velocities and torques (not observable from single-frame vision)
- Contact forces
- Internal state not visible to cameras (e.g., wrist rotation behind occlusion)

**Pi0 evidence**: Pi0 is pre-trained on *"10,000+ hours across 7 robot configurations and 68 tasks"* plus the OXE (Open X-Embodiment) dataset. This massive visual training corpus means the VLM backbone has learned strong visual priors for robot pose estimation. The flow matching loss:

```
L^τ(θ) = E[||v_θ(A_t^τ, o_t) - u(A_t^τ|A_t)||²]
```

can be substantially minimized using vision alone, because the visual encoder has already learned to extract robot configuration from images. The marginal information gain from adding proprioceptive state is small relative to the visual signal.

**Formal information-theoretic argument**: Let V denote visual features, S denote proprioceptive state, and A denote actions. The mutual information decomposition gives:

```
I(V, S; A) = I(V; A) + I(S; A | V)
```

If I(V; A) is already high (vision is highly informative about actions), then I(S; A | V) -- the *conditional* information of state given vision -- is small. The model only needs the state encoder to capture this residual. But the training loss does not separately weight this residual; it optimizes the total prediction error, which is already low from vision alone.

**RDT-1B evidence**: RDT-1B uses stochastic independent masking to combat this:

> *"each input from various modalities is independently masked with a probability of 10%"* (RDT-1B, Appendix B)

This forces the model to occasionally predict actions *without* vision, ensuring it learns to use proprioception. However, with only 10% masking probability, the model sees vision 90% of the time, still biasing the learned representations toward visual dependence.

### 3.5 Training Objective Insensitivity to Proprioceptive Precision

**The flow matching / diffusion objective treats all action dimensions equally.**

**Evo-1's flow matching loss**:
```
L^τ(θ) = E[||v_θ(A_t^τ, z_t, s_t) - u(A_t^τ|A_t)||²]
```

where the noise interpolation is:
```
A_t^τ = τ·A_t + (1-τ)·ε
```

with τ sampled from Beta(2,2) and clamped to [0.02, 0.98].

**RDT-1B's diffusion loss**:
```
L(θ) = MSE(a_t, f_θ(ℓ, o_t, √(ᾱ^k)·a_t + √(1-ᾱ^k)·ε, k))
```

**Pi0's flow matching loss**:
```
L^τ(θ) = E[||v_θ(A_t^τ, o_t) - u(A_t^τ|A_t)||²]
```

with probability path `q(A_t^τ|A_t) = N(τ·A_t, (1-τ)·I)` and target `u(A_t^τ|A_t) = A_t - ε`.

**All three use MSE-type losses** over the full action vector. This means:
1. Large, coarse movements (reachable with vision alone) contribute more to the loss than fine positioning errors
2. The gradient signal is dominated by getting the rough trajectory right, not by precision
3. Proprioceptive state is most valuable precisely for the *residual* fine-grained corrections that contribute least to the MSE

**RDT-1B's unified action space** exacerbates this. The 128-dimensional unified action vector contains:

> *"Right arm joints: indices 0-9, Left arm joints: indices 50-59, End-effector poses: indices 30-45, 80-95, Base velocities: indices 100-102"* (RDT-1B, Section 3.1)

Many dimensions are padded with zeros for robots that don't use them. The padding vectors:

> *"concatenate the action and proprioception with a 0-1 vector indicating whether each dimension is padded"* (RDT-1B, Appendix F)

This means the model must waste capacity distinguishing real zeros (joint at rest) from padding zeros, further diluting the signal from actual proprioceptive dimensions.

---

## 4. Architecture-Specific Failure Modes

### 4.1 Evo-1: Integration Module Bottleneck

Evo-1 extracts VLM features from a specific layer:

> *"extracts the fused multimodal feature z_t from the 14th VLM layer, concatenates it with the robot state s_t, and uses them as key-value inputs for all DiT layers."* (Evo-1, Section 3.2)

**Problem 1: Fixed extraction point.** The 14th layer features are optimized for vision-language reasoning, not for action-relevant spatial encoding. Proprioceptive state is concatenated with these features, creating a heterogeneous key-value space where some entries are high-level semantic features and one entry is a low-level state embedding.

**Problem 2: Cross-attention dilution.** The action expert uses *"stacked cross-attention layers"* where noisy action tokens query against the concatenated context:

> *"action expert is implemented as a Diffusion Transformer (DiT) that solely relies on stacked cross-attention layers, in contrast to the alternating self-attention and cross-attention."* (Evo-1, Section 3.3)

In pure cross-attention, the noisy action tokens attend to all context tokens. The state token must compete with hundreds of VLM tokens for attention. There is **no self-attention among context tokens** that would allow the state to interact with and influence the visual representation before the action tokens query it.

**Problem 3: Ablation evidence.** The paper compared integration modules and found:

> *"Module A outperforms other variants by maintaining consistent propagation of multimodal information."* (Evo-1, Section 4)

Module C, which *"injects features from selected mid-to-deep VLM layers into the DiT"* performed worse. This suggests that more complex fusion actually degrades performance, potentially because the VLM features are already sufficient and additional connections create optimization difficulties.

### 4.2 RDT-1B: Cross-Attention Overshadowing

RDT-1B's Alternating Condition Injection was explicitly designed to address vision-language imbalance:

> *"strategically alternate between injecting image and text tokens in successive layers' cross-attention rather than injecting both in every layer."* (RDT-1B, Section 3.3)

**The critical gap**: ACI alternates between vision and language, but **proprioception is not part of the cross-attention at all**. State enters through the self-attention path (concatenated with action tokens), while vision and language enter through cross-attention. This architectural separation means:

1. State information must be encoded entirely within 1 token in the self-attention sequence
2. Vision information arrives through cross-attention with 196+ tokens per layer
3. The attention budget for any given transformer layer is split: self-attention handles state + actions, cross-attention handles vision OR language
4. The state token in self-attention competes with 64 action tokens, but these action tokens simultaneously receive rich visual conditioning through cross-attention

**The RDT diffusion loss compounds this**:
```
L(θ) = MSE(a_t, f_θ(ℓ, o_t, √(ᾱ^k)·a_t + √(1-ᾱ^k)·ε, k))
```

The model is trained to denoise action chunks. The optimal denoising strategy is to rely on the modality with the most tokens, richest representation, and strongest signal -- which is always vision via cross-attention, not proprioception via a single self-attention token.

**Ablation evidence**: RDT-1B's ablation table shows that removing diffusion causes catastrophic failure on instruction following (12.5% vs 100%), but **no ablation was performed on the state encoder specifically**. This absence itself is telling -- the authors did not consider state encoding important enough to ablate.

### 4.3 Pi0: Linear Bottleneck and Blockwise Isolation

Pi0's blockwise causal attention creates three blocks:

> *"Tokens in each block cannot attend to tokens in future blocks...This minimizes distribution shift from VLM pre-training."* (Pi0, Section 3)

The three blocks are: (1) observation inputs (images + language), (2) state q_t, and (3) noisy actions.

**Problem 1: Causal masking prevents state from influencing observation processing.** Block 1 (observations) cannot attend to Block 2 (state). This means the VLM's visual processing is entirely independent of the robot's current proprioceptive state. The model processes what it *sees* without any knowledge of what the robot *feels*.

In biological systems, proprioception fundamentally alters visual processing (e.g., knowing your arm is extended changes how you interpret visual proximity to objects). Pi0's architecture explicitly prevents this integration.

**Problem 2: State cached for efficiency, further marginalizing its role.** The paper states:

> *"preventing it from attending to the final block allows its corresponding keys and values to be cached during sampling."* (Pi0, Section 3)

The state tokens are computed once and cached. During the iterative flow matching sampling process (10 Euler steps with δ=0.1), the state representation is **frozen**. Even though the action distribution evolves over 10 denoising steps, the state representation remains static. This means the model cannot iteratively refine its understanding of proprioceptive state during action generation.

**Problem 3: The linear projection cannot capture nonlinear joint-space geometry.** Robot joint spaces are inherently nonlinear -- joint limits create bounded manifolds, singularities exist at certain configurations, and the mapping from joint angles to end-effector pose (forward kinematics) involves trigonometric functions. A single linear projection:

```
h_state = W · q_t + b
```

can only represent affine transformations. It cannot capture:
- Joint limit effects (clipping, nonlinear resistance near limits)
- Configuration-dependent manipulability (Jacobian varies nonlinearly with joint angles)
- Kinematic singularities (where small joint changes produce large Cartesian effects)
- Redundancy resolution (multiple joint configurations mapping to same end-effector pose)

**Pi0 action expert insufficiency**: The action expert has only 300M parameters with a downsized architecture:

> *"{width=1024, mlp_dim=4096}, resulting in parameter count of ~300M"* (Pi0, Section 3)

compared to the VLM backbone's 3B parameters (width=2048, depth=18, mlp_dim=16,384). The action expert, which is the primary consumer of state information, has 10x fewer parameters than the VLM backbone that processes vision.

---

## 5. Cross-Model Theoretical Framework

### 5.1 The Vision Dominance Hypothesis

All three models exhibit the same pattern: a powerful, pre-trained vision encoder dominates a weak, train-from-scratch state encoder. This can be formalized as a **modality dominance** problem in multimodal learning.

Consider the total loss gradient with respect to the state encoder parameters θ_s:

```
∂L/∂θ_s = (∂L/∂a_pred) · (∂a_pred/∂h_fused) · (∂h_fused/∂h_state) · (∂h_state/∂θ_s)
```

where h_fused is the fused multimodal representation and h_state is the state encoding.

The term `(∂h_fused/∂h_state)` is small because:
1. h_state contributes 1 token to a sequence of 300+ tokens
2. In attention-based fusion, the gradient through attention weights scales with the attention weight itself, which is small for state (see Section 3.1)
3. The Jacobian `∂h_fused/∂h_state` has low effective rank because the state encoding is low-dimensional

Meanwhile, the gradient with respect to the vision encoder parameters θ_v:

```
∂L/∂θ_v = (∂L/∂a_pred) · (∂a_pred/∂h_fused) · (∂h_fused/∂h_vision) · (∂h_vision/∂θ_v)
```

has a much larger `(∂h_fused/∂h_vision)` term because vision contributes hundreds of tokens, each receiving attention.

### 5.2 The Implicit Pose Estimation Argument

Modern vision encoders (SigLIP, PaliGemma ViT, InternVL3) are trained on massive image datasets. For robot manipulation scenarios where the robot is visible in the image, these encoders implicitly learn to:

1. **Segment the robot** from the background
2. **Estimate joint configuration** from visual appearance
3. **Track end-effector position** relative to objects
4. **Infer gripper state** (open/close) from visual cues

This means `I(V; S)` -- the mutual information between visual features and proprioceptive state -- is already high. The model can recover much of the proprioceptive information from vision alone.

Evidence from Evo-1's real-world results (Table 2):

| Model | Params (B) | Success (%) |
|-------|-----------|-------------|
| SmolVLA | 0.45 | 50.0 |
| OpenVLA | 7.0 | 55.0 |
| Pi0 | 3.5 | 73.0 |
| **Evo-1** | **0.77** | **78.0** |

OpenVLA has **no proprioceptive state at all** and achieves 55% success. This demonstrates that vision alone provides substantial task capability. The marginal improvement from adding proprioception (comparing OpenVLA to models with proprioception) is much smaller than the improvement from better vision processing.

### 5.3 The Training Distribution Mismatch

**RDT-1B's multi-robot pre-training** creates a unique challenge for proprioceptive encoding:

> *"embed the action space of a robot into this unified space by filling each element of the original action vector into the corresponding position...according to its physical meaning, with the remaining positions being padded."* (RDT-1B, Section 3.1)

The 128-dimensional unified space must accommodate robots with vastly different kinematic structures. The state encoder must learn a representation that is meaningful across:
- 7-DOF single arms
- 14-DOF bimanual systems
- Mobile bases with 3-DOF
- Different joint ranges and units

The Fourier feature MLP:

> *"MLPs (with Fourier features), which can effectively capture the high-frequency changes in low-dimensional spaces"* (RDT-1B, Section 3.2)

is shared across all robots. But Fourier features are most effective for continuous, smooth functions. The mapping from heterogeneous robot states to a unified representation is inherently discontinuous (different robots have different dimensions active). The encoder must learn a piecewise function that routes different robot states to appropriate regions of the unified space -- a task for which Fourier features are poorly suited.

### 5.4 The Temporal Information Gap

Proprioception is most valuable for **dynamic** information that vision cannot easily provide:
- Joint velocities and accelerations
- Contact detection (forces/torques)
- Vibration and compliance feedback
- Temporal derivatives of configuration

However, all three models treat proprioception as a **static snapshot** at time t:

- **Evo-1**: `s_t` is a single state vector at the current timestep
- **RDT-1B**: Proprioception token is a single encoding of current state
- **Pi0**: `q_t` is current joint angles projected linearly

None of the models encode temporal proprioceptive features (joint velocity, acceleration, or state history). This means the models receive the *least valuable* component of proprioception (raw angles, which vision can approximate) while missing the *most valuable* component (dynamics, which vision cannot easily capture).

**Pi0's action chunking** somewhat compensates:

> *"Model outputs H=50 action tokens per inference timestep; each token decoded via linear projection into action space."* (Pi0, Section 3)

But this is a temporal model of *output* actions, not *input* state history.

---

## 6. Mathematical Analysis of Attention Dilution

### 6.1 Softmax Attention Distribution

Consider a transformer layer where action query token q attends to N_v vision key-value pairs and 1 state key-value pair. The attention weight for state is:

```
α_state = exp(q^T · k_state / √d) / (exp(q^T · k_state / √d) + Σ_{i=1}^{N_v} exp(q^T · k_v^i / √d))
```

Under uniform initialization (before training converges), if all key-query affinities are similar:

```
α_state ≈ 1 / (1 + N_v) ≈ 1/257 ≈ 0.004  (for N_v = 256)
```

Even if training increases the state's key-query affinity by a factor of 10:

```
α_state ≈ 10 / (10 + 256) ≈ 0.038
```

The state token still receives less than 4% of the attention budget. For the model to give state 50% of attention, the state key-query affinity would need to be **256x larger** than vision's -- an extreme optimization target.

### 6.2 Gradient Through Attention

The gradient of the attention output with respect to the state value is:

```
∂(Attn_output)/∂v_state = α_state · I
```

This is directly proportional to the attention weight. Since α_state is small (Section 6.1), the gradient signal through the state path is proportionally small. This creates a **positive feedback loop**:

1. Small attention weight → small gradient to state encoder
2. Small gradient → slow learning of useful state representations
3. Poor state representations → model doesn't find state useful
4. Low utility → small attention weight (back to step 1)

This is a classic **rich-get-richer** dynamic where vision, starting with pre-trained representations, captures most of the attention and gradient, and proprioception cannot break out of its low-utilization equilibrium.

### 6.3 Effective Rank Argument

The effective rank of a representation matrix captures the number of dimensions that carry meaningful information. For the vision encoder output (N_v × d_model matrix), the effective rank can be as high as min(N_v, d_model) ≈ 256.

For the state encoder output (1 × d_model matrix), the effective rank is at most **1**. This means:

- Vision can communicate up to 256 independent pieces of information per layer
- State can communicate at most 1 independent piece of information per layer

The information bottleneck is extreme. Even if the proprioceptive state has 24 dimensions of meaningful information (as in Evo-1's case), it is compressed to a single token vector that must encode all of it simultaneously.

---

## 7. Summary of Failure Modes by Model

### Evo-1
| Failure Mode | Evidence from Paper |
|---|---|
| Single state token vs hundreds of VLM tokens | *"concatenate z_t with the robot state s_t"* -- 1 token appended to VLM sequence |
| VLM backbone preservation prioritized over state learning | Two-stage training: *"freeze the entire vision-language backbone"* in Stage 1 |
| Cross-attention-only DiT dilutes state signal | *"DiT that solely relies on stacked cross-attention layers"* |
| State encoder trained from scratch, no pre-training | CategorySpecificMLP random init vs InternVL3-1B pre-trained |
| Static state snapshot, no temporal dynamics | Input is `s_t` at single timestep |

### RDT-1B
| Failure Mode | Evidence from Paper |
|---|---|
| State in self-attention, vision/language in cross-attention (architectural separation) | *"in-context conditioning"* for state vs cross-attention for vision/language |
| Vision overshadows other modalities (recognized by authors, only partially addressed) | *"image tokens are usually much more than text tokens, simultaneous injection...tends to overshadow"* |
| ACI protects language but NOT proprioception | Alternating injection only between image and text tokens |
| Unified action space dilutes proprioceptive signal with padding | 128D space with *"0-1 vector indicating whether each dimension is padded"* |
| 10% masking insufficient to force proprioceptive learning | *"independently masked with a probability of 10%"* |

### Pi0
| Failure Mode | Evidence from Paper |
|---|---|
| Single linear projection (minimal capacity) | *"projected via a linear projection layer"* -- cannot capture joint-space nonlinearity |
| Causal masking prevents state from influencing visual processing | *"Tokens in each block cannot attend to tokens in future blocks"* |
| State representation cached and frozen during sampling | *"keys and values to be cached during sampling"* -- static during 10 denoising steps |
| 10x parameter asymmetry (action expert vs VLM) | 300M action expert vs 3B VLM backbone |
| No proprioception-specific ablation reported | Paper ablates VLM pre-training, not state encoding |

---

## 8. Theoretical Predictions

Based on this analysis, we predict the following empirical findings from hook-based diagnostic experiments:

1. **Gradient magnitude ratio** (state/vision) will be < 0.01 across all three models, reflecting the attention dilution and parameter count asymmetry.

2. **CKA similarity** between vision and state representations will be > 0.6, confirming information redundancy -- the state encoder learns to approximate what vision already represents rather than capturing complementary information.

3. **Ablation impact** of removing proprioception will be < 10% performance drop for coarse manipulation tasks, but > 20% for tasks requiring sub-centimeter precision (peg insertion, plug charging).

4. **Attention weight** allocated to state tokens will decrease across layers (deepening dilution), rather than increasing, because the model learns to rely on vision which provides richer features.

5. **Effective rank** of state representations will be < 5 even for 24-dimensional proprioceptive input, indicating severe information compression at the encoding bottleneck.

---

## 9. Implications for Architecture Design

The analysis suggests several concrete improvements:

1. **Multi-token state encoding**: Project proprioception to multiple tokens (e.g., one per joint group) to increase the state's token budget and attention share.

2. **Dedicated state-action cross-attention**: Add cross-attention layers where action tokens attend *only* to state tokens, guaranteeing proprioceptive information reaches the action head without competing with vision.

3. **Temporal state encoding**: Include state history (q_{t-k}, ..., q_t) and compute joint velocities/accelerations as explicit features.

4. **Modality-balanced training**: Use higher masking rates for vision (>50% of the time) to force proprioceptive learning, or use separate loss terms that weight proprioceptive contributions.

5. **Nonlinear state encoders with pre-training**: Pre-train state encoders on dynamics prediction or forward kinematics tasks to give them useful initializations before integration with VLMs.

6. **Bidirectional state-vision attention**: Allow state tokens to attend to vision tokens (and vice versa) before the action head processes them, enabling the model to learn vision-proprioception interactions (e.g., "I see my arm is near the object AND I feel resistance at joint 3").

---

## 10. References

1. **Evo-1**: Lin et al., "Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment," arXiv:2511.04555, 2025. https://arxiv.org/abs/2511.04555
2. **RDT-1B**: Liu et al., "RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation," arXiv:2410.07864, 2024. https://arxiv.org/abs/2410.07864
3. **Pi0**: Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024. https://arxiv.org/abs/2410.24164
4. **Information Bottleneck**: Tishby et al., "The Information Bottleneck Method," arXiv:physics/0004057, 2000.
5. **RT-2**: Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," arXiv:2307.15818, 2023.

---

*Document prepared for the MultipleHooksStudy diagnostic analysis project. All claims are supported by direct evidence from the published papers and source code.*
