# State Encoder Experiments — Reference README

## Overview

This document describes the full experimental plan for investigating improved state encoding in the Evo-1 Vision-Language-Action model. The core hypothesis is that the current single-token state encoder is architecturally marginalised by the softmax competition with VLM tokens, and that temporal state history combined with richer feature extraction can improve proprioceptive grounding in the action expert.

All experiments share the same base architecture modifications and differ only in which features are encoded and how type embeddings are initialised.

---

## Background and Motivation

### Current architecture

The current state encoder in Evo-1 is:

```
s_t ∈ R^7  →  MLP(7 → 1024 → 896)  →  e_st ∈ R^896
```

This single token is concatenated with the VLM output and used as keys and values in all 8 cross-attention blocks of the DiT:

```
context_tokens = [z_t || e_st] ∈ R^(B × (N+1) × 896)
```

### Ablation finding

Zeroing the state encoder output produces no measurable change in task performance. The model has learned to ignore the state token entirely.

### Root causes

1. **Architectural marginalisation** — 1 state token competes against N VLM tokens in a single softmax. Maximum attention budget is 1/(N+1).
2. **No direct gradient signal** — the flow matching loss can be minimised using VLM tokens alone. Gradient reaching the state encoder is indirect and weak.
3. **Single snapshot** — s_t = [q_1, ..., q_7] contains no temporal information. Velocity, acceleration, and force are invisible.

---

## Proposed Architecture

### State history matrix

Instead of a single state vector, encode a k-step history:

```
S = [s_t, s_{t-1}, ..., s_{t-k}] ∈ R^((k+1) × 7)
```

Where each row s_{t-i} = [s_{1,t-i}, ..., s_{7,t-i}] ∈ R^7 is one complete joint state snapshot. Rows index timesteps, columns index joint dimensions.

### Feature extraction

All features are extracted column-wise (along the time axis) from S:

**Position tokens** — k+1 tokens
```
s_{t-i} ∈ R^7,  i = 0, ..., k
```

**Velocity tokens** — k tokens (finite difference, column-wise)
```
ṡ_{t-i} = s_{t-i+1} - s_{t-i} ∈ R^7,  i = 0, ..., k-1
```

**Acceleration tokens** — k-1 tokens (finite difference of velocity)
```
s̈_{t-i} = ṡ_{t-i+1} - ṡ_{t-i} ∈ R^7,  i = 0, ..., k-2
```

**Eligibility trace** — 1 token (MDP-inspired decaying sum)
```
z_t^tr = Σ_{i=0}^{k} ρ^i · s_{t-i} ∈ R^7,  ρ ∈ (0,1) fixed scalar
```

**Deviation from mean** — 1 token (detects motion transitions)
```
δ_t = s_t - (1/(k+1)) · Σ_{i=0}^{k} s_{t-i} ∈ R^7
```

All features are in R^7. Total state tokens: (k+1) + k + (k-1) + 1 + 1 = 3k+2.

### Shared encoding pipeline

All features pass through one shared MLP regardless of type:

```
e_i^feat = ρ_feat^i · LN(MLP(f_{t-i})) + η_feat
```

Where:
- `MLP: R^7 → R^1024 → R^896` — shared weights, identical to current state encoder architecture
- `LN` — LayerNorm(896) applied to MLP output before type embedding addition
- `η_feat ∈ R^896` — type embedding vector, one per feature type
- `ρ_feat = σ(w_feat) ∈ (0,1)` — learned decay logit per feature type, applied as ρ^i to weight timestep i
- `i` — timestep index, i=0 is most recent

### Full enriched context

```
z̃_t = [z_t || E_pos || E_vel || E_acc || e_{z^tr} || e_δ] ∈ R^(B × (N+3k+2) × 896)
```

Where z_t ∈ R^(N×896) is the VLM output unchanged.

---

## Training Strategy

Training proceeds in three sequential phases.

### Phase 0 — State encoder pretraining

**What trains:** Shared MLP φ, LayerNorm γ,β, decoder D_ψ, and (depending on embedding strain) type embeddings η

**What is frozen:** Everything else — DiT, VLM

**Data required:** State vectors only. No images, no actions, no VLM forward pass needed. Extract all feature vectors from state history windows in the dataset.

**Duration:** Short. The MLP is small (≈930k parameters) and converges fast on a simple reconstruction task.

**Loss:**

```
L_pretrain = L_recon(φ) + λ_orth · L_orth(η)
```

Reconstruction loss:
```
L_recon(φ) = (1/|F|) · Σ_{f ∈ F} || D_ψ(LN(MLP_φ(f))) - f ||^2
```

Where:
- `φ` — shared MLP weights
- `F` — set of all feature vectors from the history window: positions, velocities, accelerations, eligibility trace, deviation
- `LN` — LayerNorm applied to MLP output
- `D_ψ: R^896 → R^7` — small decoder MLP with parameters ψ, discarded after Phase 0
- `f ∈ R^7` — individual feature vector being reconstructed

Orthogonality loss (used in embedding Strain B only):
```
L_orth(η) = Σ_{i≠j} (η_i · η_j / (||η_i|| · ||η_j||))^2
```

Where:
- `η_i, η_j ∈ R^896` — type embedding vectors for feature types i ≠ j
- Penalises cosine similarity between any two type embeddings
- λ_orth ≈ 0.01 — small weight preventing orthogonality from dominating reconstruction

**After Phase 0:** Freeze MLP φ, LayerNorm γ,β, and type embeddings η. Discard decoder D_ψ. Only decay logits w remain unfrozen going into Phase 1.

### Phase 1 — Action expert alignment

**What trains:** DiT blocks, decay logits w

**What is frozen:** MLP, LayerNorm, type embeddings η, VLM

**Data required:** Full trajectories with images, states, and actions

**Duration:** Longer than the original paper's Stage 1. The DiT must discover a frozen token distribution it has never seen during initialisation. Recommend running 1.5–2× the original Stage 1 step count.

**Loss:**
```
L^τ(θ) = E_{p(A_t | z̃_t), q(A_t^τ | A_t)} [ || v_θ(A_t^τ, z̃_t) - u(A_t^τ | A_t) ||^2 ]
```

With masked scale correction from code:
```
L = ||(v_θ - u) ⊙ M||^2 · |M| / ΣM
```

Where:
- `θ` — all trainable parameters at this phase (DiT blocks + decay logits)
- `A_t^τ = τA_t + (1-τ)ε` — interpolated noisy action
- `τ ~ Beta(2,2), τ ∈ [0.02, 0.98]` — flow timestep
- `ε ~ U(-1,1)` — sampled noise
- `u(A_t^τ | A_t) = A_t - ε` — target velocity field
- `v_θ` — predicted velocity field from DiT
- `z̃_t` — enriched context tokens
- `M ∈ {0,1}^(H×7)` — action mask zeroing inactive DOFs
- `|M|/ΣM` — scale correction so masked dimensions do not deflate the loss

### Phase 2 — Full fine-tuning

**What trains:** Everything — DiT, decay logits, VLM backbone

**What is frozen:** Nothing

**Data required:** Full trajectories

**Duration:** Standard, same as original paper Stage 2

**Loss:** Same flow matching loss as Phase 1

---

## Experiment Plan

### Experiment 1 — Pure state history (baseline ablation)

**Description:** Encode only the raw position history. No velocity, acceleration, or MDP features. No type embeddings needed since all tokens are the same type.

**Features encoded:**
```
E_pos = [MLP(s_t), MLP(s_{t-1}), ..., MLP(s_{t-k})]  ∈ R^((k+1) × 896)
```

**Context tokens:**
```
z̃_t = [z_t || E_pos] ∈ R^(B × (N+k+1) × 896)
```

**Encoding pipeline:**
```
e_i^pos = ρ_pos^i · LN(MLP(s_{t-i}))
```
No type embedding needed (single feature type).

**Phase 0 loss:** Reconstruction loss only, no orthogonality term.

**Purpose:** Isolates the effect of increased token count alone. If this helps, the attention budget increase 1/(N+1) → (k+1)/(N+k+1) is sufficient explanation. If it does not help, richer features are necessary.

**What to measure:**
- Task success rate vs baseline (single token, current model)
- Mean attention weight to E_pos tokens vs VLM tokens
- Whether attention weight scales with recency (i.e. s_t gets more attention than s_{t-k})

---

### Experiment 2 — Full feature set

**Description:** Encode position history, velocity, acceleration, eligibility trace, and deviation from mean. This is the full proposed architecture.

**Features encoded:**
```
E_pos  ∈ R^((k+1) × 896)   — position history
E_vel  ∈ R^(k × 896)        — velocity
E_acc  ∈ R^((k-1) × 896)    — acceleration
e_ztr  ∈ R^(1 × 896)        — eligibility trace
e_δ    ∈ R^(1 × 896)        — deviation from mean
```

**Context tokens:**
```
z̃_t = [z_t || E_pos || E_vel || E_acc || e_ztr || e_δ] ∈ R^(B × (N+3k+2) × 896)
```

**Encoding pipeline:**
```
e_i^feat = ρ_feat^i · LN(MLP(f_{t-i})) + η_feat
```

This experiment is run in three sub-strains differing only in type embedding initialisation and training. See Embedding Strains below.

**Purpose:** Tests whether richer temporal features improve performance beyond the attention budget increase alone. The three sub-strains isolate the effect of type embedding structure.

**What to measure:**
- Task success rate vs Experiment 1 and baseline
- Per-feature-type attention weight: does α_vel > α_pos as hypothesised?
- Attention weight to e_ztr and e_δ — are MDP features used?
- Embedding geometry: PCA/UMAP of type embeddings before and after Phase 1 and 2

---

## Embedding Strains (Experiment 2 sub-conditions)

### Strain A — Random init, no orthogonality loss

**Initialisation:**
```python
self.type_embedding = nn.Embedding(6, 896)
# Default PyTorch init: N(0, 1)
```

**Phase 0 loss:**
```
L_pretrain = L_recon(φ)
```
Type embeddings receive gradient only indirectly through the reconstruction loss.

**What this tests:** Whether type embeddings can learn useful structure from the reconstruction signal alone without any explicit orthogonality supervision. Serves as the lower bound on type embedding quality.

**Expected behaviour:** Type embeddings may collapse or become similar to each other since nothing explicitly prevents this.

---

### Strain B — Random init, orthogonality loss

**Initialisation:**
```python
self.type_embedding = nn.Embedding(6, 896)
# Default PyTorch init: N(0, 1)
```

**Phase 0 loss:**
```
L_pretrain = L_recon(φ) + λ_orth · L_orth(η),  λ_orth = 0.01
```

**What this tests:** Whether explicitly training type embeddings to be orthogonal during Phase 0 improves downstream attention specialisation compared to Strain A.

**Expected behaviour:** Type embeddings will be approximately orthogonal after Phase 0. Not guaranteed to be exactly orthogonal — depends on convergence.

---

### Strain C — Orthogonal init, no orthogonality loss

**Initialisation:**
```python
def init_orthogonal_embeddings(num_types, embed_dim):
    random_matrix = torch.randn(embed_dim, embed_dim)
    Q, _ = torch.linalg.qr(random_matrix)
    return Q[:num_types, :]  # [6, 896] — exactly orthogonal

self.type_embedding = nn.Embedding(6, 896)
self.type_embedding.weight.data = init_orthogonal_embeddings(6, 896)
self.type_embedding.weight.requires_grad = False  # frozen throughout all phases
```

**Phase 0 loss:**
```
L_pretrain = L_recon(φ)
```
Type embeddings are frozen — they do not train at all.

**What this tests:** Whether guaranteed exact orthogonality by construction outperforms approximate orthogonality learned via loss. The cleanest experimental condition — type embeddings are a fixed property of the system.

**Expected behaviour:** Strongest structural prior on type embeddings. The DiT sees maximally distinct type tags from the first step of Phase 1 with no risk of collapse or instability.

---

## Summary of All Experimental Conditions

| Condition | Features | Type embeddings | Orth loss | Total state tokens |
|---|---|---|---|---|
| Baseline (current) | s_t only | none | no | 1 |
| Exp 1 | Position history | none | no | k+1 |
| Exp 2A | Full feature set | random init | no | 3k+2 |
| Exp 2B | Full feature set | random init | yes | 3k+2 |
| Exp 2C | Full feature set | orth init, frozen | no | 3k+2 |

---

## Hyperparameters

### History length k

Recommend ablating over k ∈ {3, 5, 10}. Start with k=5 as the primary condition.

- k=3: minimum for computing acceleration (needs at least 3 timesteps)
- k=5: moderate history, 17 state tokens, reasonable compute overhead
- k=10: longer history, 32 state tokens, tests whether longer memory helps

### Decay parameter ρ (eligibility trace)

Fixed scalar, not learned. Recommend ρ=0.9 as default. Can ablate over {0.7, 0.9} if time permits.

### Decay logits w (token decay)

Initialised to w=0 so ρ = σ(0) = 0.5 at the start of Phase 1. Learned from there alongside DiT blocks.

### Orthogonality weight λ_orth (Strain B only)

Default λ_orth = 0.01. Too large and orthogonality dominates reconstruction. Too small and embeddings do not separate. Can ablate over {0.001, 0.01, 0.1} if time permits.

### Phase 0 training steps

Recommend 500–1000 steps. Monitor L_recon until convergence — should plateau well before 1000 steps.

### Phase 1 training steps

Recommend 1.5× the original Stage 1 step count. The DiT must discover a frozen token distribution it has never seen during initialisation.

---

## Evaluation Metrics

### Primary

**Task success rate** — measured across all benchmark tasks. Compare each experimental condition against the baseline.

### Diagnostic

**State token attention weight** — hook into BasicTransformerBlock.attn during evaluation and log the attention matrix. For each block l and each action token h compute:

```
α_feat^(l) = (1/H) · Σ_h Σ_i Attn_{h,i}^(l)
```

Where i ranges over the indices of that feature type's tokens in context_tokens. Key hypothesis to test:

```
α_vel > α_pos
```

Since velocity is orthogonal to vision while position is partially visible to the camera.

Note: nn.MultiheadAttention returns attention weights as the second element of the output tuple only when need_weights=True. Verify this is not being overridden anywhere in the codebase before logging.

**Embedding geometry** — compute PCA and UMAP of:
1. Type embeddings η_feat — do they cluster by feature type?
2. State token embeddings before and after Phase 1 and 2 — does the pretrained structure survive?

**Edge case performance** — construct test scenarios where state should matter most:
- Occlusion: target object hidden from camera but arm position known from proprioception
- Similar visual configurations with different joint states: same scene appearance but arm at different positions
- Near joint limits: arm approaching mechanical stops, camera cannot see this

---

## Expected Results and Interpretation

### If Experiment 1 improves over baseline

The attention budget increase 1/(N+1) → (k+1)/(N+k+1) is the primary driver. Richer features in Experiment 2 should then improve further.

### If Experiment 1 does not improve over baseline

Attention budget alone is insufficient. The DiT does not learn to use additional tokens even when they are available. This suggests the marginalisation problem is deeper than token count and additive injection should be considered as a follow-up.

### If α_vel > α_pos in Experiment 2

Direct evidence that the action head is using information that vision cannot provide. This is the key result supporting the research hypothesis.

### If Strain C outperforms Strain B which outperforms Strain A

Stronger type embedding structure leads to better performance. Supports the argument that orthogonal initialisation is preferable to learned orthogonality.

### If frozen state encoder matches or exceeds unfrozen

The flow matching loss corrupts proprioceptive structure when the state encoder is left unfrozen. This is a genuinely novel finding about the interaction between flow matching and state encoding.

### If frozen state encoder underperforms unfrozen

The DiT and state encoder benefit from co-adaptation during Phase 1 and 2. Freezing the MLP is too restrictive and the system needs joint optimisation to find a useful representation.

---

## Checklist Before Running

### Critical — must pass before any experiment run

- [ ] Smoke test passes: `--max_steps 10 --skip_pretrain --batch_size 2 --history_len 3`
      - imports resolve without error
      - dataset returns `[B, k+1, max_state_dim]` states
      - state encoder produces `[B, 3k+2, 896]` tokens without shape errors
      - forward pass completes and loss is finite
      - `run_attention_eval` runs without crash and logs to wandb
- [ ] k >= 3 (minimum for acceleration computation)
- [ ] Dataset cache directory is empty or partitioned by `k{history_len}` — stale cache from different k will silently load wrong history length
- [ ] Baseline re-run on same data split for fair comparison before launching new experiments

### Phase 0 verification

- [ ] Phase 0 pretraining converges — `L_recon` plateaus before main training begins
- [ ] MLP and LayerNorm frozen after Phase 0 (verify `requires_grad = False`)
- [ ] Decay logits unfrozen going into Phase 1 (verify `requires_grad = True`)
- [ ] For Strain C: type embeddings verified orthogonal after QR init — check pairwise dot products are ~0
- [ ] For Strain B: orthogonality loss `L_orth` decreasing during Phase 0

### Architecture verification

- [ ] Dataset returns state history `[B, k+1, max_state_dim]` not single snapshot `[B, max_state_dim]`
- [ ] `torch.flip` applied in `__getitem__` — index 0 = s_t (most recent), index k = s_{t-k} (oldest)
- [ ] `_need_weights = False` initialised in `BasicTransformerBlock.__init__()` so attribute always exists
- [ ] `enable_attention_weights()` called before eval forward pass — verify `block._need_weights = True` on all blocks
- [ ] Attention hook capturing non-None weights — check `attention_store` is not empty after eval pass
- [ ] Phase 1 step count increased by 1.5× vs original Stage 1 — DiT needs extra steps to discover frozen token distribution

### Known limitations and future work

- [ ] `state_mask` parameter is accepted throughout the pipeline but never used — currently a no-op. If mixed-embodiment batches show unexpected behaviour, wire `state_mask` into `TemporalStateEncoder.forward()` to explicitly zero out padded joint dimensions before MLP encoding. Shape is now `[B, k+1, max_state_dim]` after the history change.
- [ ] `prepare_state` in `Evo1.py` passes `[B, 7]` or `[B, k+1, 7]` through as-is — caller is responsible for correct shape. No inference-time shape validation exists beyond the `ndim == 1` single vector case.
- [ ] Jerk, joint limit proximity, and manipulability features are documented in the README but not yet implemented as extractor classes. When adding: register in `FEATURE_TYPE_REGISTRY`, write extractor class with `feature_name` attribute, add to `EXTRACTOR_REGISTRY` in `state_encoder.py`. No other files need to change.
- [ ] `action_shape` dead assignment in `flow_matching.py` — harmless, clean up when convenient
- [ ] Commented out code blocks in `flow_matching.py` and `Evo1.py` — clean up when convenient