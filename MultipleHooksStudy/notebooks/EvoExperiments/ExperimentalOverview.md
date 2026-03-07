# Evo-1 DiT Speed & Efficiency Experiments
> Reference document for proposed experiments on the [MINT-SJTU/Evo-1](https://github.com/MINT-SJTU/Evo-1) codebase  
> Paper: [arXiv 2511.04555](https://arxiv.org/abs/2511.04555)

---

## Architecture Quick Reference

| Component | Detail |
|---|---|
| VLM Backbone | InternVL3-1B (14 layers retained) |
| Visual Encoder | InternViT-300M, 448×448, 4× pixel-unshuffle |
| Integration Module | Module A — mid-layer cross-attention + concat with proprioceptive state |
| Action Expert (DiT) | 8-layer cross-attention DiT (`--num_layers 8`) |
| DiT Attention Type | Cross-attention only — action tokens as Q, VLM+state as KV (no self-attention between action tokens) |
| Action Head | Flow matching (ODE from t=1 noise → t=0 action) |
| Action Horizon | 50 steps (`--horizon 50`) |
| Action Dim | 24 per step (`--per_action_dim 24`) |
| State Dim | 24 (`--state_dim 24`) |
| Total Params | 0.77B |
| GPU Memory | 2.3 GB |
| Baseline Inference | **16.4 Hz** on RTX 4090d |
| Benchmarks | MetaWorld 80.6% · LIBERO 94.8% (Long: 92.3%) · RoboTwin 37.8% · Real-world 78% |

### Inference Pipeline (per observation)
```
Client → [obs dict: 3× 448×448 images, state, prompt]
                        ↓ WebSocket (Evo1_server.py)
VLM Encode (InternVL3-1B, 14 layers)
                        ↓
Integration Module (mid-layer cross-attn → concat state)
                        ↓ KV fixed for all ODE steps
DiT Denoise — N ODE steps × 8 layers (cross-attn Q=action, KV=vlm+state)
                        ↓
Action chunk [50 × 24] → Client
```

---

## Experiment 1: NFE (ODE Steps) Reduction

### Core Idea
Flow matching solves this ODE at inference:

```
dx/dt = v_θ(x_t, t, conditioning)    t: 1→0
```

Each ODE step = 1 full DiT forward pass. Reducing the number of steps (NFE) directly reduces inference time proportionally.

### Where to Look in Code
- **`Evo_1/scripts/Evo1_server.py`** — around line 149 (checkpoint load). Find the inference loop; it will look approximately like:

```python
timesteps = torch.linspace(1.0, 0.0, num_steps)  # default likely 10 or 100
x = torch.randn(batch, horizon, action_dim)
for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
    v = dit_model(x, t_curr, conditioning)   # ← 1 DiT forward pass per step
    x = x + (t_next - t_curr) * v           # Euler step
```

### Phase 1 — Zero-Retraining Sweep
Change `num_steps` only at inference. No weight changes.

| `num_steps` | Expected DiT speedup | Notes |
|---|---|---|
| 10 (baseline) | 1× | Assumed default |
| 5 | ~2× | Safe starting point |
| 3 | ~3.3× | Usually acceptable for FM |
| 2 | ~5× | Often works well due to straight FM trajectories |
| 1 | ~10× | Single Euler step; surprising how often this holds |

### Phase 2 — Reflow Fine-tuning (if needed)
If `num_steps=2` loses more than ~5% success rate, do a reflow distillation:

1. Run the trained model at 10 steps on training data → generate "teacher" straight trajectories.
2. Retrain DiT to predict the constant velocity field:
   ```
   v_student(x_t, t) ≈ (x_0_teacher - x_1_noise) / 1.0
   ```
3. Only needs **Stage 1 equivalent (~5k steps)** — no full Stage 2 retraining.

### Metrics to Track
- Success rate on MetaWorld (short-horizon) and LIBERO-Long (long-horizon)
- Wall-clock DiT time (ms) per action chunk
- Full-stack Hz (VLM + integration + DiT)

### Key Hypothesis
LIBERO-Long (92.3% baseline) will degrade faster than MetaWorld with fewer NFE steps, because long-horizon tasks are more sensitive to action trajectory precision. The differential degradation rate is itself a publishable finding.

### Expected Result
- `num_steps=2` → ~5× DiT speedup, likely <5% drop on MetaWorld, possibly more on LIBERO-Long.
- Reflow at `num_steps=1` → ~10× DiT speedup with <3% drop after fine-tuning.

---

## Experiment 2: DiT Layer Ablation / Depth Pruning

### Core Idea
The action expert uses `--num_layers 8` of **cross-attention only** (no self-attention between action tokens). Because action tokens are refined independently by the conditioning signal at each layer (no inter-token communication), early layers may saturate faster. The depth may be over-parameterized.

### Why Cross-Attention-Only DiT Is Different
In standard image DiTs: self-attention builds spatial structure across layers → depth matters a lot.
In Evo-1's DiT: each layer independently refines action tokens using the (fixed) VLM KV → the marginal value of additional layers may fall off faster.

### Phase 1 — Retrain from Scratch (Cleanest Comparison)

All other hyperparameters identical. Only `--num_layers` changes.

```bash
# Ablation A
accelerate launch scripts/train.py ... --num_layers 6 ...

# Ablation B
accelerate launch scripts/train.py ... --num_layers 4 ...

# Ablation C
accelerate launch scripts/train.py ... --num_layers 2 ...
```

Each run = Stage 1 (5k steps) + Stage 2 (80k steps). Can check trend at 40k steps before committing.

### Phase 2 — Structured Pruning from Trained Checkpoint (Cheaper)
Skip retraining from scratch. Instead:

1. Hook into each of the 8 DiT layers and measure activation change ratio:
   ```python
   # For each layer i:
   ratio_i = ||layer_i_output - layer_i_input||_2 / ||layer_i_input||_2
   # Low ratio → layer changes very little → pruning candidate
   ```
2. Remove the 2 layers with lowest ratio.
3. Fine-tune for 5k–10k steps (Stage 1 equivalent only).

### FLOP Analysis
Approximate DiT FLOPs per action chunk:
```
FLOPs ≈ NFE × num_layers × 2 × seq_len × hidden_dim²
      = 10  ×  8  ×  2  × 50  × hidden_dim²
```

Reducing layers 8→4 = **2× FLOP reduction** in the DiT.  
Combined with Exp 1 (NFE 10→2) = **10× total DiT FLOP reduction**.  
These compose multiplicatively.

> **Note:** The VLM's KV computation (integration module output) is computed once per observation and is amortized across all NFE steps. The per-step cost is dominated by the DiT's cross-attention QKV projection. Layer reduction directly proportionally reduces per-step inference time.

### Metrics to Track
- Success rate vs. `num_layers` on both benchmarks
- FLOPs per action chunk (compute the formula above)
- GPU time (ms) for DiT only (profile with `torch.profiler`)

### Expected Result
Hypothesis: 4-layer DiT retains >95% of 8-layer performance due to the cross-attention-only structure (no accumulated self-attention structure to lose). This would be a clean result showing the architecture is over-parameterized at depth 8.

---

## Experiment 3: Temporal Reuse / Adaptive Replanning

### Core Idea
Evo-1 generates a 50-step action chunk but calls the server every step by default. Many consecutive observations are nearly identical (robot moves slowly). Skip server calls and reuse the cached chunk, only replanning when needed.

### The Server-Client Architecture
```
# Evo_1/scripts/Evo1_client_xarm6.py — main loop

obs = {
    "image": [base_proc, wrist_proc, dummy_proc],  # 448×448 each
    "image_mask": [1, 1, 0],
    "state": state.tolist(),
    "action_mask": action_mask,
    "prompt": task_instruction
}
await ws.send(json.dumps(obs))
result = await ws.recv()
action_chunk = torch.tensor(json.loads(result))  # shape: [50, 24]
```

### Option A — Fixed-k Reuse (Easiest, No Model Changes)
Only call the server every `k` steps; execute from cached chunk otherwise.

```python
if step_count % k == 0:
    action_chunk = await query_server(obs)
    chunk_pointer = 0

action = action_chunk[chunk_pointer]
chunk_pointer += 1
step_count += 1
```

Sweep `k` ∈ {1, 2, 5, 10, 25}:

| `k` | Effective replan rate | Effective system Hz (control still at 16.4 Hz) |
|---|---|---|
| 1 | 16.4 Hz | Baseline |
| 2 | 8.2 Hz | 2× compute saving |
| 5 | 3.3 Hz | 5× compute saving |
| 10 | 1.6 Hz | 10× compute saving |
| 25 | 0.66 Hz | 25× compute saving (still within 50-step chunk) |

### Option B — Similarity Gate (Smarter Replanning)
Run a lightweight client-side proxy encoder to decide whether to replan:

```python
import torch.nn.functional as F

def should_replan(curr_frame, prev_frame, threshold=0.97):
    # Cheap option: downsampled pixel L2 difference
    curr_small = F.avg_pool2d(curr_frame, 16)  # 28×28
    prev_small = F.avg_pool2d(prev_frame, 16)
    diff = (curr_small - prev_small).norm() / curr_small.norm()
    return diff > (1 - threshold)

# Or use a frozen MobileNetV3 client-side for better semantic sensitivity
```

### Option C — Chunk Blending (Reduces Boundary Discontinuities)
Query server every 25 steps, but blend chunk boundaries with a linear interpolation window:

```python
BLEND_WINDOW = 5
# At step 25 (new chunk arrives):
for i in range(BLEND_WINDOW):
    alpha = i / BLEND_WINDOW
    blended_action = (1 - alpha) * old_chunk[25+i] + alpha * new_chunk[i]
```

### Metrics to Track
- Success rate vs. `k` on MetaWorld (short-horizon) and LIBERO-Long (long-horizon)
- Effective system-level Hz
- Failure mode analysis: does the robot fail at chunk boundaries? During fast motions?

### Key Scientific Hypothesis
The degradation with larger `k` will be **task-dependent**: MetaWorld (fast, reactive, short-horizon) should degrade quickly with `k>2`. LIBERO-Long (slower, more sustained, longer-horizon) should be tolerant up to `k=10`. This differential reveals the actual temporal correlation structure of the learned policy — how long the action chunk stays "valid" as a function of task dynamics.

### Expected Result
- `k=5` on LIBERO-Long: <5% degradation (chunk stays valid for many steps in slow manipulation tasks)
- `k=5` on MetaWorld: potentially 10–20% degradation (fast, reactive tasks need fresh observations)

---

## Recommended Execution Order

```
Exp 3 (temporal reuse) → Exp 1 (NFE sweep) → Exp 2 (layer ablation)
  Zero retraining         Zero retraining        Retraining required
  ~1 day                  ~1 day                 ~1–2 weeks (per variant)
```

**Rationale:**
- Exp 3 and Exp 1 require zero retraining — run them first to understand the performance floor before committing GPU time to Exp 2.
- If NFE=2 (Exp 1) already meets your target Hz, you may not need Exp 2 at all.
- Exp 2 (layer ablation) is the most expensive — only run if NFE reduction alone is insufficient.

---

## Combined Pareto Front (Target Result)

The publishable outcome is a 3D surface:

```
Axes: (num_layers) × (NFE steps) × (k reuse interval)
Color: success rate on MetaWorld / LIBERO-Long
```

The hypothesis is that the optimal point achieves **5–10× total compute reduction** with **<5% success rate drop** — showing that Evo-1's DiT is significantly over-provisioned at its default settings.

| Config | DiT speedup | Control Hz (est.) | Risk |
|---|---|---|---|
| Baseline (8L, NFE=10, k=1) | 1× | 16.4 Hz | — |
| NFE=2, k=1 | ~5× DiT | ~50 Hz (est.) | Low |
| NFE=2, k=5 | ~5× DiT + 5× replan | ~50 Hz control, 10 Hz replan | Medium |
| 4L, NFE=2, k=5 | ~10× DiT + 5× replan | ~100 Hz (est.) | Medium |
| 2L, NFE=1, k=10 | ~40× DiT | Very fast | High |

---

## Key Files Reference

| File | Relevance |
|---|---|
| `Evo_1/scripts/Evo1_server.py` | DiT inference loop, NFE steps, checkpoint loading (L149) |
| `Evo_1/scripts/train.py` | `--num_layers`, `--horizon`, `--per_action_dim`, `--num_steps` args |
| `Evo_1/scripts/Evo1_client_xarm6.py` | Client loop — where temporal reuse logic goes |
| `MetaWorld_evaluation/mt50_evo1_client_prompt.py` | MetaWorld eval client (port config L40) |
| `LIBERO_evaluation/libero_client_4tasks.py` | LIBERO eval client |
| `Evo_1/dataset/config.yaml` | Dataset + camera mapping config |
| `Evo_1/ds_config.json` | DeepSpeed ZeRO config for training |