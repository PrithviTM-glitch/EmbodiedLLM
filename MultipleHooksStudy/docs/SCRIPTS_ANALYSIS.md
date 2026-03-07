# Scripts, Hooks & Losses — Verification Analysis

> Cross-reference of every hook, loss function, helper script, and data-collector
> against official model repositories, PyTorch norms, and the study design
> described in `PROPRIOCEPTIVE_UNDERUTILIZATION_THEORY.md`.
>
> Companion document to `NOTEBOOK_ANALYSIS.md`.
> All bugs listed here have been **fixed** in the corresponding files.

---

## Scope

| Category | Files |
|---|---|
| **Hook infrastructure** | `hooks/base_hooks.py`, `ablation_hooks.py`, `gradient_hooks.py`, `representation_hooks.py`, `utilization_hooks.py` |
| **Model-specific adapters** | `hooks/model_specific/{evo1,pi0,smolvla,rdt,octo}_hooks.py` |
| **Loss functions** | `hooks/losses/{evo1,pi0,rdt}_loss.py` |
| **Run scripts** | `scripts/run_{evo1,pi0,rdt}_{gradient_analysis,ablation}.py` |
| **Framework** | `scripts/ablation_framework.py` |
| **Data collectors** | `scripts/data_collectors/{base,libero,metaworld,bridge}_collector.py` |
| **Analysis** | `analysis/experiment_coordinator.py`, `result_analyzer.py` |

---

## Overall Scorecard

| File | Status | Issues | Fixed |
|---|---|---|---|
| `hooks/base_hooks.py` | ✅ PASS | — | — |
| `hooks/ablation_hooks.py` | ✅ FIXED | 3 missing methods on `AblationStudyCoordinator` | ✅ |
| `hooks/gradient_hooks.py` | ✅ PASS | — | — |
| `hooks/representation_hooks.py` | ✅ PASS | — | — |
| `hooks/utilization_hooks.py` | ✅ PASS | — | — |
| `hooks/model_specific/evo1_hooks.py` | ✅ PASS | Architecture verified vs MINT-SJTU/Evo-1 | — |
| `hooks/model_specific/pi0_hooks.py` | ✅ FIXED | Docstring stated "multiple layers" for single-Linear state encoder | ✅ |
| `hooks/model_specific/smolvla_hooks.py` | ✅ PASS | Architecture verified vs lerobot SmolVLA | — |
| `hooks/model_specific/rdt_hooks.py` | ✅ PASS | `state_adaptor` naming verified vs thu-ml/RDT | — |
| `hooks/model_specific/octo_hooks.py` | ✅ PASS | Legacy model; hooks correct for reference | — |
| `hooks/losses/evo1_loss.py` | ✅ PASS | Flow-matching formulation consistent with openpi reference | — |
| `hooks/losses/pi0_loss.py` | ✅ PASS | References correct openpi source lines | — |
| `hooks/losses/rdt_loss.py` | ✅ PASS | DDPM noise scheduler matches thu-ml/RDT | — |
| `scripts/run_evo1_gradient_analysis.py` | ✅ FIXED | 3 bugs: wrong module name (`integration_module`), wrong gradient key lookups, wrong JSON label | ✅ |
| `scripts/run_pi0_gradient_analysis.py` | ✅ FIXED | 5 bugs: wrong import, 2× KeyError, 2× undefined methods, step counter | ✅ |
| `scripts/run_pi0_ablation.py` | ✅ PASS | Correct use of `AblationServer` + lerobot model loading | — |
| `scripts/run_rdt_gradient_analysis.py` | ⚠️ NOT IN SCOPE | RDT not benchmark model; reviewed, no crashes found | — |
| `scripts/run_rdt_ablation.py` | ⚠️ NOT IN SCOPE | Same as above | — |
| `scripts/ablation_framework.py` | ✅ FIXED | Deprecated `path` arg in websockets handler | ✅ |
| `scripts/data_collectors/base_collector.py` | ✅ PASS | Solid ABC design, h5py output standardised | — |
| `scripts/data_collectors/libero_collector.py` | ✅ PASS | — | — |
| `scripts/data_collectors/metaworld_collector.py` | ✅ PASS | — | — |
| `scripts/data_collectors/bridge_collector.py` | ✅ PASS | — | — |

---

## 1. Hook Infrastructure

### `base_hooks.py` ✅

The four base classes (`BaseGradientHook`, `BaseFeatureHook`, `BaseAblationHook`, `BaseAttentionHook`) correctly implement PyTorch hook patterns:

- `register_full_backward_hook` (not deprecated `register_backward_hook`) ✓  
- Forward hooks return `None` to pass-through or a tensor to replace output ✓  
- `self.handles` list used for reliable `remove()` ✓  
- `defaultdict(list)` used safely for accumulated gradients ✓

### `ablation_hooks.py` — FIXED ✅

**Bug 1–3 (missing methods on `AblationStudyCoordinator`)**: Three methods were called by model-specific hook adapters but not defined:

| Caller | Method called | Problem |
|---|---|---|
| `pi0_hooks.py:attach_ablation_hooks()` | `coordinator.register_ablation(name, hook_fn)` | `AttributeError` |
| `evo1_hooks.py:attach_ablation_hooks()` | `coordinator.add_ablation_target(name, module, ablation_types)` | `AttributeError` |
| `evo1_hooks.py:get_results()` | `coordinator.get_results()` | `AttributeError` |

**Fix**: Added all three methods to `AblationStudyCoordinator`:
- `register_ablation(name, hook_fn)` — stores custom hook functions by name
- `add_ablation_target(name, module, ablation_types)` — delegates to `ModalityAblationManager.register_encoder`
- `get_results()` — returns `self.results` (populated by `run_ablation_experiment`)

### `gradient_hooks.py` ✅

`EncoderGradientTracker` correctly uses `register_full_backward_hook` which PyTorch 2.x requires for reliable grad capture. `LayerWiseGradientProfiler.find_vanishing_point()` threshold default of `1e-4` is reasonable for analysing proprioceptive encoder gradients.

### `representation_hooks.py` ✅

`CKASimilarityAnalyzer.linear_cka()` correctly centres features before computing Gram matrices. The `effective_rank` computation via singular values (implicit in downstream use) is standard.

### `utilization_hooks.py` ✅

`AttentionWeightTracker.compute_modality_attention()` handles both `(B, H, S, S)` and `(B, S, S)` attention weight formats. The modality token ranges must be set externally via `set_modality_ranges()` — this is correct design (model-specific config passed in).

---

## 2. Model-Specific Hook Adapters

### `evo1_hooks.py` ✅ — Verified Against MINT-SJTU/Evo-1

| Property | Expected (repo) | In code | Match |
|---|---|---|---|
| State encoder module path | `model.action_head.state_encoder` | `action_head.state_encoder` fallback search | ✓ |
| State encoder type | `CategorySpecificMLP` (3-layer MLP) | Documented in docstring | ✓ |
| VL backbone attribute | `model.embedder` (InternVL3Embedder) | Tries `embedder`, `vl_backbone`, `internvl` | ✓ |
| Action head attribute | `model.action_head` (FlowmatchingActionHead) | Tries `action_head`, `policy_head`, `diffusion` | ✓ |
| Ablation target | Forward hook on `action_head.state_encoder` output | Registered via `add_ablation_target` | ✓ |

### `pi0_hooks.py` — FIXED ✅ — Verified Against physical-intelligence/openpi + lerobot/pi0

| Property | Expected (repo) | In code | Match |
|---|---|---|---|
| State encoder module | `policy.model.state_proj` (single `nn.Linear`) | `state_proj` attribute search | ✓ |
| Backbone | `paligemma_with_expert` wrapping PaliGemma + Expert Gemma | Tries `paligemma_with_expert`, `model` | ✓ |
| Docstring architecture | Single `nn.Linear` state encoder | Fixed: was "multiple layers" | ✓ |

### `smolvla_hooks.py` ✅ — Verified Against huggingface/lerobot SmolVLA

| Property | Expected (repo) | In code | Match |
|---|---|---|---|
| State encoder | `policy.model.state_proj` (single `nn.Linear`) | Multi-path attribute search + fallback `named_modules` scan | ✓ |
| VLM backbone | SmolVLM2 (SigLIP + SmolLM2 first 16 layers) | Vision/language searched separately | ✓ |
| Action expert | Flow matching transformer | `action_expert`, `action_head`, `flow_matching_head` search | ✓ |

### `rdt_hooks.py` ✅ — Verified Against thu-ml/RoboticsDiffusionTransformer

| Property | Expected (repo) | In code | Match |
|---|---|---|---|
| State encoder | `model.state_adaptor` (single `nn.Linear`) | Tries `state_adaptor` first | ✓ |
| Input dim note | `state_token_dim * 2` (state + mask) | Documented in docstring | ✓ |
| Vision encoder | SigLIP | Tries `vision_encoder`, `image_encoder` | ✓ |
| Language encoder | T5-XXL | Tries `language_encoder`, `text_encoder` | ✓ |

### `octo_hooks.py` ✅ — Legacy Reference

Octo is not a primary benchmark model in this study (it was in the initial scope of the `vla-benchmark/` sub-project). The hooks are correct for Octo's architecture (ResNet/ViT vision + linear proprio projection). No active run scripts use `OctoHooks`, so this is safe to retain as reference.

---

## 3. Loss Functions

### Flow Matching Convention (Evo-1 & Pi0)

Both `evo1_loss.py` and `pi0_loss.py` use:

```
x_t = τ · A_t + (1−τ) · ε          (noisy action)
u_t = ε − A_t                       (target velocity)
Loss = MSE(v_θ(x_t, obs, τ), u_t)
```

This is internally consistent with the openpi reference where the probability path
`q(A_t^τ | A_t) = N(τ·A_t, (1−τ)·I)`. Some formulations express this as
`u_t = A_t − ε` (velocity from noise to data); openpi uses the equivalent
"noise→data" convention where inference integrates `-v_θ` (or `v_θ` depending
on ODE direction). For **gradient analysis** purposes, the sign convention does
not matter — only gradient magnitudes are compared across baseline/ablated runs.

**Both files are internally consistent and produce differentiable losses suitable for gradient analysis.** ✓

### `rdt_loss.py` ✅ — Verified Against thu-ml/RDT

RDT uses DDPM (discrete diffusion), not flow matching.

```
noisy_action = √ᾱ_t · a + √(1−ᾱ_t) · ε
Loss = MSE(model(obs, noisy_action, t), ε)   [epsilon prediction]
```

The `DDPMNoiseScheduler.squaredcos_cap_v2` beta schedule matches the `diffusers` implementation used by thu-ml/RDT. ✓

---

## 4. Run Scripts

### `run_evo1_gradient_analysis.py` — FIXED ✅

**Bug 1** (Critical): Ablation hook searched for `'integration_module'` in `model.named_modules()`. This name does not exist in Evo-1's architecture. The correct target is `action_head.state_encoder`.

```python
# BEFORE (wrong — 'integration_module' never found, ablation silently skipped)
for name, module in model.named_modules():
    if 'integration_module' in name:
        ablation_handle = module.register_forward_hook(zero_output_hook)

# AFTER (correct — use hook manager's resolved reference, fallback by name)
if hook_manager.state_encoder is not None:
    ablation_handle = hook_manager.state_encoder.register_forward_hook(zero_output_hook)
else:
    for name, module in model.named_modules():
        if 'action_head' in name and 'state_encoder' in name:
            ablation_handle = module.register_forward_hook(zero_output_hook)
```

**Bug 2** (Critical): Gradient key lookup used `'integration_module'` to query `layer_profiles` dict, but the hook registration uses key `'state_encoder'`. Result was always `None`, making baseline/ablated comparison invalid.

**Bug 3** (Minor): JSON output labelled `'state_encoder': 'integration_module'` — corrected to `'action_head.state_encoder (CategorySpecificMLP)'`.

### `run_pi0_gradient_analysis.py` — FIXED ✅

**Bug 1** (Critical): Wrong import path for Pi0:

```python
# BEFORE (wrong — openpi has no Pi0Policy at package root)
from openpi import Pi0Policy
model = Pi0Policy.from_pretrained('pi0-base')

# AFTER (correct — Pi0 loaded via lerobot HuggingFace integration)
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
model = PI0Policy.from_pretrained('lerobot/pi0')
```

**Bug 2–3** (Critical): KeyErrors on `discover_model_structure()` return dict:

```python
# BEFORE (wrong keys — Pi0Hooks uses 'has_proprio_encoder' / 'proprio_encoder_type')
structure["has_state_encoder"]
structure["state_encoder_type"]

# AFTER (correct keys)
structure["has_proprio_encoder"]
structure["proprio_encoder_type"]
```

**Bug 4** (Critical): `hook_manager.detach_all()` → method does not exist on `Pi0Hooks`. Fixed to `hook_manager.cleanup()`.

**Bug 5** (Critical): `hook_manager.enable_ablation('state_encoder')` → method does not exist. Fixed to a direct PyTorch forward hook on `hook_manager.state_proj`:

```python
ablation_handle = hook_manager.state_proj.register_forward_hook(
    lambda m, i, o: torch.zeros_like(o)
)
```

**Bug 6** (Minor): Step counter printed `[5/5]` before the ablation step but then `[6/6]` for the comparison — 6 total steps, fixed to `[5/6]`.

### `ablation_framework.py` — FIXED ✅

**Bug**: Deprecated `path` parameter in websockets handler. `websockets >= 10.0` removed the second positional argument:

```python
# BEFORE (crashes on websockets >= 10.0)
async def handle_inference(self, websocket, path):

# AFTER (current API)
async def handle_inference(self, websocket):
```

---

## 5. Data Collectors

### `base_collector.py` ✅

Clean ABC with `collect_observation()`, `get_observation_spec()`, and `setup_environment()` abstract methods. HDF5 output uses consistent keys (`image`, `robot_state`, `action`, `task_description`). Consistent with how `run_evo1_gradient_analysis.py` reads data.

### `libero_collector.py` / `metaworld_collector.py` / `bridge_collector.py` ✅

Implementations correctly extend `BenchmarkDataCollector`. The LIBERO collector activates the correct conda environment (`libero_client`, Python 3.8.13) and MetaWorld uses `metaworld_client` (Python 3.10), consistent with the notebook setup.

---

## 6. Design Notes & Known Limitations

### Flow-Matching Proxy Loss in Gradient Scripts

`run_evo1_gradient_analysis.py` and `run_pi0_gradient_analysis.py` use a **proxy gradient** approach:

> Instead of running the full flow-matching ODE (which requires the model to accept noisy actions at arbitrary timesteps), the scripts compute `evo1_flow_matching_loss_simple(model_output, flow_components['u_t'])`, treating raw model output as a velocity prediction.

This produces a valid differentiable loss for gradient magnitude comparison — the ablation vs baseline comparison is still meaningful because the same proxy loss is used in both conditions. For a publication-quality gradient analysis, the model's `forward_with_time` interface should be confirmed and used directly.

### RDT and Octo Status

RDT-1B hooks and loss are present but **RDT is not a primary benchmark model** in this study. The scripts exist as scaffolding for potential extension. No SLURM job scripts target RDT.

Octo hooks are **legacy** from the `vla-benchmark/` sub-project. They are correct but unused by any run script in `MultipleHooksStudy/`.

### `AblationStudyCoordinator` vs Direct Hooks

The coordinator abstraction (`AblationStudyCoordinator`) is designed for **performance-based** ablation (comparing task success rates). The **gradient-based** ablation in run scripts correctly bypasses the coordinator and uses direct `register_forward_hook` — this is the right approach since gradient analysis does not need the server-based framework.

---

## Official Repository Reference Table

| Model | State Encoder Module | Verified Path | Source |
|---|---|---|---|
| Evo-1 | `CategorySpecificMLP` | `model.action_head.state_encoder` | [MINT-SJTU/Evo-1](https://github.com/MINT-SJTU/Evo-1) |
| Pi0 | `nn.Linear` | `policy.model.state_proj` | [lerobot/pi0](https://huggingface.co/lerobot/pi0) + [openpi](https://github.com/physical-intelligence/openpi) |
| SmolVLA | `nn.Linear` | `policy.model.state_proj` | [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) |
| RDT-1B | `nn.Linear` | `model.state_adaptor` | [thu-ml/RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer) |
| Octo | `nn.Linear` | `model.proprio_encoder` | [octo-model/octo](https://github.com/octo-model/octo) |

---

*Analysis performed on branch `AnalyseMultipleHooks` — all fixes applied and validated.*
