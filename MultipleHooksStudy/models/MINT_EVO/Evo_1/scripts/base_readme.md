# Baseline Train.py — Changes & Training Commands

## Context

The baseline uses the Evo-1 codebase from `MINT_EVO/Evo_1/`. The original vanilla `train.py` from the GitHub repo had several issues that prevented it from running correctly with our setup. This document describes every change made and why, plus the exact commands for Stage 1 and Stage 2.

---

## Changes from Vanilla train.py

### 1 — `Accelerator()` moved inside `train()`

**Vanilla:**
```python
# At module level (line ~22)
accelerator = Accelerator()
```

**Modified:**
```python
# Removed from module level
# Added as first line inside train():
def train(config):
    global accelerator
    accelerator = Accelerator(mixed_precision="bf16")
```

**Why:** When `accelerate launch --deepspeed_config_file` spawns the process, DeepSpeed environment variables are already set. `Accelerator()` called at import time (module level) silently kills the process before `train()` ever runs. Moving it inside `train()` ensures it's only called after the process is fully initialised by the launcher.

---

### 2 — Scheduler registered for checkpointing

**Vanilla:**
```python
scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=step))
# no registration
```

**Modified:**
```python
scheduler = LambdaLR(optimizer, get_lr_lambda(warmup_steps, max_steps, resume_step=0))
accelerator.register_for_checkpointing(scheduler)
```

**Why:** `accelerator.save_state()` / `accelerator.load_state()` only captures objects that have been explicitly registered. Without registration, the scheduler state is lost on resume and the LR curve restarts from the beginning. The scheduler is created with `resume_step=0` (not `resume_step=step`) so that `load_state` can overwrite it with the saved state for within-stage resumes.

---

### 3 — Checkpoint format switched from DeepSpeed to Accelerate

**Vanilla `save_checkpoint`:**
```python
model_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
# Produces: mp_rank_00_model_states.pt
```

**Modified `save_checkpoint`:**
```python
accelerator.save_state(checkpoint_dir)
# Produces: model.safetensors, optimizer.bin, scheduler pkl, RNG state
```

**Why:** Stage 1 was originally trained using `accelerator.save_state()` due to the DeepSpeed crash issue at module level. Stage 2 must use the same format to be able to resume. The vanilla DeepSpeed format (`mp_rank_00_model_states.pt`) is incompatible with `accelerator.load_state()`.

**Note:** Meta is saved as `meta.json` (not `checkpoint_meta.json` as in our modified `train.py`).

---

### 4 — `load_checkpoint` rewritten for Accelerate format + cross-stage support

**Vanilla (`load_checkpoint_with_deepspeed`):**
```python
model_engine.load_checkpoint(load_dir, tag=tag, load_optimizer_states=..., ...)
# Loads mp_rank_00_model_states.pt via DeepSpeed
```

**Modified (`load_checkpoint`):**
```python
def load_checkpoint(model_engine, load_dir, accelerator, tag,
                    resume_pretrain=False, model_only=False):
    checkpoint_dir = os.path.join(load_dir, tag)
    if model_only:
        # Cross-stage resume: load weights only, skip optimizer/scheduler
        from safetensors.torch import load_file
        unwrapped = accelerator.unwrap_model(model_engine)
        unwrapped.load_state_dict(
            load_file(os.path.join(checkpoint_dir, "model.safetensors")), strict=True
        )
    else:
        # Within-stage resume: load full state
        accelerator.load_state(checkpoint_dir)
    ...
```

**Why:** The Stage 1 checkpoint is in Accelerate format. The vanilla DeepSpeed loader cannot read it. `model_only=True` (triggered by `--resume_model_only`) is needed for Stage 1 → Stage 2 cross-stage resume because the optimizer param groups differ — Stage 1 trains action head only, Stage 2 adds VLM — so loading the full optimizer state would crash with a param group mismatch.

---

### 5 — `--resume_model_only` argparse argument added

```python
parser.add_argument("--resume_model_only", action="store_true",
                    help="Load model weights only. Use for cross-stage resumes.")
```

**Why:** Needed to trigger the `model_only` path in `load_checkpoint` for Stage 1 → Stage 2.

---

### 6 — Gradient checkpointing removed

**Not present in vanilla. Added then removed.**

The block below was added to support VLM finetuning on a single A100 but was removed because it slows training by approximately 3-4x:

```python
# REMOVED — slows training 3-4x
if config.get("finetune_vlm", False):
    model.embedder.model.language_model.gradient_checkpointing_enable()
    model.embedder.model.vision_model.gradient_checkpointing_enable()
```

Without gradient checkpointing, Stage 2 with `--finetune_vlm` and `batch_size=16` requires approximately 60-65GB VRAM. This fits on an 80GB A100 but is tight. If OOM errors occur, re-enable gradient checkpointing or reduce batch size.

---

### 7 — WandB set to offline mode

**Vanilla:** `--disable_wandb` flag disables wandb entirely.

**Modified:** `wandb.init(..., mode="offline")` — wandb always runs but logs locally. Sync manually with:
```bash
wandb sync /path/to/wandb/offline-run-*/
```

---

## Summary Table

| Change | Vanilla | Modified | Reason |
|--------|---------|----------|--------|
| `Accelerator()` location | Module level | Inside `train()` | DeepSpeed crash fix |
| Scheduler registration | Not registered | `register_for_checkpointing` | Correct resume |
| Checkpoint format | DeepSpeed (`mp_rank_00_model_states.pt`) | Accelerate (`model.safetensors`) | Compatibility with Stage 1 |
| Load function | `load_checkpoint_with_deepspeed` | `load_checkpoint` with `model_only` | Cross-stage resume |
| `--resume_model_only` | Not present | Added | Cross-stage resume flag |
| Gradient checkpointing | Not present | Added then removed | Speed (3-4x overhead) |
| WandB mode | Disabled via flag | Offline | Local logging |

---

## Training Commands

### Stage 1

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/tmprithvi/MINT_EVO/Evo_1

accelerate launch \
  --num_processes 1 --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  scripts/train.py \
  --run_name Evo1_metaworld_baseline_stage1 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 \
  --batch_size 16 --image_size 448 \
  --max_steps 10000 --warmup_steps 1000 \
  --log_interval 10 --ckpt_interval 2500 \
  --grad_clip_norm 1.0 --num_layers 8 \
  --horizon 50 --per_action_dim 24 --state_dim 24 \
  --finetune_action_head \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path /home/tmprithvi/MINT_EVO/Evo_1/dataset/metaworld_config.yaml \
  --save_dir /home/tmprithvi/tmp/baseline/stage1 \
  --num_workers 4
```

**What trains:** Action head only. VLM frozen. No state encoder (baseline).

**Checkpoint format:** Accelerate (`model.safetensors`) saved at `step_2500`, `step_5000`, ..., `step_best`, `step_final`.

---

### Stage 2

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/tmprithvi/MINT_EVO/Evo_1

accelerate launch \
  --num_processes 1 --num_machines 1 \
  --deepspeed_config_file ds_config.json \
  scripts/train.py \
  --run_name Evo1_metaworld_baseline_stage2 \
  --action_head flowmatching \
  --use_augmentation \
  --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 \
  --batch_size 16 --image_size 448 \
  --max_steps 80000 --warmup_steps 1000 \
  --log_interval 10 --ckpt_interval 2500 \
  --grad_clip_norm 1.0 --num_layers 8 \
  --horizon 50 --per_action_dim 24 --state_dim 24 \
  --finetune_vlm --finetune_action_head \
  --vlm_name OpenGVLab/InternVL3-1B \
  --dataset_config_path /home/tmprithvi/MINT_EVO/Evo_1/dataset/metaworld_config.yaml \
  --save_dir /home/tmprithvi/tmp/baseline/stage2 \
  --num_workers 4 \
  --resume --resume_model_only \
  --resume_path /home/tmprithvi/tmp/baseline/stage2/step_12500
```

**What trains:** Full model — VLM + action head. Resumes from Stage 1 weights only (optimizer state discarded because param groups differ between stages).

**Key flags:**
- `--resume_model_only` — loads `model.safetensors` only, skips optimizer/scheduler state. Required for cross-stage resume.
- `--resume_path` — point to whichever Stage 1 checkpoint you want to start from (`step_10000`, `step_best`, `step_final`).

**Checkpoint format:** Accelerate (`model.safetensors`) saved at every `ckpt_interval` steps and whenever a new best loss is achieved after step 1000.

---

### Within-Stage Resume (Stage 2 → Stage 2 after preemption)

```bash
accelerate launch \
  ... (same flags as Stage 2) \
  --resume \
  --resume_path /home/tmprithvi/tmp/baseline/stage2/step_XXXX
  # Note: NO --resume_model_only — loads full state including optimizer and scheduler
```

---

## Evaluation

Use `eval_server.py` (not the original `evo1_server_json.py`) since checkpoints are in Accelerate format:

```bash
python eval_server.py \
  --ckpt_dir /home/tmprithvi/tmp/baseline/stage2/step_best \
  --port 9000
```

The original server loads `mp_rank_00_model_states.pt` which does not exist in Accelerate checkpoints.