# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2026-06-02]

### Added

- **`Mod_server.py`** — New inference server derived from `Evo1_server.py` with minimal changes to support Accelerate safetensors checkpoints (produced by `train.py`). Replaces the DeepSpeed `mp_rank_00_model_states.pt` loading path. Accepts `--ckpt_dir` and `--port` via argparse.

### Changed

- **`Mod_server.py` vs `Evo1_server.py` — checkpoint loading** — `load_model_and_normalizer()` now uses `safetensors.torch.load_file("model.safetensors")` instead of `torch.load("mp_rank_00_model_states.pt")["module"]`.
- **`Mod_server.py` vs `Evo1_server.py` — `__main__` entry point** — `ckpt_dir` and `port` are now CLI arguments via `argparse` instead of hardcoded string literals.
- **`Mod_server.py` vs `Evo1_server.py` — context managers** — `with torch.no_grad() and torch.amp.autocast(...)` corrected to `with torch.no_grad(), torch.amp.autocast(...)` so both context managers are active during inference.

### Removed

- **`Mod_server.py` vs `Evo1_server.py` — `fvcore` import** — `from fvcore.nn import FlopCountAnalysis` removed; it was imported but never used and would cause `ImportError` on environments without `fvcore`.

---

## [Unreleased]

### Added

- **`vanilla_train.py`** — Original unmodified training script preserved as a reference baseline.
- **`dataset/metaworld_config.yaml`** — New MetaWorld-specific dataset configuration file.
- **`train.py` — `--resume_model_only` CLI flag** — Loads only model weights from a safetensors file, skipping optimizer and scheduler state. Intended for cross-stage resumes (e.g. pretraining → finetuning).
- **`train.py` — `load_checkpoint()`** — New Accelerate-native checkpoint loading function. Supports both full state restoration (`accelerator.load_state()`) and model-weights-only loading via safetensors. Reads step and loss metadata from `meta.json`.
- **`train.py` — `accelerator.register_for_checkpointing(scheduler)`** — Registers the LR scheduler with Accelerate so it is correctly saved and restored by `save_state` / `load_state`.
- **`train.py` — `meta.json` in checkpoint output** — Simple JSON file (`step`, `best_loss`) written alongside `config.json` and `norm_stats.json` for each checkpoint.

### Changed

- **`train.py` — Accelerator initialisation** — Moved from module level to inside `train()`, and `mixed_precision="bf16"` enabled. The module-level `accelerator = Accelerator()` is now a commented-out placeholder; the live instance is created as `Accelerator(mixed_precision="bf16")` at the start of training.
- **`train.py` — Checkpoint save logic** — `save_checkpoint()` rewritten to use `accelerator.save_state()` instead of DeepSpeed's `model_engine.save_checkpoint()`. Metadata previously stored as `client_state` is now written to `meta.json`.
- **`train.py` — LR scheduler creation order** — Scheduler is now created and registered *before* `load_state()` is called (required by Accelerate for correct state restoration on resume).
- **`ds_config.json` — `train_micro_batch_size_per_gpu`** — Increased from `8` to `16`.
- **`ds_config.json` — ZeRO optimisation** — Downgraded from stage 2 (with allgather/reduce-scatter partitioning) to stage 0 (disabled).
- **`dataset/config.yaml` — MetaWorld dataset path** — Updated from `/home/dell/code/lintao/Evo_1/Evo1_training_dataset/Evo1_MetaWorld_Dataset` to `/home/tmprithvi/Evo1_training_dataset/Metaworld`.
- **`dataset/lerobot_dataset_pretrain_mp.py` — Default cache directory** — Updated from `/home/dell/code/lintao/Evo_1/training_data_cache/` to `/home/tmprithvi/Evo1_training_dataset/cache/`.

### Removed

- **`train.py` — SwanLab integration** — `import swanlab` commented out; `init_swanlab()` stubbed to `pass`; SwanLab logging block removed from `log_training_step()`.
- **`train.py` — `load_checkpoint_with_deepspeed()`** — Replaced by the new `load_checkpoint()` function.
- **`train.py` — DeepSpeed `model_engine.save_checkpoint()`** — Replaced by `accelerator.save_state()`.
- **`train.py` — `best_ckpt_path` variable** — No longer needed after the checkpoint rewrite.
- **`train.py` — `checkpoint.json` metadata file** — Replaced by `meta.json` with a simpler schema.
- **`ds_config.json` — ZeRO stage 2 options** — `allgather_partitions`, `allgather_bucket_size`, `reduce_scatter`, `reduce_bucket_size`, `overlap_comm`, `contiguous_gradients` all removed along with the stage downgrade.
