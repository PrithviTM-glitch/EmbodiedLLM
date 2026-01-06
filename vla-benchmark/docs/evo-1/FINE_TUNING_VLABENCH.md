# Fine-tuning Evo-1 on VLABench

This repo includes a wrapper script that stitches together the *existing* upstream pipelines:

- VLABench trajectory generation → HDF5
- VLABench HDF5 → LeRobot v2.1 dataset
- Evo-1 LeRobot loader + `scripts/train.py` fine-tuning

The wrapper is: `vla-benchmark/scripts/finetune_evo1_vlabench.py`

## Prerequisites

You need local clones (these are gitignored in this repo by default):

- Evo-1: https://github.com/MINT-SJTU/Evo-1
- VLABench: https://github.com/OpenMOSS/VLABench

And you need VLABench assets installed:

- `python scripts/download_assets.py`

Also note:

- Evo-1 training expects **LeRobot v2.1** datasets.
- VLABench already provides `scripts/convert_to_lerobot.py` which produces a compatible dataset layout.

## Recommended folder layout

- `vla-benchmark/models/Evo-1/` → clone Evo-1 here
- `vla-benchmark/benchmark/VLABench/` → clone VLABench here

(They are ignored by `.gitignore` in this repo.)

## End-to-end command

From the repo root:

```bash
python vla-benchmark/scripts/finetune_evo1_vlabench.py \
  --vlabench-root vla-benchmark/benchmark/VLABench \
  --evo1-root vla-benchmark/models/Evo-1 \
  --tasks select_toy select_fruit select_drink \
  --n-samples-per-task 50 \
  --dataset-name vlabench_3task_ft \
  --run collect convert stage1 stage2 \
  --use-deepspeed
```

What it does:

1) Runs VLABench expert rollout generation into `vla-benchmark/datasets/vlabench_hdf5/<task>/*.hdf5`
2) Converts those HDF5 files into a LeRobot dataset at `${HF_HOME}/lerobot/<dataset-name>`
3) Writes an Evo-1 dataset config YAML at `vla-benchmark/configs/evo1/vlabench_dataset.yaml`
4) Launches Evo-1 stage1 then stage2 training with `accelerate`

## Notes / common adjustments

- If you use conda envs, pass interpreters explicitly:
  - `--vlabench-python $(which python)` inside your `vlabench` env
  - `--evo1-python $(which python)` inside your `Evo1` env

- Set `HF_HOME` if you want LeRobot datasets written somewhere else:
  - `--hf-home /path/to/hf_cache`

- If stage1 ran with a different step count, stage2 resume path must match:
  - by default the wrapper assumes `save_dir_stage1/step_<stage1-max-steps>`

- VLABench generation can be slow. Start with small values:
  - `--n-samples-per-task 1` and a few tasks

## Outputs

- HDF5 rollouts: `vla-benchmark/datasets/vlabench_hdf5/`
- LeRobot dataset: `${HF_HOME}/lerobot/<dataset-name>/`
- Evo-1 dataset config yaml: `vla-benchmark/configs/evo1/vlabench_dataset.yaml`
- Evo-1 checkpoints:
  - `vla-benchmark/results/evo1_vlabench_finetune/stage1/`
  - `vla-benchmark/results/evo1_vlabench_finetune/stage2/`
