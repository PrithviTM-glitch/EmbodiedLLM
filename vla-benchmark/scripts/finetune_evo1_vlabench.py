#!/usr/bin/env python3
"""Fine-tune Evo-1 on VLABench via LeRobot (v2.1) dataset format.

Pipeline:
1) Generate trajectories with VLABench expert policies -> HDF5 files.
2) Convert HDF5 -> LeRobot dataset (parquet + videos + meta).
3) Generate an Evo-1 dataset config.yaml pointing at the LeRobot dataset.
4) Launch Evo-1 training (stage 1 and/or stage 2) using accelerate.

This script intentionally does NOT re-implement dataset logic; it shells out to:
- VLABench/scripts/trajectory_generation.py
- VLABench/scripts/convert_to_lerobot.py
- Evo-1/Evo_1/scripts/train.py

Prereqs:
- VLABench installed (and assets downloaded)
- Evo-1 installed (requirements + accelerate + deepspeed config)
- LeRobot installed in the environment that runs VLABench conversion

Typical usage:
python vla-benchmark/scripts/finetune_evo1_vlabench.py \
  --vlabench-root /path/to/VLABench \
  --evo1-root /path/to/Evo-1 \
  --tasks select_toy select_fruit \
  --n-samples-per-task 50 \
  --dataset-name vlabench_ft_select \
  --run collect convert stage1 stage2
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class UserError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    vlabench_root: Path
    evo1_root: Path
    hdf5_out_dir: Path
    hf_home: Path
    evo1_dataset_config_out: Path


def _echo(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _run(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        print(f"[dry-run] {(_echo(cmd))}")
        return

    print(f"[run] {(_echo(cmd))}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _default_hf_home() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface").resolve()


def _lerobot_dataset_path(hf_home: Path, dataset_name: str) -> Path:
    # VLABench converter writes to: HF_HOME/lerobot/<dataset_name>
    return hf_home / "lerobot" / dataset_name


def _require_dir(path: Path, what: str) -> None:
    if not path.exists():
        raise UserError(f"{what} not found: {path}")
    if not path.is_dir():
        raise UserError(f"{what} is not a directory: {path}")


def _require_file(path: Path, what: str) -> None:
    if not path.exists():
        raise UserError(f"{what} not found: {path}")
    if not path.is_file():
        raise UserError(f"{what} is not a file: {path}")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Evo-1 on VLABench (collect → convert → train)")

    p.add_argument("--vlabench-root", type=Path, required=True, help="Path to cloned VLABench repo")
    p.add_argument("--evo1-root", type=Path, required=True, help="Path to cloned Evo-1 repo")

    p.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="VLABench task names to collect (e.g., select_toy select_fruit)",
    )

    p.add_argument(
        "--hdf5-out-dir",
        type=Path,
        default=Path("vla-benchmark/datasets/vlabench_hdf5"),
        help="Where to store VLABench-generated HDF5 trajectories (task subdirs will be created)",
    )

    p.add_argument(
        "--n-samples-per-task",
        type=int,
        default=10,
        help="How many successful trajectories to attempt to generate per task",
    )

    p.add_argument(
        "--vlabench-robot",
        type=str,
        default="franka",
        help="Robot name passed to VLABench trajectory generation (--robot)",
    )

    p.add_argument(
        "--vlabench-eval-unseen",
        action="store_true",
        help="Pass --eval-unseen to VLABench trajectory generation",
    )

    p.add_argument(
        "--vlabench-early-stop",
        action="store_true",
        help="Pass --early-stop to VLABench trajectory generation",
    )

    p.add_argument(
        "--vlabench-python",
        type=str,
        default=sys.executable,
        help="Python executable/env for running VLABench scripts",
    )

    p.add_argument(
        "--evo1-python",
        type=str,
        default=sys.executable,
        help="Python executable/env for running Evo-1 training scripts (usually same as Evo1 conda env)",
    )

    p.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="LeRobot dataset name (VLABench converter uses this as repo_id; stored under HF_HOME/lerobot/)",
    )

    p.add_argument(
        "--max-files",
        type=int,
        default=500,
        help="Max number of hdf5 files to convert per task (VLABench convert_to_lerobot.py --max-files)",
    )

    p.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="HF_HOME to use for LeRobot output (default: env HF_HOME or ~/.cache/huggingface)",
    )

    p.add_argument(
        "--arm-group-name",
        type=str,
        default="vlabench_franka",
        help="Top-level key under data_groups in Evo-1 dataset config.yaml",
    )

    p.add_argument(
        "--evo1-dataset-config-out",
        type=Path,
        default=Path("vla-benchmark/configs/evo1/vlabench_dataset.yaml"),
        help="Where to write the Evo-1 dataset config.yaml used by scripts/train.py",
    )

    p.add_argument(
        "--run",
        nargs="+",
        default=["collect", "convert"],
        choices=["collect", "convert", "stage1", "stage2"],
        help="Which stages to run. Default runs only collect+convert.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )

    # Training knobs (defaults match Evo-1 README as closely as possible)
    p.add_argument("--num-processes", type=int, default=1, help="accelerate --num_processes")
    p.add_argument("--num-machines", type=int, default=1, help="accelerate --num_machines")
    p.add_argument("--use-deepspeed", action="store_true", help="Use Evo-1 ds_config.json with accelerate")

    p.add_argument("--vlm-name", type=str, default="OpenGVLab/InternVL3-1B")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)

    p.add_argument("--stage1-max-steps", type=int, default=5000)
    p.add_argument("--stage2-max-steps", type=int, default=80000)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--ckpt-interval", type=int, default=2500)

    p.add_argument(
        "--save-dir-stage1",
        type=Path,
        default=Path("vla-benchmark/results/evo1_vlabench_finetune/stage1"),
    )
    p.add_argument(
        "--save-dir-stage2",
        type=Path,
        default=Path("vla-benchmark/results/evo1_vlabench_finetune/stage2"),
    )

    p.add_argument(
        "--disable-wandb",
        action="store_true",
        default=True,
        help="Pass --disable_wandb to Evo-1 train.py (default: True)",
    )

    return p.parse_args(argv)


def _make_paths(args: argparse.Namespace) -> Paths:
    vlabench_root = args.vlabench_root.expanduser().resolve()
    evo1_root = args.evo1_root.expanduser().resolve()

    hf_home = args.hf_home.expanduser().resolve() if args.hf_home else _default_hf_home()

    hdf5_out_dir = args.hdf5_out_dir.expanduser().resolve()
    evo1_dataset_config_out = args.evo1_dataset_config_out.expanduser().resolve()

    return Paths(
        vlabench_root=vlabench_root,
        evo1_root=evo1_root,
        hdf5_out_dir=hdf5_out_dir,
        hf_home=hf_home,
        evo1_dataset_config_out=evo1_dataset_config_out,
    )


def _write_evo1_dataset_config(
    *,
    out_path: Path,
    arm_group_name: str,
    dataset_name: str,
    dataset_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Minimal config matching Evo-1's expected YAML schema.
    # VLABench's convert_to_lerobot uses keys: image, wrist_image.
    # Evo-1's loader expects video folders named by these view_map values.
    config_text = """# Auto-generated by finetune_evo1_vlabench.py
# Compatible with Evo-1 LeRobot loader (LeRobot v2.1 format).
max_action_dim: 24
max_state_dim: 24
max_views: 3

data_groups:
  {arm_group_name}:
    {dataset_name}:
      path: {dataset_path}
      view_map:
        image_1: observation.images.image
        image_2: observation.images.wrist_image
""".format(
        arm_group_name=arm_group_name,
        dataset_name=dataset_name,
        dataset_path=str(dataset_path),
    )

    out_path.write_text(config_text)


def _count_existing_hdf5(task_dir: Path) -> int:
    if not task_dir.exists():
        return 0
    return len(list(task_dir.glob("*.hdf5")))


def collect_trajectories(
    *,
    args: argparse.Namespace,
    paths: Paths,
    dry_run: bool,
) -> None:
    _require_dir(paths.vlabench_root, "VLABench root")
    gen_script = paths.vlabench_root / "scripts" / "trajectory_generation.py"
    _require_file(gen_script, "VLABench trajectory generation script")

    paths.hdf5_out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")

    for task in args.tasks:
        task_out_dir = paths.hdf5_out_dir / task
        task_out_dir.mkdir(parents=True, exist_ok=True)
        start_id = _count_existing_hdf5(task_out_dir)

        cmd = [
            args.vlabench_python,
            str(gen_script),
            "--task-name",
            task,
            "--save-dir",
            str(paths.hdf5_out_dir),
            "--n-sample",
            str(args.n_samples_per_task),
            "--start-id",
            str(start_id),
            "--robot",
            args.vlabench_robot,
        ]

        if args.vlabench_eval_unseen:
            cmd.append("--eval-unseen")
        if args.vlabench_early_stop:
            cmd.append("--early-stop")

        _run(cmd, cwd=paths.vlabench_root, env=env, dry_run=dry_run)


def convert_to_lerobot(
    *,
    args: argparse.Namespace,
    paths: Paths,
    dry_run: bool,
) -> Path:
    _require_dir(paths.vlabench_root, "VLABench root")
    convert_script = paths.vlabench_root / "scripts" / "convert_to_lerobot.py"
    _require_file(convert_script, "VLABench convert_to_lerobot.py")

    env = os.environ.copy()
    env["HF_HOME"] = str(paths.hf_home)

    cmd = [
        args.vlabench_python,
        str(convert_script),
        "--dataset-name",
        args.dataset_name,
        "--dataset-path",
        str(paths.hdf5_out_dir),
        "--max-files",
        str(args.max_files),
    ]

    # convert_to_lerobot supports restricting by task-list
    if args.tasks:
        cmd += ["--task-list", *args.tasks]

    _run(cmd, cwd=paths.vlabench_root, env=env, dry_run=dry_run)

    out_path = _lerobot_dataset_path(paths.hf_home, args.dataset_name)
    print(f"LeRobot dataset path (expected): {out_path}")
    return out_path


def _accelerate_cmd(
    *,
    evo1_python: str,
    evo1_evo1_dir: Path,
    num_processes: int,
    num_machines: int,
    use_deepspeed: bool,
    train_args: List[str],
) -> List[str]:
    # We run accelerate as a module via python -m accelerate to avoid PATH issues.
    cmd = [
        evo1_python,
        "-m",
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        str(num_machines),
    ]

    if use_deepspeed:
        ds_cfg = evo1_evo1_dir / "ds_config.json"
        cmd += ["--deepspeed_config_file", str(ds_cfg)]

    cmd += ["scripts/train.py", *train_args]
    return cmd


def train_stage1(
    *,
    args: argparse.Namespace,
    paths: Paths,
    dataset_config_path: Path,
    dry_run: bool,
) -> Path:
    evo1_evo1_dir = paths.evo1_root / "Evo_1"
    _require_dir(evo1_evo1_dir, "Evo-1 Evo_1 directory")

    args.save_dir_stage1.mkdir(parents=True, exist_ok=True)

    train_args = [
        "--run_name",
        f"Evo1_vlabench_{args.dataset_name}_stage1",
        "--action_head",
        "flowmatching",
        "--use_augmentation",
        "--lr",
        str(args.lr),
        "--dropout",
        str(args.dropout),
        "--weight_decay",
        str(args.weight_decay),
        "--batch_size",
        str(args.batch_size),
        "--image_size",
        str(args.image_size),
        "--max_steps",
        str(args.stage1_max_steps),
        "--log_interval",
        str(args.log_interval),
        "--ckpt_interval",
        str(args.ckpt_interval),
        "--warmup_steps",
        str(args.warmup_steps),
        "--grad_clip_norm",
        str(args.grad_clip_norm),
        "--num_layers",
        str(args.num_layers),
        "--horizon",
        str(args.horizon),
        "--finetune_action_head",
        "--vlm_name",
        args.vlm_name,
        "--dataset_config_path",
        str(dataset_config_path),
        "--per_action_dim",
        "24",
        "--state_dim",
        "24",
        "--save_dir",
        str(args.save_dir_stage1),
    ]

    if args.disable_wandb:
        train_args.append("--disable_wandb")

    cmd = _accelerate_cmd(
        evo1_python=args.evo1_python,
        evo1_evo1_dir=evo1_evo1_dir,
        num_processes=args.num_processes,
        num_machines=args.num_machines,
        use_deepspeed=args.use_deepspeed,
        train_args=train_args,
    )

    _run(cmd, cwd=evo1_evo1_dir, env=os.environ.copy(), dry_run=dry_run)

    resume_tag = f"step_{args.stage1_max_steps}"
    return args.save_dir_stage1 / resume_tag


def train_stage2(
    *,
    args: argparse.Namespace,
    paths: Paths,
    dataset_config_path: Path,
    resume_path: Path,
    dry_run: bool,
) -> None:
    evo1_evo1_dir = paths.evo1_root / "Evo_1"
    _require_dir(evo1_evo1_dir, "Evo-1 Evo_1 directory")

    args.save_dir_stage2.mkdir(parents=True, exist_ok=True)

    train_args = [
        "--run_name",
        f"Evo1_vlabench_{args.dataset_name}_stage2",
        "--action_head",
        "flowmatching",
        "--use_augmentation",
        "--lr",
        str(args.lr),
        "--dropout",
        str(args.dropout),
        "--weight_decay",
        str(args.weight_decay),
        "--batch_size",
        str(args.batch_size),
        "--image_size",
        str(args.image_size),
        "--max_steps",
        str(args.stage2_max_steps),
        "--log_interval",
        str(args.log_interval),
        "--ckpt_interval",
        str(args.ckpt_interval),
        "--warmup_steps",
        str(args.warmup_steps),
        "--grad_clip_norm",
        str(args.grad_clip_norm),
        "--num_layers",
        str(args.num_layers),
        "--horizon",
        str(args.horizon),
        "--finetune_vlm",
        "--finetune_action_head",
        "--vlm_name",
        args.vlm_name,
        "--dataset_config_path",
        str(dataset_config_path),
        "--per_action_dim",
        "24",
        "--state_dim",
        "24",
        "--save_dir",
        str(args.save_dir_stage2),
        "--resume",
        "--resume_pretrain",
        "--resume_path",
        str(resume_path),
    ]

    if args.disable_wandb:
        train_args.append("--disable_wandb")

    cmd = _accelerate_cmd(
        evo1_python=args.evo1_python,
        evo1_evo1_dir=evo1_evo1_dir,
        num_processes=args.num_processes,
        num_machines=args.num_machines,
        use_deepspeed=args.use_deepspeed,
        train_args=train_args,
    )

    _run(cmd, cwd=evo1_evo1_dir, env=os.environ.copy(), dry_run=dry_run)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = _parse_args(argv)
        paths = _make_paths(args)

        _require_dir(paths.vlabench_root, "VLABench root")
        _require_dir(paths.evo1_root, "Evo-1 root")

        stages = set(args.run)
        dry_run = bool(args.dry_run)

        if "collect" in stages:
            collect_trajectories(args=args, paths=paths, dry_run=dry_run)

        lerobot_path = _lerobot_dataset_path(paths.hf_home, args.dataset_name)
        if "convert" in stages:
            lerobot_path = convert_to_lerobot(args=args, paths=paths, dry_run=dry_run)

        # Write Evo-1 dataset config.yaml (always; it's small and deterministic)
        _write_evo1_dataset_config(
            out_path=paths.evo1_dataset_config_out,
            arm_group_name=args.arm_group_name,
            dataset_name=args.dataset_name,
            dataset_path=lerobot_path,
        )
        print(f"Wrote Evo-1 dataset config: {paths.evo1_dataset_config_out}")

        resume_path = None
        if "stage1" in stages:
            resume_path = train_stage1(
                args=args,
                paths=paths,
                dataset_config_path=paths.evo1_dataset_config_out,
                dry_run=dry_run,
            )
            print(f"Stage1 resume path (expected): {resume_path}")

        if "stage2" in stages:
            if resume_path is None:
                # Default assumption if stage1 was run previously
                resume_path = args.save_dir_stage1 / f"step_{args.stage1_max_steps}"
            train_stage2(
                args=args,
                paths=paths,
                dataset_config_path=paths.evo1_dataset_config_out,
                resume_path=resume_path,
                dry_run=dry_run,
            )

        return 0

    except UserError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"command failed: {e}", file=sys.stderr)
        return e.returncode


if __name__ == "__main__":
    raise SystemExit(main())
