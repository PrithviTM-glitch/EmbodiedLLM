"""
launch_stage1.py — Interactive Stage 1 launcher for Evo-1 MetaWorld experiments.

Usage:
    python launch_stage1.py

The script asks which experiments to run, then launches them as parallel
subprocesses. Stage 1 always starts fresh (no resume) — action-head only,
Phase 0 pretraining active.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRESET VALUES — edit these before running
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess
import sys
import os
import time

# ── Environment detection ───────────────────────────────────────────────
BASE_DIR = "/content" if os.path.exists("/content") else os.path.expanduser("~")

# ── Shared training hyperparameters ────────────────────────────────────
SHARED = {
    "action_head":       "flowmatching",
    "lr":                "1e-5",
    "dropout":           "0.2",
    "weight_decay":      "1e-3",
    "batch_size":        "16",
    "image_size":        "448",
    "max_steps":         "5000",
    "warmup_steps":      "1000",
    "log_interval":      "10",
    "ckpt_interval":     "2500",
    "eval_interval":     "500",
    "grad_clip_norm":    "1.0",
    "num_layers":        "8",
    "horizon":           "50",
    "per_action_dim":    "24",
    "state_dim":         "24",
    "history_len":       "5",
    "pretrain_steps":    "1000",
    "pretrain_lr":       "0.0001",
    "lambda_orth":       "0.1",
    "trace_decay":       "0.9",
    "vlm_name":          "OpenGVLab/InternVL3-1B",
    "wandb_project":     "evo1_metaworld",
    "dataset_config":    f"{BASE_DIR}/Evo_1/Evo_1/dataset/metaworld_config.yaml",
    "cache_dir":         f"{BASE_DIR}/Evo1_training_dataset/cache/metaworld",
}

# ── Accelerate launcher flags ───────────────────────────────────────────
ACCELERATE = [
    "accelerate", "launch",
    "--num_processes", "1",
    "--num_machines", "1",
    "--mixed_precision", "bf16",
    "--dynamo_backend", "no",
    "--deepspeed_config_file", "ds_config.json",
]

# GCS base prefix — basename of save_dir is appended automatically by the sync code
# Result: gs://model-checkpointing/all_check_7500steps/all_check_7500steps/expX/stage1/
_GCS_BASE = "gs://model-checkpointing/all_check_7500steps/all_check_7500steps"

# ── Per-experiment config ───────────────────────────────────────────────
# save_dir   : local path on GCP instance (fast SSD under /tmp)
# gcs_bucket : GCS prefix to rsync save_dir into (train.py appends basename of save_dir)
# features   : list of feature names
# embedding_strain
EXPERIMENTS = {
    "exp1": {
        "run_name":         "Evo1_metaworld_exp1_stage1",
        "features":         ["position"],
        "embedding_strain": "none",
        "save_dir":         "/tmp/exp1/stage1",
        "gcs_bucket":       f"{_GCS_BASE}/exp1",
    },
    "exp2A": {
        "run_name":         "Evo1_metaworld_exp2A_stage1",
        "features":         ["position", "velocity", "acceleration", "trace", "deviation"],
        "embedding_strain": "A",
        "save_dir":         "/tmp/exp2A/stage1",
        "gcs_bucket":       f"{_GCS_BASE}/exp2A",
    },
    "exp2B": {
        "run_name":         "Evo1_metaworld_exp2B_stage1",
        "features":         ["position", "velocity", "acceleration", "trace", "deviation"],
        "embedding_strain": "B",
        "save_dir":         "/tmp/exp2B/stage1",
        "gcs_bucket":       f"{_GCS_BASE}/exp2B",
    },
    "exp2C": {
        "run_name":         "Evo1_metaworld_exp2C_stage1",
        "features":         ["position", "velocity", "acceleration", "trace", "deviation"],
        "embedding_strain": "C",
        "save_dir":         "/tmp/exp2C/stage1",
        "gcs_bucket":       f"{_GCS_BASE}/exp2C",
    },
}

# ───────────────────────────────────────────────────────────────────────

def ask(question, options):
    """Ask a choice question, return the chosen string from options."""
    print(f"\n{question}")
    for i, opt in enumerate(options):
        print(f"  [{i+1}] {opt}")
    while True:
        ans = input("  -> ").strip()
        if ans in [str(i+1) for i in range(len(options))]:
            return options[int(ans) - 1]
        if len(options) == 2:
            if ans.lower() in ("y", "yes"):
                return options[0]
            if ans.lower() in ("n", "no"):
                return options[1]
        print(f"  Please enter a number between 1 and {len(options)}.")


def build_command(exp_key):
    """Build the accelerate launch command list for a given experiment."""
    exp = EXPERIMENTS[exp_key]
    sh = SHARED

    return ACCELERATE + [
        "scripts/train.py",
        "--run_name",            exp["run_name"],
        "--wandb_project",       sh["wandb_project"],
        "--action_head",         sh["action_head"],
        "--use_augmentation",
        "--lr",                  sh["lr"],
        "--dropout",             sh["dropout"],
        "--weight_decay",        sh["weight_decay"],
        "--batch_size",          sh["batch_size"],
        "--image_size",          sh["image_size"],
        "--max_steps",           sh["max_steps"],
        "--warmup_steps",        sh["warmup_steps"],
        "--log_interval",        sh["log_interval"],
        "--ckpt_interval",       sh["ckpt_interval"],
        "--eval_interval",       sh["eval_interval"],
        "--grad_clip_norm",      sh["grad_clip_norm"],
        "--num_layers",          sh["num_layers"],
        "--horizon",             sh["horizon"],
        "--per_action_dim",      sh["per_action_dim"],
        "--state_dim",           sh["state_dim"],
        "--finetune_action_head",
        "--vlm_name",            sh["vlm_name"],
        "--dataset_config_path", sh["dataset_config"],
        "--cache_dir",           sh["cache_dir"],
        "--save_dir",            exp["save_dir"],
        "--gcs_bucket",          exp["gcs_bucket"],
        "--history_len",         sh["history_len"],
        "--features",            *exp["features"],
        "--embedding_strain",    exp["embedding_strain"],
        "--pretrain_steps",      sh["pretrain_steps"],
        "--pretrain_lr",         sh["pretrain_lr"],
        "--lambda_orth",         sh["lambda_orth"],
        "--trace_decay",         sh["trace_decay"],
        "--disable_swanlab",
    ]


def main():
    print("=" * 60)
    print("  Evo-1 MetaWorld — Stage 1 Launcher")
    print(f"  BASE_DIR = {BASE_DIR}")
    print("=" * 60)

    os.chdir(f"{BASE_DIR}/Evo_1/Evo_1")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    selected = []
    for exp_key in ["exp1", "exp2A", "exp2B", "exp2C"]:
        run = ask(f"Run {exp_key}?", ["Yes", "No"])
        if run == "Yes":
            selected.append(exp_key)

    if not selected:
        print("\nNo experiments selected. Exiting.")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("  Launching the following experiments in parallel:")
    for exp_key in selected:
        exp = EXPERIMENTS[exp_key]
        print(f"  * {exp_key}  [fresh start]")
        print(f"    save_dir   : {exp['save_dir']}")
        print(f"    gcs_bucket : {exp['gcs_bucket']}")
    print("=" * 60)

    confirm = ask("\nProceed?", ["Yes", "No"])
    if confirm == "No":
        print("Aborted.")
        sys.exit(0)

    # Launch all selected experiments as parallel subprocesses
    processes = {}
    for exp_key in selected:
        cmd = build_command(exp_key)
        log_path = f"/tmp/{exp_key}_stage1_launch.log"
        print(f"\n[{exp_key}] Launching... log -> {log_path}")
        with open(log_path, "w") as logf:
            p = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env={**os.environ}
            )
        processes[exp_key] = (p, log_path)
        print(f"[{exp_key}] PID {p.pid}")

    print("\n" + "=" * 60)
    print("  All processes launched.")
    print("  Monitor logs with:  tail -f /tmp/expX_stage1_launch.log")
    print("=" * 60)

    # Poll all processes until each finishes, reporting completions as they occur
    remaining = dict(processes)
    while remaining:
        for exp_key, (p, log_path) in list(remaining.items()):
            ret = p.poll()
            if ret is not None:
                status = "completed" if ret == 0 else f"FAILED (exit {ret})"
                print(f"[{exp_key}] {status} — {log_path}")
                del remaining[exp_key]
        if remaining:
            time.sleep(30)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
