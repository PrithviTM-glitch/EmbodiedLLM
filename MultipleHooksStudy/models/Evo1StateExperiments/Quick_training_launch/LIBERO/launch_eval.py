"""
launch_eval.py — Pull a LIBERO checkpoint from GCS and run evaluation.

Steps:
  1. Pull step_XXXXX checkpoint from GCS (skip with --ckpt-dir if already local)
  2. Start eval_server.py as a background subprocess
  3. Wait until ws://127.0.0.1:{port} is connectable (model loaded)
  4. Run libero_eval_client.py in the foreground
  5. Terminate the server when the client finishes

Usage:
    python launch_eval.py --exp exp1 --step best
    python launch_eval.py --exp exp2A --step 80000 --suites libero_spatial libero_goal
    python launch_eval.py --ckpt-dir /local/path/to/step_80000  # skip GCS pull
"""

import argparse
import os
import socket
import subprocess
import sys
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SERVER_SCRIPT = os.path.join(_REPO_ROOT, "Quick_training_launch", "server", "eval_server.py")
_CLIENT_SCRIPT = os.path.join(_REPO_ROOT, "LIBERO_evaluation", "libero_eval_client.py")
_CLIENT_CWD    = os.path.join(_REPO_ROOT, "LIBERO_evaluation")

# ── GCS layout ────────────────────────────────────────────────────────────────
_GCS_EVAL_BASE       = "gs://model-checkpointing/libero/all_check_freq_saves"
_DEFAULT_LOCAL_CACHE = "/home/tmprithvi/tmp/libero_eval_cache"

EXP_KEYS = ["exp1", "exp2A", "exp2B", "exp2C", "exp_k0"]
ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def pull_from_gcs(gcs_path: str, local_base: str) -> str:
    suffix     = gcs_path.replace("gs://model-checkpointing/", "").replace("/", "_")
    local_path = os.path.join(local_base, suffix)
    os.makedirs(local_path, exist_ok=True)
    print(f"  [GCS] {gcs_path}")
    print(f"        → {local_path}")
    subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", gcs_path, local_path],
        check=True,
    )
    return local_path


def wait_for_server(host: str = "127.0.0.1", port: int = 9000, timeout: int = 300) -> bool:
    print(f"  Waiting for server on {host}:{port}  (timeout={timeout}s) ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("  Server is ready.")
                return True
        except OSError:
            time.sleep(5)
    return False


# ── Arg parsing ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Launch Evo-1 LIBERO evaluation")

    # Checkpoint source
    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--ckpt-dir", metavar="PATH",
                      help="Local step_XXXXX directory (skips GCS pull)")
    ckpt.add_argument("--step",
                      help="Checkpoint tag to pull from GCS, e.g. 80000 or best or final (requires --exp)")

    p.add_argument("--exp",   choices=EXP_KEYS, default=None,
                   help="Experiment key (required when using --step)")
    p.add_argument("--stage", default="stage2",
                   help="Training stage subfolder in GCS (default: stage2)")

    # Server options
    p.add_argument("--port",         type=int, default=9000)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--timesteps",    type=int, default=32)
    p.add_argument("--evo1-root",    default=None)
    p.add_argument("--ablate-state", action="store_true",
                   help="Zero state encoder output (ablation run)")

    # Client options
    p.add_argument("--episodes",          type=int, default=10)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--history-len",       type=int, default=6)
    p.add_argument("--inference-horizon", type=int, default=14)
    p.add_argument("--suites",            nargs="+", default=ALL_SUITES,
                   choices=ALL_SUITES)
    p.add_argument("--max-steps",         type=int, default=None,
                   help="Override max_steps for all suites")
    p.add_argument("--no-video",          action="store_true")
    p.add_argument("--log-dir",           default=None,
                   help="Directory for eval result logs (default: LIBERO_evaluation/log_file/)")

    # Colab / conda options
    p.add_argument("--colab",      action="store_true",
                   help="Run the LIBERO client inside the 'libero' conda env "
                        "(required on Colab where LIBERO needs Python 3.8)")
    p.add_argument("--conda-env",  default="libero",
                   help="Conda env name to use with --colab (default: libero)")
    p.add_argument("--conda-dir",  default="/root/miniconda3",
                   help="Miniconda installation directory (default: /root/miniconda3)")

    # Cache
    p.add_argument("--local-cache", default=_DEFAULT_LOCAL_CACHE)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1) Resolve checkpoint
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
        print(f"[eval] Using local checkpoint: {ckpt_dir}")
    else:
        if not args.exp:
            print("[ERROR] --exp is required when using --step")
            sys.exit(1)
        gcs_path = f"{_GCS_EVAL_BASE}/{args.exp}/{args.stage}/step_{args.step}"
        print(f"\n{'='*60}\n  Pulling checkpoint from GCS...\n{'='*60}")
        ckpt_dir = pull_from_gcs(gcs_path, args.local_cache)

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {ckpt_dir}")
    print(f"  Port       : {args.port}")
    print(f"  Episodes   : {args.episodes}  History: {args.history_len}")
    print(f"  Suites     : {args.suites}")
    if args.ablate_state:
        print("  *** ABLATION MODE: state encoder zeroed ***")
    print(f"{'='*60}")

    # 2) Start server
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MUJOCO_GL", "osmesa")

    server_log = f"/tmp/libero_eval_server_{os.path.basename(ckpt_dir)}.log"
    print(f"\n[server] Starting — log → {server_log}")

    server_cmd = [sys.executable, _SERVER_SCRIPT,
                  "--ckpt_dir",  ckpt_dir,
                  "--port",      str(args.port),
                  "--device",    args.device,
                  "--timesteps", str(args.timesteps)]
    if args.evo1_root:
        server_cmd += ["--evo1-root", args.evo1_root]
    if args.ablate_state:
        server_cmd.append("--ablate-state")

    with open(server_log, "w") as logf:
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ},
        )
    print(f"[server] PID {server_proc.pid}")

    # 3) Wait for server
    if not wait_for_server(port=args.port, timeout=300):
        print(f"[ERROR] Server did not start in time. Check log: {server_log}")
        server_proc.terminate()
        sys.exit(1)

    # 4) Run client
    # --colab: run inside the libero conda env (Python 3.8 + LIBERO package)
    # uses `conda run` which works without activating the env in the shell
    if args.colab:
        conda_bin = os.path.join(args.conda_dir, "bin", "conda")
        if not os.path.exists(conda_bin):
            print(f"[ERROR] conda not found at {conda_bin}. "
                  f"Run LIBERO_evaluation/setup_libero_env.sh first.")
            server_proc.terminate()
            sys.exit(1)
        client_python = [conda_bin, "run", "--no-capture-output",
                         "-n", args.conda_env, "python"]
    else:
        client_python = [sys.executable]

    client_cmd = [
        *client_python, _CLIENT_SCRIPT,
        "--server-url",        f"ws://127.0.0.1:{args.port}",
        "--episodes",          str(args.episodes),
        "--seed",              str(args.seed),
        "--history-len",       str(args.history_len),
        "--inference-horizon", str(args.inference_horizon),
        "--suites",            *args.suites,
    ]
    if args.max_steps:
        client_cmd += ["--max-steps", str(args.max_steps)]
    if args.no_video:
        client_cmd.append("--no-video")
    if args.log_dir:
        client_cmd += ["--log-dir", args.log_dir]

    print(f"\n[client] Starting LIBERO eval...")
    try:
        subprocess.run(client_cmd, cwd=_CLIENT_CWD, env={**os.environ})
    finally:
        print("\n[server] Stopping...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("[server] Done.")


if __name__ == "__main__":
    main()
