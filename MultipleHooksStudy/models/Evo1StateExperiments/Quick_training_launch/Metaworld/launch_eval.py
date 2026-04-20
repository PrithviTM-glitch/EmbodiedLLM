"""
launch_eval.py — Pull a MetaWorld checkpoint from GCS and run the MT50 evaluation.

Steps:
  1. Pull step_XXXXX checkpoint from GCS (skip with --no-gcs if already local)
  2. Start eval_server.py as a background subprocess
  3. Wait until ws://127.0.0.1:{port} is connectable (model loaded)
  4. Run mt50_eval_client.py in the foreground
  5. Terminate the server when the client finishes

Usage:
    python launch_eval.py --exp exp1 --step 80000
    python launch_eval.py --exp exp2A --step 62500 --episodes 5 --target-level easy
    python launch_eval.py --ckpt-dir /local/path/to/step_80000  # skip GCS pull
"""

import argparse
import os
import socket
import subprocess
import sys
import time

# ── Paths (relative to this script) ──────────────────────────────────────────
_REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SERVER_SCRIPT = os.path.join(_REPO_ROOT, "Quick_training_launch", "server", "eval_server.py")
_CLIENT_SCRIPT = os.path.join(_REPO_ROOT, "MetaWorld_evaluation", "mt50_eval_client.py")
_CLIENT_CWD    = os.path.join(_REPO_ROOT, "MetaWorld_evaluation")

# ── GCS layout ────────────────────────────────────────────────────────────────
# Checkpoints are synced by train.py to:
#   gs://model-checkpointing/all_check_freq_saves/{exp}/stage2/step_{N}/
_GCS_EVAL_BASE    = "gs://model-checkpointing/all_check_freq_saves"
_DEFAULT_LOCAL_CACHE = "/home/tmprithvi/tmp/eval_cache"

EXP_KEYS = ["exp1", "exp2A", "exp2B", "exp2C"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def pull_from_gcs(gcs_path: str, local_base: str) -> str:
    suffix     = gcs_path.replace("gs://model-checkpointing/", "").replace("/", "_")
    local_path = os.path.join(local_base, suffix)
    os.makedirs(local_path, exist_ok=True)
    print(f"  [GCS] {gcs_path}")
    print(f"        → {local_path}")
    subprocess.run(["gsutil", "-m", "rsync", "-r", gcs_path, local_path], check=True)
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
    p = argparse.ArgumentParser(description="Launch Evo-1 MetaWorld evaluation")

    # Checkpoint source — either GCS pull or a local path
    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--ckpt-dir", metavar="PATH",
                      help="Local step_XXXXX directory (skips GCS pull)")
    ckpt.add_argument("--step",     type=int,
                      help="Checkpoint step to pull from GCS (requires --exp)")

    p.add_argument("--exp",     choices=EXP_KEYS, default=None,
                   help="Experiment key (required when using --step)")
    p.add_argument("--stage",   default="stage2",
                   help="Training stage subfolder in GCS (default: stage2)")

    # Server options
    p.add_argument("--port",      type=int, default=9000)
    p.add_argument("--device",    default="cuda")
    p.add_argument("--timesteps", type=int, default=32,
                   help="Flow-matching inference timesteps")

    # Client options (passed through)
    p.add_argument("--episodes",     type=int, default=10)
    p.add_argument("--horizon",      type=int, default=400,
                   help="Max env steps per episode")
    p.add_argument("--seed",         type=int, default=4042)
    p.add_argument("--target-level", default="all",
                   choices=["all","easy","medium","hard","very_hard"])
    p.add_argument("--history-len",  type=int, default=6)
    p.add_argument("--state-take",   type=int, default=8)
    p.add_argument("--inference-horizon", type=int, default=15)
    p.add_argument("--no-video",  action="store_true")
    p.add_argument("--no-window", action="store_true")

    # GCS / cache options
    p.add_argument("--local-cache", default=_DEFAULT_LOCAL_CACHE,
                   help="Local directory for GCS checkpoint cache")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1) Resolve local checkpoint directory
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
    print(f"  Episodes   : {args.episodes}  Horizon: {args.horizon}")
    print(f"  Target     : {args.target_level}")
    print(f"  History    : {args.history_len}  State-take: {args.state_take}")
    print(f"{'='*60}")

    # 2) Start server
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    server_log = f"/tmp/eval_server_{os.path.basename(ckpt_dir)}.log"
    print(f"\n[server] Starting — log → {server_log}")

    with open(server_log, "w") as logf:
        server_proc = subprocess.Popen(
            [sys.executable, _SERVER_SCRIPT,
             "--ckpt_dir",  ckpt_dir,
             "--port",      str(args.port),
             "--device",    args.device,
             "--timesteps", str(args.timesteps)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ},
        )
    print(f"[server] PID {server_proc.pid}")

    # 3) Wait for server to be ready
    if not wait_for_server(port=args.port, timeout=300):
        print(f"[ERROR] Server did not start in time. Check log: {server_log}")
        server_proc.terminate()
        sys.exit(1)

    # 4) Run client
    client_cmd = [
        sys.executable, _CLIENT_SCRIPT,
        "--server-url",        f"ws://127.0.0.1:{args.port}",
        "--episodes",          str(args.episodes),
        "--horizon",           str(args.horizon),
        "--seed",              str(args.seed),
        "--target-level",      args.target_level,
        "--history-len",       str(args.history_len),
        "--state-take",        str(args.state_take),
        "--inference-horizon", str(args.inference_horizon),
    ]
    if args.no_video:  client_cmd.append("--no-video")
    if args.no_window: client_cmd.append("--no-window")

    print(f"\n[client] Starting MT50 eval...")
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
