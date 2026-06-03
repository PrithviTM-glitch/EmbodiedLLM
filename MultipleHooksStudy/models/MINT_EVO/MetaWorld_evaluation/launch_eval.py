"""
launch_eval.py — Pull a MINT_EVO baseline checkpoint from GCS and run MT50 evaluation.

Steps:
  1. Pull step_XXXXX checkpoint from GCS (skip with --ckpt-dir if already local)
  2. Start Mod_server.py as a background subprocess
  3. Wait until ws://127.0.0.1:{port} is connectable (model loaded)
  4. Run mt50_eval_client.py in the foreground
  5. Terminate the server when the client finishes

Usage:
    python launch_eval.py --step 80000
    python launch_eval.py --step best --stage stage1
    python launch_eval.py --ckpt-dir /local/path/to/step_80000
"""

import argparse
import os
import socket
import subprocess
import sys
import time

# ── Paths (relative to this script) ──────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
_MINT_EVO_ROOT  = os.path.abspath(os.path.join(_HERE, ".."))

_SERVER_SCRIPT  = os.path.join(_MINT_EVO_ROOT, "Evo_1", "scripts", "Mod_server.py")
_CLIENT_SCRIPT  = os.path.join(_HERE, "mt50_eval_client.py")
_CLIENT_CWD     = _HERE

# ── GCS layout ────────────────────────────────────────────────────────────────
# gs://model-checkpointing/baseline/baseline/{stage}/step_{N}/
_GCS_BASE            = "gs://model-checkpointing/baseline/baseline"
_DEFAULT_LOCAL_CACHE = os.path.expanduser("~/tmp/eval_cache")


# ── Helpers ───────────────────────────────────────────────────────────────────
def pull_from_gcs(gcs_path: str, local_base: str) -> str:
    suffix     = gcs_path.replace("gs://", "").replace("/", "_")
    local_path = os.path.join(local_base, suffix)
    os.makedirs(local_path, exist_ok=True)
    print(f"  [GCS] {gcs_path}")
    print(f"        → {local_path}")
    subprocess.run(["gcloud", "storage", "rsync", "--recursive", gcs_path, local_path],
                   check=True)
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
    p = argparse.ArgumentParser(description="Launch MINT_EVO baseline MetaWorld evaluation")

    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument("--ckpt-dir", metavar="PATH",
                      help="Local step_XXXXX directory (skips GCS pull)")
    ckpt.add_argument("--step",
                      help="Checkpoint tag to pull from GCS, e.g. 80000 or best or final")

    p.add_argument("--stage",   default="stage2",
                   help="Training stage subfolder in GCS (default: stage2)")

    # Server options
    p.add_argument("--port",      type=int, default=9000)
    p.add_argument("--timesteps", type=int, default=32,
                   help="Flow-matching inference timesteps")

    # Client options (passed through to mt50_eval_client.py)
    p.add_argument("--episodes",          type=int, default=10)
    p.add_argument("--horizon",           type=int, default=400)
    p.add_argument("--seed",              type=int, default=4042)
    p.add_argument("--target-level",      default="all",
                   choices=["all", "easy", "medium", "hard", "very_hard"])
    p.add_argument("--history-len",       type=int, default=1,
                   help="State history length (default 1 = current frame only, matches baseline training)")
    p.add_argument("--state-take",        type=int, default=8,
                   help="Number of obs dims to use per frame (default 8, matches baseline training)")
    p.add_argument("--inference-horizon", type=int, default=15)
    p.add_argument("--no-video",          action="store_true")
    p.add_argument("--no-window",         action="store_true")
    p.add_argument("--log-dir",           default=None)

    # GCS cache
    p.add_argument("--local-cache", default=_DEFAULT_LOCAL_CACHE)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1) Resolve checkpoint directory
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
        print(f"[eval] Using local checkpoint: {ckpt_dir}")
    else:
        gcs_path = f"{_GCS_BASE}/{args.stage}/step_{args.step}"
        print(f"\n{'='*60}\n  Pulling checkpoint from GCS...\n{'='*60}")
        ckpt_dir = pull_from_gcs(gcs_path, args.local_cache)

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {ckpt_dir}")
    print(f"  Server     : {_SERVER_SCRIPT}")
    print(f"  Client     : {_CLIENT_SCRIPT}")
    print(f"  Port       : {args.port}")
    print(f"  Episodes   : {args.episodes}  Horizon: {args.horizon}")
    print(f"  Target     : {args.target_level}")
    print(f"  History    : {args.history_len}  State-take: {args.state_take}")
    print(f"{'='*60}")

    # 2) Start server
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    server_log = f"/tmp/mod_server_{os.path.basename(ckpt_dir.rstrip('/'))}.log"
    print(f"\n[server] Starting — log → {server_log}")

    server_cmd = [
        sys.executable, _SERVER_SCRIPT,
        "--ckpt_dir",   ckpt_dir,
        "--port",       str(args.port),
        "--timesteps",  str(args.timesteps),
    ]
    with open(server_log, "w") as logf:
        server_proc = subprocess.Popen(
            server_cmd,
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
    if args.no_video:   client_cmd.append("--no-video")
    if args.no_window:  client_cmd.append("--no-window")
    if args.log_dir:    client_cmd += ["--log-dir", args.log_dir]

    print(f"\n[client] Starting MT50 eval...")
    try:
        subprocess.run(client_cmd, cwd=_CLIENT_CWD, env={**os.environ}, check=True)
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
