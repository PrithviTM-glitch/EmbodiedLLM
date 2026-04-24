"""
libero_eval_client.py — LIBERO evaluation client for Evo-1 with state history.

Evaluates all 4 suites (libero_spatial, libero_object, libero_goal, libero_10)
sequentially. Maintains a rolling state buffer so each inference call receives
the full [history_len, state_dim] context the temporal encoder expects.

Usage:
    python libero_eval_client.py [options]

Key options (all have defaults):
    --server-url      ws://127.0.0.1:9000
    --episodes        10
    --seed            42
    --history-len     6     (k+1 for k=5, must match training config)
    --inference-horizon 14  (action chunk size from model)
    --suites          libero_spatial libero_object libero_goal libero_10
    --no-video
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")  # must be set before mujoco import

import argparse
import asyncio
import datetime
import json
import logging
import math
import pathlib
import random
from collections import deque
from typing import Dict, List, Optional

import imageio
import numpy as np
import websockets

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


# ── Suite config ───────────────────────────────────────────────────────────────
# Default max_steps per suite (total env steps across the action chunk loop)
SUITE_MAX_STEPS = {
    "libero_spatial": 250,
    "libero_object":  250,
    "libero_goal":    250,
    "libero_10":      600,
}
ALL_SUITES = list(SUITE_MAX_STEPS.keys())

LIBERO_DUMMY_ACTION = [0.0] * 7


# ── Arg parsing ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LIBERO Evo-1 eval client (with state history)")
    p.add_argument("--server-url",        default="ws://127.0.0.1:9000")
    p.add_argument("--episodes",          type=int, default=10)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--history-len",       type=int, default=6,
                   help="State history length k+1 (default 6 = k+1 for k=5, must match training config)")
    p.add_argument("--inference-horizon", type=int, default=14,
                   help="Action chunk size returned by model")
    p.add_argument("--resolution",        type=int, default=448)
    p.add_argument("--suites",            nargs="+", default=ALL_SUITES,
                   choices=ALL_SUITES, help="Which suites to evaluate")
    p.add_argument("--max-steps",         type=int, default=None,
                   help="Override max_steps for all suites (default: per-suite values)")
    p.add_argument("--log-dir",           default="log_file")
    p.add_argument("--video-dir",         default="video_log_file")
    p.add_argument("--no-video",          action="store_true")
    return p.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}_{ts}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)
    log.info(f"Log file: {log_path}")
    return log


# ── State utils ───────────────────────────────────────────────────────────────
def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.clip(quat, -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_state_vec(obs: dict) -> List[float]:
    """Extract 8-dim state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)."""
    return np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).tolist()


# ── Image utils ───────────────────────────────────────────────────────────────
def encode_image(img: np.ndarray) -> list:
    return np.ascontiguousarray(img[::-1, ::-1]).astype(np.uint8).tolist()


def obs_to_images(obs: dict) -> list:
    """Returns [agentview_flipped, wrist_flipped, dummy] as uint8 lists."""
    dummy = np.zeros((448, 448, 3), dtype=np.uint8)
    return [
        encode_image(obs["agentview_image"]),
        encode_image(obs["robot0_eye_in_hand_image"]),
        dummy.tolist(),
    ]


# ── Video saving ──────────────────────────────────────────────────────────────
def save_video(frames: list, path: str, fps: int = 30):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if frames:
        imageio.mimsave(path, frames, fps=fps)


# ── Environment setup ─────────────────────────────────────────────────────────
def get_libero_env(task, resolution: int = 448, seed: int = 42):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, str(task.language)


# ── WebSocket inference ───────────────────────────────────────────────────────
async def evo1_infer(
    ws,
    obs: dict,
    state_history: List[List[float]],  # [history_len, state_dim], oldest first
    prompt: str,
) -> np.ndarray:
    payload = {
        "image":       obs_to_images(obs),
        "state":       state_history,
        "prompt":      prompt,
        "image_mask":  [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }
    await ws.send(json.dumps(payload))
    return np.array(json.loads(await ws.recv()), dtype=np.float32)


# ── Gripper binarization ──────────────────────────────────────────────────────
def binarize_gripper(action: List[float]) -> List[float]:
    action = list(action)
    action[6] = -1.0 if action[6] > 0.5 else 1.0
    return action


# ── Single suite evaluation ───────────────────────────────────────────────────
async def eval_suite(ws, suite_name: str, args, log: logging.Logger) -> Dict:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = args.max_steps or SUITE_MAX_STEPS[suite_name]

    log.info(f"\n{'='*60}")
    log.info(f"  Suite: {suite_name}  ({num_tasks} tasks, max_steps={max_steps})")
    log.info(f"{'='*60}")

    suite_success = 0
    suite_episodes = 0

    per_task_results: Dict[int, Dict] = {}

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=args.resolution, seed=args.seed)

        log.info(f"\n--- Task {task_id+1}/{num_tasks}: {task_description} ---")

        num_episodes = min(args.episodes, len(initial_states))
        task_success = 0

        for ep in range(num_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[ep])

            # 10-step warm-up (physics settle)
            for _ in range(10):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Initialise history buffer — oldest first, newest last
            init_vec = obs_to_state_vec(obs)
            state_buf = deque([init_vec] * args.history_len, maxlen=args.history_len)

            frames = []
            episode_done = False

            for step in range(max_steps):
                # state_buf[0]=oldest, state_buf[-1]=most recent — matches dataset ordering
                state_history = list(state_buf)

                actions = await evo1_infer(ws, obs, state_history, task_description)

                for i in range(args.inference_horizon):
                    action = binarize_gripper(actions[i].tolist())

                    try:
                        obs, reward, done, info = env.step(action[:7])
                    except ValueError as e:
                        log.warning(f"Invalid action at step {step}: {e}")
                        episode_done = False
                        break

                    # Update history — append keeps oldest at index 0
                    state_buf.append(obs_to_state_vec(obs))

                    if not args.no_video:
                        frame = np.hstack([
                            np.rot90(obs["agentview_image"], 2),
                            np.rot90(obs["robot0_eye_in_hand_image"], 2),
                        ])
                        frames.append(frame)

                    if done:
                        episode_done = True
                        task_success += 1
                        suite_success += 1
                        break

                if episode_done:
                    break

            suite_episodes += 1

            status = "SUCCESS" if episode_done else "FAIL"
            log.info(f"  Task {task_id+1} | Ep {ep+1}: {status}")

            if not args.no_video and frames:
                vpath = os.path.join(
                    args.video_dir, suite_name,
                    f"task{task_id+1:02d}_ep{ep+1:03d}.mp4"
                )
                save_video(frames, vpath)

        rate = task_success / max(1, num_episodes)
        log.info(f"  Task {task_id+1} summary: {task_success}/{num_episodes}  rate={rate:.3f}")
        per_task_results[task_id] = {
            "description": task_description,
            "success": task_success,
            "episodes": num_episodes,
            "rate": rate,
        }
        env.close()

    suite_rate = suite_success / max(1, suite_episodes)
    log.info(f"\n  {suite_name} TOTAL: {suite_success}/{suite_episodes}  rate={suite_rate:.3f}")
    return {"per_task": per_task_results, "success": suite_success,
            "episodes": suite_episodes, "rate": suite_rate}


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    log = setup_logging(args.log_dir, "libero_eval")
    log.info(f"Server: {args.server_url}  episodes={args.episodes}  "
             f"history_len={args.history_len}  inference_horizon={args.inference_horizon}")
    log.info(f"Suites: {args.suites}")

    np.random.seed(args.seed)
    random.seed(args.seed)

    all_results: Dict[str, Dict] = {}

    async with websockets.connect(args.server_url, max_size=100_000_000) as ws:
        for suite_name in args.suites:
            all_results[suite_name] = await eval_suite(ws, suite_name, args, log)

    # ── Final summary ──────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("  FINAL SUMMARY")
    log.info("=" * 60)
    total_s, total_e = 0, 0
    for suite_name, res in all_results.items():
        log.info(f"  {suite_name:20s}  {res['success']:3d}/{res['episodes']:3d}  "
                 f"rate={res['rate']:.3f}")
        total_s += res["success"]
        total_e += res["episodes"]
    overall = total_s / max(1, total_e)
    log.info(f"\n  Overall: {total_s}/{total_e}  rate={overall:.3f}")

    return all_results


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
