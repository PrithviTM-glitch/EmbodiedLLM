#!/usr/bin/env python3
"""
libero_client_4tasks.py  --  patched for sweep orchestration
Changes vs upstream:
  - argparse replaces hardcoded Args class
  - --server-address / --server-port  drive SERVER_URL
  - --benchmark selects suite(s): a single name OR "all" to run all 4 and aggregate
  - --num-episodes sets episodes per task
  - --output writes a JSON result file that collect_sr() expects
  - --horizon overrides action-chunk horizon
  - --seed overrides RNG seed
"""
import argparse
import asyncio
import json
import logging
import math
import os
import pathlib
import random

import imageio
import numpy as np
import websockets

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["MUJOCO_GL"] = "egl"

LIBERO_DUMMY_ACTION = [0.0] * 7

ALL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-address", default="localhost")
    p.add_argument("--server-port",    type=int, default=9001)
    p.add_argument("--benchmark",      default="all",
                   help="'all' to run all 4 suites aggregated, or a single suite name: "
                        "libero_spatial / libero_object / libero_goal / libero_10")
    p.add_argument("--num-episodes",   type=int, default=10,
                   help="Episodes per task")
    p.add_argument("--max-steps",      type=int, default=None,
                   help="Max env steps per episode (default: auto per suite)")
    p.add_argument("--horizon",        type=int, default=14,
                   help="Action-chunk horizon (how many actions to execute per server call)")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--output",         default=None,
                   help="Path to write JSON result (required for sweep)")
    p.add_argument("--log-file",       default=None)
    return p.parse_args()

args = parse_args()

SERVER_URL = f"ws://{args.server_address}:{args.server_port}"

# Per-suite sensible defaults for max_steps
_SUITE_MAX_STEPS = {
    "libero_spatial": 25,
    "libero_object":  25,
    "libero_goal":    25,
    "libero_10":      95,
}

# Resolve which suites to run
if args.benchmark == "all":
    SUITES_TO_RUN = ALL_SUITES
else:
    if args.benchmark not in _SUITE_MAX_STEPS:
        raise ValueError(
            f"Unknown benchmark '{args.benchmark}'. "
            f"Use 'all' or one of: {list(_SUITE_MAX_STEPS.keys())}"
        )
    SUITES_TO_RUN = [args.benchmark]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_bench_tag = "all" if args.benchmark == "all" else args.benchmark
log_file = args.log_file or f"/tmp/libero_{_bench_tag}_{args.server_port}.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_json_dict(obs, prompt, resize_size=448):
    img       = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy     = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)
    return {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy),
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt":      prompt,
        "image_mask":  [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }


def get_libero_env(task, resolution=448, seed=42):
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def save_video(frames, filename, fps=30, save_dir="./videos"):
    os.makedirs(save_dir, exist_ok=True)
    if frames:
        imageio.mimsave(os.path.join(save_dir, filename), frames, fps=fps)

# ---------------------------------------------------------------------------
# Eval one suite over an existing WebSocket connection
# ---------------------------------------------------------------------------
async def run_suite(ws, suite_name: str, total_success: int, total_episodes: int,
                    episode_results: list):
    """Run all tasks in suite_name; returns updated (total_success, total_episodes)."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    num_tasks  = task_suite.n_tasks
    max_steps  = args.max_steps or _SUITE_MAX_STEPS.get(suite_name, 95)

    log.info(f"\n{'='*60}")
    log.info(f"Suite={suite_name}  tasks={num_tasks}  "
             f"eps/task={args.num_episodes}  max_steps={max_steps}  "
             f"horizon={args.horizon}")

    for task_id in range(num_tasks):
        task           = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_desc = get_libero_env(task, seed=args.seed)

        log.info(f"\n=== [{suite_name}] Task {task_id+1}/{num_tasks}: {task_desc} ===")

        task_success  = 0
        task_episodes = min(args.num_episodes, len(initial_states))

        for ep in range(task_episodes):
            env.reset()
            obs = env.set_init_state(initial_states[ep])

            # warm-up steps
            for _ in range(10):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            episode_done = False
            frames       = []

            for step in range(max_steps):
                send_data = obs_to_json_dict(obs, str(task_desc))
                await ws.send(json.dumps(send_data))

                raw = await ws.recv()
                try:
                    actions = np.array(json.loads(raw))
                except Exception as e:
                    log.error(f"Action parse error step={step}: {e}")
                    break

                for i in range(args.horizon):
                    action = actions[i].tolist()
                    action[6] = -1 if action[6] > 0.5 else 1
                    try:
                        obs, reward, done, info = env.step(action[:7])
                    except ValueError as ve:
                        log.error(f"Invalid action: {ve}")
                        break

                    frames.append(np.hstack([
                        np.rot90(obs["agentview_image"], 2),
                        np.rot90(obs["robot0_eye_in_hand_image"], 2),
                    ]))

                    if done:
                        episode_done = True
                        break

                if episode_done:
                    break

            success = int(episode_done)
            task_success   += success
            total_success  += success
            total_episodes += 1
            episode_results.append({
                "suite":   suite_name,
                "task_id": task_id,
                "task":    str(task_desc),
                "episode": ep,
                "success": success,
            })
            log.info(f"  ep {ep+1}: {'✅' if success else '❌'}")

            save_video(
                frames,
                f"{suite_name}_task{task_id+1}_ep{ep+1}.mp4",
                save_dir=f"./videos/{suite_name}",
            )

        log.info(f"[{suite_name}] Task {task_id+1}: {task_success}/{task_episodes}")

    return total_success, total_episodes


# ---------------------------------------------------------------------------
# Main eval coroutine — runs all suites over a single WS connection
# ---------------------------------------------------------------------------
async def run():
    log.info(f"Suites={SUITES_TO_RUN}  server={SERVER_URL}")

    total_success  = 0
    total_episodes = 0
    episode_results = []

    async with websockets.connect(
        SERVER_URL, max_size=100_000_000,
        ping_interval=120, ping_timeout=300, open_timeout=60,
    ) as ws:
        for suite_name in SUITES_TO_RUN:
            total_success, total_episodes = await run_suite(
                ws, suite_name, total_success, total_episodes, episode_results
            )

    # -----------------------------------------------------------------------
    # Write output JSON
    # -----------------------------------------------------------------------
    overall_sr = total_success / max(1, total_episodes)
    result = {
        "success_rate":  overall_sr,
        "total_success": total_success,
        "total_episodes": total_episodes,
        "suites":        SUITES_TO_RUN,
        "server_port":   args.server_port,
        "results":       episode_results,
    }
    log.info(f"\nOverall SR across all suites: {overall_sr:.3f}  "
             f"({total_success}/{total_episodes})")

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        log.info(f"Results written to {args.output}")

    return result


if __name__ == "__main__":
    np.random.seed(args.seed)
    random.seed(args.seed)
    asyncio.run(run())
