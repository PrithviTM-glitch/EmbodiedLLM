#!/usr/bin/env python3
"""
mt50_evo1_client_prompt.py  --  patched for sweep orchestration
Changes vs upstream:
  - argparse replaces all module-level constants that matter for sweeps
  - --server-port  drives SERVER_URL
  - --output       writes a JSON result file that collect_sr() expects
  - --episodes     overrides episodes per task
  - --horizon      overrides action-chunk horizon
  - --seed         overrides RNG seed
  - --target-level filters task difficulty (default: all)
  - video/image saving disabled by default in sweep mode (--save-video to re-enable)
  All original eval logic (ordering, group bucketing, per-task rates) is preserved.
"""
import argparse
import asyncio
import datetime
import json
import os
import pathlib
from typing import Dict, List, Optional, Set

import cv2
import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import websockets

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server-address", default="127.0.0.1")
    p.add_argument("--server-port",    type=int, default=9001)
    p.add_argument("--output",         default=None,
                   help="Path to write JSON result (required for sweep)")
    p.add_argument("--episodes",       type=int, default=10,
                   help="Episodes per task")
    p.add_argument("--episode-horizon",type=int, default=400)
    p.add_argument("--horizon",        type=int, default=15,
                   help="Action-chunk horizon")
    p.add_argument("--seed",           type=int, default=4042)
    p.add_argument("--target-level",   default="all",
                   choices=["all", "easy", "medium", "hard", "very_hard"])
    p.add_argument("--camera",         default="corner2")
    p.add_argument("--state-take",     type=int, default=8)
    p.add_argument("--order-json",     default=None,
                   help="Path to mt50_order.json (auto-detected if not given)")
    p.add_argument("--tasks-jsonl",    default=None,
                   help="Path to tasks.jsonl for prompts")
    p.add_argument("--save-video",     action="store_true",
                   help="Save per-episode videos (slow; off by default)")
    p.add_argument("--log-file",       default=None)
    return p.parse_args()

args = parse_args()

SERVER_URL    = f"ws://{args.server_address}:{args.server_port}"
HORIZON       = args.horizon
EPISODES      = args.episodes
EPISODE_HORIZON = args.episode_horizon
SEED          = args.seed
CAMERA_NAME   = args.camera
STATE_TAKE    = args.state_take
TARGET_LEVEL  = args.target_level
IMG_SIZE      = (448, 448)
SAVE_VIDEO    = args.save_video
VIDEO_SAVE_DIR = "episode_videos"
VIDEO_FPS     = 10

# Resolve helper file paths relative to THIS script's directory first,
# then fall back to CWD (matches upstream behaviour).
_HERE = pathlib.Path(__file__).parent

def _find_file(cli_arg, filename):
    if cli_arg and os.path.exists(cli_arg):
        return cli_arg
    candidate = _HERE / filename
    if candidate.exists():
        return str(candidate)
    return filename   # let downstream code handle missing gracefully

ORDER_JSON_PATH = _find_file(args.order_json,  "mt50_order.json")
TASKS_JSONL_PATH= _find_file(args.tasks_jsonl, "tasks.jsonl")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH = args.log_file or os.path.join(LOG_DIR, f"mt50_{args.server_port}_{ts}.txt")

def log_write(text: str):
    print(text)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")

os.environ.setdefault("MUJOCO_GL", "egl")
gym.logger.min_level = gym.logger.ERROR

# ---------------------------------------------------------------------------
# Image / state helpers
# ---------------------------------------------------------------------------
def encode_image_uint8_list(img_bgr: np.ndarray):
    return img_bgr.astype(np.uint8).tolist()


def obs_to_state(obs, take: int = STATE_TAKE) -> List[float]:
    if isinstance(obs, dict):
        arr = (np.asarray(obs["observation"], dtype=np.float32).ravel()
               if "observation" in obs
               else np.concatenate([np.asarray(v).ravel() for v in obs.values()]).astype(np.float32))
    else:
        arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr[:min(take, arr.shape[0])].tolist()


def render_single_bgr(env) -> np.ndarray:
    rgb = env.render()
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)           # flip for Evo-1 convention
    if IMG_SIZE:
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr, dtype=np.uint8)


async def evo1_infer(ws, img_bgr: np.ndarray,
                     state_vec: List[float],
                     prompt: Optional[str] = None) -> np.ndarray:
    assert prompt, "prompt must be non-empty"
    dummy = np.zeros((448, 448, 3), dtype=np.uint8)
    payload = {
        "image": [
            encode_image_uint8_list(img_bgr),
            encode_image_uint8_list(dummy),
            encode_image_uint8_list(dummy),
        ],
        "state":       state_vec,
        "prompt":      prompt,
        "image_mask":  [1, 0, 0],
        "action_mask": [1, 1, 1, 1] + [0] * 20,
    }
    await ws.send(json.dumps(payload))
    data = json.loads(await ws.recv())
    return np.asarray(data, dtype=np.float32)

# ---------------------------------------------------------------------------
# Video helpers (only used when --save-video)
# ---------------------------------------------------------------------------
def create_video_writer(env, video_name: str):
    os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
    frame = render_single_bgr(env)
    h, w  = frame.shape[:2]
    path  = os.path.join(VIDEO_SAVE_DIR, video_name)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
    writer.write(frame)
    return writer


def close_video(writer, video_name, task_idx, slug, ep_num):
    if writer is None:
        return
    try:
        writer.release()
        log_write(f"[video] task={task_idx} {slug} ep={ep_num} -> {video_name}")
    except Exception as e:
        log_write(f"[video][ERROR] {e}")

# ---------------------------------------------------------------------------
# Prompt book
# ---------------------------------------------------------------------------
class PromptBook:
    def __init__(self, jsonl_path: str):
        self.by_idx:  Dict[int, str] = {}
        self.by_slug: Dict[str, str] = {}
        self.seq:     List[str]      = []
        if not os.path.exists(jsonl_path):
            log_write(f"[WARN] {jsonl_path} not found; prompts will be empty.")
            return
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj  = json.loads(line)
                idx  = obj.get("index", len(self.seq))
                slug = obj.get("slug", "")
                text = obj.get("prompt", obj.get("text", ""))
                self.by_idx[idx]  = text
                self.by_slug[slug] = text
                self.seq.append(text)

    def get(self, idx: int, slug: str = "") -> str:
        if idx in self.by_idx:
            return self.by_idx[idx]
        if slug and slug in self.by_slug:
            return self.by_slug[slug]
        return slug  # fallback: use task name as prompt


PROMPTS = PromptBook(TASKS_JSONL_PATH)

# ---------------------------------------------------------------------------
# Order / group loader
# ---------------------------------------------------------------------------
def load_order_and_groups(total_envs: int):
    if os.path.exists(ORDER_JSON_PATH):
        with open(ORDER_JSON_PATH, "r") as f:
            data = json.load(f)
        ordered_indices = list(map(int, data["ordered_indices"]))
        groups      = {k: set(v) for k, v in data["groups"].items()}
        idx_to_slug = {int(k): v for k, v in data["idx_to_slug"].items()}
        log_write(f"[INFO] Loaded order from {ORDER_JSON_PATH} ({len(ordered_indices)} tasks)")
        return ordered_indices, groups, idx_to_slug
    # fallback
    idx_list    = list(range(total_envs))
    idx_to_slug = {i: f"task-{i}" for i in idx_list}
    groups      = {"easy": set(), "medium": set(), "hard": set(), "very_hard": set()}
    log_write("[WARN] mt50_order.json not found; using all tasks in default order")
    return idx_list, groups, idx_to_slug

# ---------------------------------------------------------------------------
# Core eval
# ---------------------------------------------------------------------------
async def eval_mt50_with_groups():
    envs = gym.make_vec(
        "Meta-World/MT50",
        vector_strategy="sync",
        seed=SEED,
        render_mode="rgb_array",
        camera_name=CAMERA_NAME,
    )
    total_envs = len(envs.envs)

    ordered_indices, groups, idx_to_slug = load_order_and_groups(total_envs)
    ordered_indices = [i for i in ordered_indices if 0 <= i < total_envs]

    if TARGET_LEVEL.lower() != "all":
        allowed = groups.get(TARGET_LEVEL.lower(), set())
        ordered_indices = [i for i in ordered_indices
                           if idx_to_slug.get(i, "") in allowed]
        log_write(f"[INFO] Filtered to {TARGET_LEVEL}: {len(ordered_indices)} tasks")

    success_counts: Dict[int, int] = {i: 0 for i in ordered_indices}
    trials_counts:  Dict[int, int] = {i: 0 for i in ordered_indices}
    group_success = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}
    group_trials  = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}
    episode_results = []

    async with websockets.connect(
        SERVER_URL, max_size=100_000_000,
        ping_interval=120, ping_timeout=300, open_timeout=60,
    ) as ws:
        for idx in ordered_indices:
            sub   = envs.envs[idx]
            slug  = idx_to_slug.get(idx, f"task-{idx}")
            task_prompt = PROMPTS.get(idx, slug=slug)

            gname_for_task = next(
                (g for g in group_trials if slug in groups.get(g, set())),
                None
            )

            for ep in range(EPISODES):
                # randomise goal position when supported
                for obj in (sub, getattr(sub, "unwrapped", None)):
                    fn = getattr(obj, "iterate_goal_position", None)
                    if callable(fn):
                        try: fn()
                        except Exception: pass
                        break

                obs, _ = sub.reset(seed=SEED + ep)
                trials_counts[idx] += 1
                if gname_for_task:
                    group_trials[gname_for_task] += 1

                # warm-up
                try:
                    a0 = np.clip(
                        np.zeros(sub.action_space.shape, dtype=np.float32),
                        sub.action_space.low, sub.action_space.high
                    )
                    obs, _, _, _, _ = sub.step(a0)
                except Exception:
                    pass

                video_writer = (create_video_writer(sub, f"task{idx:02d}_{slug}_ep{ep+1:03d}.mp4")
                                if SAVE_VIDEO else None)

                steps = 0
                done  = False
                success = 0

                while steps < EPISODE_HORIZON and not done:
                    img_bgr   = render_single_bgr(sub)
                    state_vec = obs_to_state(obs)
                    actions   = await evo1_infer(ws, img_bgr, state_vec, prompt=task_prompt)

                    if SAVE_VIDEO and video_writer:
                        video_writer.write(img_bgr)

                    for i in range(HORIZON):
                        a4 = np.clip(
                            np.asarray(actions[i][:4], dtype=np.float32),
                            sub.action_space.low, sub.action_space.high,
                        )
                        obs, _, terminated, truncated, info = sub.step(a4)
                        steps += 1

                        if isinstance(info, dict) and info.get("success", 0) == 1:
                            success = 1
                            success_counts[idx] += 1
                            if gname_for_task:
                                group_success[gname_for_task] += 1
                            done = True
                            break

                        if terminated or truncated or steps >= EPISODE_HORIZON:
                            done = True
                            break

                episode_results.append({
                    "task_idx": idx,
                    "slug":     slug,
                    "episode":  ep,
                    "success":  success,
                })

                if SAVE_VIDEO:
                    close_video(video_writer,
                                f"task{idx:02d}_{slug}_ep{ep+1:03d}.mp4",
                                idx, slug, ep + 1)

            s, t = success_counts[idx], trials_counts[idx]
            log_write(f"[Task {idx:02d} {slug}] SR={s/max(1,t):.3f}  ({s}/{t})")

    envs.close()

    # metrics
    per_task: Dict[str, float] = {
        idx_to_slug.get(i, f"task-{i}"): success_counts[i] / max(1, trials_counts[i])
        for i in ordered_indices
    }
    per_group: Dict[str, float] = {
        g: group_success[g] / max(1, group_trials[g])
        for g in ["easy", "medium", "hard", "very_hard"]
    }
    overall = sum(success_counts.values()) / max(1, sum(trials_counts.values()))

    return per_task, per_group, overall, episode_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def _amain():
    log_write(f"[mt50_client] server={SERVER_URL}  episodes={EPISODES}  "
              f"horizon={HORIZON}  level={TARGET_LEVEL}")

    per_task, per_group, overall, episode_results = await eval_mt50_with_groups()

    log_write("\n==== Per-task success rate ====")
    for slug, rate in per_task.items():
        log_write(f"  {slug:30s}  {rate:.3f}")

    log_write("\n==== Difficulty buckets ====")
    for g in ["easy", "medium", "hard", "very_hard"]:
        log_write(f"  {g:<12s}  {per_group.get(g, 0.0):.3f}")

    avg4 = sum(per_group.get(g, 0.0) for g in ["easy", "medium", "hard", "very_hard"]) / 4
    log_write(f"\n==== Overall (avg of 4 buckets) ====\n  {avg4:.3f}")
    log_write(f"==== Overall (episode mean)      ====\n  {overall:.3f}")

    # ------------------------------------------------------------------
    # Write output JSON  (success_rate key is what collect_sr() looks for)
    # ------------------------------------------------------------------
    result = {
        "success_rate":     overall,
        "avg4_bucket_rate": avg4,
        "per_task":         per_task,
        "per_group":        per_group,
        "server_port":      args.server_port,
        "results":          episode_results,
    }

    if args.output:
        out = pathlib.Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        log_write(f"Results written to {args.output}")

    return result


if __name__ == "__main__":
    asyncio.run(_amain())
