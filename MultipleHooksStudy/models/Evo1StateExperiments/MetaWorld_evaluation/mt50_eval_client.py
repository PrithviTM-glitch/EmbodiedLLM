"""
mt50_eval_client.py — MT50 evaluation client for Evo-1 with state history.

Maintains a rolling state buffer (deque of length history_len) so each inference
call receives the full [history_len, state_dim] context the temporal encoder expects.

Usage:
    python mt50_eval_client.py [options]

Key options (all have defaults):
    --server-url    ws://127.0.0.1:9000
    --episodes      10
    --horizon       400
    --seed          4042
    --target-level  all  (one of: all, easy, medium, hard, very_hard)
    --history-len   6
    --state-take    8    (number of obs dims to use per frame)
    --inference-horizon  15   (action chunk size from model)
    --no-video      disable video saving
    --no-window     disable cv2 imshow
"""

import argparse
import asyncio
import datetime
import json
import os
from collections import deque
from typing import Dict, List, Optional, Set

import cv2
import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import websockets

os.environ.setdefault("MUJOCO_GL", "egl")
gym.logger.min_level = gym.logger.ERROR

# ── Constants / defaults (overridden by argparse) ─────────────────────────────
_DEFAULTS = dict(
    server_url       = "ws://127.0.0.1:9000",
    camera_name      = "corner2",
    img_size         = (448, 448),
    state_take       = 24,
    history_len      = 6,
    inference_horizon= 15,
    episodes         = 10,
    episode_horizon  = 400,
    seed             = 4042,
    target_level     = "all",
    order_json       = "mt50_order.json",
    tasks_jsonl      = "tasks.jsonl",
    log_dir          = "logs",
    video_dir        = "episode_videos",
    inspect_dir      = "inspect_frames",
    video_fps        = 10,
    crop_keep_ratio  = 2/3,
)

# ── Arg parsing ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="MT50 Evo-1 eval client (with state history)")
    p.add_argument("--server-url",        default=_DEFAULTS["server_url"])
    p.add_argument("--episodes",          type=int,   default=_DEFAULTS["episodes"])
    p.add_argument("--horizon",           type=int,   default=_DEFAULTS["episode_horizon"],
                   help="Max env steps per episode")
    p.add_argument("--seed",              type=int,   default=_DEFAULTS["seed"])
    p.add_argument("--target-level",      default=_DEFAULTS["target_level"],
                   choices=["all", "easy", "medium", "hard", "very_hard"])
    p.add_argument("--history-len",       type=int,   default=_DEFAULTS["history_len"],
                   help="State history length k+1 (default 6 = k+1 for k=5, must match training config)")
    p.add_argument("--state-take",        type=int,   default=_DEFAULTS["state_take"],
                   help="Number of obs dims to use per frame (default 24 = full state_dim; use 8 for baseline)")
    p.add_argument("--inference-horizon", type=int,   default=_DEFAULTS["inference_horizon"],
                   help="Action chunk size returned by model")
    p.add_argument("--order-json",        default=_DEFAULTS["order_json"])
    p.add_argument("--tasks-jsonl",       default=_DEFAULTS["tasks_jsonl"])
    p.add_argument("--no-video",          action="store_true")
    p.add_argument("--no-window",         action="store_true")
    return p.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────
def make_log_path(log_dir: str, prefix: str = "mt50") -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{prefix}_{ts}.txt")


def log_write(path: str, text: str):
    print(text)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ── Image utils ───────────────────────────────────────────────────────────────
def render_frame(env, img_size, crop_ratio: float, show_window: bool) -> np.ndarray:
    rgb = np.ascontiguousarray(env.render(), dtype=np.uint8)
    rgb = cv2.rotate(rgb, cv2.ROTATE_180)

    if 0.0 < crop_ratio < 1.0:
        h, w = rgb.shape[:2]
        nh, nw = max(1, int(round(h * crop_ratio))), max(1, int(round(w * crop_ratio)))
        y0, x0 = (h - nh) // 2, (w - nw) // 2
        rgb = np.ascontiguousarray(rgb[y0:y0+nh, x0:x0+nw])

    rgb = cv2.resize(rgb, img_size)
    bgr = np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if show_window:
        try:
            cv2.imshow("MetaWorld", bgr)
            cv2.waitKey(1)
        except Exception:
            pass
    return bgr


def encode_bgr(img_bgr: np.ndarray) -> list:
    return img_bgr.astype(np.uint8).tolist()


def make_video_writer(env, video_path: str, fps: int, img_size, crop_ratio: float, show_window: bool):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    frame = render_frame(env, img_size, crop_ratio, show_window)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(frame)
    return writer, frame


# ── State utils ───────────────────────────────────────────────────────────────
def obs_to_vec(obs, take: int) -> List[float]:
    if isinstance(obs, dict):
        arr = (np.asarray(obs["observation"]) if "observation" in obs
               else np.concatenate([np.asarray(v).ravel() for v in obs.values()])).astype(np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr[:min(take, arr.shape[0])].tolist()


# ── Prompt book ───────────────────────────────────────────────────────────────
class PromptBook:
    def __init__(self, jsonl_path: str):
        self.by_idx:  Dict[int, str] = {}
        self.by_slug: Dict[str, str] = {}
        self.seq:     List[str]      = []
        if not os.path.exists(jsonl_path):
            print(f"[WARN] {jsonl_path} not found — prompts will be empty")
            return
        with open(jsonl_path, encoding="utf-8") as f:
            for obj in (json.loads(l) for l in f if l.strip()):
                txt = str(obj.get("task", "")).strip()
                if "idx"  in obj: self.by_idx[int(obj["idx"])]   = txt
                if "slug" in obj: self.by_slug[str(obj["slug"])] = txt
                self.seq.append(txt)

    def get(self, idx: int, slug: Optional[str] = None) -> str:
        if idx in self.by_idx:              return self.by_idx[idx]
        if slug and slug in self.by_slug:   return self.by_slug[slug]
        if 0 <= idx < len(self.seq):        return self.seq[idx]
        return ""


# ── Order / groups ────────────────────────────────────────────────────────────
def load_order_and_groups(order_json: str, total_envs: int):
    if os.path.exists(order_json):
        with open(order_json) as f:
            data = json.load(f)
        ordered = list(map(int, data["ordered_indices"]))
        groups  = {k: set(v) for k, v in data["groups"].items()}
        idx2slug = {int(k): v for k, v in data["idx_to_slug"].items()}
        print(f"[INFO] Loaded order from {order_json} ({len(ordered)} tasks)")
        return ordered, groups, idx2slug
    idx_list = list(range(total_envs))
    print("[WARN] mt50_order.json not found — using all tasks in index order")
    return idx_list, {g: set() for g in ["easy","medium","hard","very_hard"]}, {i: f"task-{i}" for i in idx_list}


# ── WebSocket inference ───────────────────────────────────────────────────────
async def evo1_infer(
    ws,
    img_bgr: np.ndarray,
    state_history: List[List[float]],   # [history_len, state_take]
    prompt: str,
) -> np.ndarray:
    dummy = np.zeros((448, 448, 3), dtype=np.uint8)
    payload = {
        "image":       [encode_bgr(img_bgr), encode_bgr(dummy), encode_bgr(dummy)],
        "state":       state_history,
        "prompt":      prompt,
        "image_mask":  [1, 0, 0],
        "action_mask": [1, 1, 1, 1] + [0] * 20,
    }
    await ws.send(json.dumps(payload))
    return np.asarray(json.loads(await ws.recv()), dtype=np.float32)


# ── Main eval loop ────────────────────────────────────────────────────────────
async def eval_mt50(args):
    log_path = make_log_path(_DEFAULTS["log_dir"])

    def log(text): log_write(log_path, text)

    save_video  = not args.no_video
    show_window = not args.no_window
    img_size    = _DEFAULTS["img_size"]
    crop_ratio  = _DEFAULTS["crop_keep_ratio"]
    video_dir   = _DEFAULTS["video_dir"]

    prompts = PromptBook(args.tasks_jsonl)

    envs = gym.make_vec(
        "Meta-World/MT50",
        vector_strategy="sync",
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=_DEFAULTS["camera_name"],
    )
    total_envs = len(envs.envs)

    ordered, groups, idx2slug = load_order_and_groups(args.order_json, total_envs)
    ordered = [i for i in ordered if 0 <= i < total_envs]

    if args.target_level != "all":
        allowed = groups.get(args.target_level, set())
        ordered = [i for i in ordered if idx2slug.get(i,"") in allowed]
        print(f"[INFO] Filtered to {args.target_level}: {len(ordered)} tasks")

    success_counts: Dict[int,int] = {i: 0 for i in ordered}
    trials_counts:  Dict[int,int] = {i: 0 for i in ordered}
    group_success = {k: 0 for k in ["easy","medium","hard","very_hard"]}
    group_trials  = {k: 0 for k in ["easy","medium","hard","very_hard"]}

    log(f"Evaluation started  server={args.server_url}  episodes={args.episodes}  "
        f"horizon={args.horizon}  history_len={args.history_len}  state_take={args.state_take}")

    async with websockets.connect(args.server_url, max_size=100_000_000) as ws:
        for idx in ordered:
            sub   = envs.envs[idx]
            slug  = idx2slug.get(idx, f"task-{idx}")
            prompt = prompts.get(idx, slug=slug)

            gname = next((g for g in group_trials if slug in groups.get(g,set())), None)

            for ep in range(args.episodes):
                # Vary goal position each episode
                for obj in (sub, getattr(sub, "unwrapped", None)):
                    fn = getattr(obj, "iterate_goal_position", None)
                    if callable(fn):
                        try: fn()
                        except Exception: pass
                        break

                obs, _ = sub.reset(seed=args.seed + ep)
                trials_counts[idx] += 1
                if gname: group_trials[gname] += 1

                # Warm-up step (lets physics settle)
                try:
                    a0 = np.clip(np.zeros(sub.action_space.shape, np.float32),
                                 sub.action_space.low, sub.action_space.high)
                    obs, _, _, _, _ = sub.step(a0)
                except Exception:
                    pass

                # Initialise state history buffer with current obs repeated
                init_vec = obs_to_vec(obs, args.state_take)
                state_buf = deque([init_vec] * args.history_len, maxlen=args.history_len)

                video_writer = None
                if save_video:
                    vpath = os.path.join(video_dir, f"task{idx:02d}_{slug}_ep{ep+1:03d}.mp4")
                    video_writer, _ = make_video_writer(sub, vpath, _DEFAULTS["video_fps"],
                                                        img_size, crop_ratio, show_window)

                steps = 0
                done  = False

                while steps < args.horizon and not done:
                    img_bgr = render_frame(sub, img_size, crop_ratio, show_window)
                    if save_video and video_writer:
                        video_writer.write(img_bgr)

                    # state_buf[0] = oldest (s_{t-k}), state_buf[-1] = most recent (s_t)
                    # Matches dataset ordering: history_states[0]=oldest, [-1]=current
                    state_history = list(state_buf)   # [history_len, state_take], oldest first

                    actions = await evo1_infer(ws, img_bgr, state_history, prompt)

                    for i in range(args.inference_horizon):
                        a4 = np.clip(np.asarray(actions[i][:4], np.float32),
                                     sub.action_space.low, sub.action_space.high)
                        obs, _, terminated, truncated, info = sub.step(a4)
                        steps += 1

                        # append (not appendleft) keeps oldest at index 0, newest at index -1
                        state_buf.append(obs_to_vec(obs, args.state_take))

                        if isinstance(info, dict) and info.get("success", 0) == 1:
                            success_counts[idx] += 1
                            if gname: group_success[gname] += 1
                            done = True
                            break
                        if terminated or truncated or steps >= args.horizon:
                            done = True
                            break

                if video_writer:
                    final = render_frame(sub, img_size, crop_ratio, show_window)
                    video_writer.write(final)
                    video_writer.release()

            s, t = success_counts[idx], trials_counts[idx]
            log(f"[Task {idx:02d} {slug:28s}] {prompt[:60]:60s}  "
                f"success={s}/{t}  rate={s/max(1,t):.3f}")

    envs.close()

    per_task  = {idx2slug.get(i,f"task-{i}"): success_counts[i]/max(1,trials_counts[i])
                 for i in ordered}
    per_group = {g: group_success[g]/max(1,group_trials[g])
                 for g in ["easy","medium","hard","very_hard"]}
    overall   = sum(success_counts.values()) / max(1, sum(trials_counts.values()))
    avg_group = sum(per_group.values()) / 4

    log("\n==== Per-task success rate ====")
    for slug, rate in per_task.items():
        log(f"  {slug:28s}  {rate:.3f}")

    log("\n==== Difficulty buckets ====")
    for g in ["easy","medium","hard","very_hard"]:
        log(f"  {g:10s}  {per_group[g]:.3f}")

    log(f"\n  Overall (flat avg)        : {overall:.3f}")
    log(f"  Overall (avg of groups)   : {avg_group:.3f}")
    log(f"\nLog saved to: {log_path}")

    return per_task, per_group, overall


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(eval_mt50(args))
