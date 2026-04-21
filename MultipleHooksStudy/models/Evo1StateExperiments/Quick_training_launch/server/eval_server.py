"""
eval_server.py — WebSocket inference server for Evo-1 evaluation.

Loads an Accelerate safetensors checkpoint and serves inference over WebSocket.
Compatible with mt50_eval_client.py.

Usage:
    python eval_server.py --ckpt_dir /path/to/step_XXXXX [--port 9000] [--device cuda] [--timesteps 32]

The checkpoint directory must contain:
    model.safetensors   — weights saved by accelerator.save_state()
    config.json         — model architecture config
    norm_stats.json     — per-robot normalisation statistics
"""

import argparse
import asyncio
import json
import os
import sys

import cv2
import numpy as np
import torch
import websockets
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

# ── Locate Evo_1 package ───────────────────────────────────────────────────────
# Default: two levels up from Quick_training_launch/server/ → repo root / Evo_1
# Override with --evo1-root if the script is run from a different location.
_DEFAULT_EVO1_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Evo_1"))


def _setup_evo1_path(evo1_root: str):
    sys.path.insert(0, evo1_root)
    sys.path.insert(0, os.path.join(evo1_root, "scripts"))
    sys.path.insert(0, os.path.join(evo1_root, "model", "action_head"))


# ── Normalizer ───────────────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path) as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                x = torch.cat([x, torch.zeros(24 - x.shape[0])])
            elif x.shape[0] > 24:
                raise ValueError(f"Stat vector length {x.shape[0]} > 24")
            return x

        if len(stats) != 1:
            raise ValueError(f"norm_stats.json must have exactly one robot key, got: {list(stats.keys())}")
        robot_stats = stats[list(stats.keys())[0]]

        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mn = self.state_min.to(state.device, dtype=state.dtype)
        mx = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - mn) / (mx - mn + 1e-8) - 1, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mn = self.action_min.to(action.device, dtype=action.dtype)
        mx = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (mx - mn + 1e-8) + mn


# ── Model loading ─────────────────────────────────────────────────────────────
_loaded_config: dict = {}  # set by load_model_and_normalizer, read by run_inference for shape checks


def load_model_and_normalizer(ckpt_dir: str, device: str = "cuda", timesteps: int = 32):
    from Evo1 import EVO1  # _setup_evo1_path() must be called before this function

    for fname in ("config.json", "norm_stats.json", "model.safetensors"):
        p = os.path.join(ckpt_dir, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing checkpoint file: {p}")

    global _loaded_config
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats  = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))
    _loaded_config = config

    config["finetune_vlm"]            = False
    config["finetune_action_head"]    = False
    config["num_inference_timesteps"] = timesteps
    config["device"]                  = device

    print(f"[server] Building EVO1  vlm={config.get('vlm_name')}  "
          f"horizon={config.get('horizon')}  layers={config.get('num_layers')}")
    model = EVO1(config).eval()

    weights_path = os.path.join(ckpt_dir, "model.safetensors")
    print(f"[server] Loading weights from {weights_path}")
    model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
    model = model.to(device)
    print(f"[server] Ready — {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    return model, Normalizer(stats)


# ── Image decode ──────────────────────────────────────────────────────────────
def decode_image(img_list, device: str = "cuda"):
    arr = np.array(img_list, dtype=np.uint8)
    arr = cv2.resize(arr, (448, 448))
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(Image.fromarray(arr)).to(device)


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(data: dict, model, normalizer: Normalizer, device: str = "cuda"):
    """
    data["state"] : List[List[float]]  shape [history_len, state_take]
    Returns       : List[List[float]]  shape [horizon, 24]
    """
    images = [decode_image(img, device) for img in data["image"]]
    assert len(images) == 3, "Payload must contain exactly 3 images"

    # State history: [history_len, state_take] → pad → [history_len, 24] → [1, history_len, 24]
    state = torch.tensor(data["state"], dtype=torch.float32, device=device)  # [H, D]
    if state.ndim == 1:
        state = state.unsqueeze(0)                                            # [1, D] fallback
    expected_h = _loaded_config.get("history_len", 5) + 1 if _loaded_config else None
    if expected_h is not None and state.shape[0] != expected_h:
        print(f"[WARN] state history length {state.shape[0]} != expected {expected_h} "
              f"(history_len={expected_h-1} from config)")
    if state.shape[-1] < 24:
        pad = torch.zeros(*state.shape[:-1], 24 - state.shape[-1], device=device)
        state = torch.cat([state, pad], dim=-1)                               # [H, 24]
    norm_state = normalizer.normalize_state(state).unsqueeze(0).to(torch.float32)  # [1, H, 24]

    image_mask  = torch.tensor(data["image_mask"],  dtype=torch.int32,  device=device)
    action_mask = torch.tensor([data["action_mask"]], dtype=torch.int32, device=device)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=data["prompt"],
            state_input=norm_state,
            action_mask=action_mask,
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0])
        return action.cpu().numpy().tolist()


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def handle(websocket, model, normalizer, device):
    print("[server] Client connected")
    try:
        async for message in websocket:
            data    = json.loads(message)
            actions = run_inference(data, model, normalizer, device)
            await websocket.send(json.dumps(actions))
    except websockets.exceptions.ConnectionClosed:
        print("[server] Client disconnected")


# ── Entrypoint ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evo-1 WebSocket inference server")
    p.add_argument("--ckpt_dir",     required=True,        help="Path to step_XXXXX checkpoint directory")
    p.add_argument("--port",         type=int, default=9000)
    p.add_argument("--device",       default="cuda")
    p.add_argument("--timesteps",    type=int, default=32, help="Flow-matching inference timesteps")
    p.add_argument("--evo1-root",    default=None,
                   help=f"Path to Evo_1 package root (default: {_DEFAULT_EVO1_ROOT})")
    p.add_argument("--ablate-state", action="store_true",
                   help="Zero out the state encoder output (ablation: same shape, no state signal)")
    return p.parse_args()


def apply_state_ablation(model):
    """Register a forward hook that zeros the state encoder output.
    The hook preserves the output tensor shape so downstream attention layers
    are unaffected — this isolates whether the state encoder is helping."""
    se = getattr(getattr(model, "action_head", None), "state_encoder", None)
    if se is None:
        print("[server] WARNING: state_encoder not found — --ablate-state has no effect")
        return

    def _zero_output(module, input, output):
        return torch.zeros_like(output)

    se.register_forward_hook(_zero_output)
    n_params = sum(p.numel() for p in se.parameters()) / 1e6
    print(f"[server] ABLATION: state encoder output zeroed "
          f"({n_params:.2f}M params bypassed, shape preserved)")


async def serve(model, normalizer, device, port):
    print(f"[server] Listening on ws://0.0.0.0:{port}")
    async with websockets.serve(
        lambda ws: handle(ws, model, normalizer, device),
        "0.0.0.0", port,
        max_size=100_000_000,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    args = parse_args()
    _setup_evo1_path(args.evo1_root or _DEFAULT_EVO1_ROOT)
    model, normalizer = load_model_and_normalizer(args.ckpt_dir, args.device, args.timesteps)
    if args.ablate_state:
        apply_state_ablation(model)
    asyncio.run(serve(model, normalizer, args.device, args.port))
