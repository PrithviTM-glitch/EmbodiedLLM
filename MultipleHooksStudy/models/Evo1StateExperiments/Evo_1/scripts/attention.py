"""
attention.py

Attention weight logging utility for the temporal state encoder experiments.

Provides:
    register_attention_hooks()     — attach hooks to all DiT cross-attn blocks
    remove_hooks()                 — detach all hooks after logging
    enable_attention_weights()     — set need_weights=True on all blocks
    disable_attention_weights()    — set need_weights=False on all blocks
    compute_feature_attention()    — partition attention mass by feature group
    log_attention_weights()        — compute and log to wandb + swanlab
    run_attention_eval()           — full eval pass with hooks, single entry point

Usage in train.py:
    from attention import run_attention_eval

    if step % eval_interval == 0 and accelerator.is_main_process:
        run_attention_eval(model, batch, step, accelerator)

All functions are no-ops if the model has no state_encoder — safe to call
regardless of whether the temporal encoder is active.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import wandb
import swanlab

from state_encoder import TemporalStateEncoder
from features import (
    PositionFeatures,
    VelocityFeatures,
    AccelerationFeatures,
)


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def register_attention_hooks(model) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """
    Attach a forward hook to each BasicTransformerBlock.attn module.

    The hook fires after every forward pass through that attention module
    and stores the attention weight tensor in attention_store.

    Args:
        model: EVO1 model instance

    Returns:
        attention_store: dict mapping 'block_N' → attn_weights tensor
                         populated after a forward pass is run
        hooks:           list of hook handles — pass to remove_hooks() when done
    """
    attention_store: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(block_idx: int):
        def hook(module, input, output):
            """
            output is (attn_output, attn_weights) from nn.MultiheadAttention.
            attn_weights: [B, num_heads, H, N+3k+2]  if need_weights=True
                          None                         if need_weights=False (fast path)
            """
            attn_weights = output[1]
            if attn_weights is not None:
                attention_store[f"block_{block_idx}"] = attn_weights.detach().cpu()
        return hook

    for i, block in enumerate(model.action_head.transformer_blocks):
        handle = block.attn.register_forward_hook(make_hook(i))
        hooks.append(handle)

    return attention_store, hooks


def remove_hooks(hooks: list[Any]) -> None:
    """
    Remove all registered hooks.
    Call this immediately after the eval forward pass to restore normal behaviour.
    """
    for handle in hooks:
        handle.remove()


# ---------------------------------------------------------------------------
# need_weights flag management
# ---------------------------------------------------------------------------

def enable_attention_weights(model) -> None:
    """
    Set need_weights=True on all transformer blocks so the attention module
    returns weights rather than taking the fast path (which returns None).

    Must be called before the eval forward pass that you want to log.
    """
    for block in model.action_head.transformer_blocks:
        block._need_weights = True


def disable_attention_weights(model) -> None:
    """
    Restore need_weights=False on all transformer blocks.
    Call after the eval forward pass.
    """
    for block in model.action_head.transformer_blocks:
        block._need_weights = False


# ---------------------------------------------------------------------------
# Attention mass computation
# ---------------------------------------------------------------------------

def compute_feature_attention(
    attn_weights: torch.Tensor,
    num_vlm_tokens: int,
    encoder: TemporalStateEncoder,
) -> dict[str, float]:
    """
    Partition mean attention mass across context token groups.

    Context token layout (matches order in FlowmatchingActionHead._build_context_tokens):
        [VLM tokens (N)] [pos (k+1)] [vel (k)] [acc (k-1)] [trace (1)] [deviation (1)]

    Args:
        attn_weights:    [B, num_heads, H, N+3k+2] — raw attention weights from hook
        num_vlm_tokens:  N — number of VLM context tokens
        encoder:         TemporalStateEncoder — used to read history_len and extractor list

    Returns:
        dict mapping feature group name → scalar attention mass (sums to ~1.0)
    """
    # Average over batch, heads, and action token (horizon) dimensions
    # Result: [N + 3k + 2] — mean attention weight per context token
    mean_attn = attn_weights.float().mean(dim=(0, 1, 2))   # [total_context_tokens]

    k = encoder.history_len
    results: dict[str, float] = {}

    idx = 0

    # VLM tokens
    results["vlm"] = mean_attn[:num_vlm_tokens].sum().item()
    idx += num_vlm_tokens

    # State tokens — iterate extractors in the same order they were registered
    for ext in encoder.extractors:
        if isinstance(ext, PositionFeatures):
            n = k + 1
            results["position"] = mean_attn[idx : idx + n].sum().item()
        elif isinstance(ext, VelocityFeatures):
            n = k
            results["velocity"] = mean_attn[idx : idx + n].sum().item()
        elif isinstance(ext, AccelerationFeatures):
            n = k - 1
            results["acceleration"] = mean_attn[idx : idx + n].sum().item()
        else:
            # Single-token features: trace, deviation, any future additions
            n = 1
            results[ext.feature_name] = mean_attn[idx].item()
        idx += n

    return results


def compute_per_head_attention(
    attn_weights: torch.Tensor,
    num_vlm_tokens: int,
    encoder: TemporalStateEncoder,
) -> dict[str, list[float]]:
    """
    Same as compute_feature_attention but broken down per attention head.
    Useful for seeing whether different heads specialise on different feature types.

    Args:
        attn_weights: [B, num_heads, H, N+3k+2]

    Returns:
        dict mapping feature group name → list of attention mass per head
    """
    # Average over batch and action token dimensions only
    # Result: [num_heads, N+3k+2]
    mean_attn = attn_weights.float().mean(dim=(0, 2))   # [num_heads, total_context_tokens]
    num_heads = mean_attn.shape[0]

    k = encoder.history_len
    results: dict[str, list[float]] = {}
    idx = 0

    results["vlm"] = mean_attn[:, :num_vlm_tokens].sum(dim=1).tolist()
    idx += num_vlm_tokens

    for ext in encoder.extractors:
        if isinstance(ext, PositionFeatures):
            n = k + 1
            results["position"] = mean_attn[:, idx : idx + n].sum(dim=1).tolist()
        elif isinstance(ext, VelocityFeatures):
            n = k
            results["velocity"] = mean_attn[:, idx : idx + n].sum(dim=1).tolist()
        elif isinstance(ext, AccelerationFeatures):
            n = k - 1
            results["acceleration"] = mean_attn[:, idx : idx + n].sum(dim=1).tolist()
        else:
            n = 1
            results[ext.feature_name] = mean_attn[:, idx].tolist()
        idx += n

    return results


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_attention_weights(
    attention_store: dict[str, torch.Tensor],
    num_vlm_tokens: int,
    encoder: TemporalStateEncoder,
    step: int,
) -> None:
    """
    Compute feature attention for each block and log to wandb and swanlab.

    Logs:
        attn/block_N/vlm             — mean attention to VLM tokens
        attn/block_N/position        — mean attention to position tokens
        attn/block_N/velocity        — mean attention to velocity tokens
        attn/block_N/acceleration    — mean attention to acceleration tokens
        attn/block_N/trace           — mean attention to eligibility trace token
        attn/block_N/deviation       — mean attention to deviation token
        attn/mean/...                — same metrics averaged across all blocks

    Also logs the key research hypothesis metric:
        attn/hypothesis/vel_gt_pos   — 1.0 if α_vel > α_pos, else 0.0

    Args:
        attention_store: dict from register_attention_hooks(), populated after forward pass
        num_vlm_tokens:  N — number of VLM context tokens in the batch
        encoder:         TemporalStateEncoder
        step:            current training step for wandb x-axis
    """
    if not attention_store:
        logging.warning("attention_store is empty — no weights were captured. "
                        "Check that need_weights=True was set before the forward pass.")
        return

    all_block_results: list[dict[str, float]] = []

    for block_name, weights in sorted(attention_store.items()):
        feature_attn = compute_feature_attention(weights, num_vlm_tokens, encoder)
        all_block_results.append(feature_attn)

        log_dict = {f"attn/{block_name}/{k}": v for k, v in feature_attn.items()}
        log_dict["step"] = step
        wandb.log(log_dict)
        swanlab.log(log_dict)

        logging.info(
            f"[Step {step}] {block_name} attention — "
            + "  ".join(f"{k}: {v:.4f}" for k, v in feature_attn.items())
        )

    # Mean across all blocks
    if all_block_results:
        keys = all_block_results[0].keys()
        mean_results = {
            k: sum(r[k] for r in all_block_results) / len(all_block_results)
            for k in keys
        }
        mean_log = {f"attn/mean/{k}": v for k, v in mean_results.items()}
        mean_log["step"] = step
        wandb.log(mean_log)
        swanlab.log(mean_log)

        logging.info(
            f"[Step {step}] mean attention across blocks — "
            + "  ".join(f"{k}: {v:.4f}" for k, v in mean_results.items())
        )

        # Key hypothesis metric: does velocity receive more attention than position?
        vel_gt_pos = 0.0
        if "velocity" in mean_results and "position" in mean_results:
            vel_gt_pos = 1.0 if mean_results["velocity"] > mean_results["position"] else 0.0
            wandb.log({"attn/hypothesis/vel_gt_pos": vel_gt_pos, "step": step})
            swanlab.log({"attn/hypothesis/vel_gt_pos": vel_gt_pos, "step": step})
            logging.info(
                f"[Step {step}] hypothesis α_vel > α_pos: "
                f"{'SUPPORTED' if vel_gt_pos else 'NOT SUPPORTED'} "
                f"(vel={mean_results['velocity']:.4f}, pos={mean_results['position']:.4f})"
            )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_attention_eval(
    model,
    batch: dict,
    step: int,
    accelerator,
) -> None:
    """
    Full attention logging eval pass. Single entry point for train.py.

    Registers hooks, runs one forward pass with need_weights=True,
    removes hooks, computes and logs feature attention.

    Only runs on the main process. No-op if model has no state_encoder.

    Args:
        model:       EVO1 model (or accelerator-wrapped version)
        batch:       one batch dict from the dataloader
        step:        current training step
        accelerator: HuggingFace Accelerator instance
    """
    if not accelerator.is_main_process:
        return

    # Unwrap model if wrapped by accelerator (DDP etc.)
    unwrapped = accelerator.unwrap_model(model)

    if (
        not hasattr(unwrapped, "action_head")
        or not hasattr(unwrapped.action_head, "state_encoder")
        or unwrapped.action_head.state_encoder is None
    ):
        logging.info("[Attention eval] No state encoder found — skipping.")
        return

    encoder = unwrapped.action_head.state_encoder

    # Enable weight computation and register hooks
    enable_attention_weights(unwrapped)
    attention_store, hooks = register_attention_hooks(unwrapped)

    # Run one forward pass with no gradient
    try:
        with torch.no_grad():
            fused_tokens_list = []
            prompts      = batch["prompts"]
            images_batch = batch["images"]
            image_masks  = batch["image_masks"]
            states       = batch["states"].to(dtype=torch.bfloat16)
            actions_gt   = batch["actions"].to(dtype=torch.bfloat16)
            action_mask  = batch["action_mask"]

            for prompt, images, image_mask in zip(prompts, images_batch, image_masks):
                fused = unwrapped.get_vl_embeddings(
                    images=images,
                    image_mask=image_mask,
                    prompt=prompt,
                    return_cls_only=False,
                )
                fused_tokens_list.append(fused.to(dtype=torch.bfloat16))

            fused_tokens = torch.cat(fused_tokens_list, dim=0)
            num_vlm_tokens = fused_tokens.shape[1]

            unwrapped(
                fused_tokens,
                state=states,
                actions_gt=actions_gt,
                action_mask=action_mask,
            )
    finally:
        # Always remove hooks and disable weights even if forward pass throws
        remove_hooks(hooks)
        disable_attention_weights(unwrapped)

    # Compute and log
    log_attention_weights(attention_store, num_vlm_tokens, encoder, step)