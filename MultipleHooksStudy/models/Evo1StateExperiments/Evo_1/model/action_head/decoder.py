"""
decoder.py

Phase 0 pretraining components for the temporal state encoder.

Contains:
    StateDecoder      — small D_ψ: R^embed_dim → R^state_dim, discarded after Phase 0
    reconstruction_loss — L_recon for a batch of feature vectors
    pretrain_phase0   — full Phase 0 training loop

Phase 0 trains only:
    - TemporalStateEncoder MLP φ and LayerNorm γ/β
    - StateDecoder D_ψ (discarded after)
    - Type embeddings η (Strains A and B only — Strain C is already frozen)

After Phase 0:
    encoder.freeze_pretrained_params() is called to freeze MLP, LayerNorm,
    and type embeddings.  Decay logits remain trainable throughout.
    The StateDecoder is discarded — it is never used after pretraining.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import wandb

from state_encoder import TemporalStateEncoder


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class StateDecoder(nn.Module):
    """
    Small inverse MLP D_ψ: R^embed_dim → R^state_dim.

    Used only during Phase 0 pretraining to provide a direct reconstruction
    supervision signal to the shared MLP.  Discarded after Phase 0 completes.

    Architecture mirrors the encoder MLP in reverse:
        R^embed_dim → R^hidden_dim → R^state_dim
    """

    def __init__(
        self,
        embed_dim: int = 896,
        hidden_dim: int = 1024,
        state_dim: int = 7,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e: [B, embed_dim] — encoded token from LN(MLP(f))
        Returns:
            [B, state_dim] — reconstructed feature vector
        """
        return self.net(e)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def reconstruction_loss(
    encoder: TemporalStateEncoder,
    decoder: StateDecoder,
    feature_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruction loss for a batch of feature vectors.

        L_recon = (1/|F|) · Σ_{f ∈ F} ‖D_ψ(LN(MLP_φ(f))) − f‖²

    Args:
        encoder:       TemporalStateEncoder — only its MLP and LayerNorm are used here
        decoder:       StateDecoder D_ψ
        feature_batch: [N, state_dim] — flat batch of all feature vectors extracted
                       from the history window (positions, velocities, accelerations,
                       eligibility traces, deviations).  Assembled by the caller.

    Returns:
        scalar MSE loss
    """
    encoded    = encoder.layer_norm(encoder.mlp(feature_batch))   # [N, embed_dim]
    recon      = decoder(encoded)                                  # [N, state_dim]
    return F.mse_loss(recon, feature_batch)


def phase0_loss(
    encoder: TemporalStateEncoder,
    decoder: StateDecoder,
    feature_batch: torch.Tensor,
    lambda_orth: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined Phase 0 loss:

        L_pretrain = L_recon(φ) + λ_orth · L_orth(η)

    L_orth is non-zero only for Strain B (RandomInitWithOrthLoss).
    For Strains A and C it evaluates to zero with no gradient overhead.

    Args:
        encoder:       TemporalStateEncoder
        decoder:       StateDecoder
        feature_batch: [N, state_dim] — flat batch of feature vectors
        lambda_orth:   weight for orthogonality term, default 0.01

    Returns:
        total loss scalar, dict of component values for logging
    """
    l_recon = reconstruction_loss(encoder, decoder, feature_batch)
    l_orth  = encoder.orthogonality_loss().to(l_recon.device)
    total   = l_recon + lambda_orth * l_orth

    components = {
        "loss/recon": l_recon.item(),
        "loss/orth":  l_orth.item() if isinstance(l_orth, torch.Tensor) else float(l_orth),
        "loss/total": total.item(),
    }
    return total, components


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def build_feature_batch(
    encoder: TemporalStateEncoder,
    state_history: torch.Tensor,
) -> torch.Tensor:
    """
    Extract all feature vectors from a state history batch and flatten
    them into a single [N, state_dim] tensor for reconstruction training.

    This runs all feature extractors and collects the raw feature vectors
    before any MLP/LN/embedding — i.e. the targets for reconstruction.

    Args:
        encoder:       TemporalStateEncoder (extractors accessed here)
        state_history: [B, k+1, state_dim]

    Returns:
        [B * num_features_per_sample, state_dim]
    """
    feats = []
    for extractor in encoder.extractors:
        for feat, _, _ in extractor.extract(state_history):
            feats.append(feat)                  # each feat: [B, state_dim]
    return torch.cat(feats, dim=0)              # [B * num_features, state_dim]


def pretrain_phase0(
    encoder: TemporalStateEncoder,
    dataloader: torch.utils.data.DataLoader,
    accelerator,
    config: dict,
) -> StateDecoder:
    """
    Full Phase 0 pretraining loop.

    Trains:
        - encoder.mlp (φ)
        - encoder.layer_norm (γ, β)
        - decoder.net (ψ) — discarded on return
        - encoder.embedding_strategy (η) — Strains A and B only

    Does NOT train:
        - encoder.decay_logits
        - DiT blocks
        - VLM

    After this function returns the caller should call:
        encoder.freeze_pretrained_params()
    to freeze MLP, LayerNorm, and type embeddings before Phase 1.

    Args:
        encoder:     TemporalStateEncoder instance
        dataloader:  yields batches with key 'states' — [B, k+1, state_dim]
                     (state-only batches, no images needed)
        accelerator: HuggingFace Accelerator instance
        config:      dict with keys:
                         pretrain_steps  int    default 1000
                         pretrain_lr     float  default 1e-3
                         lambda_orth     float  default 0.01
                         log_interval    int    default 50
                         state_dim       int    default 7
                         embed_dim       int    default 896
                         hidden_dim      int    default 1024

    Returns:
        decoder (StateDecoder) — caller should discard after inspecting if needed
    """
    pretrain_steps = config.get("pretrain_steps", 1000)
    lr             = config.get("pretrain_lr", 1e-3)
    lambda_orth    = config.get("lambda_orth", 0.01)
    log_interval   = config.get("log_interval", 50)
    state_dim      = config.get("state_dim", 7)
    embed_dim      = config.get("embed_dim", 896)
    hidden_dim     = config.get("hidden_dim", 1024)

    decoder = StateDecoder(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        state_dim=state_dim,
    )

    # Freeze decay logits during Phase 0 — they have no meaning yet
    encoder.decay_logits.requires_grad = False

    # Collect trainable parameters
    trainable_params = (
        list(encoder.mlp.parameters())
        + list(encoder.layer_norm.parameters())
        + list(decoder.parameters())
    )
    # Add embedding strategy parameters if trainable (Strains A and B)
    for p in encoder.embedding_strategy.parameters():
        if p.requires_grad:
            trainable_params.append(p)

    optimizer = AdamW(trainable_params, lr=lr)

    # Do NOT prepare the encoder here — it is a submodule of the full model and
    # will be wrapped by accelerator.prepare(model, ...) in the main training loop.
    # Wrapping it twice causes DistributedDataParallel to be applied twice on
    # multi-GPU runs.  Move the decoder to the correct device manually instead.
    decoder = decoder.to(accelerator.device)
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

    encoder.train()
    decoder.train()

    step = 0
    while step < pretrain_steps:
        for batch in dataloader:
            if step >= pretrain_steps:
                break

            # Dataloader yields full trajectory batches — extract state history
            # Expected key: 'states' with shape [B, k+1, state_dim]
            state_history = batch["states"].to(dtype=torch.float32)

            if state_history.ndim == 2:
                # Fallback: single snapshot [B, state_dim] — unsqueeze for compatibility
                state_history = state_history.unsqueeze(1)

            feature_batch = build_feature_batch(encoder, state_history)

            optimizer.zero_grad(set_to_none=True)
            loss, components = phase0_loss(
                encoder, decoder, feature_batch, lambda_orth=lambda_orth
            )
            accelerator.backward(loss)
            optimizer.step()

            if step % log_interval == 0 and accelerator.is_main_process:
                logging.info(
                    f"[Phase 0 | step {step}/{pretrain_steps}] "
                    f"recon={components['loss/recon']:.4f}  "
                    f"orth={components['loss/orth']:.4f}  "
                    f"total={components['loss/total']:.4f}"
                )
                try:
                    wandb.log({
                        "phase0/recon":  components["loss/recon"],
                        "phase0/orth":   components["loss/orth"],
                        "phase0/total":  components["loss/total"],
                        "phase0/step":   step,
                    })
                except Exception:
                    pass

            step += 1

    # Restore decay logits to trainable for Phase 1
    encoder.decay_logits.requires_grad = True

    if accelerator.is_main_process:
        logging.info("Phase 0 pretraining complete.")
        try:
            wandb.log({
                "phase0/final_recon": components["loss/recon"],
                "loss":               components["loss/recon"],
            })
        except Exception:
            pass
        logging.info(
            "Call encoder.freeze_pretrained_params() before starting Phase 1."
        )

    return decoder   # caller discards this after Phase 0