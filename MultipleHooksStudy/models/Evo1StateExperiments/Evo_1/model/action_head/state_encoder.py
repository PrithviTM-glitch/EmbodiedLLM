"""
state_encoder.py

TemporalStateEncoder — assembles feature extractors, embedding strategy,
shared MLP, and LayerNorm into a single nn.Module that produces context
tokens for the DiT cross-attention blocks.

Also exposes build_encoder() — a factory that constructs the right encoder
from a flat config dict so the training script never needs to import
individual extractor or strategy classes directly.

Encoding formula per token:
    e_i^feat = ρ_feat^i · LN(MLP(f_{t-i})) + η_feat

    where ρ_feat = σ(decay_logit[feat_type]) is learned per feature type
    and η_feat is the bias vector from the embedding strategy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from features import (
    BaseFeatureExtractor,
    PositionFeatures,
    VelocityFeatures,
    AccelerationFeatures,
    EligibilityTrace,
    DeviationFromMean,
    FEATURE_TYPE_REGISTRY,
)
from embeddings import (
    BaseEmbeddingStrategy,
    build_embedding_strategy,
)


# ---------------------------------------------------------------------------
# Extractor registry — maps config string → class
# ---------------------------------------------------------------------------

EXTRACTOR_REGISTRY: dict[str, type[BaseFeatureExtractor]] = {
    "position":    PositionFeatures,
    "velocity":    VelocityFeatures,
    "acceleration": AccelerationFeatures,
    "trace":       EligibilityTrace,
    "deviation":   DeviationFromMean,
}


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------

class TemporalStateEncoder(nn.Module):
    """
    Temporal state encoder.

    Args:
        feature_extractors:   ordered list of BaseFeatureExtractor instances.
                              The token sequence in forward() follows this order.
        embedding_strategy:   BaseEmbeddingStrategy instance.
                              Use NoEmbedding for single-type experiments.
        state_dim:            input dimension of each feature vector (default 7)
        hidden_dim:           MLP hidden dimension (default 1024)
        embed_dim:            MLP output / token dimension (default 896)
        num_feature_types:    number of distinct type IDs used by the extractors.
                              Must match the number of rows in the embedding table.

    The shared MLP and LayerNorm are the only learnable parameters here
    beyond the embedding strategy.  Decay logits (one per unique type ID)
    are also learnable and intentionally left unfrozen after Phase 0 so
    the DiT and encoder jointly learn the right temporal weighting.

    History convention for forward():
        state_history[:, 0, :] = s_t        (most recent)
        state_history[:, k, :] = s_{t-k}    (oldest)
    """

    def __init__(
        self,
        feature_extractors: list[BaseFeatureExtractor],
        embedding_strategy: BaseEmbeddingStrategy,
        history_len: int = 5,
        state_dim: int = 7,
        hidden_dim: int = 1024,
        embed_dim: int = 896,
        num_feature_types: int = 5,
    ):
        super().__init__()

        if not feature_extractors:
            raise ValueError("feature_extractors must not be empty.")

        self.extractors          = feature_extractors
        self.embedding_strategy  = embedding_strategy
        self.history_len         = history_len
        self.state_dim           = state_dim
        self.embed_dim           = embed_dim
        self.num_feature_types   = num_feature_types

        # Shared MLP: R^state_dim → R^hidden_dim → R^embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

        # LayerNorm on MLP output — normalises scale to match VLM tokens
        self.layer_norm = nn.LayerNorm(embed_dim)

        # One learnable decay logit per feature type.
        # Initialised to 0 → ρ = σ(0) = 0.5 at the start of Phase 1.
        # Intentionally NOT frozen after Phase 0.
        self.decay_logits = nn.Parameter(torch.zeros(num_feature_types))

    # ------------------------------------------------------------------
    # Phase 0 interface
    # ------------------------------------------------------------------

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Delegates to the embedding strategy.
        Returns zero for NoEmbedding and OrthogonalInitFrozen.
        Non-zero only for RandomInitWithOrthLoss (Strain B).
        """
        return self.embedding_strategy.orthogonality_loss()

    def freeze_pretrained_params(self) -> None:
        """
        Freeze MLP φ, LayerNorm γ/β, and the embedding strategy η
        after Phase 0 pretraining completes.
        Decay logits w remain trainable (updated jointly with DiT in Phase 1).
        """
        for p in self.mlp.parameters():
            p.requires_grad = False
        for p in self.layer_norm.parameters():
            p.requires_grad = False
        self.embedding_strategy.freeze()

    # ------------------------------------------------------------------
    # Internal encoding
    # ------------------------------------------------------------------

    def _encode(
        self,
        feat: torch.Tensor,
        type_id: int,
        timestep_idx: int,
    ) -> torch.Tensor:
        """
        Encode a single feature vector into a context token.

            e = ρ^i · LN(MLP(feat)) + η_feat

        Args:
            feat:         [B, state_dim]
            type_id:      integer from FEATURE_TYPE_REGISTRY
            timestep_idx: i — 0 is most recent, used as the decay exponent

        Returns:
            [B, embed_dim]
        """
        x   = self.layer_norm(self.mlp(feat))                         # [B, embed_dim]
        rho = torch.sigmoid(self.decay_logits[type_id])
        x   = (rho ** timestep_idx) * x
        x   = x + self.embedding_strategy(type_id, feat.device)       # add type bias
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, state_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_history: [B, k+1, state_dim]
                           state_history[:, 0, :] = most recent state s_t

        Returns:
            tokens: [B, num_tokens, embed_dim]
        """
        tokens: list[torch.Tensor] = []

        for extractor in self.extractors:
            for feat, type_id, t_idx in extractor.extract(state_history):
                tokens.append(self._encode(feat, type_id, t_idx))

        return torch.stack(tokens, dim=1)    # [B, num_tokens, embed_dim]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_tokens(self) -> int:
        """
        Total number of tokens produced per forward pass.
        Computed analytically from the extractor list and history_len.

            PositionFeatures    → k+1 tokens
            VelocityFeatures    → k tokens
            AccelerationFeatures → k-1 tokens
            everything else     → 1 token  (trace, deviation, future types)
        """
        k = self.history_len
        total = 0
        for ext in self.extractors:
            if isinstance(ext, PositionFeatures):
                total += k + 1
            elif isinstance(ext, VelocityFeatures):
                total += k
            elif isinstance(ext, AccelerationFeatures):
                total += k - 1
            else:
                total += 1
        return total


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encoder(config: dict) -> TemporalStateEncoder:
    """
    Build a TemporalStateEncoder from a flat config dict.

    Expected config keys:
        features        : list[str]  — ordered list of extractor names
                          e.g. ['position', 'velocity', 'acceleration', 'trace', 'deviation']
        history_len     : int        — k, must be >= 3 for acceleration
        embedding_strain: str        — 'none', 'A', 'B', or 'C'
        state_dim       : int        — default 7
        hidden_dim      : int        — default 1024
        embed_dim       : int        — default 896
        trace_decay     : float      — default 0.9 (only used if 'trace' in features)

    Example:
        config = {
            'features':         ['position', 'velocity', 'acceleration', 'trace', 'deviation'],
            'history_len':      5,
            'embedding_strain': 'C',
        }
        encoder = build_encoder(config)
    """
    feature_names   = config.get("features", ["position"])
    history_len     = config.get("history_len", 5)
    strain          = config.get("embedding_strain", "none")
    state_dim       = config.get("state_dim", 7)
    hidden_dim      = config.get("hidden_dim", 1024)
    embed_dim       = config.get("embed_dim", 896)
    trace_decay     = config.get("trace_decay", 0.9)

    if history_len < 3 and "acceleration" in feature_names:
        raise ValueError(
            f"history_len k={history_len} is too small for acceleration. "
            "Need k >= 3."
        )

    # Build extractors in the order specified by config
    extractors: list[BaseFeatureExtractor] = []
    for name in feature_names:
        if name not in EXTRACTOR_REGISTRY:
            raise ValueError(
                f"Unknown feature '{name}'. "
                f"Available: {list(EXTRACTOR_REGISTRY.keys())}"
            )
        cls = EXTRACTOR_REGISTRY[name]
        if name == "trace":
            extractors.append(cls(decay=trace_decay))
        else:
            extractors.append(cls())

    # Collect unique type IDs used by these extractors — needed for embedding table size
    unique_type_ids: set[int] = set()
    dummy_S = torch.zeros(1, history_len + 1, state_dim)
    for ext in extractors:
        for _, type_id, _ in ext.extract(dummy_S):
            unique_type_ids.add(type_id)
    num_types = max(unique_type_ids) + 1   # table must cover the highest ID

    # If only one type is used, override strain to 'none' — no need for embeddings
    if len(unique_type_ids) <= 1:
        strain = "none"

    embedding_strategy = build_embedding_strategy(strain, num_types, embed_dim)

    return TemporalStateEncoder(
        feature_extractors=extractors,
        embedding_strategy=embedding_strategy,
        history_len=history_len,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
        num_feature_types=num_types,
    )