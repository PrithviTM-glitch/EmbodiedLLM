"""
embeddings.py

Type embedding strategies for the temporal state encoder.
Each strategy is a small nn.Module that maps a feature type integer ID
to a bias vector η ∈ R^embed_dim added to the encoded token.

    e_i^feat = decay^i · LN(MLP(f)) + strategy(type_id)

Strategies differ only in how η is initialised and whether it is trained:

    NoEmbedding            — returns zero, used for single-type encoders
                             (Experiment 1, position history only)
    RandomInitEmbedding    — N(0,1) init, fully trainable (Strain A)
    RandomInitWithOrthLoss — N(0,1) init, trainable + orthogonality loss (Strain B)
    OrthogonalInitFrozen   — QR orthogonal init, frozen throughout (Strain C)

Adding a new strategy:
    1. Inherit BaseEmbeddingStrategy
    2. Override __init__, forward, and optionally orthogonality_loss / freeze
    3. Add to EMBEDDING_STRATEGY_REGISTRY
    No other files need to change.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbeddingStrategy(nn.Module):
    """
    Base class for type embedding strategies.

    All strategies expose the same interface so the encoder and training
    loop can use them interchangeably.
    """

    def forward(self, type_id: int, device: torch.device) -> torch.Tensor:
        """
        Return the bias vector for a given feature type.

        Args:
            type_id: integer index from FEATURE_TYPE_REGISTRY
            device:  target device for the returned tensor

        Returns:
            [embed_dim] bias vector
        """
        raise NotImplementedError

    def orthogonality_loss(self) -> torch.Tensor:
        """
        Pairwise cosine similarity penalty on the embedding table.

            L_orth = Σ_{i≠j} (η_i · η_j / (‖η_i‖ · ‖η_j‖))²

        Returns zero by default.  Override in Strain B.
        """
        return torch.tensor(0.0)

    def freeze(self) -> None:
        """
        Freeze all embedding parameters after Phase 0 pretraining.
        No-op by default (NoEmbedding, OrthogonalInitFrozen already frozen).
        Override in RandomInitEmbedding and RandomInitWithOrthLoss.
        """
        pass


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class NoEmbedding(BaseEmbeddingStrategy):
    """
    Returns a zero vector — no type distinction.
    Used for Experiment 1 (position history only) where all tokens
    are the same feature type and no tagging is needed.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, type_id: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.embed_dim, device=device)


class RandomInitEmbedding(BaseEmbeddingStrategy):
    """
    Strain A — random N(0,1) initialisation, fully trainable.

    Type embeddings receive gradient indirectly through the reconstruction
    loss during Phase 0.  No explicit orthogonality supervision.
    After freeze_pretrained_params() is called they become fixed.
    """

    def __init__(self, num_types: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_types, embed_dim)
        # default PyTorch init is N(0,1) — intentionally left as-is

    def forward(self, type_id: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor(type_id, dtype=torch.long, device=device)
        return self.embedding(idx)

    def freeze(self) -> None:
        self.embedding.weight.requires_grad = False


class RandomInitWithOrthLoss(BaseEmbeddingStrategy):
    """
    Strain B — random N(0,1) initialisation, trainable, with orthogonality loss.

    During Phase 0 pretraining the training loop adds:
        λ_orth · strategy.orthogonality_loss()
    to the reconstruction loss.  This explicitly pushes type embeddings
    toward orthogonality without guaranteeing it exactly.
    """

    def __init__(self, num_types: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_types, embed_dim)

    def forward(self, type_id: int, device: torch.device) -> torch.Tensor:
        idx = torch.tensor(type_id, dtype=torch.long, device=device)
        return self.embedding(idx)

    def orthogonality_loss(self) -> torch.Tensor:
        """
        L_orth = Σ_{i≠j} (η_i · η_j / (‖η_i‖ · ‖η_j‖))²
        """
        eta = self.embedding.weight                              # [N, embed_dim]
        eta_norm = F.normalize(eta, dim=1)                       # [N, embed_dim]
        cos_sim  = eta_norm @ eta_norm.T                         # [N, N]
        mask     = 1.0 - torch.eye(
            cos_sim.size(0), device=cos_sim.device
        )
        return (cos_sim.pow(2) * mask).sum()

    def freeze(self) -> None:
        self.embedding.weight.requires_grad = False


class OrthogonalInitFrozen(BaseEmbeddingStrategy):
    """
    Strain C — QR-orthogonal initialisation, frozen throughout all phases.

    Guarantees exact mutual orthogonality by construction.  The DiT sees
    maximally distinct type tags from the first step of Phase 1 with no
    risk of collapse or gradient-driven corruption.

    The embedding is registered as a buffer (not a parameter) so it is
    never included in any optimizer and moves with the model on .to(device).
    """

    def __init__(self, num_types: int, embed_dim: int):
        super().__init__()
        embeddings = self._init_orthogonal(num_types, embed_dim)
        self.register_buffer("embeddings", embeddings)

    @staticmethod
    def _init_orthogonal(num_types: int, embed_dim: int) -> torch.Tensor:
        Q, _ = torch.linalg.qr(torch.randn(embed_dim, embed_dim))
        return Q[:num_types, :]    # [num_types, embed_dim] — exactly orthogonal rows

    def forward(self, type_id: int, device: torch.device) -> torch.Tensor:
        return self.embeddings[type_id]

    def freeze(self) -> None:
        pass   # already a buffer, never in optimizer


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EMBEDDING_STRATEGY_REGISTRY: dict[str, type[BaseEmbeddingStrategy]] = {
    "none":   NoEmbedding,
    "A":      RandomInitEmbedding,
    "B":      RandomInitWithOrthLoss,
    "C":      OrthogonalInitFrozen,
}


def build_embedding_strategy(
    strain: str,
    num_types: int,
    embed_dim: int,
) -> BaseEmbeddingStrategy:
    """
    Factory function.  strain must be one of 'none', 'A', 'B', 'C'.
    """
    if strain not in EMBEDDING_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown embedding strain '{strain}'. "
            f"Choose from: {list(EMBEDDING_STRATEGY_REGISTRY.keys())}"
        )
    cls = EMBEDDING_STRATEGY_REGISTRY[strain]
    if strain == "none":
        return cls(embed_dim)
    return cls(num_types, embed_dim)