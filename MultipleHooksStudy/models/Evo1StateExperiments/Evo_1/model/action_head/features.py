"""
features.py

Feature extractor classes for the temporal state encoder.
Each extractor takes a state history matrix S ∈ R^(B × (k+1) × state_dim)
and returns a list of (feature_vector, type_id, timestep_idx) tuples.

    feature_vector : [B, state_dim]
    type_id        : int  — index into FEATURE_TYPE_REGISTRY
    timestep_idx   : int  — 0 = most recent, used as decay exponent

History convention throughout:
    S[:, 0, :] = s_t        (most recent)
    S[:, k, :] = s_{t-k}   (oldest)

Adding a new feature:
    1. Add its name to FEATURE_TYPE_REGISTRY
    2. Write a class inheriting BaseFeatureExtractor
    3. Plug it into the extractor list when building TemporalStateEncoder
    No other files need to change.
"""

from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# Global type registry
# ---------------------------------------------------------------------------

FEATURE_TYPE_REGISTRY: dict[str, int] = {
    "position":    0,
    "velocity":    1,
    "acceleration": 2,
    "trace":       3,
    "deviation":   4,
    # future entries go here — existing indices must never be renumbered
}


def register_feature_type(name: str) -> int:
    """
    Register a new feature type and return its assigned integer ID.
    Idempotent — calling twice with the same name returns the same ID.
    """
    if name not in FEATURE_TYPE_REGISTRY:
        FEATURE_TYPE_REGISTRY[name] = len(FEATURE_TYPE_REGISTRY)
    return FEATURE_TYPE_REGISTRY[name]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseFeatureExtractor:
    """
    Stateless feature extractor base class.

    Subclasses implement extract() which takes the full history matrix and
    returns a list of (feat, type_id, timestep_idx) tuples ready for encoding.
    """

    def extract(
        self,
        S: torch.Tensor,
    ) -> list[tuple[torch.Tensor, int, int]]:
        """
        Args:
            S: [B, k+1, state_dim]  — full state history

        Returns:
            list of (feat [B, state_dim], type_id int, timestep_idx int)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete extractors
# ---------------------------------------------------------------------------

class PositionFeatures(BaseFeatureExtractor):
    """
    Raw joint position snapshots from the history window.
    Produces k+1 tokens, one per timestep.

        feat[i] = s_{t-i} = S[:, i, :]
    """

    TYPE_ID = FEATURE_TYPE_REGISTRY["position"]

    def extract(self, S: torch.Tensor) -> list[tuple[torch.Tensor, int, int]]:
        k = S.shape[1] - 1
        return [
            (S[:, i, :], self.TYPE_ID, i)
            for i in range(k + 1)
        ]


class VelocityFeatures(BaseFeatureExtractor):
    """
    First-order finite differences (column-wise along time axis).
    Produces k tokens.

        vel[i] = s_{t-i} - s_{t-i-1}
               = S[:, i, :] - S[:, i+1, :]
    """

    TYPE_ID = FEATURE_TYPE_REGISTRY["velocity"]

    def extract(self, S: torch.Tensor) -> list[tuple[torch.Tensor, int, int]]:
        k = S.shape[1] - 1
        return [
            (S[:, i, :] - S[:, i + 1, :], self.TYPE_ID, i)
            for i in range(k)
        ]


class AccelerationFeatures(BaseFeatureExtractor):
    """
    Second-order finite differences (finite difference of velocity).
    Produces k-1 tokens.  Requires k >= 2.

        acc[i] = vel[i] - vel[i+1]
               = S[:,i,:] - 2·S[:,i+1,:] + S[:,i+2,:]
    """

    TYPE_ID = FEATURE_TYPE_REGISTRY["acceleration"]

    def extract(self, S: torch.Tensor) -> list[tuple[torch.Tensor, int, int]]:
        k = S.shape[1] - 1
        if k < 2:
            raise ValueError(
                f"AccelerationFeatures requires history_len k >= 2, got k={k}"
            )
        result = []
        for i in range(k - 1):
            vel_i  = S[:, i,     :] - S[:, i + 1, :]
            vel_i1 = S[:, i + 1, :] - S[:, i + 2, :]
            acc    = vel_i - vel_i1
            result.append((acc, self.TYPE_ID, i))
        return result


class EligibilityTrace(BaseFeatureExtractor):
    """
    Exponentially-decaying weighted sum of history states.
    Produces 1 token.

        z_t = Σ_{i=0}^{k} ρ^i · s_{t-i}

    Args:
        decay: fixed scalar ρ ∈ (0,1), default 0.9.
               This is the *trace* decay, distinct from the per-type
               learned decay logit used in the encoding pipeline.
    """

    TYPE_ID = FEATURE_TYPE_REGISTRY["trace"]

    def __init__(self, decay: float = 0.9):
        self.decay = decay

    def extract(self, S: torch.Tensor) -> list[tuple[torch.Tensor, int, int]]:
        k = S.shape[1] - 1
        powers = torch.tensor(
            [self.decay ** i for i in range(k + 1)],
            dtype=S.dtype,
            device=S.device,
        )                                                      # [k+1]
        trace = (S * powers.view(1, -1, 1)).sum(dim=1)        # [B, state_dim]
        return [(trace, self.TYPE_ID, 0)]


class DeviationFromMean(BaseFeatureExtractor):
    """
    Difference between the current state and the mean of the history window.
    Produces 1 token.  Captures deviation from the recent average trajectory.

        δ_t = s_t - (1/(k+1)) · Σ_{i=0}^{k} s_{t-i}
            = S[:,0,:] - mean(S, dim=1)
    """

    TYPE_ID = FEATURE_TYPE_REGISTRY["deviation"]

    def extract(self, S: torch.Tensor) -> list[tuple[torch.Tensor, int, int]]:
        mean_s = S.mean(dim=1)                 # [B, state_dim]
        delta  = S[:, 0, :] - mean_s
        return [(delta, self.TYPE_ID, 0)]