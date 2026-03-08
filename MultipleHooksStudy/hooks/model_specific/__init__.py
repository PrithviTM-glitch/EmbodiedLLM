"""
Model-Specific Hook Adapters

Adapters for attaching hooks to specific VLA model architectures.
Each model has different internal structure requiring custom hook attachment points.

Models:
- Evo-1: Integration module aligns VL with proprioceptive state
- RDT-1B: Single Linear layer (state_adaptor) for proprioceptive encoding
- π0: Single Linear projection (state_proj) for proprioceptive encoding
- π0.5: MLP pair (time_mlp_in/time_mlp_out); PyTorch: PI05Policy (lerobot/pi05_base)
- Octo: Linear projection + position embeddings
- SmolVLA: Single Linear layer (state_proj) → 1 VLM token, flow matching action expert
"""

from .evo1_hooks import Evo1Hooks
from .octo_hooks import OctoHooks
from .rdt_hooks import RDTHooks
from .pi0_hooks import Pi0Hooks
from .pi05_hooks import Pi05Hooks
from .smolvla_hooks import SmolVLAHooks

__all__ = [
    'Evo1Hooks',
    'OctoHooks',
    'RDTHooks',
    'Pi0Hooks',
    'Pi05Hooks',
    'SmolVLAHooks',
]
