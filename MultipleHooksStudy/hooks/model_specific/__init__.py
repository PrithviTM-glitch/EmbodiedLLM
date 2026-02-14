"""
Model-Specific Hook Adapters

Adapters for attaching hooks to specific VLA model architectures.
Each model has different internal structure requiring custom hook attachment points.

Models:
- OpenVLA: No proprio encoder (baseline)
- Octo: Linear projection + position embeddings
- RDT-1B: MLP encoder with Fourier features
- π0: Separate encoder with block-wise causal masking
"""

from .openvla_hooks import OpenVLAHooks
from .octo_hooks import OctoHooks
from .rdt_hooks import RDTHooks
from .pi0_hooks import Pi0Hooks

__all__ = [
    'OpenVLAHooks',
    'OctoHooks',
    'RDTHooks',
    'Pi0Hooks'
]
