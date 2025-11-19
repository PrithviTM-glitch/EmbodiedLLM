"""
Adapters module for VLA benchmark.

This module contains adapter classes that wrap different VLA models
to provide a standardized interface for benchmarking.
"""

from .base_adapter import BaseAdapter
from .octo_adapter import OctoAdapter

__all__ = ['BaseAdapter', 'OctoAdapter']
