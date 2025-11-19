"""
Benchmarks module for VLA evaluation.

This module contains benchmark classes for evaluating VLA models
on different datasets and tasks.
"""

from .base_benchmark import BaseBenchmark
from .openx_benchmark import OpenXBenchmark

__all__ = ['BaseBenchmark', 'OpenXBenchmark']
