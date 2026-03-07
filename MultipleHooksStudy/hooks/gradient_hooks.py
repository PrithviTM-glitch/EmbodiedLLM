"""
Gradient Flow Analysis Hooks

Hooks for measuring gradient flow through different encoders in VLA models.
Tracks gradient magnitudes, layer-wise gradients, and computes ratios
to identify underutilization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import numpy as np
from collections import defaultdict

from .base_hooks import BaseGradientHook


class EncoderGradientTracker(BaseGradientHook):
    """
    Track gradient magnitudes at encoder outputs.
    
    Measures gradient norm flowing back to vision, language, and proprioceptive
    encoders to identify utilization patterns.
    """
    
    def __init__(self, name: str = "encoder_gradient_tracker"):
        super().__init__(name)
        self.encoder_names = []
    
    def attach_to_encoders(
        self,
        vision_encoder: Optional[nn.Module] = None,
        language_encoder: Optional[nn.Module] = None,
        proprio_encoder: Optional[nn.Module] = None,
        custom_encoders: Optional[Dict[str, nn.Module]] = None
    ):
        """
        Attach hooks to multiple encoders.
        
        Args:
            vision_encoder: Vision encoder module
            language_encoder: Language encoder module
            proprio_encoder: Proprioceptive encoder module (if exists)
            custom_encoders: Dict of custom encoder names to modules
        """
        if vision_encoder is not None:
            self.attach(vision_encoder, "vision_encoder")
            self.encoder_names.append("vision_encoder")
        
        if language_encoder is not None:
            self.attach(language_encoder, "language_encoder")
            self.encoder_names.append("language_encoder")
        
        if proprio_encoder is not None:
            self.attach(proprio_encoder, "proprio_encoder")
            self.encoder_names.append("proprio_encoder")
        
        if custom_encoders:
            for name, encoder in custom_encoders.items():
                self.attach(encoder, name)
                self.encoder_names.append(name)
    
    def compute_ratios(self) -> Dict[str, float]:
        """
        Compute gradient ratios between encoders.
        
        Returns:
            Dict with gradient ratios (e.g., proprio/vision, proprio/language)
        """
        results = self.get_results()
        ratios = {}
        
        if "proprio_encoder" in results and "vision_encoder" in results:
            proprio_mean = results["proprio_encoder"]["mean"]
            vision_mean = results["vision_encoder"]["mean"]
            if vision_mean > 0:
                ratios["proprio_vision_ratio"] = proprio_mean / vision_mean
        
        if "proprio_encoder" in results and "language_encoder" in results:
            proprio_mean = results["proprio_encoder"]["mean"]
            language_mean = results["language_encoder"]["mean"]
            if language_mean > 0:
                ratios["proprio_language_ratio"] = proprio_mean / language_mean
        
        if "vision_encoder" in results and "language_encoder" in results:
            vision_mean = results["vision_encoder"]["mean"]
            language_mean = results["language_encoder"]["mean"]
            if language_mean > 0:
                ratios["vision_language_ratio"] = vision_mean / language_mean
        
        return ratios
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary including ratios."""
        results = self.get_results()
        ratios = self.compute_ratios()
        
        return {
            "gradient_stats": results,
            "gradient_ratios": ratios,
            "encoder_names": self.encoder_names
        }


class LayerWiseGradientProfiler(BaseGradientHook):
    """
    Track gradients at each layer of an encoder.
    
    Identifies at which layer gradients start to vanish, indicating
    bottlenecks in information flow.
    """
    
    def __init__(self, name: str = "layerwise_gradient_profiler"):
        super().__init__(name)
        self.layer_indices = []
    
    def attach_to_layers(self, layers: List[nn.Module], layer_names: Optional[List[str]] = None):
        """
        Attach hooks to a list of layers.
        
        Args:
            layers: List of layer modules
            layer_names: Optional custom names for layers
        """
        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(layers))]
        
        assert len(layers) == len(layer_names), "Layers and names must match in length"
        
        for layer, name in zip(layers, layer_names):
            self.attach(layer, name)
            self.layer_indices.append(name)
    
    def find_vanishing_point(self, threshold: float = 1e-4) -> Optional[str]:
        """
        Find the layer where gradients drop below threshold.
        
        Args:
            threshold: Gradient magnitude threshold for "vanishing"
        
        Returns:
            Name of first layer where gradient drops below threshold
        """
        results = self.get_results()
        
        for layer_name in self.layer_indices:
            if layer_name in results:
                mean_grad = results[layer_name]["mean"]
                if mean_grad < threshold:
                    return layer_name
        
        return None
    
    def get_gradient_profile(self) -> Dict[str, float]:
        """
        Get gradient magnitude profile across layers.
        
        Returns:
            Dict mapping layer names to mean gradient magnitudes
        """
        results = self.get_results()
        profile = {}
        
        for layer_name in self.layer_indices:
            if layer_name in results:
                profile[layer_name] = results[layer_name]["mean"]
        
        return profile
    
    def compute_gradient_decay(self) -> List[float]:
        """
        Compute gradient decay ratios between consecutive layers.
        
        Returns:
            List of decay ratios (later_layer_grad / earlier_layer_grad)
        """
        profile = self.get_gradient_profile()
        decay_ratios = []
        
        layer_grads = [profile.get(name, 0.0) for name in self.layer_indices]
        
        for i in range(1, len(layer_grads)):
            if layer_grads[i-1] > 0:
                decay_ratios.append(layer_grads[i] / layer_grads[i-1])
            else:
                decay_ratios.append(0.0)
        
        return decay_ratios


class GradientFlowAnalyzer:
    """
    High-level analyzer that combines multiple gradient hooks.
    
    Provides comprehensive gradient flow analysis across all encoders
    and layers in a VLA model.
    """
    
    def __init__(self):
        self.encoder_tracker = EncoderGradientTracker()
        self.layer_profilers: Dict[str, LayerWiseGradientProfiler] = {}
    
    def setup_encoder_tracking(
        self,
        vision_encoder: Optional[nn.Module] = None,
        language_encoder: Optional[nn.Module] = None,
        proprio_encoder: Optional[nn.Module] = None,
        custom_encoders: Optional[Dict[str, nn.Module]] = None
    ):
        """Setup gradient tracking for encoders."""
        self.encoder_tracker.attach_to_encoders(
            vision_encoder, language_encoder, proprio_encoder, custom_encoders
        )
    
    def setup_layer_profiling(
        self,
        encoder_name: str,
        layers: List[nn.Module],
        layer_names: Optional[List[str]] = None
    ):
        """
        Setup layer-wise profiling for a specific encoder.
        
        Args:
            encoder_name: Name of the encoder being profiled
            layers: List of layers in the encoder
            layer_names: Optional custom layer names
        """
        profiler = LayerWiseGradientProfiler(name=f"{encoder_name}_profiler")
        profiler.attach_to_layers(layers, layer_names)
        self.layer_profilers[encoder_name] = profiler
    
    def reset_all(self):
        """Reset all trackers and profilers."""
        self.encoder_tracker.reset()
        for profiler in self.layer_profilers.values():
            profiler.reset()
    
    def remove_all(self):
        """Remove all hooks."""
        self.encoder_tracker.remove()
        for profiler in self.layer_profilers.values():
            profiler.remove()
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Get comprehensive gradient flow report.
        
        Returns:
            Dict containing:
            - Encoder-level gradient statistics
            - Gradient ratios between encoders
            - Layer-wise profiles for each encoder
            - Vanishing gradient points
        """
        report = {
            "encoder_summary": self.encoder_tracker.get_summary(),
            "layer_profiles": {},
            "vanishing_points": {},
            "gradient_decay": {}
        }
        
        for encoder_name, profiler in self.layer_profilers.items():
            report["layer_profiles"][encoder_name] = profiler.get_gradient_profile()
            report["vanishing_points"][encoder_name] = profiler.find_vanishing_point()
            report["gradient_decay"][encoder_name] = profiler.compute_gradient_decay()
        
        return report
    
    def print_summary(self):
        """Print human-readable summary of gradient flow."""
        report = self.get_comprehensive_report()
        
        print("=" * 80)
        print("GRADIENT FLOW ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Encoder summary
        print("\n### Encoder Gradient Statistics ###")
        for encoder_name, stats in report["encoder_summary"]["gradient_stats"].items():
            print(f"\n{encoder_name}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std:  {stats['std']:.6f}")
            print(f"  Min:  {stats['min']:.6f}")
            print(f"  Max:  {stats['max']:.6f}")
        
        # Gradient ratios
        print("\n### Gradient Ratios ###")
        for ratio_name, ratio_value in report["encoder_summary"]["gradient_ratios"].items():
            print(f"{ratio_name}: {ratio_value:.4f}")
        
        # Vanishing points
        print("\n### Vanishing Gradient Points ###")
        for encoder_name, vanishing_point in report["vanishing_points"].items():
            if vanishing_point:
                print(f"{encoder_name}: {vanishing_point}")
            else:
                print(f"{encoder_name}: No vanishing detected")
        
        print("=" * 80)
