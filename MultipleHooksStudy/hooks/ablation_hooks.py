"""
Ablation Analysis Hooks

Hooks for ablating different encoders to measure their contribution.
Includes zero-out ablation, noise injection, and coordinated multi-encoder ablation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
import numpy as np

from .base_hooks import BaseAblationHook


class ZeroOutAblationHook(BaseAblationHook):
    """
    Zero out encoder outputs for ablation studies.
    
    Measures performance drop when an encoder is completely removed.
    """
    
    def __init__(self, name: str = "zero_ablation"):
        super().__init__(name, ablation_type="zero")


class NoiseInjectionHook(BaseAblationHook):
    """
    Inject Gaussian noise into encoder outputs.
    
    Tests robustness - well-utilized encoders should be sensitive to noise.
    """
    
    def __init__(self, name: str = "noise_injection", noise_std: float = 0.1):
        super().__init__(name, ablation_type="noise")
        self.noise_std = noise_std
    
    def set_noise_level(self, noise_std: float):
        """Set noise standard deviation."""
        self.noise_std = noise_std
    
    def forward_hook_fn(self, module_name: str):
        """Create forward hook with configurable noise level."""
        def hook(module, input, output):
            if not self.enabled or not self.ablate:
                return None
            
            if isinstance(output, torch.Tensor):
                noise = torch.randn_like(output) * self.noise_std
                return output + noise
            elif isinstance(output, tuple):
                return tuple(
                    o + torch.randn_like(o) * self.noise_std if isinstance(o, torch.Tensor) else o
                    for o in output
                )
            
            return None
        return hook


class ModalityAblationManager:
    """
    Coordinate ablation across multiple encoders.
    
    Allows systematic testing of each modality's contribution while
    keeping others intact.
    """
    
    def __init__(self):
        self.ablation_hooks: Dict[str, BaseAblationHook] = {}
        self.encoder_modules: Dict[str, nn.Module] = {}
    
    def register_encoder(
        self,
        encoder_name: str,
        encoder_module: nn.Module,
        ablation_type: str = "zero"
    ):
        """
        Register an encoder for ablation.
        
        Args:
            encoder_name: Name of encoder (e.g., "vision", "proprio")
            encoder_module: The encoder module
            ablation_type: "zero" or "noise"
        """
        if ablation_type == "zero":
            hook = ZeroOutAblationHook(name=f"{encoder_name}_ablation")
        else:
            hook = NoiseInjectionHook(name=f"{encoder_name}_ablation")
        
        hook.attach(encoder_module, encoder_name)
        hook.set_ablate(False)  # Start disabled
        
        self.ablation_hooks[encoder_name] = hook
        self.encoder_modules[encoder_name] = encoder_module
    
    def ablate_encoder(self, encoder_name: str, ablate: bool = True):
        """Enable or disable ablation for a specific encoder."""
        if encoder_name in self.ablation_hooks:
            self.ablation_hooks[encoder_name].set_ablate(ablate)
    
    def ablate_all_except(self, keep_encoder: str):
        """Ablate all encoders except one."""
        for encoder_name in self.ablation_hooks.keys():
            if encoder_name == keep_encoder:
                self.ablate_encoder(encoder_name, False)
            else:
                self.ablate_encoder(encoder_name, True)
    
    def ablate_only(self, ablate_encoder: str):
        """Ablate only one encoder, keep others intact."""
        for encoder_name in self.ablation_hooks.keys():
            if encoder_name == ablate_encoder:
                self.ablate_encoder(encoder_name, True)
            else:
                self.ablate_encoder(encoder_name, False)
    
    def disable_all_ablations(self):
        """Disable all ablations (full model)."""
        for hook in self.ablation_hooks.values():
            hook.set_ablate(False)
    
    def remove_all_hooks(self):
        """Remove all ablation hooks."""
        for hook in self.ablation_hooks.values():
            hook.remove()
        self.ablation_hooks.clear()
    
    def get_ablation_status(self) -> Dict[str, bool]:
        """Get current ablation status for all encoders."""
        return {
            name: hook.ablate
            for name, hook in self.ablation_hooks.items()
        }


class AblationStudyCoordinator:
    """
    High-level coordinator for systematic ablation studies.
    
    Runs ablation experiments and tracks results.
    """
    
    def __init__(self):
        self.manager = ModalityAblationManager()
        self.results: Dict[str, Any] = {}
    
    def setup(
        self,
        vision_encoder: Optional[nn.Module] = None,
        language_encoder: Optional[nn.Module] = None,
        proprio_encoder: Optional[nn.Module] = None,
        custom_encoders: Optional[Dict[str, nn.Module]] = None,
        ablation_type: str = "zero"
    ):
        """Register all encoders for ablation."""
        if vision_encoder is not None:
            self.manager.register_encoder("vision", vision_encoder, ablation_type)
        
        if language_encoder is not None:
            self.manager.register_encoder("language", language_encoder, ablation_type)
        
        if proprio_encoder is not None:
            self.manager.register_encoder("proprio", proprio_encoder, ablation_type)
        
        if custom_encoders:
            for name, encoder in custom_encoders.items():
                self.manager.register_encoder(name, encoder, ablation_type)
    
    def run_ablation_experiment(
        self,
        eval_function,
        ablation_configs: List[Dict[str, bool]],
        config_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run systematic ablation experiment.
        
        Args:
            eval_function: Function that evaluates model and returns metrics
            ablation_configs: List of dicts specifying which encoders to ablate
            config_names: Names for each configuration
        
        Returns:
            Dict mapping configuration names to evaluation results
        """
        results = {}
        
        for config, name in zip(ablation_configs, config_names):
            # Set ablation state
            for encoder_name, should_ablate in config.items():
                self.manager.ablate_encoder(encoder_name, should_ablate)
            
            # Run evaluation
            metrics = eval_function()
            results[name] = {
                "metrics": metrics,
                "ablation_config": config,
                "ablation_status": self.manager.get_ablation_status()
            }
        
        # Reset to no ablation
        self.manager.disable_all_ablations()
        
        self.results = results
        return results
    
    def run_standard_ablations(self, eval_function) -> Dict[str, Any]:
        """
        Run standard ablation suite (full model + each encoder ablated).
        
        Args:
            eval_function: Function to evaluate model
        
        Returns:
            Dict with results for each ablation configuration
        """
        encoders = list(self.manager.ablation_hooks.keys())
        
        # Configuration 1: Full model (no ablation)
        configs = [
            {enc: False for enc in encoders}
        ]
        names = ["full_model"]
        
        # Configuration 2-N: Ablate each encoder individually
        for encoder in encoders:
            config = {enc: (enc == encoder) for enc in encoders}
            configs.append(config)
            names.append(f"ablate_{encoder}")
        
        return self.run_ablation_experiment(eval_function, configs, names)
    
    def compute_ablation_deltas(self, metric_key: str = "success_rate") -> Dict[str, float]:
        """
        Compute performance delta caused by ablating each encoder.
        
        Args:
            metric_key: Key to extract from metrics dict
        
        Returns:
            Dict mapping encoder names to performance delta
        """
        if not self.results:
            return {}
        
        baseline = self.results.get("full_model", {}).get("metrics", {}).get(metric_key, 0)
        deltas = {}
        
        for config_name, result in self.results.items():
            if config_name.startswith("ablate_"):
                encoder_name = config_name.replace("ablate_", "")
                ablated_perf = result["metrics"].get(metric_key, 0)
                delta = ablated_perf - baseline
                deltas[encoder_name] = delta
        
        return deltas
    
    def get_encoder_importance_ranking(self, metric_key: str = "success_rate") -> List[Tuple[str, float]]:
        """
        Rank encoders by importance (magnitude of ablation delta).
        
        Returns:
            List of (encoder_name, abs_delta) tuples, sorted by importance
        """
        deltas = self.compute_ablation_deltas(metric_key)
        ranking = [(name, abs(delta)) for name, delta in deltas.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def print_ablation_report(self, metric_key: str = "success_rate"):
        """Print human-readable ablation report."""
        print("=" * 80)
        print("ABLATION STUDY REPORT")
        print("=" * 80)
        
        if not self.results:
            print("No results available. Run ablation experiment first.")
            return
        
        # Baseline performance
        baseline = self.results.get("full_model", {}).get("metrics", {}).get(metric_key, 0)
        print(f"\nBaseline ({metric_key}): {baseline:.4f}")
        
        # Individual ablations
        print("\n### Ablation Deltas ###")
        deltas = self.compute_ablation_deltas(metric_key)
        for encoder_name, delta in deltas.items():
            print(f"{encoder_name}: {delta:+.4f} ({delta/baseline*100:+.2f}%)")
        
        # Importance ranking
        print("\n### Encoder Importance Ranking ###")
        ranking = self.get_encoder_importance_ranking(metric_key)
        for i, (encoder_name, importance) in enumerate(ranking, 1):
            print(f"{i}. {encoder_name}: {importance:.4f}")
        
        print("=" * 80)
    
    def cleanup(self):
        """Remove all hooks."""
        self.manager.remove_all_hooks()
