"""
Base Hook Classes for VLA Encoder Analysis

Abstract base classes for all hook types used in the study.
Provides common interface for gradient tracking, feature extraction,
ablation, and attention weight monitoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np


class BaseHook(ABC):
    """Abstract base class for all hooks."""
    
    def __init__(self, name: str = "hook"):
        self.name = name
        self.enabled = True
        self.handles = []  # Store hook handles for removal
    
    def enable(self):
        """Enable this hook."""
        self.enabled = True
    
    def disable(self):
        """Disable this hook."""
        self.enabled = False
    
    def remove(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    @abstractmethod
    def attach(self, module: nn.Module, **kwargs):
        """Attach hook to a module."""
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Get hook results."""
        pass
    
    def reset(self):
        """Reset hook state."""
        pass


class BaseGradientHook(BaseHook):
    """Base class for gradient tracking hooks."""
    
    def __init__(self, name: str = "gradient_hook"):
        super().__init__(name)
        self.gradients = defaultdict(list)
    
    def backward_hook_fn(self, module_name: str):
        """Create a backward hook function for a specific module."""
        def hook(module, grad_input, grad_output):
            if not self.enabled:
                return None
            
            if grad_output[0] is not None:
                grad_norm = grad_output[0].detach().norm().item()
                self.gradients[module_name].append(grad_norm)
            
            return None
        return hook
    
    def attach(self, module: nn.Module, module_name: str = "module"):
        """Attach backward hook to module."""
        handle = module.register_full_backward_hook(
            self.backward_hook_fn(module_name)
        )
        self.handles.append(handle)
    
    def get_results(self) -> Dict[str, Any]:
        """Get gradient statistics."""
        results = {}
        for module_name, grads in self.gradients.items():
            if grads:
                results[module_name] = {
                    'mean': np.mean(grads),
                    'std': np.std(grads),
                    'min': np.min(grads),
                    'max': np.max(grads),
                    'all_values': grads
                }
        return results
    
    def reset(self):
        """Reset gradient accumulation."""
        self.gradients.clear()


class BaseFeatureHook(BaseHook):
    """Base class for feature extraction hooks."""
    
    def __init__(self, name: str = "feature_hook", store_all: bool = False):
        super().__init__(name)
        self.features = defaultdict(list) if store_all else {}
        self.store_all = store_all
    
    def forward_hook_fn(self, module_name: str):
        """Create a forward hook function for feature extraction."""
        def hook(module, input, output):
            if not self.enabled:
                return None
            
            # Handle different output types
            if isinstance(output, torch.Tensor):
                feature = output.detach().cpu()
            elif isinstance(output, tuple):
                feature = output[0].detach().cpu()
            else:
                feature = output
            
            if self.store_all:
                self.features[module_name].append(feature)
            else:
                self.features[module_name] = feature
            
            return None
        return hook
    
    def attach(self, module: nn.Module, module_name: str = "module"):
        """Attach forward hook to module."""
        handle = module.register_forward_hook(
            self.forward_hook_fn(module_name)
        )
        self.handles.append(handle)
    
    def get_results(self) -> Dict[str, Any]:
        """Get extracted features."""
        return dict(self.features)
    
    def reset(self):
        """Reset feature storage."""
        if self.store_all:
            self.features = defaultdict(list)
        else:
            self.features.clear()


class BaseAblationHook(BaseHook):
    """Base class for ablation hooks."""
    
    def __init__(self, name: str = "ablation_hook", ablation_type: str = "zero"):
        super().__init__(name)
        self.ablation_type = ablation_type
        self.ablate = False
    
    def set_ablate(self, ablate: bool):
        """Enable or disable ablation."""
        self.ablate = ablate
    
    def forward_hook_fn(self, module_name: str):
        """Create forward hook for ablation."""
        def hook(module, input, output):
            if not self.enabled or not self.ablate:
                return None
            
            if self.ablation_type == "zero":
                # Zero out the output
                if isinstance(output, torch.Tensor):
                    return torch.zeros_like(output)
                elif isinstance(output, tuple):
                    return tuple(torch.zeros_like(o) if isinstance(o, torch.Tensor) else o 
                               for o in output)
            elif self.ablation_type == "noise":
                # Add noise to output
                if isinstance(output, torch.Tensor):
                    noise = torch.randn_like(output) * 0.1
                    return output + noise
                elif isinstance(output, tuple):
                    return tuple(
                        o + torch.randn_like(o) * 0.1 if isinstance(o, torch.Tensor) else o
                        for o in output
                    )
            
            return None
        return hook
    
    def attach(self, module: nn.Module, module_name: str = "module"):
        """Attach forward hook to module for ablation."""
        handle = module.register_forward_hook(
            self.forward_hook_fn(module_name)
        )
        self.handles.append(handle)
    
    def get_results(self) -> Dict[str, Any]:
        """Get ablation status."""
        return {
            "ablation_type": self.ablation_type,
            "ablate": self.ablate,
            "enabled": self.enabled
        }


class BaseAttentionHook(BaseHook):
    """Base class for attention weight tracking hooks."""
    
    def __init__(self, name: str = "attention_hook", store_all: bool = False):
        super().__init__(name)
        self.attention_weights = defaultdict(list) if store_all else {}
        self.store_all = store_all
    
    def forward_hook_fn(self, module_name: str):
        """Create forward hook for attention weight extraction."""
        def hook(module, input, output):
            if not self.enabled:
                return None
            
            # Attention weights are typically the second element in output
            # Format varies by architecture, needs model-specific handling
            if isinstance(output, tuple) and len(output) >1:
                attn = output[1]
                if attn is not None:
                    attn_weights = attn.detach().cpu()
                    if self.store_all:
                        self.attention_weights[module_name].append(attn_weights)
                    else:
                        self.attention_weights[module_name] = attn_weights
            
            return None
        return hook
    
    def attach(self, module: nn.Module, module_name: str = "module"):
        """Attach forward hook to attention layer."""
        handle = module.register_forward_hook(
            self.forward_hook_fn(module_name)
        )
        self.handles.append(handle)
    
    def get_results(self) -> Dict[str, Any]:
        """Get attention weights."""
        return dict(self.attention_weights)
    
    def reset(self):
        """Reset attention weight storage."""
        if self.store_all:
            self.attention_weights = defaultdict(list)
        else:
            self.attention_weights.clear()


class HookManager:
    """Manager for coordinating multiple hooks."""
    
    def __init__(self):
        self.hooks: Dict[str, BaseHook] = {}
    
    def register_hook(self, hook: BaseHook):
        """Register a hook with the manager."""
        self.hooks[hook.name] = hook
    
    def remove_hook(self, name: str):
        """Remove a hook by name."""
        if name in self.hooks:
            self.hooks[name].remove()
            del self.hooks[name]
    
    def remove_all(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
    
    def enable_all(self):
        """Enable all hooks."""
        for hook in self.hooks.values():
            hook.enable()
    
    def disable_all(self):
        """Disable all hooks."""
        for hook in self.hooks.values():
            hook.disable()
    
    def reset_all(self):
        """Reset all hooks."""
        for hook in self.hooks.values():
            hook.reset()
    
    def get_all_results(self) -> Dict[str, Any]:
        """Get results from all hooks."""
        return {name: hook.get_results() for name, hook in self.hooks.items()}
