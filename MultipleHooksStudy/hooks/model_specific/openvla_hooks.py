"""
OpenVLA Model-Specific Hooks

Adapter for attaching hooks to OpenVLA (7B) model.
OpenVLA has NO proprioceptive encoder - only vision and language.

Architecture:
- Vision: SigLIP-400M (vision encoder)
- Language: Llama-2-7B (language encoder)
- Fusion: Vision-as-prefix (vision tokens prepended to language)
- No proprioceptive state input
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from ..gradient_hooks import GradientFlowAnalyzer
from ..representation_hooks import RepresentationQualityAnalyzer
from ..ablation_hooks import AblationStudyCoordinator  
from ..utilization_hooks import DownstreamUtilizationAnalyzer


class OpenVLAHooks:
    """
    Hook adapter for OpenVLA model.
    
    NOTE: OpenVLA does not have a proprioceptive encoder.
    This adapter focuses on vision and language encoders only.
    """
    
    def __init__(self, model):
        """
        Initialize hook adapter.
        
        Args:
            model: OpenVLA model instance
        """
        self.model = model
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.representation_analyzer = RepresentationQualityAnalyzer()
        self.ablation_coordinator = AblationStudyCoordinator()
        self.utilization_analyzer = DownstreamUtilizationAnalyzer()
        
        # Model components (to be discovered)
        self.vision_encoder = None
        self.language_encoder = None
        self.fusion_layer = None
    
    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover OpenVLA model structure.
        
        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "OpenVLA",
            "has_proprio_encoder": False,
            "components": {}
        }
        
        # Try to find vision encoder (SigLIP)
        # PrismaticVLM (actual OpenVLA) uses 'vision_backbone', others may use vision_encoder, etc.
        for attr in ['vision_backbone', 'vision_encoder', 'vision_model', 'image_encoder', 'visual_encoder']:
            if hasattr(self.model, attr):
                self.vision_encoder = getattr(self.model, attr)
                structure["components"]["vision_encoder"] = attr
                break
        
        # Try to find language encoder (Llama)
        # PrismaticVLM (actual OpenVLA) uses 'llm_backbone', others may use language_model, etc.
        for attr in ['llm_backbone', 'language_model', 'llm', 'text_encoder', 'language_encoder']:
            if hasattr(self.model, attr):
                self.language_encoder = getattr(self.model, attr)
                structure["components"]["language_encoder"] = attr
                break
        
        # Try to find fusion mechanism
        for attr in ['projector', 'connector', 'vision_projector']:
            if hasattr(self.model, attr):
                self.fusion_layer = getattr(self.model, attr)
                structure["components"]["fusion_layer"] = attr
                break
        
        return structure
    
    def attach_gradient_hooks(self):
        """Attach gradient flow analysis hooks."""
        if self.vision_encoder is None or self.language_encoder is None:
            self.discover_model_structure()
        
        # Attach encoder-level tracking
        self.gradient_analyzer.setup_encoder_tracking(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=None  # OpenVLA has no proprio encoder
        )
        
        # Attach layer-wise profiling for vision encoder
        if self.vision_encoder is not None:
            vision_layers = self._get_vision_layers()
            if vision_layers:
                self.gradient_analyzer.setup_layer_profiling(
                    "vision_encoder",
                    vision_layers,
                    [f"vision_layer_{i}" for i in range(len(vision_layers))]
                )
        
        # Attach layer-wise profiling for language encoder
        if self.language_encoder is not None:
            language_layers = self._get_language_layers()
            if language_layers:
                self.gradient_analyzer.setup_layer_profiling(
                    "language_encoder",
                    language_layers[:6],  # First 6 layers only (7B is too large)
                    [f"language_layer_{i}" for i in range(min(6, len(language_layers)))]
                )
    
    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.vision_encoder is None or self.language_encoder is None:
            self.discover_model_structure()
        
        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=None  # No proprio encoder
        )
    
    def attach_ablation_hooks(self):
        """Attach ablation hooks."""
        if self.vision_encoder is None or self.language_encoder is None:
            self.discover_model_structure()
        
        self.ablation_coordinator.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=None,  # No proprio encoder
            ablation_type="zero"
        )
    
    def _get_vision_layers(self) -> List[nn.Module]:
        """Get vision encoder layers."""
        if self.vision_encoder is None:
            return []
        
        # Try common layer attributes
        for attr in ['layers', 'encoder', 'blocks', 'transformer']:
            if hasattr(self.vision_encoder, attr):
                layers_obj = getattr(self.vision_encoder, attr)
                if hasattr(layers_obj, 'layers'):
                    return list(layers_obj.layers)
                elif isinstance(layers_obj, nn.ModuleList):
                    return list(layers_obj)
        
        return []
    
    def _get_language_layers(self) -> List[nn.Module]:
        """Get language encoder layers."""
        if self.language_encoder is None:
            return []
        
        # Llama-style models typically have model.layers
        if hasattr(self.language_encoder, 'model'):
            if hasattr(self.language_encoder.model, 'layers'):
                return list(self.language_encoder.model.layers)
        
        if hasattr(self.language_encoder, 'layers'):
            return list(self.language_encoder.layers)
        
        return []
    
    def get_modality_token_ranges(self, input_ids=None) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality.
        
        In OpenVLA, vision tokens are prepended to language tokens.
        
        Returns:
            Dict with (start, end) ranges for vision and language
        """
        # Vision tokens typically come first (e.g., 0-255)
        # Language tokens follow (e.g., 256-end)
        # Exact numbers depend on input
        
        # Default assumption: 256 vision tokens (16x16 patches)
        vision_tokens = 256
        
        ranges = {
            "vision": (0, vision_tokens),
            "language": (vision_tokens, -1)  # -1 means "to end"
        }
        
        return ranges
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """
        Run complete diagnostic analysis.
        
        Returns:
            Comprehensive diagnostic report
        """
        # Discover structure
        structure = self.discover_model_structure()
        
        # Attach hooks
        self.attach_gradient_hooks()
        self.attach_representation_hooks()
        
        report = {
            "model_structure": structure,
            "note": "OpenVLA has no proprioceptive encoder - analysis focuses on vision/language only"
        }
        
        return report
    
    def cleanup(self):
        """Remove all hooks."""
        self.gradient_analyzer.remove_all()
        self.representation_analyzer.remove_hooks()
        self.ablation_coordinator.cleanup()
        self.utilization_analyzer.cleanup()
    
    def print_model_info(self):
        """Print human-readable model information."""
        structure = self.discover_model_structure()
        
        print("=" * 80)
        print("OpenVLA MODEL STRUCTURE")
        print("=" * 80)
        print(f"Model: {structure['model_name']}")
        print(f"Has Proprioceptive Encoder: {structure['has_proprio_encoder']}")
        print("\nDiscovered Components:")
        for comp_name, attr_name in structure["components"].items():
            print(f"  - {comp_name}: model.{attr_name}")
        
        print("\nVision Encoder Layers:", len(self._get_vision_layers()))
        print("Language Encoder Layers:", len(self._get_language_layers()))
        print("\nNote: OpenVLA uses vision-as-prefix fusion (no separate fusion layer)")
        print("=" * 80)
