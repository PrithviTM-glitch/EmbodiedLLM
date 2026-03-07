"""
Octo Model-Specific Hooks

Adapter for attaching hooks to Octo (93M) model.
Octo has a LINEAR proprioceptive encoder (simplest case).

Architecture:
- Vision: ResNet-34 or ViT
- Proprio: Linear projection + position embeddings
- Fusion: Concatenate at transformer input
- Conditioning: Diffusion-based with cross-attention
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from ..gradient_hooks import GradientFlowAnalyzer
from ..representation_hooks import RepresentationQualityAnalyzer
from ..ablation_hooks import AblationStudyCoordinator
from ..utilization_hooks import DownstreamUtilizationAnalyzer


class OctoHooks:
    """
    Hook adapter for Octo model.
    
    Octo uses LINEAR state encoding:
    - State → Linear projection → Add position embeddings → Concat to transformer
    """
    
    def __init__(self, model):
        """
        Initialize hook adapter.
        
        Args:
            model: Octo model instance
        """
        self.model = model
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.representation_analyzer = RepresentationQualityAnalyzer()
        self.ablation_coordinator = AblationStudyCoordinator()
        self.utilization_analyzer = DownstreamUtilizationAnalyzer()
        
        # Model components
        self.vision_encoder = None
        self.proprio_encoder = None  # Linear layer
        self.language_encoder = None
        self.transformer_layers = None
        self.cross_attention_layers = None
    
    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover Octo model structure.
        
        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "Octo",
            "has_proprio_encoder": True,
            "proprio_encoder_type": "linear",
            "components": {}
        }
        
        # Vision encoder (ResNet or ViT)
        for attr in ['vision_encoder', 'image_encoder', 'visual_backbone']:
            if hasattr(self.model, attr):
                self.vision_encoder = getattr(self.model, attr)
                structure["components"]["vision_encoder"] = attr
                break
        
        # Proprio encoder (linear projection)
        # Common names: proprio_encoder, state_encoder, state_projection
        for attr in ['proprio_encoder', 'state_encoder', 'state_projection', 'proprio_projection']:
            if hasattr(self.model, attr):
                self.proprio_encoder = getattr(self.model, attr)
                structure["components"]["proprio_encoder"] = attr
                structure["proprio_encoder_type"] = self._identify_encoder_type(self.proprio_encoder)
                break
        
        # Language encoder (for task conditioning)
        for attr in ['language_encoder', 'text_encoder', 'task_encoder']:
            if hasattr(self.model, attr):
                self.language_encoder = getattr(self.model, attr)
                structure["components"]["language_encoder"] = attr
                break
        
        # Transformer layers
        for attr in ['transformer', 'backbone', 'model']:
            if hasattr(self.model, attr):
                transformer = getattr(self.model, attr)
                if hasattr(transformer, 'layers'):
                    self.transformer_layers = transformer.layers
                    structure["components"]["transformer_layers"] = f"{attr}.layers"
                    break
        
        # Cross-attention layers (for diffusion conditioning)
        if self.transformer_layers is not None:
            self.cross_attention_layers = []
            for i, layer in enumerate(self.transformer_layers):
                if hasattr(layer, 'cross_attn') or hasattr(layer, 'cross_attention'):
                    self.cross_attention_layers.append((i, layer))
            structure["num_cross_attention_layers"] = len(self.cross_attention_layers)
        
        return structure
    
    def _identify_encoder_type(self, encoder: nn.Module) -> str:
        """Identify the type of encoder (linear, mlp, etc)."""
        if isinstance(encoder, nn.Linear):
            return "linear"
        elif isinstance(encoder, nn.Sequential):
            # Check if it's just linear + position embedding
            has_linear = any(isinstance(m, nn.Linear) for m in encoder.children())
            has_mlp = sum(isinstance(m, nn.Linear) for m in encoder.children()) > 1
            return "mlp" if has_mlp else "linear+embedding"
        else:
            return str(type(encoder).__name__)
    
    def attach_gradient_hooks(self):
        """Attach gradient flow analysis hooks."""
        if self.vision_encoder is None or self.proprio_encoder is None:
            self.discover_model_structure()
        
        # Attach encoder-level tracking (all three modalities)
        self.gradient_analyzer.setup_encoder_tracking(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder
        )
        
        # For linear encoder, we don't need layer-wise profiling (only 1 layer)
        # But let's profile transformer layers to see gradient flow
        if self.transformer_layers is not None:
            # Profile first 6 transformer layers
            layers_to_profile = list(self.transformer_layers)[:6]
            self.gradient_analyzer.setup_layer_profiling(
                "transformer",
                layers_to_profile,
                [f"transformer_layer_{i}" for i in range(len(layers_to_profile))]
            )
    
    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.proprio_encoder is None:
            self.discover_model_structure()
        
        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder
        )
    
    def attach_ablation_hooks(self):
        """Attach ablation hooks."""
        if self.proprio_encoder is None:
            self.discover_model_structure()
        
        self.ablation_coordinator.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder,
            ablation_type="zero"
        )
    
    def attach_utilization_hooks(self):
        """Attach downstream utilization tracking hooks."""
        if self.transformer_layers is None:
            self.discover_model_structure()
        
        # Setup attention tracking on cross-attention layers
        if self.cross_attention_layers:
            for layer_idx, layer in self.cross_attention_layers:
                attn_module = getattr(layer, 'cross_attn', None) or getattr(layer, 'cross_attention', None)
                if attn_module:
                    self.utilization_analyzer.attention_tracker.attach(
                        attn_module,
                        name=f"cross_attn_layer_{layer_idx}"
                    )
        
        # Set modality ranges for attention tracking
        modality_ranges = self.get_modality_token_ranges()
        self.utilization_analyzer.attention_tracker.set_modality_ranges(modality_ranges)
        
        # Attach feature similarity tracking across transformer
        if self.transformer_layers:
            layers_to_track = list(self.transformer_layers)[:8]
            for i, layer in enumerate(layers_to_track):
                self.utilization_analyzer.similarity_tracker.attach(
                    layer,
                    name=f"transformer_layer_{i}"
                )
    
    def get_modality_token_ranges(self) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality in Octo.
        
        Octo concatenates: [vision_tokens, proprio_tokens, language_tokens]
        Exact numbers depend on input configuration.
        
        Returns:
            Dict with (start, end) ranges
        """
        # Default Octo configuration (may vary):
        # Vision: ResNet features or ViT patches (e.g., 49 for 7x7, 256 for 16x16)
        # Proprio: Typically 7-14 dimensions (joint positions + gripper state)
        # Language: Variable length task description
        
        # Common default: 196 vision tokens (14x14), 1 proprio token (projected), variable language
        vision_tokens = 196
        proprio_tokens = 1  # Linear projection often outputs single token
        
        ranges = {
            "vision": (0, vision_tokens),
            "proprio": (vision_tokens, vision_tokens + proprio_tokens),
            "language": (vision_tokens + proprio_tokens, -1)
        }
        
        return ranges
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """
        Run complete diagnostic analysis for Octo.
        
        Returns:
            Comprehensive diagnostic report
        """
        # Discover structure
        structure = self.discover_model_structure()
        
        # Attach all hooks
        self.attach_gradient_hooks()
        self.attach_representation_hooks()
        self.attach_utilization_hooks()
        
        report = {
            "model_structure": structure,
            "note": "Octo uses linear state encoder - simplest proprioceptive encoding",
            "analysis_focus": [
                "Gradient flow to linear proprio projection",
                "CKA similarity: vision vs proprio features",
                "Cross-attention weights to proprio tokens",
                "Ablation impact: removing single proprio token"
            ]
        }
        
        return report
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """
        Get summary of diagnostic results.
        
        Returns:
            Dict with analysis summaries
        """
        summary = {}
        
        # Gradient analysis
        if self.gradient_analyzer.encoder_tracker.results:
            summary["gradients"] = self.gradient_analyzer.get_comprehensive_report()
        
        # Representation analysis
        if self.representation_analyzer.feature_extractor.features:
            summary["representations"] = {
                "cka": self.representation_analyzer.cka_analyzer.get_similarity_matrix(),
                "effective_rank": {
                    name: calc.get_summary()
                    for name, calc in self.representation_analyzer.rank_calculators.items()
                }
            }
        
        # Attention utilization
        if self.utilization_analyzer.attention_tracker.results:
            summary["attention"] = self.utilization_analyzer.attention_tracker.compute_modality_attention()
        
        return summary
    
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
        print("OCTO MODEL STRUCTURE")
        print("=" * 80)
        print(f"Model: {structure['model_name']}")
        print(f"Has Proprioceptive Encoder: {structure['has_proprio_encoder']}")
        print(f"Proprio Encoder Type: {structure.get('proprio_encoder_type', 'unknown')}")
        
        print("\nDiscovered Components:")
        for comp_name, attr_name in structure["components"].items():
            print(f"  - {comp_name}: model.{attr_name}")
        
        if self.transformer_layers:
            print(f"\nTransformer Layers: {len(self.transformer_layers)}")
        if self.cross_attention_layers:
            print(f"Cross-Attention Layers: {len(self.cross_attention_layers)}")
        
        print("\nFusion Mechanism: Concatenation at transformer input")
        print("Conditioning: Diffusion-based with cross-attention")
        print("=" * 80)
