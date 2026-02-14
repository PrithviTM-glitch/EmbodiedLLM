"""
RDT-1B Model-Specific Hooks

Adapter for attaching hooks to RDT-1B (1.2B) model.
RDT has an MLP proprioceptive encoder with Fourier features.

Architecture:
- Vision: ViT or CNN backbone
- Proprio: Fourier features → MLP (multiple layers) → Concatenate
- Language: Pre-trained language model
- Fusion: Concatenate all modalities at transformer input
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from ..gradient_hooks import GradientFlowAnalyzer
from ..representation_hooks import RepresentationQualityAnalyzer
from ..ablation_hooks import AblationStudyCoordinator
from ..utilization_hooks import DownstreamUtilizationAnalyzer


class RDTHooks:
    """
    Hook adapter for RDT-1B model.
    
    RDT uses MLP state encoding with Fourier features:
    - State → Fourier features → MLP (2-3 layers) → Concat to transformer
    """
    
    def __init__(self, model):
        """
        Initialize hook adapter.
        
        Args:
            model: RDT-1B model instance
        """
        self.model = model
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.representation_analyzer = RepresentationQualityAnalyzer()
        self.ablation_coordinator = AblationStudyCoordinator()
        self.utilization_analyzer = DownstreamUtilizationAnalyzer()
        
        # Model components
        self.vision_encoder = None
        self.proprio_encoder = None  # state_adaptor (single Linear layer in RDT-1B)
        self.language_encoder = None
        self.transformer_layers = None
    
    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover RDT model structure.
        
        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "RDT-1B",
            "has_proprio_encoder": True,
            "proprio_encoder_type": "linear",  # Single Linear layer (state_adaptor)
            "components": {}
        }
        
        # Vision encoder
        for attr in ['vision_encoder', 'image_encoder', 'visual_backbone', 'backbone']:
            if hasattr(self.model, attr):
                self.vision_encoder = getattr(self.model, attr)
                structure["components"]["vision_encoder"] = attr
                break
        
        # Proprio encoder (state_adaptor - single Linear layer in RDT-1B)
        # Priority: state_adaptor (actual RDT-1B name), then fallbacks
        for attr in ['state_adaptor', 'proprio_encoder', 'state_encoder', 'state_mlp', 'robot_state_encoder']:
            if hasattr(self.model, attr):
                self.proprio_encoder = getattr(self.model, attr)
                structure["components"]["proprio_encoder"] = attr
                
                # Check if it's a Linear layer
                if isinstance(self.proprio_encoder, nn.Linear):
                    structure["proprio_encoder_architecture"] = "single_linear"
                    structure["proprio_input_dim"] = self.proprio_encoder.in_features
                    structure["proprio_output_dim"] = self.proprio_encoder.out_features
                break
        
        # Language encoder
        for attr in ['language_encoder', 'text_encoder', 'language_model']:
            if hasattr(self.model, attr):
                self.language_encoder = getattr(self.model, attr)
                structure["components"]["language_encoder"] = attr
                break
        
        # Transformer layers
        for attr in ['transformer', 'backbone', 'model', 'encoder']:
            if hasattr(self.model, attr):
                transformer = getattr(self.model, attr)
                if hasattr(transformer, 'layers') or hasattr(transformer, 'blocks'):
                    self.transformer_layers = getattr(transformer, 'layers', None) or getattr(transformer, 'blocks')
                    structure["components"]["transformer"] = f"{attr}.layers"
                    structure["num_transformer_layers"] = len(self.transformer_layers)
                    break
        
        return structure
    
    def attach_gradient_hooks(self):
        """Attach gradient flow analysis hooks."""
        if self.proprio_encoder is None:
            self.discover_model_structure()
        
        # Attach encoder-level tracking
        self.gradient_analyzer.setup_encoder_tracking(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder
        )
        
        # Profile the state_adaptor (single Linear layer in RDT-1B)
        if self.proprio_encoder:
            self.gradient_analyzer.setup_layer_profiling(
                "state_adaptor",
                [self.proprio_encoder],
                ["state_adaptor"]
            )
        
        # Profile transformer layers to see how gradients flow back
        if self.transformer_layers:
            transformer_sample = list(self.transformer_layers)[:6]
            self.gradient_analyzer.setup_layer_profiling(
                "transformer",
                transformer_sample,
                [f"transformer_layer_{i}" for i in range(len(transformer_sample))]
            )
    
    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.proprio_encoder is None:
            self.discover_model_structure()
        
        # Standard setup for encoders
        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder
        )
        
        # Extract features from the state_adaptor output
        if self.proprio_encoder:
            self.representation_analyzer.feature_extractor.attach(
                self.proprio_encoder,
                name="state_adaptor_output"
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
        
        # ADDITIONAL: Ablate individual MLP layers
        # Test: Does removing Fourier features hurt more than removing final MLP layer?
        if self.fourier_layer:
            self.ablation_coordinator.ablation_manager.register_encoder(
                "fourier_only",
                self.fourier_layer
            )
        
        if self.mlp_layers and len(self.mlp_layers) > 1:
            # Register final MLP layer
            final_name, final_layer = self.mlp_layers[-1]
            self.ablation_coordinator.ablation_manager.register_encoder(
                "final_mlp_layer",
                final_layer
            )
    
    def attach_utilization_hooks(self):
        """Attach downstream utilization tracking hooks."""
        if self.transformer_layers is None:
            self.discover_model_structure()
        
        # Attach attention tracking to transformer self-attention
        if self.transformer_layers:
            for i, layer in enumerate(list(self.transformer_layers)[:8]):
                # Find self-attention module
                attn_module = None
                if hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                elif hasattr(layer, 'attn'):
                    attn_module = layer.attn
                elif hasattr(layer, 'attention'):
                    attn_module = layer.attention
                
                if attn_module:
                    self.utilization_analyzer.attention_tracker.attach(
                        attn_module,
                        name=f"transformer_attn_{i}"
                    )
        
        # Set modality ranges
        modality_ranges = self.get_modality_token_ranges()
        self.utilization_analyzer.attention_tracker.set_modality_ranges(modality_ranges)
        
        # Feature similarity across MLP layers
        if self.mlp_layers:
            for name, layer in self.mlp_layers:
                self.utilization_analyzer.similarity_tracker.attach(
                    layer,
                    name=f"mlp_{name}"
                )
    
    def get_modality_token_ranges(self) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality in RDT.
        
        RDT concatenates: [vision_tokens, proprio_tokens, language_tokens]
        
        Returns:
            Dict with (start, end) ranges
        """
        # Typical RDT configuration:
        # Vision: ViT patches (e.g., 196 for 14x14)
        # Proprio: MLP output (e.g., 8-16 tokens)
        # Language: Variable length instruction
        
        vision_tokens = 196
        proprio_tokens = 8  # MLP often outputs a sequence
        
        ranges = {
            "vision": (0, vision_tokens),
            "proprio": (vision_tokens, vision_tokens + proprio_tokens),
            "language": (vision_tokens + proprio_tokens, -1)
        }
        
        return ranges
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """
        Run complete diagnostic analysis for RDT.
        
        Returns:
            Comprehensive diagnostic report
        """
        # Discover structure
        structure = self.discover_model_structure()
        
        # Attach all hooks
        self.attach_gradient_hooks()
        self.attach_representation_hooks()
        self.attach_ablation_hooks()
        self.attach_utilization_hooks()
        
        report = {
            "model_structure": structure,
            "note": "RDT uses MLP+Fourier state encoder - intermediate complexity",
            "analysis_focus": [
                "Layer-wise gradient flow through MLP (Fourier → Layer1 → Layer2 → ...)",
                "Representation quality at each MLP layer",
                "CKA similarity: Does MLP add structure beyond Fourier?",
                "Ablation: Fourier-only vs Full MLP vs No-proprio",
                "Attention to proprio tokens vs vision tokens"
            ],
            "key_questions": [
                "Do gradients vanish in MLP layers?",
                "Does each MLP layer add meaningful representations?",
                "Are Fourier features sufficient, or does MLP matter?",
                "How much attention do proprio tokens receive?"
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
        
        # Gradient analysis - focus on MLP layer-wise
        if self.gradient_analyzer.layer_profilers:
            mlp_profiler = self.gradient_analyzer.layer_profilers.get("proprio_mlp")
            if mlp_profiler:
                summary["mlp_gradients"] = {
                    "layer_norms": mlp_profiler.get_summary(),
                    "vanishing_point": mlp_profiler.find_vanishing_point()
                }
        
        # Representation at each MLP layer
        if self.representation_analyzer.feature_extractor.features:
            mlp_features = {
                name: feats 
                for name, feats in self.representation_analyzer.feature_extractor.features.items()
                if "mlp" in name.lower()
            }
            if mlp_features:
                summary["mlp_representations"] = list(mlp_features.keys())
        
        # Overall encoder comparison
        summary["encoder_comparison"] = self.gradient_analyzer.get_comprehensive_report()
        
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
        print("RDT-1B MODEL STRUCTURE")
        print("=" * 80)
        print(f"Model: {structure['model_name']}")
        print(f"Has Proprioceptive Encoder: {structure['has_proprio_encoder']}")
        print(f"Proprio Encoder Type: {structure.get('proprio_encoder_type', 'unknown')}")
        
        print("\nDiscovered Components:")
        for comp_name, attr_name in structure["components"].items():
            print(f"  - {comp_name}: model.{attr_name}")
        
        print(f"\nProprio Encoder Details:")
        print(f"  - Has Fourier Features: {structure.get('has_fourier', False)}")
        print(f"  - MLP Layers: {structure.get('mlp_layers', 0)}")
        
        if self.mlp_layers:
            print("\n  MLP Layer Structure:")
            for name, layer in self.mlp_layers:
                if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                    print(f"    - {name}: {layer.in_features} → {layer.out_features}")
        
        if self.transformer_layers:
            print(f"\nTransformer Layers: {len(self.transformer_layers)}")
        
        print("\nFusion Mechanism: Concatenation at transformer input")
        print("Key Feature: Fourier features for physical state representation")
        print("=" * 80)
