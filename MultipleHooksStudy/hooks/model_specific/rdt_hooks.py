"""
RDT-1B Model-Specific Hooks

Adapter for attaching hooks to RDT-1B (1.2B) model.
RDT uses a single Linear layer (state_adaptor) for state encoding.

Architecture:
- Vision: SigLIP encoder
- State: state_adaptor (Linear layer, input_dim = state_token_dim * 2 for state + mask)
- Language: T5-XXL encoder
- Fusion: Concatenate all modalities at diffusion transformer input
- Action: Diffusion-based action prediction
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
    
    RDT uses a single Linear layer for state encoding:
    - State (+ mask) → state_adaptor (Linear) → Concat to diffusion transformer
    - Note: state_adaptor input_dim = state_token_dim * 2 (state + mask indicator)
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
        
        # Note: Input dimension is state_token_dim * 2 (state + mask concatenated)
        for attr in ['state_adaptor', 'proprio_encoder', 'state_encoder', 'robot_state_encoder']:
            if hasattr(self.model, attr):
                self.proprio_encoder = getattr(self.model, attr)
                structure["components"]["proprio_encoder"] = attr
                
                # Check if it's a Linear layer
                if isinstance(self.proprio_encoder, nn.Linear):
                    structure["proprio_encoder_architecture"] = "single_linear"
                    structure["proprio_input_dim"] = self.proprio_encoder.in_features
                    structure["proprio_output_dim"] = self.proprio_encoder.out_features
                    structure["note"] = "Input includes state + mask (dim = state_token_dim * 2)"
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
    
    def get_modality_token_ranges(self) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality in RDT.
        
        RDT concatenates: [vision_tokens, state_tokens, language_tokens]
        
        Returns:
            Dict with (start, end) ranges
        """
        # RDT-1B configuration:
        # Vision: SigLIP patches (typically 256 patches)
        # State: state_adaptor output (1 token or few tokens)
        # Language: T5-XXL encoded instruction tokens
        
        vision_tokens = 256
        state_tokens = 1  # Single state token from state_adaptor
        
        ranges = {
            "vision": (0, vision_tokens),
            "state": (vision_tokens, vision_tokens + state_tokens),
            "language": (vision_tokens + state_tokens, -1)
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
            "note": "RDT-1B uses single Linear layer (state_adaptor) for state encoding",
            "analysis_focus": [
                "Gradient flow through state_adaptor layer",
                "Representation quality of state encoding",
                "CKA: State vs Vision features",
                "Attention patterns in diffusion transformer",
                "Ablation: With state vs Without state"
            ],
            "key_questions": [
                "Does state_adaptor learn meaningful representations?",
                "How does diffusion transformer utilize state features?",
                "Is state→action coupling stronger than vision→action?",
                "What happens when state input is zeroed?"
            ],
            "architectural_features": {
                "vision_encoder": "SigLIP",
                "state_encoder": "Single Linear layer (state_adaptor)",
                "language_encoder": "T5-XXL",
                "action_generation": "Diffusion-based",
                "note": "state_adaptor input_dim = state_token_dim * 2 (state + mask)"
            }
        }
        
        return report
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """
        Get summary of diagnostic results.
        
        Returns:
            Dict with analysis summaries
        """
        summary = {}
        
        # State adaptor gradients
        if self.gradient_analyzer.layer_profilers:
            state_profiler = self.gradient_analyzer.layer_profilers.get("state_adaptor")
            if state_profiler:
                summary["state_adaptor_gradients"] = state_profiler.get_summary()
            
            transformer_profiler = self.gradient_analyzer.layer_profilers.get("transformer")
            if transformer_profiler:
                summary["transformer_gradients"] = transformer_profiler.get_summary()
        
        # State representation features
        if self.representation_analyzer.feature_extractor.features:
            state_features = {
                name: feats 
                for name, feats in self.representation_analyzer.feature_extractor.features.items()
                if "state_adaptor" in name
            }
            if state_features:
                summary["state_representations"] = list(state_features.keys())
        
        # Attention patterns
        if self.utilization_analyzer.attention_tracker.results:
            summary["attention_patterns"] = self.utilization_analyzer.attention_tracker.compute_modality_attention()
        
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
        print(f"Has State Encoder: {structure['has_proprio_encoder']}")
        print(f"State Encoder Type: {structure.get('proprio_encoder_type', 'unknown')}")
        
        print("\nDiscovered Components:")
        for comp_name, attr_name in structure["components"].items():
            print(f"  - {comp_name}: model.{attr_name}")
        
        print(f"\nState Encoder Details:")
        print(f"  - Type: Single Linear layer (state_adaptor)")
        print(f"  - Architecture: Linear projection to transformer embedding space")
        if 'proprio_input_dim' in structure and 'proprio_output_dim' in structure:
            print(f"  - Dimensions: {structure['proprio_input_dim']} → {structure['proprio_output_dim']}")
        if structure.get('note'):
            print(f"  - Note: {structure['note']}")
        
        if self.transformer_layers:
            print(f"\nDiffusion Transformer Layers: {len(self.transformer_layers)}")
        
        print("\nArchitectural Features:")
        print("  - Vision: SigLIP encoder")
        print("  - State: Linear projection (state + mask)")
        print("  - Language: T5-XXL encoder")
        print("  - Action Generation: Diffusion-based prediction")
        print("  - Model: RDT-1B (1.2B parameters)")
        print("\nFusion: Concatenation at diffusion transformer input")
        print("Key Feature: Lightweight state encoding via single projection layer")
        print("=" * 80)
