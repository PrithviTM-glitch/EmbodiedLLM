"""
π0 (Pi0) Model-Specific Hooks

Adapter for attaching hooks to π0 (3.3B) model.
π0 has the most advanced proprioceptive encoding with separate encoder and causal masking.

Architecture:
- Vision: Pre-trained vision transformer
- Proprio: Separate dedicated encoder with multiple layers
- Language: Pre-trained language model  
- Fusion: Block-wise causal masking, asymmetric conditioning
- Action: Flow matching architecture
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from ..gradient_hooks import GradientFlowAnalyzer
from ..representation_hooks import RepresentationQualityAnalyzer
from ..ablation_hooks import AblationStudyCoordinator
from ..utilization_hooks import DownstreamUtilizationAnalyzer


class Pi0Hooks:
    """
    Hook adapter for π0 (Pi0) model.
    
    π0 uses ADVANCED state encoding:
    - State → Separate encoder (multiple layers) → Block-wise causal masking
    - Asymmetric conditioning of action on state
    - Flow matching for action generation
    """
    
    def __init__(self, model):
        """
        Initialize hook adapter.
        
        Args:
            model: π0 model instance
        """
        self.model = model
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.representation_analyzer = RepresentationQualityAnalyzer()
        self.ablation_coordinator = AblationStudyCoordinator()
        self.utilization_analyzer = DownstreamUtilizationAnalyzer()
        
        # Model components
        self.vision_encoder = None
        self.proprio_encoder = None  # Separate multi-layer encoder
        self.proprio_encoder_layers = None
        self.language_encoder = None
        self.causal_attention_layers = None  # Layers with causal masking
        self.flow_matching_layers = None
        self.transformer_layers = None
    
    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover π0 model structure.
        
        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "π0 (Pi0)",
            "has_proprio_encoder": True,
            "proprio_encoder_type": "separate+causal",
            "components": {}
        }
        
        # Vision encoder
        for attr in ['vision_encoder', 'image_encoder', 'visual_encoder', 'vision_backbone']:
            if hasattr(self.model, attr):
                self.vision_encoder = getattr(self.model, attr)
                structure["components"]["vision_encoder"] = attr
                break
        
        # Proprio encoder (separate multi-layer)
        for attr in ['proprio_encoder', 'state_encoder', 'robot_encoder', 'physical_state_encoder']:
            if hasattr(self.model, attr):
                self.proprio_encoder = getattr(self.model, attr)
                structure["components"]["proprio_encoder"] = attr
                
                # Extract layers from encoder
                self._extract_encoder_layers()
                structure["proprio_encoder_layers"] = len(self.proprio_encoder_layers) if self.proprio_encoder_layers else 0
                break
        
        # Language encoder
        for attr in ['language_encoder', 'text_encoder', 'language_model', 'instruction_encoder']:
            if hasattr(self.model, attr):
                self.language_encoder = getattr(self.model, attr)
                structure["components"]["language_encoder"] = attr
                break
        
        # Main transformer/backbone
        for attr in ['transformer', 'backbone', 'model', 'decoder']:
            if hasattr(self.model, attr):
                transformer = getattr(self.model, attr)
                if hasattr(transformer, 'layers') or hasattr(transformer, 'blocks'):
                    self.transformer_layers = getattr(transformer, 'layers', None) or getattr(transformer, 'blocks')
                    structure["components"]["transformer"] = f"{attr}.layers"
                    structure["num_transformer_layers"] = len(self.transformer_layers)
                    
                    # Find causal attention layers
                    self._find_causal_attention_layers()
                    structure["num_causal_layers"] = len(self.causal_attention_layers) if self.causal_attention_layers else 0
                    break
        
        # Flow matching components
        for attr in ['flow_matcher', 'action_head', 'flow_network']:
            if hasattr(self.model, attr):
                self.flow_matching_layers = getattr(self.model, attr)
                structure["components"]["flow_matching"] = attr
                break
        
        return structure
    
    def _extract_encoder_layers(self):
        """Extract individual layers from separate proprio encoder."""
        if self.proprio_encoder is None:
            return
        
        self.proprio_encoder_layers = []
        
        # Check for transformer-style layers
        if hasattr(self.proprio_encoder, 'layers'):
            self.proprio_encoder_layers = list(self.proprio_encoder.layers)
        elif hasattr(self.proprio_encoder, 'blocks'):
            self.proprio_encoder_layers = list(self.proprio_encoder.blocks)
        # Check if it's a Sequential
        elif isinstance(self.proprio_encoder, nn.Sequential):
            self.proprio_encoder_layers = list(self.proprio_encoder.children())
        else:
            # Try to find transformer blocks in encoder
            for name, module in self.proprio_encoder.named_modules():
                if 'layer' in name.lower() or 'block' in name.lower():
                    # Avoid duplicates - only add direct children
                    if '.' not in name or name.count('.') == 1:
                        self.proprio_encoder_layers.append(module)
    
    def _find_causal_attention_layers(self):
        """Find transformer layers that use causal masking."""
        if self.transformer_layers is None:
            return
        
        self.causal_attention_layers = []
        
        for i, layer in enumerate(self.transformer_layers):
            # Check for causal attention indicators
            has_causal = False
            
            # Check layer attributes
            if hasattr(layer, 'causal') and layer.causal:
                has_causal = True
            elif hasattr(layer, 'is_causal') and layer.is_causal:
                has_causal = True
            
            # Check attention module
            attn = None
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
            elif hasattr(layer, 'attn'):
                attn = layer.attn
            
            if attn:
                if hasattr(attn, 'causal') and attn.causal:
                    has_causal = True
                elif hasattr(attn, 'is_causal') and attn.is_causal:
                    has_causal = True
            
            if has_causal:
                self.causal_attention_layers.append((i, layer))
        
        # If no explicit causal markers, assume later layers use causal
        # (π0 uses block-wise causal masking)
        if not self.causal_attention_layers and self.transformer_layers:
            # Assume second half of layers use causal masking
            mid_point = len(self.transformer_layers) // 2
            self.causal_attention_layers = [
                (i, layer) 
                for i, layer in enumerate(self.transformer_layers[mid_point:], start=mid_point)
            ]
    
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
        
        # CRITICAL: Layer-wise profiling of separate proprio encoder
        # This is key for π0 - see gradient flow through each encoder layer
        if self.proprio_encoder_layers:
            self.gradient_analyzer.setup_layer_profiling(
                "proprio_encoder",
                self.proprio_encoder_layers,
                [f"proprio_layer_{i}" for i in range(len(self.proprio_encoder_layers))]
            )
        
        # Profile causal attention layers
        if self.causal_attention_layers:
            causal_modules = [layer for _, layer in self.causal_attention_layers]
            causal_names = [f"causal_layer_{i}" for i, _ in self.causal_attention_layers]
            
            self.gradient_analyzer.setup_layer_profiling(
                "causal_attention",
                causal_modules[:6],  # First 6 causal layers
                causal_names[:6]
            )
    
    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.proprio_encoder is None:
            self.discover_model_structure()
        
        # Standard encoder setup
        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder
        )
        
        # ADDITIONAL: Extract features at each layer of separate encoder
        # Critical for π0 - understand what separate encoder learns
        if self.proprio_encoder_layers:
            for i, layer in enumerate(self.proprio_encoder_layers):
                self.representation_analyzer.feature_extractor.attach(
                    layer,
                    name=f"proprio_encoder_layer_{i}"
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
        
        # ADDITIONAL: Test causal masking
        # Can we ablate causal attention and measure impact?
        if self.causal_attention_layers:
            # Register first causal layer for ablation testing
            first_causal_idx, first_causal_layer = self.causal_attention_layers[0]
            self.ablation_coordinator.ablation_manager.register_encoder(
                f"causal_layer_{first_causal_idx}",
                first_causal_layer
            )
    
    def attach_utilization_hooks(self):
        """Attach downstream utilization tracking hooks."""
        if self.transformer_layers is None:
            self.discover_model_structure()
        
        # CRITICAL: Track attention in causal layers
        # π0's causal masking is key architectural feature
        if self.causal_attention_layers:
            for layer_idx, layer in self.causal_attention_layers:
                attn_module = None
                if hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                elif hasattr(layer, 'attn'):
                    attn_module = layer.attn
                
                if attn_module:
                    self.utilization_analyzer.attention_tracker.attach(
                        attn_module,
                        name=f"causal_attn_{layer_idx}"
                    )
        
        # Set modality ranges for attention tracking
        modality_ranges = self.get_modality_token_ranges()
        self.utilization_analyzer.attention_tracker.set_modality_ranges(modality_ranges)
        
        # Feature similarity across proprio encoder layers
        if self.proprio_encoder_layers:
            for i, layer in enumerate(self.proprio_encoder_layers):
                self.utilization_analyzer.similarity_tracker.attach(
                    layer,
                    name=f"proprio_layer_{i}"
                )
        
        # Mutual information with action
        # π0 uses flow matching, so track MI between proprio features and action
        if self.proprio_encoder:
            self.utilization_analyzer.mi_estimator.attach_feature_extractor(
                self.proprio_encoder,
                name="proprio_features"
            )
    
    def get_modality_token_ranges(self) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality in π0.
        
        π0 uses separate encoding then concatenation with causal masking.
        
        Returns:
            Dict with (start, end) ranges
        """
        # π0 configuration:
        # Vision: ViT patches
        # Proprio: Separate encoder output (sequence of tokens)
        # Language: Instruction tokens
        # Order may vary - π0 uses block-wise masking
        
        vision_tokens = 196
        proprio_tokens = 16  # Separate encoder outputs a sequence
        
        ranges = {
            "vision": (0, vision_tokens),
            "proprio": (vision_tokens, vision_tokens + proprio_tokens),
            "language": (vision_tokens + proprio_tokens, -1)
        }
        
        return ranges
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """
        Run complete diagnostic analysis for π0.
        
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
            "note": "π0 uses separate encoder + causal masking - most advanced proprio encoding",
            "analysis_focus": [
                "Gradient flow through separate proprio encoder layers",
                "Representation evolution across encoder (Layer 0 → Layer N)",
                "CKA: Does separate encoder learn distinct features from vision?",
                "Attention patterns in causal layers: proprio vs vision",
                "MI(proprio_features, actions) via MINE",
                "Ablation: Full encoder vs No-causal vs No-proprio"
            ],
            "key_questions": [
                "Does separate encoder learn meaningful proprio representations?",
                "How does block-wise causal masking affect proprio utilization?",
                "Do later encoder layers add value beyond early layers?",
                "Is proprio→action MI higher than vision→action MI?",
                "What happens when causal masking is removed?"
            ],
            "architectural_features": {
                "separate_encoder": True,
                "causal_masking": "block-wise",
                "action_generation": "flow matching",
                "conditioning": "asymmetric"
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
        
        # Proprio encoder layer-wise gradients
        if self.gradient_analyzer.layer_profilers:
            encoder_profiler = self.gradient_analyzer.layer_profilers.get("proprio_encoder")
            if encoder_profiler:
                summary["encoder_gradients"] = {
                    "layer_norms": encoder_profiler.get_summary(),
                    "vanishing_point": encoder_profiler.find_vanishing_point()
                }
            
            causal_profiler = self.gradient_analyzer.layer_profilers.get("causal_attention")
            if causal_profiler:
                summary["causal_gradients"] = causal_profiler.get_summary()
        
        # Representation progression through encoder
        if self.representation_analyzer.feature_extractor.features:
            encoder_features = {
                name: feats
                for name, feats in self.representation_analyzer.feature_extractor.features.items()
                if "proprio_encoder_layer" in name
            }
            if encoder_features:
                summary["encoder_layer_features"] = list(encoder_features.keys())
        
        # Causal attention patterns
        if self.utilization_analyzer.attention_tracker.results:
            causal_attention = {
                name: result
                for name, result in self.utilization_analyzer.attention_tracker.results.items()
                if "causal" in name
            }
            if causal_attention:
                summary["causal_attention_patterns"] = self.utilization_analyzer.attention_tracker.compute_modality_attention()
        
        # MI estimation
        if hasattr(self.utilization_analyzer.mi_estimator, 'mi_estimates'):
            summary["mutual_information"] = self.utilization_analyzer.mi_estimator.mi_estimates
        
        # Overall comparison
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
        print("π0 (PI0) MODEL STRUCTURE")
        print("=" * 80)
        print(f"Model: {structure['model_name']}")
        print(f"Has Proprioceptive Encoder: {structure['has_proprio_encoder']}")
        print(f"Proprio Encoder Type: {structure.get('proprio_encoder_type', 'unknown')}")
        
        print("\nDiscovered Components:")
        for comp_name, attr_name in structure["components"].items():
            print(f"  - {comp_name}: model.{attr_name}")
        
        print(f"\nProprio Encoder Details:")
        print(f"  - Encoder Type: Separate multi-layer encoder")
        print(f"  - Number of Layers: {structure.get('proprio_encoder_layers', 0)}")
        
        if self.proprio_encoder_layers:
            print("\n  Encoder Layer Structure:")
            for i, layer in enumerate(self.proprio_encoder_layers[:5]):  # Show first 5
                print(f"    - Layer {i}: {type(layer).__name__}")
        
        if self.transformer_layers:
            print(f"\nTransformer Layers: {len(self.transformer_layers)}")
        
        if self.causal_attention_layers:
            print(f"Causal Attention Layers: {len(self.causal_attention_layers)}")
            causal_indices = [idx for idx, _ in self.causal_attention_layers]
            print(f"  - Layer indices: {causal_indices[:10]}...")  # Show first 10
        
        print("\nArchitectural Features:")
        print("  - Fusion: Block-wise causal masking")
        print("  - Action Generation: Flow matching")
        print("  - Conditioning: Asymmetric (action conditioned on state)")
        print("\nKey Advantage: Dedicated encoder learns proprio-specific representations")
        print("=" * 80)
