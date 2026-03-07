"""
π0 (Pi0) Model-Specific Hooks

Adapter for attaching hooks to π0 (3.3B) model.
π0 uses a single Linear layer (state_proj) to project proprioceptive state
to a single VLM embedding token, concatenated with vision/language tokens.

Architecture:
- Vision: PaliGemma ViT encoder
- Proprio: state_proj (Single nn.Linear: state_dim → embed_dim)
- Language: PaliGemma language model
- Fusion: State token concat with vision/language tokens, processed by Expert Gemma
- Action: Flow matching architecture (Expert Gemma transformer)
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
        self.state_proj = None  # State projection layer (Linear in π0)
        self.action_in_proj = None  # Action input projection
        self.action_out_proj = None  # Action output projection
        self.language_encoder = None
        self.paligemma = None  # PaliGemma backbone
        self.gemma_expert = None  # Expert Gemma for actions
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
            "proprio_encoder_type": "linear_projection",  # π0 uses single Linear layer
            "components": {}
        }
        
        # Check for PaliGemma with Expert wrapper (actual π0 structure)
        for attr in ['paligemma_with_expert', 'model']:
            if hasattr(self.model, attr):
                wrapper = getattr(self.model, attr)
                if hasattr(wrapper, 'paligemma'):
                    self.paligemma = wrapper.paligemma
                    structure["components"]["paligemma"] = f"{attr}.paligemma"
                if hasattr(wrapper, 'gemma_expert'):
                    self.gemma_expert = wrapper.gemma_expert
                    structure["components"]["gemma_expert"] = f"{attr}.gemma_expert"
                break
        
        # State projection (π0's actual state encoder - single Linear layer)
        for attr in ['state_proj', 'proprio_proj', 'state_encoder']:
            if hasattr(self.model, attr):
                self.state_proj = getattr(self.model, attr)
                structure["components"]["state_proj"] = attr
                if isinstance(self.state_proj, nn.Linear):
                    structure["state_proj_architecture"] = "single_linear"
                    structure["state_input_dim"] = self.state_proj.in_features
                    structure["state_output_dim"] = self.state_proj.out_features
                break
        
        # Action projections
        for attr in ['action_in_proj', 'action_input_proj']:
            if hasattr(self.model, attr):
                self.action_in_proj = getattr(self.model, attr)
                structure["components"]["action_in_proj"] = attr
                break
        
        for attr in ['action_out_proj', 'action_output_proj']:
            if hasattr(self.model, attr):
                self.action_out_proj = getattr(self.model, attr)
                structure["components"]["action_out_proj"] = attr
                break
        
        # Vision encoder (from PaliGemma)
        if self.paligemma and hasattr(self.paligemma, 'model'):
            if hasattr(self.paligemma.model, 'vision_tower'):
                self.vision_encoder = self.paligemma.model.vision_tower
                structure["components"]["vision_encoder"] = "paligemma.model.vision_tower"
        
        # Language encoder (from PaliGemma)
        if self.paligemma and hasattr(self.paligemma, 'model'):
            if hasattr(self.paligemma.model, 'language_model'):
                self.language_encoder = self.paligemma.model.language_model
                structure["components"]["language_encoder"] = "paligemma.model.language_model"
        
        return structure
    
    def attach_gradient_hooks(self):
        """Attach gradient flow analysis hooks."""
        if self.state_proj is None:
            self.discover_model_structure()
        
        # Attach encoder-level tracking
        self.gradient_analyzer.setup_encoder_tracking(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.state_proj  # Pass state_proj as proprio_encoder
        )
        
        # Profile state projection layer (π0's state encoder)
        if self.state_proj:
            self.gradient_analyzer.setup_layer_profiling(
                "state_proj",
                [self.state_proj],
                ["state_proj"]
            )
        
        # Profile action projections
        if self.action_in_proj and self.action_out_proj:
            self.gradient_analyzer.setup_layer_profiling(
                "action_projections",
                [self.action_in_proj, self.action_out_proj],
                ["action_in_proj", "action_out_proj"]
            )
    
    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.state_proj is None:
            self.discover_model_structure()
        
        # Standard setup for encoders
        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.state_proj  # Pass state_proj as proprio_encoder
        )
        
        # Extract features from state projection output
        if self.state_proj:
            self.representation_analyzer.feature_extractor.attach(
                self.state_proj,
                name="state_proj_output"
            )
    
    def _create_zero_output_hook(self, module):
        """Create a hook that zeros out module output for ablation studies."""
        def hook(module, input, output):
            return torch.zeros_like(output)
        return hook
    
    def attach_ablation_hooks(self, ablation_type: str = "zero_state"):
        """Attach ablation study hooks."""
        if self.state_proj is None:
            self.discover_model_structure()
        
        if ablation_type == "zero_state":
            # Zero out state projection output
            self.ablation_coordinator.register_ablation(
                "zero_state_proj_output",
                self._create_zero_output_hook(self.state_proj)
            )
    
    def attach_utilization_hooks(self):
        """Attach downstream utilization analysis hooks."""
        if self.state_proj is None:
            self.discover_model_structure()
        
        # Track how state projection features are used downstream
        # Note: π0 uses flow matching, so downstream layers may be in gemma_expert
        downstream = None
        if self.gemma_expert and hasattr(self.gemma_expert, 'model'):
            if hasattr(self.gemma_expert.model, 'layers'):
                downstream = self.gemma_expert.model.layers
        
        self.utilization_analyzer.setup(
            encoder=self.state_proj,
            downstream_layers=downstream
        )
        
        # Track mutual information between state features and action
        if self.state_proj:
            self.utilization_analyzer.mi_estimator.attach_feature_extractor(
                self.state_proj,
                name="state_features"
            )
    
    def get_modality_token_ranges(self) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality in π0.
        
        π0 uses PaliGemma with state projection - tokens are fused in transformer.
        
        Returns:
            Dict with (start, end) ranges
        """
        # π0 (π0.5-DROID) configuration:
        # Vision: PaliGemma ViT patches (typically 256 patches)
        # State: Single projection layer output (1 token or few tokens)
        # Language: Instruction tokens from PaliGemma
        
        vision_tokens = 256  # PaliGemma default
        state_tokens = 1     # Single state projection
        
        ranges = {
            "vision": (0, vision_tokens),
            "state": (vision_tokens, vision_tokens + state_tokens),
            "language": (vision_tokens + state_tokens, -1)
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
            "note": "π0 uses PaliGemma + Expert Gemma with state_proj Linear layer",
            "analysis_focus": [
                "Gradient flow through state_proj layer",
                "Representation quality of state encoding",
                "CKA: State vs Vision features",
                "Attention patterns in expert model",
                "MI(state_features, actions) via MINE",
                "Ablation: With state vs Without state"
            ],
            "key_questions": [
                "Does state_proj learn meaningful representations?",
                "How does flow matching utilize state features?",
                "Is state→action MI higher than vision→action MI?",
                "What happens when state input is zeroed?"
            ],
            "architectural_features": {
                "vision_encoder": "PaliGemma",
                "state_encoder": "Single Linear projection",
                "action_generation": "Flow matching (Expert Gemma)",
                "conditioning": "Proprioceptive state"
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
        
        # State projection gradients
        if self.gradient_analyzer.layer_profilers:
            state_profiler = self.gradient_analyzer.layer_profilers.get("state_proj")
            if state_profiler:
                summary["state_proj_gradients"] = state_profiler.get_summary()
            
            action_profiler = self.gradient_analyzer.layer_profilers.get("action_projections")
            if action_profiler:
                summary["action_proj_gradients"] = action_profiler.get_summary()
        
        # State representation features
        if self.representation_analyzer.feature_extractor.features:
            state_features = {
                name: feats
                for name, feats in self.representation_analyzer.feature_extractor.features.items()
                if "state_proj" in name
            }
            if state_features:
                summary["state_features"] = list(state_features.keys())
        
        # Attention patterns (from expert model if available)
        if self.utilization_analyzer.attention_tracker.results:
            summary["attention_patterns"] = self.utilization_analyzer.attention_tracker.compute_modality_attention()
        
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
        print(f"Has State Encoder: {structure['has_proprio_encoder']}")
        print(f"State Encoder Type: {structure.get('proprio_encoder_type', 'unknown')}")
        
        print("\nDiscovered Components:")
        for comp_name, attr_name in structure["components"].items():
            print(f"  - {comp_name}: model.{attr_name}")
        
        print(f"\nState Encoder Details:")
        print(f"  - Type: Single Linear projection (state_proj)")
        print(f"  - Architecture: PaliGemma + Expert Gemma with flow matching")
        
        print("\nArchitectural Features:")
        print("  - Vision: PaliGemma ViT encoder")
        print("  - State: Linear projection to embedding space")
        print("  - Action Generation: Flow matching with Expert Gemma")
        print("  - Model: π0.5-DROID (3.3B parameters)")
        print("\nKey Feature: Lightweight state encoding via single projection layer")
        print("=" * 80)