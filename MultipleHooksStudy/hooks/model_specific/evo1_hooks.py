"""
Evo-1 specific hook implementations.

Evo-1 Architecture (VERIFIED from MINT-SJTU/Evo-1 repository):
- Input: {Images, Language, State}
- embedder: InternVL3Embedder (InternVL3-1B VL backbone) - processes images+text → fused_tokens
- action_head: FlowmatchingActionHead - generates actions via flow matching diffusion
  - Contains internal state_encoder: CategorySpecificMLP (state → embeddings)
  - Transformer blocks process action tokens with fused_tokens + state embeddings as context
  - MLP head outputs final actions
- Output: Action

Key Implementation Detail:
State encoding happens INSIDE action_head via state_encoder (CategorySpecificMLP),
NOT in a separate integration module. The state encoder is a simple 3-layer MLP
that projects proprioceptive state to the same embedding dimension as VL features.

References:
- Paper: https://arxiv.org/abs/2512.06951
- GitHub: https://github.com/MINT-SJTU/Evo-1
- Actual implementation: Evo_1/scripts/Evo1.py, Evo_1/model/action_head/flow_matching.py
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn

from ..gradient_hooks import GradientFlowAnalyzer
from ..representation_hooks import RepresentationQualityAnalyzer
from ..ablation_hooks import AblationStudyCoordinator
from ..utilization_hooks import DownstreamUtilizationAnalyzer


class Evo1Hooks:
    """Hook manager specialized for Evo-1 VLA model."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        # Initialize analyzers
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.representation_analyzer = RepresentationQualityAnalyzer()
        self.ablation_coordinator = AblationStudyCoordinator()
        self.utilization_analyzer = DownstreamUtilizationAnalyzer()
        
        # Model components (verified from actual Evo-1 implementation)
        self.embedder = None  # InternVL3Embedder (VL backbone)
        self.action_head = None  # FlowmatchingActionHead (diffusion-based action generation)
        self.state_encoder = None  # action_head.state_encoder (CategorySpecificMLP)
    
    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover Evo-1 model structure.
        
        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "Evo-1",
            "has_proprio_encoder": True,
            "proprio_encoder_type": "state_encoder_in_action_head",
            "architecture_type": "embedder+action_head",
            "components": {}
        }
        
        # Find embedder (InternVL3Embedder - VL backbone)
        for attr in ['embedder', 'vl_backbone', 'vision_language_backbone', 'internvl', 'vlm']:
            if hasattr(self.model, attr):
                self.embedder = getattr(self.model, attr)
                structure["components"]["embedder"] = attr
                break
        
        # Find action_head (FlowmatchingActionHead)
        for attr in ['action_head', 'policy_head', 'diffusion_transformer', 'diffusion', 'action_decoder']:
            if hasattr(self.model, attr):
                self.action_head = getattr(self.model, attr)
                structure["components"]["action_head"] = attr
                
                # Extract state_encoder from inside action_head
                if hasattr(self.action_head, 'state_encoder'):
                    self.state_encoder = self.action_head.state_encoder
                    structure["components"]["state_encoder"] = f"{attr}.state_encoder"
                    structure["state_encoder_found"] = True
                else:
                    structure["state_encoder_found"] = False
                    structure["warning"] = "state_encoder not found in action_head"
                
                break
        
        # Verify we found the expected components
        if self.embedder is None:
            structure["warning"] = "embedder (VL backbone) not found - check model attribute naming"
        if self.action_head is None:
            structure["warning"] = "action_head not found - check model attribute naming"
        
        return structure
    
    def attach_gradient_hooks(self):
        """Attach gradient flow analysis hooks."""
        if self.embedder is None:
            self.discover_model_structure()
        
        # Profile state_encoder (CategorySpecificMLP inside action_head)
        # This is the actual proprioceptive encoder in Evo-1
        if self.state_encoder:
            self.gradient_analyzer.setup_encoder_tracking(
                vision_encoder=None,  # embedder is monolithic VL model
                language_encoder=None,
                proprio_encoder=self.state_encoder
            )
            
            # Additional layer-level profiling for state_encoder
            self.gradient_analyzer.setup_layer_profiling(
                "state_encoder",
                [self.state_encoder],
                ["state_mlp"]
            )
        
        # Profile action_head transformer blocks
        if self.action_head:
            # Try to get transformer blocks from action_head
            transformer_blocks = None
            for attr in ['transformer_blocks', 'blocks', 'layers']:
                if hasattr(self.action_head, attr):
                    transformer_blocks = getattr(self.action_head, attr)
                    break
            
            if transformer_blocks:
                # Sample first few blocks for profiling (FlowmatchingActionHead typically has 8 layers)
                sample_blocks = list(transformer_blocks)[:6]
                self.gradient_analyzer.setup_layer_profiling(
                    "action_head_transformer",
                    sample_blocks,
                    [f"action_block_{i}" for i in range(len(sample_blocks))]
                )
    
    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.embedder is None:
            self.discover_model_structure()
        
        # Setup representation analyzer with state_encoder
        self.representation_analyzer.setup(
            vision_encoder=None,  # embedder is monolithic
            language_encoder=None,
            proprio_encoder=self.state_encoder
        )
        
        # Extract features from state_encoder output
        if self.state_encoder:
            self.representation_analyzer.feature_extractor.attach(
                self.state_encoder,
                name="state_encoder_output"
            )
        
        # Extract features from action_head output (final action representation)
        if self.action_head:
            self.representation_analyzer.feature_extractor.attach(
                self.action_head,
                name="action_head_output"
            )
    
    def attach_ablation_hooks(self):
        """Attach ablation study hooks."""
        if self.embedder is None:
            self.discover_model_structure()
        
        # Setup ablation with state_encoder
        self.ablation_coordinator.setup(
            vision_encoder=None,
            language_encoder=None,
            proprio_encoder=self.state_encoder
        )
        
        # ADDITIONAL: Ablate state_encoder
        # Key research question: How important is proprioceptive state encoding?
        if self.state_encoder:
            self.ablation_coordinator.add_ablation_target(
                name="state_encoder",
                module=self.state_encoder,
                ablation_types=['zero', 'noise', 'freeze']
            )
    
    def attach_utilization_hooks(self):
        """Attach downstream utilization analysis hooks."""
        if self.embedder is None:
            self.discover_model_structure()
        
        # Track state_encoder utilization for action generation
        self.utilization_analyzer.setup(
            vision_encoder=None,
            language_encoder=None,
            proprio_encoder=self.state_encoder
        )
        
        # Track state_encoder module directly
        if self.state_encoder:
            self.utilization_analyzer.track_module(
                self.state_encoder,
                name="state_encoder"
            )
    
    def get_results(self) -> Dict[str, Any]:
        """Get all analysis results."""
        return {
            "gradient_flow": self.gradient_analyzer.get_results(),
            "representation_quality": self.representation_analyzer.get_results(),
            "ablation_study": self.ablation_coordinator.get_results(),
            "downstream_utilization": self.utilization_analyzer.get_results(),
            "model_structure": self.discover_model_structure()
        }
    
    def get_research_insights(self) -> Dict[str, Any]:
        """
        Get Evo-1-specific research insights.
        
        Focus on:
        1. State encoder effectiveness (proprioceptive encoding via CategorySpecificMLP)
        2. Action head patterns (flow matching diffusion for action generation)
        3. State vs VL feature utilization balance
        """
        results = self.get_results()
        
        insights = {
            "state_encoder_analysis": {},
            "action_head_patterns": {},
            "feature_utilization": {}
        }
        
        # Analyze state_encoder (the actual proprio encoder)
        if "state_encoder" in results.get("gradient_flow", {}).get("layer_profiles", {}):
            state_grads = results["gradient_flow"]["layer_profiles"]["state_encoder"]
            insights["state_encoder_analysis"] = {
                "gradient_magnitude": state_grads.get("mean_gradient", 0),
                "gradient_stability": state_grads.get("gradient_variance", 0),
                "critical_for_training": state_grads.get("mean_gradient", 0) > 1e-4
            }
        
        # Check representation quality from state_encoder output
        if "state_encoder_output" in results.get("representation_quality", {}).get("features", {}):
            state_features = results["representation_quality"]["features"]["state_encoder_output"]
            insights["state_encoder_analysis"]["representation_quality"] = {
                "feature_rank": state_features.get("effective_rank", 0),
                "condition_number": state_features.get("condition_number", 0),
                "well_conditioned": state_features.get("condition_number", float('inf')) < 100
            }
        
        # Analyze action_head transformer patterns
        if "action_head_transformer" in results.get("gradient_flow", {}).get("layer_profiles", {}):
            action_grads = results["gradient_flow"]["layer_profiles"]["action_head_transformer"]
            insights["action_head_patterns"] = {
                "layer_gradient_distribution": action_grads,
                "training_stability": "stable" if action_grads.get("gradient_variance", 0) < 1.0 else "unstable"
            }
        
        # Analyze feature utilization (state vs VL)
        ablation_results = results.get("ablation_study", {})
        if "state_encoder" in ablation_results:
            insights["feature_utilization"]["state_importance"] = {
                "performance_drop_when_ablated": ablation_results["state_encoder"],
                "critical": ablation_results["state_encoder"].get("zero_ablation_impact", 0) > 0.1
            }
        
        return insights
