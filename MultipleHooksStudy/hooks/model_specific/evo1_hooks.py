"""
Evo-1 specific hook implementations.

Evo-1 Architecture:
- Input: {Images, Language, State}
- Vision-Language Backbone: InternVL3-1B (pretrained VLM)
- Integration Module: Aligns VL features + proprioceptive state
- Cross-Modulated Diffusion Transformer: Generates actions
- Output: Action

Key Research Insight:
Two-stage training preserves semantic attention patterns, avoiding semantic drift.
This is critical for VLA performance.

References:
- Paper: https://arxiv.org/abs/2512.06951
- GitHub: https://github.com/MINT-SJTU/Evo-1
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
        
        # Model components
        self.vl_backbone = None  # InternVL3-1B
        self.vision_encoder = None  # Vision part of VL backbone
        self.language_encoder = None  # Language part of VL backbone
        self.integration_module = None  # Aligns VL + state
        self.diffusion_transformer = None  # Action generation
        self.proprio_encoder = None  # State encoding (if separate)
    
    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover Evo-1 model structure.
        
        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "Evo-1",
            "has_proprio_encoder": True,
            "proprio_encoder_type": "integration_module",
            "architecture_type": "vl_backbone+integration+diffusion",
            "components": {}
        }
        
        # Try to find VL backbone (InternVL3-1B)
        for attr in ['vl_backbone', 'vision_language_backbone', 'internvl', 'backbone', 'vlm']:
            if hasattr(self.model, attr):
                self.vl_backbone = getattr(self.model, attr)
                structure["components"]["vl_backbone"] = attr
                
                # Try to extract vision and language components from VL backbone
                if hasattr(self.vl_backbone, 'vision_model'):
                    self.vision_encoder = self.vl_backbone.vision_model
                    structure["components"]["vision_encoder"] = f"{attr}.vision_model"
                elif hasattr(self.vl_backbone, 'vision_tower'):
                    self.vision_encoder = self.vl_backbone.vision_tower
                    structure["components"]["vision_encoder"] = f"{attr}.vision_tower"
                
                if hasattr(self.vl_backbone, 'language_model'):
                    self.language_encoder = self.vl_backbone.language_model
                    structure["components"]["language_encoder"] = f"{attr}.language_model"
                elif hasattr(self.vl_backbone, 'llm'):
                    self.language_encoder = self.vl_backbone.llm
                    structure["components"]["language_encoder"] = f"{attr}.llm"
                
                break
        
        # If VL backbone not found as single component, try separate vision/language
        if self.vision_encoder is None:
            for attr in ['vision_encoder', 'vision_model', 'image_encoder', 'visual_encoder']:
                if hasattr(self.model, attr):
                    self.vision_encoder = getattr(self.model, attr)
                    structure["components"]["vision_encoder"] = attr
                    break
        
        if self.language_encoder is None:
            for attr in ['language_model', 'llm', 'text_encoder', 'language_encoder']:
                if hasattr(self.model, attr):
                    self.language_encoder = getattr(self.model, attr)
                    structure["components"]["language_encoder"] = attr
                    break
        
        # Try to find integration module (critical for Evo-1)
        # This module aligns VL features + proprioceptive state
        for attr in ['integration_module', 'integration', 'align_module', 'fusion_module', 
                     'state_integrator', 'vl_state_align']:
            if hasattr(self.model, attr):
                self.integration_module = getattr(self.model, attr)
                structure["components"]["integration_module"] = attr
                structure["integration_module_found"] = True
                break
        
        # Try to find diffusion transformer (action generation)
        for attr in ['diffusion_transformer', 'diffusion', 'action_head', 'policy_head',
                     'transformer', 'action_decoder']:
            if hasattr(self.model, attr):
                diffusion_module = getattr(self.model, attr)
                # Check if it's actually a diffusion model
                if any(keyword in str(type(diffusion_module)).lower() 
                       for keyword in ['diffusion', 'dit', 'transformer']):
                    self.diffusion_transformer = diffusion_module
                    structure["components"]["diffusion_transformer"] = attr
                    break
        
        # Try to find separate proprioceptive encoder (might not exist if integrated)
        for attr in ['proprio_encoder', 'state_encoder', 'robot_state_encoder']:
            if hasattr(self.model, attr):
                self.proprio_encoder = getattr(self.model, attr)
                structure["components"]["proprio_encoder"] = attr
                break
        
        # Mark integration module as the effective proprio encoder if no separate encoder found
        if self.proprio_encoder is None and self.integration_module is not None:
            self.proprio_encoder = self.integration_module
            structure["components"]["effective_proprio_encoder"] = "integration_module"
        
        return structure
    
    def attach_gradient_hooks(self):
        """Attach gradient flow analysis hooks."""
        if self.vl_backbone is None and self.vision_encoder is None:
            self.discover_model_structure()
        
        # Attach encoder-level tracking for VL backbone components
        self.gradient_analyzer.setup_encoder_tracking(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder  # Integration module
        )
        
        # CRITICAL: Track integration module gradients
        # This is the key innovation in Evo-1 - how VL + state align
        if self.integration_module:
            self.gradient_analyzer.setup_layer_profiling(
                "integration_module",
                [self.integration_module],
                ["integration"]
            )
        
        # Profile diffusion transformer if available
        if self.diffusion_transformer:
            # Try to get transformer layers/blocks
            transformer_layers = None
            for attr in ['layers', 'blocks', 'transformer_blocks']:
                if hasattr(self.diffusion_transformer, attr):
                    transformer_layers = getattr(self.diffusion_transformer, attr)
                    break
            
            if transformer_layers:
                # Sample first few layers for profiling
                sample_layers = list(transformer_layers)[:6]
                self.gradient_analyzer.setup_layer_profiling(
                    "diffusion_transformer",
                    sample_layers,
                    [f"diffusion_layer_{i}" for i in range(len(sample_layers))]
                )
    
    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.vl_backbone is None and self.vision_encoder is None:
            self.discover_model_structure()
        
        # Standard setup for encoders
        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder  # Integration module
        )
        
        # CRITICAL: Extract features from integration module output
        # This shows how VL + state alignment affects representation quality
        if self.integration_module:
            self.representation_analyzer.feature_extractor.attach(
                self.integration_module,
                name="integration_output"
            )
        
        # Extract features from diffusion output (final action representation)
        if self.diffusion_transformer:
            self.representation_analyzer.feature_extractor.attach(
                self.diffusion_transformer,
                name="diffusion_output"
            )
    
    def attach_ablation_hooks(self):
        """Attach ablation study hooks."""
        if self.vl_backbone is None and self.vision_encoder is None:
            self.discover_model_structure()
        
        # Standard ablation setup
        self.ablation_coordinator.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder
        )
        
        # ADDITIONAL: Ablate integration module
        # Key research question: How critical is the integration module?
        if self.integration_module:
            self.ablation_coordinator.add_ablation_target(
                name="integration_module",
                module=self.integration_module,
                ablation_types=['zero', 'noise', 'freeze']
            )
    
    def attach_utilization_hooks(self):
        """Attach downstream utilization analysis hooks."""
        if self.vl_backbone is None and self.vision_encoder is None:
            self.discover_model_structure()
        
        # Track which parts of VL backbone are used for action generation
        self.utilization_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.proprio_encoder
        )
        
        # Track integration module utilization
        if self.integration_module:
            self.utilization_analyzer.track_module(
                self.integration_module,
                name="integration_module"
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
        1. Integration module effectiveness (VL + state alignment)
        2. Semantic preservation (two-stage training impact)
        3. Diffusion transformer utilization patterns
        """
        results = self.get_results()
        
        insights = {
            "integration_module_analysis": {},
            "semantic_preservation": {},
            "diffusion_patterns": {}
        }
        
        # Analyze integration module
        if "integration_module" in results.get("gradient_flow", {}).get("layer_profiles", {}):
            integration_grads = results["gradient_flow"]["layer_profiles"]["integration_module"]
            insights["integration_module_analysis"] = {
                "gradient_magnitude": integration_grads.get("mean_gradient", 0),
                "gradient_stability": integration_grads.get("gradient_variance", 0),
                "critical_for_training": integration_grads.get("mean_gradient", 0) > 1e-4
            }
        
        # Check representation quality from integration output
        if "integration_output" in results.get("representation_quality", {}).get("features", {}):
            integration_features = results["representation_quality"]["features"]["integration_output"]
            insights["semantic_preservation"] = {
                "feature_rank": integration_features.get("effective_rank", 0),
                "condition_number": integration_features.get("condition_number", 0),
                "well_conditioned": integration_features.get("condition_number", float('inf')) < 100
            }
        
        # Analyze diffusion transformer patterns
        if "diffusion_transformer" in results.get("gradient_flow", {}).get("layer_profiles", {}):
            diffusion_grads = results["gradient_flow"]["layer_profiles"]["diffusion_transformer"]
            insights["diffusion_patterns"] = {
                "layer_gradient_distribution": diffusion_grads,
                "training_stability": "stable" if diffusion_grads.get("gradient_variance", 0) < 1.0 else "unstable"
            }
        
        return insights
