"""
SmolVLA Model-Specific Hooks

Adapter for attaching hooks to SmolVLA (450M) model.
SmolVLA uses a single Linear layer (state_proj) to project robot state
to a VLM token, concatenated with image/language tokens.

Architecture:
- VLM Backbone: SmolVLM2 (SigLIP vision + SmolLM2 first 16 layers)
- State Encoder: state_proj (Single Linear layer, state → 1 VLM token)
- Action Expert: Flow matching transformer (~100M params)
- Loss Function: Flow matching MSE on vector fields
- Framework: LeRobot (HuggingFace)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from ..gradient_hooks import GradientFlowAnalyzer
from ..representation_hooks import RepresentationQualityAnalyzer
from ..ablation_hooks import AblationStudyCoordinator
from ..utilization_hooks import DownstreamUtilizationAnalyzer


class SmolVLAHooks:
    """
    Hook adapter for SmolVLA model.

    SmolVLA uses LIGHTWEIGHT state encoding:
    - State → Single Linear layer (state_proj) → 1 VLM token
    - Concatenated with image/language tokens into VLM backbone
    - Flow matching for action generation via action expert

    The model is loaded via lerobot:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')
    """

    def __init__(self, model):
        """
        Initialize hook adapter.

        Args:
            model: SmolVLA policy instance (SmolVLAPolicy or inner model)
        """
        self.model = model
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.representation_analyzer = RepresentationQualityAnalyzer()
        self.ablation_coordinator = AblationStudyCoordinator()
        self.utilization_analyzer = DownstreamUtilizationAnalyzer()

        # Model components (populated by discover_model_structure)
        self.state_proj = None          # Linear layer: state → 1 VLM token
        self.vision_encoder = None      # SigLIP vision encoder
        self.language_encoder = None    # SmolLM2 language model
        self.action_expert = None       # Flow matching transformer
        self.vlm_backbone = None        # SmolVLM2 backbone
        self.transformer_layers = None  # VLM or action expert layers

        # Hook handles for cleanup
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover SmolVLA model structure.

        SmolVLAPolicy wraps an inner model — we probe both levels.

        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "SmolVLA",
            "has_proprio_encoder": True,
            "proprio_encoder_type": "single_linear_projection",
            "architecture_type": "vlm_backbone+action_expert",
            "components": {}
        }

        # Unwrap SmolVLAPolicy → inner model
        inner = getattr(self.model, 'model', self.model)

        # --- State projection (state_proj) ---
        for attr in ['state_proj', 'proprio_proj', 'state_token_proj',
                      'state_encoder', 'robot_state_proj']:
            candidate = self._find_attr_recursive(inner, attr)
            if candidate is not None:
                self.state_proj = candidate
                structure["components"]["state_proj"] = attr
                if isinstance(candidate, nn.Linear):
                    structure["state_input_dim"] = candidate.in_features
                    structure["state_output_dim"] = candidate.out_features
                    structure["state_proj_params"] = (
                        candidate.in_features * candidate.out_features
                        + (candidate.out_features if candidate.bias is not None else 0)
                    )
                break

        # Fallback: scan named_modules for state-related Linear layers
        if self.state_proj is None:
            for name, module in self.model.named_modules():
                if 'state' in name.lower() and isinstance(module, nn.Linear):
                    self.state_proj = module
                    structure["components"]["state_proj"] = name
                    structure["state_input_dim"] = module.in_features
                    structure["state_output_dim"] = module.out_features
                    break

        if self.state_proj is None:
            structure["warning_state"] = (
                "state_proj not found. Run `policy.named_modules()` to inspect manually."
            )

        # --- Vision encoder (SigLIP) ---
        for attr_path in [
            'vision_encoder', 'vision_tower', 'image_encoder',
            'vlm.vision_tower', 'backbone.vision_encoder',
        ]:
            candidate = self._find_nested_attr(inner, attr_path)
            if candidate is not None:
                self.vision_encoder = candidate
                structure["components"]["vision_encoder"] = attr_path
                n = sum(p.numel() for p in candidate.parameters())
                structure["vision_encoder_params"] = n
                break

        # --- Language model (SmolLM2) ---
        for attr_path in [
            'language_model', 'language_encoder', 'lm_head',
            'vlm.language_model', 'backbone.language_model',
        ]:
            candidate = self._find_nested_attr(inner, attr_path)
            if candidate is not None:
                self.language_encoder = candidate
                structure["components"]["language_encoder"] = attr_path
                n = sum(p.numel() for p in candidate.parameters())
                structure["language_encoder_params"] = n
                break

        # --- Action expert (flow matching transformer) ---
        for attr in ['action_expert', 'action_head', 'flow_matching_head',
                      'action_decoder', 'policy_head']:
            candidate = self._find_attr_recursive(inner, attr)
            if candidate is not None:
                self.action_expert = candidate
                structure["components"]["action_expert"] = attr
                n = sum(p.numel() for p in candidate.parameters())
                structure["action_expert_params"] = n
                break

        # --- VLM backbone (SmolVLM2) ---
        for attr in ['vlm', 'backbone', 'smolvlm', 'vlm_backbone']:
            candidate = self._find_attr_recursive(inner, attr)
            if candidate is not None:
                self.vlm_backbone = candidate
                structure["components"]["vlm_backbone"] = attr
                break

        # --- Transformer layers (for layer-wise profiling) ---
        for candidate_module in [self.action_expert, self.vlm_backbone, inner]:
            if candidate_module is None:
                continue
            for attr in ['layers', 'blocks', 'transformer_blocks',
                          'decoder_layers', 'encoder_layers']:
                if hasattr(candidate_module, attr):
                    layers = getattr(candidate_module, attr)
                    if hasattr(layers, '__len__') and len(layers) > 0:
                        self.transformer_layers = layers
                        structure["components"]["transformer_layers"] = attr
                        structure["n_transformer_layers"] = len(layers)
                        break
            if self.transformer_layers is not None:
                break

        # Total params
        total = sum(p.numel() for p in self.model.parameters())
        structure["total_parameters"] = total

        return structure

    # ------------------------------------------------------------------
    # Hook attachment
    # ------------------------------------------------------------------

    def attach_gradient_hooks(self):
        """Attach gradient flow analysis hooks to SmolVLA components."""
        if self.state_proj is None:
            self.discover_model_structure()

        # Encoder-level tracking
        self.gradient_analyzer.setup_encoder_tracking(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.state_proj,
        )

        # State projection layer profiling
        if self.state_proj:
            self.gradient_analyzer.setup_layer_profiling(
                "state_proj",
                [self.state_proj],
                ["state_proj"],
            )

        # Action expert profiling (sample first N layers)
        if self.transformer_layers is not None:
            sample = list(self.transformer_layers)[:6]
            self.gradient_analyzer.setup_layer_profiling(
                "action_expert_layers",
                sample,
                [f"layer_{i}" for i in range(len(sample))],
            )

    def attach_representation_hooks(self):
        """Attach representation quality analysis hooks."""
        if self.state_proj is None:
            self.discover_model_structure()

        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.state_proj,
        )

        if self.state_proj:
            self.representation_analyzer.feature_extractor.attach(
                self.state_proj,
                name="state_proj_output",
            )

        if self.action_expert:
            self.representation_analyzer.feature_extractor.attach(
                self.action_expert,
                name="action_expert_output",
            )

    def attach_ablation_hooks(self, ablation_type: str = "zero_state"):
        """
        Attach ablation study hooks.

        Args:
            ablation_type: Type of ablation ('zero_state', 'noise_state', 'freeze_state')
        """
        if self.state_proj is None:
            self.discover_model_structure()

        if self.state_proj is None:
            raise RuntimeError("Cannot attach ablation hooks: state_proj not found")

        if ablation_type == "zero_state":
            handle = self.state_proj.register_forward_hook(self._zero_output_hook)
            self._hook_handles.append(handle)
        elif ablation_type == "noise_state":
            handle = self.state_proj.register_forward_hook(self._noise_output_hook)
            self._hook_handles.append(handle)
        elif ablation_type == "freeze_state":
            for param in self.state_proj.parameters():
                param.requires_grad = False

    def attach_utilization_hooks(self):
        """Attach downstream utilization analysis hooks."""
        if self.state_proj is None:
            self.discover_model_structure()

        self.utilization_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=self.state_proj,
        )

        if self.state_proj:
            self.utilization_analyzer.track_module(
                self.state_proj,
                name="state_proj",
            )

    # ------------------------------------------------------------------
    # Hook functions
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_output_hook(module, input, output):
        """Forward hook that zeros out the module output (Pass0 ablation)."""
        return torch.zeros_like(output)

    @staticmethod
    def _noise_output_hook(module, input, output):
        """Forward hook that replaces output with Gaussian noise of same scale."""
        return torch.randn_like(output) * output.detach().std()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, Any]:
        """Get all analysis results."""
        return {
            "gradient_flow": self.gradient_analyzer.get_results(),
            "representation_quality": self.representation_analyzer.get_results(),
            "ablation_study": self.ablation_coordinator.get_results(),
            "downstream_utilization": self.utilization_analyzer.get_results(),
            "model_structure": self.discover_model_structure(),
        }

    def get_research_insights(self) -> Dict[str, Any]:
        """
        Get SmolVLA-specific research insights.

        Focus on:
        1. State encoder effectiveness (proprioceptive encoding via single Linear)
        2. Flow matching loss sensitivity to state input
        3. State vs VL feature utilization balance
        """
        results = self.get_results()

        insights = {
            "state_encoder_analysis": {},
            "action_expert_patterns": {},
            "feature_utilization": {},
        }

        # State projection gradient analysis
        grad_profiles = results.get("gradient_flow", {}).get("layer_profiles", {})
        if "state_proj" in grad_profiles:
            state_grads = grad_profiles["state_proj"]
            insights["state_encoder_analysis"] = {
                "gradient_magnitude": state_grads.get("mean_gradient", 0),
                "gradient_stability": state_grads.get("gradient_variance", 0),
                "critical_for_training": state_grads.get("mean_gradient", 0) > 1e-4,
            }

        # Representation quality
        rep_features = results.get("representation_quality", {}).get("features", {})
        if "state_proj_output" in rep_features:
            sf = rep_features["state_proj_output"]
            insights["state_encoder_analysis"]["representation_quality"] = {
                "feature_rank": sf.get("effective_rank", 0),
                "condition_number": sf.get("condition_number", 0),
                "well_conditioned": sf.get("condition_number", float("inf")) < 100,
            }

        # Action expert patterns
        if "action_expert_layers" in grad_profiles:
            ae_grads = grad_profiles["action_expert_layers"]
            insights["action_expert_patterns"] = {
                "layer_gradient_distribution": ae_grads,
                "training_stability": (
                    "stable" if ae_grads.get("gradient_variance", 0) < 1.0
                    else "unstable"
                ),
            }

        # Feature utilization
        ablation_res = results.get("ablation_study", {})
        if "state_proj" in ablation_res:
            insights["feature_utilization"]["state_importance"] = {
                "performance_drop_when_ablated": ablation_res["state_proj"],
                "critical": ablation_res["state_proj"].get(
                    "zero_ablation_impact", 0
                ) > 0.1,
            }

        return insights

    def get_modality_token_ranges(self) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality in SmolVLA.

        SmolVLA concatenates: [image_tokens, state_token, language_tokens]
        into a single VLM input sequence.

        Returns:
            Dict with (start, end) ranges
        """
        # SmolVLA default: SigLIP patches + 1 state token + instruction tokens
        vision_tokens = 256   # SigLIP default patches
        state_tokens = 1      # state_proj produces single token

        ranges = {
            "vision": (0, vision_tokens),
            "state": (vision_tokens, vision_tokens + state_tokens),
            "language": (vision_tokens + state_tokens, -1),
        }
        return ranges

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """
        Run complete diagnostic analysis for SmolVLA.

        Returns:
            Comprehensive diagnostic report
        """
        structure = self.discover_model_structure()

        self.attach_gradient_hooks()
        self.attach_representation_hooks()
        self.attach_utilization_hooks()

        report = {
            "model_structure": structure,
            "note": "SmolVLA uses SmolVLM2 backbone + state_proj Linear + flow matching action expert",
            "analysis_focus": [
                "Gradient flow through state_proj layer",
                "Representation quality of state encoding",
                "CKA: State vs Vision features",
                "Flow matching loss sensitivity to state input",
                "Ablation: With state vs Without state (zero output)",
            ],
            "key_questions": [
                "Does the single Linear state_proj learn meaningful representations?",
                "How does flow matching loss respond to zeroed state input?",
                "Is state_proj/vision gradient ratio < 0.1 (underutilization)?",
                "Does performance drop when state_proj output is zeroed?",
            ],
            "architectural_features": {
                "vision_encoder": "SigLIP (via SmolVLM2)",
                "state_encoder": "Single Linear projection (state_proj)",
                "backbone": "SmolVLM2 (SmolLM2 first 16 layers)",
                "action_generation": "Flow matching transformer (~100M params)",
                "parameters": "~450M total",
            },
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
            sp = self.gradient_analyzer.layer_profilers.get("state_proj")
            if sp:
                summary["state_proj_gradients"] = sp.get_summary()

            ae = self.gradient_analyzer.layer_profilers.get("action_expert_layers")
            if ae:
                summary["action_expert_gradients"] = ae.get_summary()

        # State representation features
        if self.representation_analyzer.feature_extractor.features:
            state_feats = {
                n: f for n, f in self.representation_analyzer.feature_extractor.features.items()
                if "state_proj" in n
            }
            if state_feats:
                summary["state_features"] = list(state_feats.keys())

        # Overall comparison
        summary["encoder_comparison"] = self.gradient_analyzer.get_comprehensive_report()

        return summary

    def cleanup(self):
        """Remove all hooks and restore state."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        self.gradient_analyzer.remove_all()
        self.representation_analyzer.remove_hooks()
        self.ablation_coordinator.cleanup()
        self.utilization_analyzer.cleanup()

    def print_model_info(self):
        """Print human-readable model information."""
        structure = self.discover_model_structure()

        print("=" * 80)
        print("SmolVLA MODEL STRUCTURE")
        print("=" * 80)
        print(f"Model: {structure['model_name']}")
        print(f"Has State Encoder: {structure['has_proprio_encoder']}")
        print(f"State Encoder Type: {structure.get('proprio_encoder_type', 'unknown')}")

        print("\nDiscovered Components:")
        for comp_name, attr_name in structure.get("components", {}).items():
            print(f"  - {comp_name}: model.{attr_name}")

        if "state_input_dim" in structure:
            print(f"\nState Projector:")
            print(f"  - Input dim:  {structure['state_input_dim']}")
            print(f"  - Output dim: {structure['state_output_dim']}")
            print(f"  - Parameters: {structure.get('state_proj_params', 'N/A')}")

        print(f"\nTotal Parameters: {structure.get('total_parameters', 0) / 1e6:.0f}M")
        print("\nArchitectural Features:")
        print("  - Vision: SigLIP (via SmolVLM2)")
        print("  - State: Single Linear projection (state_proj → 1 VLM token)")
        print("  - Backbone: SmolVLM2 (SmolLM2 first 16 layers)")
        print("  - Action Expert: Flow matching transformer (~100M params)")
        print("  - Loss: Flow matching MSE on vector fields")
        print("\nKey Feature: Minimal state encoding via single projection layer")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_attr_recursive(module: nn.Module, attr_name: str):
        """Find attribute by name on module or its direct children."""
        if hasattr(module, attr_name):
            return getattr(module, attr_name)
        for name, child in module.named_children():
            if name == attr_name:
                return child
        return None

    @staticmethod
    def _find_nested_attr(module: nn.Module, dotted_path: str):
        """Find attribute by dotted path (e.g. 'vlm.vision_tower')."""
        parts = dotted_path.split(".")
        current = module
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current
