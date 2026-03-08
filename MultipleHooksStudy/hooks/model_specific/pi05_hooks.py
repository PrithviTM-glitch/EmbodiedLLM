"""
π0.5 (Pi0.5) Model-Specific Hooks

Adapter for attaching hooks to π0.5 (3.3B) model.
π0.5 replaces π0's single Linear state_proj with a pair of MLPs:
  - time_mlp_in:  Encodes temporal/state information into the model
  - time_mlp_out: Decodes temporal/state information for action prediction

Architecture:
- Vision: PaliGemma ViT encoder
- Proprio: time_mlp_in / time_mlp_out (MLP pair)
- Language: PaliGemma language model
- Fusion: Block-wise causal masking with co-training on heterogeneous data
- Action: Flow matching architecture (Expert Gemma transformer)

Checkpoints:
- JAX (openpi):  pi05_libero  (GCS: gs://openpi-assets/checkpoints/pi05_libero)
                 Ablation via serve_policy_ablated_pi05.py monkey-patch
- PyTorch (LeRobot): lerobot/pi05_libero_finetuned  (PI05Policy)
                     lerobot/pi05_base               (generalist, no fine-tuning)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any

from ..gradient_hooks import GradientFlowAnalyzer
from ..representation_hooks import RepresentationQualityAnalyzer
from ..ablation_hooks import AblationStudyCoordinator
from ..utilization_hooks import DownstreamUtilizationAnalyzer


class Pi05Hooks:
    """
    Hook adapter for π0.5 (Pi0.5) model.

    π0.5 uses a MLP pair for state encoding (replaces π0's single Linear state_proj):
    - time_mlp_in:  Encodes temporal/state information
    - time_mlp_out: Decodes temporal/state information for action prediction

    PyTorch checkpoint available: lerobot/pi05_libero_finetuned (PI05Policy)
    JAX checkpoint: pi05_libero via openpi — ablation uses serve_policy_ablated_pi05.py
    """

    def __init__(self, model=None):
        """
        Initialize hook adapter.

        Args:
            model: π0.5 model instance (PI05Policy or None for deferred attachment)
        """
        self.model = model
        self.gradient_analyzer = GradientFlowAnalyzer()
        self.representation_analyzer = RepresentationQualityAnalyzer()
        self.ablation_coordinator = AblationStudyCoordinator()
        self.utilization_analyzer = DownstreamUtilizationAnalyzer()

        # Model components
        self.vision_encoder = None
        self.time_mlp_in = None   # Temporal/state input MLP  (replaces state_proj)
        self.time_mlp_out = None  # Temporal/state output MLP (replaces state_proj)
        self.action_in_proj = None
        self.action_out_proj = None
        self.language_encoder = None
        self.paligemma = None
        self.gemma_expert = None
        self.transformer_layers = None

    def discover_model_structure(self) -> Dict[str, Any]:
        """
        Discover π0.5 model structure.

        Returns:
            Dict with discovered model components
        """
        structure = {
            "model_name": "π0.5 (Pi0.5)",
            "has_proprio_encoder": True,
            "proprio_encoder_type": "mlp_pair",  # π0.5 uses time_mlp_in/time_mlp_out
            "components": {}
        }

        if self.model is None:
            structure["note"] = "No model provided — structure is architecture documentation only"
            return structure

        # Check for PaliGemma with Expert wrapper
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

        # State encoder: time_mlp_in / time_mlp_out (π0.5's replacement for state_proj)
        for attr in ['time_mlp_in']:
            if hasattr(self.model, attr):
                self.time_mlp_in = getattr(self.model, attr)
                structure["components"]["time_mlp_in"] = attr
                structure["time_mlp_in_type"] = type(self.time_mlp_in).__name__
                break

        for attr in ['time_mlp_out']:
            if hasattr(self.model, attr):
                self.time_mlp_out = getattr(self.model, attr)
                structure["components"]["time_mlp_out"] = attr
                structure["time_mlp_out_type"] = type(self.time_mlp_out).__name__
                break

        if self.time_mlp_in is None and self.time_mlp_out is None:
            structure["warning"] = (
                "time_mlp_in/time_mlp_out not found at model root. "
                "These may be nested inside a sub-module. "
                "Verify PI05Policy attribute names with: "
                "[(name, type(m).__name__) for name, m in model.named_modules()]"
            )

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
        """Attach gradient flow analysis hooks to time_mlp_in/time_mlp_out."""
        if self.time_mlp_in is None and self.time_mlp_out is None:
            self.discover_model_structure()

        # Track both time_mlp modules as the proprioceptive encoder
        # Use time_mlp_in as primary (analogous to state_proj in π0)
        proprio_encoder = self.time_mlp_in or self.time_mlp_out
        self.gradient_analyzer.setup_encoder_tracking(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=proprio_encoder
        )

        # Profile time_mlp layers
        mlp_modules = []
        mlp_names = []
        if self.time_mlp_in:
            mlp_modules.append(self.time_mlp_in)
            mlp_names.append("time_mlp_in")
        if self.time_mlp_out:
            mlp_modules.append(self.time_mlp_out)
            mlp_names.append("time_mlp_out")

        if mlp_modules:
            self.gradient_analyzer.setup_layer_profiling(
                "time_mlp",
                mlp_modules,
                mlp_names
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
        if self.time_mlp_in is None and self.time_mlp_out is None:
            self.discover_model_structure()

        proprio_encoder = self.time_mlp_in or self.time_mlp_out
        self.representation_analyzer.setup(
            vision_encoder=self.vision_encoder,
            language_encoder=self.language_encoder,
            proprio_encoder=proprio_encoder
        )

        # Extract features from time_mlp output
        if self.time_mlp_in:
            self.representation_analyzer.feature_extractor.attach(
                self.time_mlp_in,
                name="time_mlp_in_output"
            )
        if self.time_mlp_out:
            self.representation_analyzer.feature_extractor.attach(
                self.time_mlp_out,
                name="time_mlp_out_output"
            )

    def _create_zero_output_hook(self, module):
        """Create a hook that zeros out module output for ablation studies."""
        def hook(module, input, output):
            return torch.zeros_like(output)
        return hook

    def attach_ablation_hooks(self, ablation_type: str = "zero_state"):
        """
        Attach ablation study hooks targeting time_mlp_in/time_mlp_out.

        Uses PyTorch forward hooks on lerobot/pi05_libero_finetuned (PI05Policy).
        For JAX-based evaluation (openpi), ablation is done server-side via
        serve_policy_ablated_pi05.py (see pi05_complete.ipynb).
        """
        if self.time_mlp_in is None and self.time_mlp_out is None:
            self.discover_model_structure()

        if ablation_type == "zero_state":
            # Zero out both time_mlp modules to remove all state/temporal information
            if self.time_mlp_in:
                self.ablation_coordinator.register_ablation(
                    "zero_time_mlp_in_output",
                    self._create_zero_output_hook(self.time_mlp_in)
                )
            if self.time_mlp_out:
                self.ablation_coordinator.register_ablation(
                    "zero_time_mlp_out_output",
                    self._create_zero_output_hook(self.time_mlp_out)
                )

    def attach_utilization_hooks(self):
        """Attach downstream utilization analysis hooks."""
        if self.time_mlp_in is None and self.time_mlp_out is None:
            self.discover_model_structure()

        downstream = None
        if self.gemma_expert and hasattr(self.gemma_expert, 'model'):
            if hasattr(self.gemma_expert.model, 'layers'):
                downstream = self.gemma_expert.model.layers

        proprio_encoder = self.time_mlp_in or self.time_mlp_out
        self.utilization_analyzer.setup(
            encoder=proprio_encoder,
            downstream_layers=downstream
        )

        if self.time_mlp_in:
            self.utilization_analyzer.mi_estimator.attach_feature_extractor(
                self.time_mlp_in,
                name="time_mlp_in_features"
            )

    def get_modality_token_ranges(self) -> Dict[str, tuple]:
        """
        Get token index ranges for each modality in π0.5.

        π0.5 shares the same PaliGemma token layout as π0.

        Returns:
            Dict with (start, end) ranges
        """
        # π0.5 configuration:
        # Vision: PaliGemma ViT patches (typically 256 patches)
        # State: time_mlp_in/out output (1 token or few tokens)
        # Language: Instruction tokens from PaliGemma

        vision_tokens = 256  # PaliGemma default
        state_tokens = 1     # time_mlp projection output

        ranges = {
            "vision": (0, vision_tokens),
            "state": (vision_tokens, vision_tokens + state_tokens),
            "language": (vision_tokens + state_tokens, -1)
        }

        return ranges

    def run_full_diagnostic(self) -> Dict[str, Any]:
        """
        Run complete diagnostic analysis for π0.5.

        Returns:
            Comprehensive diagnostic report
        """
        structure = self.discover_model_structure()

        self.attach_gradient_hooks()
        self.attach_representation_hooks()
        self.attach_ablation_hooks()
        self.attach_utilization_hooks()

        report = {
            "model_structure": structure,
            "note": "π0.5 uses PaliGemma + Expert Gemma with time_mlp_in/time_mlp_out MLP pair",
            "analysis_focus": [
                "Gradient flow through time_mlp_in/time_mlp_out layers",
                "Representation quality of state/temporal encoding",
                "CKA: State vs Vision features",
                "Attention patterns in expert model",
                "MI(time_mlp_features, actions) via MINE",
                "Ablation: With state vs Without state"
            ],
            "key_questions": [
                "Do time_mlp_in/time_mlp_out learn meaningful representations?",
                "How does flow matching utilize temporal/state features?",
                "Is state→action MI higher than in π0 (which uses single Linear)?",
                "What happens when time_mlp output is zeroed?"
            ],
            "architectural_features": {
                "vision_encoder": "PaliGemma",
                "state_encoder": "MLP pair (time_mlp_in/time_mlp_out)",
                "action_generation": "Flow matching (Expert Gemma)",
                "conditioning": "Proprioceptive state + temporal information",
                "key_advance": "Co-training on heterogeneous data for open-world generalization"
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

        # time_mlp gradients
        if self.gradient_analyzer.layer_profilers:
            time_mlp_profiler = self.gradient_analyzer.layer_profilers.get("time_mlp")
            if time_mlp_profiler:
                summary["time_mlp_gradients"] = time_mlp_profiler.get_summary()

            action_profiler = self.gradient_analyzer.layer_profilers.get("action_projections")
            if action_profiler:
                summary["action_proj_gradients"] = action_profiler.get_summary()

        # State representation features
        if self.representation_analyzer.feature_extractor.features:
            state_features = {
                name: feats
                for name, feats in self.representation_analyzer.feature_extractor.features.items()
                if "time_mlp" in name
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
        print("π0.5 (PI05) MODEL STRUCTURE")
        print("=" * 80)
        print(f"Model: {structure['model_name']}")
        print(f"Has State Encoder: {structure['has_proprio_encoder']}")
        print(f"State Encoder Type: {structure.get('proprio_encoder_type', 'unknown')}")

        print("\nDiscovered Components:")
        for comp_name, attr_name in structure["components"].items():
            print(f"  - {comp_name}: model.{attr_name}")

        if "warning" in structure:
            print(f"\n⚠️  Warning: {structure['warning']}")

        print(f"\nState Encoder Details:")
        print(f"  - Type: MLP pair (time_mlp_in / time_mlp_out)")
        print(f"  - Architecture: PaliGemma + Expert Gemma with flow matching")
        print(f"  - Difference from π0: Replaces single Linear state_proj with MLP pair")

        print("\nCheckpoints:")
        print("  - JAX  (openpi):   pi05_libero (GCS) — served via openpi WebSocket")
        print("  - PyTorch (LeRobot): lerobot/pi05_libero_finetuned  (PI05Policy)")
        print("  -                    lerobot/pi05_base  (generalist, negative control)")

        print("\nArchitectural Features:")
        print("  - Vision: PaliGemma ViT encoder")
        print("  - State: time_mlp_in/time_mlp_out MLP pair")
        print("  - Action Generation: Flow matching with Expert Gemma")
        print("  - Model: π0.5 (3.3B parameters)")
        print("  - Key Advance: Co-training on heterogeneous data for open-world generalization")
        print("\nKey Feature: MLP pair state encoding vs π0's single Linear projection")
        print("=" * 80)
