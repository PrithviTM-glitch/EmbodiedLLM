"""
Experiment Coordinator

Coordinates running diagnostic experiments across all VLA models.
Ensures consistent experimental protocol and result collection.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime

from ..hooks.model_specific import OpenVLAHooks, OctoHooks, RDTHooks, Pi0Hooks


class ExperimentCoordinator:
    """
    Coordinates diagnostic experiments across multiple VLA models.
    
    Usage:
        coordinator = ExperimentCoordinator()
        coordinator.register_model("octo", octo_model)
        results = coordinator.run_gradient_analysis(dataset)
    """
    
    def __init__(self, output_dir: str = "./results"):
        """
        Initialize experiment coordinator.
        
        Args:
            output_dir: Directory to save results
        """
        self.models = {}
        self.hook_adapters = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Track experiment metadata
        self.experiment_metadata = {
            "start_time": None,
            "end_time": None,
            "models": [],
            "experiments_run": []
        }
    
    def register_model(self, model_name: str, model, hook_adapter_class=None):
        """
        Register a model for experiments.
        
        Args:
            model_name: Name identifier (openvla, octo, rdt, pi0)
            model: Model instance
            hook_adapter_class: Optional custom hook adapter (auto-detected if None)
        """
        self.models[model_name] = model
        
        # Auto-detect hook adapter if not provided
        if hook_adapter_class is None:
            adapter_map = {
                "openvla": OpenVLAHooks,
                "octo": OctoHooks,
                "rdt": RDTHooks,
                "pi0": Pi0Hooks
            }
            hook_adapter_class = adapter_map.get(model_name.lower())
            
            if hook_adapter_class is None:
                raise ValueError(f"Unknown model '{model_name}'. Provide hook_adapter_class explicitly.")
        
        # Create hook adapter instance
        self.hook_adapters[model_name] = hook_adapter_class(model)
        
        # Add to metadata
        if model_name not in self.experiment_metadata["models"]:
            self.experiment_metadata["models"].append(model_name)
        
        print(f"✓ Registered model: {model_name}")
    
    def run_gradient_analysis(
        self,
        data_loader,
        num_batches: int = 10,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run gradient flow analysis across models.
        
        Args:
            data_loader: DataLoader providing batches
            num_batches: Number of batches to analyze
            models: List of model names (None = all registered models)
        
        Returns:
            Dict mapping model_name → gradient analysis results
        """
        print("\n" + "="*80)
        print("GRADIENT FLOW ANALYSIS")
        print("="*80)
        
        models = models or list(self.models.keys())
        results = {}
        
        for model_name in models:
            print(f"\n[{model_name.upper()}]")
            
            model = self.models[model_name]
            adapter = self.hook_adapters[model_name]
            
            # Attach gradient hooks
            adapter.attach_gradient_hooks()
            
            # Run forward+backward on batches
            model.train()
            for i, batch in enumerate(data_loader):
                if i >= num_batches:
                    break
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.get('loss') or outputs.get('logits').mean()
                
                # Backward pass
                loss.backward()
                
                # Clear gradients for next iteration
                model.zero_grad()
                
                if (i + 1) % 5 == 0:
                    print(f"  Processed batch {i+1}/{num_batches}")
            
            # Collect results
            results[model_name] = adapter.gradient_analyzer.get_comprehensive_report()
            
            # Cleanup
            adapter.gradient_analyzer.remove_all()
            
            print(f"  ✓ Gradient analysis complete")
        
        # Save results
        self._save_results("gradient_analysis", results)
        self.experiment_metadata["experiments_run"].append("gradient_analysis")
        
        return results
    
    def run_representation_analysis(
        self,
        data_loader,
        num_batches: int = 50,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run representation quality analysis across models.
        
        Args:
            data_loader: DataLoader providing batches
            num_batches: Number of batches to collect features from
            models: List of model names (None = all)
        
        Returns:
            Dict mapping model_name → representation analysis results
        """
        print("\n" + "="*80)
        print("REPRESENTATION QUALITY ANALYSIS")
        print("="*80)
        
        models = models or list(self.models.keys())
        results = {}
        
        for model_name in models:
            print(f"\n[{model_name.upper()}]")
            
            model = self.models[model_name]
            adapter = self.hook_adapters[model_name]
            
            # Attach representation hooks
            adapter.attach_representation_hooks()
            
            # Run forward passes to collect features
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= num_batches:
                        break
                    
                    outputs = model(**batch)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed batch {i+1}/{num_batches}")
            
            # Compute CKA similarity
            print("  Computing CKA similarity...")
            similarity_matrix = adapter.representation_analyzer.cka_analyzer.get_similarity_matrix()
            
            # Compute effective rank
            print("  Computing effective rank...")
            rank_summary = {}
            for name, calculator in adapter.representation_analyzer.rank_calculators.items():
                rank_summary[name] = calculator.get_summary()
            
            # Compile results
            results[model_name] = {
                "cka_similarity": similarity_matrix,
                "effective_rank": rank_summary,
                "high_redundancy_pairs": adapter.representation_analyzer.find_redundant_pairs(threshold=0.7)
            }
            
            # Cleanup
            adapter.representation_analyzer.remove_hooks()
            
            print(f"  ✓ Representation analysis complete")
        
        # Save results
        self._save_results("representation_analysis", results)
        self.experiment_metadata["experiments_run"].append("representation_analysis")
        
        return results
    
    def run_ablation_study(
        self,
        data_loader,
        eval_fn: Callable,
        num_batches: int = 20,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run ablation study across models.
        
        Args:
            data_loader: DataLoader providing batches
            eval_fn: Function(model, batch) → performance metric
            num_batches: Number of batches for evaluation
            models: List of model names (None = all)
        
        Returns:
            Dict mapping model_name → ablation results
        """
        print("\n" + "="*80)
        print("ABLATION STUDY")
        print("="*80)
        
        models = models or list(self.models.keys())
        results = {}
        
        for model_name in models:
            print(f"\n[{model_name.upper()}]")
            
            model = self.models[model_name]
            adapter = self.hook_adapters[model_name]
            
            # Setup ablation coordinator
            adapter.attach_ablation_hooks()
            
            # Run standard ablation suite
            print("  Running ablation configurations...")
            ablation_configs = adapter.ablation_coordinator.run_standard_ablations()
            
            # Evaluate each configuration
            config_results = {}
            for config_name, config in ablation_configs.items():
                print(f"  Evaluating: {config_name}")
                
                # Apply ablation
                for encoder_name in config["ablated_encoders"]:
                    adapter.ablation_coordinator.ablation_manager.ablate_only([encoder_name])
                
                # Evaluate
                model.eval()
                metrics = []
                with torch.no_grad():
                    for i, batch in enumerate(data_loader):
                        if i >= num_batches:
                            break
                        metric = eval_fn(model, batch)
                        metrics.append(metric)
                
                config_results[config_name] = {
                    "mean_performance": np.mean(metrics),
                    "std_performance": np.std(metrics)
                }
                
                # Reset ablation
                adapter.ablation_coordinator.ablation_manager.reset_all()
            
            # Compute deltas and ranking
            deltas = adapter.ablation_coordinator.compute_ablation_deltas(
                config_results,
                baseline_key="full_model"
            )
            ranking = adapter.ablation_coordinator.get_encoder_importance_ranking(deltas)
            
            results[model_name] = {
                "configurations": config_results,
                "ablation_deltas": deltas,
                "encoder_importance_ranking": ranking
            }
            
            # Cleanup
            adapter.ablation_coordinator.cleanup()
            
            print(f"  ✓ Ablation study complete")
        
        # Save results
        self._save_results("ablation_study", results)
        self.experiment_metadata["experiments_run"].append("ablation_study")
        
        return results
    
    def run_utilization_analysis(
        self,
        data_loader,
        num_batches: int = 30,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run downstream utilization analysis across models.
        
        Args:
            data_loader: DataLoader providing batches
            num_batches: Number of batches to analyze
            models: List of model names (None = all)
        
        Returns:
            Dict mapping model_name → utilization results
        """
        print("\n" + "="*80)
        print("DOWNSTREAM UTILIZATION ANALYSIS")
        print("="*80)
        
        models = models or list(self.models.keys())
        results = {}
        
        for model_name in models:
            print(f"\n[{model_name.upper()}]")
            
            model = self.models[model_name]
            adapter = self.hook_adapters[model_name]
            
            # Attach utilization hooks
            adapter.attach_utilization_hooks()
            
            # Set modality ranges
            modality_ranges = adapter.get_modality_token_ranges()
            adapter.utilization_analyzer.attention_tracker.set_modality_ranges(modality_ranges)
            
            # Run forward passes
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(data_loader):
                    if i >= num_batches:
                        break
                    
                    outputs = model(**batch)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed batch {i+1}/{num_batches}")
            
            # Compute modality attention percentages
            print("  Computing attention distribution...")
            attention_dist = adapter.utilization_analyzer.attention_tracker.compute_modality_attention()
            
            # Find stagnant layers
            print("  Finding stagnant layers...")
            stagnant = adapter.utilization_analyzer.similarity_tracker.find_stagnant_layers(threshold=0.95)
            
            # Compile results
            results[model_name] = {
                "attention_distribution": attention_dist,
                "stagnant_layers": stagnant,
                "modality_ranges": modality_ranges
            }
            
            # Cleanup
            adapter.utilization_analyzer.cleanup()
            
            print(f"  ✓ Utilization analysis complete")
        
        # Save results
        self._save_results("utilization_analysis", results)
        self.experiment_metadata["experiments_run"].append("utilization_analysis")
        
        return results
    
    def run_full_diagnostic(
        self,
        data_loader,
        eval_fn: Callable,
        gradient_batches: int = 10,
        representation_batches: int = 50,
        ablation_batches: int = 20,
        utilization_batches: int = 30,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete diagnostic suite on all registered models.
        
        Args:
            data_loader: DataLoader providing batches
            eval_fn: Evaluation function for ablation study
            *_batches: Number of batches for each analysis type
            models: List of model names (None = all)
        
        Returns:
            Complete diagnostic results
        """
        self.experiment_metadata["start_time"] = datetime.now().isoformat()
        
        print("\n" + "="*80)
        print("FULL DIAGNOSTIC SUITE")
        print("="*80)
        print(f"Models: {', '.join(models or self.models.keys())}")
        print(f"Output Directory: {self.output_dir}")
        print("="*80)
        
        results = {
            "gradient_analysis": self.run_gradient_analysis(
                data_loader, gradient_batches, models
            ),
            "representation_analysis": self.run_representation_analysis(
                data_loader, representation_batches, models
            ),
            "ablation_study": self.run_ablation_study(
                data_loader, eval_fn, ablation_batches, models
            ),
            "utilization_analysis": self.run_utilization_analysis(
                data_loader, utilization_batches, models
            )
        }
        
        self.experiment_metadata["end_time"] = datetime.now().isoformat()
        
        # Save metadata
        self._save_metadata()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC SUITE COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        
        return results
    
    def _save_results(self, experiment_name: str, results: Dict[str, Any]):
        """Save experiment results to JSON."""
        output_file = self.output_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n  Results saved to: {output_file}")
    
    def _save_metadata(self):
        """Save experiment metadata."""
        metadata_file = self.output_dir / "experiment_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
    
    def _make_serializable(self, obj):
        """Convert tensors/arrays to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def cleanup_all(self):
        """Remove all hooks from all models."""
        for model_name, adapter in self.hook_adapters.items():
            adapter.cleanup()
        print("✓ All hooks cleaned up")
    
    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Registered Models: {len(self.models)}")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
        
        print(f"\nExperiments Run: {len(self.experiment_metadata['experiments_run'])}")
        for exp in self.experiment_metadata["experiments_run"]:
            print(f"  - {exp}")
        
        if self.experiment_metadata["start_time"]:
            print(f"\nStart Time: {self.experiment_metadata['start_time']}")
        if self.experiment_metadata["end_time"]:
            print(f"End Time: {self.experiment_metadata['end_time']}")
        
        print(f"\nResults Directory: {self.output_dir}")
        print("="*80)
