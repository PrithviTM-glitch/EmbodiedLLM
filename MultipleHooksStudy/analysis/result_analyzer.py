"""
Result Analysis Utilities

Tools for analyzing and visualizing diagnostic results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
import json


class ResultAnalyzer:
    """
    Analyzes and visualizes diagnostic experiment results.
    
    Usage:
        analyzer = ResultAnalyzer("results/")
        analyzer.load_all_results()
        analyzer.plot_gradient_comparison()
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize result analyzer.
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        
    def load_all_results(self):
        """Load all results from directory."""
        for json_file in self.results_dir.glob("*.json"):
            if json_file.name == "experiment_metadata.json":
                continue
            
            experiment_type = json_file.stem.rsplit('_', 2)[0]  # Remove timestamp
            
            with open(json_file, 'r') as f:
                self.results[experiment_type] = json.load(f)
        
        print(f"Loaded {len(self.results)} result files")
        for exp_type in self.results.keys():
            print(f"  - {exp_type}")
    
    def plot_gradient_comparison(self, save_path: Optional[str] = None):
        """
        Plot gradient flow comparison across models.
        
        Args:
            save_path: Optional path to save figure
        """
        if "gradient_analysis" not in self.results:
            print("No gradient analysis results found")
            return
        
        data = self.results["gradient_analysis"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Gradient Flow Analysis Across Models", fontsize=16, fontweight='bold')
        
        # Plot 1: Encoder gradient norms
        ax = axes[0, 0]
        models = list(data.keys())
        encoders = ['vision', 'language', 'proprio']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, encoder in enumerate(encoders):
            norms = []
            for model in models:
                encoder_stats = data[model].get("encoder_stats", {}).get(f"{encoder}_encoder", {})
                norm = encoder_stats.get("mean_norm", 0)
                norms.append(norm)
            
            ax.bar(x + i*width, norms, width, label=encoder.capitalize())
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Mean Gradient Norm")
        ax.set_title("Encoder-Level Gradient Norms")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Proprio/Vision gradient ratio
        ax = axes[0, 1]
        ratios = []
        for model in models:
            encoder_stats = data[model].get("encoder_stats", {})
            proprio_stats = encoder_stats.get("proprio_encoder", {})
            ratio = proprio_stats.get("proprio_vision_ratio", 0)
            ratios.append(ratio)
        
        colors = ['green' if r > 0.3 else 'orange' if r > 0.1 else 'red' for r in ratios]
        ax.barh(models, ratios, color=colors)
        ax.set_xlabel("Proprio/Vision Gradient Ratio")
        ax.set_title("Proprioceptive Encoder Gradient Strength\n(relative to vision)")
        ax.axvline(0.3, color='green', linestyle='--', alpha=0.5, label='Good (>0.3)')
        ax.axvline(0.1, color='orange', linestyle='--', alpha=0.5, label='Weak (>0.1)')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: Vanishing point location
        ax = axes[1, 0]
        vanishing_points = []
        for model in models:
            layer_profiles = data[model].get("layer_profiles", {})
            # Find first profile with vanishing point
            vp = None
            for profile_name, profile_data in layer_profiles.items():
                vp = profile_data.get("vanishing_point")
                if vp is not None and vp != "No vanishing point":
                    break
            vanishing_points.append(vp if isinstance(vp, int) else -1)
        
        colors = ['green' if vp == -1 else 'red' if vp < 3 else 'orange' 
                  for vp in vanishing_points]
        bars = ax.barh(models, vanishing_points, color=colors)
        ax.set_xlabel("Vanishing Point (layer index)")
        ax.set_title("Gradient Vanishing Point Location\n(-1 = no vanishing)")
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 4: Gradient decay rate
        ax = axes[1, 1]
        decay_rates = []
        for model in models:
            layer_profiles = data[model].get("layer_profiles", {})
            # Calculate average decay across profiles
            decays = []
            for profile_data in layer_profiles.values():
                decay = profile_data.get("gradient_decay", {}).get("percent_decrease", 0)
                if decay > 0:
                    decays.append(decay)
            decay_rates.append(np.mean(decays) if decays else 0)
        
        colors = ['green' if d < 50 else 'orange' if d < 80 else 'red' for d in decay_rates]
        ax.barh(models, decay_rates, color=colors)
        ax.set_xlabel("Gradient Decay (%)")
        ax.set_title("Average Gradient Decay Across Layers")
        ax.axvline(50, color='green', linestyle='--', alpha=0.5, label='Good (<50%)')
        ax.axvline(80, color='orange', linestyle='--', alpha=0.5, label='Severe (>80%)')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
    
    def plot_representation_comparison(self, save_path: Optional[str] = None):
        """
        Plot representation quality comparison.
        
        Args:
            save_path: Optional path to save figure
        """
        if "representation_analysis" not in self.results:
            print("No representation analysis results found")
            return
        
        data = self.results["representation_analysis"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Representation Quality Analysis", fontsize=16, fontweight='bold')
        
        models = list(data.keys())
        
        # Plot 1: Effective rank
        ax = axes[0, 0]
        encoders = ['vision_encoder', 'proprio_encoder', 'language_encoder']
        encoder_labels = ['Vision', 'Proprio', 'Language']
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, (encoder, label) in enumerate(zip(encoders, encoder_labels)):
            ranks = []
            for model in models:
                rank_data = data[model].get("effective_rank", {}).get(encoder, {})
                rank = rank_data.get("effective_rank", 0)
                ranks.append(rank)
            
            ax.bar(x + i*width, ranks, width, label=label)
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Effective Rank")
        ax.set_title("Effective Dimensionality of Encoder Outputs")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Utilization percentage
        ax = axes[0, 1]
        for i, (encoder, label) in enumerate(zip(encoders, encoder_labels)):
            utilizations = []
            for model in models:
                rank_data = data[model].get("effective_rank", {}).get(encoder, {})
                util = rank_data.get("utilization_percent", 0)
                utilizations.append(util)
            
            ax.bar(x + i*width, utilizations, width, label=label)
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Utilization (%)")
        ax.set_title("Dimension Utilization (Effective Rank / Total Dims)")
        ax.set_xticks(x + width)
        ax.set_xticklabels(models)
        ax.legend()
        ax.axhline(50, color='red', linestyle='--', alpha=0.3, label='50% threshold')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 3: CKA similarity heatmaps (for first model)
        ax = axes[1, 0]
        first_model = models[0]
        cka_matrix = data[first_model].get("cka_similarity", {})
        
        if cka_matrix:
            # Convert to numpy array
            feature_names = list(cka_matrix.keys())
            n = len(feature_names)
            matrix = np.zeros((n, n))
            
            for i, name1 in enumerate(feature_names):
                for j, name2 in enumerate(feature_names):
                    matrix[i, j] = cka_matrix.get(name1, {}).get(name2, 0)
            
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
                       xticklabels=feature_names, yticklabels=feature_names,
                       ax=ax, vmin=0, vmax=1)
            ax.set_title(f"CKA Similarity Matrix ({first_model})")
        
        # Plot 4: High redundancy pairs count
        ax = axes[1, 1]
        redundancy_counts = []
        for model in models:
            pairs = data[model].get("high_redundancy_pairs", [])
            redundancy_counts.append(len(pairs))
        
        colors = ['green' if c == 0 else 'orange' if c < 3 else 'red' 
                  for c in redundancy_counts]
        ax.bar(models, redundancy_counts, color=colors)
        ax.set_ylabel("Number of Redundant Pairs")
        ax.set_title("High Redundancy Pairs (CKA > 0.7)")
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
    
    def plot_ablation_results(self, save_path: Optional[str] = None):
        """
        Plot ablation study results.
        
        Args:
            save_path: Optional path to save figure
        """
        if "ablation_study" not in self.results:
            print("No ablation study results found")
            return
        
        data = self.results["ablation_study"]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Ablation Study Results", fontsize=16, fontweight='bold')
        
        models = list(data.keys())
        
        # Plot 1: Performance drop by encoder
        ax = axes[0]
        
        # Collect deltas for each encoder across models
        encoder_deltas = {}
        for model in models:
            deltas = data[model].get("ablation_deltas", {})
            for encoder, delta in deltas.items():
                if encoder not in encoder_deltas:
                    encoder_deltas[encoder] = []
                encoder_deltas[encoder].append(abs(delta))
        
        # Plot grouped bars
        x = np.arange(len(models))
        width = 0.2
        encoders = list(encoder_deltas.keys())
        
        for i, encoder in enumerate(encoders):
            values = encoder_deltas[encoder]
            ax.bar(x + i*width, values, width, label=encoder.replace('_encoder', ''))
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Performance Drop (absolute)")
        ax.set_title("Performance Impact of Encoder Ablation")
        ax.set_xticks(x + width * (len(encoders) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Encoder importance ranking
        ax = axes[1]
        
        # For each model, show ranking
        for i, model in enumerate(models):
            ranking = data[model].get("encoder_importance_ranking", [])
            
            # Extract encoder names and deltas
            if ranking:
                encoders = [item[0].replace('_encoder', '') for item in ranking]
                importances = [abs(item[1]) for item in ranking]
                
                y_positions = np.arange(len(encoders)) + i * (len(encoders) + 0.5)
                ax.barh(y_positions, importances, height=0.3, label=model)
                
                # Add encoder labels for first model only
                if i == 0:
                    for j, encoder in enumerate(encoders):
                        ax.text(-0.01, y_positions[j], encoder, 
                               ha='right', va='center', fontsize=8)
        
        ax.set_xlabel("Importance (ablation delta)")
        ax.set_title("Encoder Importance Ranking by Model")
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
    
    def plot_attention_distribution(self, save_path: Optional[str] = None):
        """
        Plot attention distribution across modalities.
        
        Args:
            save_path: Optional path to save figure
        """
        if "utilization_analysis" not in self.results:
            print("No utilization analysis results found")
            return
        
        data = self.results["utilization_analysis"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Attention Distribution Across Modalities", fontsize=16, fontweight='bold')
        
        models = list(data.keys())
        modalities = ['vision', 'proprio', 'language']
        
        # Collect attention percentages
        attention_data = {mod: [] for mod in modalities}
        
        for model in models:
            attn_dist = data[model].get("attention_distribution", {})
            for mod in modalities:
                # Average across layers
                layer_attentions = []
                for layer_name, layer_dist in attn_dist.items():
                    if mod in layer_dist:
                        layer_attentions.append(layer_dist[mod])
                
                avg_attn = np.mean(layer_attentions) if layer_attentions else 0
                attention_data[mod].append(avg_attn * 100)  # Convert to percentage
        
        # Stacked bar chart
        x = np.arange(len(models))
        bottom = np.zeros(len(models))
        
        colors = {'vision': '#1f77b4', 'proprio': '#ff7f0e', 'language': '#2ca02c'}
        
        for mod in modalities:
            values = attention_data[mod]
            ax.bar(x, values, bottom=bottom, label=mod.capitalize(), color=colors[mod])
            bottom += values
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Attention Percentage (%)")
        ax.set_title("Average Attention Distribution by Modality")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add 33% reference line
        ax.axhline(33.33, color='red', linestyle='--', alpha=0.3, label='Equal (33.3%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        plt.show()
    
    def generate_summary_report(self) -> str:
        """
        Generate text summary of all results.
        
        Returns:
            Formatted summary string
        """
        report = []
        report.append("="*80)
        report.append("DIAGNOSTIC ANALYSIS SUMMARY")
        report.append("="*80)
        
        # Gradient Analysis
        if "gradient_analysis" in self.results:
            report.append("\n### GRADIENT FLOW ANALYSIS ###\n")
            data = self.results["gradient_analysis"]
            
            for model, results in data.items():
                report.append(f"\n[{model.upper()}]")
                
                encoder_stats = results.get("encoder_stats", {})
                for encoder_name, stats in encoder_stats.items():
                    mean_norm = stats.get("mean_norm", 0)
                    report.append(f"  {encoder_name}: {mean_norm:.4f}")
                
                # Proprio/Vision ratio
                proprio_stats = encoder_stats.get("proprio_encoder", {})
                ratio = proprio_stats.get("proprio_vision_ratio", 0)
                report.append(f"  Proprio/Vision Ratio: {ratio:.4f}")
                
                if ratio < 0.1:
                    report.append("    ⚠️  WARNING: Weak proprio gradients!")
        
        # Representation Analysis
        if "representation_analysis" in self.results:
            report.append("\n\n### REPRESENTATION QUALITY ###\n")
            data = self.results["representation_analysis"]
            
            for model, results in data.items():
                report.append(f"\n[{model.upper()}]")
                
                rank_data = results.get("effective_rank", {})
                for encoder, stats in rank_data.items():
                    eff_rank = stats.get("effective_rank", 0)
                    util_pct = stats.get("utilization_percent", 0)
                    report.append(f"  {encoder}: Rank={eff_rank:.1f}, Util={util_pct:.1f}%")
                
                redundant = results.get("high_redundancy_pairs", [])
                if redundant:
                    report.append(f"  ⚠️  {len(redundant)} redundant pairs detected")
        
        # Ablation Study
        if "ablation_study" in self.results:
            report.append("\n\n### ABLATION STUDY ###\n")
            data = self.results["ablation_study"]
            
            for model, results in data.items():
                report.append(f"\n[{model.upper()}]")
                
                ranking = results.get("encoder_importance_ranking", [])
                report.append("  Importance Ranking:")
                for i, (encoder, delta) in enumerate(ranking, 1):
                    report.append(f"    {i}. {encoder}: {abs(delta):.4f}")
        
        # Utilization Analysis
        if "utilization_analysis" in self.results:
            report.append("\n\n### DOWNSTREAM UTILIZATION ###\n")
            data = self.results["utilization_analysis"]
            
            for model, results in data.items():
                report.append(f"\n[{model.upper()}]")
                
                attn_dist = results.get("attention_distribution", {})
                # Average across layers
                avg_dist = {}
                for layer_dist in attn_dist.values():
                    for mod, pct in layer_dist.items():
                        avg_dist[mod] = avg_dist.get(mod, []) + [pct]
                
                report.append("  Average Attention Distribution:")
                for mod, pcts in avg_dist.items():
                    avg_pct = np.mean(pcts) * 100
                    report.append(f"    {mod}: {avg_pct:.1f}%")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def save_summary_report(self, output_path: str):
        """Save summary report to file."""
        report = self.generate_summary_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to {output_path}")
    
    def plot_all(self, output_dir: Optional[str] = None):
        """Generate all plots."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        print("Generating plots...")
        
        self.plot_gradient_comparison(
            save_path=output_dir / "gradient_comparison.png" if output_dir else None
        )
        
        self.plot_representation_comparison(
            save_path=output_dir / "representation_comparison.png" if output_dir else None
        )
        
        self.plot_ablation_results(
            save_path=output_dir / "ablation_results.png" if output_dir else None
        )
        
        self.plot_attention_distribution(
            save_path=output_dir / "attention_distribution.png" if output_dir else None
        )
        
        print("✓ All plots generated")
