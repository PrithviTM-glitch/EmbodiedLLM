"""
VLA Proprioceptive Encoder Diagnostic Analysis - Example Usage

This script demonstrates how to use the diagnostic infrastructure
to analyze proprioceptive encoder utilization in VLA models.

Can be run in Google Colab or local Jupyter notebook.
"""

# %% Setup and Imports
import torch
import numpy as np
from pathlib import Path

# Add MultipleHooksStudy to path if needed
import sys
sys.path.append('/content/EmbodiedLLM/MultipleHooksStudy')

from hooks.model_specific import OpenVLAHooks, OctoHooks, RDTHooks, Pi0Hooks
from analysis import ExperimentCoordinator
from analysis.result_analyzer import ResultAnalyzer

# %% Load Models
print("Loading models...")

# Example: Load Octo model (smallest, good for testing)
from transformers import AutoModel, AutoTokenizer

# For Octo
octo_model = AutoModel.from_pretrained("octo-base")
octo_model.eval()

# For OpenVLA (example)
# openvla_model = AutoModel.from_pretrained("openvla/openvla-7b")

# For RDT-1B (example)
# rdt_model = AutoModel.from_pretrained("robot-transformer/rdt-1b")

# For π0 (example)
# pi0_model = AutoModel.from_pretrained("physical-intelligence/pi0")

print("✓ Models loaded")

# %% Create Hook Adapters
print("\nCreating hook adapters...")

# Create adapters for each model
octo_hooks = OctoHooks(octo_model)

# Discover model structure
structure = octo_hooks.discover_model_structure()
octo_hooks.print_model_info()

print("✓ Hook adapters created")

# %% Manual Hook Testing (Optional)
print("\nTesting manual hook attachment...")

# Attach gradient hooks
octo_hooks.attach_gradient_hooks()

# Run a forward+backward pass
dummy_input = {
    "images": torch.randn(2, 3, 224, 224),
    "proprio_state": torch.randn(2, 7),
    "task_description": torch.zeros(2, 512)
}

output = octo_model(**dummy_input)
loss = output.logits.mean() if hasattr(output, 'logits') else output.mean()
loss.backward()

# Get gradient analysis
grad_report = octo_hooks.gradient_analyzer.get_comprehensive_report()
print("\nGradient Analysis:")
print(f"  Proprio/Vision Ratio: {grad_report['encoder_stats']['proprio_encoder']['proprio_vision_ratio']:.4f}")

# Cleanup
octo_hooks.cleanup()

print("✓ Manual testing complete")

# %% Automated Experiment Coordinator
print("\n" + "="*80)
print("RUNNING FULL DIAGNOSTIC SUITE")
print("="*80)

# Create coordinator
coordinator = ExperimentCoordinator(output_dir="./results")

# Register models
coordinator.register_model("octo", octo_model)
# coordinator.register_model("openvla", openvla_model)
# coordinator.register_model("rdt", rdt_model)
# coordinator.register_model("pi0", pi0_model)

# Create a dummy dataloader (replace with actual dataset)
from torch.utils.data import DataLoader, TensorDataset

dummy_dataset = TensorDataset(
    torch.randn(100, 3, 224, 224),  # images
    torch.randn(100, 7),             # proprio state
    torch.zeros(100, 512),           # task embeddings
    torch.randn(100, 7)              # actions (for eval)
)
data_loader = DataLoader(dummy_dataset, batch_size=8, shuffle=False)

# Define evaluation function for ablation study
def eval_fn(model, batch):
    """Compute performance metric on batch."""
    images, proprio, task, actions = batch
    
    with torch.no_grad():
        outputs = model(images=images, proprio_state=proprio, task_description=task)
        # For ablation, we measure action prediction accuracy
        pred_actions = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # MSE between predicted and actual actions
        mse = torch.nn.functional.mse_loss(pred_actions, actions)
        return mse.item()

# Run full diagnostic suite
results = coordinator.run_full_diagnostic(
    data_loader=data_loader,
    eval_fn=eval_fn,
    gradient_batches=10,
    representation_batches=50,
    ablation_batches=20,
    utilization_batches=30
)

# Print summary
coordinator.print_summary()

print("\n✓ Full diagnostic complete")

# %% Analyze Results
print("\n" + "="*80)
print("ANALYZING RESULTS")
print("="*80)

# Create result analyzer
analyzer = ResultAnalyzer("./results")

# Load all results
analyzer.load_all_results()

# Generate text summary
summary = analyzer.generate_summary_report()
print(summary)

# Save summary
analyzer.save_summary_report("./results/summary_report.txt")

# %% Visualize Results
print("\nGenerating visualizations...")

# Plot gradient comparison
analyzer.plot_gradient_comparison(save_path="./results/gradient_comparison.png")

# Plot representation comparison
analyzer.plot_representation_comparison(save_path="./results/representation_comparison.png")

# Plot ablation results
analyzer.plot_ablation_results(save_path="./results/ablation_results.png")

# Plot attention distribution
analyzer.plot_attention_distribution(save_path="./results/attention_distribution.png")

# Or generate all plots at once
# analyzer.plot_all(output_dir="./results/plots")

print("✓ Visualizations saved")

# %% Key Findings Analysis

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Extract key metrics
if "gradient_analysis" in analyzer.results:
    grad_data = analyzer.results["gradient_analysis"]
    
    print("\n### Gradient Flow ###")
    for model_name, data in grad_data.items():
        encoder_stats = data.get("encoder_stats", {})
        proprio_stats = encoder_stats.get("proprio_encoder", {})
        
        ratio = proprio_stats.get("proprio_vision_ratio", 0)
        print(f"\n{model_name.upper()}:")
        print(f"  Proprio/Vision Gradient Ratio: {ratio:.4f}")
        
        if ratio < 0.1:
            print("  ❌ FINDING: Weak proprioceptive gradients!")
        elif ratio < 0.3:
            print("  ⚠️  FINDING: Moderate proprioceptive gradients")
        else:
            print("  ✅ FINDING: Strong proprioceptive gradients")

if "ablation_study" in analyzer.results:
    abl_data = analyzer.results["ablation_study"]
    
    print("\n### Ablation Impact ###")
    for model_name, data in abl_data.items():
        ranking = data.get("encoder_importance_ranking", [])
        
        print(f"\n{model_name.upper()} Encoder Importance:")
        for i, (encoder, delta) in enumerate(ranking, 1):
            print(f"  {i}. {encoder}: {abs(delta):.4f} performance drop")
        
        # Check if proprio is most important
        if ranking and 'proprio' in ranking[0][0]:
            print("  ✅ FINDING: Proprio encoder is most important!")
        else:
            print("  ❌ FINDING: Proprio encoder is NOT most important")

if "utilization_analysis" in analyzer.results:
    util_data = analyzer.results["utilization_analysis"]
    
    print("\n### Attention Distribution ###")
    for model_name, data in util_data.items():
        attn_dist = data.get("attention_distribution", {})
        
        # Average across layers
        avg_dist = {}
        for layer_dist in attn_dist.values():
            for mod, pct in layer_dist.items():
                avg_dist[mod] = avg_dist.get(mod, []) + [pct]
        
        print(f"\n{model_name.upper()}:")
        for mod, pcts in avg_dist.items():
            avg_pct = np.mean(pcts) * 100
            print(f"  {mod}: {avg_pct:.1f}%")
        
        proprio_pct = np.mean(avg_dist.get('proprio', [0])) * 100
        if proprio_pct < 10:
            print("  ❌ FINDING: Very low attention to proprio!")
        elif proprio_pct < 20:
            print("  ⚠️  FINDING: Low attention to proprio")
        else:
            print("  ✅ FINDING: Reasonable attention to proprio")

print("\n" + "="*80)

# %% Cleanup
coordinator.cleanup_all()
print("\n✓ All hooks cleaned up")
print("✓ Analysis complete!")

# %%
