#!/usr/bin/env python3
"""
Run complete gradient flow analysis on Evo-1 model.
This script is designed to run in the evo1_server conda environment.

Usage:
    # With synthetic data (default)
    conda run -n evo1_server python run_evo1_gradient_analysis.py --checkpoint metaworld
    
    # With real LIBERO data
    conda run -n evo1_server python run_evo1_gradient_analysis.py \
        --checkpoint metaworld \
        --data-path /content/benchmark_observations/libero_libero_90_seed42_50samples.h5 \
        --num-samples 50
    
    # With real MetaWorld data
    conda run -n evo1_server python run_evo1_gradient_analysis.py \
        --checkpoint metaworld \
        --data-path /content/benchmark_observations/metaworld_ml10_seed42_50samples.h5 \
        --num-samples 50

This script:
1. Loads the Evo-1 model
2. Attaches gradient hooks
3. Runs baseline analysis (normal state) on real or synthetic observations
4. Runs ablation analysis (zeroed state encoder output)
5. Compares gradients and outputs results using proper flow matching loss

Results are saved to: /content/evo1_gradient_analysis_results.json

To collect real data, first run:
    python scripts/data_collectors/libero_collector.py --num-samples 50
    python scripts/data_collectors/metaworld_collector.py --num-samples 50
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import json
import numpy as np
from hooks.losses.evo1_loss import compute_evo1_flow_matching_components, evo1_flow_matching_loss_simple

def main():
    parser = argparse.ArgumentParser(description='Run Evo-1 gradient analysis')
    parser.add_argument('--checkpoint', type=str, default='metaworld',
                       choices=['metaworld', 'libero'],
                       help='Which checkpoint to use')
    parser.add_argument('--output', type=str, default='/content/evo1_gradient_analysis_results.json',
                       help='Where to save results')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to real observation HDF5 file (if None, uses synthetic data)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to use from real data')
    args = parser.parse_args()
    
    print('🔬 Evo-1 Gradient Flow Analysis')
    print('='*60)
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Output: {args.output}')
    print(f'Data source: {"Real data: " + args.data_path if args.data_path else "Synthetic (torch.randn)"}')
    if args.data_path:
        print(f'Num samples: {args.num_samples}')
    print('='*60)
    
    # ========================================
    # Step 1: Load Model
    # ========================================
    print('\n[1/5] Loading Evo-1 model...')
    
    evo1_repo_path = '/content/Evo-1'
    checkpoint_dir = Path(f'/content/checkpoints/{args.checkpoint}')
    evo1_code_path = f'{evo1_repo_path}/Evo_1'
    hooks_path = '/content/EmbodiedLLM/MultipleHooksStudy'
    
    # Verify paths
    if not os.path.exists(evo1_repo_path):
        raise FileNotFoundError(f"Evo-1 repository not found at: {evo1_repo_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_dir}")
    if not os.path.exists(hooks_path):
        raise FileNotFoundError(f"Hook framework not found at: {hooks_path}")
    
    # Add to path
    sys.path.insert(0, evo1_code_path)
    sys.path.insert(0, f'{evo1_repo_path}/Evo_1/scripts')
    sys.path.insert(0, hooks_path)
    
    # Import and load model
    from Evo1_server import load_model_and_normalizer
    model, normalizer = load_model_and_normalizer(str(checkpoint_dir))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f'✅ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params on {device}')
    
    # ========================================
    # Step 2: Import Hook Framework
    # ========================================
    print('\n[2/5] Importing hook framework...')
    from hooks.model_specific.evo1_hooks import Evo1Hooks
    
    hook_manager = Evo1Hooks(model)
    structure = hook_manager.discover_model_structure()
    
    print(f'✅ Model structure discovered:')
    print(f'   - Model: {structure["model_name"]}')
    print(f'   - Has proprio encoder: {structure["has_proprio_encoder"]}')
    print(f'   - Encoder type: {structure["proprio_encoder_type"]}')
    
    # ========================================
    # Step 3: Baseline Analysis (Normal State)
    # ========================================
    print('\n[3/5] Running baseline analysis (normal state)...')
    
    # Load real data if provided
    if args.data_path:
        print(f'   Loading real observations from: {args.data_path}')
        import h5py
        
        with h5py.File(args.data_path, 'r') as f:
            images = f['image'][:args.num_samples]
            robot_states = f['robot_state'][:args.num_samples]
            actions = f['action'][:args.num_samples] if 'action' in f else None
            # Convert images from (H, W, C) to (C, H, W) and normalize
            images = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
            robot_states = torch.from_numpy(robot_states).float()
            if actions is not None:
                actions = torch.from_numpy(actions).float()
            
            print(f'   Loaded {len(images)} real observations')
            print(f'   Image shape: {images.shape}')
            print(f'   State shape: {robot_states.shape}')
            if actions is not None:
                print(f'   Action shape: {actions.shape}')
    else:
        print(f'   Using synthetic data (torch.randn)')
        images = torch.randn(args.num_samples, 3, 224, 224)
        robot_states = torch.randn(args.num_samples, 7)
        actions = torch.randn(args.num_samples, 50, 7)  # (batch, horizon, action_dim)
    
    # Move to device
    images = images.to(device).half()
    robot_states = robot_states.to(device).half()
    if actions is not None:
        actions = actions.to(device).half()
    
    # Average gradients across all samples
    total_results_baseline = None
    
    # Attach hooks
    hook_manager.attach_gradient_hooks()
    
    # Iterate through samples
    for i in range(len(images)):
        # Prepare inputs
        inputs = {
            'pixel_values': images[i:i+1],
            'input_ids': torch.randint(0, 50000, (1, 10)).to(device),  # Still random for text
            'robot_state': robot_states[i:i+1]
        }
        
        # Forward + backward with proper flow matching loss
        model.train()
        model.zero_grad()
        outputs = model(**inputs)  # Model predictions
        
        # Compute flow matching loss
        if actions is not None:
            # Use real ground truth actions
            action_gt = actions[i:i+1]
            flow_components = compute_evo1_flow_matching_components(action_gt)
            # Predict velocity field from noisy actions
            # NOTE: This requires model to support forward_with_time interface
            # For now, compute simplified loss using model outputs
            loss_baseline = evo1_flow_matching_loss_simple(outputs, flow_components['u_t']).mean()
        else:
            # Fallback for synthetic data without actions
            # Use random target for demonstration purposes
            target = torch.randn_like(outputs)
            loss_baseline = torch.nn.functional.mse_loss(outputs, target)
        
        loss_baseline.backward()
        
        # Get baseline results
        sample_results = hook_manager.get_results()
        
        # Accumulate results
        if total_results_baseline is None:
            total_results_baseline = sample_results
        else:
            # Average gradients
            for key in total_results_baseline:
                if isinstance(total_results_baseline[key], dict):
                    for subkey in total_results_baseline[key]:
                        if isinstance(total_results_baseline[key][subkey], (int, float)):
                            total_results_baseline[key][subkey] += sample_results[key][subkey]
                elif isinstance(total_results_baseline[key], (int, float)):
                    total_results_baseline[key] += sample_results[key]
        
        hook_manager.reset()
    
    # Average the accumulated results
    num_samples = len(images)
    for key in total_results_baseline:
        if isinstance(total_results_baseline[key], dict):
            for subkey in total_results_baseline[key]:
                if isinstance(total_results_baseline[key][subkey], (int, float)):
                    total_results_baseline[key][subkey] /= num_samples
        elif isinstance(total_results_baseline[key], (int, float)):
            total_results_baseline[key] /= num_samples
    
    results_baseline = total_results_baseline
    gradient_baseline = results_baseline.get('gradient_flow', {})
    
    baseline_integration_grad = None
    if 'layer_profiles' in gradient_baseline:
        layer_profiles = gradient_baseline['layer_profiles']
        if 'integration_module' in layer_profiles:
            baseline_integration_grad = layer_profiles['integration_module']
    
    if baseline_integration_grad:
        baseline_norm = baseline_integration_grad.get('norm', 0.0)
        print(f'✅ Baseline gradient norm (state encoder): {baseline_norm:.6f}')
    else:
        baseline_norm = 0.0
        print('⚠️  integration_module gradients not found in baseline')
    
    # Compute total model gradient norm (baseline only - no ablation needed)
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.data.norm(2).item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    # Calculate baseline contribution ratio (state encoder / total model)
    baseline_contribution_ratio = baseline_norm / total_grad_norm if total_grad_norm > 0 else 0.0
    
    print(f'✅ Total model gradient norm: {total_grad_norm:.6f}')
    print(f'✅ Baseline contribution ratio: {baseline_contribution_ratio:.6f} ({baseline_contribution_ratio*100:.2f}% of total gradients)')
    
    # ========================================
    # Step 4: Ablation Analysis (Zero State Encoder)
    # ========================================
    print('\n[4/5] Running ablation analysis (zero state encoder output)...')
    
    # Attach ablation hook
    ablation_handle = None
    def zero_output_hook(module, input, output):
        return torch.zeros_like(output)
    
    for name, module in model.named_modules():
        if 'integration_module' in name:
            ablation_handle = module.register_forward_hook(zero_output_hook)
            print(f'   Hooked: {name}')
            break
    
    # Reset and run ablation
    hook_manager.reset()
    
    # Average ablation results across all samples
    total_results_ablated = None
    
    # Iterate through the same samples
    for i in range(len(images)):
        # Prepare inputs (same as baseline)
        inputs = {
            'pixel_values': images[i:i+1],
            'input_ids': torch.randint(0, 50000, (1, 10)).to(device),
            'robot_state': robot_states[i:i+1]
        }
        
        model.zero_grad()
        outputs_ablated = model(**inputs)
        
        # Compute flow matching loss (same as baseline)
        if actions is not None:
            action_gt = actions[i:i+1]
            flow_components = compute_evo1_flow_matching_components(action_gt)
            loss_ablated = evo1_flow_matching_loss_simple(outputs_ablated, flow_components['u_t']).mean()
        else:
            # Fallback for synthetic data
            target = torch.randn_like(outputs_ablated)
            loss_ablated = torch.nn.functional.mse_loss(outputs_ablated, target)
        
        loss_ablated.backward()
        
        # Get ablation results
        sample_results = hook_manager.get_results()
        
        # Accumulate results
        if total_results_ablated is None:
            total_results_ablated = sample_results
        else:
            for key in total_results_ablated:
                if isinstance(total_results_ablated[key], dict):
                    for subkey in total_results_ablated[key]:
                        if isinstance(total_results_ablated[key][subkey], (int, float)):
                            total_results_ablated[key][subkey] += sample_results[key][subkey]
                elif isinstance(total_results_ablated[key], (int, float)):
                    total_results_ablated[key] += sample_results[key]
        
        hook_manager.reset()
    
    # Average the accumulated results
    for key in total_results_ablated:
        if isinstance(total_results_ablated[key], dict):
            for subkey in total_results_ablated[key]:
                if isinstance(total_results_ablated[key][subkey], (int, float)):
                    total_results_ablated[key][subkey] /= num_samples
        elif isinstance(total_results_ablated[key], (int, float)):
            total_results_ablated[key] /= num_samples
    
    # Remove hook
    if ablation_handle:
        ablation_handle.remove()
    
    # Get ablation results
    results_ablated = total_results_ablated
    gradient_ablated = results_ablated.get('gradient_flow', {})
    
    ablated_integration_grad = None
    if 'layer_profiles' in gradient_ablated:
        layer_profiles = gradient_ablated['layer_profiles']
        if 'integration_module' in layer_profiles:
            ablated_integration_grad = layer_profiles['integration_module']
    
    if ablated_integration_grad:
        ablated_norm = ablated_integration_grad.get('norm', 0.0)
        print(f'✅ Ablation gradient norm: {ablated_norm:.6f}')
    else:
        ablated_norm = 0.0
        print('⚠️  integration_module gradients not found in ablation')
    
    # ========================================
    # Step 5: Compare and Generate Verdict
    # ========================================
    print('\n[5/5] Comparing gradients and generating verdict...')
    
    # Calculate percentage change
    grad_change_pct = abs(baseline_norm - ablated_norm) / baseline_norm * 100 if baseline_norm > 0 else 0
    
    # Calculate magnitude ratios
    grad_retention_ratio = ablated_norm / baseline_norm if baseline_norm > 0 else 0
    grad_reduction_ratio = (baseline_norm - ablated_norm) / baseline_norm if baseline_norm > 0 else 0
    
    # Calculate absolute and relative magnitudes
    grad_absolute_reduction = baseline_norm - ablated_norm
    
    if grad_change_pct < 10:
        verdict = "❌ UNDERUTILIZED"
        explanation = "When integration_module output is zeroed, gradients barely change. The model doesn't rely on state encoder's contribution."
    elif grad_change_pct < 30:
        verdict = "⚠️ PARTIALLY UTILIZED"
        explanation = "Some gradient sensitivity when state encoder is removed, but the dependency is weak."
    else:
        verdict = "✅ WELL UTILIZED"
        explanation = "Strong gradient response when state encoder output is ablated. The model meaningfully uses state information."
    
    print(f'\n{"="*60}')
    print(f'VERDICT: {verdict}')
    print(f'{"="*60}')
    print(f'Baseline gradient norm:     {baseline_norm:.6f}')
    print(f'Total model gradient norm:  {total_grad_norm:.6f}')
    print(f'Baseline contribution:      {baseline_contribution_ratio:.6f} ({baseline_contribution_ratio*100:.2f}% of total)')
    print(f'\nAblation gradient norm:     {ablated_norm:.6f}')
    print(f'Absolute reduction:         {grad_absolute_reduction:.6f}')
    print(f'Gradient change:            {grad_change_pct:.1f}%')
    print(f'Retention ratio:            {grad_retention_ratio:.3f} ({grad_retention_ratio*100:.1f}% retained)')
    print(f'Reduction ratio:            {grad_reduction_ratio:.3f} ({grad_reduction_ratio*100:.1f}% lost)')
    print(f'\n{explanation}')
    print(f'{"="*60}')
    
    # Save results
    results = {
        'model': 'Evo-1 (0.77B)',
        'checkpoint': args.checkpoint,
        'state_encoder': 'integration_module',
        'ablation_method': 'output_ablation',
        'baseline_grad_norm': float(baseline_norm),
        'total_model_grad_norm': float(total_grad_norm),
        'baseline_contribution_ratio': float(baseline_contribution_ratio),
        'ablated_grad_norm': float(ablated_norm),
        'gradient_absolute_reduction': float(grad_absolute_reduction),
        'gradient_change_pct': float(grad_change_pct),
        'gradient_retention_ratio': float(grad_retention_ratio),
        'gradient_reduction_ratio': float(grad_reduction_ratio),
        'verdict': verdict,
        'explanation': explanation,
        'loss_baseline': float(loss_baseline.item()),
        'loss_ablated': float(loss_ablated.item()),
        'device': str(device),
        'model_structure': structure
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n✅ Results saved to: {args.output}')
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f'\n❌ Analysis failed: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
