#!/usr/bin/env python3
"""
Run gradient flow analysis on Pi0 model.

Usage:
    # With real LIBERO data
    python run_pi0_gradient_analysis.py \
        --data-path /path/to/libero_data.h5 \
        --num-samples 50
    
    # With real VLABench data
    python run_pi0_gradient_analysis.py \
        --benchmark vlabench \
        --data-path /path/to/vlabench_data.h5 \
        --num-samples 50

This script:
1. Loads the Pi0 model
2. Attaches gradient hooks to state_proj layer
3. Runs baseline analysis (normal state)
4. Runs ablation analysis (zeroed state encoder output)
5. Compares gradients using proper flow matching loss

Results saved to: /content/pi0_gradient_analysis_results.json
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import json
import numpy as np
from hooks.losses.pi0_loss import compute_flow_matching_components, pi0_flow_matching_loss_simple


def main():
    parser = argparse.ArgumentParser(description='Run Pi0 gradient analysis')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to HDF5 data file with observations and actions')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to analyze')
    parser.add_argument('--benchmark', type=str, default='libero',
                       choices=['libero', 'vlabench', 'metaworld'],
                       help='Which benchmark the data is from')
    parser.add_argument('--output', type=str, default='/content/pi0_gradient_analysis_results.json',
                       help='Where to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    
    args = parser.parse_args()
    device = args.device
    
    print('='*80)
    print('Pi0 Gradient Flow Analysis with Flow Matching Loss')
    print('='*80)
    
    # ========================================
    # Step 1: Load Pi0 Model
    # ========================================
    print('\n[1/5] Loading Pi0 model...')
    
    try:
        # Import Pi0 model (adjust import path as needed)
        from openpi import Pi0Policy
        
        # Load pretrained model
        model = Pi0Policy.from_pretrained('pi0-base')
        model = model.to(device)
        model.eval()
        
        print(f'✅ Pi0 model loaded successfully')
        print(f'   Device: {device}')
        
    except Exception as e:
        print(f'❌ Error loading Pi0 model: {e}')
        print('   Make sure openpi package is installed and model weights are available')
        return
    
    # ========================================
    # Step 2: Setup Hooks
    # ========================================
    print('\n[2/5] Importing hook framework...')
    from hooks.model_specific.pi0_hooks import Pi0Hooks
    
    hook_manager = Pi0Hooks(model)
    structure = hook_manager.discover_model_structure()
    
    print(f'✅ Model structure discovered:')
    print(f'   - Model: {structure["model_name"]}')
    print(f'   - Has state encoder: {structure["has_state_encoder"]}')
    print(f'   - Encoder type: {structure["state_encoder_type"]}')
    
    # ========================================
    # Step 3: Load Data
    # ========================================
    print(f'\n[3/5] Loading data from: {args.data_path}')
    
    import h5py
    
    with h5py.File(args.data_path, 'r') as f:
        images = f['image'][:args.num_samples]
        robot_states = f['robot_state'][:args.num_samples]
        actions = f['action'][:args.num_samples]
        
        # Convert to tensors
        images = torch.from_numpy(images).float()
        robot_states = torch.from_numpy(robot_states).float()
        actions = torch.from_numpy(actions).float()
        
        # Normalize images if needed
        if images.max() > 1.0:
            images = images / 255.0
        
        # Transpose images from (H, W, C) to (C, H, W) if needed
        if images.ndim == 4 and images.shape[-1] in [1, 3]:
            images = images.permute(0, 3, 1, 2)
        
        print(f'✅ Loaded {len(images)} observations')
        print(f'   Image shape: {images.shape}')
        print(f'   State shape: {robot_states.shape}')
        print(f'   Action shape: {actions.shape}')
    
    # Move to device
    images = images.to(device)
    robot_states = robot_states.to(device)
    actions = actions.to(device)
    
    # ========================================
    # Step 4: Baseline Analysis
    # ========================================
    print('\n[4/5] Running baseline analysis (normal state)...')
    
    total_results_baseline = None
    hook_manager.attach_gradient_hooks()
    
    for i in range(len(images)):
        # Prepare observation dict
        observation = {
            'image': images[i:i+1],
            'state': robot_states[i:i+1]
        }
        
        # Get ground truth action
        action_gt = actions[i:i+1]
        
        # Forward pass
        model.train()
        model.zero_grad()
        predicted_action = model(observation)
        
        # Compute flow matching loss
        flow_components = compute_flow_matching_components(action_gt)
        
        # For proper flow matching, we need velocity prediction
        # Simplified: use model output as velocity prediction
        loss_baseline = pi0_flow_matching_loss_simple(
            predicted_action, 
            flow_components['u_t']
        ).mean()
        
        loss_baseline.backward()
        
        # Collect results
        sample_results = hook_manager.get_results()
        
        if total_results_baseline is None:
            total_results_baseline = sample_results
        else:
            for key in total_results_baseline:
                if 'gradient_norm' in key or 'gradient_mean' in key:
                    total_results_baseline[key] += sample_results[key]
    
    # Average results
    num_samples = len(images)
    for key in total_results_baseline:
        if 'gradient_norm' in key or 'gradient_mean' in key:
            total_results_baseline[key] /= num_samples
    
    print(f'✅ Baseline analysis complete')
    print(f'   State encoder gradient norm: {total_results_baseline.get("state_encoder_gradient_norm", 0):.6f}')
    
    # ========================================
    # Step 5: Ablation Analysis
    # ========================================
    print('\n[5/5] Running ablation analysis (zeroed state)...')
    
    # Detach and attach ablation hooks
    hook_manager.detach_all()
    hook_manager.attach_representation_hooks()
    hook_manager.attach_gradient_hooks()
    
    # Enable ablation
    hook_manager.enable_ablation('state_encoder')
    
    total_results_ablated = None
    
    for i in range(len(images)):
        observation = {
            'image': images[i:i+1],
            'state': robot_states[i:i+1]
        }
        
        action_gt = actions[i:i+1]
        
        model.zero_grad()
        predicted_action_ablated = model(observation)
        
        # Compute flow matching loss
        flow_components = compute_flow_matching_components(action_gt)
        loss_ablated = pi0_flow_matching_loss_simple(
            predicted_action_ablated,
            flow_components['u_t']
        ).mean()
        
        loss_ablated.backward()
        
        sample_results = hook_manager.get_results()
        
        if total_results_ablated is None:
            total_results_ablated = sample_results
        else:
            for key in total_results_ablated:
                if 'gradient_norm' in key or 'gradient_mean' in key:
                    total_results_ablated[key] += sample_results[key]
    
    # Average results
    for key in total_results_ablated:
        if 'gradient_norm' in key or 'gradient_mean' in key:
            total_results_ablated[key] /= num_samples
    
    print(f'✅ Ablation analysis complete')
    print(f'   State encoder gradient norm: {total_results_ablated.get("state_encoder_gradient_norm", 0):.6f}')
    
    # ========================================
    # Step 6: Compare and Save Results
    # ========================================
    print('\n[6/6] Comparing results...')
    
    comparison = {
        'baseline': total_results_baseline,
        'ablated': total_results_ablated,
        'comparison': {}
    }
    
    # Compute gradient reduction
    for key in total_results_baseline:
        if 'gradient_norm' in key:
            baseline_val = total_results_baseline[key]
            ablated_val = total_results_ablated.get(key, 0)
            
            if baseline_val > 0:
                reduction_pct = ((baseline_val - ablated_val) / baseline_val) * 100
            else:
                reduction_pct = 0
            
            comparison['comparison'][key] = {
                'baseline': float(baseline_val),
                'ablated': float(ablated_val),
                'reduction_percent': float(reduction_pct)
            }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f'\n✅ Results saved to: {output_path}')
    print('\nKey Findings:')
    for key, values in comparison['comparison'].items():
        print(f'   {key}:')
        print(f'      Baseline: {values["baseline"]:.6f}')
        print(f'      Ablated:  {values["ablated"]:.6f}')
        print(f'      Reduction: {values["reduction_percent"]:.2f}%')
    
    hook_manager.detach_all()
    print('\n✅ Analysis complete!')


if __name__ == '__main__':
    main()
