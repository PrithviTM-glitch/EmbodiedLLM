#!/usr/bin/env python3
"""
Run gradient flow analysis on RDT-1B model.

Usage:
    # With real LIBERO data
    python run_rdt_gradient_analysis.py \
        --data-path /path/to/libero_data.h5 \
        --num-samples 50
    
    # With real VLABench data
    python run_rdt_gradient_analysis.py \
        --benchmark vlabench \
        --data-path /path/to/vlabench_data.h5 \
        --num-samples 50

This script:
1. Loads the RDT-1B model
2. Attaches gradient hooks to state_adaptor layer
3. Runs baseline analysis (normal state)
4. Runs ablation analysis (zeroed state encoder output)
5. Compares gradients using proper diffusion loss

Results saved to: /content/rdt_gradient_analysis_results.json
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import json
import numpy as np
from hooks.losses.rdt_loss import rdt_diffusion_loss_simple, create_noise_scheduler


def main():
    parser = argparse.ArgumentParser(description='Run RDT-1B gradient analysis')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to HDF5 data file with observations and actions')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to analyze')
    parser.add_argument('--benchmark', type=str, default='libero',
                       choices=['libero', 'vlabench', 'metaworld'],
                       help='Which benchmark the data is from')
    parser.add_argument('--output', type=str, default='/content/rdt_gradient_analysis_results.json',
                       help='Where to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--pred-type', type=str, default='epsilon',
                       choices=['epsilon', 'sample'],
                       help='Diffusion prediction type')
    parser.add_argument('--num-train-timesteps', type=int, default=100,
                       help='Number of diffusion timesteps')
    
    args = parser.parse_args()
    device = args.device
    
    print('='*80)
    print('RDT-1B Gradient Flow Analysis with Diffusion Loss')
    print('='*80)
    
    # ========================================
    # Step 1: Load RDT Model
    # ========================================
    print('\n[1/6] Loading RDT-1B model...')
    
    try:
        # Import RDT model (adjust import path as needed)
        # This assumes RDT code is available in the environment
        from rdt.models import RDT1B
        
        # Load pretrained model
        model = RDT1B.from_pretrained('rdt-1b')
        model = model.to(device)
        model.eval()
        
        print(f'✅ RDT-1B model loaded successfully')
        print(f'   Device: {device}')
        
    except Exception as e:
        print(f'❌ Error loading RDT model: {e}')
        print('   Make sure RDT package is installed and model weights are available')
        return
    
    # ========================================
    # Step 2: Setup Noise Scheduler
    # ========================================
    print('\n[2/6] Setting up noise scheduler...')
    
    noise_scheduler = create_noise_scheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule='squaredcos_cap_v2'
    )
    
    # Attach to model for convenience
    model.noise_scheduler = noise_scheduler
    model.num_train_timesteps = args.num_train_timesteps
    
    print(f'✅ Noise scheduler created')
    print(f'   Timesteps: {args.num_train_timesteps}')
    print(f'   Beta schedule: squaredcos_cap_v2')
    
    # ========================================
    # Step 3: Setup Hooks
    # ========================================
    print('\n[3/6] Importing hook framework...')
    from hooks.model_specific.rdt_hooks import RDTHooks
    
    hook_manager = RDTHooks(model)
    structure = hook_manager.discover_model_structure()
    
    print(f'✅ Model structure discovered:')
    print(f'   - Model: {structure["model_name"]}')
    print(f'   - Has proprio encoder: {structure["has_proprio_encoder"]}')
    print(f'   - Encoder type: {structure["proprio_encoder_type"]}')
    
    # ========================================
    # Step 4: Load Data
    # ========================================
    print(f'\n[4/6] Loading data from: {args.data_path}')
    
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
    # Step 5: Baseline Analysis
    # ========================================
    print('\n[5/6] Running baseline analysis (normal state)...')
    
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
        
        # Sample noise and timesteps for diffusion
        batch_size = action_gt.shape[0]
        noise = torch.randn_like(action_gt)
        timesteps = torch.randint(
            0, args.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        # Add noise to actions
        noisy_action = noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        # Forward pass
        model.train()
        model.zero_grad()
        
        # Model predicts noise or clean sample
        predicted = model(observation, noisy_action, timesteps)
        
        # Determine target based on prediction type
        if args.pred_type == 'epsilon':
            target = noise
        else:  # 'sample'
            target = action_gt
        
        # Compute diffusion loss
        loss_baseline = rdt_diffusion_loss_simple(predicted, target)
        
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
    # Step 6: Ablation Analysis
    # ========================================
    print('\n[6/7] Running ablation analysis (zeroed state)...')

    # Clean up baseline hooks, then re-attach for ablation run
    hook_manager.cleanup()
    hook_manager.attach_representation_hooks()
    hook_manager.attach_gradient_hooks()

    # Enable ablation via direct forward hook on state_adaptor (proprio_encoder)
    ablation_handle = None
    if hook_manager.proprio_encoder is not None:
        ablation_handle = hook_manager.proprio_encoder.register_forward_hook(
            lambda m, i, o: torch.zeros_like(o)
        )
        print('   \u2705 Zero-ablation hook attached to state_adaptor')
    else:
        print('   \u26a0\ufe0f  proprio_encoder not found \u2014 ablation skipped')

    total_results_ablated = None
    
    for i in range(len(images)):
        observation = {
            'image': images[i:i+1],
            'state': robot_states[i:i+1]
        }
        
        action_gt = actions[i:i+1]
        
        # Sample noise and timesteps
        batch_size = action_gt.shape[0]
        noise = torch.randn_like(action_gt)
        timesteps = torch.randint(
            0, args.num_train_timesteps,
            (batch_size,),
            device=device
        ).long()
        
        noisy_action = noise_scheduler.add_noise(action_gt, noise, timesteps)
        
        model.zero_grad()
        predicted_ablated = model(observation, noisy_action, timesteps)
        
        # Determine target
        if args.pred_type == 'epsilon':
            target = noise
        else:
            target = action_gt
        
        loss_ablated = rdt_diffusion_loss_simple(predicted_ablated, target)
        
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
    
    # Remove ablation hook
    if ablation_handle is not None:
        ablation_handle.remove()

    print(f'✅ Ablation analysis complete')
    print(f'   State encoder gradient norm: {total_results_ablated.get("state_encoder_gradient_norm", 0):.6f}')
    
    # ========================================
    # Step 7: Compare and Save Results
    # ========================================
    print('\n[7/7] Comparing results...')
    
    comparison = {
        'config': {
            'model': 'RDT-1B',
            'pred_type': args.pred_type,
            'num_train_timesteps': args.num_train_timesteps,
            'num_samples': args.num_samples,
            'benchmark': args.benchmark
        },
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
    
    hook_manager.cleanup()
    print('\n✅ Analysis complete!')


if __name__ == '__main__':
    main()
