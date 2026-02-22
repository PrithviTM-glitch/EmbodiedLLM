#!/usr/bin/env python3
"""
RDT-1B Ablation Study

Runs performance-based ablation study on RDT-1B model across benchmarks.
Compares task success rates with normal vs zeroed state encoder.

Usage:
    # Run on LIBERO
    python run_rdt_ablation.py --benchmark libero --tasks libero_90 --num-episodes 50
    
    # Run on VLABench
    python run_rdt_ablation.py --benchmark vlabench --tasks select_box --num-episodes 50
    
    # Run on all benchmarks
    python run_rdt_ablation.py --all-benchmarks --num-episodes 30

Results saved to: /content/rdt_ablation_results.json
"""

import argparse
import sys
import torch
import json
from pathlib import Path
from scripts.ablation_framework import (
    AblationServer, run_ablation_trial, compare_results,
    save_results, print_results_summary
)


# RDT-1B benchmark configuration
RDT_BENCHMARKS = {
    'libero': {
        'tasks': ['libero_90', 'libero_spatial', 'libero_object', 'libero_goal'],
        'port_base': 8765
    },
    'vlabench': {
        'tasks': ['select_box', 'insert_peg', 'add_liquid'],
        'port_base': 9765
    },
    'metaworld': {
        'tasks': ['pick-place', 'push', 'reach', 'drawer-open'],
        'port_base': 10765
    }
}


def load_rdt_model():
    """Load RDT-1B model and identify state encoder."""
    try:
        from rdt.models import RDT1B
        
        print("Loading RDT-1B model...")
        model = RDT1B.from_pretrained('rdt-1b')
        model.eval()
        
        # Identify state_adaptor layer
        state_adaptor = None
        for name, module in model.named_modules():
            if 'state_adaptor' in name or 'state_adapter' in name:
                if isinstance(module, torch.nn.Linear):
                    state_adaptor = module
                    print(f"✅ Found state encoder: {name}")
                    break
        
        if state_adaptor is None:
            raise ValueError("Could not find state_adaptor layer in RDT model")
        
        return model, state_adaptor
        
    except Exception as e:
        print(f"❌ Error loading RDT model: {e}")
        sys.exit(1)


def run_benchmark_ablation(
    model,
    state_encoder,
    benchmark_name: str,
    task_name: str,
    num_episodes: int,
    port: int
) -> dict:
    """Run ablation study on single benchmark task."""
    print(f"\n{'='*60}")
    print(f"Running: {benchmark_name} - {task_name}")
    print(f"{'='*60}")
    
    # Start baseline server (no ablation)
    print("\n[1/4] Starting baseline server...")
    baseline_server = AblationServer(
        model=model,
        state_encoder=state_encoder,
        port=port,
        enable_ablation=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    baseline_server.start_background()
    
    # Run baseline trials
    print(f"\n[2/4] Running {num_episodes} baseline episodes...")
    baseline_results = run_ablation_trial(
        benchmark_name=benchmark_name,
        task_name=task_name,
        num_episodes=num_episodes,
        server_port=port,
        enable_ablation=False
    )
    
    print(f"✅ Baseline success rate: {baseline_results['success_rate']:.1%}")
    
    # Cleanup baseline server
    baseline_server.cleanup()
    
    # Start ablated server (zero state encoder)
    print("\n[3/4] Starting ablated server...")
    ablated_server = AblationServer(
        model=model,
        state_encoder=state_encoder,
        port=port,
        enable_ablation=True,  # Enable zero injection
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    ablated_server.start_background()
    
    # Run ablated trials
    print(f"\n[4/4] Running {num_episodes} ablated episodes...")
    ablated_results = run_ablation_trial(
        benchmark_name=benchmark_name,
        task_name=task_name,
        num_episodes=num_episodes,
        server_port=port,
        enable_ablation=True
    )
    
    print(f"✅ Ablated success rate: {ablated_results['success_rate']:.1%}")
    
    # Cleanup ablated server
    ablated_server.cleanup()
    
    # Compare results
    comparison = compare_results(baseline_results, ablated_results)
    print_results_summary(comparison)
    
    return {
        'benchmark': benchmark_name,
        'task': task_name,
        'baseline': baseline_results,
        'ablated': ablated_results,
        'comparison': comparison
    }


def main():
    parser = argparse.ArgumentParser(description='Run RDT-1B ablation study')
    parser.add_argument('--benchmark', type=str, choices=['libero', 'vlabench', 'metaworld'],
                       help='Which benchmark to run')
    parser.add_argument('--tasks', type=str, nargs='+',
                       help='Specific tasks to run (default: all tasks in benchmark)')
    parser.add_argument('--all-benchmarks', action='store_true',
                       help='Run on all benchmarks')
    parser.add_argument('--num-episodes', type=int, default=50,
                       help='Number of episodes per condition')
    parser.add_argument('--output', type=str, default='/content/rdt_ablation_results.json',
                       help='Where to save results')
    
    args = parser.parse_args()
    
    if not args.all_benchmarks and not args.benchmark:
        parser.error("Must specify --benchmark or --all-benchmarks")
    
    print('='*60)
    print('RDT-1B Ablation Study - Performance Analysis')
    print('='*60)
    
    # Load model
    model, state_encoder = load_rdt_model()
    
    # Determine which benchmarks to run
    if args.all_benchmarks:
        benchmarks_to_run = list(RDT_BENCHMARKS.keys())
    else:
        benchmarks_to_run = [args.benchmark]
    
    # Run ablation studies
    all_results = {}
    
    for benchmark_name in benchmarks_to_run:
        config = RDT_BENCHMARKS[benchmark_name]
        tasks = args.tasks if args.tasks else config['tasks']
        port_base = config['port_base']
        
        benchmark_results = {}
        
        for i, task_name in enumerate(tasks):
            port = port_base + i
            
            task_results = run_benchmark_ablation(
                model=model,
                state_encoder=state_encoder,
                benchmark_name=benchmark_name,
                task_name=task_name,
                num_episodes=args.num_episodes,
                port=port
            )
            
            benchmark_results[task_name] = task_results
        
        all_results[benchmark_name] = benchmark_results
    
    # Save comprehensive results
    save_results(all_results, args.output)
    
    # Print final summary
    print("\n" + "="*60)
    print("ABLATION STUDY COMPLETE - RDT-1B")
    print("="*60)
    
    for benchmark_name, benchmark_results in all_results.items():
        print(f"\n{benchmark_name.upper()}:")
        for task_name, task_results in benchmark_results.items():
            comp = task_results['comparison']
            print(f"  {task_name}:")
            print(f"    Drop: {comp['absolute_drop']:.1%} "
                  f"({comp['relative_drop_percent']:.1f}%), "
                  f"p={comp['p_value']:.4f}, "
                  f"d={comp['cohens_d']:.2f}")
    
    print("\n✅ All ablation studies complete!")
    print(f"Full results saved to: {args.output}")


if __name__ == '__main__':
    main()
