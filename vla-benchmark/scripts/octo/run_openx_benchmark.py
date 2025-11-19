#!/usr/bin/env python3
"""
Run OCTO on Open X-Embodiment benchmark.

This script demonstrates the complete benchmark evaluation pipeline:
1. Initialize OCTO adapter
2. Load OpenX benchmark (Bridge V2 dataset)
3. Run evaluation
4. Save and display results

Usage:
    python run_openx_benchmark.py [--model MODEL_NAME] [--max-episodes N]
    
    # With logging to file:
    python run_openx_benchmark.py --dataset bridge_dataset --max-episodes 50 --log-file bridge_benchmark.log
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from adapters.octo_adapter import OctoAdapter
from benchmarks.openx_benchmark import OpenXBenchmark


def main():
    parser = argparse.ArgumentParser(description='Run OCTO on OpenX benchmark')
    parser.add_argument(
        '--model',
        type=str,
        default='hf://rail-berkeley/octo-small-1.5',
        help='OCTO model checkpoint (default: octo-small-1.5)'
    )
    parser.add_argument(
        '--max-episodes',
        type=int,
        default=10,
        help='Maximum episodes to evaluate (default: 10)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='fractal20220817_data',
        help='Dataset name from OXE (default: fractal20220817_data - smaller dataset for testing)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Base path to OXE data (default: uses config file path or project data/open-x/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/openx',
        help='Output directory for results (default: results/openx)'
    )
    parser.add_argument(
        '--use-language',
        action='store_true',
        default=True,
        help='Use language instructions (default: True)'
    )
    parser.add_argument(
        '--use-debug',
        action='store_true',
        default=False,
        help='Use a small synthetic debug dataset (no downloads)'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        default=False,
        help='Do not attempt to download datasets from remote storage; fail fast if missing'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Save output to log file in logs/ folder (default: no logging to file)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_handlers = [logging.StreamHandler(sys.stdout)]
    
    if args.log_file:
        # Handle log file path - support both relative and absolute paths
        log_file = Path(args.log_file)
        
        if log_file.is_absolute():
            # Absolute path provided
            log_path = log_file
        elif str(log_file).startswith('logs/'):
            # Already includes logs/ prefix
            log_path = project_root / log_file
        else:
            # Just filename - add logs/ prefix
            logs_dir = project_root / 'logs'
            logs_dir.mkdir(exist_ok=True)
            log_path = logs_dir / log_file
        
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_path, mode='w'))
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=log_handlers
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("OCTO on Open X-Embodiment Benchmark")
    print("=" * 80)
    
    # Step 1: Initialize OCTO adapter
    print("\n[1/4] Initializing OCTO adapter...")
    adapter = OctoAdapter(model_path=args.model)
    
    try:
        adapter.load_model()
        print(f"✅ OCTO model loaded: {args.model}")
        print(f"   - Action dimension: {adapter.action_dim}")
        print(f"   - Action space: {adapter.action_space}")
    except Exception as e:
        print(f"❌ Failed to load OCTO model: {e}")
        return 1
    
    # Step 2: Initialize benchmark
    print("\n[2/4] Initializing OpenX benchmark...")
    
    benchmark_config = {
        'max_episodes_per_task': args.max_episodes,
        'action_mse_threshold': 0.1,
        'use_language': args.use_language,
        'use_synthetic_debug': args.use_debug,
        'no_download': args.no_download,
    }
    
    benchmark = OpenXBenchmark(
        data_path=args.data_path,
        dataset_name=args.dataset,
        config=benchmark_config
    )
    
    try:
        benchmark.setup()
        tasks = benchmark.get_task_list()
        print(f"✅ Benchmark initialized")
        print(f"   - Dataset: {args.dataset}")
        print(f"   - Episodes: {tasks[0]['num_episodes']}")
        print(f"   - Using language: {args.use_language}")
    except Exception as e:
        print(f"❌ Failed to setup benchmark: {e}")
        print("\nNote: If you see dataset not found errors, you may need to:")
        print("  1. Download the dataset manually")
        print("  2. Or use the debug dataset in models/octo/tests/debug_dataset/")
        return 1
    
    # Step 3: Run evaluation
    print("\n[3/4] Running evaluation...")
    print(f"Evaluating {args.max_episodes} episodes...")
    
    try:
        results = benchmark.run_evaluation(
            adapter=adapter,
            output_dir=args.output_dir,
            verbose=True
        )
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Display results
    print("\n[4/4] Results Summary")
    print("=" * 80)
    
    summary = results.get('summary', {})
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Success Rate: {summary.get('success_rate', 0):.2%}")
    print(f"   Total Episodes: {summary.get('total_episodes', 0)}")
    print(f"   Successful Episodes: {summary.get('successful_episodes', 0)}")
    print(f"   Failed Episodes: {summary.get('failed_episodes', 0)}")
    
    metrics = summary.get('metrics', {})
    print(f"\n📈 Action Prediction Metrics:")
    print(f"   MSE (Mean): {metrics.get('action_mse_mean', 0):.6f} ± {metrics.get('action_mse_std', 0):.6f}")
    print(f"   MAE (Mean): {metrics.get('action_mae_mean', 0):.6f} ± {metrics.get('action_mae_std', 0):.6f}")
    print(f"   Cosine Similarity: {metrics.get('cosine_similarity_mean', 0):.4f} ± {metrics.get('cosine_similarity_std', 0):.4f}")
    
    print(f"\n⏱️  Performance:")
    print(f"   Avg Time per Episode: {summary.get('avg_time_per_episode', 0):.2f}s")
    print(f"   Total Evaluation Time: {summary.get('total_time', 0):.2f}s")
    
    # Results saved location
    results_file = Path(summary.get('results_file', ''))
    if results_file.exists():
        print(f"\n💾 Full results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark evaluation completed!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
