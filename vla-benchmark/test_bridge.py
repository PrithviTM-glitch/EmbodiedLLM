#!/usr/bin/env python3
"""Quick test script for bridge dataset loading."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.openx_benchmark import OpenXBenchmark

print("Creating benchmark...")
benchmark = OpenXBenchmark(
    dataset_name='bridge_dataset',
    data_path='data/open-x',
    config={'max_episodes_per_task': 2}
)

print("Running setup...")
benchmark.setup()
print("Setup complete!")

if benchmark.episodes:
    print(f"\nLoaded {len(benchmark.episodes)} episodes")
    print(f"Episode 0 keys: {benchmark.episodes[0].keys()}")
    print(f"Episode 0 observation keys: {benchmark.episodes[0]['observation'].keys()}")
    print(f"Episode 0 action shape: {benchmark.episodes[0]['action'].shape}")
    print("\nObservation shapes:")
    for key, val in benchmark.episodes[0]['observation'].items():
        print(f"  {key}: {val.shape}")
    
    print("\nTest completed successfully!")
