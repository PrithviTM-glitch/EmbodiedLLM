#!/usr/bin/env python3
"""
List available Open X-Embodiment datasets.

This script shows all datasets that can be used with the OpenX benchmark.
"""

import sys
from pathlib import Path

# Add OCTO to path
octo_path = Path(__file__).parent.parent.parent / "models" / "octo"
if str(octo_path) not in sys.path:
    sys.path.insert(0, str(octo_path))

try:
    from octo.data.oxe import OXE_DATASET_CONFIGS
    
    print("=" * 80)
    print("Available Open X-Embodiment Datasets")
    print("=" * 80)
    print()
    print(f"Total datasets: {len(OXE_DATASET_CONFIGS)}")
    print()
    
    # Group datasets by type
    print("Datasets (sorted alphabetically):")
    print("-" * 80)
    
    for i, (name, config) in enumerate(sorted(OXE_DATASET_CONFIGS.items()), 1):
        print(f"{i:2d}. {name}")
        
        # Show image sources
        img_keys = config.get('image_obs_keys', {})
        images = [k for k, v in img_keys.items() if v is not None]
        if images:
            print(f"    Images: {', '.join(images)}")
        
        # Show encoding types
        proprio = config.get('proprio_encoding')
        action = config.get('action_encoding')
        print(f"    Proprio: {proprio.name if hasattr(proprio, 'name') else proprio}")
        print(f"    Action: {action.name if hasattr(action, 'name') else action}")
        print()
    
    print("=" * 80)
    print("Recommended datasets for testing:")
    print("-" * 80)
    print("1. fractal20220817_data  - Smaller dataset, good for quick tests")
    print("2. kuka                  - Standard manipulation tasks")
    print("3. bridge_dataset        - Diverse manipulation (needs separate download)")
    print()
    print("Data source:")
    print("  Default: gs://gresearch/robotics (Google Cloud Storage)")
    print("  Note: Cloud access requires Google Cloud authentication")
    print()
    print("Usage:")
    print("  python run_openx_benchmark.py --dataset fractal20220817_data")
    print("=" * 80)
    
except ImportError as e:
    print("❌ Error: OCTO not installed properly")
    print(f"   {e}")
    print()
    print("Please install OCTO:")
    print("  cd models/octo")
    print("  pip install -e .")
    sys.exit(1)
