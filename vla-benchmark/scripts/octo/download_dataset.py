#!/usr/bin/env python3
"""
Download Open X-Embodiment datasets from Google Cloud Storage.

Simple script to download OXE datasets to the project data folder.

Usage:
    python download_dataset.py --dataset fractal20220817_data --shards 10
    python download_dataset.py --dataset bridge_data_v2 --shards 20
    python download_dataset.py --list
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Dataset version mapping
# Maps OCTO dataset names to their GCS version directories
DATASET_VERSIONS = {
    'bridge_dataset': '0.0.1',  # GCS path: bridge_data_v2/0.0.1
    'fractal20220817_data': '0.1.0',
    'kuka': '0.1.0',
    'roboturk': '0.1.0',
    'berkeley_autolab_ur5': '0.1.0',
}

# Maps OCTO dataset names to GCS storage names (when different)
GCS_DATASET_NAMES = {
    'bridge_dataset': 'bridge_data_v2',  # OCTO calls it bridge_dataset, GCS has bridge_data_v2
}

POPULAR_DATASETS = [
    'fractal20220817_data',
    'bridge_dataset',  # Note: stored as bridge_data_v2 in GCS
    'kuka',
    'roboturk',
    'berkeley_autolab_ur5',
    'taco_play',
    'jaco_play',
]

def run_gsutil(args):
    """Run gsutil command and return success status."""
    try:
        result = subprocess.run(
            ['gsutil'] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def list_available_datasets():
    """List available OXE datasets in GCS."""
    print("\n" + "="*80)
    print("Available Open X-Embodiment Datasets")
    print("="*80 + "\n")
    
    success, output = run_gsutil(['ls', 'gs://gresearch/robotics/'])
    if not success:
        print(f"❌ Failed to list datasets: {output}")
        return
    
    datasets = []
    for line in output.strip().split('\n'):
        if line.endswith('/'):
            name = line.split('/')[-2]
            if not name.endswith('_$folder$'):
                datasets.append(name)
    
    print("Popular datasets (from OCTO training mix):")
    for ds in POPULAR_DATASETS:
        if ds in datasets:
            version = DATASET_VERSIONS.get(ds, '0.1.0')
            print(f"  ✓ {ds} (v{version})")
    
    print(f"\n{len(datasets)} total datasets available in gs://gresearch/robotics/")
    print("\nUse --dataset <name> to download a specific dataset.")

def download_dataset(dataset_name, num_shards=10):
    """Download dataset from GCS to project data folder."""
    
    # Get version for this dataset
    version = DATASET_VERSIONS.get(dataset_name, '0.1.0')
    
    # Get GCS storage name (may differ from OCTO dataset name)
    gcs_name = GCS_DATASET_NAMES.get(dataset_name, dataset_name)
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent.parent / "data" / "open-x" / dataset_name / version
    data_dir.mkdir(parents=True, exist_ok=True)
    
    gcs_base = f"gs://gresearch/robotics/{gcs_name}/{version}"
    
    print("\n" + "="*80)
    print(f"Downloading OXE Dataset: {dataset_name}")
    print("="*80)
    print(f"\n📊 Dataset: {dataset_name} (v{version})")
    print(f"📊 Shards: {num_shards}")
    print(f"📁 Output: {data_dir}")
    print(f"🌐 Source: {gcs_base}\n")
    
    # Download metadata files
    print("[1/2] Downloading metadata files...")
    for filename in ['dataset_info.json', 'features.json']:
        gcs_path = f"{gcs_base}/{filename}"
        local_path = data_dir / filename
        
        if local_path.exists():
            print(f"  ⏭  {filename} (already exists)")
            continue
        
        print(f"  ⬇  {filename}...")
        success, _ = run_gsutil(['cp', gcs_path, str(local_path)])
        
        if not success:
            print(f"  ❌ Failed to download {filename}")
            return False
        print(f"  ✅ {filename}")
    
    # Download TFRecord shards
    print(f"\n[2/2] Downloading {num_shards} TFRecord shards...")
    
    for i in range(num_shards):
        shard_num = f"{i:05d}"
        shard_name = f"{gcs_name}-train.tfrecord-{shard_num}-of-01024"
        local_path = data_dir / shard_name
        
        if local_path.exists():
            print(f"  ⏭  Shard {i+1}/{num_shards} (already exists)")
            continue
        
        gcs_path = f"{gcs_base}/{shard_name}"
        
        print(f"  ⬇  Shard {i+1}/{num_shards}...", end='', flush=True)
        success, _ = run_gsutil(['cp', gcs_path, str(local_path)])
        
        if not success:
            print(f" ❌")
            print(f"\nFailed to download shard {shard_num}")
            return False
        
        # Get file size
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f" ✅ ({size_mb:.1f} MB)")
    
    # Summary
    total_size = sum(f.stat().st_size for f in data_dir.glob("*.tfrecord-*")) / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"✅ Download complete!")
    print(f"   Total size: {total_size:.1f} MB")
    print(f"   Location: {data_dir}")
    print(f"   Files: {len(list(data_dir.glob('*.tfrecord-*')))} shards + 2 metadata files")
    print(f"{'='*80}\n")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Download Open X-Embodiment datasets from Google Cloud Storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --dataset fractal20220817_data --shards 10
  %(prog)s --dataset bridge_data_v2 --shards 20
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Name of the dataset to download'
    )
    parser.add_argument(
        '--shards',
        type=int,
        default=10,
        help='Number of TFRecord shards to download (default: 10, max: 1024)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
        return 0
    
    if not args.dataset:
        parser.print_help()
        print("\n❌ Error: --dataset is required (or use --list to see available datasets)")
        return 1
    
    # Validate shard count
    if args.shards < 1 or args.shards > 1024:
        print(f"❌ Error: --shards must be between 1 and 1024 (got {args.shards})")
        return 1
    
    # Download the dataset
    success = download_dataset(args.dataset, args.shards)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
