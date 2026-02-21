#!/usr/bin/env python3
"""
Load Evo-1 model using official loading function.
This script is designed to run in the evo1_server conda environment.

Usage:
    conda run -n evo1_server python load_evo1_model.py --checkpoint metaworld

Requirements:
    - Evo-1 repository cloned at /content/Evo-1
    - Checkpoint downloaded to /content/checkpoints/{checkpoint_name}
    - Running in evo1_server conda environment
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import pickle

def main():
    parser = argparse.ArgumentParser(description='Load Evo-1 model')
    parser.add_argument('--checkpoint', type=str, default='metaworld',
                       choices=['metaworld', 'libero'],
                       help='Which checkpoint to load (metaworld or libero)')
    parser.add_argument('--output', type=str, default='/content/model_state.pkl',
                       help='Where to save model state')
    args = parser.parse_args()
    
    print('🏗️ Loading Evo-1 model...')
    print('='*60)
    
    # Paths
    evo1_repo_path = '/content/Evo-1'
    checkpoint_dir = Path(f'/content/checkpoints/{args.checkpoint}')
    evo1_code_path = f'{evo1_repo_path}/Evo_1'
    
    # Verify paths exist
    if not os.path.exists(evo1_repo_path):
        raise FileNotFoundError(f"Evo-1 repository not found at: {evo1_repo_path}")
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_dir}")
    
    # Add Evo-1 to path
    if evo1_code_path not in sys.path:
        sys.path.insert(0, evo1_code_path)
    
    # Import official loading function
    print('[1/2] Importing official Evo-1 server module...')
    sys.path.insert(0, f'{evo1_repo_path}/Evo_1/scripts')
    from Evo1_server import load_model_and_normalizer
    print('✅ Official server module imported')
    
    # Load model
    print(f'\n[2/2] Loading model from {args.checkpoint}...')
    print('⏳ This takes ~1-2 minutes (downloading & loading InternVL3-1B)...')
    print('   Note: First run downloads InternVL3-1B (~2GB)')
    print('   Security warning about trust_remote_code is expected')
    print('   Using evo1_server conda environment with transformers==4.57.6')
    
    try:
        model, normalizer = load_model_and_normalizer(str(checkpoint_dir))
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print('\n' + '='*60)
        print('✅ Evo-1 model loaded successfully!')
        print(f'   Checkpoint: {args.checkpoint}')
        print(f'   Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B')
        print(f'   Device: {device}')
        print(f'   Environment: evo1_server (transformers==4.57.6)')
        print('='*60)
        
        # Save model reference for notebook to use
        # We'll save metadata about the loaded model
        model_info = {
            'checkpoint': args.checkpoint,
            'device': str(device),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'checkpoint_dir': str(checkpoint_dir),
            'success': True
        }
        
        with open(args.output, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f'\n✅ Model info saved to: {args.output}')
        print('   (Model object stays in memory for notebook to use)')
        
        return 0
        
    except Exception as e:
        print('\n' + '='*60)
        print('❌ Model loading failed!')
        print('='*60)
        print(f'Error: {str(e)}')
        print('\nTroubleshooting:')
        print('1. CRITICAL: Verify Flash Attention is installed in evo1_server')
        print('   - Run: conda run -n evo1_server python -c "import flash_attn"')
        print('2. Verify transformers==4.57.6 (NOT 5.0.0) in evo1_server')
        print('   - Run: conda run -n evo1_server python -c "import transformers; print(transformers.__version__)"')
        print('3. Verify checkpoint files exist in:', checkpoint_dir)
        print('4. Check CUDA/GPU availability')
        print('5. If "meta tensor" error: Wrong transformers version or missing Flash Attention')
        
        # Save error state
        model_info = {
            'checkpoint': args.checkpoint,
            'success': False,
            'error': str(e)
        }
        with open(args.output, 'wb') as f:
            pickle.dump(model_info, f)
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
