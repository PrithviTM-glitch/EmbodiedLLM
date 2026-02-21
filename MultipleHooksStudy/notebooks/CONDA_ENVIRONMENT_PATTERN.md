# Conda Environment Pattern for Model Notebooks

## Why Conda Environments?

Each VLA model (Evo-1, RDT-1B, π0) has different dependency requirements, especially for transformers versions:
- **Evo-1**: Requires transformers 4.46.3 (to avoid meta tensor issues with InternVL3)
- **RDT-1B**: May require different transformers version
- **π0**: Uses JAX/PyTorch with specific dependencies

Using separate conda environments prevents version conflicts.

## Standard Pattern

### Cell 1: Environment Info (Markdown)
```markdown
## 1.5 Setup Isolated Environment

**Critical**: [MODEL_NAME] requires [KEY_DEPENDENCY with version].  
**Solution**: Use conda environment `[env_name]` to isolate dependencies.
```

### Cell 2: Create Conda Environment
```python
# Create isolated conda environment for [MODEL_NAME]
print('🔧 Setting up [MODEL_NAME] environment...')
print('='*60)

# Step 1: Create conda environment
print('\n[1/N] Creating conda environment: [env_name]')
!conda create -n [env_name] python=[version] -y -q
print('✅ Environment created')

# Step 2: Install base dependencies
print('\n[2/N] Installing base dependencies...')
!conda run -n [env_name] pip install -q torch torchvision [--index-url ...]
!conda run -n [env_name] pip install -q [other packages]
print('✅ Base dependencies installed')

# Step 3: Install CRITICAL locked versions
print('\n[3/N] Installing [critical_package] [version]...')
!conda run -n [env_name] pip install -q "[critical_package]==[version]"
print('✅ [critical_package] locked in place')

# Continue with model-specific dependencies...
```

### Cell 3: Clone and Install Repository
```python
# Clone and install [MODEL_NAME] repository
import os
from pathlib import Path

print('📦 Setting up [MODEL_NAME] repository...')
print('='*60)

# Determine paths (Colab vs local)
if IN_COLAB:
    repo_path = '/content/[REPO_NAME]'
    checkpoint_base = '/content'
else:
    repo_path = './[REPO_NAME]'
    checkpoint_base = '.'

# Step 1: Clone repository
if not Path(f'{repo_path}/.git').exists():
    print('\n📥 Cloning [MODEL_NAME] repository...')
    !git clone [REPO_URL] {repo_path}
    print('✅ [MODEL_NAME] cloned')
else:
    print('✅ [MODEL_NAME] already cloned')

# Step 2: Apply any necessary patches
# (e.g., for Evo-1, patch InternVL3Embedder)

# Step 3: Install repository dependencies in conda environment
requirements_path = f'{repo_path}/[path]/requirements.txt'
if os.path.exists(requirements_path):
    print('\n📦 Installing [MODEL_NAME] dependencies in [env_name]...')
    !conda run -n [env_name] pip install -q -r {requirements_path}
    print('✅ Dependencies installed')
    
    # CRITICAL: Re-lock critical dependency after requirements.txt
    print('\n⚠️ Re-locking [critical_package] to [version]...')
    !conda run -n [env_name] pip install -q --force-reinstall "[critical_package]==[version]"
    print('✅ [critical_package] re-locked')

# Step 4: Verification
print('\n🔍 Verifying installation...')
!conda run -n [env_name] python -c "import [package]; print(f'[Package]: {[package].__version__}')"
```

### Cell 4: Load Model
```python
# Load [MODEL_NAME] model (using conda environment)
import sys
sys.path.insert(0, f'{repo_path}/[code_path]')

from [module] import [ModelClass]
# ... rest of loading code
```

## Evo-1 Example (Complete)

See `evo1_complete.ipynb` cells 6-9 for the full implementation:
- Cell 6: Markdown explanation
- Cell 7: Create `evo1_env` conda environment
- Cell 8: Clone Evo-1, patch InternVL3Embedder, install dependencies
- Cell 9: Load model

## RDT-1B Pattern (TODO)

Environment: `rdt_env`
- Determine required transformers version from thu-ml/RoboticsDiffusionTransformer docs
- Clone thu-ml/RoboticsDiffusionTransformer
- Install dependencies including SigLIP and T5-XXL encoders
- Use `scripts.agilex_model.create_model` for loading

## π0 Pattern (TODO)

Environment: `pi0_env`
- Clone Physical-Intelligence/openpi
- Install with `uv` or `pip install -e`
- Download checkpoints from GCS
- Use `openpi.policies.policy_config.create_trained_policy` for loading

## Key Principles

1. **One environment per model** - Prevents dependency conflicts
2. **Lock critical versions** - Re-install after requirements.txt with `--force-reinstall`
3. **Verify installations** - Always verify key packages after setup
4. **Patch when needed** - Apply code patches before importing modules
5. **Use conda run** - Execute all commands in the target environment with `conda run -n [env_name]`

## Benefits

- ✅ No version conflicts between models
- ✅ Clean, reproducible environments
- ✅ Easy to reset (just delete and recreate conda env)
- ✅ Works on both Colab and local
- ✅ Clear separation of concerns
