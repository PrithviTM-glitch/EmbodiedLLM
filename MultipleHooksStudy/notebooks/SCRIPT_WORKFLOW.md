# Script-Based Workflow for Evo-1 Analysis

## Overview

The notebook now uses runner scripts that execute in the correct conda environment (`evo1_server`). This solves the problem where notebook Python cells run in Colab's default environment (missing Flash Attention, wrong transformers version).

## Architecture

```
Colab Notebook (orchestrator)
    ↓
    [calls via !conda run -n evo1_server]
    ↓
Runner Scripts (MultipleHooksStudy/scripts/)
    ├── load_evo1_model.py           # Model loading only
    └── run_evo1_gradient_analysis.py # Complete analysis pipeline
    ↓
    [executes in evo1_server conda environment]
    ↓
Environment with correct dependencies:
    - Flash Attention (compiled from source)
    - transformers==4.57.6 (NOT 5.0.0)
    - PyTorch 2.5.1+cu121
```

## Notebook Structure

### Section 1: Environment Setup
- Installs Miniconda
- Creates 3 conda environments (evo1_server, libero_client, metaworld_client)
- Compiles Flash Attention in evo1_server
- Clones EmbodiedLLM repository (makes scripts available)
- Downloads Evo-1 checkpoints (metaworld, libero)

### Section 2: Run Gradient Flow Analysis ⭐ NEW
**Main execution cell** - runs complete analysis in one command:
```python
!conda run -n evo1_server python scripts/run_evo1_gradient_analysis.py --checkpoint metaworld
```

### Section 3: Display Analysis Results
Shows the JSON results from Section 2 with formatted output

### Section 4: Manual Walkthrough (Optional)
Detailed step-by-step cells for educational purposes or custom experimentation

## Runner Scripts

### `run_evo1_gradient_analysis.py` (245 lines)

**Purpose**: Complete end-to-end gradient flow analysis

**Usage**:
```bash
conda run -n evo1_server python scripts/run_evo1_gradient_analysis.py --checkpoint {metaworld|libero} [--output PATH]
```

**Pipeline**:
1. Load Evo-1 model using official loading function
2. Import Evo1Hooks framework and discover model structure
3. Run baseline analysis (normal state encoder)
4. Run ablation analysis (zeroed integration_module output)
5. Compare gradients and generate verdict

**Output**: JSON file with comprehensive results
```json
{
  "model": "Evo-1 (0.77B)",
  "checkpoint": "metaworld",
  "state_encoder": "integration_module",
  "ablation_method": "output_ablation",
  "baseline_grad_norm": 0.123456,
  "ablated_grad_norm": 0.067890,
  "gradient_change_pct": 45.0,
  "verdict": "✅ WELL UTILIZED",
  "explanation": "Strong gradient response...",
  "loss_baseline": 0.5432,
  "loss_ablated": 0.6789,
  "device": "cuda",
  "model_structure": {...}
}
```

**Verdict Thresholds**:
- `< 10%` gradient change → ❌ UNDERUTILIZED
- `< 30%` gradient change → ⚠️ PARTIALLY UTILIZED
- `≥ 30%` gradient change → ✅ WELL UTILIZED

### `load_evo1_model.py` (189 lines)

**Purpose**: Load Evo-1 model only (without analysis)

**Usage**:
```bash
conda run -n evo1_server python scripts/load_evo1_model.py --checkpoint {metaworld|libero} [--output PATH]
```

**Output**: Pickle file with model metadata
```python
{
    'checkpoint': 'metaworld',
    'device': 'cuda',
    'num_parameters': 770000000,
    'checkpoint_dir': '/content/checkpoints/metaworld',
    'success': True
}
```

## Why This Approach?

### ❌ Problem: Direct notebook execution
```python
# This cell runs in Colab's default Python
import transformers  # ❌ Version 5.0.0 (wrong!)
import flash_attn     # ❌ Not installed!
```

### ✅ Solution: Script execution in conda environment
```python
# Notebook cell (orchestrator)
!conda run -n evo1_server python scripts/run_analysis.py
```
```python
# Script runs in evo1_server with:
# ✅ transformers==4.57.6
# ✅ Flash Attention compiled
# ✅ PyTorch 2.5.1+cu121
```

## Benefits

1. **Dependency Isolation**: Scripts run in correct environment with all dependencies
2. **Version Control**: Scripts are in git repo (branch: AnalyseMultipleHooks)
3. **Reusability**: Same scripts work across multiple notebooks
4. **Maintainability**: Update script once, all notebooks benefit
5. **Testability**: Scripts can be tested independently
6. **No Kernel Switching**: No need to restart notebook kernel or install kernel packages

## Quick Commands

### Run complete analysis (MetaWorld)
```bash
!conda run -n evo1_server python /content/EmbodiedLLM/MultipleHooksStudy/scripts/run_evo1_gradient_analysis.py --checkpoint metaworld
```

### Run analysis on LIBERO checkpoint
```bash
!conda run -n evo1_server python /content/EmbodiedLLM/MultipleHooksStudy/scripts/run_evo1_gradient_analysis.py --checkpoint libero
```

### Custom output location
```bash
!conda run -n evo1_server python /content/EmbodiedLLM/MultipleHooksStudy/scripts/run_evo1_gradient_analysis.py --checkpoint metaworld --output /content/my_results.json
```

### Load model only
```bash
!conda run -n evo1_server python /content/EmbodiedLLM/MultipleHooksStudy/scripts/load_evo1_model.py --checkpoint metaworld
```

## Verification

After running Section 1 (Environment Setup), verify conda environments:

```bash
# Check Flash Attention is installed
!conda run -n evo1_server python -c "import flash_attn; print(flash_attn.__version__)"

# Check transformers version (should be 4.57.6)
!conda run -n evo1_server python -c "import transformers; print(transformers.__version__)"

# Check PyTorch version
!conda run -n evo1_server python -c "import torch; print(torch.__version__)"

# List all conda environments
!conda env list
```

## Troubleshooting

### Script not found
**Problem**: `scripts/run_evo1_gradient_analysis.py: No such file or directory`

**Solution**: Run Section 1, Cell 8 to clone repository:
```python
!git clone https://github.com/PrithviTM-glitch/EmbodiedLLM.git
%cd EmbodiedLLM
!git checkout AnalyseMultipleHooks
```

### Flash Attention errors
**Problem**: `ModuleNotFoundError: No module named 'flash_attn'`

**Solution**: Re-run Section 1, Cell 7 and wait for full compilation (10-15 minutes)

### Wrong transformers version
**Problem**: Meta tensor initialization errors

**Solution**: Verify transformers version in evo1_server:
```bash
!conda run -n evo1_server python -c "import transformers; print(transformers.__version__)"
```
Should show `4.57.6`, NOT `5.0.0`

### conda command not found
**Problem**: `conda: command not found`

**Solution**: Re-run Section 1, Cell 7 to install Miniconda

## File Locations

| Path | Description |
|------|-------------|
| `/content/EmbodiedLLM/` | Cloned repository root |
| `/content/EmbodiedLLM/MultipleHooksStudy/scripts/` | Runner scripts location |
| `/content/checkpoints/metaworld/` | MetaWorld checkpoint |
| `/content/checkpoints/libero/` | LIBERO checkpoint |
| `/content/evo1_gradient_analysis_results.json` | Analysis results (default) |
| `/opt/conda/envs/evo1_server/` | evo1_server conda environment |

## Development Workflow

To modify analysis logic:

1. Edit script in local repository:
   ```bash
   vim MultipleHooksStudy/scripts/run_evo1_gradient_analysis.py
   ```

2. Commit and push to github:
   ```bash
   git add scripts/run_evo1_gradient_analysis.py
   git commit -m "Updated gradient analysis logic"
   git push origin AnalyseMultipleHooks
   ```

3. In Colab notebook, pull latest changes:
   ```python
   %cd /content/EmbodiedLLM
   !git pull origin AnalyseMultipleHooks
   ```

4. Re-run analysis cell (Section 2)

## Next Steps

1. Test complete workflow in Google Colab
2. Verify JSON results are generated correctly
3. Consider adding more analysis options (e.g., different ablation strategies)
4. Add logging to scripts for better debugging
5. Create pi0 and rdt versions of analysis scripts
