# Server Script Fix - Complete

## Problem Solved
**Original Issue**: `FileNotFoundError: 'Your/Path/To/Checkpoint/config.json'`

The Evo-1 complete notebook was calling the original `Evo1_server.py` from the Evo-1 repo, which has hardcoded placeholder checkpoint paths and doesn't properly handle the `--checkpoint` argument.

## Solution Implemented
Following the pattern from `ablation_state_encoder_pass0.ipynb`, created custom server scripts with proper checkpoint loading:

### 1. Baseline Server Script (✅ Complete)
**File**: `/content/Evo-1/Evo_1/scripts/Evo1_baseline_server.py`

**Features**:
- Complete Normalizer class with proper state normalization
- `normalize_state()`: Returns `(state - min)/(max - min) * 2 - 1` (NORMAL normalization)
- `load_model_and_normalizer()`: Loads from `args.checkpoint` directory
- Proper argument parsing: `--port`, `--checkpoint`, `--name`
- WebSocket server with ping_interval=120, ping_timeout=300

**Updated Launch Commands**:
- ✅ LIBERO Baseline (Cell 4, ~line 225): Uses `Evo1_baseline_server.py`
- ✅ MetaWorld Baseline (Cell 5, ~line 493): Uses `Evo1_baseline_server.py`

### 2. Ablated Server Script (✅ Complete)
**File**: `/content/Evo-1/Evo_1/scripts/Evo1_ablated_server.py`

**Features**:
- Identical to baseline except for one function
- `normalize_state()`: Returns `torch.zeros_like(state)` (ABLATION)
- Tests: Can vision alone solve tasks without proprioceptive state?
- All other code identical to baseline: same inference flow, same outputs

**Status**:
- ✅ Script creation cell added (Cell 6, ~line 555)
- ℹ️ Actual ablation benchmark runs not implemented yet (notebook says "See original notebook")
- ℹ️ When implementing ablation benchmarks, use ports 9021-9030 (LIBERO) and 9031-9040 (MetaWorld)

## Port Allocation Summary
All ports now sequential starting from 9001 (updated in previous fix):

### Evo-1 Port Ranges:
- **LIBERO Baseline**: 9001-9010 (10 parallel servers)
- **MetaWorld Baseline**: 9011-9020 (10 parallel servers)
- **LIBERO Ablation**: 9021-9030 (10 parallel servers) - when implemented
- **MetaWorld Ablation**: 9031-9040 (10 parallel servers) - when implemented

## Key Differences from Original Evo1_server.py

### Original Script Issues:
```python
# Hardcoded placeholder path
ckpt_dir = "Your/Path/To/Checkpoint"
config = json.load(open(os.path.join(ckpt_dir, "config.json")))  # ❌ FAILS
```

### Fixed Scripts:
```python
# Proper command-line argument
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

# Load from provided checkpoint directory
def load_model_and_normalizer(ckpt_dir):
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))  # ✅ WORKS
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))
    # ... load model ...
```

## Usage Examples

### Baseline Server:
```bash
conda run -n evo1_server python /content/Evo-1/Evo_1/scripts/Evo1_baseline_server.py \
  --port 9001 \
  --checkpoint /content/checkpoints/evo1/libero \
  --name baseline_trial_1
```

### Ablated Server (when implemented):
```bash
conda run -n evo1_server python /content/Evo-1/Evo_1/scripts/Evo1_ablated_server.py \
  --port 9021 \
  --checkpoint /content/checkpoints/evo1/libero \
  --name ablation_trial_1
```

## Verification
✅ Baseline server creation cell added before Cell 4
✅ Ablated server creation cell added in Cell 6
✅ LIBERO baseline launch command updated (Cell 4)
✅ MetaWorld baseline launch command updated (Cell 5)
✅ Both scripts accept proper checkpoint paths via --checkpoint argument
✅ Both scripts include proper Normalizer class
✅ Only difference: normalize_state() returns normalized values (baseline) vs zeros (ablation)

## Next Steps (When Implementing Ablation Benchmarks)
1. Add ablation server startup code (similar to baseline, use ports 9021-9030 for LIBERO)
2. Add ablation client execution code
3. Update ablation server launch commands to use `Evo1_ablated_server.py`
4. Use ports 9031-9040 for MetaWorld ablation servers

## References
- **Working Pattern**: `ablation_state_encoder_pass0.ipynb` lines 285-400
- **Updated Notebook**: `evo1_complete.ipynb`
- **Port Documentation**: `PORT_UPDATE.md`
- **Validation**: `VALIDATION_REPORT.md`
