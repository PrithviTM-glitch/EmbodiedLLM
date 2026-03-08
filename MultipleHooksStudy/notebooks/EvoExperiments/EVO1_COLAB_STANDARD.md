# Evo-1 Colab Notebook Standard

> **Canonical Source**: `ablation_state_encoder_pass0.ipynb`
> **Last verified**: 8 March 2026

Every Evo-1 experiment notebook **must** follow the four-cell setup
sequence below **verbatim**. Only the `WORKSPACE` path and
experiment-specific directory creation differ between notebooks.

---

## Why this standard exists

Different notebooks evolved different installation patterns (some use
`subprocess.run`, some use `%%capture`, some forget conda TOS accept,
etc.). The `ablation_state_encoder_pass0.ipynb` is the only notebook
that has been **run end-to-end on Colab and produced results**.
This document codifies its exact setup so every future notebook works
first-try.

---

## Hardware Requirements

| Resource | Minimum |
|----------|---------|
| GPU | A100 40 GB (Colab Pro/Pro+) |
| Disk | ~25 GB free in `/content` |
| Runtime | Python 3 (default) |
| Drive | Mounted for persistent results |

---

## Cell 1: Setup Paths, Mount Drive, Check GPU

```python
from google.colab import drive
from pathlib import Path
import os

# Mount Drive
drive.mount('/content/drive')

# Path definitions — Drive (persistent)
WORKSPACE = '/content/drive/MyDrive/<experiment_name>'   # ← CHANGE THIS
RESULTS_DIR = Path(f'{WORKSPACE}/Results')

# Path definitions — Local (ephemeral)
CHECKPOINTS_DIR = Path('/content/checkpoints')
LOGS_DIR = Path('/content/logs')

# Create directories (add experiment-specific subdirs here)
for d in [RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Clean environment
os.environ.pop('PYTHONPATH', None)
os.environ['MUJOCO_GL'] = 'egl'
os.environ.pop('DISPLAY', None)

# Check GPU
import torch
if not torch.cuda.is_available():
    raise RuntimeError("❌ No GPU! Enable in Runtime > Change runtime type")

gpu_name = torch.cuda.get_device_name(0)
total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"✅ GPU: {gpu_name} ({total_mem:.1f} GB)")
print(f"✅ Workspace: {WORKSPACE}")
print(f"✅ Results: {RESULTS_DIR}")
```

### Rules
- Always call `os.environ.pop('PYTHONPATH', None)` to prevent stale paths.
- Always set `MUJOCO_GL = 'egl'` and remove `DISPLAY` for headless rendering.
- GPU check **must** raise `RuntimeError` so the notebook stops immediately.

---

## Cell 2: Install Dependencies (System + Miniconda + 3 Conda Envs)

> **All in ONE cell.** Uses `!` shell commands. No `%%capture`, no
> `subprocess.run()` wrappers.

```python
print('📦 Installing system dependencies...')
!apt-get update -qq
!apt-get install -y -qq git wget ffmpeg libosmesa6-dev patchelf libgl1-mesa-glx libegl1-mesa > /dev/null 2>&1

print('📦 Installing Miniconda...')
!wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
!bash /tmp/miniconda.sh -b -p /opt/conda > /dev/null 2>&1
!rm /tmp/miniconda.sh
os.environ['PATH'] = f"/opt/conda/bin:{os.environ['PATH']}"
!conda init bash
!conda config --set always_yes yes

!conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
!conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

print('\n📦 Creating 3 conda environments (official Evo-1 structure)...')
print('='*60)

# Environment 1: Evo-1 server (Python 3.10)
print('\n[1/3] evo1_server (Python 3.10) - for Evo-1 model')
!conda create -n evo1_server python=3.10 -y
!conda run -n evo1_server pip install  \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# CRITICAL: transformers must be 4.57.6, NOT 5.x
# 28 Jan 2026: transformers==5.0.0 causes meta tensor / InternViT init failure
!conda run -n evo1_server pip install \
  'numpy>=1.26.4,<2.0' 'transformers==4.57.6' accelerate diffusers \
  einops timm pillow opencv-python-headless \
  websockets pyyaml huggingface_hub tqdm
print('📦 Installing flash-attn (required, ~10-15 min)...')
!conda run -n evo1_server pip install flash-attn --no-build-isolation 2>&1 | tail -5 || echo '⚠️ Flash-attn failed'
print('✅ evo1_server ready')

# Environment 2: LIBERO client (Python 3.8.13 — OFFICIAL requirement)
print('\n[2/3] libero_client (Python 3.8.13) - for LIBERO benchmark')
!conda create -n libero_client python=3.8.13 -y
!conda run -n libero_client pip install \
  'numpy>=1.20,<2.0' robosuite==1.4.1 mujoco==2.3.7 \
  imageio h5py bddl websockets huggingface_hub
!conda run -n libero_client pip install \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
  --extra-index-url https://download.pytorch.org/whl/cu113
print('✅ libero_client ready')

# Environment 3: MetaWorld client (Python 3.10)
print('\n[3/3] metaworld_client (Python 3.10) - for MetaWorld benchmark')
!conda create -n metaworld_client python=3.10 -y
!conda run -n metaworld_client pip install \
  mujoco websockets opencv-python packaging huggingface_hub metaworld gymnasium

print('✅ metaworld_client ready')

print('\n' + '='*60)
print('✅ All 3 environments created!')
!conda run -n evo1_server python -c "import sys; print(f'  evo1_server: Python {sys.version.split()[0]}')"
!conda run -n libero_client python -c "import sys; print(f'  libero_client: Python {sys.version.split()[0]}')"
!conda run -n metaworld_client python -c "import sys; print(f'  metaworld_client: Python {sys.version.split()[0]}')"
```

### Rules
- **One cell, no `%%capture`**. You need to see output to debug failures.
- Use `!` shell prefix, **not** `subprocess.run()`.
- Always run `conda tos accept` — Colab prompts block non-interactively.
- Use `torch==2.5.1+cu121` (with CUDA suffix), not bare `torch==2.5.1`.
- Pin `transformers==4.57.6`. Version 5.x breaks InternViT initialization.
- Pin `numpy>=1.26.4,<2.0` for evo1_server to avoid numpy 2.0 ABI breaks.
- Include `flash-attn` with `--no-build-isolation` and `| tail -5` to limit output.

---

## Cell 3: Clone Repositories, Install in Envs, Configure LIBERO

```python
import os
import yaml
from pathlib import Path

print('📦 Setting up repositories...')
print('='*60)

# ==================== Clone Evo-1 ====================
print('\n[1/3] Cloning Evo-1...')
if not Path('/content/Evo-1/.git').exists():
    !git clone https://github.com/MINT-SJTU/Evo-1.git /content/Evo-1
    print('✅ Cloned Evo-1')
else:
    print('✅ Evo-1 already cloned')

# Install Evo-1 dependencies in server environment
print('\n📦 Installing Evo-1 dependencies in evo1_server...')
evo1_requirements = Path('/content/Evo-1/Evo_1/requirements.txt')
if evo1_requirements.exists():
    !conda run -n evo1_server pip install -q -r /content/Evo-1/Evo_1/requirements.txt
    print('✅ Evo-1 dependencies installed in evo1_server')
else:
    print('⚠️ WARNING: Evo-1 requirements.txt not found')

# ==================== Clone LIBERO ====================
print('\n[2/3] Cloning LIBERO...')
if not Path('/content/LIBERO_evaluation/LIBERO/.git').exists():
    !git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /content/LIBERO_evaluation/LIBERO
    print('✅ Cloned LIBERO')
else:
    print('✅ LIBERO already cloned')

# Install LIBERO in client environment (Python 3.8.13)
print('\n📦 Installing LIBERO in libero_client...')
libero_requirements = Path('/content/LIBERO_evaluation/LIBERO/requirements.txt')
if libero_requirements.exists():
    !conda run -n libero_client pip install -q -r /content/LIBERO_evaluation/LIBERO/requirements.txt
    !conda run -n libero_client pip install -q -e /content/LIBERO_evaluation/LIBERO
    print('✅ LIBERO installed in libero_client')
else:
    print('⚠️ WARNING: LIBERO requirements.txt not found')

# ==================== Configure LIBERO ====================
print('\n[3/3] Configuring LIBERO...')
os.makedirs(os.path.expanduser('~/.libero'), exist_ok=True)
libero_cfg = os.path.expanduser('~/.libero/config.yaml')

if not os.path.exists(libero_cfg):
    cfg = {
        'benchmark_root': '/content/LIBERO_evaluation/LIBERO/libero/libero',
        'bddl_files': '/content/LIBERO_evaluation/LIBERO/libero/libero/bddl_files',
        'init_states': '/content/LIBERO_evaluation/LIBERO/libero/libero/init_files',
        'datasets': '/content/LIBERO_evaluation/LIBERO/datasets',
        'assets': '/content/LIBERO_evaluation/LIBERO/libero/libero/assets'
    }
    with open(libero_cfg, 'w') as f:
        yaml.dump(cfg, f)
    print('✅ LIBERO config created at ~/.libero/config.yaml')
else:
    print('✅ LIBERO config already exists')

# ==================== Install MetaWorld ====================
print('\n📦 Installing MetaWorld in metaworld_client...')
!conda run -n metaworld_client pip install -q metaworld
!conda run -n metaworld_client pip install -q opencv-python
print('✅ MetaWorld and dependencies installed')

# ==================== Verification ====================
print('\n' + '='*60)
print('🔍 Verifying installations...')
print('='*60)

verification_commands = [
    ('evo1_server', 'python -c "import torch; print(f\'PyTorch: {torch.__version__}\')"'),
    ('libero_client', 'python -c "import libero; print(\'LIBERO imported successfully\')"'),
    ('metaworld_client', 'python -c "import metaworld; print(\'MetaWorld imported successfully\')"'),
]

for env_name, cmd in verification_commands:
    print(f'\n{env_name}:')
    !conda run -n {env_name} {cmd}

print('\n' + '='*60)
print('✅ All repositories installed and configured!')
print('='*60)
```

### Rules
- Clone Evo-1 to `/content/Evo-1`.
- Clone LIBERO to `/content/LIBERO_evaluation/LIBERO` (NOT `/content/LIBERO`).
- Install Evo-1 deps from its `requirements.txt` — do NOT `pip install -e /content/Evo-1`.
- Install LIBERO from its `requirements.txt` AND as editable (`pip install -e`).
- Always create `~/.libero/config.yaml` with the paths above.
- Verify imports at the end of this cell.

---

## Cell 4: Download Checkpoints

```python
from huggingface_hub import snapshot_download

CHECKPOINTS_DIR = Path('/content/checkpoints/')

CHECKPOINTS = {
    'libero': {'repo': 'MINT-SJTU/Evo1_LIBERO', 'dir': CHECKPOINTS_DIR / 'libero'},
    'metaworld': {'repo': 'MINT-SJTU/Evo1_MetaWorld', 'dir': CHECKPOINTS_DIR / 'metaworld'}
}

print('📥 Downloading checkpoints...')
for name, cfg in CHECKPOINTS.items():
    cfg['dir'].mkdir(parents=True, exist_ok=True)
    model_file = cfg['dir'] / 'mp_rank_00_model_states.pt'

    if model_file.exists() and model_file.stat().st_size > 1_000_000:
        print(f"✅ {name}: {model_file.stat().st_size / 1e9:.2f} GB")
    else:
        print(f"⏳ Downloading {name}...")
        snapshot_download(
            repo_id=cfg['repo'],
            local_dir=str(cfg['dir']),
            local_dir_use_symlinks=False,
        )
        print(f"✅ {name} downloaded")

print('\n✅ Checkpoints ready')
```

### Rules
- Check for existing files by size (`> 1 MB`) to allow re-runs.
- Use `local_dir_use_symlinks=False` — Colab ephemeral disk doesn't benefit from symlinks.

---

## LIBERO Eval Client Reference

When running LIBERO evaluation clients, use:

| Item | Value |
|------|-------|
| Eval script | `/content/Evo-1/LIBERO_evaluation/libero_client_4tasks.py` |
| PYTHONPATH | `/content/LIBERO_evaluation/LIBERO` |
| Conda env | `libero_client` |

Example shell command in experiment cells:
```python
cmd = (
    'export PYTHONPATH=/content/LIBERO_evaluation/LIBERO && '
    'conda run -n libero_client python '
    '/content/Evo-1/LIBERO_evaluation/libero_client_4tasks.py '
    '--server-address localhost '
    f'--server-port {port} '
    '--benchmark libero_90 --num-episodes 50 '
    f'--output {output_path} '
    f'> {log_path} 2>&1'
)
```

Or via Python `subprocess.Popen` env dict:
```python
client_env = {**os.environ, "PYTHONPATH": "/content/LIBERO_evaluation/LIBERO"}
```

---

## Forbidden Patterns

| ❌ Don't | ✅ Do instead |
|----------|---------------|
| `%%capture install_log` | Print output so failures are visible |
| `subprocess.run()` for installs | `!` shell commands |
| `torch==2.5.1` (bare) | `torch==2.5.1+cu121` |
| `transformers>=5` or latest | `transformers==4.57.6` |
| Clone LIBERO to `/content/LIBERO` | Clone to `/content/LIBERO_evaluation/LIBERO` |
| `pip install -e /content/Evo-1` | `pip install -r /content/Evo-1/Evo_1/requirements.txt` |
| Skip conda TOS accept | Always include the two `conda tos accept` lines |
| Skip `~/.libero/config.yaml` | Always create it with standard paths |

---

## Conda Environment Summary

### evo1_server (Python 3.10)
```
torch==2.5.1+cu121
torchvision==0.20.1+cu121
numpy>=1.26.4,<2.0
transformers==4.57.6
accelerate
diffusers
einops
timm
pillow
opencv-python-headless
websockets
pyyaml
huggingface_hub
tqdm
flash-attn  (compiled from source, ~10-15 min)
```

### libero_client (Python 3.8.13)
```
numpy>=1.20,<2.0
robosuite==1.4.1
mujoco==2.3.7
imageio
h5py
bddl
websockets
huggingface_hub
torch==1.11.0+cu113
torchvision==0.12.0+cu113
torchaudio==0.11.0
```

### metaworld_client (Python 3.10)
```
mujoco
websockets
opencv-python
packaging
huggingface_hub
metaworld
gymnasium
```

---

## File Layout on Colab

```
/content/
├── Evo-1/                          # MINT-SJTU/Evo-1 repo
│   ├── Evo_1/
│   │   ├── requirements.txt
│   │   └── scripts/
│   ├── LIBERO_evaluation/
│   │   └── libero_client_4tasks.py  # LIBERO eval client
│   └── MetaWorld_evaluation/
│       └── mt50_evo1_client_prompt.py
├── LIBERO_evaluation/
│   └── LIBERO/                     # Lifelong-Robot-Learning/LIBERO repo
│       └── libero/
├── checkpoints/
│   ├── libero/
│   │   ├── config.json
│   │   ├── norm_stats.json
│   │   └── mp_rank_00_model_states.pt
│   └── metaworld/
│       ├── config.json
│       ├── norm_stats.json
│       └── mp_rank_00_model_states.pt
├── logs/
└── drive/                          # Google Drive mount
    └── MyDrive/
        └── <experiment>/Results/
```
