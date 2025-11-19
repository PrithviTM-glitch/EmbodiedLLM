# vla-benchmark

A reproducible evaluation pipeline for Vision-Language-Action (VLA) models on embodied manipulation and interaction benchmarks.

---

## 🎯 Project Goals

- **Reproduce and evaluate** OCTO model (27M & 93M variants) and extend to other VLA models
- **Create lightweight pipeline** that runs on reasonable hardware (<1B parameter models)
- **Provide data adapters** to standardize datasets into model-consumable formats
- **Generate comparable metrics** (success rate, task completion, action accuracy)
- **Keep artifacts clean** - large models/data stay out of source control

---

## 🗂️ Project Structure

```
vla-benchmark/
│
├── 📄 README.md                    # Project overview and guide
├── 📄 requirements.txt             # Python dependencies
├── 🔧 setup.sh                     # Environment setup script
├── 📄 .gitignore                   # Excludes models/, data/, checkpoints/
│
├── 🔌 adapters/                    # Data format converters
│   ├── __init__.py
│   ├── base_adapter.py             # Abstract base class
│   ├── octo_adapter.py             # OCTO-specific adapter
│   ├── minivla_adapter.py          # MiniVLA adapter
│   ├── smolvla_adapter.py          # SmolVLA adapter
│   └── tinyvla_adapter.py          # TinyVLA adapter
│   │
│   └── 📝 Purpose: Convert dataset observations/actions
│                   to model-specific input/output formats
│
├── 🧪 benchmarks/                  # Evaluation runners
│   ├── __init__.py
│   ├── base_benchmark.py           # Abstract benchmark interface
│   ├── libero_runner.py            # LIBERO-90 evaluation
│   ├── embodiedbench_runner.py     # EmbodiedBench evaluation
│   ├── behaviour1k_runner.py       # Behaviour-1K evaluation
│   └── openx_runner.py             # Open X-Embodiment evaluation
│   │
│   └── 📝 Purpose: Metric computation, logging,
│                   experiment orchestration
│
├── ⚙️ config/                      # Experiment configurations
│   ├── models/
│   │   ├── octo_27m.yaml
│   │   ├── octo_93m.yaml
│   │   ├── minivla.yaml
│   │   └── smolvla.yaml
│   └── benchmarks/
│       ├── libero.yaml
│       ├── embodiedbench.yaml
│       └── behaviour1k.yaml
│   │
│   └── 📝 Purpose: Paths, hyperparameters, model variants,
│                   dataset splits (human-readable YAML/JSON)
│
├── 💾 data/                        # Dataset storage (gitignored)
│   ├── examples/                   # Small demo datasets (tracked)
│   ├── libero/                     # LIBERO-90 data (gitignored)
│   ├── embodiedbench/              # EmbodiedBench data (gitignored)
│   ├── behaviour1k/                # Behaviour-1K data (gitignored)
│   └── openx/                      # Open X-Embodiment (gitignored)
│   │
│   └── 📝 Purpose: Download scripts, pointers to raw storage
│                   Large datasets NOT stored in git
│
├── 🤖 models/                      # Model repositories (gitignored)
│   ├── octo/                       # Cloned from github.com/octo-models/octo
│   ├── minivla/                    # MiniVLA repository
│   ├── smolvla/                    # SmolVLA/LeRobot repository
│   └── tinyvla/                    # TinyVLA repository
│   │
│   └── 📝 Purpose: Local clones of model code and weights
│                   NOT tracked in git to avoid binary bloat
│
├── 🔬 notebooks/                   # Interactive analysis
│   ├── 01_octo_baseline.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_visualizations.ipynb
│   │
│   └── 📝 Purpose: Quick analysis, visualization,
│                   interactive debugging
│
├── 📊 results/                     # Evaluation outputs
│   ├── octo/
│   │   ├── libero_results.json
│   │   ├── embodiedbench_results.json
│   │   └── logs/
│   ├── minivla/
│   ├── smolvla/
│   └── analysis/
│       ├── comparative_analysis.csv
│       └── plots/
│   │
│   └── 📝 Purpose: Logs, metrics, summaries
│                   Only small examples tracked in git
│
├── 🛠️ scripts/                     # Utility scripts
│   ├── setup_octo.sh               # OCTO-specific setup
│   ├── setup_minivla.sh            # MiniVLA setup
│   ├── download_checkpoints.sh     # Download model weights
│   ├── download_data.sh            # Download datasets
│   └── run_benchmark.py            # Main evaluation script
│   │
│   └── 📝 Purpose: Dataset prep, batch runners,
│                   job submission helpers
│
└── 📚 docs/                        # Documentation
    ├── adaptations.md              # Dataset/model-specific adaptations
    └── troubleshooting.md          # Common issues and solutions
    │
    └── 📝 Purpose: Single source of truth for decisions
                    (preprocessing, transforms, mappings)
```

### 🏗️ Design Principles

- **Clear separation of concerns**: Adapters convert data → Benchmarks run experiments → Models remain local
- **Easy extensibility**: Add new models by dropping code in `models/` and creating adapter in `adapters/`
- **No evaluation code changes needed** when adding models

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM
- 8GB+ VRAM (for OCTO-93M)

### Setup

```bash
# 1. Clone repository

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Clone model repositories
bash setup.sh

# 5. Setup OCTO specifically
bash scripts/setup_octo.sh

# 6. Download OCTO checkpoints
python scripts/download_checkpoints.py --model octo --variant base-93m
```

---

## 📋 Roadmap & Next Steps

### ✅ Phase 1: OCTO Baseline (Week 1-2)

**1.1 Setup OCTO**
- [ ] Clone OCTO into `models/octo/` (gitignored)
- [ ] Verify installation: `cd models/octo && python examples/01_inference_pretrained.py`
- [ ] Download pre-trained checkpoint (27M or 93M variant)

**1.2 Create OCTO Adapter**
- [ ] Implement `adapters/octo_adapter.py` inheriting from `base_adapter.py`
- [ ] Key methods:
  - `load_model()` - Load checkpoint
  - `predict()` - Run inference
  - `preprocess_observation()` - Convert benchmark → OCTO format
  - `postprocess_action()` - Convert OCTO output → benchmark format

**1.3 Test Adapter**
- [ ] Create small sample data in `data/examples/`
- [ ] Run: `python tests/test_adapters.py --model octo`
- [ ] Validate output shapes and types

### 🔄 Phase 2: Reproduce OCTO Results (Week 2-3)

**2.1 Setup Open X-Embodiment**
- [ ] Download dataset: `bash scripts/download_data.sh --dataset openx --subset bridge`
- [ ] Document data layout in `docs/adaptations.md`

**2.2 Implement Benchmark Runner**
- [ ] Create `benchmarks/openx_runner.py`
- [ ] Implement evaluation loop (load episodes → run inference → compute metrics)

**2.3 Validate Against Published Results**
- [ ] Run: `python scripts/run_benchmark.py --model octo --benchmark openx`
- [ ] Compare your metrics with OCTO paper
- [ ] **Validation checkpoint**: Results match within ±5%

### 🎯 Phase 3: Expand to Target Benchmarks (Week 3-5)

**3.1 Setup EmbodiedBench**
- [ ] Download: `bash scripts/download_data.sh --dataset embodiedbench`
- [ ] Study observation format (RGB, depth, proprioception)
- [ ] Understand action space (discrete vs continuous)

**3.2 Create EmbodiedBench Adapter**
- [ ] Implement `benchmarks/embodiedbench_runner.py`
- [ ] Map observations: EmbodiedBench → OCTO format
- [ ] Map actions: OCTO → EmbodiedBench action space
- [ ] Document all adaptations in `docs/adaptations.md`

**3.3 Run Initial Evaluation**
- [ ] Execute benchmark: `python scripts/run_benchmark.py --model octo --benchmark embodiedbench`
- [ ] Save results to `results/octo/embodiedbench_initial.json`
- [ ] Document challenges in `results/octo/embodiedbench_notes.md`

### 🔬 Phase 4: Add More Models (Week 5-8)

Repeat Phases 1-3 for additional models in this order:

1. **MiniVLA** - Similar architecture, good efficiency comparison
2. **SmolVLA** - HuggingFace integration simplifies testing
3. **TinyVLA** - Diffusion-based, interesting architectural contrast

### 📊 Phase 5: Comparative Analysis (Week 8+)

- [ ] Create comparison notebooks in `notebooks/`
- [ ] Generate plots: success rate vs model size, inference speed
- [ ] Analyze trade-offs and document findings
- [ ] Compile results into comparative tables

---

## 🎓 Models Under Evaluation

Focus on efficient VLA models with **<1B parameters**:

| Model | Parameters | Architecture | Key Features |
|-------|-----------|--------------|--------------|
| **OCTO** | 27M-93M | CNN + Diffusion Policy | 500x smaller than RT-2, matches performance |
| **MiniVLA** | ~1B | Qwen 2.5 0.5B + ViT | 82% on LIBERO-90, 7x smaller than OpenVLA |
| **SmolVLA** | 450M | SmolVLM-based | HuggingFace ecosystem, LeRobot training |
| **TinyVLA** | <1B | Small VLM + Diffusion | 20x faster inference, LoRA fine-tuning |

---

## 📚 Benchmarks

- **Open X-Embodiment**: Large-scale robot manipulation (OCTO training data)
- **LIBERO-90**: Manipulation tasks in simulation
- **EmbodiedBench**: Vision-driven agent evaluation (reasoning, perception)
- **Behaviour-1K**: 1000 everyday activities across 50 environments
- **CLIPort**: Robot manipulation task subsets

---

## ⚠️ Expected Challenges

| Challenge | Solution |
|-----------|----------|
| **Action space mismatch** | Write small action mappers, document in `docs/adaptations.md` |
| **Observation format differences** | Centralize preprocessing in adapters, list accepted shapes |
| **Task instruction variations** | Create normalized templates per dataset |
| **Coordinate transforms** | Document all transforms in `docs/adaptations.md` |

---

## 🔬 Reproducibility Guidelines

### Keep Experiments Traceable
- ✅ Save configs under `config/` with random seeds
- ✅ Record all hyperparameters in results
- ✅ Document adaptations in `docs/adaptations.md`

### Keep Repository Clean
- ❌ No large weights or datasets in git
- ✅ Clone to `models/`, add to `.gitignore`
- ✅ Use small example datasets in `data/examples/` for CI checks

### Version Control
- Track: Code, configs, small example results
- Ignore: Models, checkpoints, full datasets, large result files

---

## 🔗 Resources

### OCTO (Start Here)
- 🌐 **Project**: https://octo-models.github.io/
- 💻 **GitHub**: https://github.com/octo-models/octo
- 🤗 **Model Hub**: https://huggingface.co/rail-berkeley/octo-base
- 📄 **Paper**: https://arxiv.org/pdf/2405.12213 

### Benchmark Papers
- **Behaviour-1K**: https://arxiv.org/abs/2403.09227
- **EmbodiedBench**: https://arxiv.org/abs/2502.09560
- **CLIPort**: https://arxiv.org/abs/2109.12098
- **Open X-Embodiment**: https://robotics-transformer-x.github.io/
- **LIBERO-90**: https://github.com/Lifelong-Robot-Learning/LIBERO

### Other VLA Models
- **MiniVLA**: https://ai.stanford.edu/blog/minivla
- **SmolVLA**: https://huggingface.co/docs/lerobot
- **TinyVLA**: https://tiny-vla.github.io/

---

## 🛠️ Contributing

### Adding a New Model
1. Clone model repo to `models/<model_name>/`
2. Create `adapters/<model_name>_adapter.py` inheriting from `base_adapter.py`
3. Add tests: `tests/test_<model_name>_adapter.py`
4. Update this README

### Adding a New Benchmark
1. Download dataset to `data/<benchmark_name>/`
2. Create `benchmarks/<benchmark_name>_runner.py` inheriting from `base_benchmark.py`
3. Create config: `config/benchmarks/<benchmark_name>.yaml`
4. Document adaptations in `docs/adaptations.md`

---

## 📝 Notes

- CI/smoke tests can validate single-sample OCTO inference on PRs
- Maintain tiny reference dataset in `data/examples/` for automated testing
- All dataset→model adaptations go in `docs/adaptations.md` (single source of truth)

---

## 📧 Contact

[Your contact information or research group details]

---

**Status**: 🚧 In Development - Phase 1 (OCTO Baseline)  
**Last Updated**: [Current Date]