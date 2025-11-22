## Run LIBERO 90 Benchmark 

Aim is to run the LIBERO 90 benchmark on different VLA architectures and try to recreate the results from their respective papers.

### Folder Structure
- `adapters/` : Contains adapters for different models to run the LIBERO benchmark
this folder will be populated only if the model requires adapters to run the benchmark.
- `results/` : Contains results obtained after running the benchmark on different models. Each model will have a seperate folder containing the results.
- `models/` : Contains the cloned model repositories and will be gitignored to avoid git conflicts.
- `logs/` : Contains logs generated while running the benchmark on different models. each model will have a seperate log folder with all logs.
- `scripts/` : Contains scripts to run the benchmark on different models. This will have a base class to run the benchmark on any model and child classed in different files to run the benchmark on specific models.
- `docs/` : Contains all READMEs related to the models initialisation and results.

### Models

| Model | LIBERO-90 Success Rate | Parameters | Paper/Link |
|-------|------------------------|------------|------------|
| **OpenVLA-OFT** | 97.1% | 7B | [arxiv:2502.19645](https://arxiv.org/abs/2502.19645) |
| **Evo-1** | 94.8% | 0.77B | [arxiv:2511.04555](https://arxiv.org/abs/2511.04555) |
| **Pi0.5** | High 90s | 0.5B| [arxiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| **ECoT-Lite (full)** | 90.8% | 1B | [ecot-lite.github.io](https://ecot-lite.github.io) |
| **ECoT-Lite (dropout)** | 89.4% | 1B | [ecot-lite.github.io](https://ecot-lite.github.io) |
| **SmolVLA** | 87.3% | 0.45B | [arxiv:2506.01844](https://arxiv.org/abs/2506.01844) |
| **MiniVLA** | 82% | 1B | [Stanford AI Blog](https://ai.stanford.edu/blog/minivla) |
| **OpenVLA** | 62% | 7B | [arxiv:2406.09246](https://arxiv.org/abs/2406.09246) |


### Model Setup 
The models setup instructions will be documented in a seperate README file for each model. 

### Benchmark Setup

The git repository contains the instructions to run the benchmark on a model. Now There maybe some issues and all models may not run out of the box. In the case that the model requires adapters to run the LIBERO benchmakr, a custom adapter can be created in the `adapters` folder.

The following are installation instructions for the benchmark:

```bash
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -e .
```

### TO DOS

- [ ] Setup LIBERO benchmark environment

- Clone and setup model repositories for following models:
  - [ ] Evo-1
  - [ ] Pi0.5
  - [ ] ECoT-Lite
  - [ ] SmolVLA
  - [ ] OCTO 

- [ ] Run LIBERO benchmark on all models
    - [ ] Create adapters if necessary for relevant models
    - [ ] Collect results and compare with papers

- [ ] Document results in seperate README files one for each model 
    - [ ] Include results and comparison with paper results