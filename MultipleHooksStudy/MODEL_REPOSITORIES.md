# VLA Model Repositories & Checkpoints

**Last Updated**: February 14, 2026

This document contains all GitHub repositories and HuggingFace model checkpoints for the VLA models being analyzed.

---

## 1. OpenVLA (7B Parameters)

**Description**: Vision-Language-Action model with no proprioceptive encoder (baseline)

### Links
- **GitHub**: https://github.com/openvla/openvla
- **HuggingFace Model**: https://huggingface.co/openvla/openvla-7b
- **Paper**: https://arxiv.org/abs/2406.09246

### Architecture
- **Vision**: SigLIP-400M (ViT-SO-400M)
- **Language**: Llama-2 7B
- **Proprio Encoder**: None (vision-as-prefix fusion)
- **Framework**: PyTorch ✅
- **Key Classes**: `PrismaticVLM`, `VisionBackbone`, `LLMBackbone`

### Model Loading
```python
from transformers import AutoModelForVision2Seq, AutoProcessor

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
```

---

## 2. Octo (93M Parameters)

**Description**: Diffusion-based VLA with linear state encoder

### Links
- **GitHub**: https://github.com/octo-models/octo
- **HuggingFace Model**: https://huggingface.co/rail-berkeley/octo-base
- **Paper**: https://arxiv.org/abs/2405.12213

### Architecture
- **Vision**: ResNet or ViT variants
- **Language**: T5 encoder
- **Proprio Encoder**: Linear projection + position embeddings
- **Framework**: JAX/Flax ❌ (INCOMPATIBLE with PyTorch hooks)
- **Key Classes**: `OctoModule`, `OctoTransformer`, `ImageTokenizer`

### Model Loading
```python
from octo.model.octo_model import OctoModel

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
```

### ⚠️ Critical Note
**Octo uses JAX/Flax, NOT PyTorch**. All PyTorch hook infrastructure (`register_forward_hook`, `register_full_backward_hook`) will NOT work. Would require complete rewrite using JAX's `jax.grad()` and custom transformation functions.

---

## 3. RDT-1B (1.2B Parameters)

**Description**: Robotics Diffusion Transformer with state encoder

### Links
- **GitHub**: https://github.com/thu-ml/RoboticsDiffusionTransformer
- **HuggingFace Model**: https://huggingface.co/robotics-diffusion-transformer/rdt-1b
- **Paper**: https://arxiv.org/abs/2410.21257

### Architecture
- **Vision**: SigLIP-SO-400M or CLIP variants
- **Language**: T5-XXL encoder
- **Proprio Encoder**: `state_adaptor` (Linear layer)
- **Framework**: PyTorch ✅
- **Key Classes**: `RDTRunner`, `RDT`, `RDTBlock`

### Model Loading
```python
from models.rdt_runner import RDTRunner

model = RDTRunner.from_pretrained("robotics-diffusion-transformer/rdt-1b")
```

### Architecture Details (Verified)
```python
# From models/rdt_runner.py
self.state_adaptor = nn.Linear(
    in_features=state_token_dim * 2,    # state + state mask
    out_features=hidden_size
)
self.img_adaptor = nn.Linear(img_token_dim, hidden_size)
self.lang_adaptor = nn.Linear(lang_token_dim, hidden_size)
```

**Note**: No separate "Fourier layer" found in model code. Fourier features may be in data preprocessing pipeline.

---

## 4. π0 (Pi-Zero, 3.3B Parameters)

**Description**: Flow matching VLA with separate state encoder

### Links
- **GitHub**: https://github.com/physical-intelligence/pi0 (⚠️ Access currently failing)
- **HuggingFace Model**: https://huggingface.co/physical-intelligence/pi0 (if available)
- **Paper**: https://www.physicalintelligence.company/blog/pi0

### Architecture (Unverified)
- **Vision**: Pre-trained ViT
- **Language**: T5 or similar
- **Proprio Encoder**: Separate multi-layer encoder (assumed)
- **Action Generation**: Flow matching
- **Framework**: Likely PyTorch ⚠️
- **Key Features**: Block-wise causal masking, asymmetric conditioning

### Status
❓ **Repository access failed** - needs manual verification

---

## 5. Evo-1 (0.77B Parameters - Lightweight VLA)

**Description**: Lightweight VLA with preserved semantic alignment and two-stage training

### Links
- **GitHub**: https://github.com/MINT-SJTU/Evo-1 ✅
- **HuggingFace**: Not yet released (check MINT-SJTU or authors)
- **Paper**: https://arxiv.org/abs/2512.06951

### Architecture (Verified from Paper)
- **Vision-Language Backbone**: InternVL3-1B (native multimodal VLM)
- **Integration Module**: Aligns VL representations with robot state (proprio)
- **Action Generator**: Cross-modulated diffusion transformer
- **State Input**: Robot state s_t (proprioception)
- **Framework**: PyTorch ✅
- **Size**: 0.77B parameters (smallest VLA model)

### Key Components
```python
# From paper architecture description:
Input: {Images: I_t^i, Language: L_t, State: s_t}
↓
Vision-Language Backbone (InternVL3-1B)
↓
Integration Module (aligns VL + state)
↓
Cross-Modulated Diffusion Transformer
↓
Output: Action a_t
```

### Training Strategy
- **Two-Stage Training**: Preserves VLM semantic representations
  - Stage 1: Align integration module while freezing VLM
  - Stage 2: Fine-tune entire model with careful learning rates
- **No Robot Pretraining**: Trained from scratch on downstream tasks

### Evaluation Benchmarks
- Meta-World: 80.6% (SOTA, +12.4% vs previous best)
- LIBERO: 94.8% (competitive)
- RoboTwin: SOTA (+6.9% vs previous best)
- Real Robot: 78% success rate

### Status
✅ **Fully Verified** - Architecture confirmed from paper

---

## 6. EvoVLA (Alternative: Self-Evolving VLA)

**Description**: Self-supervised VLA framework (different from Evo-1)

### Links
- **GitHub**: https://github.com/AIGeeksGroup/EvoVLA
- **Paper**: https://arxiv.org/abs/2511.16166
- **Website**: https://aigeeksgroup.github.io/EvoVLA/

### Architecture
- **Base Model**: Built on OpenVLA
- **Key Features**:
  - Stage-Aligned Reward (triplet contrastive learning)
  - Pose-Based Object Exploration
  - Long-Horizon Memory
- **Framework**: PyTorch (inherits from OpenVLA)

### Benchmark
- Discoverse-L (long-horizon manipulation)
- Real-world robot deployment

---

## Summary Table

| Model | Size | Framework | Proprio Encoder | GitHub Status | Hook Compatible |
|-------|------|-----------|-----------------|---------------|-----------------|
| OpenVLA | 7B | PyTorch | ❌ None (baseline) | ✅ Verified | ✅ Yes |
| Octo | 93M | JAX/Flax | Linear projection | ✅ Verified | ❌ No (JAX) |
| RDT-1B | 1.2B | PyTorch | ✅ Linear adaptor | ✅ Verified | ✅ Yes |
| Evo-1 | 0.77B | PyTorch | ✅ Integration module | ✅ Verified | ✅ Yes |
| π0 | 3.3B | PyTorch? | ✅ Multi-layer? | ❓ Access failed | ⚠️ TBD |
| EvoVLA | 7B | PyTorch | OpenVLA-based | ✅ Verified | ✅ Yes |

---

## Recommended Model Selection for Hook Analysis

### Priority 1: PyTorch Models with Diverse Encoding Strategies ✅
1. **OpenVLA (7B)** - Baseline with NO proprio encoder
2. **Evo-1 (0.77B)** - Integration module aligns VL + state
3. **RDT-1B (1.2B)** - Linear state adaptor
4. **π0 (3.3B)** - Multi-layer encoder (if we can access it)

### Priority 2: Advanced Variants (Optional)
- **EvoVLA (7B)** - Long-horizon extension of OpenVLA

### Skip: Framework Incompatible
- ❌ **Octo (93M)** - JAX/Flax, requires complete rewrite for hooks

---

## Next Steps

### ✅ Completed Verification
1. ✅ Documented all repository links
2. ✅ Verified OpenVLA architecture (PyTorch, vision_backbone/llm_backbone)
3. ✅ Verified Octo architecture (JAX/Flax - incompatible)
4. ✅ Verified RDT-1B architecture (PyTorch, state_adaptor)
5. ✅ Verified Evo-1 architecture (PyTorch, integration module)

### 🔄 In Progress
1. **Fix OpenVLA Adapter** - Update attribute name priorities
2. **Fix RDT-1B Adapter** - Target state_adaptor instead of Fourier+MLP
3. **Create Evo-1 Adapter** - New adapter for integration module
4. **Document Octo Limitation** - Mark as JAX-incompatible
5. **Decide on π0** - Find implementation or skip

### 📋 Implementation Plan
1. **Update OpenVLA adapter** (5 min fix)
   - Add `vision_backbone`, `llm_backbone` to priority search
   - Test with `openvla/openvla-7b` checkpoint

2. **Rewrite RDT-1B adapter** (15 min fix)
   - Change target from `fourier_layer` + MLP to `state_adaptor`
   - Remove layer-wise profiling assumptions
   - Focus on single Linear layer analysis

3. **Create Evo-1 adapter** (30 min new code)
   - Discover InternVL3-1B backbone
   - Hook integration module (aligns VL + state)
   - Hook cross-modulated diffusion transformer
   - Add two-stage training awareness

4. **Test model loading** from HuggingFace
   - Load actual checkpoints
   - Verify `discover_model_structure()` works
   - Test hook attachment on real models

5. **Update documentation**
   - Mark Octo as JAX-incompatible
   - Update HOOKS_GUIDE.md with verified models
   - Update example_usage.py with real model loading
