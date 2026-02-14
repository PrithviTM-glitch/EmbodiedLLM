# State Encoder Analysis in Vision-Language-Action Models

## Research Objective

Investigate why proprioceptive state encoders (robot joint positions, gripper state, end-effector pose) fail to contribute meaningfully in modern VLA architectures, and develop methods for more information-dense state representations that improve multi-encoder fusion.

---

## Background

VLA models integrate multiple modalities:
- **Vision**: RGB/depth camera observations
- **Language**: Task instructions and goals
- **Proprioception**: Joint angles, gripper state, end-effector position/orientation
- **Actions**: Target joint velocities, gripper commands

**Core Problem**: Proprioceptive state encoders are consistently underutilized across architectures. Despite being critical for precise manipulation, these encoders receive minimal gradient flow, show high redundancy with visual encoders, and contribute <5-10% to task performance in ablation studies.

---

## Research Questions

1. **Why do proprioceptive state encoders fail in Evo-1 and other VLAs?**
   - Measure gradient flow, information content, and downstream utilization
   - Identify architectural bottlenecks preventing effective integration

2. **How can we encode proprioceptive state more efficiently?**
   - Design information-dense representations that capture essential spatial/kinematic state
   - Reduce dimensionality while maintaining or improving task performance

3. **What fusion mechanisms effectively integrate proprioceptive state?**
   - Compare concatenation, cross-attention, gating, and hierarchical approaches
   - Determine when and why certain mechanisms outperform others

---

## Representative Models (Core Study)

### Model Selection Criteria
- Architecture diversity (autoregressive, diffusion, flow-based)
- Different fusion mechanisms
- Varying treatment of proprioceptive state
- Availability of pretrained checkpoints

### Recommended 3-Model Set

**1. Evo-1 (MINT-SJTU)**
- Evolutionary optimization approach to VLA design
- Baseline for understanding proprioceptive encoding failures
- Medium scale (~3-7B parameters)

**2. OpenVLA (7B)**
- Vision-as-prefix architecture (SigLIP + LLaMA)
- Late fusion: vision features prepended to language stream
- Well-documented, reproducible baseline

**3. Octo (93M - 1B)**
- Diffusion transformer for action prediction
- Observation-centric with explicit proprioception conditioning
- Different paradigm from autoregressive models

### Alternative 3-Model Sets

**Option A: Focus on Fusion Mechanisms**
- Evo-1 (evolutionary fusion)
- Pi0 (flow matching fusion)
- RT-2 (cross-attention fusion)

**Option B: Scale Analysis**
- Octo-Base (93M)
- Evo-1 or Pi0 (3-7B)
- RT-2 (1B+ variant)

---

## Extended Study Models (Optional)

**4. Pi0 (3B)**
- Flow matching for action generation
- Modern efficient architecture
- Different state conditioning mechanism

**5. RT-2 / RT-2-X (1B-55B variants)**
- Google's production VLA
- Cross-attention between modalities
- Industry-standard baseline

**6. GR-1 (1.4B)**
- Unified tokenization approach
- All modalities (vision, language, proprio, actions) as tokens
- GPT-style architecture

**7. VIMA (200M+)**
- Object-centric multimodal transformer
- Prompt-based manipulation
- Explicit object state encoding

**8. RoboFlamingo (7B+)**
- Flamingo-style interleaved vision-language
- Frozen encoder approach
- Tests whether frozen encoders affect proprioception usage

---

## Methodology

### Week 1: Diagnostic Analysis

#### Days 1-2: Model Setup & Architecture Analysis
**For each model (Evo-1, OpenVLA, Octo):**

- [ ] Load pretrained checkpoints
- [ ] Map architecture components (identify where proprioception enters)
- [ ] Document fusion mechanisms (concatenation, cross-attention, etc.)
- [ ] Identify proprioceptive encoder architecture (MLP, transformer, etc.)
- [ ] Note dimensionality: vision features, language features, proprio features

#### Days 3-4: Encoder Utilization Measurements

**Gradient Flow Analysis:**
- [ ] Measure gradient magnitude to proprioceptive encoder during eval
- [ ] Compare against vision encoder and language encoder gradients
- [ ] Track gradient ratios: proprio_grad / vision_grad
- [ ] Identify if gradients vanish or saturate at specific layers

**Representation Quality:**
- [ ] Compute effective rank of proprioceptive encoder outputs
- [ ] Measure intrinsic dimensionality (participation ratio)
- [ ] Calculate mutual information between proprio features and predicted actions
- [ ] Compare information content: proprio vs vision vs language

**Downstream Utilization:**
- [ ] Measure cosine similarity: proprio features at layer N vs layer N+5
- [ ] If similarity is high (>0.9), features aren't being transformed/used
- [ ] Track attention weights if cross-attention is used (what % goes to proprio?)
- [ ] Probe intermediate layers: what information from proprio is retained?

#### Days 4-5: Ablation Studies

**Encoder Ablation:**
- [ ] Baseline: full model performance (success rate on LIBERO/MetaWorld)
- [ ] Ablate vision encoder: measure performance drop
- [ ] Ablate language encoder: measure performance drop  
- [ ] Ablate proprioceptive encoder: measure performance drop
- [ ] Expected finding: proprio ablation causes <5-10% drop (indicates underutilization)

**Fusion Point Ablation:**
- [ ] Zero out proprio features before fusion
- [ ] Zero out proprio features after fusion (at action decoder input)
- [ ] Compare drops to understand where information is lost

**Task-Specific Ablation:**
- [ ] Test on tasks requiring precise spatial reasoning (peg insertion, stacking)
- [ ] Test on tasks with less spatial precision (pushing, toppling)
- [ ] Hypothesis: proprio should matter more for precision tasks

#### Day 5: Root Cause Identification

**Test Hypotheses:**

**H1: Redundancy with Vision**
- [ ] Compute CKA/SVCCA between vision features and proprio features
- [ ] High similarity (>0.7) suggests vision encoder already captures spatial state
- [ ] Check if vision encoder implicitly learns joint positions from visual feedback

**H2: Optimization Dynamics**
- [ ] Track quando model "learns to ignore" proprio during training
- [ ] Plot proprio encoder gradient norms over training steps
- [ ] Check learning rate: is proprio encoder undertrained?

**H3: Architectural Bottleneck**
- [ ] Measure fusion bottleneck: is dimensionality mismatch an issue?
- [ ] Vision: 768-dim, Language: 512-dim, Proprio: 64-dim → concatenation favors higher-dim inputs
- [ ] Check if increasing proprio dimensionality improves utilization

**H4: Information Density**
- [ ] Measure bits per dimension in proprio encoder
- [ ] Compare against vision/language: is proprio encoding less efficiently?
- [ ] Calculate compression ratio: input state dimension / encoded dimension

---

### Week 2: State Encoding Improvements

#### Days 6-8: Design & Implement State Encoding Variants

**Variant 1: Hierarchical Spatial-Temporal Encoding**

*Rationale*: Separate spatial (joint positions, end-effector pose) from temporal (velocities, accelerations) state

- [ ] Encode joint positions/gripper state as spatial features
- [ ] Encode joint velocities/recent actions as temporal features
- [ ] Use SSM/Mamba for temporal encoding (captures dynamics)
- [ ] Fuse hierarchically: spatial + temporal → unified proprio state

**Variant 2: Task-Conditioned Proprioception Encoding**

*Rationale*: Make proprio encoding query-dependent (what action are we predicting?)

- [ ] Use cross-attention: action query attends to proprio features
- [ ] Different tasks emphasize different joints (gripper vs arm vs base)
- [ ] Encoding adapts based on task instruction embedding
- [ ] Reduces redundancy by encoding only task-relevant state

**Variant 3: Compressed Multi-Resolution Encoding**

*Rationale*: Force information density through bottleneck, multiple resolutions

- [ ] Encode proprio at multiple resolutions (fine-grained joints, coarse end-effector)
- [ ] Use vector quantization (VQ) or learned compression
- [ ] Target: reduce from 64-dim to 16-32 dim without information loss
- [ ] Measure: does compression force more information-dense encoding?

**Variant 4: Differential State Encoding**

*Rationale*: Encode differences/changes rather than absolute positions

- [ ] Encode delta from reference pose (e.g., neutral position)
- [ ] Encode velocity/acceleration directly (dynamics-focused)
- [ ] Reduces redundancy with vision (which shows absolute pose)
- [ ] Emphasizes information vision can't easily capture

#### Days 9-11: Integration & Benchmarking

**Hook Variants into Models:**

For each model (Evo-1, OpenVLA, Octo) and each variant:

- [ ] Add state encoder as module injected via forward hooks
- [ ] Use residual connection to avoid disrupting pretrained weights
- [ ] Start with small mixing coefficient (0.1-0.2)
- [ ] Gradually increase if utilization improves

**Measurement Protocol:**

- [ ] Run on LIBERO benchmark (long-horizon tasks, 90 tasks across 10 suites)
- [ ] Run on MetaWorld (50 diverse manipulation tasks)
- [ ] If available: real robot data (Bridge V2, RT-X datasets)

**Metrics to Track:**

*Performance:*
- [ ] Task success rate (primary metric)
- [ ] Precision on spatial tasks (peg insertion, stacking)
- [ ] Generalization to unseen task variations

*Utilization:*
- [ ] Gradient magnitude to state encoder
- [ ] Ablation delta: performance drop when state encoder removed
- [ ] Information flow: mutual information with actions

*Efficiency:*
- [ ] Parameters added
- [ ] Inference latency impact
- [ ] Compression ratio achieved

**Analysis by Task Type:**

- [ ] Spatial precision tasks (does proprio encoding help more?)
- [ ] Dynamic tasks (does temporal encoding help more?)
- [ ] Language-conditioned tasks (does task-conditioned encoding help?)

#### Days 12-14: Multi-Encoder Fusion Analysis

**Compare Fusion Mechanisms:**

**Baseline Fusion (model's default):**
- [ ] Document current approach (concatenation, cross-attention, etc.)
- [ ] Measure information flow through fusion layer
- [ ] Identify bottlenecks

**Alternative Fusion 1: Gated Fusion**
- [ ] Learn per-modality gates: gate_vision, gate_language, gate_proprio
- [ ] Fusion = gate_v * vision + gate_l * language + gate_p * proprio
- [ ] Measure: do gates learn to balance modalities better?
- [ ] Track gate values over inference: which modality dominates when?

**Alternative Fusion 2: Cross-Attention Fusion**
- [ ] Language/proprio as queries, vision as key/value
- [ ] Vision as query, language/proprio as key/value
- [ ] Compare: which configuration utilizes proprio better?

**Alternative Fusion 3: Hierarchical Fusion**
- [ ] Stage 1: fuse vision + proprio (spatial modalities)
- [ ] Stage 2: fuse result with language (task conditioning)
- [ ] Rationale: group modalities by information type
- [ ] Measure: does staging improve proprio utilization?

**Alternative Fusion 4: Modality-Specific Decoders**
- [ ] Separate decoder branches for each modality
- [ ] Late fusion only at action prediction head
- [ ] Prevents modality dominance in shared decoder
- [ ] Measure: does parallel processing help?

**Integration Method Analysis:**

For each fusion mechanism:
- [ ] Measure gradient flow to each modality encoder
- [ ] Compute information contribution (Shapley values or ablation)
- [ ] Track attention weights if applicable
- [ ] Identify optimal architecture for proprio integration

**Generalization Across Models:**

- [ ] Does best fusion mechanism generalize across Evo-1, OpenVLA, Octo?
- [ ] Or is optimal fusion architecture-specific?
- [ ] Identify principles that transfer vs model-specific tuning

---

## Benchmarking Environments

### Simulation (Recommended for 2-week timeline)

**LIBERO (Long-Horizon Tasks):**
- 90 tasks across 10 suites
- Multi-step manipulation requiring spatial precision
- Language-conditioned tasks
- Good for testing proprioception on precise, multi-stage manipulation

**MetaWorld (Diverse Manipulation):**
- 50 tasks with standardized robot
- Varied spatial requirements (reaching, pushing, picking)
- Quick to run, good for ablations

**RLBench (Optional):**
- 100+ tasks with vision + language
- More complex than MetaWorld
- Slower but comprehensive

### Real Robot (If Available)

**Bridge V2:**
- Real-world manipulation dataset
- Tests generalization from simulation

**RT-X (Open X-Embodiment):**
- Cross-embodiment evaluation
- Tests if state encoding transfers across robot morphologies

---

## Expected Findings

### Diagnostic Phase (Week 1)

**Finding 1: Gradient Flow Imbalance**
- Vision encoder receives 10-50x more gradient than proprioceptive encoder
- Language encoder receives 5-20x more gradient than proprioceptive encoder
- Indicates optimization favors vision/language over proprio

**Finding 2: High Redundancy**
- CKA similarity between vision and proprio features: 0.6-0.8
- Vision encoder implicitly captures spatial state from visual feedback
- Proprioceptive encoder provides minimal unique information

**Finding 3: Low Ablation Impact**
- Removing vision: 30-50% success rate drop
- Removing language: 20-40% success rate drop
- Removing proprio: 5-15% success rate drop (confirms underutilization)

**Finding 4: Architecture-Specific Bottlenecks**
- Concatenation fusion: high-dimensional modalities dominate
- Cross-attention: query modality determines information flow
- Diffusion conditioning: proprioception poorly integrated into noise prediction

### Improvement Phase (Week 2)

**Finding 5: Information-Dense Encoding Helps**
- Compressed encoding (16-32 dim) performs similarly to full (64 dim)
- Forces removal of redundant information
- Improved utilization: gradient magnitude increases 2-3x

**Finding 6: Task-Conditioned Encoding Reduces Redundancy**
- Query-dependent encoding attends to task-relevant joints
- Reduces overlap with vision (CKA drops from 0.7 to 0.4)
- Improves precision task performance by 10-15%

**Finding 7: Temporal Encoding Captures Dynamics**
- SSM-based temporal encoding of velocities/accelerations
- Provides information vision can't easily extract
- Helps on dynamic tasks (throwing, dynamic grasping)

**Finding 8: Fusion Mechanism Matters More Than Encoding**
- Cross-attention fusion improves proprio utilization 2-3x over concatenation
- Hierarchical fusion (vision+proprio first) helps spatial tasks
- Gated fusion learns to balance modalities (prevents vision dominance)

**Finding 9: Architecture-Specific Optimal Strategies**
- Autoregressive models (Evo-1, OpenVLA): hierarchical fusion works best
- Diffusion models (Octo): task-conditioned encoding works best
- Large models benefit more from compression (redundancy is higher)

---

## Deliverables

### Technical Outputs

**1. Diagnostic Report**
- Quantified encoder utilization across 3 models
- Gradient flow analysis with visualizations
- Ablation study results table
- Root cause identification for each model

**2. State Encoding Comparison**
- Performance table: baseline vs 4 encoding variants
- Utilization metrics: gradient magnitude, ablation delta
- Task-specific analysis: which tasks benefit from which encoding
- Compression analysis: information density measurements

**3. Fusion Mechanism Analysis**
- Comparison table: 4 fusion mechanisms across 3 models
- Gradient flow and attention weight visualizations
- Recommendations: optimal fusion for each architecture type
- Generalization analysis: transferable principles vs model-specific

**4. Implementation Guide**
- How to add state encoders to existing VLA architectures
- Hook-based integration (no retraining required)
- Best practices for each architecture type
- Code snippets for measurement tools

### Research Contributions

**1. Characterization of the Problem**
- First systematic study of proprioceptive encoder underutilization
- Quantified across multiple VLA architectures
- Identified root causes: redundancy, optimization dynamics, fusion bottlenecks

**2. Solution Framework**
- Information-dense encoding methods
- Architecture-specific fusion strategies
- Guidelines for effective multi-encoder integration

**3. Actionable Recommendations**
- For model developers: how to design better state encoders
- For practitioners: which existing models use proprio effectively
- For researchers: open problems and future directions

---

## Success Metrics

### Primary Metrics
- **Proprioceptive encoder utilization**: gradient magnitude ratio (proprio/vision) increases from <0.1 to >0.3
- **Task performance**: success rate improvement on precision tasks (>10%)
- **Information efficiency**: maintain performance with 50% reduction in proprio encoding dimension

### Secondary Metrics
- **Ablation impact**: removing improved proprio encoder causes >20% drop (vs <10% baseline)
- **Redundancy reduction**: CKA similarity (vision/proprio) decreases from >0.7 to <0.5
- **Generalization**: findings hold across 3+ different architectures

---

## Timeline Summary

### Week 1: Diagnosis
- **Days 1-2**: Model setup, architecture analysis
- **Days 3-4**: Utilization measurements (gradients, representations, ablations)
- **Day 5**: Root cause identification

### Week 2: Solutions
- **Days 6-8**: Design and implement 4 encoding variants
- **Days 9-11**: Integration, benchmarking, measurement
- **Days 12-14**: Fusion mechanism analysis, write-up

---

## Extensions (Beyond 2 Weeks)

### Extended Model Analysis (Week 3-4)
- Add Pi0, RT-2, GR-1, VIMA to model set
- Test generalization of findings across 7-8 architectures
- Identify architecture families with similar challenges

### Fine-Tuning Study (Week 3-6)
- Fine-tune models with improved state encoders
- Measure sample efficiency improvements
- Test on real robot deployments

### Theoretical Analysis (Week 3-4)
- Information-theoretic characterization of state encoding
- Optimal compression bounds for proprioceptive state
- Formal analysis of fusion mechanisms (capacity, identifiability)

### Cross-Domain Generalization (Week 3-4)
- Test on different robot morphologies (arms, mobile manipulators, hands)
- Test on different tasks (navigation, locomotion, dexterous manipulation)
- Identify domain-specific vs universal findings

---

## Resources Required

### Compute
- **Minimum**: 1x A100 (40GB) for inference benchmarking
- **Recommended**: 4x A100 for parallel experiments
- **Storage**: ~100GB for models and datasets

### Software
- PyTorch, HuggingFace Transformers
- Model-specific repos: Evo-1, OpenVLA, Octo
- Simulation environments: LIBERO, MetaWorld
- Measurement tools: scikit-learn (CKA), custom gradient hooks

### Data
- LIBERO benchmark suite
- MetaWorld tasks
- Optional: Bridge V2, RT-X datasets

---

## Open Questions for Future Work

1. **Does proprioceptive state encoding quality affect sim-to-real transfer?**
   - Better encoding might reduce reality gap

2. **Can we learn to compress proprioception end-to-end?**
   - Learned compression vs hand-designed features

3. **Does human teleoperation data encode proprioception differently?**
   - Compare human demonstrations vs autonomous policies

4. **What's the information-theoretic lower bound?**
   - Minimum bits needed to encode task-relevant proprioceptive state

5. **Do larger models naturally learn better proprioceptive encoding?**
   - Scaling laws for state encoder utilization

---

## References to Review

- OpenVLA paper (Kim et al.) -- https://arxiv.org/abs/2406.09246 
- Octo paper (generalist robot policies) -- https://arxiv.org/abs/2405.12213 
- RT-2 paper (vision-language-action models) -- https://arxiv.org/abs/2307.15818 
- Evo-1 paper (evolutionary VLA design) -- https://arxiv.org/abs/2511.04555 
- Information bottleneck theory (Tishby) -- https://arxiv.org/abs/physics/0004057 
- Representation learning in multimodal models -- https://arxiv.org/abs/2503.08497 
- Fusion mechanisms in vision-language models -- https://arxiv.org/abs/2504.09925 