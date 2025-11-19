# Dataset ↔ Model Adaptations

This document tracks all adaptations and mapping decisions made when converting dataset observations/actions to a model's expected inputs/outputs. Use this as the canonical place to record exact preprocessing, coordinate transforms, and validation/tests for reproducibility.

## Template entry (one per dataset-model pair)

- Date: YYYY-MM-DD
- Author: <name>
- Dataset: (e.g. Open X-Embodiment / LIBERO-90 / Behaviour-1K)
- Dataset version / split: (e.g. v1.0 / test)
- Model: (e.g. OCTO 27M)

### Observation mapping
- Raw observation fields available: (list)
- Selected fields passed to model: (list)
- Image preprocessing: (resize, normalization, color space)
- Camera intrinsics / extrinsics handling: (if applicable)
- Multi-view handling: (concatenate / stack / per-view processing)

### Action mapping
- Dataset action format: (discrete cmds / continuous control / tokenized)
- Model action format: (continuous vector / diffusion token / action tokens)
- Mapping strategy: (e.g. discrete -> continuous via mapping table, or quantization scheme)

### Timing / Episode handling
- Frame rate assumptions
- Episode segmentation and termination conditions

### Metric / Evaluation alignment
- Dataset metric definitions vs model metric computation
- Any normalization or thresholds adjusted for fair comparison

### Tests & Validation
- Small smoke test used: (path to `data/` sample)
- Expected shapes & sample outputs
- Unit tests added: (file paths)

### Notes / Open questions
- (List any approximations, assumptions, or unresolved issues.)

---

Add one filled section per dataset-model pair. Link to runs in `results/` with commitable config files in `config/` for full traceability.
