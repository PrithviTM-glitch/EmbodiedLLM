# OCTO Reproducibility Research Plan

## Objective
Reproduce OCTO's evaluation results to validate the reproducibility of their published findings and benchmark our implementation.

## OCTO Training Data: "OXE Magic Soup"

OCTO was trained on 25 datasets from Open X-Embodiment with the following weights:

### Top Priority Datasets (Highest Weights)
1. **nyu_franka_play_dataset** (weight: 3.0)
2. **taco_play** (weight: 2.0)
3. **viola** (weight: 2.0)
4. **roboturk** (weight: 2.0)
5. **berkeley_autolab_ur5** (weight: 2.0)
6. **stanford_hydra_dataset** (weight: 2.0)
7. **ucsd_kitchen_dataset** (weight: 2.0)
8. **berkeley_fanuc_manipulation** (weight: 2.0)

### Medium Priority Datasets
9. **bridge_dataset** (weight: 1.0) - **Primary evaluation dataset in paper**
10. **kuka** (weight: 0.83)
11. **fractal20220817_data** (weight: 0.54) - ✅ **We have this!**
12. **jaco_play** (weight: 1.0)
13. **berkeley_cable_routing** (weight: 1.0)

### Low Priority (Small weights)
- language_table (0.1)
- furniture_bench (0.1)
- bc_z (0.2)

---

## Evaluation Strategy

### Phase 1: Current Dataset (✅ In Progress)
- [x] Downloaded fractal20220817_data (10 shards, 1.2GB)
- [x] Ran evaluation on 3 episodes
- [x] Ran evaluation on 50 episodes
- [ ] Run evaluation on 100+ episodes for stronger statistics
- [ ] Download remaining fractal shards for comprehensive testing

**Current Results (50 episodes):**
```
MSE: 0.025
MAE: 0.079
Cosine Similarity: -0.013
Success Rate (MSE < 0.1): 46%
```

### Phase 2: Bridge Dataset (High Priority)
**bridge_dataset** is OCTO's primary evaluation dataset mentioned in the paper.

**Actions:**
1. Download bridge_dataset (10-20 shards initially)
2. Run same 50-episode evaluation
3. Compare metrics with fractal results
4. Check if OCTO paper reports any offline metrics for bridge

**Expected Download:**
```bash
cd scripts/octo
python download_from_gcs.py --dataset bridge_dataset --shards 20
```

### Phase 3: Additional Datasets
Based on training weights and availability:

1. **kuka** - RT-1 style robot (weight: 0.83)
2. **roboturk** - Diverse manipulation (weight: 2.0)
3. **berkeley_autolab_ur5** - UR5 robot (weight: 2.0)

### Phase 4: Multi-Dataset Analysis
- Aggregate results across all datasets
- Compute per-dataset metrics
- Identify which datasets OCTO performs best/worst on
- Compare with training weights (hypothesis: higher weight → better performance)

---

## Metrics to Track

### Current Metrics (✅ Implemented)
1. Action MSE (overall)
2. Action MAE
3. Cosine Similarity
4. Success Rate (MSE threshold-based)
5. Per-dimension MSE statistics

### Additional Metrics to Implement
1. **Per-Dimension Analysis**
   - Individual MSE for each of 7 action dimensions
   - Visualization: bar chart showing error per DoF
   
2. **Temporal Metrics**
   - Action smoothness (jerk)
   - Prediction stability over episode
   
3. **Task-Specific Metrics**
   - Language instruction compliance (if available)
   - Goal-reaching accuracy
   
4. **Statistical Measures**
   - Confidence intervals (95%)
   - Median values (robust to outliers)
   - Distribution plots

---

## Expected Results vs. Paper Claims

### What OCTO Paper Reports

**Online (Real Robot) Results:**
- WidowX BridgeV2 (language): 50% success
- WidowX BridgeV2 (goal image): 70% success
- Stanford Blocks: 80% success
- Average across 6 tasks: 72% success

**Comparison to Baselines:**
- RT-1-X: 35% average
- RT-2-X: 85% on WidowX, 50% average
- From Scratch: 20-25% average

**Offline Metrics:**
- ⚠️ **Not reported in paper** - they focus on online success rates

### What We Can Measure

Since we don't have robots, we measure **offline prediction accuracy**:

1. **Action MSE/MAE** - How accurate are predictions?
2. **Directional accuracy** - Are actions pointing the right way?
3. **Consistency** - Are predictions stable over time?

**Hypothesis:** Lower offline MSE/MAE should correlate with higher online success rates.

---

## Potential Discrepancies

### Why Our Results Might Differ:

1. **Different Evaluation Protocol**
   - We use offline metrics (prediction accuracy)
   - Paper uses online metrics (task success)
   - These are related but not identical

2. **Dataset Subset**
   - We evaluate on 10/1024 shards of fractal
   - May not be representative of full distribution
   - Solution: Download more shards

3. **Model Version**
   - We use octo-small-1.5 (27M params)
   - Paper may have used different checkpoints
   - Solution: Test both octo-small and octo-base

4. **Preprocessing Differences**
   - Our RLDS parsing may differ slightly
   - Action standardization might be different
   - Solution: Compare with OCTO's own data loaders

5. **Evaluation Episodes**
   - We randomly sample 50 episodes
   - Paper may use specific test set
   - Solution: Evaluate on larger sample (100-200 episodes)

---

## Action Items

### Immediate (Next 1-2 Hours)
- [x] Complete 50-episode fractal evaluation
- [ ] Analyze results in detail
- [ ] Create visualization of prediction vs. ground truth
- [ ] Download bridge_dataset (10-20 shards)

### Short Term (Next Session)
- [ ] Run 50-episode evaluation on bridge_dataset
- [ ] Compare fractal vs. bridge performance
- [ ] Implement per-dimension metric visualization
- [ ] Download kuka dataset

### Medium Term (This Week)
- [ ] Evaluate on 5+ different OXE datasets
- [ ] Generate comprehensive comparison report
- [ ] Implement confidence intervals
- [ ] Test octo-base model (93M params) for comparison

### Long Term (Research Goals)
- [ ] Correlate offline metrics with paper's online success rates
- [ ] Identify dataset-specific performance patterns
- [ ] Document reproducibility findings
- [ ] Publish comparison results

---

## Success Criteria

### Minimum Viable Reproducibility
✅ Evaluate OCTO on 3+ OXE datasets
✅ Report MSE, MAE, cosine similarity for each
✅ Compare relative performance across datasets
✅ Document any significant discrepancies

### Strong Reproducibility
✅ Evaluate on 5+ datasets from OXE Magic Soup
✅ 100+ episodes per dataset
✅ Statistical significance testing
✅ Per-dimension analysis
✅ Temporal consistency metrics

### Full Reproducibility (Ideal)
✅ Evaluate on all 25 datasets from Magic Soup
✅ Match offline metrics to online success rates (if correlation exists)
✅ Test multiple OCTO model versions
✅ Reproduce any reported offline benchmarks
✅ Validate training data distribution effects

---

## Data Storage Plan

### Current Storage
```
vla-benchmark/data/open-x/
├── fractal20220817_data/  (1.2 GB)
```

### Planned Storage
```
vla-benchmark/data/open-x/
├── fractal20220817_data/  (1.2 GB → expand to 5-10 GB)
├── bridge_dataset/         (target: 2-5 GB)
├── kuka/                   (target: 1-3 GB)
├── roboturk/               (target: 2-5 GB)
└── ...
```

**Total Expected:** 10-25 GB for comprehensive evaluation

---

## Timeline

| Phase | Duration | Datasets | Episodes | Deliverable |
|-------|----------|----------|----------|-------------|
| 1 | ✅ Done | fractal (1) | 50 | Initial metrics |
| 2 | 2-3 hours | bridge (1) | 50 | Two-dataset comparison |
| 3 | 1 day | +kuka, roboturk (3 total) | 150 total | Multi-dataset analysis |
| 4 | 2-3 days | 5+ datasets | 500+ total | Comprehensive report |

---

## Questions to Answer

1. ✅ **Can we measure OCTO's performance offline?** Yes - MSE, MAE, cosine sim
2. **Do offline metrics correlate with paper's online success rates?** TBD
3. **Which datasets does OCTO perform best on?** TBD - need more data
4. **Is octo-small different from octo-base?** TBD - test both
5. **Are our results reproducible across runs?** TBD - run multiple times
6. **What's the minimum data needed for valid evaluation?** TBD - test sample sizes

---

## References

- OCTO Website: https://octo-models.github.io/
- OCTO GitHub: https://github.com/octo-models/octo
- OXE Datasets: gs://gresearch/robotics/
- Paper Citation: Octo Model Team et al., RSS 2024
