# Vision-Language-Action (VLA) Model Evaluation Metrics

## Overview

This document describes the standard evaluation metrics for VLA (Vision-Language-Action) models like OCTO, RT-1, RT-2, and other generalist robotic policies.

## Evaluation Categories

VLA models are evaluated using two main categories of metrics:

### 1. **Online (Closed-Loop) Metrics** 🤖
These require actual robot execution and measure real-world performance.

### 2. **Offline (Open-Loop) Metrics** 📊
These use pre-recorded datasets and measure prediction accuracy without robot execution.

---

## Online Evaluation Metrics

### **Primary Metric: Success Rate**
- **Definition**: Percentage of task attempts that successfully achieve the goal
- **Range**: 0.0 to 1.0 (or 0% to 100%)
- **Example**: "Pick up the can" - did the robot grasp and lift the can?

**OCTO Paper Results (from website):**
- WidowX BridgeV2 (Language): 50% success
- WidowX BridgeV2 (Goal Image): 70% success  
- Stanford Blocks: 80% success
- Average across 6 finetuning tasks: 72% success

### **Secondary Online Metrics:**
1. **Episode Length**: Number of timesteps to complete task
2. **Completion Time**: Wall-clock time to finish
3. **Collision Rate**: Frequency of unintended collisions
4. **Failure Mode Analysis**: Why did failures occur?
   - Dropped object
   - Wrong object grasped
   - Timeout
   - Safety violation

---

## Offline Evaluation Metrics

These are what we can measure **without a robot**, using pre-recorded datasets like Open X-Embodiment.

### **1. Action Prediction Accuracy**

#### **Mean Squared Error (MSE)**
```
MSE = (1/N) * Σ(predicted_action - ground_truth_action)²
```
- Measures average squared difference between predicted and actual actions
- **Lower is better**
- Penalizes large errors more than small ones
- **Our Results**: 0.025 (50 episodes on fractal)

#### **Mean Absolute Error (MAE)**
```
MAE = (1/N) * Σ|predicted_action - ground_truth_action|
```
- Measures average absolute difference
- **Lower is better**
- More interpretable than MSE (same units as actions)
- **Our Results**: 0.079 (50 episodes on fractal)

#### **Per-Dimension MSE**
- MSE computed separately for each action dimension
- Useful for understanding which DoFs (Degrees of Freedom) are harder to predict
- Example: gripper action vs. XYZ position vs. rotation
- **Our Results**: Mean=0.025, Std=0.036

### **2. Directional Accuracy**

#### **Cosine Similarity**
```
cos_sim = (predicted · ground_truth) / (||predicted|| × ||ground_truth||)
```
- Measures alignment of action direction (ignoring magnitude)
- **Range**: -1.0 to 1.0
- **Higher is better** (closer to 1.0)
- **Our Results**: -0.013 (suggests nearly perpendicular predictions)

### **3. Temporal Consistency**

#### **Action Smoothness**
```
smoothness = Σ||action[t] - action[t-1]||²
```
- Measures jerkiness of predicted action sequence
- Lower is better (smoother trajectories)

#### **Autocorrelation**
- Measures how well the model maintains consistent predictions over time
- Important for stable robot control

---

## Comparison: Online vs Offline Metrics

| Aspect | Online (Robot) | Offline (Dataset) |
|--------|---------------|-------------------|
| **Cost** | High (robot time, wear) | Low (just compute) |
| **Speed** | Slow (real-time) | Fast (batch processing) |
| **Risk** | Physical damage possible | No risk |
| **Realism** | Most realistic | Approximation |
| **Reproducibility** | Harder (environmental variation) | Easier (fixed dataset) |
| **Primary Use** | Final validation | Development & debugging |

---

## OCTO Paper Evaluation Setup

Based on the OCTO paper and website, here's their evaluation methodology:

### **Training Data**
- **Dataset**: Open X-Embodiment (25 datasets, 800k episodes)
- **Size**: ~1.2TB preprocessed
- **Mix**: "oxe_magic_soup" - weighted combination of datasets

### **Offline Evaluation**
OCTO paper doesn't report detailed offline metrics (MSE/MAE), focusing on online success rates.

### **Online Evaluation - 9 Robot Setups:**

1. **WidowX BridgeV2** (UC Berkeley)
   - Language conditioning: 50% success
   - Goal image conditioning: 70% success
   - Tasks: Diverse object manipulation

2. **Stanford Blocks**
   - Success: 80%
   - Long-horizon task

3. **Berkeley Peg Insert**
   - Requires precise manipulation
   - Uses force-torque sensors (new observation modality)

4. **Berkeley Pick-Up**
   - Joint position control (new action space)

5. **Additional setups** at CMU, other institutions

### **Comparison Baselines:**
- **RT-1-X**: 35% avg success (language only)
- **RT-2-X**: 85% on WidowX, 50% avg (55B parameters)
- **From Scratch**: 20-25% avg
- **VC-1**: 15-30% avg
- **Octo**: 72% avg across 6 finetuning tasks

---

## What We're Currently Measuring

### ✅ **Implemented (Offline Metrics)**
1. Action MSE (overall and per-dimension)
2. Action MAE
3. Cosine Similarity
4. Success rate (based on MSE threshold)

### 🔄 **In Progress**
1. Extended evaluation on 50 episodes ✓ (Just completed!)
2. Multiple dataset evaluation

### ❌ **Not Yet Implemented**
1. Online robot evaluation (requires hardware)
2. Task-specific success criteria
3. Temporal smoothness metrics
4. Per-action-dimension analysis visualization

---

## Interpreting Our Results

### **From 50-Episode Evaluation on fractal20220817_data:**

```
Total Episodes: 50
MSE (Mean): 0.025
MAE (Mean): 0.079
Cosine Similarity: -0.013
Success Rate (MSE < 0.1): 46%
Avg Time per Episode: 12.73s
```

### **What This Means:**

1. **MSE of 0.025**: 
   - Actions are in range [-1, 1]
   - RMS error ≈ sqrt(0.025) ≈ 0.158
   - This is ~15.8% of the action range
   - **Interpretation**: Moderate prediction accuracy

2. **MAE of 0.079**:
   - Average absolute error is 7.9% of action range
   - **Interpretation**: Predictions deviate by ~8% on average

3. **Cosine Similarity of -0.013**:
   - Nearly orthogonal to ground truth
   - **Warning**: This suggests the model may not be capturing action direction well
   - Could indicate:
     - Model needs more finetuning on this dataset
     - Dataset distribution differs from training data
     - Action space mismatch

4. **46% Success Rate**:
   - Based on MSE < 0.1 threshold
   - Better than random, but room for improvement
   - OCTO paper shows 50-85% success on real robots (different metric)

---

## Recommendations for Reproducibility Research

To match OCTO's paper methodology:

### **1. Use Same Datasets**
- ✅ fractal20220817_data (we have this)
- ⬜ bridge_dataset (Berkeley Bridge V2)
- ⬜ Additional OXE datasets from their "magic soup" mix

### **2. Evaluate on Multiple Tasks**
- Download and evaluate on 3-5 different OXE datasets
- Compare offline metrics across datasets

### **3. Implement Additional Metrics**
- Per-dimension action analysis
- Temporal smoothness
- Success criteria specific to task types

### **4. Statistical Significance**
- ✅ Evaluate on 50+ episodes per dataset
- Compute confidence intervals
- Report mean ± std for all metrics

### **5. Comparison Benchmarks**
- If possible, evaluate RT-1-X or other baselines
- Compare with OCTO's reported offline performance (if available)

---

## Next Steps

1. **Download bridge_dataset** - OCTO's primary evaluation dataset
2. **Evaluate on 100+ episodes** for stronger statistics
3. **Implement per-dimension metrics** for detailed analysis
4. **Create visualization tools** for action prediction vs ground truth
5. **Document discrepancies** between our results and paper claims

---

## References

1. OCTO Paper: [https://octo-models.github.io/](https://octo-models.github.io/)
2. OCTO GitHub: [https://github.com/octo-models/octo](https://github.com/octo-models/octo)
3. Open X-Embodiment: [https://robotics-transformer-x.github.io/](https://robotics-transformer-x.github.io/)
4. RT-1: [https://robotics-transformer1.github.io/](https://robotics-transformer1.github.io/)
5. RT-2: [https://robotics-transformer2.github.io/](https://robotics-transformer2.github.io/)
