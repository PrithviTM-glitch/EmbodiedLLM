# Action Prediction Metrics Explanation

This document explains how each evaluation metric is calculated in the VLA benchmark.

## Context

For robot manipulation, we predict **action vectors** at each timestep. For example, bridge_dataset uses 7-dimensional actions:
- **[0-2]**: End-effector position (x, y, z) delta
- **[3-5]**: End-effector rotation (roll, pitch, yaw) delta  
- **[6]**: Gripper open/close command

**Input shapes:**
- `predictions`: Shape `(T, 7)` - T timesteps, 7 action dimensions
- `ground_truth`: Shape `(T, 7)` - T timesteps, 7 action dimensions

---

## 1. Mean Squared Error (MSE)

**Formula:**
```
MSE = mean((predictions - ground_truth)²)
```

**Code:**
```python
mse = np.mean((predictions - ground_truth) ** 2)
```

**What it measures:**
- **Average squared difference** across all timesteps and dimensions
- Heavily penalizes large errors (squaring amplifies them)
- Units: squared action units (e.g., if actions are in meters, MSE is in m²)

**Example:**
```python
# Timestep 1: prediction=[0.1, 0.2, 0.3], ground_truth=[0.15, 0.18, 0.32]
# Difference: [-0.05, 0.02, -0.02]
# Squared: [0.0025, 0.0004, 0.0004]
# Average across all timesteps and dimensions
```

**Interpretation:**
- **Lower is better** (0 = perfect prediction)
- `MSE = 0.052` means predictions differ by ~0.23 units on average (sqrt(0.052))
- We use threshold `0.1` for "success" - episodes below this are considered good

---

## 2. Mean Absolute Error (MAE)

**Formula:**
```
MAE = mean(|predictions - ground_truth|)
```

**Code:**
```python
mae = np.mean(np.abs(predictions - ground_truth))
```

**What it measures:**
- **Average absolute difference** across all timesteps and dimensions
- Linear penalty (no squaring) - easier to interpret
- Units: same as actions (e.g., meters, radians)

**Example:**
```python
# Same example as above
# Difference: [-0.05, 0.02, -0.02]
# Absolute: [0.05, 0.02, 0.02]
# Average: 0.03
```

**Interpretation:**
- **Lower is better** (0 = perfect)
- `MAE = 0.088` means predictions differ by 0.088 units on average
- More intuitive than MSE because it's in the same units as the actions
- Less sensitive to outliers than MSE

**MSE vs MAE:**
- If `MSE` is much larger than `MAE²`, you have some large errors (outliers)
- Example: MSE=0.052, MAE²=0.078 → relatively consistent errors

---

## 3. Cosine Similarity

**Formula:**
```
cosine_similarity = mean(normalized_pred · normalized_gt)
```

**Code:**
```python
# Normalize each action vector to unit length
norm_pred = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
norm_gt = ground_truth / (np.linalg.norm(ground_truth, axis=1, keepdims=True) + 1e-8)

# Dot product of normalized vectors
cosine_sim = np.mean(np.sum(norm_pred * norm_gt, axis=1))
```

**What it measures:**
- **Direction similarity** between predicted and ground truth action vectors
- Ignores magnitude, focuses on whether actions point in the same direction
- Range: -1 (opposite direction) to +1 (same direction)

**Example:**
```python
# Timestep 1:
# pred = [0.3, 0.4, 0.0]  (normalized to [0.6, 0.8, 0.0])
# gt   = [0.6, 0.8, 0.0]  (normalized to [0.6, 0.8, 0.0])
# cosine = 0.6*0.6 + 0.8*0.8 + 0.0*0.0 = 1.0 (perfect direction match)

# Timestep 2:
# pred = [1.0, 0.0, 0.0]  (normalized to [1.0, 0.0, 0.0])
# gt   = [0.0, 1.0, 0.0]  (normalized to [0.0, 1.0, 0.0])
# cosine = 1.0*0.0 + 0.0*1.0 + 0.0*0.0 = 0.0 (perpendicular)
```

**Interpretation:**
- **Higher is better** (+1 = perfect alignment)
- `cosine_sim = 0.464` means predictions are pointing in roughly the right direction
- `0.464` indicates ~62° average angle between predicted and true actions
  - cos(62°) ≈ 0.47
- Values near 0: predictions are perpendicular to ground truth
- Negative values: predictions point opposite direction (very bad!)

**Why it's useful:**
- Even if magnitude is wrong, if direction is right, the robot moves the right way
- Complements MSE/MAE which care about exact values

---

## 4. MSE Per Dimension (Mean)

**Formula:**
```
mse_per_dim = mean((predictions - ground_truth)², axis=timesteps)
mse_per_dim_mean = mean(mse_per_dim)
```

**Code:**
```python
# Compute MSE for each dimension separately
mse_per_dim = np.mean((predictions - ground_truth) ** 2, axis=0)  # Shape: (7,)

# Then average across dimensions
mse_per_dim_mean = np.mean(mse_per_dim)
```

**What it measures:**
- MSE computed **separately for each action dimension**, then averaged
- Shows if errors are evenly distributed across dimensions

**Example:**
```python
# predictions.shape = (100, 7)  # 100 timesteps, 7 dimensions
# ground_truth.shape = (100, 7)

# Step 1: Compute MSE for each dimension
# mse_per_dim = [0.04, 0.05, 0.06, 0.03, 0.02, 0.08, 0.01]
#                 x     y     z    roll  pitch  yaw   grip

# Step 2: Average across dimensions
# mse_per_dim_mean = mean([0.04, 0.05, 0.06, 0.03, 0.02, 0.08, 0.01])
#                  = 0.041
```

**Interpretation:**
- Should be close to overall MSE (0.052 in our case)
- If much different from overall MSE, indicates non-uniform error distribution

---

## 5. MSE Per Dimension (Std)

**Formula:**
```
mse_per_dim_std = std(mse_per_dim)
```

**Code:**
```python
# Standard deviation of per-dimension MSEs
mse_per_dim_std = np.std(mse_per_dim)
```

**What it measures:**
- **Variance in prediction quality** across different action dimensions
- Shows if some dimensions are predicted much worse than others

**Example:**
```python
# Scenario 1: Uniform errors
# mse_per_dim = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
# std = 0.0 → all dimensions equally well predicted

# Scenario 2: Non-uniform errors  
# mse_per_dim = [0.01, 0.02, 0.03, 0.08, 0.15, 0.20, 0.01]
#                 x     y     z    roll  pitch  yaw   grip
# std = 0.072 → rotation predictions much worse than position
```

**Interpretation:**
- **Lower is better** (0 = all dimensions equally accurate)
- `mse_per_dim_std = 0.122` indicates significant variation
- Could mean:
  - Some dimensions (e.g., gripper) are easy to predict
  - Other dimensions (e.g., rotation) are hard to predict
  - Model specializes in certain action types

**Why it's useful:**
- Identifies which action dimensions need improvement
- Could guide targeted model training or architecture changes

---

## Real Example from Bridge Dataset

From your 10-episode run:

```
MSE:                 0.051977
MAE:                 0.088397
Cosine Similarity:   0.464
MSE per dim mean:    0.051977
MSE per dim std:     0.122
```

**Analysis:**

1. **MSE = 0.052**: Predictions differ by ~0.23 units on average
   - Below threshold (0.1) → considered "successful"

2. **MAE = 0.088**: Average absolute error is 0.088 units
   - √(MAE) ≈ 0.30, which is similar to √(MSE) ≈ 0.23
   - Suggests relatively consistent errors (few outliers)

3. **Cosine = 0.464**: Predictions point in roughly correct direction
   - ~62° average angle from true action
   - Room for improvement in directional alignment

4. **MSE per dim mean = 0.052**: Matches overall MSE
   - Confirms our MSE calculation is consistent

5. **MSE per dim std = 0.122**: High variation across dimensions
   - Some action dimensions predicted much better than others
   - Std is larger than mean! Indicates high variability
   - Example: gripper (binary) might be easy (MSE~0.01)
   - While rotation angles might be hard (MSE~0.15)

---

## Success Criterion

**Threshold: MSE < 0.1**

An episode is marked "successful" if `action_mse < 0.1`:
- ✅ Episode 2: MSE=0.009 (excellent!)
- ✅ Episode 10: MSE=0.027 (very good)
- ❌ Episode 7: MSE=0.111 (failed - above threshold)

This threshold means predictions must differ by less than ~0.32 units (√0.1) on average.

---

## Visualization Example

Here's what the metrics mean visually:

```
Ground Truth Action Sequence:
t=0: [0.10, 0.20, 0.15, 0.00, 0.00, 0.05, 1.0]
t=1: [0.12, 0.22, 0.14, 0.01, -0.01, 0.04, 1.0]
t=2: [0.15, 0.25, 0.13, 0.02, -0.02, 0.03, 0.0]

Predicted Actions:
t=0: [0.11, 0.19, 0.16, 0.01, 0.00, 0.04, 1.0]  ✅ Close!
t=1: [0.13, 0.21, 0.15, 0.02, -0.01, 0.05, 1.0]  ✅ Close!
t=2: [0.14, 0.24, 0.12, 0.01, -0.01, 0.02, 0.0]  ✅ Close!

Errors per timestep:
t=0: [-0.01, 0.01, -0.01, -0.01, 0.00, 0.01, 0.0]
t=1: [-0.01, 0.01, -0.01, -0.01, 0.00, -0.01, 0.0]
t=2: [ 0.01, 0.01, 0.01,  0.01, 0.01,  0.01, 0.0]

→ MSE ≈ 0.0001 (excellent prediction!)
→ MAE ≈ 0.008
→ Cosine ≈ 0.99 (nearly perfect direction)
```

---

## Summary Table

| Metric | Range | Better | Units | What it measures |
|--------|-------|--------|-------|------------------|
| **MSE** | [0, ∞) | Lower | action² | Overall squared error (penalizes outliers) |
| **MAE** | [0, ∞) | Lower | action | Overall absolute error (interpretable) |
| **Cosine Sim** | [-1, +1] | Higher | unitless | Direction alignment (ignores magnitude) |
| **MSE per dim (mean)** | [0, ∞) | Lower | action² | Average error per dimension |
| **MSE per dim (std)** | [0, ∞) | Lower | action² | Consistency across dimensions |

All metrics work together to give a complete picture of prediction quality!
