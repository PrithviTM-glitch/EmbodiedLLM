import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, levene

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# MetaWorld Data - Using only first 2 runs from each trial (like LIBERO)
baseline_data = np.array([
    [0.791, 0.799],
    [0.788, 0.788],
    [0.778, 0.787],
    [0.767, 0.755],
    [0.777, 0.800]
])

ablation_data = np.array([
    [0.801, 0.775],
    [0.798, 0.775],
    [0.748, 0.756],
    [0.795, 0.806],
    [0.794, 0.775]
])

# Flatten to get all individual runs
baseline = baseline_data.flatten()
ablation = ablation_data.flatten()

# Calculate trial averages
baseline_trial_means = baseline_data.mean(axis=1)
ablation_trial_means = ablation_data.mean(axis=1)

# Create comprehensive DataFrame
df_runs = pd.DataFrame({
    'Run': list(range(1, 11)) * 2,
    'Success_Rate': np.concatenate([baseline, ablation]),
    'Condition': ['Baseline (with state encoder)'] * 10 + ['Ablation (without state encoder)'] * 10
})
df_runs['Success_Rate_Pct'] = df_runs['Success_Rate'] * 100

print("=" * 90)
print("STATISTICAL ANALYSIS: State Encoder Ablation Study - METAWORLD BENCHMARK")
print("(Using 10 runs per condition - 2 runs × 5 trials - to match LIBERO methodology)")
print("=" * 90)

# Descriptive Statistics
print("\n1. DESCRIPTIVE STATISTICS")
print("-" * 90)
print(f"Baseline (with state encoder):")
print(f"  Mean: {np.mean(baseline):.4f} ({np.mean(baseline)*100:.2f}%)")
print(f"  Std Dev: {np.std(baseline, ddof=1):.4f} ({np.std(baseline, ddof=1)*100:.2f}%)")
print(f"  Min: {np.min(baseline):.4f} ({np.min(baseline)*100:.2f}%)")
print(f"  Max: {np.max(baseline):.4f} ({np.max(baseline)*100:.2f}%)")
print(f"  Median: {np.median(baseline):.4f} ({np.median(baseline)*100:.2f}%)")
print(f"  Range: {(np.max(baseline) - np.min(baseline))*100:.2f}%")

print(f"\nAblation (without state encoder):")
print(f"  Mean: {np.mean(ablation):.4f} ({np.mean(ablation)*100:.2f}%)")
print(f"  Std Dev: {np.std(ablation, ddof=1):.4f} ({np.std(ablation, ddof=1)*100:.2f}%)")
print(f"  Min: {np.min(ablation):.4f} ({np.min(ablation)*100:.2f}%)")
print(f"  Max: {np.max(ablation):.4f} ({np.max(ablation)*100:.2f}%)")
print(f"  Median: {np.median(ablation):.4f} ({np.median(ablation)*100:.2f}%)")
print(f"  Range: {(np.max(ablation) - np.min(ablation))*100:.2f}%")

print(f"\nDifference:")
mean_diff = np.mean(baseline) - np.mean(ablation)
print(f"  Mean difference: {mean_diff:.4f} ({mean_diff*100:.2f}%)")
print(f"  Relative change: {(mean_diff/np.mean(baseline)*100):.2f}%")

print("\n2. VARIANCE ANALYSIS")
print("-" * 90)
print(f"Baseline Std Dev: {np.std(baseline, ddof=1):.4f} ({np.std(baseline, ddof=1)*100:.2f}%)")
print(f"Ablation Std Dev: {np.std(ablation, ddof=1):.4f} ({np.std(ablation, ddof=1)*100:.2f}%)")
variance_ratio = np.var(ablation, ddof=1)/np.var(baseline, ddof=1)
print(f"Variance Ratio (Ablation/Baseline): {variance_ratio:.2f}")
std_ratio = np.std(ablation, ddof=1)/np.std(baseline, ddof=1)
print(f"Std Dev Ratio (Ablation/Baseline): {std_ratio:.2f}")
print(f"\n⚠️  OBSERVATION: Ablation has {((std_ratio - 1)*100):.1f}% MORE standard deviation")
print(f"    This suggests removing the state encoder increases performance instability!")

# Levene's test for equality of variances
levene_stat, levene_p = levene(baseline, ablation)
print(f"\nLevene's Test for Equal Variances:")
print(f"  Statistic: {levene_stat:.4f}")
print(f"  p-value: {levene_p:.4f}")
print(f"  Result: Variances are {'DIFFERENT' if levene_p < 0.05 else 'NOT significantly different'} (α=0.05)")

# Statistical Tests
print("\n3. STATISTICAL TESTS (MEAN COMPARISON)")
print("-" * 90)

# Two-sample t-test
t_stat, p_value_ttest = ttest_ind(baseline, ablation)
print(f"Independent samples t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.4f}")
print(f"  Result: {'NOT significant' if p_value_ttest >= 0.05 else 'SIGNIFICANT'} at α=0.05")

# Welch's t-test (doesn't assume equal variances)
t_stat_welch, p_value_welch = ttest_ind(baseline, ablation, equal_var=False)
print(f"\nWelch's t-test (unequal variances):")
print(f"  t-statistic: {t_stat_welch:.4f}")
print(f"  p-value: {p_value_welch:.4f}")
print(f"  Result: {'NOT significant' if p_value_welch >= 0.05 else 'SIGNIFICANT'} at α=0.05")

# Mann-Whitney U test
u_stat, p_value_mw = mannwhitneyu(baseline, ablation, alternative='two-sided')
print(f"\nMann-Whitney U test (non-parametric):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_value_mw:.4f}")
print(f"  Result: {'NOT significant' if p_value_mw >= 0.05 else 'SIGNIFICANT'} at α=0.05")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(baseline)-1)*np.var(baseline, ddof=1) + 
                       (len(ablation)-1)*np.var(ablation, ddof=1)) / 
                      (len(baseline) + len(ablation) - 2))
cohens_d = (np.mean(baseline) - np.mean(ablation)) / pooled_std
print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"
print(f"  Interpretation: {effect_interpretation} effect")

# Confidence Intervals
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - margin, mean + margin

ci_baseline = confidence_interval(baseline)
ci_ablation = confidence_interval(ablation)

print(f"\n95% Confidence Intervals:")
print(f"  Baseline: [{ci_baseline[0]:.4f}, {ci_baseline[1]:.4f}] or [{ci_baseline[0]*100:.2f}%, {ci_baseline[1]*100:.2f}%]")
print(f"  Ablation: [{ci_ablation[0]:.4f}, {ci_ablation[1]:.4f}] or [{ci_ablation[0]*100:.2f}%, {ci_ablation[1]*100:.2f}%]")
print(f"  CI Overlap: {'YES (not significant)' if ci_baseline[0] <= ci_ablation[1] and ci_ablation[0] <= ci_baseline[1] else 'NO (likely significant)'}")

# Normality tests
print("\n4. NORMALITY TESTS (Shapiro-Wilk)")
print("-" * 90)
stat_baseline, p_baseline = stats.shapiro(baseline)
stat_ablation, p_ablation = stats.shapiro(ablation)
print(f"Baseline: W={stat_baseline:.4f}, p={p_baseline:.4f} {'(Normal)' if p_baseline > 0.05 else '(Not normal)'}")
print(f"Ablation: W={stat_ablation:.4f}, p={p_ablation:.4f} {'(Normal)' if p_ablation > 0.05 else '(Not normal)'}")

print("\n" + "=" * 90)
print("KEY FINDINGS")
print("=" * 90)
print(f"1. MEAN PERFORMANCE: Baseline slightly higher ({np.mean(baseline)*100:.2f}% vs {np.mean(ablation)*100:.2f}%)")
print(f"   Difference: {mean_diff*100:.2f}% (p={p_value_ttest:.3f})")
print(f"2. VARIANCE: Ablation has {((std_ratio - 1)*100):.0f}% higher standard deviation")
print(f"3. STATISTICAL SIGNIFICANCE: {'NO' if p_value_ttest >= 0.05 else 'YES'} difference in means (p={p_value_ttest:.3f})")
print(f"4. PRACTICAL INTERPRETATION:")
if p_value_ttest >= 0.05:
    print(f"   → State encoder doesn't significantly improve MEAN performance")
else:
    print(f"   → State encoder provides small but significant performance improvement")
print(f"   → State encoder MAY improve STABILITY (reduces variance by {((1 - 1/std_ratio)*100):.0f}%)")
print(f"   → Removing it leads to more variable results")

print("\n" + "=" * 90)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('State Encoder Ablation Study - MetaWorld Benchmark Analysis\n(10 runs per condition, matching LIBERO methodology)', 
             fontsize=16, fontweight='bold', y=1.00)

# 1. Box plot with individual points
ax1 = axes[0, 0]
sns.boxplot(data=df_runs, x='Condition', y='Success_Rate_Pct', ax=ax1, 
            palette=['#3498db', '#e74c3c'], width=0.5)
sns.stripplot(data=df_runs, x='Condition', y='Success_Rate_Pct', ax=ax1, 
              color='black', alpha=0.5, size=6)
ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax1.set_xlabel('')
ax1.set_title('Distribution Comparison (Box Plot)', fontsize=12, fontweight='bold')
ax1.set_xticklabels(['Baseline\n(with encoder)', 'Ablation\n(without encoder)'])
ax1.grid(axis='y', alpha=0.3)

# 2. Violin plot
ax2 = axes[0, 1]
sns.violinplot(data=df_runs, x='Condition', y='Success_Rate_Pct', ax=ax2,
               palette=['#3498db', '#e74c3c'], inner='box')
ax2.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('')
ax2.set_title('Distribution Comparison (Violin Plot)', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['Baseline\n(with encoder)', 'Ablation\n(without encoder)'])
ax2.grid(axis='y', alpha=0.3)

# 3. Histogram with KDE
ax3 = axes[0, 2]
ax3.hist(baseline * 100, bins=7, alpha=0.6, label='Baseline', color='#3498db', edgecolor='black')
ax3.hist(ablation * 100, bins=7, alpha=0.6, label='Ablation', color='#e74c3c', edgecolor='black')
ax3.axvline(np.mean(baseline) * 100, color='#3498db', linestyle='--', linewidth=2, label=f'Baseline mean: {np.mean(baseline)*100:.2f}%')
ax3.axvline(np.mean(ablation) * 100, color='#e74c3c', linestyle='--', linewidth=2, label=f'Ablation mean: {np.mean(ablation)*100:.2f}%')
ax3.set_xlabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Distribution Histogram', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# 4. Run-by-run comparison
ax4 = axes[1, 0]
runs = np.arange(1, 11)
ax4.plot(runs, baseline * 100, marker='o', linewidth=2, markersize=8, 
         label='Baseline', color='#3498db')
ax4.plot(runs, ablation * 100, marker='s', linewidth=2, markersize=8, 
         label='Ablation', color='#e74c3c')
ax4.axhline(np.mean(baseline) * 100, color='#3498db', linestyle='--', alpha=0.5)
ax4.axhline(np.mean(ablation) * 100, color='#e74c3c', linestyle='--', alpha=0.5)
ax4.fill_between(runs, 
                 (np.mean(baseline) - np.std(baseline, ddof=1)) * 100,
                 (np.mean(baseline) + np.std(baseline, ddof=1)) * 100,
                 alpha=0.2, color='#3498db')
ax4.fill_between(runs, 
                 (np.mean(ablation) - np.std(ablation, ddof=1)) * 100,
                 (np.mean(ablation) + np.std(ablation, ddof=1)) * 100,
                 alpha=0.2, color='#e74c3c')
ax4.set_xlabel('Run Number', fontsize=11, fontweight='bold')
ax4.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('Run-by-Run Performance', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
ax4.set_xticks(runs)

# 5. Mean with confidence intervals
ax5 = axes[1, 1]
conditions = ['Baseline\n(with encoder)', 'Ablation\n(without encoder)']
means = [np.mean(baseline) * 100, np.mean(ablation) * 100]
cis = [(ci_baseline[1] - ci_baseline[0]) * 100 / 2, 
       (ci_ablation[1] - ci_ablation[0]) * 100 / 2]
colors = ['#3498db', '#e74c3c']

bars = ax5.bar(conditions, means, yerr=cis, capsize=10, alpha=0.7, 
               color=colors, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax5.set_title('Mean Success Rate with 95% CI', fontsize=12, fontweight='bold')
ax5.set_ylim([74, 82])
ax5.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + ci + 0.3,
             f'{mean:.2f}%\n±{ci:.2f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 6. Statistical summary text
ax6 = axes[1, 2]
ax6.axis('off')
summary_text = f"""
STATISTICAL SUMMARY

Baseline (with state encoder):
  Mean: {np.mean(baseline)*100:.2f}%
  Std Dev: {np.std(baseline, ddof=1)*100:.2f}%
  95% CI: [{ci_baseline[0]*100:.2f}%, {ci_baseline[1]*100:.2f}%]

Ablation (without state encoder):
  Mean: {np.mean(ablation)*100:.2f}%
  Std Dev: {np.std(ablation, ddof=1)*100:.2f}%
  95% CI: [{ci_ablation[0]*100:.2f}%, {ci_ablation[1]*100:.2f}%]

Statistical Test Results:
  t-statistic: {t_stat:.4f}
  p-value: {p_value_ttest:.4f}
  Cohen's d: {cohens_d:.4f} ({effect_interpretation})
  
  Significant? {'NO' if p_value_ttest >= 0.05 else 'YES'} (α=0.05)

Mean Difference: {mean_diff*100:.2f}%
Std Dev Increase: {((std_ratio - 1)*100):.0f}%

Conclusion: The state encoder has
{'NO statistically significant' if p_value_ttest >= 0.05 else 'a statistically significant'}
effect on mean performance, but MAY
reduce performance variance.
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
         fontsize=9.5, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
         family='monospace')

plt.tight_layout()
plt.savefig('metaworld_state_encoder_ablation_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'metaworld_state_encoder_ablation_analysis.png'")
plt.show()

# Summary table
summary_df = pd.DataFrame({
    'Condition': ['Baseline (with encoder)', 'Ablation (without encoder)'],
    'Mean': [np.mean(baseline), np.mean(ablation)],
    'Std Dev': [np.std(baseline, ddof=1), np.std(ablation, ddof=1)],
    'Min': [np.min(baseline), np.min(ablation)],
    'Max': [np.max(baseline), np.max(ablation)],
    'Median': [np.median(baseline), np.median(ablation)],
    'CI_Lower': [ci_baseline[0], ci_ablation[0]],
    'CI_Upper': [ci_baseline[1], ci_ablation[1]]
})

print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)
print(summary_df.to_string(index=False))
print("\n")

# Export data
df_runs.to_csv('metaworld_all_runs.csv', index=False)
summary_df.to_csv('metaworld_summary.csv', index=False)
print("✓ Data exported to 'metaworld_all_runs.csv' and 'metaworld_summary.csv'\n")