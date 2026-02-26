import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Data
baseline = np.array([0.9325, 0.9350, 0.9175, 0.9375, 0.9300, 
                     0.9325, 0.9375, 0.9100, 0.9400, 0.9325])

ablation = np.array([0.9375, 0.9300, 0.9300, 0.9100, 0.9175, 
                     0.9300, 0.9350, 0.9325, 0.9300, 0.9375])

# Create DataFrame for easier plotting
df = pd.DataFrame({
    'Run': list(range(1, 11)) * 2,
    'Success_Rate': np.concatenate([baseline, ablation]),
    'Condition': ['Baseline (with state encoder)'] * 10 + ['Ablation (without state encoder)'] * 10
})

# Convert to percentage for better readability
df['Success_Rate_Pct'] = df['Success_Rate'] * 100

print("=" * 80)
print("STATISTICAL ANALYSIS: State Encoder Ablation Study")
print("=" * 80)

# Descriptive Statistics
print("\n1. DESCRIPTIVE STATISTICS")
print("-" * 80)
print(f"Baseline (with state encoder):")
print(f"  Mean: {np.mean(baseline):.4f} ({np.mean(baseline)*100:.2f}%)")
print(f"  Std Dev: {np.std(baseline, ddof=1):.4f} ({np.std(baseline, ddof=1)*100:.2f}%)")
print(f"  Min: {np.min(baseline):.4f} ({np.min(baseline)*100:.2f}%)")
print(f"  Max: {np.max(baseline):.4f} ({np.max(baseline)*100:.2f}%)")
print(f"  Median: {np.median(baseline):.4f} ({np.median(baseline)*100:.2f}%)")

print(f"\nAblation (without state encoder):")
print(f"  Mean: {np.mean(ablation):.4f} ({np.mean(ablation)*100:.2f}%)")
print(f"  Std Dev: {np.std(ablation, ddof=1):.4f} ({np.std(ablation, ddof=1)*100:.2f}%)")
print(f"  Min: {np.min(ablation):.4f} ({np.min(ablation)*100:.2f}%)")
print(f"  Max: {np.max(ablation):.4f} ({np.max(ablation)*100:.2f}%)")
print(f"  Median: {np.median(ablation):.4f} ({np.median(ablation)*100:.2f}%)")

print(f"\nDifference:")
print(f"  Mean difference: {(np.mean(baseline) - np.mean(ablation)):.4f} ({(np.mean(baseline) - np.mean(ablation))*100:.2f}%)")
print(f"  Relative change: {((np.mean(baseline) - np.mean(ablation))/np.mean(baseline)*100):.2f}%")

# Statistical Tests
print("\n2. STATISTICAL TESTS")
print("-" * 80)

# Two-sample t-test (assumes normality and equal variances)
t_stat, p_value_ttest = ttest_ind(baseline, ablation)
print(f"Independent samples t-test:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.4f}")
print(f"  Result: {'NOT significant' if p_value_ttest >= 0.05 else 'SIGNIFICANT'} at α=0.05")

# Mann-Whitney U test (non-parametric alternative)
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

# Normality tests
print("\n3. NORMALITY TESTS (Shapiro-Wilk)")
print("-" * 80)
stat_baseline, p_baseline = stats.shapiro(baseline)
stat_ablation, p_ablation = stats.shapiro(ablation)
print(f"Baseline: W={stat_baseline:.4f}, p={p_baseline:.4f} {'(Normal)' if p_baseline > 0.05 else '(Not normal)'}")
print(f"Ablation: W={stat_ablation:.4f}, p={p_ablation:.4f} {'(Normal)' if p_ablation > 0.05 else '(Not normal)'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if p_value_ttest >= 0.05:
    print("The state encoder does NOT have a statistically significant effect on")
    print("LIBERO performance. The observed difference could be due to random chance.")
    print("\nImplication: The state encoder appears to be redundant for these tasks.")
else:
    print("The state encoder HAS a statistically significant effect on LIBERO performance.")

print("\n" + "=" * 80)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('State Encoder Ablation Study - LIBERO Benchmark Analysis', 
             fontsize=16, fontweight='bold', y=1.00)

# 1. Box plot with individual points
ax1 = axes[0, 0]
sns.boxplot(data=df, x='Condition', y='Success_Rate_Pct', ax=ax1, 
            palette=['#3498db', '#e74c3c'], width=0.5)
sns.stripplot(data=df, x='Condition', y='Success_Rate_Pct', ax=ax1, 
              color='black', alpha=0.5, size=6)
ax1.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax1.set_xlabel('')
ax1.set_title('Distribution Comparison (Box Plot)', fontsize=12, fontweight='bold')
ax1.set_xticklabels(['Baseline\n(with encoder)', 'Ablation\n(without encoder)'])
ax1.grid(axis='y', alpha=0.3)

# 2. Violin plot
ax2 = axes[0, 1]
sns.violinplot(data=df, x='Condition', y='Success_Rate_Pct', ax=ax2,
               palette=['#3498db', '#e74c3c'], inner='box')
ax2.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('')
ax2.set_title('Distribution Comparison (Violin Plot)', fontsize=12, fontweight='bold')
ax2.set_xticklabels(['Baseline\n(with encoder)', 'Ablation\n(without encoder)'])
ax2.grid(axis='y', alpha=0.3)

# 3. Histogram with KDE
ax3 = axes[0, 2]
ax3.hist(baseline * 100, bins=8, alpha=0.6, label='Baseline', color='#3498db', edgecolor='black')
ax3.hist(ablation * 100, bins=8, alpha=0.6, label='Ablation', color='#e74c3c', edgecolor='black')
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
ax5.set_ylim([90, 95])
ax5.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + ci + 0.2,
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

Mean Difference: {(np.mean(baseline) - np.mean(ablation))*100:.2f}%

Conclusion: The state encoder has
{'NO statistically significant' if p_value_ttest >= 0.05 else 'a statistically significant'}
effect on LIBERO performance.
"""

ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, 
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
         family='monospace')

plt.tight_layout()
plt.savefig('state_encoder_ablation_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'state_encoder_ablation_analysis.png'")
plt.show()

# Additional: Create a summary DataFrame
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

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))
print("\n")