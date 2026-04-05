"""
Step 2: Sweep thresholds on a grid to find balanced class distributions.

Goal: Find threshold combinations that give:
- CHOP: 5-15% of bars
- TREND: 20-30% of bars
- TRANSITION: rest

We'll sweep thresholds and check label balance WITHOUT training models (fast).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

# Load labeling features from Step 1
print("="*80)
print("STEP 2: SWEEPING THRESHOLDS TO FIND BALANCED DISTRIBUTIONS")
print("="*80)
print()

features_path = "output/threshold_analysis/labeling_features.csv"
print(f"Loading labeling features from {features_path}...")
df = pd.read_csv(features_path, index_col=0, parse_dates=True)
print(f"Loaded {len(df):,} rows")
print()

# Define threshold grids to sweep
# Current thresholds (from config.py):
#   eff_trend_min = 0.35
#   abs_return_atr_trend_min = 5.0
#   eff_chop_max = 0.10
#   abs_return_atr_chop_max = 3.0

# From Step 1 statistics:
# CHOP: Eff mean=0.11, p90=0.25; |R_ATR| mean=1.08, p90=2.52
# TREND: Eff mean=0.27, p10=0.05; |R_ATR| mean=2.71, p10=0.50
# TRANSITION: Eff mean=0.20; |R_ATR| mean=1.91

# Let's sweep a range around these values
eff_trend_min_grid = [0.25, 0.30, 0.35, 0.40, 0.45]
abs_return_atr_trend_min_grid = [3.0, 4.0, 5.0, 6.0]

eff_chop_max_grid = [0.10, 0.15, 0.20, 0.25, 0.30]
abs_return_atr_chop_max_grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

print("Threshold grids:")
print(f"  eff_trend_min: {eff_trend_min_grid}")
print(f"  abs_return_atr_trend_min: {abs_return_atr_trend_min_grid}")
print(f"  eff_chop_max: {eff_chop_max_grid}")
print(f"  abs_return_atr_chop_max: {abs_return_atr_chop_max_grid}")
print()

total_combinations = (len(eff_trend_min_grid) * len(abs_return_atr_trend_min_grid) *
                     len(eff_chop_max_grid) * len(abs_return_atr_chop_max_grid))
print(f"Total combinations to test: {total_combinations:,}")
print()

# Labeling function
def label_regimes(eff, r_atr_abs, eff_trend_min, r_atr_trend_min, eff_chop_max, r_atr_chop_max):
    """
    Label regimes based on thresholds.

    TREND: eff >= eff_trend_min AND r_atr_abs >= r_atr_trend_min
    CHOP: eff <= eff_chop_max AND r_atr_abs <= r_atr_chop_max
    TRANSITION: everything else
    """
    labels = np.full(len(eff), 'TRANSITION')

    # TREND conditions
    trend_mask = (eff >= eff_trend_min) & (r_atr_abs >= r_atr_trend_min)
    labels[trend_mask] = 'TREND'

    # CHOP conditions (but don't override TREND)
    chop_mask = (eff <= eff_chop_max) & (r_atr_abs <= r_atr_chop_max) & (~trend_mask)
    labels[chop_mask] = 'CHOP'

    return labels

# Sweep thresholds
print("Sweeping thresholds...")
results = []

for eff_t_min, r_atr_t_min, eff_c_max, r_atr_c_max in product(
    eff_trend_min_grid,
    abs_return_atr_trend_min_grid,
    eff_chop_max_grid,
    abs_return_atr_chop_max_grid
):
    # Generate labels
    labels = label_regimes(
        df['efficiency_ratio'].values,
        df['r_atr_abs'].values,
        eff_t_min, r_atr_t_min,
        eff_c_max, r_atr_c_max
    )

    # Count distribution
    trend_count = (labels == 'TREND').sum()
    chop_count = (labels == 'CHOP').sum()
    transition_count = (labels == 'TRANSITION').sum()

    trend_pct = 100 * trend_count / len(labels)
    chop_pct = 100 * chop_count / len(labels)
    transition_pct = 100 * transition_count / len(labels)

    # Store result
    results.append({
        'eff_trend_min': eff_t_min,
        'r_atr_trend_min': r_atr_t_min,
        'eff_chop_max': eff_c_max,
        'r_atr_chop_max': r_atr_c_max,
        'trend_count': trend_count,
        'trend_pct': trend_pct,
        'chop_count': chop_count,
        'chop_pct': chop_pct,
        'transition_count': transition_count,
        'transition_pct': transition_pct
    })

results_df = pd.DataFrame(results)
print(f"Completed sweep: {len(results_df):,} combinations tested")
print()

# Save results
output_dir = Path("output/threshold_analysis")
results_df.to_csv(output_dir / "step2_threshold_sweep_results.csv", index=False)
print(f"Saved full results to {output_dir / 'step2_threshold_sweep_results.csv'}")
print()

# ============================================================================
# FILTER PROMISING CANDIDATES
# ============================================================================

print("="*80)
print("FILTERING PROMISING THRESHOLD COMBINATIONS")
print("="*80)
print()

# Target criteria:
# - CHOP: 5-15% (goal: make it learnable)
# - TREND: 20-35% (reasonable range)
# - TRANSITION: rest

chop_min_pct = 5.0
chop_max_pct = 15.0
trend_min_pct = 20.0
trend_max_pct = 35.0

print(f"Filter criteria:")
print(f"  CHOP: {chop_min_pct:.1f}% - {chop_max_pct:.1f}%")
print(f"  TREND: {trend_min_pct:.1f}% - {trend_max_pct:.1f}%")
print()

filtered = results_df[
    (results_df['chop_pct'] >= chop_min_pct) &
    (results_df['chop_pct'] <= chop_max_pct) &
    (results_df['trend_pct'] >= trend_min_pct) &
    (results_df['trend_pct'] <= trend_max_pct)
]

print(f"Promising candidates: {len(filtered)} / {len(results_df)}")
print()

if len(filtered) == 0:
    print("⚠️  No candidates meet the strict criteria. Let's relax the constraints...")
    # Relax constraints
    chop_min_pct = 3.0
    chop_max_pct = 20.0
    trend_min_pct = 15.0
    trend_max_pct = 40.0

    print(f"Relaxed criteria:")
    print(f"  CHOP: {chop_min_pct:.1f}% - {chop_max_pct:.1f}%")
    print(f"  TREND: {trend_min_pct:.1f}% - {trend_max_pct:.1f}%")
    print()

    filtered = results_df[
        (results_df['chop_pct'] >= chop_min_pct) &
        (results_df['chop_pct'] <= chop_max_pct) &
        (results_df['trend_pct'] >= trend_min_pct) &
        (results_df['trend_pct'] <= trend_max_pct)
    ]

    print(f"Promising candidates (relaxed): {len(filtered)} / {len(results_df)}")
    print()

# Sort by CHOP percentage (descending) to prioritize higher CHOP representation
filtered_sorted = filtered.sort_values('chop_pct', ascending=False).reset_index(drop=True)

# Save filtered results
filtered_sorted.to_csv(output_dir / "step2_promising_thresholds.csv", index=False)
print(f"Saved promising candidates to {output_dir / 'step2_promising_thresholds.csv'}")
print()

# Display top 20 candidates
print("="*80)
print("TOP 20 PROMISING THRESHOLD COMBINATIONS")
print("="*80)
print()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 2)

top_20 = filtered_sorted.head(20)
print(top_20.to_string(index=True))
print()

# ============================================================================
# VISUALIZATION
# ============================================================================

print("Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. CHOP percentage distribution
ax = axes[0, 0]
ax.hist(results_df['chop_pct'], bins=50, alpha=0.7, color='red', edgecolor='black')
ax.axvline(chop_min_pct, color='green', linestyle='--', linewidth=2, label=f'Target min: {chop_min_pct}%')
ax.axvline(chop_max_pct, color='green', linestyle='--', linewidth=2, label=f'Target max: {chop_max_pct}%')
ax.set_xlabel('CHOP Percentage', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of CHOP % Across All Threshold Combinations', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. TREND percentage distribution
ax = axes[0, 1]
ax.hist(results_df['trend_pct'], bins=50, alpha=0.7, color='green', edgecolor='black')
ax.axvline(trend_min_pct, color='red', linestyle='--', linewidth=2, label=f'Target min: {trend_min_pct}%')
ax.axvline(trend_max_pct, color='red', linestyle='--', linewidth=2, label=f'Target max: {trend_max_pct}%')
ax.set_xlabel('TREND Percentage', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of TREND % Across All Threshold Combinations', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Scatter: CHOP vs TREND percentages
ax = axes[1, 0]
scatter = ax.scatter(results_df['trend_pct'], results_df['chop_pct'],
                     c=results_df['transition_pct'], cmap='viridis', alpha=0.6, s=20)
if len(filtered_sorted) > 0:
    ax.scatter(filtered_sorted['trend_pct'], filtered_sorted['chop_pct'],
              c='red', marker='o', s=100, edgecolors='black', linewidths=2,
              label=f'Promising ({len(filtered_sorted)})', zorder=10)
ax.set_xlabel('TREND %', fontsize=12)
ax.set_ylabel('CHOP %', fontsize=12)
ax.set_title('CHOP % vs TREND % (color = TRANSITION %)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('TRANSITION %', fontsize=10)

# 4. Best candidate distributions (if any)
ax = axes[1, 1]
if len(filtered_sorted) > 0:
    best = filtered_sorted.iloc[0]
    categories = ['TREND', 'TRANSITION', 'CHOP']
    percentages = [best['trend_pct'], best['transition_pct'], best['chop_pct']]
    colors_bar = ['green', 'orange', 'red']

    bars = ax.bar(categories, percentages, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title(f'Best Candidate Distribution\n'
                 f'(eff_t_min={best["eff_trend_min"]}, r_atr_t_min={best["r_atr_trend_min"]}, '
                 f'eff_c_max={best["eff_chop_max"]}, r_atr_c_max={best["r_atr_chop_max"]})',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No promising candidates found\nwith current criteria',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.tight_layout()
plot_path = output_dir / "step2_threshold_sweep.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {plot_path}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("STEP 2 COMPLETE")
print("="*80)
print()

if len(filtered_sorted) > 0:
    print(f"✅ Found {len(filtered_sorted)} promising threshold combinations!")
    print()
    print("Recommended threshold sets for Step 3 (top 5):")
    print()
    for i, row in filtered_sorted.head(5).iterrows():
        print(f"{i+1}. eff_trend_min={row['eff_trend_min']:.2f}, "
              f"r_atr_trend_min={row['r_atr_trend_min']:.1f}, "
              f"eff_chop_max={row['eff_chop_max']:.2f}, "
              f"r_atr_chop_max={row['r_atr_chop_max']:.1f}")
        print(f"   → TREND={row['trend_pct']:.1f}%, CHOP={row['chop_pct']:.1f}%, "
              f"TRANSITION={row['transition_pct']:.1f}%")
        print()

    print(f"Next: Run step3_evaluate_thresholds.py to train models on these candidates")
else:
    print("⚠️  No candidates found. Consider:")
    print("   1. Further relaxing the target percentages")
    print("   2. Expanding the threshold grids")
    print("   3. Using a different labeling approach (e.g., clustering)")

print()
print(f"Outputs saved to: {output_dir}/")
