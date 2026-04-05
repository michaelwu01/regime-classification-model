"""
Step 2b: Refined threshold sweep with tighter CHOP thresholds.

Based on initial sweep, we need MUCH tighter CHOP thresholds to avoid
over-labeling everything as CHOP.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

# Load labeling features
print("="*80)
print("STEP 2B: REFINED THRESHOLD SWEEP")
print("="*80)
print()

features_path = "output/threshold_analysis/labeling_features.csv"
print(f"Loading labeling features from {features_path}...")
df = pd.read_csv(features_path, index_col=0, parse_dates=True)
print(f"Loaded {len(df):,} rows")
print()

# Refined grids based on Step 1 statistics and initial sweep results
# We need TIGHTER CHOP thresholds to get 5-15% instead of 28-73%

# From Step 1:
# CHOP: Eff mean=0.11, p10=0.01, p25=0.03, p50=0.07, p75=0.15, p90=0.25
# CHOP: |R_ATR| mean=1.08, p10=0.12, p25=0.32, p50=0.74, p75=1.53, p90=2.52

# To get ~10% CHOP, we need to be stricter - use values around p10-p25 percentiles
eff_chop_max_grid = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
abs_return_atr_chop_max_grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# TREND thresholds - keep reasonable range
eff_trend_min_grid = [0.25, 0.30, 0.35, 0.40]
abs_return_atr_trend_min_grid = [3.0, 3.5, 4.0, 4.5, 5.0]

print("Refined threshold grids:")
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
    """Label regimes based on thresholds."""
    labels = np.full(len(eff), 'TRANSITION')

    # TREND conditions
    trend_mask = (eff >= eff_trend_min) & (r_atr_abs >= r_atr_trend_min)
    labels[trend_mask] = 'TREND'

    # CHOP conditions (but don't override TREND)
    chop_mask = (eff <= eff_chop_max) & (r_atr_abs <= r_atr_chop_max) & (~trend_mask)
    labels[chop_mask] = 'CHOP'

    return labels

# Sweep
print("Sweeping thresholds...")
results = []

for eff_t_min, r_atr_t_min, eff_c_max, r_atr_c_max in product(
    eff_trend_min_grid,
    abs_return_atr_trend_min_grid,
    eff_chop_max_grid,
    abs_return_atr_chop_max_grid
):
    labels = label_regimes(
        df['efficiency_ratio'].values,
        df['r_atr_abs'].values,
        eff_t_min, r_atr_t_min,
        eff_c_max, r_atr_c_max
    )

    trend_count = (labels == 'TREND').sum()
    chop_count = (labels == 'CHOP').sum()
    transition_count = (labels == 'TRANSITION').sum()

    trend_pct = 100 * trend_count / len(labels)
    chop_pct = 100 * chop_count / len(labels)
    transition_pct = 100 * transition_count / len(labels)

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

# Save
output_dir = Path("output/threshold_analysis")
results_df.to_csv(output_dir / "step2b_refined_sweep_results.csv", index=False)
print(f"Saved results to {output_dir / 'step2b_refined_sweep_results.csv'}")
print()

# Filter promising candidates
print("="*80)
print("FILTERING PROMISING CANDIDATES")
print("="*80)
print()

# Target: CHOP 5-15%, TREND 20-35%
chop_min, chop_max = 5.0, 15.0
trend_min, trend_max = 20.0, 35.0

filtered = results_df[
    (results_df['chop_pct'] >= chop_min) &
    (results_df['chop_pct'] <= chop_max) &
    (results_df['trend_pct'] >= trend_min) &
    (results_df['trend_pct'] <= trend_max)
]

print(f"Target: CHOP {chop_min}-{chop_max}%, TREND {trend_min}-{trend_max}%")
print(f"Candidates found: {len(filtered)} / {len(results_df)}")
print()

if len(filtered) == 0:
    print("Relaxing to: CHOP 3-20%, TREND 15-40%...")
    chop_min, chop_max = 3.0, 20.0
    trend_min, trend_max = 15.0, 40.0

    filtered = results_df[
        (results_df['chop_pct'] >= chop_min) &
        (results_df['chop_pct'] <= chop_max) &
        (results_df['trend_pct'] >= trend_min) &
        (results_df['trend_pct'] <= trend_max)
    ]
    print(f"Candidates found (relaxed): {len(filtered)} / {len(results_df)}")
    print()

# Sort by a balance score: prioritize CHOP in target range
# Score = CHOP_pct (higher is better, as long as within range)
filtered_sorted = filtered.sort_values('chop_pct', ascending=False).reset_index(drop=True)

# Save
filtered_sorted.to_csv(output_dir / "step2b_promising_thresholds.csv", index=False)
print(f"Saved to {output_dir / 'step2b_promising_thresholds.csv'}")
print()

# Display top candidates
print("="*80)
print("TOP 20 PROMISING THRESHOLD COMBINATIONS")
print("="*80)
print()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.precision', 2)

if len(filtered_sorted) > 0:
    top_20 = filtered_sorted.head(20)
    print(top_20[['eff_trend_min', 'r_atr_trend_min', 'eff_chop_max', 'r_atr_chop_max',
                  'trend_pct', 'chop_pct', 'transition_pct']].to_string(index=True))
    print()

    print("="*80)
    print("RECOMMENDED THRESHOLDS FOR STEP 3 (Top 5)")
    print("="*80)
    print()

    for i, row in filtered_sorted.head(5).iterrows():
        print(f"{i+1}. Thresholds:")
        print(f"   - TREND: eff >= {row['eff_trend_min']:.2f} AND |R_ATR| >= {row['r_atr_trend_min']:.1f}")
        print(f"   - CHOP:  eff <= {row['eff_chop_max']:.2f} AND |R_ATR| <= {row['r_atr_chop_max']:.1f}")
        print(f"   Distribution: TREND={row['trend_pct']:.1f}%, CHOP={row['chop_pct']:.1f}%, TRANSITION={row['transition_pct']:.1f}%")
        print()
else:
    print("No candidates found. Distribution summary:")
    print(f"CHOP %: min={results_df['chop_pct'].min():.2f}, max={results_df['chop_pct'].max():.2f}, mean={results_df['chop_pct'].mean():.2f}")
    print(f"TREND %: min={results_df['trend_pct'].min():.2f}, max={results_df['trend_pct'].max():.2f}, mean={results_df['trend_pct'].mean():.2f}")

print("="*80)
print("STEP 2B COMPLETE")
print("="*80)
print()
print(f"Outputs saved to: {output_dir}/")

if len(filtered_sorted) > 0:
    print(f"Next: Run step3_evaluate_thresholds.py to train models on top {min(5, len(filtered_sorted))} candidates")
else:
    print("⚠️  Consider further refining threshold ranges or using alternative labeling methods")
