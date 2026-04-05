"""
Step 1: Analyze empirical distribution of labeling features.

Plot the joint distribution of Efficiency Ratio vs |R_ATR| to understand
where the data naturally clusters and identify potential threshold boundaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the preprocessed features
features_path = "output/expanded_preprocess/features.csv"
labels_path = "output/expanded_label/regime_labels.csv"

print("="*80)
print("STEP 1: ANALYZING FEATURE DISTRIBUTION FOR THRESHOLD SELECTION")
print("="*80)
print()

# Load data
print(f"Loading features from {features_path}...")
features = pd.read_csv(features_path, index_col=0, parse_dates=True)
print(f"Loaded {len(features):,} bars")

print(f"\nLoading current labels from {labels_path}...")
labels = pd.read_csv(labels_path, index_col=0, parse_dates=True)
print(f"Loaded {len(labels):,} labels")

# Merge
merged = features.join(labels[['regime_str']], how='inner')
print(f"\nMerged dataset: {len(merged):,} rows")

# We need to compute the labeling features (Efficiency Ratio and |R_ATR|)
# These are computed over a horizon H. Let's use H=20 (same as in labeler.py)

print("\nComputing labeling features (Efficiency Ratio and |R_ATR|)...")
print("Using horizon H=20 bars...")

H = 20

# Load bars for price data
bars_path = "expanded_data/GC_bars_5min_2020_2022.csv"
bars = pd.read_csv(bars_path, index_col=0, parse_dates=True)
print(f"Loaded {len(bars):,} bars from {bars_path}")

# Compute Efficiency Ratio: |close[t+H] - close[t]| / sum(|price changes|)
close_change = bars['close'].diff(H).abs()
sum_abs_changes = bars['close'].diff().abs().rolling(H).sum()
efficiency_ratio = close_change / sum_abs_changes

# Compute |R_ATR|: |close[t+H] - close[t]| / ATR
atr = merged['feat_atr_z']  # We can use the ATR from features
# Actually, we need raw ATR, not z-scored. Let's compute it fresh.
from ta.volatility import AverageTrueRange

atr_indicator = AverageTrueRange(high=bars['high'], low=bars['low'], close=bars['close'], window=14)
atr_raw = atr_indicator.average_true_range()

r_atr = close_change / atr_raw
r_atr_abs = r_atr.abs()

# Combine into DataFrame
labeling_features = pd.DataFrame({
    'efficiency_ratio': efficiency_ratio,
    'r_atr_abs': r_atr_abs,
    'regime_current': merged['regime_str']
})

# Drop NaNs
labeling_features = labeling_features.dropna()
print(f"\nLabeling features computed: {len(labeling_features):,} valid rows")

# Current label distribution
print("\nCurrent label distribution:")
for regime in ['TREND', 'TRANSITION', 'CHOP']:
    count = (labeling_features['regime_current'] == regime).sum()
    pct = 100 * count / len(labeling_features)
    print(f"  {regime:12s}: {count:8,} ({pct:5.2f}%)")

# Save labeling features for later use
output_dir = Path("output/threshold_analysis")
output_dir.mkdir(parents=True, exist_ok=True)
labeling_features.to_csv(output_dir / "labeling_features.csv")
print(f"\nSaved labeling features to {output_dir / 'labeling_features.csv'}")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Joint distribution (all data)
ax = axes[0, 0]
ax.scatter(labeling_features['efficiency_ratio'], labeling_features['r_atr_abs'],
           alpha=0.1, s=1, c='blue')
ax.set_xlabel('Efficiency Ratio', fontsize=12)
ax.set_ylabel('|R_ATR|', fontsize=12)
ax.set_title('Joint Distribution: Efficiency Ratio vs |R_ATR|\n(All Data)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add current threshold lines
# From regime_sandbox/label/config.py:
# eff_trend_min = 0.35
# abs_return_atr_trend_min = 5.0
# eff_chop_max = 0.10
# abs_return_atr_chop_max = 3.0
ax.axvline(x=0.35, color='green', linestyle='--', linewidth=2, label='TREND: eff_min=0.35', alpha=0.7)
ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='TREND: |R_ATR|_min=5.0', alpha=0.7)
ax.axvline(x=0.10, color='red', linestyle='--', linewidth=2, label='CHOP: eff_max=0.10', alpha=0.7)
ax.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='CHOP: |R_ATR|_max=3.0', alpha=0.7)
ax.legend(loc='upper right', fontsize=9)

# 2. Joint distribution colored by current regime
ax = axes[0, 1]
regime_colors = {'TREND': 'green', 'TRANSITION': 'orange', 'CHOP': 'red'}
for regime in ['TREND', 'TRANSITION', 'CHOP']:
    mask = labeling_features['regime_current'] == regime
    data = labeling_features[mask]
    ax.scatter(data['efficiency_ratio'], data['r_atr_abs'],
               alpha=0.2, s=1, c=regime_colors[regime], label=regime)
ax.set_xlabel('Efficiency Ratio', fontsize=12)
ax.set_ylabel('|R_ATR|', fontsize=12)
ax.set_title('Joint Distribution Colored by Current Regime Labels', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)

# 3. Marginal distribution: Efficiency Ratio
ax = axes[1, 0]
for regime in ['TREND', 'TRANSITION', 'CHOP']:
    mask = labeling_features['regime_current'] == regime
    data = labeling_features[mask]['efficiency_ratio']
    ax.hist(data, bins=100, alpha=0.5, label=regime, color=regime_colors[regime], density=True)
ax.axvline(x=0.35, color='green', linestyle='--', linewidth=2, label='TREND threshold', alpha=0.7)
ax.axvline(x=0.10, color='red', linestyle='--', linewidth=2, label='CHOP threshold', alpha=0.7)
ax.set_xlabel('Efficiency Ratio', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Marginal Distribution: Efficiency Ratio', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

# 4. Marginal distribution: |R_ATR|
ax = axes[1, 1]
for regime in ['TREND', 'TRANSITION', 'CHOP']:
    mask = labeling_features['regime_current'] == regime
    data = labeling_features[mask]['r_atr_abs']
    # Clip to reasonable range for visualization
    data_clipped = np.clip(data, 0, 15)
    ax.hist(data_clipped, bins=100, alpha=0.5, label=regime, color=regime_colors[regime], density=True)
ax.axvline(x=5.0, color='green', linestyle='--', linewidth=2, label='TREND threshold', alpha=0.7)
ax.axvline(x=3.0, color='red', linestyle='--', linewidth=2, label='CHOP threshold', alpha=0.7)
ax.set_xlabel('|R_ATR|', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Marginal Distribution: |R_ATR| (clipped at 15 for viz)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 15)

plt.tight_layout()
plot_path = output_dir / "step1_feature_distribution.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Saved plot to {plot_path}")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n" + "="*80)
print("FEATURE STATISTICS BY CURRENT REGIME")
print("="*80)

for regime in ['TREND', 'TRANSITION', 'CHOP']:
    mask = labeling_features['regime_current'] == regime
    data = labeling_features[mask]

    print(f"\n{regime}:")
    print(f"  Count: {len(data):,} ({100*len(data)/len(labeling_features):.2f}%)")
    print(f"  Efficiency Ratio: mean={data['efficiency_ratio'].mean():.4f}, "
          f"median={data['efficiency_ratio'].median():.4f}, "
          f"std={data['efficiency_ratio'].std():.4f}")
    print(f"  |R_ATR|: mean={data['r_atr_abs'].mean():.4f}, "
          f"median={data['r_atr_abs'].median():.4f}, "
          f"std={data['r_atr_abs'].std():.4f}")
    print(f"  Percentiles (Eff): p10={data['efficiency_ratio'].quantile(0.10):.4f}, "
          f"p25={data['efficiency_ratio'].quantile(0.25):.4f}, "
          f"p75={data['efficiency_ratio'].quantile(0.75):.4f}, "
          f"p90={data['efficiency_ratio'].quantile(0.90):.4f}")
    print(f"  Percentiles (|R_ATR|): p10={data['r_atr_abs'].quantile(0.10):.4f}, "
          f"p25={data['r_atr_abs'].quantile(0.25):.4f}, "
          f"p75={data['r_atr_abs'].quantile(0.75):.4f}, "
          f"p90={data['r_atr_abs'].quantile(0.90):.4f}")

# Overall statistics
print(f"\n{'OVERALL':}")
print(f"  Count: {len(labeling_features):,}")
print(f"  Efficiency Ratio: min={labeling_features['efficiency_ratio'].min():.4f}, "
      f"max={labeling_features['efficiency_ratio'].max():.4f}")
print(f"  |R_ATR|: min={labeling_features['r_atr_abs'].min():.4f}, "
      f"max={labeling_features['r_atr_abs'].max():.4f}")

print("\n" + "="*80)
print("STEP 1 COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {output_dir}/")
print("Next: Run step2_sweep_thresholds.py to find optimal threshold combinations")
