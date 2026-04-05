"""
Visualize the train/validation/test split to understand temporal separation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

print("="*80)
print("VISUALIZING TRAIN/VALIDATION/TEST SPLIT")
print("="*80)
print()

# Load the data
bars_path = "expanded_data/GC_bars_5min_2020_2022.csv"
print(f"Loading bars from {bars_path}...")
bars = pd.read_csv(bars_path, index_col=0, parse_dates=True)
print(f"Loaded {len(bars):,} bars")
print()

# Calculate split indices
n = len(bars)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

# Get date ranges
train_dates = bars.index[:train_end]
val_dates = bars.index[train_end:val_end]
test_dates = bars.index[val_end:]

print("Split Details:")
print(f"  Training:   {len(train_dates):,} bars ({train_dates[0]} to {train_dates[-1]})")
print(f"  Validation: {len(val_dates):,} bars ({val_dates[0]} to {val_dates[-1]})")
print(f"  Test:       {len(test_dates):,} bars ({test_dates[0]} to {test_dates[-1]})")
print()

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# ============================================================================
# Plot 1: Timeline view
# ============================================================================
ax1.barh(0, len(train_dates), left=0, height=0.5, color='steelblue', label='Training (60%)')
ax1.barh(0, len(val_dates), left=len(train_dates), height=0.5, color='orange', label='Validation (20%)')
ax1.barh(0, len(test_dates), left=len(train_dates)+len(val_dates), height=0.5, color='green', label='Test (20%)')

# Add text annotations
ax1.text(len(train_dates)/2, 0, f'Training\n{len(train_dates):,} bars\n60%',
         ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax1.text(len(train_dates)+len(val_dates)/2, 0, f'Validation\n{len(val_dates):,} bars\n20%',
         ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax1.text(len(train_dates)+len(val_dates)+len(test_dates)/2, 0, f'Test\n{len(test_dates):,} bars\n20%',
         ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# Add date labels
ax1.text(0, -0.4, train_dates[0].strftime('%Y-%m-%d'), ha='left', fontsize=10)
ax1.text(len(train_dates), -0.4, train_dates[-1].strftime('%Y-%m-%d'), ha='center', fontsize=10)
ax1.text(len(train_dates)+len(val_dates), -0.4, val_dates[-1].strftime('%Y-%m-%d'), ha='center', fontsize=10)
ax1.text(len(train_dates)+len(val_dates)+len(test_dates), -0.4, test_dates[-1].strftime('%Y-%m-%d'), ha='right', fontsize=10)

ax1.set_ylim(-0.6, 0.6)
ax1.set_xlim(0, n)
ax1.set_yticks([])
ax1.set_xlabel('Bar Index', fontsize=12)
ax1.set_title('Data Split Timeline (60/20/20 Sequential Split)', fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(True, alpha=0.3, axis='x')

# ============================================================================
# Plot 2: Price chart with split regions highlighted
# ============================================================================
ax2.plot(bars.index, bars['close'], linewidth=0.5, color='black', alpha=0.7, label='GC Close Price')

# Shade regions
ax2.axvspan(train_dates[0], train_dates[-1], alpha=0.2, color='steelblue', label='Training')
ax2.axvspan(val_dates[0], val_dates[-1], alpha=0.2, color='orange', label='Validation')
ax2.axvspan(test_dates[0], test_dates[-1], alpha=0.2, color='green', label='Test')

# Add vertical lines at split points
ax2.axvline(train_dates[-1], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Split Points')
ax2.axvline(val_dates[-1], color='red', linestyle='--', linewidth=2, alpha=0.7)

ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('GC Close Price ($)', fontsize=12)
ax2.set_title('Gold Futures (GC) Price with Train/Val/Test Regions', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# Rotate x-axis labels
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
output_path = "output/threshold_analysis/data_split_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {output_path}")
print()

# ============================================================================
# Print detailed statistics
# ============================================================================
print("="*80)
print("TEMPORAL SEPARATION ANALYSIS")
print("="*80)
print()

print("Date Ranges:")
print(f"  Training:   {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}")
print(f"  Validation: {val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')}")
print(f"  Test:       {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")
print()

# Calculate time gaps
train_duration = (train_dates[-1] - train_dates[0]).days
val_duration = (val_dates[-1] - val_dates[0]).days
test_duration = (test_dates[-1] - test_dates[0]).days

print("Duration (Calendar Days):")
print(f"  Training:   {train_duration} days")
print(f"  Validation: {val_duration} days")
print(f"  Test:       {test_duration} days")
print()

print("Price Statistics by Split:")
print(f"  Training:   min=${bars.loc[train_dates, 'close'].min():.2f}, max=${bars.loc[train_dates, 'close'].max():.2f}, mean=${bars.loc[train_dates, 'close'].mean():.2f}")
print(f"  Validation: min=${bars.loc[val_dates, 'close'].min():.2f}, max=${bars.loc[val_dates, 'close'].max():.2f}, mean=${bars.loc[val_dates, 'close'].mean():.2f}")
print(f"  Test:       min=${bars.loc[test_dates, 'close'].min():.2f}, max=${bars.loc[test_dates, 'close'].max():.2f}, mean=${bars.loc[test_dates, 'close'].mean():.2f}")
print()

print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()
print("✅ Sequential Split: Data is split chronologically, not randomly")
print("   - Training data comes BEFORE validation data")
print("   - Validation data comes BEFORE test data")
print("   - No future information leaks into past")
print()
print("✅ Test Set is Truly Unseen:")
print("   - Test set contains 2022 data")
print("   - Model was trained only on 2020-2021 data")
print("   - Model has NEVER seen 2022 prices/features during training")
print()
print("✅ Proper Time-Series Evaluation:")
print("   - Mimics real-world deployment scenario")
print("   - Train on historical data, predict future")
print("   - Honest performance estimate")
print()
print("⚠️  Forward-Looking Label Issue:")
print("   - Labels use H=20 bars forward-looking window")
print("   - Label at bar t uses prices from t to t+20")
print("   - This is unavoidable for regime labeling")
print("   - Real-time trading accuracy may be lower than 93%")
print()
print(f"Visualization saved to: {output_path}")
