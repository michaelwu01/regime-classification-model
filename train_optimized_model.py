"""
Train final production model with optimized thresholds.

Uses the optimized thresholds from threshold optimization:
- TREND: eff >= 0.30, |R_ATR| >= 3.0
- CHOP: eff <= 0.07, |R_ATR| <= 0.5

Expected performance:
- Macro F1: ~0.93
- TREND Recall: ~96%
- CHOP Recall: ~98%
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score
import pickle

print("="*80)
print("TRAINING FINAL MODEL WITH OPTIMIZED THRESHOLDS")
print("="*80)
print()

# Load data
bars_path = "expanded_data/GC_bars_5min_2020_2022.csv"
features_path = "output/expanded_preprocess/features.csv"

print(f"Loading bars from {bars_path}...")
bars = pd.read_csv(bars_path, index_col=0, parse_dates=True)
print(f"Loaded {len(bars):,} bars")

print(f"Loading features from {features_path}...")
features = pd.read_csv(features_path, index_col=0, parse_dates=True)
print(f"Loaded {len(features):,} feature rows")
print()

# Compute labeling features
print("Computing labeling features (Efficiency Ratio and |R_ATR|)...")
H = 20  # Horizon

close_change = bars['close'].diff(H).abs()
sum_abs_changes = bars['close'].diff().abs().rolling(H).sum()
efficiency_ratio = close_change / sum_abs_changes

atr_indicator = AverageTrueRange(high=bars['high'], low=bars['low'], close=bars['close'], window=14)
atr_raw = atr_indicator.average_true_range()
r_atr_abs = (close_change / atr_raw).abs()

print("Done")
print()

# Apply optimized thresholds
print("Applying optimized thresholds:")
print("  TREND: eff >= 0.30 AND |R_ATR| >= 3.0")
print("  CHOP: eff <= 0.07 AND |R_ATR| <= 0.5")
print()

# Label regimes
labels = np.full(len(bars), 'TRANSITION')

trend_mask = (efficiency_ratio >= 0.30) & (r_atr_abs >= 3.0)
labels[trend_mask] = 'TREND'

chop_mask = (efficiency_ratio <= 0.07) & (r_atr_abs <= 0.5) & (~trend_mask)
labels[chop_mask] = 'CHOP'

# Merge with features
df_merged = features.copy()
df_merged['regime'] = labels
df_merged = df_merged.dropna()

print(f"Dataset: {len(df_merged):,} samples after dropping NaN")
print()

# Label distribution
print("Label Distribution:")
for regime in ['TREND', 'TRANSITION', 'CHOP']:
    count = (df_merged['regime'] == regime).sum()
    pct = 100 * count / len(df_merged)
    print(f"  {regime:12s}: {count:8,} ({pct:5.2f}%)")
print()

# Split data (60/20/20)
n = len(df_merged)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

train_df = df_merged.iloc[:train_end]
val_df = df_merged.iloc[train_end:val_end]
test_df = df_merged.iloc[val_end:]

print(f"Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
print()

# Prepare feature matrices
feature_cols = [c for c in df_merged.columns if c.startswith('feat_')]
X_train = train_df[feature_cols].values
y_train = train_df['regime'].values
X_val = val_df[feature_cols].values
y_val = val_df['regime'].values
X_test = test_df[feature_cols].values
y_test = test_df['regime'].values

print(f"Features: {len(feature_cols)} columns")
print(f"  {', '.join(feature_cols)}")
print()

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Grid search for best C
print("="*80)
print("TRAINING LOGISTIC REGRESSION (class_weight='balanced')")
print("="*80)
print()

print("Grid search for best C...")
best_c, best_f1 = None, -1
for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
    lr = LogisticRegression(solver='lbfgs', C=C, max_iter=2000, random_state=42, class_weight='balanced')
    lr.fit(X_train_s, y_train)
    val_pred = lr.predict(X_val_s)
    val_f1 = f1_score(y_val, val_pred, labels=['TREND', 'TRANSITION', 'CHOP'], average='macro', zero_division=0)
    print(f"  C={C:5.1f}  val_macro_f1={val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_c = C

print()
print(f"Best C={best_c} (val macro F1={best_f1:.4f})")
print()

# Train final model on train+val
print("Training final model on train+val data...")
X_trainval = np.vstack([X_train_s, X_val_s])
y_trainval = np.concatenate([y_train, y_val])

final_model = LogisticRegression(solver='lbfgs', C=best_c, max_iter=2000, random_state=42, class_weight='balanced')
final_model.fit(X_trainval, y_trainval)
print("Training complete")
print()

# Evaluate on test set
print("="*80)
print("TEST SET EVALUATION")
print("="*80)
print()

test_pred = final_model.predict(X_test_s)

# Overall metrics
test_acc = (test_pred == y_test).mean()
test_bal_acc = balanced_accuracy_score(y_test, test_pred)
test_f1_macro = f1_score(y_test, test_pred, labels=['TREND', 'TRANSITION', 'CHOP'], average='macro', zero_division=0)

print(f"Overall Metrics:")
print(f"  Accuracy:          {test_acc:.4f}")
print(f"  Balanced Accuracy: {test_bal_acc:.4f}")
print(f"  Macro F1:          {test_f1_macro:.4f}")
print()

# Per-class metrics
print("Per-Class Performance:")
print(classification_report(y_test, test_pred, labels=['TREND', 'TRANSITION', 'CHOP'], zero_division=0))

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, test_pred, labels=['TREND', 'TRANSITION', 'CHOP'])
print("                 Predicted →")
print("Actual ↓        TREND    TRANSITION    CHOP")
for i, regime in enumerate(['TREND', 'TRANSITION', 'CHOP']):
    print(f"{regime:12s}  {cm[i, 0]:6,}    {cm[i, 1]:6,}    {cm[i, 2]:6,}")
print()

# Save model artifacts
output_dir = Path("output/optimized_model")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("SAVING MODEL ARTIFACTS")
print("="*80)
print()

# Save model
model_path = output_dir / "logistic_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"✓ Model saved to {model_path}")

# Save scaler
scaler_path = output_dir / "scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved to {scaler_path}")

# Save feature names
feature_names_path = output_dir / "feature_names.txt"
with open(feature_names_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"✓ Feature names saved to {feature_names_path}")

# Save metrics
metrics_path = output_dir / "test_metrics.txt"
with open(metrics_path, 'w') as f:
    f.write(f"Test Set Metrics (Optimized Model)\n")
    f.write(f"="*50 + "\n\n")
    f.write(f"Overall:\n")
    f.write(f"  Accuracy:          {test_acc:.4f}\n")
    f.write(f"  Balanced Accuracy: {test_bal_acc:.4f}\n")
    f.write(f"  Macro F1:          {test_f1_macro:.4f}\n\n")
    f.write(f"Per-Class:\n")
    f.write(classification_report(y_test, test_pred, labels=['TREND', 'TRANSITION', 'CHOP'], zero_division=0))
    f.write(f"\nConfusion Matrix:\n")
    f.write(f"                 Predicted →\n")
    f.write(f"Actual ↓        TREND    TRANSITION    CHOP\n")
    for i, regime in enumerate(['TREND', 'TRANSITION', 'CHOP']):
        f.write(f"{regime:12s}  {cm[i, 0]:6,}    {cm[i, 1]:6,}    {cm[i, 2]:6,}\n")
print(f"✓ Metrics saved to {metrics_path}")

# Save thresholds
thresholds_path = output_dir / "thresholds.txt"
with open(thresholds_path, 'w') as f:
    f.write("Optimized Labeling Thresholds\n")
    f.write("="*50 + "\n\n")
    f.write("TREND:\n")
    f.write("  efficiency_ratio >= 0.30\n")
    f.write("  |R_ATR| >= 3.0\n\n")
    f.write("CHOP:\n")
    f.write("  efficiency_ratio <= 0.07\n")
    f.write("  |R_ATR| <= 0.5\n\n")
    f.write("TRANSITION:\n")
    f.write("  everything else\n")
print(f"✓ Thresholds saved to {thresholds_path}")

print()
print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print()

print(f"📊 Final Results:")
print(f"   Macro F1:      {test_f1_macro:.4f}")
print(f"   Accuracy:      {test_acc:.4f}")
print(f"   TREND Recall:  {(test_pred[y_test == 'TREND'] == 'TREND').mean():.4f}")
print(f"   CHOP Recall:   {(test_pred[y_test == 'CHOP'] == 'CHOP').mean():.4f}")
print()

print(f"📁 Model artifacts saved to: {output_dir}/")
print(f"   - logistic_model.pkl (Logistic Regression model)")
print(f"   - scaler.pkl (StandardScaler)")
print(f"   - feature_names.txt (Feature column names)")
print(f"   - test_metrics.txt (Complete test metrics)")
print(f"   - thresholds.txt (Labeling thresholds)")
print()

print("🎉 Production-ready model with optimized thresholds!")
print()

# Print usage example
print("="*80)
print("USAGE EXAMPLE")
print("="*80)
print("""
import pickle
import pandas as pd
import numpy as np

# Load model artifacts
with open('output/optimized_model/logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('output/optimized_model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare your features (6 columns)
# features = ['feat_delta_di', 'feat_slope', 'feat_er', 'feat_atr_z', 'feat_bbw', 'feat_imbalance']
X = your_feature_dataframe[features].values

# Scale and predict
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)

# predictions will be: 'TREND', 'TRANSITION', or 'CHOP'
""")
