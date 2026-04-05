#!/usr/bin/env python3
"""
Train regime model with expanded 2020-2022 data.
Uses the converted 5-minute bars from DBN files.
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from regime_sandbox.constants import (
    FEATURE_COLUMNS, LABEL_STR, REGIME_LABELS,
    TREND, TRANSITION, CHOP,
)
from regime_sandbox.preprocess.config import PreprocessConfig
from regime_sandbox.label.config import RegimeLabelConfig
from regime_sandbox.train.config import TrainConfig
from regime_sandbox.preprocess.features import compute_features
from regime_sandbox.label.labeler import compute_raw_labels
from regime_sandbox.label.smoother import smooth_labels
from regime_sandbox.train.trainer import train_and_evaluate

# Configuration for expanded data
BARS_CSV = "expanded_data/GC_bars_5min_2020_2022.csv"
START = "2020-01-01 00:00:00"
END = "2022-12-31 23:59:59"

PREPROCESS_OUT = "output/expanded_preprocess"
LABEL_OUT = "output/expanded_label"
TRAIN_OUT_LOGISTIC_WEIGHTED = "output/expanded_train_logistic_weighted"
TRAIN_OUT_MLP = "output/expanded_train_mlp"

print("=" * 80)
print("REGIME MODEL TRAINING - EXPANDED DATASET (2020-2022)")
print("=" * 80)

# Check if bars file exists
if not os.path.exists(BARS_CSV):
    print(f"\nERROR: Bars file not found: {BARS_CSV}")
    print("Please run convert_dbn_to_bars.py first!")
    sys.exit(1)

# Load bars from CSV
print(f"\nLoading bars from {BARS_CSV}...")
bars_df = pd.read_csv(BARS_CSV)
bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])
print(f"Loaded {len(bars_df):,} bars")
print(f"Date range: {bars_df['timestamp'].min()} to {bars_df['timestamp'].max()}")

# Step 1: Preprocess - Compute Features
print("\n[1/4] PREPROCESSING - Computing Features...")
print("-" * 80)

# For features, we need to compute without ticks (no imbalance feature initially)
# We'll compute 5 features and set imbalance to 0 or remove it
features_df = bars_df.copy()

# Compute technical indicators manually since we don't have ticks
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# 1. Delta DI (from ADX)
adx_indicator = ADXIndicator(high=features_df['high'], low=features_df['low'],
                             close=features_df['close'], window=14)
features_df['feat_delta_di'] = adx_indicator.adx_pos() - adx_indicator.adx_neg()

# 2. Slope
window_slope = 20
features_df['ma'] = features_df['close'].rolling(window=window_slope).mean()
features_df['feat_slope'] = (features_df['ma'] - features_df['ma'].shift(window_slope)) / window_slope

# 3. Efficiency Ratio
window_er = 20
features_df['change'] = (features_df['close'] - features_df['close'].shift(window_er)).abs()
features_df['volatility'] = features_df['close'].diff().abs().rolling(window=window_er).sum()
features_df['feat_er'] = features_df['change'] / (features_df['volatility'] + 1e-10)

# 4. ATR Z-score
atr_indicator = AverageTrueRange(high=features_df['high'], low=features_df['low'],
                                close=features_df['close'], window=14)
features_df['atr'] = atr_indicator.average_true_range()
atr_mean = features_df['atr'].rolling(window=100).mean()
atr_std = features_df['atr'].rolling(window=100).std()
features_df['feat_atr_z'] = (features_df['atr'] - atr_mean) / (atr_std + 1e-10)

# 5. Bollinger Band Width
bb_indicator = BollingerBands(close=features_df['close'], window=20, window_dev=2)
bb_upper = bb_indicator.bollinger_hband()
bb_lower = bb_indicator.bollinger_lband()
bb_middle = bb_indicator.bollinger_mavg()
features_df['feat_bbw'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)

# 6. Imbalance (set to 0 since we don't have tick data)
features_df['feat_imbalance'] = 0.0

# Save features
Path(PREPROCESS_OUT).mkdir(parents=True, exist_ok=True)
features_csv = f"{PREPROCESS_OUT}/features.csv"
features_df[["timestamp"] + FEATURE_COLUMNS].to_csv(features_csv, index=False)

print(f"\nFeature Statistics:")
for col in FEATURE_COLUMNS:
    valid = features_df[col].notna().sum()
    print(f"  {col}: {valid:,} valid samples")

print(f"\nSaved features to {features_csv}")

# Step 2: Label - Generate Regime Labels
print("\n[2/4] LABELING - Generating Regime Labels...")
print("-" * 80)

# Compute labels
label_df = features_df.copy()

# Compute efficiency and abs_r_atr for labeling
horizon = 40
label_df['efficiency'] = label_df['feat_er']  # Already computed
label_df['abs_r_atr'] = label_df['feat_atr_z'].abs() * 5  # Approximate

# Apply labeling rules
label_df['raw_label'] = TRANSITION
label_df.loc[(label_df['efficiency'] >= 0.15) & (label_df['abs_r_atr'] >= 5.0), 'raw_label'] = TREND
label_df.loc[(label_df['efficiency'] <= 0.10) & (label_df['abs_r_atr'] <= 3.0), 'raw_label'] = CHOP

# Smooth labels
valid_mask = label_df["raw_label"].notna()
raw = label_df.loc[valid_mask, "raw_label"].values.astype(int)
n_valid = int(valid_mask.sum())

if n_valid == 0:
    print("WARNING: no valid raw labels")
    sys.exit(1)

# Simple smoothing (majority vote in window)
smoothed = raw.copy()
window = 24
for i in range(len(smoothed)):
    start = max(0, i - window // 2)
    end = min(len(smoothed), i + window // 2 + 1)
    window_labels = smoothed[start:end]
    smoothed[i] = np.bincount(window_labels).argmax()

label_df["regime_label"] = np.nan
label_df.loc[valid_mask, "regime_label"] = smoothed.astype(float)
label_df["regime_str"] = label_df["regime_label"].map(
    {float(k): v for k, v in LABEL_STR.items()}
)

# Save labels
Path(LABEL_OUT).mkdir(parents=True, exist_ok=True)
labels_csv = f"{LABEL_OUT}/regime_labels.csv"
label_df[["timestamp", "open", "high", "low", "close", "volume",
          "atr", "efficiency", "abs_r_atr",
          "raw_label", "regime_label", "regime_str"]].to_csv(labels_csv, index=False)

print(f"\nLabel Distribution:")
for li, ls in LABEL_STR.items():
    count = int((smoothed == li).sum())
    pct = count/n_valid*100 if n_valid > 0 else 0
    print(f"  {ls:12s}: {count:6d} ({pct:5.1f}%)")

print(f"\nSaved labels to {labels_csv}")

# Step 3: Train Logistic with class_weight='balanced'
print("\n[3/4] TRAINING - Logistic Regression with class_weight='balanced'...")
print("-" * 80)

train_cfg_log = TrainConfig(
    labels_csv=labels_csv,
    features_csv=features_csv,
    output_dir=TRAIN_OUT_LOGISTIC_WEIGHTED,
    model_type="logistic",
)

# We need to modify the trainer to support class weights
# For now, let's manually train
from regime_sandbox.train.trainer import load_and_merge, split_chronological
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle

df = load_and_merge(train_cfg_log)
train_df, val_df, test_df = split_chronological(df, train_cfg_log.train_frac, train_cfg_log.val_frac)

X_train = train_df[FEATURE_COLUMNS].values
y_train = train_df['regime_label'].values.astype(int)
X_val = val_df[FEATURE_COLUMNS].values
y_val = val_df['regime_label'].values.astype(int)
X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df['regime_label'].values.astype(int)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# Grid search with class_weight='balanced'
best_c = None
best_f1 = -1
for C in [0.1, 0.3, 1.0, 3.0, 10.0]:
    model = LogisticRegression(solver='lbfgs', C=C, max_iter=2000,
                              random_state=42, class_weight='balanced')
    model.fit(X_train_s, y_train)
    val_pred = model.predict(X_val_s)
    from sklearn.metrics import f1_score
    val_f1 = f1_score(y_val, val_pred, labels=REGIME_LABELS, average='macro', zero_division=0)
    print(f"  C={C:5.1f}  val_macro_f1={val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_c = C

print(f"\nBest C={best_c} (val macro F1={best_f1:.4f})")

# Retrain on train+val with best params
trainval_df = pd.concat([train_df, val_df])
X_trainval = trainval_df[FEATURE_COLUMNS].values
y_trainval = trainval_df['regime_label'].values.astype(int)

final_scaler = StandardScaler()
X_trainval_s = final_scaler.fit_transform(X_trainval)
X_test_s = final_scaler.transform(X_test)

final_model = LogisticRegression(solver='lbfgs', C=best_c, max_iter=2000,
                                 random_state=42, class_weight='balanced')
final_model.fit(X_trainval_s, y_trainval)

# Evaluate
test_pred = final_model.predict(X_test_s)
from sklearn.metrics import accuracy_score, balanced_accuracy_score

test_acc = accuracy_score(y_test, test_pred)
test_bal_acc = balanced_accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, labels=REGIME_LABELS, average='macro', zero_division=0)
test_cm = confusion_matrix(y_test, test_pred, labels=REGIME_LABELS)
test_report = classification_report(y_test, test_pred, labels=REGIME_LABELS,
                                   target_names=[LABEL_STR[l] for l in REGIME_LABELS],
                                   output_dict=True, zero_division=0)

print(f"\nTest Results (Logistic Weighted):")
print(f"  Accuracy:          {test_acc:.4f}")
print(f"  Balanced Accuracy: {test_bal_acc:.4f}")
print(f"  Macro F1:          {test_f1:.4f}")
print(f"\nConfusion Matrix:")
print(test_cm)
print(f"\nPer-Class Metrics:")
for class_name in ['TREND', 'TRANSITION', 'CHOP']:
    metrics = test_report[class_name]
    print(f"  {class_name:12s}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

# Save model
Path(TRAIN_OUT_LOGISTIC_WEIGHTED).mkdir(parents=True, exist_ok=True)
with open(f"{TRAIN_OUT_LOGISTIC_WEIGHTED}/model.pkl", "wb") as f:
    pickle.dump(final_model, f)
with open(f"{TRAIN_OUT_LOGISTIC_WEIGHTED}/scaler.pkl", "wb") as f:
    pickle.dump(final_scaler, f)

# Step 4: Train MLP
print("\n[4/4] TRAINING - MLP...")
print("-" * 80)

train_cfg_mlp = TrainConfig(
    labels_csv=labels_csv,
    features_csv=features_csv,
    output_dir=TRAIN_OUT_MLP,
    model_type="mlp",
)

report_mlp = train_and_evaluate(train_cfg_mlp)

print("\n" + "=" * 80)
print("TRAINING COMPLETE - EXPANDED DATASET")
print("=" * 80)

print("\n📊 FINAL COMPARISON:")
print("\nLogistic (Weighted) vs MLP:")
print(f"  Logistic Weighted - Macro F1: {test_f1:.4f}, TREND Recall: {test_report['TREND']['recall']:.4f}")
print(f"  MLP               - Macro F1: {report_mlp['test_macro_f1']:.4f}, TREND Recall: {report_mlp['test_classification_report']['TREND']['recall']:.4f}")

print("\n✅ All models trained successfully!")
print(f"Results saved to:")
print(f"  - {TRAIN_OUT_LOGISTIC_WEIGHTED}/")
print(f"  - {TRAIN_OUT_MLP}/")
