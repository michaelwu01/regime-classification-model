#!/usr/bin/env python3
"""
Run the complete regime model training pipeline and display results.
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Add project root to path
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
from regime_sandbox.data_loader import load_bars, load_ticks
from regime_sandbox.preprocess.features import compute_features
from regime_sandbox.label.labeler import compute_raw_labels
from regime_sandbox.label.smoother import smooth_labels
from regime_sandbox.train.trainer import train_and_evaluate

# Configuration
CATALOG_PATH = "sample_data/catalog"
BAR_TYPE_STR = "GC.n.0.GLBX-5-MINUTE-LAST-EXTERNAL"
TICK_INSTR = "GC.n.0.GLBX"
START = "2020-02-03 00:00:00"
END = "2020-03-21 00:00:00"

PREPROCESS_OUT = "output/preprocess"
LABEL_OUT = "output/label"
TRAIN_OUT = "output/train"

MODEL_TYPE = "logistic"

print("=" * 80)
print("REGIME MODEL TRAINING PIPELINE")
print("=" * 80)

# Step 1: Preprocess - Compute Features
print("\n[1/3] PREPROCESSING - Computing Features...")
print("-" * 80)

preprocess_cfg = PreprocessConfig(
    catalog_path=CATALOG_PATH,
    bar_type_str=BAR_TYPE_STR,
    tick_type_str=TICK_INSTR,
    start=START,
    end=END,
    output_dir=PREPROCESS_OUT,
)

# Load bars
print(f"Loading bars from {CATALOG_PATH}...")
bars_df = load_bars(preprocess_cfg.catalog_path, preprocess_cfg.bar_type_str,
                    preprocess_cfg.start, preprocess_cfg.end)
print(f"  Loaded {len(bars_df)} bars")

# Load ticks for imbalance
print(f"Loading ticks...")
ticks_df = load_ticks(preprocess_cfg.catalog_path, preprocess_cfg.tick_type_str,
                      preprocess_cfg.start, preprocess_cfg.end)
print(f"  Loaded {len(ticks_df)} ticks")

# Compute features
print(f"Computing features...")
features_df = compute_features(bars_df, preprocess_cfg, ticks_df=ticks_df)

# Save
Path(PREPROCESS_OUT).mkdir(parents=True, exist_ok=True)
features_csv = f"{PREPROCESS_OUT}/features.csv"
features_df[["timestamp"] + FEATURE_COLUMNS].to_csv(features_csv, index=False)

print(f"\nFeature Statistics:")
for col in FEATURE_COLUMNS:
    valid = features_df[col].notna().sum()
    print(f"  {col}: {valid:,} valid samples")

print(f"\nSaved features to {features_csv}")

# Step 2: Label - Generate Regime Labels
print("\n[2/3] LABELING - Generating Regime Labels...")
print("-" * 80)

label_cfg = RegimeLabelConfig(
    catalog_path=CATALOG_PATH,
    bar_type_str=BAR_TYPE_STR,
    instrument_id=TICK_INSTR,
    start=START,
    end=END,
    output_dir=LABEL_OUT,
)

label_df = bars_df.copy()
label_df = compute_raw_labels(label_df, label_cfg)

# Smooth
valid_mask = label_df["raw_label"].notna()
raw = label_df.loc[valid_mask, "raw_label"].values.astype(int)
n_valid = int(valid_mask.sum())

if n_valid == 0:
    print("WARNING: no valid raw labels")
    sys.exit(1)
else:
    smoothed = smooth_labels(raw, label_cfg)

label_df["regime_label"] = np.nan
label_df.loc[valid_mask, "regime_label"] = smoothed.astype(float)
label_df["regime_str"] = label_df["regime_label"].map(
    {float(k): v for k, v in LABEL_STR.items()}
)

# Save
Path(LABEL_OUT).mkdir(parents=True, exist_ok=True)
labels_csv = f"{LABEL_OUT}/regime_labels.csv"
label_df[["timestamp", "open", "high", "low", "close", "volume",
          "atr", "efficiency", "abs_r_atr",
          "raw_label", "regime_label", "regime_str"]].to_csv(labels_csv, index=False)

print(f"\nLabel Distribution:")
for li, ls in LABEL_STR.items():
    count = int((smoothed == li).sum())
    pct = count/n_valid*100 if n_valid > 0 else 0
    print(f"  {ls:12s}: {count:5d} ({pct:5.1f}%)")

print(f"\nSaved labels to {labels_csv}")

# Step 3: Train - Fit Classifier
print("\n[3/3] TRAINING - Fitting Classifier Model...")
print("-" * 80)

train_cfg = TrainConfig(
    labels_csv=labels_csv,
    features_csv=features_csv,
    output_dir=TRAIN_OUT,
    model_type=MODEL_TYPE,
)

print(f"Model type: {MODEL_TYPE}")
print(f"Training classifier with grid search...")

report = train_and_evaluate(train_cfg)

# Display Results
print("\n" + "=" * 80)
print("TRAINING RESULTS")
print("=" * 80)

print(f"\nModel Configuration:")
print(f"  Type: {report['model_type']}")
print(f"  Best params: {report['best_params']}")

print(f"\nTest Set Performance:")
print(f"  Accuracy:          {report['test_accuracy']:.4f}")
print(f"  Balanced Accuracy: {report['test_balanced_accuracy']:.4f}")
print(f"  Macro F1-Score:    {report['test_macro_f1']:.4f}")

print(f"\nPer-Class Metrics:")
class_report = report.get('test_classification_report_dict', {})
if class_report:
    for class_name in ['CHOP', 'TRANSITION', 'TREND']:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"\n  {class_name}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {int(metrics['support'])}")

print(f"\nConfusion Matrix:")
cm = np.array(report["test_confusion_matrix"])
class_names = report.get("report_label_order",
                         [LABEL_STR.get(c, str(c)) for c in report["classes"]])

# Print header
print(f"\n  {'':12s} | " + " | ".join([f"{name:12s}" for name in class_names]) + " |")
print(f"  {'-'*12} | " + " | ".join(['-'*12 for _ in class_names]) + " |")

# Print rows
for i, actual_class in enumerate(class_names):
    row_str = f"  {actual_class:12s} | "
    row_str += " | ".join([f"{cm[i, j]:12d}" for j in range(len(class_names))])
    row_str += " |"
    print(row_str)

print(f"\nHMM Training Labels:")
hmm_counts = report.get("hmm_train_label_counts", {})
for label_name, count in hmm_counts.items():
    print(f"  {label_name:12s}: {count:5d}")

print(f"\nModel saved to: {TRAIN_OUT}/model.pkl")
print(f"Scaler saved to: {TRAIN_OUT}/scaler.pkl")
print(f"Report saved to: {TRAIN_OUT}/training_report.json")

# Save detailed report
report_path = f"{TRAIN_OUT}/training_report.json"
with open(report_path, 'w') as f:
    json.dump({k: v for k, v in report.items() if k != 'model'},
              f, indent=2, default=str)

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
