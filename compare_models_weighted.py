#!/usr/bin/env python3
"""
Compare model performance with class weighting and MLP.
Trains 3 models:
1. Logistic Regression (baseline, no class weights)
2. Logistic Regression with class_weight='balanced'
3. MLP with grid search
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

from regime_sandbox.constants import FEATURE_COLUMNS, LABEL_STR, REGIME_LABELS
from regime_sandbox.train.config import TrainConfig
from regime_sandbox.train.trainer import (
    load_and_merge, split_chronological, _ensure_non_empty_splits,
    _ensure_required_regime_classes, _macro_f1_score,
    _classification_report_3class, _grid_search_logistic, _grid_search_mlp,
    _build_logistic_model, _build_mlp_model
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import pickle

# Configuration
LABELS_CSV = "output/label/regime_labels.csv"
FEATURES_CSV = "output/preprocess/features.csv"
OUTPUT_BASE = "output/comparison"

print("=" * 80)
print("MODEL COMPARISON: Class Weights + MLP")
print("=" * 80)

# Load and prepare data
train_cfg = TrainConfig(
    labels_csv=LABELS_CSV,
    features_csv=FEATURES_CSV,
    output_dir=OUTPUT_BASE,
)

df = load_and_merge(train_cfg)
train_df, val_df, test_df = split_chronological(df, train_cfg.train_frac, train_cfg.val_frac)
_ensure_non_empty_splits(train_df, val_df, test_df)

X_train = train_df[FEATURE_COLUMNS].values
y_train = train_df[train_cfg.label_column].values.astype(int)
X_val = val_df[FEATURE_COLUMNS].values
y_val = val_df[train_cfg.label_column].values.astype(int)
X_test = test_df[FEATURE_COLUMNS].values
y_test = test_df[train_cfg.label_column].values.astype(int)
_ensure_required_regime_classes(y_train, "train")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# Prepare train+val for final models
trainval_df = pd.concat([train_df, val_df], axis=0)
X_trainval = trainval_df[FEATURE_COLUMNS].values
y_trainval = trainval_df[train_cfg.label_column].values.astype(int)

final_scaler = StandardScaler()
X_trainval_s = final_scaler.fit_transform(X_trainval)
X_test_s = final_scaler.transform(X_test)

# Container for results
results = {}

def evaluate_model(model_name, model, X_test_s, y_test):
    """Evaluate a model and return metrics."""
    test_pred = model.predict(X_test_s)

    test_acc = accuracy_score(y_test, test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)
    test_f1 = _macro_f1_score(y_test, test_pred)
    test_cm = confusion_matrix(y_test, test_pred, labels=REGIME_LABELS).tolist()
    test_report = _classification_report_3class(y_test, test_pred)

    return {
        "model_name": model_name,
        "test_accuracy": round(test_acc, 4),
        "test_balanced_accuracy": round(test_bal_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_confusion_matrix": test_cm,
        "test_classification_report": test_report,
    }

# Model 1: Logistic Regression (baseline)
print("\n" + "=" * 80)
print("MODEL 1: Logistic Regression (Baseline - No Class Weights)")
print("=" * 80)
best_model_1, best_params_1, grid_results_1 = _grid_search_logistic(
    X_train_s, y_train, X_val_s, y_val, train_cfg, class_weight=None
)
final_model_1 = _build_logistic_model(best_params_1["C"], train_cfg, class_weight=None)
final_model_1.fit(X_trainval_s, y_trainval)
results["logistic_baseline"] = evaluate_model("Logistic (Baseline)", final_model_1, X_test_s, y_test)
results["logistic_baseline"]["best_params"] = best_params_1

# Model 2: Logistic Regression with class_weight='balanced'
print("\n" + "=" * 80)
print("MODEL 2: Logistic Regression with class_weight='balanced'")
print("=" * 80)
best_model_2, best_params_2, grid_results_2 = _grid_search_logistic(
    X_train_s, y_train, X_val_s, y_val, train_cfg, class_weight='balanced'
)
final_model_2 = _build_logistic_model(best_params_2["C"], train_cfg, class_weight='balanced')
final_model_2.fit(X_trainval_s, y_trainval)
results["logistic_weighted"] = evaluate_model("Logistic (Weighted)", final_model_2, X_test_s, y_test)
results["logistic_weighted"]["best_params"] = best_params_2
results["logistic_weighted"]["class_weight"] = "balanced"

# Model 3: MLP
print("\n" + "=" * 80)
print("MODEL 3: MLP (Single Hidden Layer)")
print("=" * 80)
best_model_3, best_params_3, grid_results_3 = _grid_search_mlp(
    X_train_s, y_train, X_val_s, y_val, train_cfg
)
final_model_3 = _build_mlp_model(best_params_3["hidden_width"], best_params_3["alpha"], train_cfg)
final_model_3.fit(X_trainval_s, y_trainval)
results["mlp"] = evaluate_model("MLP", final_model_3, X_test_s, y_test)
results["mlp"]["best_params"] = best_params_3

# Save models
Path(OUTPUT_BASE).mkdir(parents=True, exist_ok=True)
with open(f"{OUTPUT_BASE}/logistic_baseline_model.pkl", "wb") as f:
    pickle.dump(final_model_1, f)
with open(f"{OUTPUT_BASE}/logistic_weighted_model.pkl", "wb") as f:
    pickle.dump(final_model_2, f)
with open(f"{OUTPUT_BASE}/mlp_model.pkl", "wb") as f:
    pickle.dump(final_model_3, f)
with open(f"{OUTPUT_BASE}/scaler.pkl", "wb") as f:
    pickle.dump(final_scaler, f)

# Print comparison
print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

print("\nOverall Performance:")
print(f"{'Model':<30} {'Accuracy':>10} {'Bal.Acc':>10} {'Macro F1':>10}")
print("-" * 62)
for key in ["logistic_baseline", "logistic_weighted", "mlp"]:
    r = results[key]
    print(f"{r['model_name']:<30} {r['test_accuracy']:>10.4f} {r['test_balanced_accuracy']:>10.4f} {r['test_macro_f1']:>10.4f}")

print("\nPer-Class Performance (TREND - Most Important):")
print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 62)
for key in ["logistic_baseline", "logistic_weighted", "mlp"]:
    r = results[key]
    trend_metrics = r['test_classification_report']['TREND']
    print(f"{r['model_name']:<30} {trend_metrics['precision']:>10.4f} {trend_metrics['recall']:>10.4f} {trend_metrics['f1-score']:>10.4f}")

print("\nPer-Class Performance (TRANSITION):")
print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 62)
for key in ["logistic_baseline", "logistic_weighted", "mlp"]:
    r = results[key]
    trans_metrics = r['test_classification_report']['TRANSITION']
    print(f"{r['model_name']:<30} {trans_metrics['precision']:>10.4f} {trans_metrics['recall']:>10.4f} {trans_metrics['f1-score']:>10.4f}")

print("\nPer-Class Performance (CHOP):")
print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 62)
for key in ["logistic_baseline", "logistic_weighted", "mlp"]:
    r = results[key]
    chop_metrics = r['test_classification_report']['CHOP']
    print(f"{r['model_name']:<30} {chop_metrics['precision']:>10.4f} {chop_metrics['recall']:>10.4f} {chop_metrics['f1-score']:>10.4f}")

print("\nConfusion Matrices:")
for key in ["logistic_baseline", "logistic_weighted", "mlp"]:
    r = results[key]
    cm = np.array(r['test_confusion_matrix'])
    print(f"\n{r['model_name']}:")
    class_names = [LABEL_STR[label] for label in REGIME_LABELS]
    print(f"  {'':12s} | " + " | ".join([f"{name:12s}" for name in class_names]) + " |")
    print(f"  {'-'*12} | " + " | ".join(['-'*12 for _ in class_names]) + " |")
    for i, actual_class in enumerate(class_names):
        row_str = f"  {actual_class:12s} | "
        row_str += " | ".join([f"{cm[i, j]:12d}" for j in range(len(class_names))])
        row_str += " |"
        print(row_str)

# Determine best model
best_key = max(results.keys(), key=lambda k: results[k]['test_macro_f1'])
print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print(f"\nBest Model: {results[best_key]['model_name']}")
print(f"  Macro F1: {results[best_key]['test_macro_f1']:.4f}")
print(f"  Balanced Accuracy: {results[best_key]['test_balanced_accuracy']:.4f}")
print(f"  TREND Recall: {results[best_key]['test_classification_report']['TREND']['recall']:.4f}")

# Calculate improvements
baseline_f1 = results['logistic_baseline']['test_macro_f1']
baseline_trend_recall = results['logistic_baseline']['test_classification_report']['TREND']['recall']
best_f1 = results[best_key]['test_macro_f1']
best_trend_recall = results[best_key]['test_classification_report']['TREND']['recall']

print(f"\nImprovements over baseline:")
print(f"  Macro F1: {baseline_f1:.4f} → {best_f1:.4f} (+{(best_f1-baseline_f1)*100:.1f}%)")
print(f"  TREND Recall: {baseline_trend_recall:.4f} → {best_trend_recall:.4f} (+{(best_trend_recall-baseline_trend_recall)*100:.1f}%)")

# Save comparison report
with open(f"{OUTPUT_BASE}/comparison_report.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nDetailed report saved to: {OUTPUT_BASE}/comparison_report.json")
print(f"Models saved to: {OUTPUT_BASE}/")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)
