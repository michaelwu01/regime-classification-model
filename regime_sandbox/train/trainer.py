"""Grid search, evaluation, and artifact saving for regime classifiers.

Supports two emission models per spec:
    1. Softmax (logistic regression)
    2. Single hidden layer MLP (ReLU + softmax)
"""
from __future__ import annotations

import inspect
import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

from regime_sandbox.constants import FEATURE_COLUMNS, LABEL_STR, REGIME_LABELS
from regime_sandbox.postprocess.hmm_decode import (
    estimate_initial_distribution,
    estimate_transition_matrix,
)
from regime_sandbox.train.config import TrainConfig


def load_and_merge(cfg: TrainConfig) -> pd.DataFrame:
    """Read labels CSV + features CSV, inner join on timestamp, drop NaN, sort."""
    labels_df = pd.read_csv(cfg.labels_csv)
    features_df = pd.read_csv(cfg.features_csv)

    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], utc=True).dt.tz_localize(None)
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True).dt.tz_localize(None)

    df = pd.merge(labels_df, features_df, on="timestamp", how="inner")
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    required_cols = FEATURE_COLUMNS + [cfg.label_column]
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError(
            "Merged dataset is empty after dropping NaN rows. "
            "Check the generated features, labels, and preprocessing outputs."
        )

    print(f"Merged dataset: {len(df):,} rows (after dropping NaN)")
    return df


def split_chronological(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe chronologically into train/val/test."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}.")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}.")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1, got {train_frac + val_frac:.3f}."
        )

    n = len(df)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    print(f"Split: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
    return train_df, val_df, test_df


def _ensure_non_empty_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    empty = [
        name
        for name, split_df in (
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        )
        if split_df.empty
    ]
    if empty:
        raise ValueError(
            "Chronological split produced empty subset(s): "
            f"{', '.join(empty)}. Adjust train/val fractions or add more data."
        )


def _ensure_required_regime_classes(labels: np.ndarray, split_name: str) -> None:
    present = {int(label) for label in np.unique(labels)}
    missing = [LABEL_STR[label] for label in REGIME_LABELS if label not in present]
    if missing:
        raise ValueError(
            f"{split_name} split is missing regime classes: {missing}. "
            "The guide defines a 3-class regime model, so the in-sample data must "
            "contain TREND, TRANSITION, and CHOP."
        )


def _macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=REGIME_LABELS,
            average="macro",
            zero_division=0,
        )
    )


def _classification_report_3class(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=REGIME_LABELS,
        target_names=[LABEL_STR[label] for label in REGIME_LABELS],
        output_dict=True,
        zero_division=0,
    )


def _extract_logistic_parameters(
    model: LogisticRegression,
    classes: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    coef = model.coef_
    intercept = model.intercept_

    if coef.shape[0] == len(classes) and intercept.shape[0] == len(classes):
        coef_by_class = {
            int(cls): coef[idx]
            for idx, cls in enumerate(classes)
        }
        intercept_by_class = {
            int(cls): float(intercept[idx])
            for idx, cls in enumerate(classes)
        }
        return coef_by_class, intercept_by_class

    if len(classes) == 2 and coef.shape[0] == 1 and intercept.shape[0] == 1:
        pos_class = int(classes[1])
        neg_class = int(classes[0])
        coef_by_class = {
            neg_class: -coef[0],
            pos_class: coef[0],
        }
        intercept_by_class = {
            neg_class: float(-intercept[0]),
            pos_class: float(intercept[0]),
        }
        return coef_by_class, intercept_by_class

    raise ValueError(
        "Unexpected logistic regression parameter shape: "
        f"coef_={coef.shape}, intercept_={intercept.shape}, classes={classes}."
    )


def _build_logistic_model(C: float, cfg: TrainConfig, class_weight=None) -> LogisticRegression:
    kwargs = dict(
        solver="lbfgs",
        C=C,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        class_weight=class_weight,
    )
    if "multi_class" in inspect.signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


def _build_mlp_model(hidden_width: int, alpha: float, cfg: TrainConfig) -> MLPClassifier:
    """Single hidden layer MLP: h = ReLU(W1*x + b1), z = W2*h + b2, softmax(z)."""
    return MLPClassifier(
        hidden_layer_sizes=(hidden_width,),
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=cfg.mlp_learning_rate_init,
        max_iter=cfg.mlp_max_iter,
        random_state=cfg.random_state,
    )


def _grid_search_logistic(
    X_train_s, y_train, X_val_s, y_val, cfg, class_weight=None,
) -> tuple[object, dict, list[dict]]:
    best_model = None
    best_params = {}
    best_val_f1 = -1.0
    grid_results = []

    for C in cfg.c_grid:
        model = _build_logistic_model(C, cfg, class_weight=class_weight)
        model.fit(X_train_s, y_train)
        val_pred = model.predict(X_val_s)
        val_f1 = _macro_f1_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred)
        grid_results.append({"C": C, "val_macro_f1": round(val_f1, 4), "val_accuracy": round(val_acc, 4)})
        print(f"  C={C:5.1f}  val_macro_f1={val_f1:.4f}  val_acc={val_acc:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"C": C}

    weight_str = f" (class_weight={class_weight})" if class_weight else ""
    print(f"\nBest logistic: C={best_params['C']} (val macro F1={best_val_f1:.4f}){weight_str}")
    return best_model, best_params, grid_results


def _grid_search_mlp(
    X_train_s, y_train, X_val_s, y_val, cfg,
) -> tuple[object, dict, list[dict]]:
    best_model = None
    best_params = {}
    best_val_f1 = -1.0
    grid_results = []

    for hidden_width, alpha in product(cfg.mlp_hidden_widths, cfg.mlp_alpha_grid):
        model = _build_mlp_model(hidden_width, alpha, cfg)
        model.fit(X_train_s, y_train)
        val_pred = model.predict(X_val_s)
        val_f1 = _macro_f1_score(y_val, val_pred)
        val_acc = accuracy_score(y_val, val_pred)
        grid_results.append({
            "hidden_width": hidden_width, "alpha": alpha,
            "val_macro_f1": round(val_f1, 4), "val_accuracy": round(val_acc, 4),
        })
        print(f"  m={hidden_width:3d}  alpha={alpha:.0e}  val_macro_f1={val_f1:.4f}  val_acc={val_acc:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model
            best_params = {"hidden_width": hidden_width, "alpha": alpha}

    print(f"\nBest MLP: m={best_params['hidden_width']}, alpha={best_params['alpha']:.0e} (val macro F1={best_val_f1:.4f})")
    return best_model, best_params, grid_results


def train_and_evaluate(cfg: TrainConfig) -> dict:
    """Full training pipeline: load, split, grid search, retrain, evaluate, save."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(cfg)
    train_df, val_df, test_df = split_chronological(df, cfg.train_frac, cfg.val_frac)
    _ensure_non_empty_splits(train_df, val_df, test_df)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[cfg.label_column].values.astype(int)
    X_val = val_df[FEATURE_COLUMNS].values
    y_val = val_df[cfg.label_column].values.astype(int)
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[cfg.label_column].values.astype(int)
    _ensure_required_regime_classes(y_train, "train")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print(f"\nModel type: {cfg.model_type}")
    if cfg.model_type == "logistic":
        best_model, best_params, grid_results = _grid_search_logistic(X_train_s, y_train, X_val_s, y_val, cfg)
    elif cfg.model_type == "mlp":
        best_model, best_params, grid_results = _grid_search_mlp(X_train_s, y_train, X_val_s, y_val, cfg)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}. Use 'logistic' or 'mlp'.")

    # Retrain on train+val
    trainval_df = pd.concat([train_df, val_df], axis=0)
    X_trainval = trainval_df[FEATURE_COLUMNS].values
    y_trainval = trainval_df[cfg.label_column].values.astype(int)
    _ensure_required_regime_classes(y_trainval, "train+val")

    final_scaler = StandardScaler()
    X_trainval_s = final_scaler.fit_transform(X_trainval)
    X_test_s = final_scaler.transform(X_test)

    if cfg.model_type == "logistic":
        final_model = _build_logistic_model(best_params["C"], cfg)
    else:
        final_model = _build_mlp_model(best_params["hidden_width"], best_params["alpha"], cfg)
    final_model.fit(X_trainval_s, y_trainval)

    # Evaluate on test
    test_pred = final_model.predict(X_test_s)

    test_acc = accuracy_score(y_test, test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)
    test_f1 = _macro_f1_score(y_test, test_pred)
    test_cm = confusion_matrix(y_test, test_pred, labels=REGIME_LABELS).tolist()
    test_report = _classification_report_3class(y_test, test_pred)

    print(f"\nTest results:")
    print(f"  Accuracy:          {test_acc:.4f}")
    print(f"  Balanced accuracy: {test_bal_acc:.4f}")
    print(f"  Macro F1:          {test_f1:.4f}")
    print(f"  Confusion matrix:\n{np.array(test_cm)}")

    # Save artifacts
    label_mapping = {int(k): v for k, v in LABEL_STR.items()}
    classes = final_model.classes_.tolist()

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(final_scaler, f)
    np.save(output_dir / "hmm_train_labels.npy", y_trainval)
    with open(output_dir / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    hmm_transition_matrix = estimate_transition_matrix(y_trainval)
    hmm_initial_distribution = estimate_initial_distribution(y_trainval)

    training_report = {
        "model_type": cfg.model_type,
        "best_params": best_params,
        "grid_results": grid_results,
        "classes": classes,
        "report_label_order": [LABEL_STR[label] for label in REGIME_LABELS],
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "test_accuracy": round(test_acc, 4),
        "test_balanced_accuracy": round(test_bal_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_confusion_matrix": test_cm,
        "test_classification_report": test_report,
        "hmm_train_label_counts": {
            LABEL_STR[label]: int((y_trainval == label).sum())
            for label in REGIME_LABELS
        },
        "hmm_transition_matrix": hmm_transition_matrix.tolist(),
        "hmm_initial_distribution": hmm_initial_distribution.tolist(),
    }

    if cfg.model_type == "logistic":
        coef_by_class, intercept_by_class = _extract_logistic_parameters(final_model, classes)
        training_report["coefficients"] = {
            label_mapping.get(c, str(c)): {
                feat: round(float(coef), 6) for feat, coef in zip(FEATURE_COLUMNS, coef_by_class[int(c)])
            } for c in classes
        }
        training_report["intercepts"] = {
            label_mapping.get(c, str(c)): round(intercept_by_class[int(c)], 6)
            for c in classes
        }
    elif cfg.model_type == "mlp":
        training_report["mlp_n_layers"] = len(final_model.coefs_)
        training_report["mlp_layer_sizes"] = [w.shape for w in final_model.coefs_]

    with open(output_dir / "training_report.json", "w") as f:
        json.dump(training_report, f, indent=2, default=str)

    print(f"\nArtifacts saved to {output_dir}/")
    return training_report
