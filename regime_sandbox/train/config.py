from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    labels_csv: str = "regime_sandbox/label/output/regime_labels.csv"
    features_csv: str = "regime_sandbox/preprocess/output/features.csv"
    output_dir: str = "regime_sandbox/train/output"
    label_column: str = "regime_label"

    train_frac: float = 0.60
    val_frac: float = 0.20

    # Model selection: "logistic" or "mlp"
    model_type: str = "logistic"

    # Logistic regression hyperparameters
    c_grid: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    max_iter: int = 2000

    # MLP hyperparameters (single hidden layer + ReLU + softmax)
    mlp_hidden_widths: tuple[int, ...] = (16, 32, 64)   # grid search over m
    mlp_alpha_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2)  # L2 regularization
    mlp_learning_rate_init: float = 1e-3
    mlp_max_iter: int = 500

    random_state: int = 42
