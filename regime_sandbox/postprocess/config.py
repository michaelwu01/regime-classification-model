from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostprocessConfig:
    model_dir: str = "regime_sandbox/train/output"
    features_csv: str = "regime_sandbox/preprocess/output/features.csv"
    labels_csv: str = "regime_sandbox/label/output/regime_labels.csv"  # for price data
    output_dir: str = "regime_sandbox/postprocess/output"

    # ── HMM transition matrix ──
    alpha_smooth: float = 1.0         # Laplace smoothing constant for transition counts

    # ── Postprocess transition rules ──
    trend_confirm_bars: int = 3
    chop_confirm_bars: int = 3

    # Minimum time-in-state: bars_in_state >= T_min before allowing exit
    t_min: int = 5

    eps: float = 1e-8
