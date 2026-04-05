"""
Standalone regime labeling system.

Usage: python -m regime_sandbox.label.run
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from regime_sandbox.constants import LABEL_STR
from regime_sandbox.data_loader import load_bars
from regime_sandbox.label.config import RegimeLabelConfig
from regime_sandbox.label.labeler import compute_raw_labels
from regime_sandbox.label.smoother import smooth_labels


def main(cfg: RegimeLabelConfig | None = None) -> None:
    if cfg is None:
        cfg = RegimeLabelConfig()

    # 1. Load bars
    df = load_bars(cfg.catalog_path, cfg.bar_type_str, cfg.start, cfg.end)

    # 2. Compute features and raw labels
    df = compute_raw_labels(df, cfg)

    # 3. Smooth labels
    valid_mask = df["raw_label"].notna()
    raw = df.loc[valid_mask, "raw_label"].values.astype(int)
    n_valid = int(valid_mask.sum())

    if n_valid == 0:
        print("WARNING: no valid raw labels were produced; skipping smoothing")
        smoothed = np.array([], dtype=int)
    else:
        smoothed = smooth_labels(raw, cfg)

    df["regime_label"] = np.nan
    df.loc[valid_mask, "regime_label"] = smoothed.astype(float)
    df["regime_str"] = df["regime_label"].map(
        {float(k): v for k, v in LABEL_STR.items()}
    )

    if n_valid > 0:
        for label_int, label_str in LABEL_STR.items():
            count = int((smoothed == label_int).sum())
            print(f"  Smoothed {label_str}: {count} ({count / n_valid:.1%})")

    # 4. Save CSV
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / cfg.csv_filename

    csv_cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "atr", "efficiency", "abs_r_atr",
        "raw_label", "regime_label", "regime_str",
    ]
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
