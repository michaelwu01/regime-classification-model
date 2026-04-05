"""
Causal feature computation for regime classifier.

Usage: python -m regime_sandbox.preprocess.run
"""
from __future__ import annotations

from pathlib import Path

from regime_sandbox.constants import FEATURE_COLUMNS
from regime_sandbox.data_loader import load_bars, load_ticks
from regime_sandbox.preprocess.config import PreprocessConfig
from regime_sandbox.preprocess.features import compute_features


def main(cfg: PreprocessConfig | None = None) -> None:
    if cfg is None:
        cfg = PreprocessConfig()

    # 1. Load bars
    df = load_bars(cfg.catalog_path, cfg.bar_type_str, cfg.start, cfg.end)

    # 2. Load tick data for imbalance feature
    ticks_df = load_ticks(cfg.catalog_path, cfg.tick_type_str, cfg.start, cfg.end)

    # 3. Compute features
    features_df = compute_features(df, cfg, ticks_df=ticks_df)

    # 4. Report stats
    for col in FEATURE_COLUMNS:
        valid = features_df[col].notna().sum()
        print(f"  {col}: {valid:,} valid values")

    # 5. Save CSV
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / cfg.csv_filename

    features_df[["timestamp"] + FEATURE_COLUMNS].to_csv(csv_path, index=False)
    print(f"Saved features CSV to {csv_path}")


if __name__ == "__main__":
    main()
