from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    catalog_path: str = "sample_data/catalog"
    bar_type_str: str = "GC.n.0.GLBX-5-MINUTE-LAST-EXTERNAL"
    tick_type_str: str = "GC.n.0.GLBX"          # instrument for tick data
    start: str | None = "2020-02-03 00:00:00"
    end: str | None = "2020-03-21 00:00:00"
    output_dir: str = "regime_sandbox/preprocess/output"
    csv_filename: str = "features.csv"

    # Directional Movement (for DI+, DI-)
    dm_period: int = 14

    # Moving-average slope
    ma_period: int = 20           # MA period (EMA)
    slope_lookback: int = 5       # k in slope_{k,t} = (MA_t - MA_{t-k}) / k

    # Kaufman Efficiency Ratio
    er_lookback: int = 20

    # ATR z-score
    atr_period: int = 14          # ATR period (Wilder)
    atr_z_window: int = 100       # rolling z-score window

    # Bollinger bandwidth (raw, not z-scored)
    bbw_period: int = 20          # BB SMA period
    bbw_k: float = 2.0            # Bollinger multiplier k_B

    # Trade imbalance
    imbalance_N: int = 500        # number of recent ticks for signed imbalance

    # General
    feature_clip: float | None = None   # optional clip; disabled by default to preserve the spec formulas
    eps: float = 1e-8
