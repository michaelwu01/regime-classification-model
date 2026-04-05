from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeLabelConfig:
    # Data source
    catalog_path: str = "sample_data/catalog"
    instrument_id: str = "GC.n.0.GLBX"
    bar_type_str: str = "GC.n.0.GLBX-5-MINUTE-LAST-EXTERNAL"
    start: str | None = "2020-02-03 00:00:00"
    end: str | None = "2020-03-21 00:00:00"

    # Output
    output_dir: str = "regime_sandbox/label/output"
    csv_filename: str = "regime_labels.csv"
    html_filename: str = "regime_labels.html"

    # Core params
    horizon: int = 40             # H: forward window length
    atr_period: int = 14
    eps: float = 1e-8
    min_atr: float = 0.0         # optional ATR floor; disabled by default per the spec

    # TREND thresholds (spec Section 7.3.1)
    eff_trend_min: float = 0.15               # Eff_{t,H} >= eff_trend_min
    abs_return_atr_trend_min: float = 5.0     # |R^ATR_{t,H}| >= abs_return_atr_trend_min

    # CHOP thresholds
    eff_chop_max: float = 0.10                # Eff_{t,H} <= eff_chop_max
    abs_return_atr_chop_max: float = 3.0      # |R^ATR_{t,H}| <= abs_return_atr_chop_max

    # Smoothing
    smooth_window: int = 24       # majority-vote smoothing window
    min_segment_len: int = 12     # minimum regime segment length
