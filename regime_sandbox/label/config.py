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

    # TREND thresholds (OPTIMIZED - see THRESHOLD_OPTIMIZATION_RESULTS.md)
    eff_trend_min: float = 0.30               # Eff_{t,H} >= eff_trend_min (was 0.15)
    abs_return_atr_trend_min: float = 3.0     # |R^ATR_{t,H}| >= abs_return_atr_trend_min (was 5.0)

    # CHOP thresholds (OPTIMIZED - see THRESHOLD_OPTIMIZATION_RESULTS.md)
    eff_chop_max: float = 0.07                # Eff_{t,H} <= eff_chop_max (was 0.10)
    abs_return_atr_chop_max: float = 0.5      # |R^ATR_{t,H}| <= abs_return_atr_chop_max (was 3.0)

    # Smoothing
    smooth_window: int = 24       # majority-vote smoothing window
    min_segment_len: int = 12     # minimum regime segment length
