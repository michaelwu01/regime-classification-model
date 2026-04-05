"""Compute offline regime labels from future price behavior.

Per spec Section 7.3.1:
    Eff_{t,H}   = |P_{t+H} - P_t| / (PathLen_{t,H} + ε)
    R^ATR_{t,H} = (P_{t+H} - P_t) / (ATR_t + ε)

    TREND:      Eff >= eff_trend_min  AND  |R^ATR| >= abs_return_atr_trend_min
    CHOP:       Eff <= eff_chop_max   AND  |R^ATR| <= abs_return_atr_chop_max
    TRANSITION: otherwise
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from nautilus_trader.indicators.volatility import AverageTrueRange
from nautilus_trader.indicators.averages import MovingAverageType

from regime_sandbox.constants import TREND, TRANSITION, CHOP
from regime_sandbox.label.config import RegimeLabelConfig


def _compute_atr(highs, lows, closes, period: int) -> np.ndarray:
    """Compute ATR using NautilusTrader's AverageTrueRange (Wilder smoothing)."""
    atr = AverageTrueRange(period, MovingAverageType.WILDER)
    values = np.full(len(highs), np.nan)
    for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
        atr.update_raw(h, l, c)
        if atr.initialized:
            values[i] = atr.value
    return values


def compute_raw_labels(df: pd.DataFrame, cfg: RegimeLabelConfig) -> pd.DataFrame:
    """Compute regime labels for each bar using 2-criteria rule.

    Adds columns: atr, efficiency, abs_r_atr, raw_label.
    """
    N = len(df)
    H = cfg.horizon
    eps = cfg.eps

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # ATR
    atr_vals = _compute_atr(high, low, close, cfg.atr_period)
    df["atr"] = atr_vals

    # Initialize feature columns
    efficiency = np.full(N, np.nan)
    abs_r_atr = np.full(N, np.nan)
    raw_label = np.full(N, np.nan)

    # Valid labeling range
    atr_ready = cfg.atr_period - 1
    last_valid = N - H

    if last_valid <= atr_ready:
        print("WARNING: not enough bars to label any data")
        df["efficiency"] = efficiency
        df["abs_r_atr"] = abs_r_atr
        df["raw_label"] = raw_label
        return df

    # Vectorized computation using sliding windows
    windows = sliding_window_view(close, H + 1)

    start_prices = windows[:, 0]
    end_prices = windows[:, H]

    # PathLen_{t,H} = sum_{i=1}^{H} |P_{t+i} - P_{t+i-1}|
    step_diffs = np.abs(np.diff(windows, axis=1))
    path_len = step_diffs.sum(axis=1)

    # Eff_{t,H} = |P_{t+H} - P_t| / (PathLen_{t,H} + ε)
    net_move = np.abs(end_prices - start_prices)
    eff = net_move / (path_len + eps)

    # R^ATR_{t,H} = (P_{t+H} - P_t) / (ATR_t + ε)
    displacement = end_prices - start_prices
    atr_slice = atr_vals[:last_valid]
    r_atr = displacement / (atr_slice + eps)
    abs_r = np.abs(r_atr)

    # Build valid mask
    atr_valid = np.isfinite(atr_vals[:last_valid]) & (atr_vals[:last_valid] >= cfg.min_atr)
    full_valid = np.zeros(N, dtype=bool)
    full_valid[atr_ready:last_valid] = atr_valid[atr_ready:last_valid]

    # Store features only for valid bars
    valid_indices = np.where(full_valid)[0]
    efficiency[valid_indices] = eff[valid_indices]
    abs_r_atr[valid_indices] = abs_r[valid_indices]

    # Label assignment
    n_valid = len(valid_indices)
    if n_valid == 0:
        print("WARNING: no bars satisfied the ATR and horizon requirements for labeling")
        df["efficiency"] = efficiency
        df["abs_r_atr"] = abs_r_atr
        df["raw_label"] = raw_label
        return df

    v_eff = eff[valid_indices]
    v_abs_r = abs_r[valid_indices]

    is_trend = (v_eff >= cfg.eff_trend_min) & (v_abs_r >= cfg.abs_return_atr_trend_min)
    is_chop = (v_eff <= cfg.eff_chop_max) & (v_abs_r <= cfg.abs_return_atr_chop_max)

    labels = np.full(len(valid_indices), TRANSITION)
    labels[is_chop] = CHOP
    labels[is_trend] = TREND  # TREND takes precedence
    raw_label[valid_indices] = labels

    df["efficiency"] = efficiency
    df["abs_r_atr"] = abs_r_atr
    df["raw_label"] = raw_label

    n_skipped = N - n_valid
    n_trend = (labels == TREND).sum()
    n_chop = (labels == CHOP).sum()
    n_trans = (labels == TRANSITION).sum()
    print(
        f"Raw labels ({n_valid:,} valid, {n_skipped:,} skipped): "
        f"TREND={n_trend} ({n_trend/n_valid:.1%}), "
        f"CHOP={n_chop} ({n_chop/n_valid:.1%}), "
        f"TRANSITION={n_trans} ({n_trans/n_valid:.1%})"
    )
    return df
