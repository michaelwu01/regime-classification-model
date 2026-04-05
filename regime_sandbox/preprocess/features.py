"""Compute 6 causal feature columns from bar + tick data.

Feature vector per the spec (Section 7.1):
    x_t = [ΔDI_t, slope_{k,t}, ER_{t,L}, z_{ATR,t}, BBW_t, Imb_{N,t}]
"""
from __future__ import annotations

from collections import deque
from math import sqrt

import numpy as np
import pandas as pd

from nautilus_trader.indicators.volatility import AverageTrueRange, BollingerBands
from nautilus_trader.indicators.averages import (
    ExponentialMovingAverage,
    MovingAverageType,
)
from nautilus_trader.indicators.trend import DirectionalMovement
from nautilus_trader.indicators.momentum import EfficiencyRatio

from regime_sandbox.constants import FEATURE_COLUMNS
from regime_sandbox.preprocess.config import PreprocessConfig


class _RollingZScore:
    """Online rolling z-score using a deque-based window."""

    def __init__(self, window: int, eps: float = 1e-8):
        self._window = window
        self._eps = eps
        self._buf: deque[float] = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0

    @property
    def ready(self) -> bool:
        return len(self._buf) == self._window

    def update(self, value: float) -> float | None:
        if len(self._buf) == self._window:
            old = self._buf[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._buf.append(value)
        self._sum += value
        self._sum_sq += value * value
        if not self.ready:
            return None
        n = self._window
        mean = self._sum / n
        var = self._sum_sq / n - mean * mean
        std = sqrt(max(var, 0.0))
        return (value - mean) / (std + self._eps)


def _maybe_clip(value, clip: float | None):
    if clip is None:
        return value
    return np.clip(value, -clip, clip)


def _precompute_tick_imbalance(
    ticks_df: pd.DataFrame,
    bar_timestamps: np.ndarray,
    N: int,
) -> np.ndarray:
    """Precompute signed trade imbalance for each bar.

    Imb_{N,t} = sum(sgn(Δp_i) * v_i) / sum(v_i)
    over the most recent N ticks up to bar time t.

    Args:
        ticks_df: DataFrame with columns [timestamp, price, volume], sorted by timestamp.
        bar_timestamps: array of bar timestamps (pd.Timestamp).
        N: number of recent ticks for imbalance window.

    Returns:
        Array of imbalance values aligned to bars, NaN where insufficient ticks
        or when tick data is unavailable.
    """
    n_bars = len(bar_timestamps)
    imbalance = np.full(n_bars, np.nan)

    if ticks_df is None or len(ticks_df) == 0:
        return imbalance

    tick_ts = ticks_df["timestamp"].values
    tick_price = ticks_df["price"].values.astype(float)
    tick_vol = ticks_df["volume"].values.astype(float)

    # Compute Δp and sgn(Δp) for all ticks
    delta_p = np.diff(tick_price, prepend=tick_price[0])
    sign_dp = np.sign(delta_p)

    # For each bar, find the ticks up to that bar's timestamp
    tick_idx = 0
    n_ticks = len(tick_ts)

    for bar_i in range(n_bars):
        bar_t = bar_timestamps[bar_i]
        # Advance tick pointer to include all ticks <= bar_t
        while tick_idx < n_ticks and tick_ts[tick_idx] <= bar_t:
            tick_idx += 1

        # Window: last N ticks ending at tick_idx
        start = max(0, tick_idx - N)
        end = tick_idx

        if end - start < 2:  # need at least 2 ticks
            continue

        window_sign = sign_dp[start:end]
        window_vol = tick_vol[start:end]

        vol_sum = window_vol.sum()
        if vol_sum > 0:
            imbalance[bar_i] = (window_sign * window_vol).sum() / vol_sum

    return imbalance


def compute_features(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    ticks_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute 6 causal features for each bar.

    Features:
        1. feat_delta_di   = DI+ - DI-           directional indicator separation
        2. feat_slope      = (MA_t - MA_{t-k})/k  moving-average slope (signed)
        3. feat_er         = Kaufman efficiency ratio
        4. feat_atr_z      = z-scored ATR
        5. feat_bbw        = 2*k_B*σ / SMA        raw Bollinger bandwidth
        6. feat_imbalance  = signed trade imbalance over N ticks

    Returns a new DataFrame with columns: timestamp + FEATURE_COLUMNS.
    """
    N_bars = len(df)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    clip = cfg.feature_clip
    eps = cfg.eps

    # Pre-allocate output arrays
    delta_di = np.full(N_bars, np.nan)
    slope = np.full(N_bars, np.nan)
    er_arr = np.full(N_bars, np.nan)
    atr_z = np.full(N_bars, np.nan)
    bbw = np.full(N_bars, np.nan)

    # Initialize NautilusTrader indicators
    dm = DirectionalMovement(cfg.dm_period)
    atr = AverageTrueRange(cfg.atr_period, MovingAverageType.WILDER)
    ema = ExponentialMovingAverage(cfg.ma_period)
    er = EfficiencyRatio(cfg.er_lookback)
    bb = BollingerBands(cfg.bbw_period, cfg.bbw_k)

    # Rolling z-score for ATR
    atr_z_proc = _RollingZScore(cfg.atr_z_window, eps)

    # Buffer for MA slope lookback
    ema_history: deque[float] = deque(maxlen=cfg.slope_lookback + 1)

    for i in range(N_bars):
        h, l, c = float(high[i]), float(low[i]), float(close[i])

        # Update indicators
        dm.update_raw(h, l)
        atr.update_raw(h, l, c)
        ema.update_raw(c)
        er.update_raw(c)
        bb.update_raw(h, l, c)

        if ema.initialized:
            ema_history.append(ema.value)

        if not (dm.initialized and atr.initialized):
            continue

        # Feature 1: ΔDI_t = DI+ - DI-
        di_pos = dm.pos
        di_neg = dm.neg
        delta_di[i] = _maybe_clip(di_pos - di_neg, clip)

        # Feature 2: slope_{k,t} = (MA_t - MA_{t-k}) / k   (signed)
        if len(ema_history) == cfg.slope_lookback + 1:
            slope_raw = (ema_history[-1] - ema_history[0]) / cfg.slope_lookback
            slope[i] = _maybe_clip(slope_raw, clip)

        # Feature 3: Efficiency Ratio
        if er.initialized:
            er_arr[i] = _maybe_clip(er.value, clip)

        # Feature 4: z_{ATR,t} = (ATR_t - μ(ATR)) / σ(ATR)
        z_val = atr_z_proc.update(atr.value)
        if z_val is not None:
            atr_z[i] = _maybe_clip(z_val, clip)

        # Feature 5: BBW_t = 2 * k_B * σ_t / SMA_t   (raw bandwidth)
        if bb.initialized:
            sma_val = bb.middle
            if sma_val > eps:
                bbw_val = (bb.upper - bb.lower) / sma_val  # = 2*k_B*σ / SMA
                bbw[i] = _maybe_clip(bbw_val, clip)

    # Feature 6: Imb_{N,t} = signed trade imbalance
    bar_timestamps = pd.to_datetime(df["timestamp"].values)
    imbalance = _precompute_tick_imbalance(
        ticks_df, bar_timestamps.values, cfg.imbalance_N
    )
    imbalance = _maybe_clip(imbalance, clip)

    result = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        FEATURE_COLUMNS[0]: delta_di,
        FEATURE_COLUMNS[1]: slope,
        FEATURE_COLUMNS[2]: er_arr,
        FEATURE_COLUMNS[3]: atr_z,
        FEATURE_COLUMNS[4]: bbw,
        FEATURE_COLUMNS[5]: imbalance,
    })
    return result
