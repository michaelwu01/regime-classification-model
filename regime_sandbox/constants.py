from __future__ import annotations

# Label encoding
TREND = 2
TRANSITION = 1
CHOP = 0

REGIME_LABELS = [TREND, TRANSITION, CHOP]
LABEL_STR = {TREND: "TREND", TRANSITION: "TRANSITION", CHOP: "CHOP"}

FEATURE_COLUMNS = [
    "feat_delta_di",       # DI+ - DI-  directional indicator separation
    "feat_slope",          # (MA_t - MA_{t-k}) / k  moving-average slope
    "feat_er",             # Kaufman efficiency ratio
    "feat_atr_z",          # z-scored ATR
    "feat_bbw",            # Bollinger bandwidth  2*k_B*sigma / SMA
    "feat_imbalance",      # signed trade imbalance over N ticks
]
