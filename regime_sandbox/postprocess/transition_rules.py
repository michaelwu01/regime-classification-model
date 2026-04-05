"""Postprocess transition rules applied after HMM decode.

Per spec:
    - No direct jumps TREND <-> CHOP; both must pass through TRANSITION.
    - Confirmation required to enter TREND or CHOP from TRANSITION.
    - Minimum time-in-state: bars_in_state >= T_min before allowing exit.
"""
from __future__ import annotations

import numpy as np

from regime_sandbox.constants import TREND, TRANSITION, CHOP
from regime_sandbox.postprocess.config import PostprocessConfig


def apply_transition_rules(
    hmm_states: np.ndarray,
    gamma: np.ndarray,
    cfg: PostprocessConfig,
) -> np.ndarray:
    """Apply confirmation + min-time-in-state + no-direct-jump rules."""
    from regime_sandbox.postprocess.hmm_decode import STATE_TO_IDX

    T = len(hmm_states)
    final = np.full(T, TRANSITION, dtype=int)

    state = TRANSITION
    bars_in_state = 0
    trend_confirm_count = 0
    chop_confirm_count = 0

    for t in range(T):
        proposed = int(hmm_states[t])

        if proposed == TREND:
            trend_confirm_count += 1
        else:
            trend_confirm_count = 0

        if proposed == CHOP:
            chop_confirm_count += 1
        else:
            chop_confirm_count = 0

        if state == TRANSITION:
            if trend_confirm_count >= cfg.trend_confirm_bars:
                state = TREND
                bars_in_state = 1
                trend_confirm_count = 0
            elif chop_confirm_count >= cfg.chop_confirm_bars:
                state = CHOP
                bars_in_state = 1
                chop_confirm_count = 0
            else:
                bars_in_state += 1

        elif state == TREND:
            if bars_in_state >= cfg.t_min and proposed != TREND:
                state = TRANSITION
                bars_in_state = 1
            else:
                bars_in_state += 1

        elif state == CHOP:
            if bars_in_state >= cfg.t_min and proposed != CHOP:
                state = TRANSITION
                bars_in_state = 1
            else:
                bars_in_state += 1

        final[t] = state

    return final
