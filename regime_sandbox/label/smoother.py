from __future__ import annotations

import numpy as np

from regime_sandbox.constants import TRANSITION
from regime_sandbox.label.config import RegimeLabelConfig


def _majority_vote(window: np.ndarray, num_classes: int = 3) -> int:
    """Return majority label in window. Tie -> TRANSITION."""
    counts = np.bincount(window.astype(int), minlength=num_classes)
    max_count = counts.max()
    winners = np.where(counts == max_count)[0]
    if len(winners) == 1:
        return int(winners[0])
    return TRANSITION


def majority_vote_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    """Apply centered majority-vote smoothing."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}.")

    N = len(labels)
    half = window // 2
    result = np.empty(N, dtype=int)

    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + (window % 2))
        result[i] = _majority_vote(labels[lo:hi])

    return result


def enforce_min_segment(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Replace short segments with neighbor label or TRANSITION."""
    result = labels.copy()
    max_iters = 50

    for _ in range(max_iters):
        changed = False
        # Find run boundaries
        diffs = np.diff(result)
        boundaries = np.where(diffs != 0)[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(result)]))

        for seg_start, seg_end in zip(starts, ends):
            seg_len = seg_end - seg_start
            if seg_len >= min_len:
                continue

            left_label = result[seg_start - 1] if seg_start > 0 else None
            right_label = result[seg_end] if seg_end < len(result) else None

            if left_label is not None and right_label is not None and left_label == right_label:
                new_label = left_label
            else:
                new_label = TRANSITION

            if result[seg_start] != new_label:
                result[seg_start:seg_end] = new_label
                changed = True

        if not changed:
            break

    return result


def smooth_labels(raw_labels: np.ndarray, cfg: RegimeLabelConfig) -> np.ndarray:
    """Apply majority-vote smoothing then minimum segment enforcement."""
    smoothed = majority_vote_smooth(raw_labels, cfg.smooth_window)
    cleaned = enforce_min_segment(smoothed, cfg.min_segment_len)
    return cleaned
