"""HMM forward-backward decode layer.

Per spec Section 7.3.4:
    Transition matrix:  A_ij = (N_ij + α) / Σ_k(N_ik + α)
    Forward recursion:  α_1(j) = π_j * b_1(j)
                        α_t(j) = b_t(j) * Σ_i α_{t-1}(i) * A_ij
    Backward recursion: β_T(j) = 1
                        β_t(i) = Σ_j A_ij * B_{t+1}(j) * β_{t+1}(j)
    Posterior:          γ_t(j) = α_t(j)*β_t(j) / Σ_k α_t(k)*β_t(k)
    Assignment:         Ŝ_t = argmax_j γ_t(j)
"""
from __future__ import annotations

import numpy as np

from regime_sandbox.constants import TREND, TRANSITION, CHOP


# Canonical state ordering for the 3×3 matrices
STATES = [CHOP, TRANSITION, TREND]  # indices 0, 1, 2
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)


def _uniform_distribution() -> np.ndarray:
    return np.full(N_STATES, 1.0 / N_STATES, dtype=float)


def estimate_transition_matrix(
    labels: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Estimate transition matrix A from a label sequence with Laplace smoothing.

    A_ij = (N_ij + α) / Σ_k(N_ik + α),  α > 0
    """
    if labels is None or len(labels) == 0:
        return np.full((N_STATES, N_STATES), 1.0 / N_STATES, dtype=float)

    counts = np.zeros((N_STATES, N_STATES), dtype=float)
    for t in range(len(labels) - 1):
        i = STATE_TO_IDX[int(labels[t])]
        j = STATE_TO_IDX[int(labels[t + 1])]
        counts[i, j] += 1

    A = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * N_STATES)
    return A


def estimate_initial_distribution(labels: np.ndarray) -> np.ndarray:
    """Estimate initial state distribution π from label frequencies."""
    if labels is None or len(labels) == 0:
        return _uniform_distribution()

    pi = np.zeros(N_STATES)
    for s in range(N_STATES):
        pi[s] = (labels == STATES[s]).sum()
    total = pi.sum()
    if total <= 0:
        return _uniform_distribution()
    pi = pi / total
    return pi


def forward_backward(
    emission_probs: np.ndarray,
    A: np.ndarray,
    pi: np.ndarray,
    eps: float = 1e-300,
) -> np.ndarray:
    """Run scaled forward-backward algorithm.

    Args:
        emission_probs: shape (T, N_STATES) — b_t(j) = q_t(j) from the classifier.
        A: shape (N_STATES, N_STATES) — transition matrix.
        pi: shape (N_STATES,) — initial state distribution.
        eps: floor for numerical stability.

    Returns:
        gamma: shape (T, N_STATES) — posterior probabilities γ_t(j).
    """
    T = emission_probs.shape[0]
    if T == 0:
        return np.empty((0, N_STATES), dtype=float)

    alpha = np.zeros((T, N_STATES))
    beta = np.zeros((T, N_STATES))
    scale = np.zeros(T)

    # Forward
    alpha[0] = pi * emission_probs[0]
    scale[0] = alpha[0].sum() + eps
    alpha[0] /= scale[0]

    for t in range(1, T):
        alpha[t] = emission_probs[t] * (alpha[t - 1] @ A)
        scale[t] = alpha[t].sum() + eps
        alpha[t] /= scale[t]

    # Backward
    beta[T - 1] = 1.0

    for t in range(T - 2, -1, -1):
        beta[t] = A @ (emission_probs[t + 1] * beta[t + 1])
        beta[t] /= scale[t + 1] + eps

    # Posterior
    gamma = alpha * beta
    gamma_sum = gamma.sum(axis=1, keepdims=True)
    gamma_sum = np.maximum(gamma_sum, eps)
    gamma /= gamma_sum

    return gamma


def hmm_decode(
    emission_probs: np.ndarray,
    train_labels: np.ndarray | None = None,
    alpha_smooth: float = 1.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Full HMM decode pipeline.

    Returns:
        gamma: shape (T, 3) — posterior γ_t(j).
        states: shape (T,) — Ŝ_t = argmax_j γ_t(j), mapped back to label ints.
    """
    if emission_probs.shape[0] == 0:
        return (
            np.empty((0, N_STATES), dtype=float),
            np.empty(0, dtype=int),
        )

    if train_labels is None:
        A = np.full((N_STATES, N_STATES), 1.0 / N_STATES, dtype=float)
        pi = _uniform_distribution()
    else:
        A = estimate_transition_matrix(train_labels, alpha=alpha_smooth)
        pi = estimate_initial_distribution(train_labels)

    gamma = forward_backward(emission_probs, A, pi, eps=eps)

    state_indices = np.argmax(gamma, axis=1)
    states = np.array([STATES[idx] for idx in state_indices])

    return gamma, states
