"""
HMM decode + transition rules postprocessing.

Usage: python -m regime_sandbox.postprocess.run
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from regime_sandbox.constants import FEATURE_COLUMNS, LABEL_STR
from regime_sandbox.postprocess.config import PostprocessConfig
from regime_sandbox.postprocess.hmm_decode import hmm_decode, STATES
from regime_sandbox.postprocess.transition_rules import apply_transition_rules


def _empty_result_df() -> pd.DataFrame:
    result_df = pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns]"),
        "close": pd.Series(dtype=float),
        "hmm_state_int": pd.Series(dtype=int),
        "hmm_state_str": pd.Series(dtype=str),
        "final_state_int": pd.Series(dtype=int),
        "final_state_str": pd.Series(dtype=str),
    })
    for state in STATES:
        result_df[f"gamma_{LABEL_STR[state].lower()}"] = pd.Series(dtype=float)
    return result_df


def main(cfg: PostprocessConfig | None = None) -> None:
    if cfg is None:
        cfg = PostprocessConfig()

    model_dir = Path(cfg.model_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model and scaler
    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    classes = model.classes_.tolist()
    print(f"Loaded model (classes={[LABEL_STR.get(c, c) for c in classes]})")

    # 2. Load features
    features_df = pd.read_csv(cfg.features_csv)
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True).dt.tz_localize(None)
    features_df.sort_values("timestamp", inplace=True)
    features_df.reset_index(drop=True, inplace=True)

    # 3. Load labels CSV for aligned price data
    labels_df = pd.read_csv(cfg.labels_csv)
    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], utc=True).dt.tz_localize(None)
    price_df = pd.merge(
        features_df[["timestamp"]],
        labels_df[["timestamp", "open", "high", "low", "close", "volume"]],
        on="timestamp",
        how="left",
    )

    hmm_train_labels_path = model_dir / "hmm_train_labels.npy"
    if hmm_train_labels_path.exists():
        train_label_arr = np.load(hmm_train_labels_path).astype(int)
    else:
        print(
            "WARNING: Missing training HMM labels artifact. "
            "Falling back to uniform HMM priors."
        )
        train_label_arr = None

    # 4. Drop NaN feature rows
    valid_mask = features_df[FEATURE_COLUMNS].notna().all(axis=1)
    features_valid = features_df[valid_mask].copy()
    features_valid.reset_index(drop=True, inplace=True)
    price_valid = price_df[valid_mask].copy()
    price_valid.reset_index(drop=True, inplace=True)

    print(f"Valid feature rows: {len(features_valid):,}")
    if features_valid.empty:
        print("WARNING: no valid feature rows are available for postprocessing")
        csv_path = output_dir / "final_states.csv"
        _empty_result_df().to_csv(csv_path, index=False)
        print(f"Saved final states CSV to {csv_path}")
        print("\nPostprocessing complete.")
        return

    # 5. Scale features and get emission probabilities
    X = features_valid[FEATURE_COLUMNS].values
    X_scaled = scaler.transform(X)
    raw_probs = model.predict_proba(X_scaled)

    # Reorder columns to match STATES = [CHOP, TRANSITION, TREND]
    emission_probs = np.full((len(raw_probs), len(STATES)), cfg.eps, dtype=float)
    for i, state in enumerate(STATES):
        if state in classes:
            src_idx = classes.index(state)
            emission_probs[:, i] = raw_probs[:, src_idx]
    emission_probs /= np.maximum(
        emission_probs.sum(axis=1, keepdims=True),
        cfg.eps,
    )

    # 6. HMM decode
    gamma, hmm_states = hmm_decode(
        emission_probs, train_label_arr,
        alpha_smooth=cfg.alpha_smooth, eps=cfg.eps,
    )

    # 7. Apply transition rules
    final_states = apply_transition_rules(hmm_states, gamma, cfg)

    # 8. Save CSV
    result_df = pd.DataFrame({
        "timestamp": features_valid["timestamp"].values,
        "close": price_valid["close"].values,
        "hmm_state_int": hmm_states,
        "hmm_state_str": [LABEL_STR.get(s, "UNKNOWN") for s in hmm_states],
        "final_state_int": final_states,
        "final_state_str": [LABEL_STR.get(s, "UNKNOWN") for s in final_states],
    })
    for i, state in enumerate(STATES):
        result_df[f"gamma_{LABEL_STR[state].lower()}"] = gamma[:, i]

    csv_path = output_dir / "final_states.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved final states CSV to {csv_path}")

    print("\nFinal states:")
    for label_int, label_str in LABEL_STR.items():
        count = (final_states == label_int).sum()
        frac = count / len(final_states) if len(final_states) > 0 else 0
        print(f"  {label_str}: {count} ({frac:.1%})")

    print("\nPostprocessing complete.")


if __name__ == "__main__":
    main()
