"""
Training pipeline for regime classifier.

Usage: python -m regime_sandbox.train.run
"""
from __future__ import annotations

from regime_sandbox.train.config import TrainConfig
from regime_sandbox.train.trainer import train_and_evaluate


def main(cfg: TrainConfig | None = None) -> None:
    if cfg is None:
        cfg = TrainConfig()

    report = train_and_evaluate(cfg)
    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
