"""Training entry point for the coffee quality benchmark project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coffee_price_predictor.config import TrainingConfig
from coffee_price_predictor.training import train_and_save_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training workflow."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a coffee quality regression model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=TrainingConfig().data_path,
        help="Path to the benchmark CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TrainingConfig().output_dir,
        help="Directory where the model artifacts will be stored.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=TrainingConfig().target_column,
        help="Target column for the regression model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TrainingConfig().test_size,
        help="Fraction of the data reserved for model evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=TrainingConfig().random_state,
        help="Random seed used by train/test splitting and model tuning.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the benchmark model and print model summary metrics."""
    args = parse_args()
    config = TrainingConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    summary = train_and_save_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
