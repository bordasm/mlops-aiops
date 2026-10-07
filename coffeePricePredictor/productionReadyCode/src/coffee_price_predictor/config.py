"""Configuration for the coffee quality benchmark training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values used by the benchmark training pipeline."""

    data_path: Path = (
        Path(__file__).resolve().parents[3] / "data" / "coffee_value_benchmark.csv"
    )
    output_dir: Path = Path(__file__).resolve().parents[2] / "artifacts"
    target_column: str = "coffee_quality_score"
    test_size: float = 0.2
    random_state: int = 42
    leakage_columns: tuple[str, ...] = (
        "record_id",
        "lot_id",
        "quality_grade",
        "price_usd_per_kg",
    )
