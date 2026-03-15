from __future__ import annotations

import json

import pandas as pd

from qa_ftopsis.data import load_queue_lookup, prepare_dataset
from tests.conftest import build_synthetic_dataset


def test_prepare_dataset_outputs_expected_splits(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path / "tickets.csv")
    prepared_dir = tmp_path / "prepared"
    metadata = prepare_dataset(dataset_path, prepared_dir, random_state=7)

    assert metadata["split_sizes"] == {
        "train": 43,
        "val_cal": 7,
        "val_sim": 7,
        "test": 15,
    }

    train_df = pd.read_parquet(prepared_dir / "train.parquet")
    assert train_df["body"].isna().sum() == 0
    assert train_df["text"].isna().sum() == 0
    assert train_df["ticket_id"].nunique() == len(train_df)

    queue_lookup = load_queue_lookup(prepared_dir)
    assert queue_lookup["queue_name"].tolist() == sorted(queue_lookup["queue_name"].tolist())
    assert queue_lookup["queue_id"].tolist() == list(range(len(queue_lookup)))
    assert (prepared_dir / "split_queue_stats.csv").exists()

    saved_metadata = json.loads((prepared_dir / "metadata.json").read_text())
    assert set(saved_metadata["min_queue_count_by_split"]) == {"train", "val_cal", "val_sim", "test"}
    assert set(saved_metadata["median_queue_count_by_split"]) == {"train", "val_cal", "val_sim", "test"}
    assert set(saved_metadata["max_queue_count_by_split"]) == {"train", "val_cal", "val_sim", "test"}


def test_prepare_dataset_collapses_priority_and_uses_configured_stratification(tmp_path):
    dataset_path = tmp_path / "german42_like.csv"
    df = pd.DataFrame(
        [
            {
                "subject": f"subject {idx}",
                "body": f"body {idx}",
                "queue": f"queue_{idx % 3}",
                "priority": priority,
                "language": "de",
            }
            for idx, priority in enumerate(
                ["very_low", "low", "medium", "high", "critical"] * 6
            )
        ]
    )
    df.to_csv(dataset_path, index=False)

    prepared_dir = tmp_path / "prepared"
    prepare_dataset(
        dataset_path,
        prepared_dir,
        random_state=7,
        dataset_profile="german42",
        priority_mapping={
            "very_low": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "critical": "high",
        },
        stratify_fields=["queue"],
        expected_languages=["de"],
    )

    train_df = pd.read_parquet(prepared_dir / "train.parquet")
    assert "priority_raw" in train_df.columns
    assert set(train_df["priority"].unique()) <= {"low", "medium", "high"}

    stats_df = pd.read_csv(prepared_dir / "split_queue_stats.csv")
    assert set(stats_df.columns) == {"split", "queue_id", "queue_name", "count", "share"}
