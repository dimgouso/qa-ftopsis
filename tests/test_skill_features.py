from __future__ import annotations

import json

import pandas as pd

from qa_ftopsis.config import load_config
from qa_ftopsis.data import prepare_dataset
from qa_ftopsis.experiment import build_skill_features_command
from qa_ftopsis.models import train_classifier
from qa_ftopsis.skill_features import embedding_metadata_path
from tests.conftest import build_synthetic_dataset, build_test_config


def test_build_skill_features_adds_similarity_and_kappa_columns(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path / "tickets.csv")
    config_path = build_test_config(tmp_path / "config.yaml", dataset_path)
    config = load_config(config_path)

    prepare_dataset(config.paths.input_csv, config.paths.prepared_dir, random_state=config.data.random_state)
    train_classifier(config)
    build_skill_features_command(config)

    train_df = pd.read_parquet(config.paths.model_dir / "features" / "train.parquet")
    sim_columns = [column for column in train_df.columns if column.startswith("sim_q_")]
    kappa_columns = [column for column in train_df.columns if column.startswith("kappa_q_")]
    assert sim_columns
    assert kappa_columns
    assert "p_max" in train_df.columns
    probability_columns = [column for column in train_df.columns if column.startswith("prob_q_")]
    assert train_df["p_max"].round(8).equals(train_df[probability_columns].max(axis=1).round(8))
    metadata = json.loads(embedding_metadata_path(config.paths.model_dir).read_text())
    assert metadata["centroid_source_split"] == "train"
    assert metadata["num_centroids"] == len(sim_columns)
    first_row = train_df.iloc[0]
    assert len(set(round(float(first_row[column]), 6) for column in kappa_columns)) > 1
