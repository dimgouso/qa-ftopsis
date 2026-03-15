from __future__ import annotations

import json

import pandas as pd

from qa_ftopsis.config import load_config
from qa_ftopsis.data import prepare_dataset
from qa_ftopsis.delay_models import delay_model_path
from qa_ftopsis.experiment import build_jira_benchmark_command, run_suite
from qa_ftopsis.models import feature_split_path, train_classifier
from qa_ftopsis.skill_features import build_skill_features
from tests.conftest import build_jira_test_config, build_synthetic_jira_exports


def test_build_jira_benchmark_outputs_expected_artifacts(tmp_path):
    issues_path, history_path = build_synthetic_jira_exports(tmp_path, per_queue=10)
    config_path = build_jira_test_config(tmp_path / "jira_config.yaml", issues_path, history_path)
    config = load_config(config_path)

    result = build_jira_benchmark_command(config)

    benchmark_path = config.paths.input_csv
    assert benchmark_path.exists()
    assert (config.jira.benchmark_dir / "jira_issue_history.parquet").exists()
    assert (config.jira.benchmark_dir / "queue_distribution.csv").exists()
    assert (config.jira.benchmark_dir / "reroute_stats.csv").exists()
    assert json.loads((config.jira.benchmark_dir / "benchmark_metadata.json").read_text())["num_queues"] == 4
    assert result["num_queues"] == 4

    benchmark_df = pd.read_parquet(benchmark_path)
    assert {"issue_id", "service_units", "time_to_resolution_hours", "num_queue_changes"} <= set(
        benchmark_df.columns
    )
    assert benchmark_df["queue"].nunique() == 4
    assert benchmark_df["service_units"].min() >= 1


def test_jira_public_end_to_end_smoke(tmp_path):
    issues_path, history_path = build_synthetic_jira_exports(tmp_path, per_queue=10)
    config_path = build_jira_test_config(tmp_path / "jira_config.yaml", issues_path, history_path)
    config = load_config(config_path)

    build_jira_benchmark_command(config)
    prepare_dataset(
        config.paths.input_csv,
        config.paths.prepared_dir,
        random_state=config.data.random_state,
        dataset_profile=config.data.dataset_profile,
        priority_mapping=config.data.priority_mapping,
        stratify_fields=config.data.stratify_fields,
        expected_languages=config.data.expected_languages,
    )
    train_classifier(config)
    feature_splits = {
        split_name: pd.read_parquet(feature_split_path(config.paths.model_dir, split_name))
        for split_name in ["train", "val_cal", "val_sim", "test"]
    }
    build_skill_features(config, feature_splits)

    run_dir = run_suite(config_path)

    assert delay_model_path(config.paths.model_dir).exists()
    train_features = pd.read_parquet(feature_split_path(config.paths.model_dir, "train"))
    assert any(column.startswith("delay_q_") for column in train_features.columns)

    summary = pd.read_csv(run_dir / "summary_metrics.csv")
    assert set(summary["environment_id"].unique()) == {"empirical_embedding", "empirical_learnedjira"}
    assert {"jsq_topk", "qa_ftopsis_topk"} <= set(summary["policy"].unique())

    report_dir = config.paths.report_root / run_dir.name
    assert (report_dir / "queue_distribution.csv").exists()
    assert (report_dir / "reroute_stats.csv").exists()
    assert (report_dir / "benchmark_metadata.json").exists()


def test_build_jira_benchmark_component_combo_creates_macro_groups(tmp_path):
    issues_path, history_path = build_synthetic_jira_exports(tmp_path, per_queue=10)
    config_path = build_jira_test_config(
        tmp_path / "jira_combo_config.yaml",
        issues_path,
        history_path,
        queue_field="component_combo",
    )
    config = load_config(config_path)

    build_jira_benchmark_command(config)

    benchmark_df = pd.read_parquet(config.paths.input_csv)
    metadata = json.loads((config.jira.benchmark_dir / "benchmark_metadata.json").read_text())
    graph = pd.read_csv(config.jira.benchmark_dir / "queue_transition_graph.csv")
    macro_groups = pd.read_csv(config.jira.benchmark_dir / "macro_groups.csv")

    assert benchmark_df["queue"].str.contains(r"\+").any()
    assert not graph.empty
    assert {"issue_overlap_count", "token_overlap_count"} <= set(graph.columns)
    assert metadata["queue_field"] == "component_combo"
    assert metadata["num_macro_groups"] < benchmark_df["queue"].nunique()
    assert macro_groups["macro_group_id"].nunique() < macro_groups["queue_id"].nunique()


def test_build_jira_benchmark_quantile_spread_service_units(tmp_path):
    issues_path, history_path = build_synthetic_jira_exports(tmp_path, per_queue=10)
    config_path = build_jira_test_config(tmp_path / "jira_quantile_config.yaml", issues_path, history_path)
    config = load_config(config_path)
    config.jira.service_unit_mode = "quantile_spread"

    build_jira_benchmark_command(config)

    benchmark_df = pd.read_parquet(config.paths.input_csv)
    metadata = json.loads((config.jira.benchmark_dir / "benchmark_metadata.json").read_text())

    assert "raw_service_units" in benchmark_df.columns
    assert benchmark_df["service_units"].max() <= config.jira.max_service_units
    assert metadata["service_unit_mode"] == "quantile_spread"
    assert benchmark_df["service_units"].nunique() > 1
    assert metadata["service_unit_quantiles"]["0.99"] <= config.jira.max_service_units
