from __future__ import annotations

import pandas as pd

from qa_ftopsis.experiment import run_suite
from tests.conftest import build_synthetic_dataset, build_test_config


def test_run_suite_end_to_end_smoke(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path / "tickets.csv", per_group=12)
    config_path = build_test_config(tmp_path / "config.yaml", dataset_path)

    run_dir = run_suite(config_path)

    assert (run_dir / "raw_metrics.csv").exists()
    assert (run_dir / "decisions.parquet").exists()
    assert (run_dir / "summary_metrics.csv").exists()
    assert (run_dir / "config_snapshot.yaml").exists()

    summary = pd.read_csv(run_dir / "summary_metrics.csv")
    assert not summary.empty
    assert summary["avg_cost_mean"].notna().all()
    assert summary["p99_wait_mean"].notna().all()
    assert "capacity_mode" in summary.columns
    assert set(summary["environment_id"].unique()) == {
        "deterministic_redundant",
        "deterministic_embedding",
        "heavytail_redundant",
        "heavytail_embedding",
    }
    assert {"jsq_topk", "qa_ftopsis_topk"} <= set(summary["policy"].unique())

    report_dir = tmp_path / "artifacts" / "reports" / run_dir.name
    assert (report_dir / "summary_metrics.csv").exists()
    assert (report_dir / "relative_improvement.csv").exists()
    assert (report_dir / "wait_cdf.csv").exists()
    assert (report_dir / "class_balance_summary.csv").exists()
    assert (report_dir / "top_confusion_pairs.csv").exists()

    relative = pd.read_csv(report_dir / "relative_improvement.csv")
    assert "environment_id" in relative.columns
    assert "capacity_mode" in relative.columns
