from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def _preferred_environment(decisions_df: pd.DataFrame) -> str | None:
    primary = decisions_df[
        (decisions_df["split"] == "test") & (decisions_df["sla_profile"] == "primary")
    ]
    if primary.empty:
        return None
    available = primary["environment_id"].dropna().astype(str).unique().tolist()
    for preferred in [
        "heavytail_embedding",
        "empirical_learnedjira",
        "empirical_embedding",
        "deterministic_embedding",
    ]:
        if preferred in available:
            return preferred
    return sorted(available)[0]


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics_df.groupby(
        [
            "split",
            "sla_profile",
            "environment_id",
            "service_model",
            "delay_mode",
            "capacity_mode",
            "scenario",
            "policy",
        ],
        dropna=False,
    )
    summary = grouped.agg(
        mean_wait_mean=("mean_wait", "mean"),
        mean_wait_std=("mean_wait", "std"),
        mean_wait_median=("mean_wait", "median"),
        p95_wait_mean=("p95_wait", "mean"),
        p95_wait_std=("p95_wait", "std"),
        p95_wait_median=("p95_wait", "median"),
        p99_wait_mean=("p99_wait", "mean"),
        p99_wait_std=("p99_wait", "std"),
        p99_wait_median=("p99_wait", "median"),
        mean_backlog_mean=("mean_backlog", "mean"),
        mean_backlog_std=("mean_backlog", "std"),
        mean_backlog_median=("mean_backlog", "median"),
        sla_violation_rate_mean=("sla_violation_rate", "mean"),
        sla_violation_rate_std=("sla_violation_rate", "std"),
        sla_violation_rate_median=("sla_violation_rate", "median"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_median=("accuracy", "median"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        macro_f1_median=("macro_f1", "median"),
        misroute_rate_mean=("misroute_rate", "mean"),
        misroute_rate_std=("misroute_rate", "std"),
        misroute_rate_median=("misroute_rate", "median"),
        avg_cost_mean=("avg_cost", "mean"),
        avg_cost_std=("avg_cost", "std"),
        avg_cost_median=("avg_cost", "median"),
    )
    return summary.reset_index()


def relative_improvement_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    primary = summary_df[
        (summary_df["split"] == "test") & (summary_df["sla_profile"] == "primary")
    ].copy()
    baseline = primary[primary["policy"] == "classifier_only"].rename(
        columns={
            "avg_cost_median": "baseline_avg_cost_median",
            "p95_wait_median": "baseline_p95_wait_median",
            "p99_wait_median": "baseline_p99_wait_median",
            "sla_violation_rate_median": "baseline_sla_violation_rate_median",
            "macro_f1_median": "baseline_macro_f1_median",
        }
    )
    merged = primary.merge(
        baseline[
            [
                "environment_id",
                "capacity_mode",
                "scenario",
                "baseline_avg_cost_median",
                "baseline_p95_wait_median",
                "baseline_p99_wait_median",
                "baseline_sla_violation_rate_median",
                "baseline_macro_f1_median",
            ]
        ],
        on=["environment_id", "capacity_mode", "scenario"],
        how="left",
    )
    merged["avg_cost_improvement_pct"] = (
        (merged["baseline_avg_cost_median"] - merged["avg_cost_median"])
        / merged["baseline_avg_cost_median"]
        * 100.0
    )
    merged["p95_wait_improvement_pct"] = (
        (merged["baseline_p95_wait_median"] - merged["p95_wait_median"])
        / merged["baseline_p95_wait_median"]
        * 100.0
    )
    merged["p99_wait_improvement_pct"] = (
        (merged["baseline_p99_wait_median"] - merged["p99_wait_median"])
        / merged["baseline_p99_wait_median"]
        * 100.0
    )
    merged["sla_violation_improvement_pct"] = (
        (merged["baseline_sla_violation_rate_median"] - merged["sla_violation_rate_median"])
        / merged["baseline_sla_violation_rate_median"].replace(0, pd.NA)
        * 100.0
    )
    merged["macro_f1_delta_pp"] = (
        (merged["macro_f1_median"] - merged["baseline_macro_f1_median"]) * 100.0
    )
    return merged[
        [
            "environment_id",
            "capacity_mode",
            "scenario",
            "policy",
            "avg_cost_improvement_pct",
            "p95_wait_improvement_pct",
            "p99_wait_improvement_pct",
            "sla_violation_improvement_pct",
            "macro_f1_delta_pp",
        ]
    ]


def seed_level_win_loss(metrics_df: pd.DataFrame) -> pd.DataFrame:
    primary = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["sla_profile"] == "primary")
    ]
    baseline = primary[primary["policy"] == "classifier_only"]
    candidates = primary[primary["policy"] != "classifier_only"]
    merged = candidates.merge(
        baseline,
        on=["environment_id", "capacity_mode", "scenario", "seed"],
        suffixes=("_policy", "_base"),
    )
    merged["avg_cost_improved"] = merged["avg_cost_policy"] < merged["avg_cost_base"]
    merged["p95_wait_improved"] = merged["p95_wait_policy"] < merged["p95_wait_base"]
    merged["p99_wait_improved"] = merged["p99_wait_policy"] < merged["p99_wait_base"]
    merged["macro_f1_delta_pp"] = (merged["macro_f1_policy"] - merged["macro_f1_base"]) * 100.0
    return merged[
        [
            "environment_id",
            "capacity_mode",
            "scenario",
            "policy_policy",
            "seed",
            "avg_cost_improved",
            "p95_wait_improved",
            "p99_wait_improved",
            "macro_f1_delta_pp",
        ]
    ].rename(columns={"policy_policy": "policy"})


def wait_cdf_table(decisions_df: pd.DataFrame) -> pd.DataFrame:
    primary = decisions_df[
        (decisions_df["split"] == "test") & (decisions_df["sla_profile"] == "primary")
    ].copy()
    rows: list[dict[str, float | str | int]] = []
    grouped = primary.groupby(["environment_id", "capacity_mode", "scenario", "policy"], dropna=False)
    for (environment_id, capacity_mode, scenario, policy), frame in grouped:
        counts = frame["wait_slots"].value_counts().sort_index()
        cumulative = counts.cumsum() / counts.sum()
        for wait_slot, cdf in cumulative.items():
            rows.append(
                {
                    "environment_id": environment_id,
                    "capacity_mode": capacity_mode,
                    "scenario": scenario,
                    "policy": policy,
                    "wait_slot": int(wait_slot),
                    "cdf": float(cdf),
                }
            )
    return pd.DataFrame(rows)


def _save_metric_plot(summary_df: pd.DataFrame, metric: str, destination: Path, title: str) -> None:
    primary = summary_df[
        (summary_df["split"] == "test") & (summary_df["sla_profile"] == "primary")
    ].copy()
    if primary.empty:
        return
    plot_frame = primary.copy()
    plot_frame["env_scenario"] = (
        plot_frame["environment_id"] + "\n" + plot_frame["capacity_mode"] + "\n" + plot_frame["scenario"]
    )
    plt.figure(figsize=(14, 6))
    sns.barplot(data=plot_frame, x="env_scenario", y=metric, hue="policy")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def _save_backlog_plot(backlog_df: pd.DataFrame, destination: Path) -> None:
    primary = backlog_df[backlog_df["sla_profile"] == "primary"].copy()
    if primary.empty:
        return
    aggregated = (
        primary.groupby(
            ["environment_id", "capacity_mode", "scenario", "policy", "slot"],
            as_index=False,
        )["total_backlog"].mean()
    )
    aggregated["panel"] = (
        aggregated["environment_id"] + " | " + aggregated["capacity_mode"] + " | " + aggregated["scenario"]
    )
    panels = aggregated["panel"].unique().tolist()
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4 * max(1, len(panels))), sharex=False)
    if len(panels) == 1:
        axes = [axes]
    for axis, panel in zip(axes, panels):
        subset = aggregated[aggregated["panel"] == panel]
        sns.lineplot(data=subset, x="slot", y="total_backlog", hue="policy", ax=axis)
        axis.set_title(f"Average backlog over time: {panel}")
    plt.tight_layout()
    plt.savefig(destination)
    plt.close()


def _save_wait_cdf_plots(cdf_df: pd.DataFrame, report_dir: Path) -> None:
    for scenario_name in ["high_load", "bursty"]:
        subset = cdf_df[cdf_df["scenario"] == scenario_name]
        if subset.empty:
            continue
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=subset,
            x="wait_slot",
            y="cdf",
            hue="policy",
            style="environment_id",
        )
        plt.title(f"Wait CDF: {scenario_name}")
        plt.tight_layout()
        plt.savefig(report_dir / f"wait_cdf_{scenario_name}.png")
        plt.close()


def _save_confusion_matrices(decisions_df: pd.DataFrame, destination_dir: Path) -> None:
    preferred_environment = _preferred_environment(decisions_df)
    if preferred_environment is None:
        return
    primary = decisions_df[
        (decisions_df["split"] == "test")
        & (decisions_df["sla_profile"] == "primary")
        & (decisions_df["environment_id"] == preferred_environment)
    ]
    for policy_name in ["classifier_only", "qa_ftopsis", "qa_ftopsis_topk"]:
        subset = primary[primary["policy"] == policy_name]
        if subset.empty:
            continue
        matrix = confusion_matrix(subset["true_queue_id"], subset["chosen_queue_id"])
        display = ConfusionMatrixDisplay(confusion_matrix=matrix)
        display.plot(cmap="Blues", colorbar=False)
        plt.title(f"Confusion matrix: {policy_name} ({preferred_environment})")
        plt.tight_layout()
        plt.savefig(destination_dir / f"confusion_matrix_{policy_name}.png")
        plt.close()


def _copy_class_balance_summary(run_path: Path, report_dir: Path) -> None:
    config_snapshot = run_path / "config_snapshot.yaml"
    if not config_snapshot.exists():
        return
    snapshot = yaml.safe_load(config_snapshot.read_text()) or {}
    prepared_dir = snapshot.get("paths", {}).get("prepared_dir")
    if not prepared_dir:
        return
    source = Path(prepared_dir) / "split_queue_stats.csv"
    if not source.exists():
        return
    summary = pd.read_csv(source).sort_values(["split", "count", "queue_id"], ascending=[True, False, True])
    summary.to_csv(report_dir / "class_balance_summary.csv", index=False)


def _copy_jira_benchmark_artifacts(run_path: Path, report_dir: Path) -> None:
    config_snapshot = run_path / "config_snapshot.yaml"
    if not config_snapshot.exists():
        return
    snapshot = yaml.safe_load(config_snapshot.read_text()) or {}
    benchmark_dir = snapshot.get("jira", {}).get("benchmark_dir")
    if not benchmark_dir:
        return
    benchmark_path = Path(benchmark_dir)
    for artifact_name in [
        "queue_distribution.csv",
        "reroute_stats.csv",
        "queue_transition_graph.csv",
        "macro_groups.csv",
        "benchmark_metadata.json",
    ]:
        source = benchmark_path / artifact_name
        if source.exists():
            target = report_dir / artifact_name
            if artifact_name.endswith(".json"):
                target.write_text(source.read_text())
            else:
                try:
                    pd.read_csv(source).to_csv(target, index=False)
                except pd.errors.EmptyDataError:
                    target.write_text(source.read_text())


def top_confusion_pairs(decisions_df: pd.DataFrame) -> pd.DataFrame:
    primary = decisions_df[
        (decisions_df["split"] == "test")
        & (decisions_df["sla_profile"] == "primary")
        & (decisions_df["true_queue_id"] != decisions_df["chosen_queue_id"])
    ].copy()
    if primary.empty:
        return pd.DataFrame(
            columns=[
                "environment_id",
                "capacity_mode",
                "policy",
                "true_queue_id",
                "chosen_queue_id",
                "count",
            ]
        )
    grouped = (
        primary.groupby(
            ["environment_id", "capacity_mode", "policy", "true_queue_id", "chosen_queue_id"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["environment_id", "policy", "count"], ascending=[True, True, False])
    )
    return grouped


def generate_report(
    run_dir: str | Path,
    report_root: str | Path,
    sample_size: int = 100,
) -> Path:
    run_path = Path(run_dir).resolve()
    report_dir = Path(report_root).resolve() / run_path.name
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(run_path / "raw_metrics.csv")
    decisions_df = pd.read_parquet(run_path / "decisions.parquet")
    backlog_df = pd.read_parquet(run_path / "backlog_trace.parquet")

    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(report_dir / "summary_metrics.csv", index=False)

    relative_df = relative_improvement_table(summary_df)
    relative_df.to_csv(report_dir / "relative_improvement.csv", index=False)

    seed_win_df = seed_level_win_loss(metrics_df)
    seed_win_df.to_csv(report_dir / "seed_win_loss_vs_classifier.csv", index=False)
    seed_win_df[seed_win_df["policy"] == "qa_ftopsis"].to_csv(
        report_dir / "qa_ftopsis_seed_win_loss.csv",
        index=False,
    )

    cdf_df = wait_cdf_table(decisions_df)
    cdf_df.to_csv(report_dir / "wait_cdf.csv", index=False)
    _copy_class_balance_summary(run_path, report_dir)
    _copy_jira_benchmark_artifacts(run_path, report_dir)
    top_confusion_pairs(decisions_df).to_csv(report_dir / "top_confusion_pairs.csv", index=False)

    _save_metric_plot(summary_df, "p95_wait_median", report_dir / "p95_wait_by_policy.png", "Median p95 wait by policy")
    _save_metric_plot(summary_df, "p99_wait_median", report_dir / "p99_wait_by_policy.png", "Median p99 wait by policy")
    _save_metric_plot(summary_df, "sla_violation_rate_median", report_dir / "sla_violations_by_policy.png", "Median SLA violation rate by policy")
    _save_metric_plot(summary_df, "avg_cost_median", report_dir / "avg_cost_by_policy.png", "Median average cost by policy")
    _save_metric_plot(summary_df, "macro_f1_median", report_dir / "macro_f1_by_policy.png", "Median macro-F1 by policy")
    _save_metric_plot(summary_df, "misroute_rate_median", report_dir / "misroute_rate_by_policy.png", "Median misroute rate by policy")

    _save_backlog_plot(backlog_df, report_dir / "backlog_over_time.png")
    _save_wait_cdf_plots(cdf_df, report_dir)
    _save_confusion_matrices(decisions_df, report_dir)

    preferred_environment = _preferred_environment(decisions_df)
    explainability_candidates = decisions_df[
        (decisions_df["policy"] == "qa_ftopsis")
        & (decisions_df["split"] == "test")
        & (decisions_df["sla_profile"] == "primary")
        & (decisions_df["environment_id"] == preferred_environment)
    ]
    if not explainability_candidates.empty:
        sample = explainability_candidates.sample(
            n=min(sample_size, len(explainability_candidates)),
            random_state=7,
        )
        sample.to_csv(report_dir / "qa_ftopsis_explainability_sample.csv", index=False)

    return report_dir
