from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from qa_ftopsis.config import AppConfig, load_config, save_config_snapshot
from qa_ftopsis.data import prepare_dataset
from qa_ftopsis.delay_models import delay_metrics_path, delay_model_path, train_delay_model
from qa_ftopsis.environment import build_environment_specs, ensure_environment_fixtures
from qa_ftopsis.jira_api import fetch_jira_api_export
from qa_ftopsis.jira_benchmark import benchmark_dataset_path, build_jira_benchmark
from qa_ftopsis.models import (
    classifier_metrics_path,
    load_feature_split,
    model_bundle_path,
    train_classifier,
)
from qa_ftopsis.policies import build_policy
from qa_ftopsis.reporting import generate_report, summarize_metrics
from qa_ftopsis.sim import run_policy_across_scenarios
from qa_ftopsis.skill_features import build_skill_features, embedding_metadata_path, ensure_skill_features


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_prepared_data(config: AppConfig) -> None:
    metadata_path = config.paths.prepared_dir / "metadata.json"
    if metadata_path.exists():
        return
    if config.data.dataset_profile == "jira_public":
        benchmark_path = benchmark_dataset_path(config)
        if not benchmark_path.exists():
            if (
                config.jira.raw_issues_path is not None
                and not config.jira.raw_issues_path.exists()
                and config.jira.api_base_url is not None
            ):
                fetch_jira_api_export(config)
            build_jira_benchmark(config)
    prepare_dataset(
        input_csv=config.paths.input_csv,
        output_dir=config.paths.prepared_dir,
        random_state=config.data.random_state,
        dataset_profile=config.data.dataset_profile,
        priority_mapping=config.data.priority_mapping,
        stratify_fields=config.data.stratify_fields,
        expected_languages=config.data.expected_languages,
    )


def ensure_model_artifacts(config: AppConfig) -> None:
    if model_bundle_path(config.paths.model_dir).exists() and classifier_metrics_path(
        config.paths.model_dir
    ).exists():
        return
    train_classifier(config)


def load_feature_splits(model_dir: str | Path, splits: list[str] | None = None) -> dict[str, pd.DataFrame]:
    split_names = splits or ["train", "val_cal", "val_sim", "test"]
    return {split_name: load_feature_split(model_dir, split_name) for split_name in split_names}


def ensure_skill_artifacts(config: AppConfig) -> None:
    feature_splits = load_feature_splits(config.paths.model_dir)
    ensure_skill_features(config, feature_splits)


def ensure_split_fixtures(config: AppConfig, split_name: str) -> None:
    split_df = load_feature_split(config.paths.model_dir, split_name)
    ensure_environment_fixtures(config, split_name, split_df)


def _requires_delay_model(config: AppConfig) -> bool:
    return any(
        delay_mode == "learned_jira_delay"
        for delay_mode in config.simulation.environment_matrix.delay_modes
    )


def ensure_delay_artifacts(config: AppConfig) -> None:
    if not _requires_delay_model(config):
        return
    if delay_model_path(config.paths.model_dir).exists() and delay_metrics_path(config.paths.model_dir).exists():
        sample = load_feature_split(config.paths.model_dir, "train")
        if any(column.startswith("delay_q_") for column in sample.columns):
            return
    train_delay_model(config)


def _tuning_rank(metrics_df: pd.DataFrame, mode: str) -> tuple[float, float, float, float]:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "tail_first":
        return (
            float(metrics_df["p95_wait"].mean()),
            float(metrics_df["p99_wait"].mean()),
            float(metrics_df["avg_cost"].mean()),
            -float(metrics_df["macro_f1"].mean()),
        )
    return (
        float(metrics_df["avg_cost"].mean()),
        float(metrics_df["p95_wait"].mean()),
        float(metrics_df["p99_wait"].mean()),
        -float(metrics_df["macro_f1"].mean()),
    )


def _evaluate_policy_for_tuning(
    *,
    config: AppConfig,
    train_df: pd.DataFrame,
    val_sim_df: pd.DataFrame,
    environment,
    selected_scenarios,
    policy_name: str,
    weights: list[float] | None = None,
    confidence_gate: float | None = None,
    alpha: float = 1.0,
    hybrid_mix: float = 0.5,
) -> tuple[pd.DataFrame, tuple[float, float, float, float]]:
    policy_kwargs: dict[str, Any] = {}
    if policy_name in {"qa_ftopsis", "qa_ftopsis_hierarchical", "qa_ftopsis_topk"}:
        policy_kwargs.update(
            {
                "environment": environment,
                "weights": weights,
                "confidence_gate": confidence_gate,
            }
        )
        if policy_name == "qa_ftopsis_topk":
            policy_kwargs["topk_k"] = config.routing.topk_gate.k
    elif policy_name == "qa_ftopsis_hybrid":
        policy_kwargs.update(
            {
                "environment": environment,
                "weights": weights,
                "confidence_gate": confidence_gate,
                "alpha": alpha,
                "topk_k": config.routing.topk_gate.k,
                "hybrid_mix": hybrid_mix,
            }
        )
    elif policy_name == "maxweight_prob":
        policy_kwargs["alpha"] = alpha

    policy = build_policy(policy_name, **policy_kwargs)
    metrics_df, _, _ = run_policy_across_scenarios(
        split_df=val_sim_df,
        train_df=train_df,
        policy=policy,
        environment=environment,
        scenarios=selected_scenarios,
        seeds=config.simulation.seeds,
        sla_deadlines=config.simulation.primary_sla,
        split_name="val_sim",
        sla_profile="primary",
        model_dir=str(config.paths.model_dir),
        include_records=False,
    )
    return metrics_df, _tuning_rank(metrics_df, config.tuning.selection_mode)


def tune_hyperparameters(config: AppConfig) -> dict[str, Any]:
    ensure_prepared_data(config)
    ensure_model_artifacts(config)
    ensure_skill_artifacts(config)
    ensure_delay_artifacts(config)
    ensure_split_fixtures(config, "val_sim")

    train_df = load_feature_split(config.paths.model_dir, "train")
    val_sim_df = load_feature_split(config.paths.model_dir, "val_sim")
    selected_scenarios = {
        name: scenario
        for name, scenario in config.simulation.scenarios.items()
        if name in config.tuning.selection_scenarios
    }
    if not selected_scenarios:
        selected_scenarios = config.simulation.scenarios

    tuning_by_environment: dict[str, Any] = {}
    for environment in build_environment_specs(config):
        best_alpha = None
        best_alpha_rank = None
        alpha_scores: list[dict[str, float | str]] = []
        for alpha in config.tuning.alpha_grid:
            metrics_df, rank = _evaluate_policy_for_tuning(
                config=config,
                train_df=train_df,
                val_sim_df=val_sim_df,
                environment=environment,
                selected_scenarios=selected_scenarios,
                policy_name="maxweight_prob",
                alpha=alpha,
            )
            alpha_scores.append(
                {
                    "environment_id": environment.environment_id,
                    "alpha": alpha,
                    "avg_cost_mean": float(metrics_df["avg_cost"].mean()),
                    "p95_wait_mean": float(metrics_df["p95_wait"].mean()),
                    "p99_wait_mean": float(metrics_df["p99_wait"].mean()),
                    "macro_f1_mean": float(metrics_df["macro_f1"].mean()),
                }
            )
            if best_alpha_rank is None or rank < best_alpha_rank:
                best_alpha = alpha
                best_alpha_rank = rank

        best_qa = None
        best_qa_rank = None
        qa_scores: list[dict[str, Any]] = []
        for weights in config.tuning.qa_weights:
            for guard_threshold in config.tuning.qa_guard_thresholds:
                metrics_df, rank = _evaluate_policy_for_tuning(
                    config=config,
                    train_df=train_df,
                    val_sim_df=val_sim_df,
                    environment=environment,
                    selected_scenarios=selected_scenarios,
                    policy_name="qa_ftopsis",
                    weights=weights,
                    confidence_gate=guard_threshold,
                )
                qa_scores.append(
                    {
                        "environment_id": environment.environment_id,
                        "weights": weights,
                        "confidence_gate": guard_threshold,
                        "avg_cost_mean": float(metrics_df["avg_cost"].mean()),
                        "p95_wait_mean": float(metrics_df["p95_wait"].mean()),
                        "p99_wait_mean": float(metrics_df["p99_wait"].mean()),
                        "macro_f1_mean": float(metrics_df["macro_f1"].mean()),
                    }
                )
                if best_qa_rank is None or rank < best_qa_rank:
                    best_qa_rank = rank
                    best_qa = {
                        "weights": weights,
                        "confidence_gate": guard_threshold,
                    }

        best_hierarchical = None
        best_hierarchical_rank = None
        hierarchical_scores: list[dict[str, Any]] = []
        for weights in config.tuning.qa_weights:
            for guard_threshold in config.tuning.qa_guard_thresholds:
                metrics_df, rank = _evaluate_policy_for_tuning(
                    config=config,
                    train_df=train_df,
                    val_sim_df=val_sim_df,
                    environment=environment,
                    selected_scenarios=selected_scenarios,
                    policy_name="qa_ftopsis_hierarchical",
                    weights=weights,
                    confidence_gate=guard_threshold,
                )
                hierarchical_scores.append(
                    {
                        "environment_id": environment.environment_id,
                        "weights": weights,
                        "confidence_gate": guard_threshold,
                        "avg_cost_mean": float(metrics_df["avg_cost"].mean()),
                        "p95_wait_mean": float(metrics_df["p95_wait"].mean()),
                        "p99_wait_mean": float(metrics_df["p99_wait"].mean()),
                        "macro_f1_mean": float(metrics_df["macro_f1"].mean()),
                    }
                )
                if best_hierarchical_rank is None or rank < best_hierarchical_rank:
                    best_hierarchical_rank = rank
                    best_hierarchical = {
                        "weights": weights,
                        "confidence_gate": guard_threshold,
                    }

        best_hybrid = None
        best_hybrid_rank = None
        hybrid_scores: list[dict[str, Any]] = []
        selected_alpha = float(best_alpha or 1.0)
        for weights in config.tuning.qa_weights:
            for guard_threshold in config.tuning.qa_guard_thresholds:
                for hybrid_mix in config.tuning.hybrid_mix_grid:
                    metrics_df, rank = _evaluate_policy_for_tuning(
                        config=config,
                        train_df=train_df,
                        val_sim_df=val_sim_df,
                        environment=environment,
                        selected_scenarios=selected_scenarios,
                        policy_name="qa_ftopsis_hybrid",
                        weights=weights,
                        confidence_gate=guard_threshold,
                        alpha=selected_alpha,
                        hybrid_mix=hybrid_mix,
                    )
                    hybrid_scores.append(
                        {
                            "environment_id": environment.environment_id,
                            "weights": weights,
                            "confidence_gate": guard_threshold,
                            "hybrid_mix": hybrid_mix,
                            "alpha": selected_alpha,
                            "avg_cost_mean": float(metrics_df["avg_cost"].mean()),
                            "p95_wait_mean": float(metrics_df["p95_wait"].mean()),
                            "p99_wait_mean": float(metrics_df["p99_wait"].mean()),
                            "macro_f1_mean": float(metrics_df["macro_f1"].mean()),
                        }
                    )
                    if best_hybrid_rank is None or rank < best_hybrid_rank:
                        best_hybrid_rank = rank
                        best_hybrid = {
                            "weights": weights,
                            "confidence_gate": guard_threshold,
                            "hybrid_mix": hybrid_mix,
                            "alpha": selected_alpha,
                        }

        tuning_by_environment[environment.environment_id] = {
            "selected_alpha": best_alpha or 1.0,
            "selected_qa": best_qa or {"weights": [0.30, 0.40, 0.30], "confidence_gate": None},
            "selected_hierarchical": best_hierarchical
            or {"weights": [0.30, 0.40, 0.30], "confidence_gate": None},
            "selected_hybrid": best_hybrid
            or {
                "weights": [0.30, 0.40, 0.30],
                "confidence_gate": None,
                "hybrid_mix": 0.5,
                "alpha": selected_alpha,
            },
            "alpha_scores": alpha_scores,
            "qa_scores": qa_scores,
            "hierarchical_scores": hierarchical_scores,
            "hybrid_scores": hybrid_scores,
        }

    return tuning_by_environment


def _persist_run_outputs(
    run_dir: Path,
    config: AppConfig,
    tuning: dict[str, Any],
    metrics_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    backlog_df: pd.DataFrame,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(run_dir / "raw_metrics.csv", index=False)
    decisions_df.to_parquet(run_dir / "decisions.parquet", index=False)
    backlog_df.to_parquet(run_dir / "backlog_trace.parquet", index=False)
    summarize_metrics(metrics_df).to_csv(run_dir / "summary_metrics.csv", index=False)
    with (run_dir / "tuning_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(tuning, handle, indent=2)
    save_config_snapshot(config, run_dir / "config_snapshot.yaml")
    return run_dir


def _load_config_object(config_or_path: AppConfig | str | Path) -> AppConfig:
    if isinstance(config_or_path, AppConfig):
        return config_or_path
    return load_config(config_or_path)


def _policy_for_environment(
    config: AppConfig,
    policy_name: str,
    environment,
    tuning_for_env: dict[str, Any],
):
    if policy_name == "maxweight_prob":
        return build_policy(policy_name, alpha=float(tuning_for_env["selected_alpha"]))
    if policy_name == "qa_ftopsis":
        return build_policy(
            policy_name,
            environment=environment,
            weights=list(tuning_for_env["selected_qa"]["weights"]),
            confidence_gate=tuning_for_env["selected_qa"]["confidence_gate"],
        )
    if policy_name == "qa_ftopsis_hierarchical":
        return build_policy(
            policy_name,
            environment=environment,
            weights=list(tuning_for_env["selected_hierarchical"]["weights"]),
            confidence_gate=tuning_for_env["selected_hierarchical"]["confidence_gate"],
        )
    if policy_name == "qa_ftopsis_topk":
        return build_policy(
            policy_name,
            environment=environment,
            weights=list(tuning_for_env["selected_qa"]["weights"]),
            confidence_gate=tuning_for_env["selected_qa"]["confidence_gate"],
            topk_k=config.routing.topk_gate.k,
        )
    if policy_name == "qa_ftopsis_hybrid":
        return build_policy(
            policy_name,
            environment=environment,
            weights=list(tuning_for_env["selected_hybrid"]["weights"]),
            confidence_gate=tuning_for_env["selected_hybrid"]["confidence_gate"],
            topk_k=config.routing.topk_gate.k,
            alpha=float(tuning_for_env["selected_hybrid"]["alpha"]),
            hybrid_mix=float(tuning_for_env["selected_hybrid"]["hybrid_mix"]),
        )
    if policy_name == "maxweight_delay":
        return build_policy(policy_name, environment=environment)
    if policy_name == "jsq_topk":
        return build_policy(policy_name, topk_k=config.routing.topk_gate.k)
    return build_policy(policy_name)


def _suite_policy_names(config: AppConfig) -> list[str]:
    policy_names = [
        "classifier_only",
        "jsq",
        "maxweight_delay",
        "maxweight_prob",
        "qa_ftopsis",
        "qa_ftopsis_hierarchical",
        "qa_ftopsis_hybrid",
    ]
    if config.routing.topk_gate.enabled:
        for base_name in config.routing.topk_gate.policy_names:
            normalized = str(base_name).strip().lower()
            if normalized == "jsq":
                policy_names.insert(2, "jsq_topk")
            elif normalized == "qa_ftopsis":
                policy_names.append("qa_ftopsis_topk")
    return policy_names


def run_sim(config_or_path: AppConfig | str | Path, policy_name: str, split: str) -> Path:
    config = _load_config_object(config_or_path)
    ensure_prepared_data(config)
    ensure_model_artifacts(config)
    ensure_skill_artifacts(config)
    ensure_delay_artifacts(config)
    ensure_split_fixtures(config, split)
    tuning = tune_hyperparameters(config)

    train_df = load_feature_split(config.paths.model_dir, "train")
    split_df = load_feature_split(config.paths.model_dir, split)

    metrics_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    backlog_frames: list[pd.DataFrame] = []

    for environment in build_environment_specs(config):
        tuning_for_env = tuning[environment.environment_id]
        policy = _policy_for_environment(config, policy_name, environment, tuning_for_env)
        primary_metrics, primary_decisions, primary_backlog = run_policy_across_scenarios(
            split_df=split_df,
            train_df=train_df,
            policy=policy,
            environment=environment,
            scenarios=config.simulation.scenarios,
            seeds=config.simulation.seeds,
            sla_deadlines=config.simulation.primary_sla,
            split_name=split,
            sla_profile="primary",
            model_dir=str(config.paths.model_dir),
            include_records=True,
        )
        metrics_frames.append(primary_metrics)
        decision_frames.append(primary_decisions)
        backlog_frames.append(primary_backlog)

        if split == "test":
            robust_metrics, robust_decisions, robust_backlog = run_policy_across_scenarios(
                split_df=split_df,
                train_df=train_df,
                policy=policy,
                environment=environment,
                scenarios=config.simulation.scenarios,
                seeds=config.simulation.seeds,
                sla_deadlines=config.simulation.robustness_sla,
                split_name=split,
                sla_profile="robustness",
                model_dir=str(config.paths.model_dir),
                include_records=True,
            )
            metrics_frames.append(robust_metrics)
            decision_frames.append(robust_decisions)
            backlog_frames.append(robust_backlog)

    run_dir = config.paths.run_root / f"run_{policy_name}_{split}_{_timestamp()}"
    return _persist_run_outputs(
        run_dir=run_dir,
        config=config,
        tuning=tuning,
        metrics_df=pd.concat(metrics_frames, ignore_index=True),
        decisions_df=pd.concat(decision_frames, ignore_index=True),
        backlog_df=pd.concat(backlog_frames, ignore_index=True),
    )


def run_suite(config_or_path: AppConfig | str | Path) -> Path:
    config = _load_config_object(config_or_path)
    ensure_prepared_data(config)
    ensure_model_artifacts(config)
    ensure_skill_artifacts(config)
    ensure_delay_artifacts(config)
    ensure_split_fixtures(config, "test")
    ensure_split_fixtures(config, "val_sim")
    tuning = tune_hyperparameters(config)

    train_df = load_feature_split(config.paths.model_dir, "train")
    test_df = load_feature_split(config.paths.model_dir, "test")

    policy_names = _suite_policy_names(config)
    metrics_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    backlog_frames: list[pd.DataFrame] = []

    for environment in build_environment_specs(config):
        tuning_for_env = tuning[environment.environment_id]
        for policy_name in policy_names:
            policy = _policy_for_environment(config, policy_name, environment, tuning_for_env)
            primary_metrics, primary_decisions, primary_backlog = run_policy_across_scenarios(
                split_df=test_df,
                train_df=train_df,
                policy=policy,
                environment=environment,
                scenarios=config.simulation.scenarios,
                seeds=config.simulation.seeds,
                sla_deadlines=config.simulation.primary_sla,
                split_name="test",
                sla_profile="primary",
                model_dir=str(config.paths.model_dir),
                include_records=True,
            )
            robust_metrics, robust_decisions, robust_backlog = run_policy_across_scenarios(
                split_df=test_df,
                train_df=train_df,
                policy=policy,
                environment=environment,
                scenarios=config.simulation.scenarios,
                seeds=config.simulation.seeds,
                sla_deadlines=config.simulation.robustness_sla,
                split_name="test",
                sla_profile="robustness",
                model_dir=str(config.paths.model_dir),
                include_records=True,
            )
            metrics_frames.extend([primary_metrics, robust_metrics])
            decision_frames.extend([primary_decisions, robust_decisions])
            backlog_frames.extend([primary_backlog, robust_backlog])

    run_dir = config.paths.run_root / f"suite_{_timestamp()}"
    persisted = _persist_run_outputs(
        run_dir=run_dir,
        config=config,
        tuning=tuning,
        metrics_df=pd.concat(metrics_frames, ignore_index=True),
        decisions_df=pd.concat(decision_frames, ignore_index=True),
        backlog_df=pd.concat(backlog_frames, ignore_index=True),
    )
    generate_report(
        run_dir=persisted,
        report_root=config.paths.report_root,
        sample_size=config.reporting.explainability_sample_size,
    )
    return persisted


def build_skill_features_command(config_or_path: AppConfig | str | Path) -> dict[str, str]:
    config = _load_config_object(config_or_path)
    ensure_prepared_data(config)
    ensure_model_artifacts(config)
    feature_splits = load_feature_splits(config.paths.model_dir)
    result = build_skill_features(config, feature_splits)
    return result


def build_jira_benchmark_command(config_or_path: AppConfig | str | Path) -> dict[str, str | int | float]:
    config = _load_config_object(config_or_path)
    if (
        config.jira.raw_issues_path is not None
        and not config.jira.raw_issues_path.exists()
        and config.jira.api_base_url is not None
    ):
        fetch_jira_api_export(config)
    return build_jira_benchmark(config)


def train_delay_model_command(config_or_path: AppConfig | str | Path) -> dict[str, Any]:
    config = _load_config_object(config_or_path)
    ensure_prepared_data(config)
    ensure_model_artifacts(config)
    ensure_skill_artifacts(config)
    return train_delay_model(config)


def fetch_jira_api_command(config_or_path: AppConfig | str | Path) -> dict[str, str | int]:
    config = _load_config_object(config_or_path)
    return fetch_jira_api_export(config)
