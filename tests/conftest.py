from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


def build_synthetic_dataset(csv_path: Path, per_group: int = 12) -> Path:
    queues = [
        ("Billing and Payments", "invoice payment refund card charge"),
        ("Technical Support", "server outage crash error stacktrace code500"),
        ("Customer Service", "question information feedback order support"),
    ]
    languages = ["en", "de"]
    priorities = ["low", "medium", "high"]
    rows = []
    ticket_id = 0
    for queue_name, cue_text in queues:
        for language in languages:
            for sample_idx in range(per_group):
                priority = priorities[sample_idx % len(priorities)]
                rows.append(
                    {
                        "subject": f"{queue_name} subject {language} {sample_idx}",
                        "body": (
                            f"{cue_text} {language} sample {sample_idx} "
                            f"ERR_{sample_idx} v1.{sample_idx % 3} ticket {ticket_id}"
                        ),
                        "queue": queue_name,
                        "priority": priority,
                        "language": language,
                        "answer": "placeholder",
                        "type": "Request",
                    }
                )
                ticket_id += 1
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def build_test_config(config_path: Path, dataset_path: Path) -> Path:
    config = {
        "paths": {
            "input_csv": str(dataset_path),
            "prepared_dir": str(config_path.parent / "artifacts" / "prepared"),
            "model_dir": str(config_path.parent / "artifacts" / "models"),
            "run_root": str(config_path.parent / "artifacts" / "runs"),
            "report_root": str(config_path.parent / "artifacts" / "reports"),
        },
        "data": {
            "random_state": 7,
            "dataset_profile": "default",
            "priority_mapping": {},
            "stratify_fields": ["queue", "language"],
        },
        "model": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "word_max_features": 300,
            "char_max_features": 300,
            "max_iter": 300,
            "sublinear_tf": True,
            "prepend_language_token": True,
        },
        "simulation": {
            "queue_discipline": "fifo",
            "slot_order": "serve_then_arrive",
            "serve_new_same_slot": False,
            "capacity_mode": "proportional_to_pj",
            "seeds": [13],
            "primary_sla": {"high": 3, "medium": 6, "low": 10},
            "robustness_sla": {"high": 2, "medium": 5, "low": 8},
            "scenarios": {
                "high_load": {
                    "lambda_base": 4,
                    "rho_target": 0.85,
                    "burst_start_offset": 0,
                }
            },
            "environment_matrix": {
                "service_models": ["deterministic", "heavy_tail"],
                "delay_modes": ["redundant_baseline", "embedding_kappa"],
            },
            "heavy_tail": {
                "pareto_alpha": 1.5,
                "base_poisson_bias": 0.25,
                "base_poisson_scale": 0.75,
                "long_job_prob_base": 0.08,
                "long_job_prob_scale": 0.20,
                "max_service_units": 6,
            },
        },
        "skill_features": {
            "model_name": "hashing-mock",
            "max_length": 64,
            "batch_size": 16,
            "device": "cpu",
            "kappa_min": 0.8,
            "kappa_max": 1.4,
        },
        "routing": {
            "topk_gate": {
                "enabled": True,
                "k": 2,
                "policy_names": ["qa_ftopsis", "jsq"],
            }
        },
        "tuning": {
            "alpha_grid": [0.5, 1.0],
            "qa_guard_thresholds": [None, 0.6],
            "qa_weights": [[0.30, 0.40, 0.30], [0.25, 0.35, 0.40]],
            "selection_scenarios": ["high_load"],
        },
        "reporting": {
            "explainability_sample_size": 10,
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def build_synthetic_jira_exports(base_dir: Path, per_queue: int = 10) -> tuple[Path, Path]:
    queues = [
        ("Frontend", "ui browser css javascript form layout"),
        ("Platform", "api database infra deploy service cluster"),
        ("Billing", "invoice refund payment tax amount checkout"),
        ("Security", "auth token login audit permission access"),
    ]
    priorities = ["low", "medium", "high", "critical"]
    issue_types = ["bug", "task", "incident", "story"]
    issues: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    issue_counter = 1
    for queue_name, cue_text in queues:
        for sample_idx in range(per_queue):
            issue_id = f"ISSUE-{issue_counter}"
            created_day = 1 + (sample_idx % 6)
            created_hour = 8 + (sample_idx % 5)
            resolution_hours = 8 + (sample_idx % 4) * 6 + queues.index((queue_name, cue_text)) * 2
            issues.append(
                {
                    "issue_id": issue_id,
                    "instance_id": "apache-jira",
                    "project_key": f"PRJ{queues.index((queue_name, cue_text)) + 1}",
                    "summary": f"{queue_name} issue {sample_idx}",
                    "description": (
                        f"{cue_text} case {sample_idx} stack ERR_{issue_counter} "
                        f"version v1.{sample_idx % 3}"
                    ),
                    "created_at": f"2025-01-{created_day:02d}T{created_hour:02d}:00:00Z",
                    "resolved_at": (
                        pd.Timestamp(f"2025-01-{created_day:02d}T{created_hour:02d}:00:00Z")
                        + pd.Timedelta(hours=resolution_hours)
                    ).isoformat(),
                    "priority": priorities[(sample_idx + queues.index((queue_name, cue_text))) % len(priorities)],
                    "issue_type": issue_types[sample_idx % len(issue_types)],
                    "component": queue_name,
                    "components_json": json.dumps(
                        [queue_name]
                        + (
                            [queues[(queues.index((queue_name, cue_text)) + 1) % len(queues)][0]]
                            if sample_idx % 2 == 0
                            else []
                        )
                    ),
                }
            )
            if sample_idx % 3 == 0:
                previous_queue = queues[(queues.index((queue_name, cue_text)) - 1) % len(queues)][0]
                history.append(
                    {
                        "issue_id": issue_id,
                        "change_time": (
                            pd.Timestamp(f"2025-01-{created_day:02d}T{created_hour:02d}:00:00Z")
                            + pd.Timedelta(hours=2)
                        ).isoformat(),
                        "field_name": "component",
                        "old_value": previous_queue,
                        "new_value": queue_name,
                        "change_author_mask": "user_a",
                    }
                )
            issue_counter += 1

    issues_path = base_dir / "jira_issues.csv"
    history_path = base_dir / "jira_history.csv"
    pd.DataFrame(issues).to_csv(issues_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return issues_path, history_path


def build_jira_test_config(
    config_path: Path,
    issues_path: Path,
    history_path: Path,
    *,
    queue_field: str = "component",
) -> Path:
    config = {
        "paths": {
            "input_csv": str(config_path.parent / "artifacts" / "benchmarks" / "jira_public" / "jira_router_benchmark.parquet"),
            "prepared_dir": str(config_path.parent / "artifacts" / "prepared"),
            "model_dir": str(config_path.parent / "artifacts" / "models"),
            "run_root": str(config_path.parent / "artifacts" / "runs"),
            "report_root": str(config_path.parent / "artifacts" / "reports"),
        },
        "data": {
            "random_state": 7,
            "dataset_profile": "jira_public",
            "priority_mapping": {
                "lowest": "low",
                "low": "low",
                "minor": "low",
                "trivial": "low",
                "medium": "medium",
                "major": "medium",
                "normal": "medium",
                "high": "high",
                "highest": "high",
                "critical": "high",
                "blocker": "high",
            },
            "stratify_fields": ["queue"],
        },
        "model": {
            "word_ngram_range": [1, 2],
            "char_ngram_range": [3, 5],
            "word_max_features": 300,
            "char_max_features": 300,
            "max_iter": 300,
            "sublinear_tf": True,
            "prepend_language_token": False,
        },
        "simulation": {
            "queue_discipline": "fifo",
            "slot_order": "serve_then_arrive",
            "serve_new_same_slot": False,
            "capacity_mode": "uniform_one",
            "seeds": [13],
            "primary_sla": {"high": 3, "medium": 6, "low": 10},
            "robustness_sla": {"high": 2, "medium": 5, "low": 8},
            "scenarios": {
                "high_load": {
                    "lambda_base": 3,
                    "rho_target": 0.85,
                }
            },
            "environment_matrix": {
                "service_models": ["empirical"],
                "delay_modes": ["embedding_kappa", "learned_jira_delay"],
            },
        },
        "skill_features": {
            "model_name": "hashing-mock",
            "max_length": 64,
            "batch_size": 16,
            "device": "cpu",
            "kappa_min": 0.8,
            "kappa_max": 1.4,
        },
        "jira": {
            "raw_issues_path": str(issues_path),
            "raw_history_path": str(history_path),
            "benchmark_dir": str(config_path.parent / "artifacts" / "benchmarks" / "jira_public"),
            "instance_filter": ["apache-jira"],
            "queue_field": queue_field,
            "min_queue_size": 4,
            "macro_group_min_support": 1,
            "macro_group_top_edges": 3,
            "slot_hours": 8,
            "max_service_units": 12,
            "resolution_target": "time_to_resolution_hours",
        },
        "delay_model": {
            "model_type": "random_forest",
            "n_estimators": 32,
            "max_depth": 6,
            "min_samples_leaf": 1,
        },
        "routing": {
            "topk_gate": {
                "enabled": True,
                "k": 2,
                "policy_names": ["qa_ftopsis", "jsq"],
            }
        },
        "tuning": {
            "alpha_grid": [0.5, 1.0],
            "qa_guard_thresholds": [None, 0.6],
            "qa_weights": [[0.30, 0.40, 0.30], [0.25, 0.35, 0.40]],
            "selection_scenarios": ["high_load"],
        },
        "reporting": {
            "explainability_sample_size": 10,
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path
