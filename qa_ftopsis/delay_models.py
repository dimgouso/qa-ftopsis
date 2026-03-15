from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from qa_ftopsis.config import AppConfig
from qa_ftopsis.data import PREPARED_SPLITS
from qa_ftopsis.models import feature_split_path, load_feature_split


@dataclass(slots=True)
class DelayModelBundle:
    regressor: Any
    queue_ids: list[int]
    priority_codes: dict[str, int]
    issue_type_codes: dict[str, int]
    queue_stats: dict[int, dict[str, float]]
    feature_names: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def delay_model_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "delay_model.joblib"


def delay_metrics_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "delay_model_metrics.json"


def delay_column_names(queue_ids: list[int]) -> list[str]:
    return [f"delay_q_{queue_id}" for queue_id in queue_ids]


def load_delay_model_bundle(model_dir: str | Path) -> DelayModelBundle:
    payload = joblib.load(delay_model_path(model_dir))
    payload["queue_ids"] = [int(value) for value in payload["queue_ids"]]
    payload["queue_stats"] = {
        int(queue_id): {str(key): float(value) for key, value in stats.items()}
        for queue_id, stats in payload["queue_stats"].items()
    }
    payload["priority_codes"] = {str(key): int(value) for key, value in payload["priority_codes"].items()}
    payload["issue_type_codes"] = {str(key): int(value) for key, value in payload["issue_type_codes"].items()}
    return DelayModelBundle(**payload)


def _priority_code_map(frame: pd.DataFrame) -> dict[str, int]:
    values = sorted(frame["priority"].fillna("medium").astype(str).str.lower().unique().tolist())
    return {value: index for index, value in enumerate(values)}


def _issue_type_code_map(frame: pd.DataFrame) -> dict[str, int]:
    if "issue_type" not in frame.columns:
        return {"unknown": 0}
    values = sorted(frame["issue_type"].fillna("unknown").astype(str).str.lower().unique().tolist())
    return {value: index for index, value in enumerate(values)}


def _queue_stats(train_df: pd.DataFrame, queue_ids: list[int]) -> dict[int, dict[str, float]]:
    grouped = train_df.groupby("true_queue_id", dropna=False)
    share = train_df["true_queue_id"].astype(int).value_counts(normalize=True).to_dict()
    stats: dict[int, dict[str, float]] = {}
    for queue_id in queue_ids:
        if queue_id in grouped.groups:
            frame = grouped.get_group(queue_id)
        else:
            frame = train_df.iloc[0:0]
        stats[int(queue_id)] = {
            "queue_share": float(share.get(queue_id, 0.0)),
            "mean_service_units": float(frame["service_units"].mean()) if not frame.empty else 1.0,
            "mean_resolution_hours": (
                float(frame["time_to_resolution_hours"].mean())
                if "time_to_resolution_hours" in frame.columns and not frame.empty
                else 0.0
            ),
            "mean_queue_changes": (
                float(frame["num_queue_changes"].mean())
                if "num_queue_changes" in frame.columns and not frame.empty
                else 0.0
            ),
            "queue_change_rate": (
                float((frame["num_queue_changes"].astype(float) > 0).mean())
                if "num_queue_changes" in frame.columns and not frame.empty
                else 0.0
            ),
        }
    return stats


def _candidate_feature_frame(
    frame: pd.DataFrame,
    candidate_queue_id: int,
    *,
    priority_codes: dict[str, int],
    issue_type_codes: dict[str, int],
    queue_stats: dict[int, dict[str, float]],
) -> pd.DataFrame:
    probability_column = f"prob_q_{candidate_queue_id}"
    similarity_column = f"sim_q_{candidate_queue_id}"
    kappa_column = f"kappa_q_{candidate_queue_id}"
    stats = queue_stats[int(candidate_queue_id)]
    priority = frame["priority"].fillna("medium").astype(str).str.lower().map(
        lambda value: priority_codes.get(value, priority_codes.get("medium", 0))
    )
    issue_type = (
        frame.get("issue_type", pd.Series(["unknown"] * len(frame), index=frame.index))
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .map(lambda value: issue_type_codes.get(value, 0))
    )
    return pd.DataFrame(
        {
            "complexity_score": frame["complexity_score"].astype(float),
            "entropy": frame["entropy"].astype(float),
            "p_max": frame["p_max"].astype(float),
            "prob_for_candidate": frame[probability_column].astype(float),
            "sim_for_candidate": (
                frame[similarity_column].astype(float)
                if similarity_column in frame.columns
                else np.zeros(len(frame), dtype=float)
            ),
            "kappa_for_candidate": (
                frame[kappa_column].astype(float)
                if kappa_column in frame.columns
                else np.ones(len(frame), dtype=float)
            ),
            "priority_code": priority.astype(float),
            "issue_type_code": issue_type.astype(float),
            "queue_id": float(candidate_queue_id),
            "queue_share": float(stats["queue_share"]),
            "queue_mean_service_units": float(stats["mean_service_units"]),
            "queue_mean_resolution_hours": float(stats["mean_resolution_hours"]),
            "queue_mean_queue_changes": float(stats["mean_queue_changes"]),
            "queue_change_rate": float(stats["queue_change_rate"]),
        },
        index=frame.index,
    )


def _predict_delay_matrix(frame: pd.DataFrame, bundle: DelayModelBundle) -> np.ndarray:
    matrices = []
    for queue_id in bundle.queue_ids:
        candidate_features = _candidate_feature_frame(
            frame,
            queue_id,
            priority_codes=bundle.priority_codes,
            issue_type_codes=bundle.issue_type_codes,
            queue_stats=bundle.queue_stats,
        )
        predicted = np.asarray(bundle.regressor.predict(candidate_features[bundle.feature_names]), dtype=float)
        matrices.append(np.clip(predicted, 1.0, None))
    return np.column_stack(matrices)


def train_delay_model(config: AppConfig) -> dict[str, Any]:
    train_df = load_feature_split(config.paths.model_dir, "train")
    if "service_units" not in train_df.columns:
        raise ValueError("Delay model requires service_units column in feature splits")

    queue_ids = sorted(train_df["true_queue_id"].astype(int).unique().tolist())
    priority_codes = _priority_code_map(train_df)
    issue_type_codes = _issue_type_code_map(train_df)
    queue_stats = _queue_stats(train_df, queue_ids)

    training_frames: list[pd.DataFrame] = []
    training_targets: list[pd.Series] = []
    for queue_id in queue_ids:
        subset = train_df[train_df["true_queue_id"].astype(int) == queue_id]
        if subset.empty:
            continue
        training_frames.append(
            _candidate_feature_frame(
                subset,
                queue_id,
                priority_codes=priority_codes,
                issue_type_codes=issue_type_codes,
                queue_stats=queue_stats,
            )
        )
        training_targets.append(subset["service_units"].astype(float))
    training_matrix = pd.concat(training_frames, ignore_index=True)
    training_target = pd.concat(training_targets, ignore_index=True)
    feature_names = training_matrix.columns.tolist()

    if config.delay_model.model_type != "random_forest":
        raise ValueError(f"Unsupported delay model type: {config.delay_model.model_type}")
    regressor = RandomForestRegressor(
        n_estimators=config.delay_model.n_estimators,
        max_depth=config.delay_model.max_depth,
        min_samples_leaf=config.delay_model.min_samples_leaf,
        random_state=config.data.random_state,
        n_jobs=-1,
    )
    regressor.fit(training_matrix[feature_names], training_target)

    bundle = DelayModelBundle(
        regressor=regressor,
        queue_ids=queue_ids,
        priority_codes=priority_codes,
        issue_type_codes=issue_type_codes,
        queue_stats=queue_stats,
        feature_names=feature_names,
        metadata={
            "model_type": config.delay_model.model_type,
            "resolution_target": config.jira.resolution_target,
            "random_state": config.data.random_state,
        },
    )

    metrics: dict[str, dict[str, float]] = {}
    for split_name in PREPARED_SPLITS:
        split_df = load_feature_split(config.paths.model_dir, split_name)
        predicted_matrix = _predict_delay_matrix(split_df, bundle)
        enriched = split_df.copy()
        for column_name, values in zip(delay_column_names(queue_ids), predicted_matrix.T):
            enriched[column_name] = values.astype(float)
        enriched.to_parquet(feature_split_path(config.paths.model_dir, split_name), index=False)

        true_queue_ids = split_df["true_queue_id"].astype(int).to_numpy()
        true_predictions = np.array(
            [
                predicted_matrix[row_index, queue_ids.index(int(queue_id))]
                for row_index, queue_id in enumerate(true_queue_ids)
            ],
            dtype=float,
        )
        true_targets = split_df["service_units"].astype(float).to_numpy()
        metrics[split_name] = {
            "mae": float(mean_absolute_error(true_targets, true_predictions)),
            "rmse": float(mean_squared_error(true_targets, true_predictions) ** 0.5),
        }

    joblib.dump(bundle.to_dict(), delay_model_path(config.paths.model_dir))
    with delay_metrics_path(config.paths.model_dir).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return {
        "model_dir": str(config.paths.model_dir),
        "delay_model_path": str(delay_model_path(config.paths.model_dir)),
        "delay_metrics": metrics,
    }
