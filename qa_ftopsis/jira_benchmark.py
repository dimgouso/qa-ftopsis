from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qa_ftopsis.config import AppConfig
from qa_ftopsis.jira_ingest import (
    canonical_queue_combo,
    coerce_timestamp,
    first_present_column,
    normalize_queue_tokens,
    normalize_queue_value,
    read_tabular_export,
)


def benchmark_dataset_path(config: AppConfig) -> Path:
    return config.paths.input_csv


def benchmark_history_path(config: AppConfig) -> Path:
    return Path(config.jira.benchmark_dir or config.paths.input_csv.parent) / "jira_issue_history.parquet"


def benchmark_metadata_path(config: AppConfig) -> Path:
    return Path(config.jira.benchmark_dir or config.paths.input_csv.parent) / "benchmark_metadata.json"


def benchmark_queue_distribution_path(config: AppConfig) -> Path:
    return Path(config.jira.benchmark_dir or config.paths.input_csv.parent) / "queue_distribution.csv"


def benchmark_reroute_stats_path(config: AppConfig) -> Path:
    return Path(config.jira.benchmark_dir or config.paths.input_csv.parent) / "reroute_stats.csv"


def benchmark_transition_graph_path(config: AppConfig) -> Path:
    return Path(config.jira.benchmark_dir or config.paths.input_csv.parent) / "queue_transition_graph.csv"


def benchmark_macro_groups_path(config: AppConfig) -> Path:
    return Path(config.jira.benchmark_dir or config.paths.input_csv.parent) / "macro_groups.csv"


def _priority_mapping(config: AppConfig) -> dict[str, str]:
    if config.data.priority_mapping:
        return {str(key).lower(): str(value).lower() for key, value in config.data.priority_mapping.items()}
    return {
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
    }


def _history_source_field(queue_field: str) -> str:
    normalized = queue_field.strip().lower()
    if normalized in {"component_combo", "components_json"}:
        return "component"
    return normalized


def _queue_name_from_row(frame: pd.DataFrame, queue_field: str) -> pd.Series:
    normalized_field = queue_field.strip().lower()
    if normalized_field in {"component_combo", "components_json"}:
        return first_present_column(frame, ["components_json", "component"], "").map(canonical_queue_combo)
    return frame[queue_field].map(normalize_queue_value)


def _normalize_issue_table(config: AppConfig, issues: pd.DataFrame) -> pd.DataFrame:
    queue_field = config.jira.queue_field
    normalized = issues.copy()
    normalized["issue_id"] = first_present_column(normalized, ["issue_id", "id", "key"]).astype(str)
    normalized["instance_id"] = (
        first_present_column(normalized, ["instance_id", "instance", "repository"], "default")
        .fillna("default")
        .astype(str)
    )
    normalized["project_key"] = (
        first_present_column(normalized, ["project_key", "project", "project_id"], "unknown")
        .fillna("unknown")
        .astype(str)
    )
    normalized["subject"] = (
        first_present_column(normalized, ["summary", "subject", "title"], "")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    normalized["body"] = (
        first_present_column(normalized, ["description", "body"], "")
        .fillna("")
        .astype(str)
        .str.strip()
    )
    normalized["priority_raw"] = (
        first_present_column(normalized, ["priority", "priority_name"], "medium")
        .fillna("medium")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    normalized["issue_type"] = (
        first_present_column(normalized, ["issue_type", "type"], "unknown")
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    normalized["created_at"] = coerce_timestamp(
        first_present_column(normalized, ["created_at", "created", "created_date"])
    )
    normalized["resolved_at"] = coerce_timestamp(
        first_present_column(normalized, ["resolved_at", "resolved", "resolution_date", "closed_at"])
    )
    normalized["component_tokens"] = first_present_column(
        normalized,
        ["components_json", "component"],
        "",
    ).map(normalize_queue_tokens)
    special_fields = {"component_combo", "components_json"}
    if queue_field not in normalized.columns and queue_field not in special_fields:
        raise ValueError(f"Missing queue field '{queue_field}' in Jira issues export")
    normalized["queue_name"] = _queue_name_from_row(normalized, queue_field)
    normalized["language"] = "unknown"
    normalized["text"] = (normalized["subject"] + "\n\n" + normalized["body"]).str.strip()
    normalized["priority"] = (
        normalized["priority_raw"]
        .map(_priority_mapping(config))
        .fillna("medium")
        .astype(str)
        .str.lower()
    )
    return normalized


def _normalize_history_table(config: AppConfig, history: pd.DataFrame) -> pd.DataFrame:
    queue_field = _history_source_field(config.jira.queue_field)
    normalized = history.copy()
    normalized["issue_id"] = first_present_column(normalized, ["issue_id", "id", "key"]).astype(str)
    normalized["change_time"] = coerce_timestamp(
        first_present_column(normalized, ["change_time", "changed_at", "timestamp"])
    )
    normalized["field_name"] = (
        first_present_column(normalized, ["field_name", "field"], "")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    normalized["old_value"] = first_present_column(normalized, ["old_value", "from"], "").map(normalize_queue_value)
    normalized["new_value"] = first_present_column(normalized, ["new_value", "to"], "").map(normalize_queue_value)
    normalized["change_author_mask"] = (
        first_present_column(normalized, ["change_author_mask", "author"], "unknown")
        .fillna("unknown")
        .astype(str)
    )
    normalized = normalized[normalized["field_name"] == queue_field].copy()
    normalized = normalized.dropna(subset=["change_time"])
    normalized = normalized.sort_values(["issue_id", "change_time"]).reset_index(drop=True)
    return normalized


def _history_queue_stats(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=[
                "issue_id",
                "initial_queue_name",
                "final_queue_name",
                "num_queue_changes",
                "time_to_first_queue_change_hours",
            ]
        )
    rows: list[dict[str, object]] = []
    for issue_id, frame in history.groupby("issue_id", dropna=False):
        frame = frame.sort_values("change_time")
        initial_queue = ""
        for value in frame["old_value"].astype(str):
            if value:
                initial_queue = value
                break
        final_queue = ""
        for value in reversed(frame["new_value"].astype(str).tolist()):
            if value:
                final_queue = value
                break
        rows.append(
            {
                "issue_id": str(issue_id),
                "initial_queue_name": initial_queue,
                "final_queue_name": final_queue,
                "num_queue_changes": int(len(frame)),
                "first_change_time": frame["change_time"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def _issue_component_tokens(
    issues: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, set[str]]:
    issue_tokens: dict[str, set[str]] = {
        str(row.issue_id): set(row.component_tokens)
        for row in issues[["issue_id", "component_tokens"]].itertuples(index=False)
    }
    if history.empty:
        return issue_tokens
    for row in history.itertuples(index=False):
        tokens = issue_tokens.setdefault(str(row.issue_id), set())
        tokens.update(normalize_queue_tokens(row.old_value))
        tokens.update(normalize_queue_tokens(row.new_value))
    return issue_tokens


def _queue_tokens(queue_name: str) -> set[str]:
    return set(normalize_queue_tokens(queue_name))


def _build_transition_graph(
    history: pd.DataFrame,
    queue_map: dict[str, int],
    benchmark_frame: pd.DataFrame,
    issue_tokens: dict[str, set[str]],
    min_support: int,
    top_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reverse_map = {queue_id: queue_name for queue_name, queue_id in queue_map.items()}
    if not reverse_map:
        empty_graph = pd.DataFrame(
            columns=[
                "source_queue_id",
                "source_queue_name",
                "target_queue_id",
                "target_queue_name",
                "count",
                "direct_transition_count",
                "issue_overlap_count",
                "token_overlap_count",
                "source_probability",
                "symmetric_count",
            ]
        )
        empty_groups = pd.DataFrame(columns=["queue_id", "queue_name", "macro_group_id"])
        return empty_graph, empty_groups

    transition_counts: dict[tuple[int, int], int] = {}
    if not history.empty:
        for _, frame in history.groupby("issue_id", dropna=False):
            frame = frame.sort_values("change_time")
            for row in frame.itertuples(index=False):
                old_name = str(row.old_value).strip()
                new_name = str(row.new_value).strip()
                if old_name not in queue_map or new_name not in queue_map:
                    continue
                source_id = int(queue_map[old_name])
                target_id = int(queue_map[new_name])
                if source_id == target_id:
                    continue
                transition_counts[(source_id, target_id)] = transition_counts.get((source_id, target_id), 0) + 1

    queue_tokens = {
        int(queue_id): _queue_tokens(queue_name)
        for queue_name, queue_id in queue_map.items()
    }
    overlap_counts: dict[tuple[int, int], int] = {}
    for row in benchmark_frame[["issue_id", "queue_id"]].drop_duplicates().itertuples(index=False):
        source_queue_id = int(row.queue_id)
        tokens = issue_tokens.get(str(row.issue_id), set())
        if not tokens:
            tokens = set(queue_tokens.get(source_queue_id, set()))
        for target_queue_id, target_tokens in queue_tokens.items():
            if target_queue_id == source_queue_id:
                continue
            if not (tokens & target_tokens):
                continue
            overlap_counts[(source_queue_id, target_queue_id)] = overlap_counts.get(
                (source_queue_id, target_queue_id),
                0,
            ) + 1

    all_edges = sorted(set(transition_counts) | set(overlap_counts))
    rows: list[dict[str, int | float | str]] = []
    row_totals: dict[int, int] = {}
    for source_id, target_id in all_edges:
        total_count = int(
            transition_counts.get((source_id, target_id), 0)
            + overlap_counts.get((source_id, target_id), 0)
        )
        row_totals[source_id] = row_totals.get(source_id, 0) + total_count

    for source_id, target_id in all_edges:
        direct_count = int(transition_counts.get((source_id, target_id), 0))
        overlap_count = int(overlap_counts.get((source_id, target_id), 0))
        total_count = direct_count + overlap_count
        reverse_count = int(
            transition_counts.get((target_id, source_id), 0)
            + overlap_counts.get((target_id, source_id), 0)
        )
        token_overlap = len(queue_tokens.get(source_id, set()) & queue_tokens.get(target_id, set()))
        rows.append(
            {
                "source_queue_id": source_id,
                "source_queue_name": reverse_map[source_id],
                "target_queue_id": target_id,
                "target_queue_name": reverse_map[target_id],
                "count": total_count,
                "direct_transition_count": direct_count,
                "issue_overlap_count": overlap_count,
                "token_overlap_count": token_overlap,
                "source_probability": float(total_count / max(row_totals.get(source_id, 1), 1)),
                "symmetric_count": int(total_count + reverse_count),
            }
        )
    graph = pd.DataFrame(rows)

    parent = {queue_id: queue_id for queue_id in reverse_map}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    if not graph.empty:
        for queue_id in sorted(reverse_map):
            outgoing = graph[graph["source_queue_id"] == queue_id].sort_values(
                ["symmetric_count", "issue_overlap_count", "token_overlap_count", "count", "target_queue_id"],
                ascending=[False, False, False, False, True],
            )
            for _, edge in outgoing.head(max(1, top_edges)).iterrows():
                has_token_overlap = int(edge["token_overlap_count"]) > 0
                has_material_support = (
                    int(edge["symmetric_count"]) >= max(1, min_support)
                    and float(edge["source_probability"]) >= 0.05
                )
                if has_token_overlap or has_material_support:
                    union(int(edge["source_queue_id"]), int(edge["target_queue_id"]))

    root_to_group: dict[int, int] = {}
    macro_rows: list[dict[str, int | str]] = []
    next_group = 0
    for queue_id in sorted(reverse_map):
        root = find(queue_id)
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        macro_rows.append(
            {
                "queue_id": queue_id,
                "queue_name": reverse_map[queue_id],
                "macro_group_id": root_to_group[root],
            }
        )
    return graph, pd.DataFrame(macro_rows)


def _assign_service_units(config: AppConfig, frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    raw_units = np.ceil(
        enriched["time_to_resolution_hours"].astype(float) / max(config.jira.slot_hours, 1)
    ).astype(int)
    enriched["raw_service_units"] = np.maximum(raw_units, 1)
    mode = config.jira.service_unit_mode.strip().lower()
    if mode == "clip_linear":
        enriched["service_units"] = enriched["raw_service_units"].clip(
            lower=1,
            upper=int(config.jira.max_service_units),
        ).astype(int)
        return enriched
    if mode == "quantile_spread":
        ranks = enriched["raw_service_units"].rank(method="average", pct=True)
        scaled = np.ceil(ranks * int(config.jira.max_service_units)).astype(int)
        enriched["service_units"] = scaled.clip(lower=1, upper=int(config.jira.max_service_units)).astype(int)
        return enriched
    raise ValueError(f"Unsupported jira.service_unit_mode: {config.jira.service_unit_mode}")


def build_jira_benchmark(config: AppConfig) -> dict[str, str | int | float | dict[str, int]]:
    if config.jira.raw_issues_path is None:
        raise ValueError("jira.raw_issues_path is required for jira_public benchmark construction")

    benchmark_dir = Path(config.jira.benchmark_dir or config.paths.input_csv.parent)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dataset = benchmark_dataset_path(config)
    benchmark_dataset.parent.mkdir(parents=True, exist_ok=True)

    issues = read_tabular_export(config.jira.raw_issues_path)
    normalized = _normalize_issue_table(config, issues)

    if config.jira.instance_filter:
        allowed = {str(value) for value in config.jira.instance_filter}
        normalized = normalized[normalized["instance_id"].astype(str).isin(allowed)].copy()

    normalized = normalized[normalized["queue_name"].astype(str).str.len() > 0].copy()
    normalized = normalized[normalized["text"].astype(str).str.len() > 0].copy()
    normalized = normalized.dropna(subset=["created_at", "resolved_at"]).copy()
    normalized["time_to_resolution_hours"] = (
        (normalized["resolved_at"] - normalized["created_at"]).dt.total_seconds() / 3600.0
    )
    normalized = normalized[normalized["time_to_resolution_hours"] > 0].copy()

    history_export = pd.DataFrame(
        columns=["issue_id", "change_time", "field_name", "old_value", "new_value", "change_author_mask"]
    )
    history_stats = pd.DataFrame(
        columns=["issue_id", "initial_queue_name", "final_queue_name", "num_queue_changes", "first_change_time"]
    )
    if config.jira.raw_history_path is not None and Path(config.jira.raw_history_path).exists():
        history_export = _normalize_history_table(config, read_tabular_export(config.jira.raw_history_path))
        history_stats = _history_queue_stats(history_export)

    if not history_stats.empty:
        normalized = normalized.merge(history_stats, on="issue_id", how="left")
        normalized["initial_queue_name"] = normalized["initial_queue_name"].fillna("").astype(str)
        normalized["final_queue_name"] = normalized["final_queue_name"].fillna(normalized["queue_name"]).astype(str)
        normalized["num_queue_changes"] = normalized["num_queue_changes"].fillna(0).astype(int)
        normalized["time_to_first_queue_change_hours"] = (
            (normalized["first_change_time"] - normalized["created_at"]).dt.total_seconds() / 3600.0
        )
    else:
        normalized["initial_queue_name"] = normalized["queue_name"].astype(str)
        normalized["final_queue_name"] = normalized["queue_name"].astype(str)
        normalized["num_queue_changes"] = 0
        normalized["time_to_first_queue_change_hours"] = np.nan

    queue_counts = normalized["queue_name"].value_counts()
    if config.jira.queue_top_n is not None:
        keep_queues = queue_counts.nlargest(int(config.jira.queue_top_n)).index.tolist()
        normalized = normalized[normalized["queue_name"].isin(keep_queues)].copy()
        queue_counts = normalized["queue_name"].value_counts()
    keep_queues = queue_counts[queue_counts >= int(config.jira.min_queue_size)].index.tolist()
    normalized = normalized[normalized["queue_name"].isin(keep_queues)].copy()
    if normalized.empty:
        raise ValueError("No Jira issues remain after queue-size filtering")

    queue_names = sorted(normalized["queue_name"].astype(str).unique().tolist())
    queue_map = {queue_name: index for index, queue_name in enumerate(queue_names)}
    normalized["queue"] = normalized["queue_name"]
    normalized["queue_id"] = normalized["queue"].map(queue_map).astype(int)
    normalized["initial_queue_id"] = normalized["initial_queue_name"].map(queue_map).fillna(-1).astype(int)
    normalized["final_queue_id"] = normalized["queue_id"].astype(int)
    normalized = _assign_service_units(config, normalized)
    normalized["benchmark_split"] = ""

    benchmark_frame = normalized[
        [
            "issue_id",
            "instance_id",
            "project_key",
            "subject",
            "body",
            "text",
            "created_at",
            "resolved_at",
            "priority_raw",
            "priority",
            "issue_type",
            "queue",
            "queue_id",
            "initial_queue_name",
            "initial_queue_id",
            "final_queue_name",
            "final_queue_id",
            "num_queue_changes",
            "time_to_first_queue_change_hours",
            "time_to_resolution_hours",
            "raw_service_units",
            "service_units",
            "language",
            "benchmark_split",
            "component_tokens",
        ]
    ].sort_values("issue_id").reset_index(drop=True)

    queue_distribution = (
        benchmark_frame["queue"]
        .value_counts()
        .rename_axis("queue_name")
        .reset_index(name="count")
    )
    queue_distribution["share"] = queue_distribution["count"] / max(len(benchmark_frame), 1)
    queue_distribution["queue_id"] = queue_distribution["queue_name"].map(queue_map).astype(int)
    queue_distribution = queue_distribution.sort_values("queue_id").reset_index(drop=True)

    reroute_stats = pd.DataFrame(
        {
            "num_issues": [int(len(benchmark_frame))],
            "reroute_rate": [float((benchmark_frame["num_queue_changes"] > 0).mean())],
            "mean_num_queue_changes": [float(benchmark_frame["num_queue_changes"].mean())],
            "median_time_to_first_queue_change_hours": [
                float(benchmark_frame["time_to_first_queue_change_hours"].dropna().median())
                if benchmark_frame["time_to_first_queue_change_hours"].notna().any()
                else float("nan")
            ],
            "mean_service_units": [float(benchmark_frame["service_units"].mean())],
        }
    )

    benchmark_frame.to_parquet(benchmark_dataset, index=False)
    filtered_history = history_export[history_export["issue_id"].isin(benchmark_frame["issue_id"])].copy()
    filtered_history.to_parquet(benchmark_history_path(config), index=False)
    queue_distribution.to_csv(benchmark_queue_distribution_path(config), index=False)
    reroute_stats.to_csv(benchmark_reroute_stats_path(config), index=False)
    issue_tokens = _issue_component_tokens(normalized, filtered_history)
    transition_graph, macro_groups = _build_transition_graph(
        filtered_history,
        queue_map,
        benchmark_frame,
        issue_tokens,
        min_support=config.jira.macro_group_min_support,
        top_edges=config.jira.macro_group_top_edges,
    )
    transition_graph.to_csv(benchmark_transition_graph_path(config), index=False)
    macro_groups.to_csv(benchmark_macro_groups_path(config), index=False)

    metadata = {
        "dataset_profile": config.data.dataset_profile,
        "benchmark_dataset": str(benchmark_dataset.resolve()),
        "issues_input": str(Path(config.jira.raw_issues_path).resolve()),
        "history_input": (
            str(Path(config.jira.raw_history_path).resolve())
            if config.jira.raw_history_path is not None
            else None
        ),
        "benchmark_dir": str(benchmark_dir.resolve()),
        "instance_filter": list(config.jira.instance_filter),
        "queue_field": config.jira.queue_field,
        "macro_group_min_support": int(config.jira.macro_group_min_support),
        "macro_group_top_edges": int(config.jira.macro_group_top_edges),
        "service_unit_mode": config.jira.service_unit_mode,
        "min_queue_size": int(config.jira.min_queue_size),
        "slot_hours": int(config.jira.slot_hours),
        "max_service_units": int(config.jira.max_service_units),
        "num_issues": int(len(benchmark_frame)),
        "num_queues": int(len(queue_map)),
        "queue_counts": {
            queue_name: int(count)
            for queue_name, count in queue_distribution.set_index("queue_name")["count"].to_dict().items()
        },
        "text_sparsity": {
            "missing_subject_rate": float((benchmark_frame["subject"].astype(str).str.len() == 0).mean()),
            "missing_body_rate": float((benchmark_frame["body"].astype(str).str.len() == 0).mean()),
        },
        "priority_raw_levels": sorted(benchmark_frame["priority_raw"].astype(str).unique().tolist()),
        "priority_levels_collapsed": sorted(benchmark_frame["priority"].astype(str).unique().tolist()),
        "resolution_target": config.jira.resolution_target,
        "raw_service_unit_quantiles": {
            str(key): float(value)
            for key, value in benchmark_frame["raw_service_units"].quantile([0.5, 0.9, 0.95, 0.99, 1.0]).to_dict().items()
        },
        "service_unit_quantiles": {
            str(key): float(value)
            for key, value in benchmark_frame["service_units"].quantile([0.5, 0.9, 0.95, 0.99, 1.0]).to_dict().items()
        },
        "num_macro_groups": int(macro_groups["macro_group_id"].nunique()) if not macro_groups.empty else 0,
    }
    with benchmark_metadata_path(config).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "benchmark_dataset": str(benchmark_dataset),
        "benchmark_dir": str(benchmark_dir),
        "metadata_path": str(benchmark_metadata_path(config)),
        "num_issues": int(len(benchmark_frame)),
        "num_queues": int(len(queue_map)),
    }
