from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from qa_ftopsis.jira_ingest import read_tabular_export

RAW_REQUIRED_COLUMNS = ["subject", "body", "queue", "priority", "language"]
PREPARED_SPLITS = ("train", "val_cal", "val_sim", "test")


def load_raw_dataset(
    csv_path: str | Path,
    *,
    priority_mapping: dict[str, str] | None = None,
    expected_languages: list[str] | None = None,
) -> pd.DataFrame:
    path = Path(csv_path)
    df = read_tabular_export(path)
    missing_columns = [column for column in RAW_REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared = df.copy()
    prepared["subject"] = prepared["subject"].fillna("").astype(str).str.strip()
    prepared["body"] = prepared["body"].fillna("").astype(str).str.strip()
    prepared["queue"] = prepared["queue"].astype(str).str.strip()
    prepared["priority"] = prepared["priority"].astype(str).str.strip().str.lower()
    prepared["language"] = prepared["language"].astype(str).str.strip().str.lower()
    prepared["priority_raw"] = prepared["priority"]

    normalized_mapping = {
        str(key).lower(): str(value).lower()
        for key, value in (priority_mapping or {}).items()
    }
    if normalized_mapping:
        mapped = prepared["priority_raw"].map(normalized_mapping)
        unknown = sorted(prepared.loc[mapped.isna(), "priority_raw"].dropna().unique().tolist())
        if unknown:
            raise ValueError(f"Unmapped priority levels: {unknown}")
        prepared["priority"] = mapped.astype(str)

    if expected_languages is not None:
        allowed_languages = {str(value).lower() for value in expected_languages}
        observed_languages = set(prepared["language"].dropna().astype(str).str.lower().unique().tolist())
        unexpected = sorted(observed_languages - allowed_languages)
        if unexpected:
            raise ValueError(
                f"Unexpected languages found: {unexpected}; expected subset of {sorted(allowed_languages)}"
            )
    return prepared


def build_text_column(df: pd.DataFrame) -> pd.Series:
    subject = df["subject"].fillna("").astype(str)
    body = df["body"].fillna("").astype(str)
    return (subject + "\n\n" + body).str.strip()


def build_queue_lookup(df: pd.DataFrame) -> pd.DataFrame:
    queue_names = sorted(df["queue"].astype(str).unique().tolist())
    return pd.DataFrame(
        {
            "queue_id": list(range(len(queue_names))),
            "queue_name": queue_names,
        }
    )


def _stratify_key(df: pd.DataFrame, stratify_fields: list[str]) -> pd.Series:
    if not stratify_fields:
        raise ValueError("stratify_fields must not be empty")
    missing_fields = [field for field in stratify_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Missing stratify fields: {missing_fields}")
    key = df[stratify_fields[0]].astype(str)
    for field in stratify_fields[1:]:
        key = key + "__" + df[field].astype(str)
    return key


def _split_dataset(
    df: pd.DataFrame,
    random_state: int,
    stratify_fields: list[str],
) -> dict[str, pd.DataFrame]:
    stratify_key = _stratify_key(df, stratify_fields)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,
        random_state=random_state,
        stratify=stratify_key,
    )

    temp_key = _stratify_key(temp_df, stratify_fields)
    val_all_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_key,
    )

    val_key = _stratify_key(val_all_df, stratify_fields)
    val_cal_df, val_sim_df = train_test_split(
        val_all_df,
        test_size=0.50,
        random_state=random_state,
        stratify=val_key,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val_cal": val_cal_df.reset_index(drop=True),
        "val_sim": val_sim_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def _build_split_queue_stats(
    splits: dict[str, pd.DataFrame],
    queue_lookup: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str | int | float]] = []
    for split_name, split_df in splits.items():
        counts = split_df["true_queue_id"].astype(int).value_counts().sort_index()
        total = max(len(split_df), 1)
        for lookup in queue_lookup.to_dict(orient="records"):
            queue_id = int(lookup["queue_id"])
            count = int(counts.get(queue_id, 0))
            rows.append(
                {
                    "split": split_name,
                    "queue_id": queue_id,
                    "queue_name": str(lookup["queue_name"]),
                    "count": count,
                    "share": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def prepare_dataset(
    input_csv: str | Path,
    output_dir: str | Path,
    random_state: int = 7,
    *,
    dataset_profile: str = "default",
    priority_mapping: dict[str, str] | None = None,
    stratify_fields: list[str] | None = None,
    expected_languages: list[str] | None = None,
) -> dict[str, str | int | dict[str, int] | dict[str, float] | list[str] | dict[str, str]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset(
        input_csv,
        priority_mapping=priority_mapping,
        expected_languages=expected_languages,
    )
    df = df.reset_index(drop=True)
    if "ticket_id" in df.columns:
        df["ticket_id"] = df["ticket_id"].astype(str)
    elif "issue_id" in df.columns:
        df["ticket_id"] = df["issue_id"].astype(str)
    else:
        df["ticket_id"] = df.index.astype(int)
    df["text"] = build_text_column(df)

    queue_lookup = build_queue_lookup(df)
    queue_map = dict(zip(queue_lookup["queue_name"], queue_lookup["queue_id"]))
    df["true_queue_id"] = df["queue"].map(queue_map).astype(int)

    resolved_stratify_fields = list(stratify_fields or ["queue", "language"])
    splits = _split_dataset(
        df,
        random_state=random_state,
        stratify_fields=resolved_stratify_fields,
    )
    queue_lookup.to_csv(output_path / "queue_lookup.csv", index=False)
    split_queue_stats = _build_split_queue_stats(splits, queue_lookup)
    split_queue_stats.to_csv(output_path / "split_queue_stats.csv", index=False)

    split_sizes: dict[str, int] = {}
    for split_name, split_df in splits.items():
        split_sizes[split_name] = len(split_df)
        split_df.to_parquet(output_path / f"{split_name}.parquet", index=False)

    grouped_stats = split_queue_stats.groupby("split")["count"]
    min_queue_count_by_split = {
        split: int(value) for split, value in grouped_stats.min().to_dict().items()
    }
    median_queue_count_by_split = {
        split: float(value) for split, value in grouped_stats.median().to_dict().items()
    }
    max_queue_count_by_split = {
        split: int(value) for split, value in grouped_stats.max().to_dict().items()
    }

    metadata = {
        "input_csv": str(Path(input_csv).resolve()),
        "prepared_dir": str(output_path.resolve()),
        "dataset_profile": dataset_profile,
        "random_state": random_state,
        "stratify_fields": resolved_stratify_fields,
        "priority_mapping": {str(key): str(value) for key, value in (priority_mapping or {}).items()},
        "expected_languages": list(expected_languages) if expected_languages is not None else None,
        "split_sizes": split_sizes,
        "num_queues": int(queue_lookup["queue_id"].nunique()),
        "priority_raw_levels": sorted(df["priority_raw"].astype(str).unique().tolist()),
        "priority_levels_collapsed": sorted(df["priority"].astype(str).unique().tolist()),
        "min_queue_count_by_split": min_queue_count_by_split,
        "median_queue_count_by_split": median_queue_count_by_split,
        "max_queue_count_by_split": max_queue_count_by_split,
    }

    with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def load_prepared_split(prepared_dir: str | Path, split_name: str) -> pd.DataFrame:
    if split_name not in PREPARED_SPLITS:
        raise ValueError(f"Unsupported split: {split_name}")
    return pd.read_parquet(Path(prepared_dir) / f"{split_name}.parquet")


def load_queue_lookup(prepared_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(prepared_dir) / "queue_lookup.csv")


def load_split_queue_stats(prepared_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(prepared_dir) / "split_queue_stats.csv")
