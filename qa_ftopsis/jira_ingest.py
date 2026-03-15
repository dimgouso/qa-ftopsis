from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

QUEUE_SPLIT_PATTERN = re.compile(r"[;,|+]")


def read_tabular_export(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(file_path, lines=True)
    if suffix == ".json":
        payload = json.loads(file_path.read_text())
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        raise ValueError(f"Unsupported JSON structure in {file_path}")
    raise ValueError(f"Unsupported table format: {file_path}")


def coerce_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def normalize_queue_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "[]"}:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, list):
                raw_items = payload
            else:
                stripped = text.strip("[]")
                raw_items = [part.strip().strip("'\"") for part in stripped.split(",")]
        else:
            raw_items = QUEUE_SPLIT_PATTERN.split(text)

    tokens: list[str] = []
    for item in raw_items:
        cleaned = str(item).strip().lower()
        if cleaned and cleaned not in {"nan", "none", "null"}:
            tokens.append(cleaned)
    return sorted(dict.fromkeys(tokens))


def canonical_queue_combo(value: object) -> str:
    tokens = normalize_queue_tokens(value)
    return " + ".join(tokens)


def normalize_queue_value(value: object) -> str:
    tokens = normalize_queue_tokens(value)
    return tokens[0] if tokens else ""


def first_present_column(frame: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate]
    return pd.Series([default] * len(frame), index=frame.index)
