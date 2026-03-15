from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from qa_ftopsis.types import ComplexityStats

TECHNICAL_TOKEN_PATTERN = re.compile(
    r"(?i)(?:err[_-]?[a-z0-9]+|[a-z0-9]*\d[a-z0-9./_-]*|v\d+(?:\.\d+)+|[A-Z]{2,}[A-Z0-9_-]*)"
)


def probability_column_names(queue_ids: Iterable[int]) -> list[str]:
    return [f"prob_q_{queue_id}" for queue_id in queue_ids]


def normalized_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    safe_prob = np.clip(prob_matrix, 1e-12, 1.0)
    entropies = -np.sum(safe_prob * np.log(safe_prob), axis=1)
    max_entropy = math.log(prob_matrix.shape[1])
    if max_entropy <= 0:
        return np.zeros(prob_matrix.shape[0], dtype=float)
    return entropies / max_entropy


def _token_count(text: str) -> int:
    return len(text.split())


def _technical_token_count(text: str) -> int:
    return len(TECHNICAL_TOKEN_PATTERN.findall(text))


def compute_complexity_raw_features(texts: pd.Series, prob_matrix: np.ndarray) -> pd.DataFrame:
    entropy_values = normalized_entropy(prob_matrix)
    token_counts = texts.fillna("").astype(str).map(_token_count).astype(float)
    technical_counts = texts.fillna("").astype(str).map(_technical_token_count).astype(float)
    return pd.DataFrame(
        {
            "token_count": token_counts,
            "entropy": entropy_values,
            "technical_token_count": technical_counts,
        }
    )


def fit_complexity_stats(raw_features: pd.DataFrame) -> ComplexityStats:
    return ComplexityStats(
        token_min=float(raw_features["token_count"].min()),
        token_max=float(raw_features["token_count"].max()),
        entropy_min=float(raw_features["entropy"].min()),
        entropy_max=float(raw_features["entropy"].max()),
        technical_min=float(raw_features["technical_token_count"].min()),
        technical_max=float(raw_features["technical_token_count"].max()),
    )


def _normalize(values: pd.Series, low: float, high: float) -> pd.Series:
    denominator = high - low
    if denominator <= 0:
        return pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    return ((values - low) / denominator).clip(0.0, 1.0)


def apply_complexity_stats(raw_features: pd.DataFrame, stats: ComplexityStats) -> pd.DataFrame:
    normalized = raw_features.copy()
    normalized["norm_token_count"] = _normalize(
        normalized["token_count"], stats.token_min, stats.token_max
    )
    normalized["norm_entropy"] = _normalize(
        normalized["entropy"], stats.entropy_min, stats.entropy_max
    )
    normalized["norm_technical_token_count"] = _normalize(
        normalized["technical_token_count"], stats.technical_min, stats.technical_max
    )
    normalized["complexity_score"] = (
        0.50 * normalized["norm_token_count"]
        + 0.30 * normalized["norm_entropy"]
        + 0.20 * normalized["norm_technical_token_count"]
    ).clip(0.0, 1.0)
    return normalized


def build_feature_frame(
    split_df: pd.DataFrame,
    prob_matrix: np.ndarray,
    stats: ComplexityStats,
    queue_ids: list[int],
) -> pd.DataFrame:
    raw_features = compute_complexity_raw_features(split_df["text"], prob_matrix)
    normalized = apply_complexity_stats(raw_features, stats)
    enriched = split_df.copy()
    probability_columns = probability_column_names(queue_ids)
    probabilities_df = pd.DataFrame(prob_matrix, columns=probability_columns, index=split_df.index)
    enriched = pd.concat([enriched, probabilities_df, normalized], axis=1)
    enriched["predicted_queue_id"] = np.argmax(prob_matrix, axis=1).astype(int)
    enriched["p_max"] = np.max(prob_matrix, axis=1).astype(float)
    return enriched
