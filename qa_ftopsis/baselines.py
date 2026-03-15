from __future__ import annotations

import numpy as np


def backlog_pressure(backlog_by_queue: np.ndarray, mu_by_queue: np.ndarray) -> np.ndarray:
    return np.asarray(backlog_by_queue, dtype=float) / np.maximum(np.asarray(mu_by_queue, dtype=float), 1.0)


def misroute_cost(prob_vector: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(np.asarray(prob_vector, dtype=float), 1e-6, 1.0))


def compute_delay_risk(
    backlog_by_queue: np.ndarray,
    mu_by_queue: np.ndarray,
    complexity_score: float,
    delay_mode: str,
    kappa_vector: np.ndarray | None = None,
    delay_vector: np.ndarray | None = None,
) -> np.ndarray:
    base = ((np.asarray(backlog_by_queue, dtype=float) + 1.0) / np.maximum(np.asarray(mu_by_queue, dtype=float), 1.0)) * (
        0.5 + 0.5 * float(complexity_score)
    )
    normalized_mode = delay_mode.strip().lower()
    if normalized_mode == "redundant_baseline":
        return base
    if normalized_mode == "embedding_kappa":
        if kappa_vector is None:
            raise ValueError("embedding_kappa delay mode requires kappa_vector")
        return base * np.asarray(kappa_vector, dtype=float)
    if normalized_mode == "learned_jira_delay":
        if delay_vector is None:
            raise ValueError("learned_jira_delay mode requires delay_vector")
        return ((np.asarray(backlog_by_queue, dtype=float) + 1.0) / np.maximum(np.asarray(mu_by_queue, dtype=float), 1.0)) * np.clip(
            np.asarray(delay_vector, dtype=float),
            1.0,
            None,
        )
    raise ValueError(f"Unsupported delay mode: {delay_mode}")
