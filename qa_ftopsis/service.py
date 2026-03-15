from __future__ import annotations

import hashlib

import numpy as np

from qa_ftopsis.config import HeavyTailSettings


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def sample_service_units(
    complexity_scores: list[float],
    service_model: str,
    heavy_tail: HeavyTailSettings,
    seed: int,
) -> list[int]:
    if service_model == "deterministic":
        return [1 for _ in complexity_scores]

    if service_model != "heavy_tail":
        raise ValueError(f"Unsupported service model: {service_model}")

    rng = np.random.default_rng(seed)
    service_units: list[int] = []
    for complexity_score in complexity_scores:
        complexity = float(np.clip(complexity_score, 0.0, 1.0))
        p_long = heavy_tail.long_job_prob_base + heavy_tail.long_job_prob_scale * complexity
        poisson_rate = heavy_tail.base_poisson_bias + heavy_tail.base_poisson_scale * complexity
        base = 1 + int(rng.poisson(poisson_rate))
        tail = 0
        if float(rng.random()) < p_long:
            tail = int(np.ceil(rng.pareto(heavy_tail.pareto_alpha) + 1.0))
        service_units.append(int(np.clip(base + tail, 1, heavy_tail.max_service_units)))
    return service_units
