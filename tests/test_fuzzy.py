from __future__ import annotations

import numpy as np

from qa_ftopsis.baselines import misroute_cost
from qa_ftopsis.fuzzy import normalize_cost_column, run_fuzzy_topsis, triangular_fuzzify, vertex_distance


def test_vertex_distance_is_zero_for_identical_numbers():
    triple = triangular_fuzzify(3.5, 0.1)
    assert vertex_distance(triple, triple) == 0.0


def test_constant_cost_column_normalizes_to_equal_values():
    column = [triangular_fuzzify(5.0, 0.1) for _ in range(3)]
    normalized = normalize_cost_column(column)
    assert normalized == [(1.0, 1.0, 1.0)] * 3


def test_fuzzy_topsis_handles_queue_invariant_column_without_nan():
    raw = np.array(
        [
            [1.0, 2.0, 0.3],
            [2.0, 2.0, 0.2],
            [3.0, 2.0, 0.1],
        ]
    )
    result = run_fuzzy_topsis(raw, deltas=[0.1, 0.2, 0.15], weights=[0.35, 0.4, 0.25])
    assert np.isfinite(result["closeness"]).all()


def test_misroute_cost_stays_finite_for_tiny_probabilities():
    risk = misroute_cost(np.array([1.0, 1e-12, 0.0]))
    assert np.isfinite(risk).all()
