from __future__ import annotations

import math

import numpy as np

Triangular = tuple[float, float, float]


def triangular_fuzzify(value: float, delta: float) -> Triangular:
    return (
        max(0.0, value * (1.0 - delta)),
        value,
        value * (1.0 + delta),
    )


def normalize_cost_column(column: list[Triangular]) -> list[Triangular]:
    if len({tuple(item) for item in column}) == 1:
        return [(1.0, 1.0, 1.0) for _ in column]

    lower_bounds = [triple[0] for triple in column]
    upper_bounds = [triple[2] for triple in column]
    min_lower = min(lower_bounds)
    max_upper = max(upper_bounds)
    denominator = max_upper - min_lower
    if denominator <= 0:
        return [(1.0, 1.0, 1.0) for _ in column]

    normalized: list[Triangular] = []
    for lower, middle, upper in column:
        normalized.append(
            (
                max(0.0, min(1.0, (max_upper - upper) / denominator)),
                max(0.0, min(1.0, (max_upper - middle) / denominator)),
                max(0.0, min(1.0, (max_upper - lower) / denominator)),
            )
        )
    return normalized


def weight_column(column: list[Triangular], weight: float) -> list[Triangular]:
    return [(a * weight, b * weight, c * weight) for a, b, c in column]


def vertex_distance(left: Triangular, right: Triangular) -> float:
    return math.sqrt(
        ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2) / 3.0
    )


def vector_distance(left: np.ndarray, right: np.ndarray) -> float:
    squared = 0.0
    for criterion_index in range(left.shape[0]):
        squared += vertex_distance(tuple(left[criterion_index]), tuple(right[criterion_index])) ** 2
    return math.sqrt(squared)


def run_fuzzy_topsis(
    raw_matrix: np.ndarray,
    deltas: list[float],
    weights: list[float],
) -> dict[str, np.ndarray]:
    if raw_matrix.ndim != 2:
        raise ValueError("raw_matrix must be 2-dimensional")

    num_alternatives, num_criteria = raw_matrix.shape
    if len(deltas) != num_criteria or len(weights) != num_criteria:
        raise ValueError("deltas and weights must match criterion count")

    fuzzy_matrix = np.zeros((num_alternatives, num_criteria, 3), dtype=float)
    normalized_matrix = np.zeros_like(fuzzy_matrix)
    weighted_matrix = np.zeros_like(fuzzy_matrix)

    for criterion_index in range(num_criteria):
        fuzzy_column = [
            triangular_fuzzify(float(raw_matrix[row_index, criterion_index]), deltas[criterion_index])
            for row_index in range(num_alternatives)
        ]
        normalized_column = normalize_cost_column(fuzzy_column)
        weighted_column = weight_column(normalized_column, weights[criterion_index])

        for row_index in range(num_alternatives):
            fuzzy_matrix[row_index, criterion_index] = np.array(fuzzy_column[row_index])
            normalized_matrix[row_index, criterion_index] = np.array(normalized_column[row_index])
            weighted_matrix[row_index, criterion_index] = np.array(weighted_column[row_index])

    ideal_best = weighted_matrix.max(axis=0)
    ideal_worst = weighted_matrix.min(axis=0)

    d_plus = np.zeros(num_alternatives, dtype=float)
    d_minus = np.zeros(num_alternatives, dtype=float)
    closeness = np.zeros(num_alternatives, dtype=float)
    for alternative_index in range(num_alternatives):
        d_plus[alternative_index] = vector_distance(weighted_matrix[alternative_index], ideal_best)
        d_minus[alternative_index] = vector_distance(weighted_matrix[alternative_index], ideal_worst)
        denominator = d_plus[alternative_index] + d_minus[alternative_index]
        closeness[alternative_index] = 0.5 if denominator <= 0 else d_minus[alternative_index] / denominator

    return {
        "fuzzy_matrix": fuzzy_matrix,
        "normalized_matrix": normalized_matrix,
        "weighted_matrix": weighted_matrix,
        "ideal_best": ideal_best,
        "ideal_worst": ideal_worst,
        "d_plus": d_plus,
        "d_minus": d_minus,
        "closeness": closeness,
    }
