from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qa_ftopsis.baselines import backlog_pressure, compute_delay_risk, misroute_cost
from qa_ftopsis.fuzzy import run_fuzzy_topsis
from qa_ftopsis.types import EnvironmentSpec, QueueState, RoutingDecision


def _resolve_candidates(
    primary_scores: np.ndarray,
    maximize: bool,
    prob_vector: np.ndarray,
    q_ratio: np.ndarray,
    candidate_indices: np.ndarray | None = None,
) -> tuple[int, int | None]:
    if candidate_indices is None:
        candidate_indices = np.arange(len(primary_scores))
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    if len(candidate_indices) == 0:
        raise ValueError("candidate_indices must not be empty")

    candidate_scores = primary_scores[candidate_indices]
    target = np.max(candidate_scores) if maximize else np.min(candidate_scores)
    candidates = candidate_indices[np.isclose(candidate_scores, target)]
    if len(candidates) > 1:
        best_prob = np.max(prob_vector[candidates])
        candidates = candidates[np.isclose(prob_vector[candidates], best_prob)]
    if len(candidates) > 1:
        best_ratio = np.min(q_ratio[candidates])
        candidates = candidates[np.isclose(q_ratio[candidates], best_ratio)]
    chosen = int(np.min(candidates))

    runner_up = None
    remaining = [
        index
        for index in candidate_indices[
            np.argsort(-(primary_scores[candidate_indices]) if maximize else primary_scores[candidate_indices])
        ]
        if index != chosen
    ]
    if remaining:
        runner_up = int(remaining[0])
    return chosen, runner_up


def _reason_snippet(chosen_queue: int, top_probability_queue: int, raw_criteria: np.ndarray) -> str:
    backlog_advantage = raw_criteria[top_probability_queue, 0] - raw_criteria[chosen_queue, 0]
    if chosen_queue != top_probability_queue and backlog_advantage > 0:
        return "Lower queue pressure outweighed lower classifier confidence."
    if chosen_queue == top_probability_queue:
        return "Classifier confidence aligned with acceptable queue pressure."
    return "Balanced backlog pressure, delay risk, and confidence favored the chosen queue."


def _ticket_kappa_vector(ticket: dict, num_queues: int) -> np.ndarray | None:
    if "kappa_vector" not in ticket:
        return None
    kappa = np.asarray(ticket["kappa_vector"], dtype=float)
    if len(kappa) != num_queues:
        raise ValueError("kappa_vector length does not match queue count")
    return kappa


def _ticket_delay_vector(ticket: dict, num_queues: int) -> np.ndarray | None:
    if "delay_vector" not in ticket:
        return None
    delay = np.asarray(ticket["delay_vector"], dtype=float)
    if len(delay) != num_queues:
        raise ValueError("delay_vector length does not match queue count")
    return delay


def _topk_candidate_indices(prob_vector: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("top-k gating requires k > 0")
    ordered = sorted(range(len(prob_vector)), key=lambda index: (-float(prob_vector[index]), index))
    return np.asarray(ordered[: min(k, len(ordered))], dtype=int)


def _hierarchical_candidate_indices(ticket: dict) -> np.ndarray | None:
    values = ticket.get("macro_group_candidate_indices")
    if values is None:
        return None
    return np.asarray(sorted({int(value) for value in values}), dtype=int)


def _merge_candidate_indices(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> np.ndarray | None:
    if left is None:
        return right
    if right is None:
        return left
    merged = sorted(set(left.tolist()) & set(right.tolist()))
    if not merged:
        return left
    return np.asarray(merged, dtype=int)


def _top_utility_candidates(
    utility: np.ndarray,
    limit: int,
    prob_vector: np.ndarray,
    q_ratio: np.ndarray,
    candidate_indices: np.ndarray | None,
) -> np.ndarray:
    if candidate_indices is None:
        candidate_indices = np.arange(len(utility))
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    ordered = sorted(
        candidate_indices.tolist(),
        key=lambda index: (float(utility[index]), -float(prob_vector[index]), float(q_ratio[index]), int(index)),
    )
    return np.asarray(ordered[: max(1, min(limit, len(ordered)))], dtype=int)


def _confidence_gate_decision(
    ticket: dict,
    prob_vector: np.ndarray,
    q_ratio: np.ndarray,
    candidate_indices: np.ndarray | None,
    policy_name: str,
) -> RoutingDecision:
    chosen_queue, runner_up = _resolve_candidates(
        prob_vector,
        True,
        prob_vector,
        q_ratio,
        candidate_indices=candidate_indices,
    )
    return RoutingDecision(
        ticket_id=ticket["ticket_id"],
        chosen_queue=chosen_queue,
        score=float(prob_vector[chosen_queue]),
        policy_name=policy_name,
        runner_up_queue=runner_up,
        reason_snippet="Confidence gate kept the classifier-preferred queue.",
    )


class BasePolicy:
    name: str = ""

    def route(
        self,
        ticket: dict,
        queue_state: QueueState,
        mu_by_queue: np.ndarray,
    ) -> RoutingDecision:
        raise NotImplementedError


@dataclass(slots=True)
class ClassifierOnlyPolicy(BasePolicy):
    name: str = "classifier_only"

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        q_ratio = backlog_pressure(np.asarray(queue_state.backlog_by_queue, dtype=float), mu_by_queue)
        chosen_queue, runner_up = _resolve_candidates(prob_vector, True, prob_vector, q_ratio)
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(prob_vector[chosen_queue]),
            policy_name=self.name,
            runner_up_queue=runner_up,
            reason_snippet="Highest calibrated queue probability.",
        )


@dataclass(slots=True)
class JSQPolicy(BasePolicy):
    name: str = "jsq"

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        q_ratio = backlog_pressure(np.asarray(queue_state.backlog_by_queue, dtype=float), mu_by_queue)
        chosen_queue, runner_up = _resolve_candidates(q_ratio, False, prob_vector, q_ratio)
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(-q_ratio[chosen_queue]),
            policy_name=self.name,
            runner_up_queue=runner_up,
            reason_snippet="Smallest normalized queue backlog.",
        )


@dataclass(slots=True)
class JSQTopKPolicy(BasePolicy):
    topk_k: int = 3
    name: str = "jsq_topk"

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        q_ratio = backlog_pressure(np.asarray(queue_state.backlog_by_queue, dtype=float), mu_by_queue)
        candidates = _topk_candidate_indices(prob_vector, self.topk_k)
        chosen_queue, runner_up = _resolve_candidates(
            q_ratio,
            False,
            prob_vector,
            q_ratio,
            candidate_indices=candidates,
        )
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(-q_ratio[chosen_queue]),
            policy_name=self.name,
            runner_up_queue=runner_up,
            reason_snippet=f"Smallest normalized queue backlog within Top-{self.topk_k}.",
        )


@dataclass(slots=True)
class MaxWeightDelayPolicy(BasePolicy):
    environment: EnvironmentSpec
    name: str = "maxweight_delay"

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        backlog = np.asarray(queue_state.backlog_by_queue, dtype=float)
        q_ratio = backlog_pressure(backlog, mu_by_queue)
        delay_risk = compute_delay_risk(
            backlog_by_queue=backlog,
            mu_by_queue=mu_by_queue,
            complexity_score=float(ticket["complexity_score"]),
            delay_mode=self.environment.delay_mode,
            kappa_vector=_ticket_kappa_vector(ticket, len(mu_by_queue)),
            delay_vector=_ticket_delay_vector(ticket, len(mu_by_queue)),
        )
        chosen_queue, runner_up = _resolve_candidates(delay_risk, False, prob_vector, q_ratio)
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(-delay_risk[chosen_queue]),
            policy_name=self.name,
            runner_up_queue=runner_up,
            reason_snippet="Minimum queue-dependent delay risk.",
        )


@dataclass(slots=True)
class MaxWeightProbPolicy(BasePolicy):
    alpha: float = 1.0
    name: str = "maxweight_prob"

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        backlog = np.asarray(queue_state.backlog_by_queue, dtype=float)
        q_ratio = backlog_pressure(backlog, mu_by_queue)
        utility = ((backlog + 1.0) / np.maximum(mu_by_queue, 1)) + (
            self.alpha * (1.0 - prob_vector)
        )
        chosen_queue, runner_up = _resolve_candidates(utility, False, prob_vector, q_ratio)
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(-utility[chosen_queue]),
            policy_name=self.name,
            runner_up_queue=runner_up,
            reason_snippet="Queue pressure plus misroute risk penalty.",
        )


@dataclass(slots=True)
class QAFuzzyTopsisPolicy(BasePolicy):
    environment: EnvironmentSpec
    weights: list[float]
    confidence_gate: float | None = None
    deltas: list[float] | None = None
    name: str = "qa_ftopsis"

    def __post_init__(self) -> None:
        if self.deltas is None:
            self.deltas = [0.10, 0.20, 0.15]

    def _candidate_indices(self, ticket: dict, prob_vector: np.ndarray) -> np.ndarray | None:
        return None

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        backlog = np.asarray(queue_state.backlog_by_queue, dtype=float)
        q_ratio = backlog_pressure(backlog, mu_by_queue)
        candidates = self._candidate_indices(ticket, prob_vector)
        if self.confidence_gate is not None and float(np.max(prob_vector)) >= float(self.confidence_gate):
            return _confidence_gate_decision(
                ticket=ticket,
                prob_vector=prob_vector,
                q_ratio=q_ratio,
                candidate_indices=candidates,
                policy_name=self.name,
            )

        delay_risk = compute_delay_risk(
            backlog_by_queue=backlog,
            mu_by_queue=mu_by_queue,
            complexity_score=float(ticket["complexity_score"]),
            delay_mode=self.environment.delay_mode,
            kappa_vector=_ticket_kappa_vector(ticket, len(mu_by_queue)),
            delay_vector=_ticket_delay_vector(ticket, len(mu_by_queue)),
        )
        risk = misroute_cost(prob_vector)
        raw_criteria = np.column_stack([q_ratio, delay_risk, risk])

        result = run_fuzzy_topsis(raw_criteria, deltas=self.deltas, weights=self.weights)
        closeness = result["closeness"]
        chosen_queue, runner_up = _resolve_candidates(
            closeness,
            True,
            prob_vector,
            q_ratio,
            candidate_indices=candidates,
        )
        criteria_raw = {
            "backlog_pressure": raw_criteria[:, 0].round(6).tolist(),
            "predicted_delay_risk": raw_criteria[:, 1].round(6).tolist(),
            "misroute_risk": raw_criteria[:, 2].round(6).tolist(),
        }
        criteria_fuzzy = {
            "weighted_backlog_pressure": result["weighted_matrix"][:, 0, :].round(6).tolist(),
            "weighted_predicted_delay_risk": result["weighted_matrix"][:, 1, :].round(6).tolist(),
            "weighted_misroute_risk": result["weighted_matrix"][:, 2, :].round(6).tolist(),
        }
        top_probability_queue = int(np.argmax(prob_vector))
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(closeness[chosen_queue]),
            criteria_raw=criteria_raw,
            criteria_fuzzy=criteria_fuzzy,
            policy_name=self.name,
            runner_up_queue=runner_up,
            closeness=closeness.round(6).tolist(),
            reason_snippet=_reason_snippet(chosen_queue, top_probability_queue, raw_criteria),
        )


@dataclass(slots=True)
class QAFuzzyTopsisTopKPolicy(BasePolicy):
    environment: EnvironmentSpec
    topk_k: int
    weights: list[float]
    confidence_gate: float | None = None
    deltas: list[float] | None = None
    name: str = "qa_ftopsis_topk"

    def __post_init__(self) -> None:
        if self.deltas is None:
            self.deltas = [0.10, 0.20, 0.15]

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        backlog = np.asarray(queue_state.backlog_by_queue, dtype=float)
        q_ratio = backlog_pressure(backlog, mu_by_queue)
        candidates = _topk_candidate_indices(prob_vector, self.topk_k)
        if self.confidence_gate is not None and float(np.max(prob_vector)) >= float(self.confidence_gate):
            return _confidence_gate_decision(
                ticket=ticket,
                prob_vector=prob_vector,
                q_ratio=q_ratio,
                candidate_indices=candidates,
                policy_name=self.name,
            )

        delay_risk = compute_delay_risk(
            backlog_by_queue=backlog,
            mu_by_queue=mu_by_queue,
            complexity_score=float(ticket["complexity_score"]),
            delay_mode=self.environment.delay_mode,
            kappa_vector=_ticket_kappa_vector(ticket, len(mu_by_queue)),
            delay_vector=_ticket_delay_vector(ticket, len(mu_by_queue)),
        )
        risk = misroute_cost(prob_vector)
        raw_criteria = np.column_stack([q_ratio, delay_risk, risk])

        result = run_fuzzy_topsis(raw_criteria, deltas=self.deltas, weights=self.weights)
        closeness = result["closeness"]
        chosen_queue, runner_up = _resolve_candidates(
            closeness,
            True,
            prob_vector,
            q_ratio,
            candidate_indices=candidates,
        )
        criteria_raw = {
            "backlog_pressure": raw_criteria[:, 0].round(6).tolist(),
            "predicted_delay_risk": raw_criteria[:, 1].round(6).tolist(),
            "misroute_risk": raw_criteria[:, 2].round(6).tolist(),
        }
        criteria_fuzzy = {
            "weighted_backlog_pressure": result["weighted_matrix"][:, 0, :].round(6).tolist(),
            "weighted_predicted_delay_risk": result["weighted_matrix"][:, 1, :].round(6).tolist(),
            "weighted_misroute_risk": result["weighted_matrix"][:, 2, :].round(6).tolist(),
        }
        top_probability_queue = int(np.argmax(prob_vector))
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(closeness[chosen_queue]),
            criteria_raw=criteria_raw,
            criteria_fuzzy=criteria_fuzzy,
            policy_name=self.name,
            runner_up_queue=runner_up,
            closeness=closeness.round(6).tolist(),
            reason_snippet=(
                f"{_reason_snippet(chosen_queue, top_probability_queue, raw_criteria)} "
                f"Restricted to Top-{self.topk_k}."
            ),
        )


@dataclass(slots=True)
class QAFuzzyTopsisHierarchicalPolicy(QAFuzzyTopsisPolicy):
    name: str = "qa_ftopsis_hierarchical"

    def _candidate_indices(self, ticket: dict, prob_vector: np.ndarray) -> np.ndarray | None:
        return _hierarchical_candidate_indices(ticket)


@dataclass(slots=True)
class QAFuzzyTopsisHybridPolicy(BasePolicy):
    environment: EnvironmentSpec
    weights: list[float]
    alpha: float
    hybrid_mix: float
    topk_k: int = 2
    confidence_gate: float | None = None
    deltas: list[float] | None = None
    name: str = "qa_ftopsis_hybrid"

    def __post_init__(self) -> None:
        if self.deltas is None:
            self.deltas = [0.10, 0.20, 0.15]

    def route(self, ticket: dict, queue_state: QueueState, mu_by_queue: np.ndarray) -> RoutingDecision:
        prob_vector = np.asarray(ticket["prob_vector"], dtype=float)
        backlog = np.asarray(queue_state.backlog_by_queue, dtype=float)
        q_ratio = backlog_pressure(backlog, mu_by_queue)
        hierarchical = _hierarchical_candidate_indices(ticket)
        topk = _topk_candidate_indices(prob_vector, self.topk_k)
        candidates = _merge_candidate_indices(hierarchical, topk)
        if self.confidence_gate is not None and float(np.max(prob_vector)) >= float(self.confidence_gate):
            return _confidence_gate_decision(
                ticket=ticket,
                prob_vector=prob_vector,
                q_ratio=q_ratio,
                candidate_indices=candidates,
                policy_name=self.name,
            )

        delay_risk = compute_delay_risk(
            backlog_by_queue=backlog,
            mu_by_queue=mu_by_queue,
            complexity_score=float(ticket["complexity_score"]),
            delay_mode=self.environment.delay_mode,
            kappa_vector=_ticket_kappa_vector(ticket, len(mu_by_queue)),
            delay_vector=_ticket_delay_vector(ticket, len(mu_by_queue)),
        )
        risk = misroute_cost(prob_vector)
        raw_criteria = np.column_stack([q_ratio, delay_risk, risk])
        topsis = run_fuzzy_topsis(raw_criteria, deltas=self.deltas, weights=self.weights)
        closeness = np.asarray(topsis["closeness"], dtype=float)

        utility = ((backlog + 1.0) / np.maximum(mu_by_queue, 1)) + (self.alpha * (1.0 - prob_vector))
        utility_candidates = _top_utility_candidates(
            utility=utility,
            limit=max(2, self.topk_k),
            prob_vector=prob_vector,
            q_ratio=q_ratio,
            candidate_indices=candidates,
        )
        if np.max(utility[utility_candidates]) - np.min(utility[utility_candidates]) <= 1e-12:
            utility_score = np.ones_like(utility, dtype=float)
        else:
            normalized = np.zeros_like(utility, dtype=float)
            utility_span = float(np.max(utility[utility_candidates]) - np.min(utility[utility_candidates]))
            normalized_candidates = (
                utility[utility_candidates] - np.min(utility[utility_candidates])
            ) / utility_span
            normalized[utility_candidates] = normalized_candidates
            utility_score = 1.0 - normalized
        hybrid_score = utility_score.copy()
        hybrid_score[utility_candidates] = (
            (1.0 - float(self.hybrid_mix)) * utility_score[utility_candidates]
        ) + (float(self.hybrid_mix) * closeness[utility_candidates])

        chosen_queue, runner_up = _resolve_candidates(
            hybrid_score,
            True,
            prob_vector,
            q_ratio,
            candidate_indices=utility_candidates,
        )
        top_probability_queue = int(np.argmax(prob_vector))
        return RoutingDecision(
            ticket_id=ticket["ticket_id"],
            chosen_queue=chosen_queue,
            score=float(hybrid_score[chosen_queue]),
            criteria_raw={
                "backlog_pressure": raw_criteria[:, 0].round(6).tolist(),
                "predicted_delay_risk": raw_criteria[:, 1].round(6).tolist(),
                "misroute_risk": raw_criteria[:, 2].round(6).tolist(),
                "hybrid_score": hybrid_score.round(6).tolist(),
            },
            criteria_fuzzy={
                "weighted_backlog_pressure": topsis["weighted_matrix"][:, 0, :].round(6).tolist(),
                "weighted_predicted_delay_risk": topsis["weighted_matrix"][:, 1, :].round(6).tolist(),
                "weighted_misroute_risk": topsis["weighted_matrix"][:, 2, :].round(6).tolist(),
            },
            policy_name=self.name,
            runner_up_queue=runner_up,
            closeness=closeness.round(6).tolist(),
            reason_snippet=(
                f"{_reason_snippet(chosen_queue, top_probability_queue, raw_criteria)} "
                f"Anchored on MaxWeightProb and reranked fuzzily within the best utility band."
            ),
        )


def build_policy(
    name: str,
    *,
    environment: EnvironmentSpec | None = None,
    alpha: float = 1.0,
    weights: list[float] | None = None,
    confidence_gate: float | None = None,
    topk_k: int = 3,
    hybrid_mix: float = 0.5,
) -> BasePolicy:
    normalized_name = name.strip().lower()
    if normalized_name == "classifier_only":
        return ClassifierOnlyPolicy()
    if normalized_name == "jsq":
        return JSQPolicy()
    if normalized_name == "jsq_topk":
        return JSQTopKPolicy(topk_k=topk_k)
    if normalized_name == "maxweight_delay":
        if environment is None:
            raise ValueError("maxweight_delay requires environment")
        return MaxWeightDelayPolicy(environment=environment)
    if normalized_name == "maxweight_prob":
        return MaxWeightProbPolicy(alpha=alpha)
    if normalized_name == "qa_ftopsis":
        if environment is None:
            raise ValueError("qa_ftopsis requires environment")
        return QAFuzzyTopsisPolicy(
            environment=environment,
            weights=weights or [0.30, 0.40, 0.30],
            confidence_gate=confidence_gate,
        )
    if normalized_name == "qa_ftopsis_topk":
        if environment is None:
            raise ValueError("qa_ftopsis_topk requires environment")
        return QAFuzzyTopsisTopKPolicy(
            environment=environment,
            topk_k=topk_k,
            weights=weights or [0.30, 0.40, 0.30],
            confidence_gate=confidence_gate,
        )
    if normalized_name == "qa_ftopsis_hierarchical":
        if environment is None:
            raise ValueError("qa_ftopsis_hierarchical requires environment")
        return QAFuzzyTopsisHierarchicalPolicy(
            environment=environment,
            weights=weights or [0.30, 0.40, 0.30],
            confidence_gate=confidence_gate,
        )
    if normalized_name == "qa_ftopsis_hybrid":
        if environment is None:
            raise ValueError("qa_ftopsis_hybrid requires environment")
        return QAFuzzyTopsisHybridPolicy(
            environment=environment,
            weights=weights or [0.30, 0.40, 0.30],
            alpha=alpha,
            hybrid_mix=hybrid_mix,
            topk_k=topk_k,
            confidence_gate=confidence_gate,
        )
    raise ValueError(f"Unsupported policy: {name}")
