from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class TicketRecord:
    ticket_id: int | str
    text: str
    true_queue: int
    priority: str
    language: str


@dataclass(slots=True)
class TicketFeatures:
    ticket_id: int | str
    prob_by_queue: list[float]
    entropy: float
    complexity_score: float
    predicted_queue: int
    p_max: float


@dataclass(slots=True)
class QueueState:
    time_slot: int
    backlog_by_queue: list[int]
    served_by_queue: list[int]


@dataclass(slots=True)
class TicketState:
    ticket_id: int | str
    arrival_slot: int
    true_queue_id: int
    chosen_queue_id: int
    priority: str
    language: str
    remaining_service: int
    misrouted: bool


@dataclass(slots=True)
class EnvironmentSpec:
    environment_id: str
    service_model: str
    delay_mode: str
    capacity_mode: str
    slot_order: str
    serve_new_same_slot: bool


@dataclass(slots=True)
class ScenarioFixture:
    scenario: str
    seed: int
    shuffled_ticket_ids: list[int | str]
    arrival_counts: list[int]
    service_units_by_ticket: dict[int | str, int]
    fixture_signature: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueueSkillArtifacts:
    embedding_model_name: str
    queue_centroids: list[list[float]]
    similarity_columns: list[str]
    kappa_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RoutingDecision:
    ticket_id: int | str
    chosen_queue: int
    score: float
    criteria_raw: dict[str, list[float]] | None = None
    criteria_fuzzy: dict[str, list[list[float]]] | None = None
    policy_name: str = ""
    runner_up_queue: int | None = None
    closeness: list[float] | None = None
    reason_snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SimulationConfig:
    arrival_mode: str
    lambda_base: int
    burst_schedule: dict[str, int]
    mu_by_queue: list[int]
    sla_by_priority: dict[str, int]
    misroute_penalty_slots: int
    seeds: list[int]
    queue_discipline: str


@dataclass(slots=True)
class RunMetrics:
    mean_wait: float
    p95_wait: float
    p99_wait: float
    mean_backlog: float
    sla_violation_rate: float
    accuracy: float
    macro_f1: float
    misroute_rate: float
    avg_cost: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ComplexityStats:
    token_min: float
    token_max: float
    entropy_min: float
    entropy_max: float
    technical_min: float
    technical_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
