from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathSettings:
    input_csv: Path
    prepared_dir: Path
    model_dir: Path
    run_root: Path
    report_root: Path


@dataclass(slots=True)
class DataSettings:
    random_state: int = 7
    dataset_profile: str = "default"
    priority_mapping: dict[str, str] = field(default_factory=dict)
    stratify_fields: list[str] = field(default_factory=lambda: ["queue", "language"])
    expected_languages: list[str] | None = None
    split_ratios: dict[str, float] = field(
        default_factory=lambda: {
            "train": 0.60,
            "val_cal": 0.10,
            "val_sim": 0.10,
            "test": 0.20,
        }
    )


@dataclass(slots=True)
class ModelSettings:
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    word_max_features: int = 40_000
    char_max_features: int = 30_000
    max_iter: int = 1_000
    sublinear_tf: bool = True
    prepend_language_token: bool = True


@dataclass(slots=True)
class ScenarioSettings:
    lambda_base: int
    rho_target: float
    burst_lambda: int | None = None
    burst_interval: int | None = None
    burst_start_offset: int = 0
    arrival_mode: str = "poisson"

    def burst_schedule(self) -> dict[str, int]:
        if self.burst_lambda is None or self.burst_interval is None:
            return {}
        return {
            "burst_lambda": self.burst_lambda,
            "burst_interval": self.burst_interval,
            "burst_start_offset": self.burst_start_offset,
        }


@dataclass(slots=True)
class EnvironmentMatrixSettings:
    service_models: list[str] = field(
        default_factory=lambda: ["deterministic", "heavy_tail"]
    )
    delay_modes: list[str] = field(
        default_factory=lambda: ["redundant_baseline", "embedding_kappa"]
    )


@dataclass(slots=True)
class HeavyTailSettings:
    pareto_alpha: float = 1.5
    base_poisson_bias: float = 0.25
    base_poisson_scale: float = 0.75
    long_job_prob_base: float = 0.08
    long_job_prob_scale: float = 0.20
    max_service_units: int = 20


@dataclass(slots=True)
class SimulationSettings:
    queue_discipline: str = "fifo"
    slot_order: str = "serve_then_arrive"
    serve_new_same_slot: bool = False
    capacity_mode: str = "proportional_to_pj"
    seeds: list[int] = field(default_factory=lambda: [11, 17, 23, 29, 37])
    primary_sla: dict[str, int] = field(
        default_factory=lambda: {"high": 3, "medium": 6, "low": 10}
    )
    robustness_sla: dict[str, int] = field(
        default_factory=lambda: {"high": 2, "medium": 5, "low": 8}
    )
    scenarios: dict[str, ScenarioSettings] = field(default_factory=dict)
    environment_matrix: EnvironmentMatrixSettings = field(
        default_factory=EnvironmentMatrixSettings
    )
    heavy_tail: HeavyTailSettings = field(default_factory=HeavyTailSettings)


@dataclass(slots=True)
class SkillFeatureSettings:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    max_length: int = 256
    batch_size: int = 64
    device: str = "cpu"
    kappa_min: float = 0.8
    kappa_max: float = 1.4
    compatibility_top_k: int = 2


@dataclass(slots=True)
class JiraSettings:
    raw_issues_path: Path | None = None
    raw_history_path: Path | None = None
    benchmark_dir: Path | None = None
    api_base_url: str | None = None
    project_key: str | None = None
    jql: str | None = None
    max_issues: int = 2000
    page_size: int = 50
    instance_filter: list[str] = field(default_factory=list)
    queue_field: str = "component"
    min_queue_size: int = 300
    queue_top_n: int | None = None
    macro_group_min_support: int = 1
    macro_group_top_edges: int = 3
    service_unit_mode: str = "clip_linear"
    slot_hours: int = 8
    max_service_units: int = 40
    resolution_target: str = "time_to_resolution_hours"


@dataclass(slots=True)
class DelayModelSettings:
    model_type: str = "random_forest"
    n_estimators: int = 160
    max_depth: int | None = 8
    min_samples_leaf: int = 2


@dataclass(slots=True)
class TopKGateSettings:
    enabled: bool = False
    k: int = 3
    policy_names: list[str] = field(default_factory=lambda: ["qa_ftopsis", "jsq"])


@dataclass(slots=True)
class RoutingSettings:
    topk_gate: TopKGateSettings = field(default_factory=TopKGateSettings)


@dataclass(slots=True)
class TuningSettings:
    qa_weights: list[list[float]] = field(
        default_factory=lambda: [
            [0.30, 0.40, 0.30],
            [0.25, 0.35, 0.40],
            [0.35, 0.40, 0.25],
        ]
    )
    alpha_grid: list[float] = field(default_factory=lambda: [0.25, 0.50, 1.00, 2.00])
    qa_guard_thresholds: list[float | None] = field(
        default_factory=lambda: [None, 0.60, 0.70, 0.80]
    )
    hybrid_mix_grid: list[float] = field(default_factory=lambda: [0.25, 0.50, 0.75])
    selection_scenarios: list[str] = field(
        default_factory=lambda: ["high_load", "bursty"]
    )
    selection_mode: str = "avg_cost_first"


@dataclass(slots=True)
class ReportingSettings:
    explainability_sample_size: int = 100


@dataclass(slots=True)
class AppConfig:
    paths: PathSettings
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    skill_features: SkillFeatureSettings = field(default_factory=SkillFeatureSettings)
    jira: JiraSettings = field(default_factory=JiraSettings)
    delay_model: DelayModelSettings = field(default_factory=DelayModelSettings)
    routing: RoutingSettings = field(default_factory=RoutingSettings)
    tuning: TuningSettings = field(default_factory=TuningSettings)
    reporting: ReportingSettings = field(default_factory=ReportingSettings)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload["paths"].items():
            payload["paths"][key] = str(value)
        for key in ["raw_issues_path", "raw_history_path", "benchmark_dir"]:
            value = payload["jira"].get(key)
            if value is not None:
                payload["jira"][key] = str(value)
        return payload


def _resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _parse_scenarios(raw_scenarios: dict[str, dict[str, Any]] | None) -> dict[str, ScenarioSettings]:
    raw_scenarios = raw_scenarios or {}
    scenarios: dict[str, ScenarioSettings] = {}
    for name, values in raw_scenarios.items():
        scenarios[name] = ScenarioSettings(
            lambda_base=int(values["lambda_base"]),
            rho_target=float(values["rho_target"]),
            burst_lambda=values.get("burst_lambda"),
            burst_interval=values.get("burst_interval"),
            burst_start_offset=int(values.get("burst_start_offset", 0)),
            arrival_mode=str(values.get("arrival_mode", "poisson")),
        )
    return scenarios


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    base_dir = path.parent
    raw = yaml.safe_load(path.read_text()) or {}

    paths_raw = raw.get("paths", {})
    paths = PathSettings(
        input_csv=_resolve_path(
            base_dir,
            paths_raw.get("input_csv", "../archive(3)/dataset-tickets-multi-lang-4-20k.csv"),
        ),
        prepared_dir=_resolve_path(base_dir, paths_raw.get("prepared_dir", "../artifacts/prepared/default")),
        model_dir=_resolve_path(base_dir, paths_raw.get("model_dir", "../artifacts/models/default")),
        run_root=_resolve_path(base_dir, paths_raw.get("run_root", "../artifacts/runs")),
        report_root=_resolve_path(base_dir, paths_raw.get("report_root", "../artifacts/reports")),
    )

    data_raw = raw.get("data", {})
    model_raw = raw.get("model", {})
    simulation_raw = raw.get("simulation", {})
    skill_raw = raw.get("skill_features", {})
    jira_raw = raw.get("jira", {})
    delay_model_raw = raw.get("delay_model", {})
    routing_raw = raw.get("routing", {})
    tuning_raw = raw.get("tuning", {})
    reporting_raw = raw.get("reporting", {})

    environment_matrix_raw = simulation_raw.get("environment_matrix", {})
    heavy_tail_raw = simulation_raw.get("heavy_tail", {})
    topk_gate_raw = routing_raw.get("topk_gate", {})

    return AppConfig(
        paths=paths,
        data=DataSettings(
            random_state=int(data_raw.get("random_state", 7)),
            dataset_profile=str(data_raw.get("dataset_profile", "default")),
            priority_mapping={
                str(key).lower(): str(value).lower()
                for key, value in data_raw.get("priority_mapping", {}).items()
            },
            stratify_fields=list(data_raw.get("stratify_fields", ["queue", "language"])),
            expected_languages=(
                [str(value).lower() for value in data_raw.get("expected_languages", [])]
                if data_raw.get("expected_languages") is not None
                else None
            ),
            split_ratios=data_raw.get(
                "split_ratios",
                {"train": 0.60, "val_cal": 0.10, "val_sim": 0.10, "test": 0.20},
            ),
        ),
        model=ModelSettings(
            word_ngram_range=tuple(model_raw.get("word_ngram_range", [1, 2])),
            char_ngram_range=tuple(model_raw.get("char_ngram_range", [3, 5])),
            word_max_features=int(model_raw.get("word_max_features", 40_000)),
            char_max_features=int(model_raw.get("char_max_features", 30_000)),
            max_iter=int(model_raw.get("max_iter", 1_000)),
            sublinear_tf=bool(model_raw.get("sublinear_tf", True)),
            prepend_language_token=bool(model_raw.get("prepend_language_token", True)),
        ),
        simulation=SimulationSettings(
            queue_discipline=str(simulation_raw.get("queue_discipline", "fifo")),
            slot_order=str(simulation_raw.get("slot_order", "serve_then_arrive")),
            serve_new_same_slot=bool(simulation_raw.get("serve_new_same_slot", False)),
            capacity_mode=str(simulation_raw.get("capacity_mode", "proportional_to_pj")),
            seeds=list(simulation_raw.get("seeds", [11, 17, 23, 29, 37])),
            primary_sla=simulation_raw.get(
                "primary_sla", {"high": 3, "medium": 6, "low": 10}
            ),
            robustness_sla=simulation_raw.get(
                "robustness_sla", {"high": 2, "medium": 5, "low": 8}
            ),
            scenarios=_parse_scenarios(simulation_raw.get("scenarios")),
            environment_matrix=EnvironmentMatrixSettings(
                service_models=list(
                    environment_matrix_raw.get("service_models", ["deterministic", "heavy_tail"])
                ),
                delay_modes=list(
                    environment_matrix_raw.get("delay_modes", ["redundant_baseline", "embedding_kappa"])
                ),
            ),
            heavy_tail=HeavyTailSettings(
                pareto_alpha=float(heavy_tail_raw.get("pareto_alpha", 1.5)),
                base_poisson_bias=float(heavy_tail_raw.get("base_poisson_bias", 0.25)),
                base_poisson_scale=float(heavy_tail_raw.get("base_poisson_scale", 0.75)),
                long_job_prob_base=float(heavy_tail_raw.get("long_job_prob_base", 0.08)),
                long_job_prob_scale=float(heavy_tail_raw.get("long_job_prob_scale", 0.20)),
                max_service_units=int(heavy_tail_raw.get("max_service_units", 20)),
            ),
        ),
        skill_features=SkillFeatureSettings(
            model_name=str(
                skill_raw.get(
                    "model_name",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                )
            ),
            max_length=int(skill_raw.get("max_length", 256)),
            batch_size=int(skill_raw.get("batch_size", 64)),
            device=str(skill_raw.get("device", "cpu")),
            kappa_min=float(skill_raw.get("kappa_min", 0.8)),
            kappa_max=float(skill_raw.get("kappa_max", 1.4)),
            compatibility_top_k=int(skill_raw.get("compatibility_top_k", 2)),
        ),
        jira=JiraSettings(
            raw_issues_path=(
                _resolve_path(base_dir, jira_raw["raw_issues_path"])
                if jira_raw.get("raw_issues_path")
                else None
            ),
            raw_history_path=(
                _resolve_path(base_dir, jira_raw["raw_history_path"])
                if jira_raw.get("raw_history_path")
                else None
            ),
            benchmark_dir=(
                _resolve_path(base_dir, jira_raw["benchmark_dir"])
                if jira_raw.get("benchmark_dir")
                else _resolve_path(base_dir, "../artifacts/benchmarks/jira_public")
            ),
            api_base_url=(
                str(jira_raw.get("api_base_url"))
                if jira_raw.get("api_base_url") is not None
                else None
            ),
            project_key=(
                str(jira_raw.get("project_key"))
                if jira_raw.get("project_key") is not None
                else None
            ),
            jql=(
                str(jira_raw.get("jql"))
                if jira_raw.get("jql") is not None
                else None
            ),
            max_issues=int(jira_raw.get("max_issues", 2000)),
            page_size=int(jira_raw.get("page_size", 50)),
            instance_filter=list(jira_raw.get("instance_filter", [])),
            queue_field=str(jira_raw.get("queue_field", "component")),
            min_queue_size=int(jira_raw.get("min_queue_size", 300)),
            queue_top_n=(
                int(jira_raw["queue_top_n"])
                if jira_raw.get("queue_top_n") is not None
                else None
            ),
            macro_group_min_support=int(jira_raw.get("macro_group_min_support", 1)),
            macro_group_top_edges=int(jira_raw.get("macro_group_top_edges", 3)),
            service_unit_mode=str(jira_raw.get("service_unit_mode", "clip_linear")),
            slot_hours=int(jira_raw.get("slot_hours", 8)),
            max_service_units=int(jira_raw.get("max_service_units", 40)),
            resolution_target=str(
                jira_raw.get("resolution_target", "time_to_resolution_hours")
            ),
        ),
        delay_model=DelayModelSettings(
            model_type=str(delay_model_raw.get("model_type", "random_forest")),
            n_estimators=int(delay_model_raw.get("n_estimators", 160)),
            max_depth=(
                int(delay_model_raw["max_depth"])
                if delay_model_raw.get("max_depth") is not None
                else None
            ),
            min_samples_leaf=int(delay_model_raw.get("min_samples_leaf", 2)),
        ),
        routing=RoutingSettings(
            topk_gate=TopKGateSettings(
                enabled=bool(topk_gate_raw.get("enabled", False)),
                k=int(topk_gate_raw.get("k", 3)),
                policy_names=list(topk_gate_raw.get("policy_names", ["qa_ftopsis", "jsq"])),
            )
        ),
        tuning=TuningSettings(
            qa_weights=tuning_raw.get(
                "qa_weights",
                [[0.30, 0.40, 0.30], [0.25, 0.35, 0.40], [0.35, 0.40, 0.25]],
            ),
            alpha_grid=list(tuning_raw.get("alpha_grid", [0.25, 0.50, 1.00, 2.00])),
            qa_guard_thresholds=list(
                tuning_raw.get("qa_guard_thresholds", [None, 0.60, 0.70, 0.80])
            ),
            hybrid_mix_grid=list(tuning_raw.get("hybrid_mix_grid", [0.25, 0.50, 0.75])),
            selection_scenarios=list(
                tuning_raw.get("selection_scenarios", ["high_load", "bursty"])
            ),
            selection_mode=str(tuning_raw.get("selection_mode", "avg_cost_first")),
        ),
        reporting=ReportingSettings(
            explainability_sample_size=int(
                reporting_raw.get("explainability_sample_size", 100)
            )
        ),
    )


def save_config_snapshot(config: AppConfig, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
