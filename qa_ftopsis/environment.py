from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qa_ftopsis.config import AppConfig, ScenarioSettings
from qa_ftopsis.service import sample_service_units, stable_seed
from qa_ftopsis.types import EnvironmentSpec, ScenarioFixture


def _service_label(service_model: str) -> str:
    normalized = service_model.strip().lower()
    if normalized == "heavy_tail":
        return "heavytail"
    if normalized == "empirical":
        return "empirical"
    return normalized


def _delay_label(delay_mode: str) -> str:
    normalized = delay_mode.strip().lower()
    if normalized == "redundant_baseline":
        return "redundant"
    if normalized == "embedding_kappa":
        return "embedding"
    if normalized == "learned_jira_delay":
        return "learnedjira"
    return normalized.replace("_", "")


def build_environment_specs(config: AppConfig) -> list[EnvironmentSpec]:
    environments: list[EnvironmentSpec] = []
    for service_model in config.simulation.environment_matrix.service_models:
        for delay_mode in config.simulation.environment_matrix.delay_modes:
            service_label = _service_label(service_model)
            delay_label = _delay_label(delay_mode)
            environments.append(
                EnvironmentSpec(
                    environment_id=f"{service_label}_{delay_label}",
                    service_model=service_model,
                    delay_mode=delay_mode,
                    capacity_mode=config.simulation.capacity_mode,
                    slot_order=config.simulation.slot_order,
                    serve_new_same_slot=config.simulation.serve_new_same_slot,
                )
            )
    return environments


def fixtures_root(model_dir: str | Path) -> Path:
    return Path(model_dir) / "fixtures"


def fixture_path(
    model_dir: str | Path,
    split_name: str,
    environment_id: str,
    scenario_name: str,
    seed: int,
) -> Path:
    return (
        fixtures_root(model_dir)
        / split_name
        / environment_id
        / scenario_name
        / f"seed_{seed}.json"
    )


def generate_arrival_counts(
    total_tickets: int,
    scenario: ScenarioSettings,
    seed: int,
) -> list[int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    counts: list[int] = []
    assigned = 0
    slot_index = 0
    while assigned < total_tickets:
        current_lambda = scenario.lambda_base
        if scenario.burst_lambda is not None and scenario.burst_interval is not None:
            if (
                slot_index >= scenario.burst_start_offset
                and (slot_index - scenario.burst_start_offset) % scenario.burst_interval == 0
            ):
                current_lambda = scenario.burst_lambda
        count = int(rng.poisson(current_lambda))
        counts.append(count)
        assigned += count
        slot_index += 1
    return counts


def _fixture_signature(
    environment: EnvironmentSpec,
    scenario: ScenarioSettings,
) -> dict[str, int | float | str | None]:
    return {
        "lambda_base": int(scenario.lambda_base),
        "rho_target": float(scenario.rho_target),
        "burst_lambda": int(scenario.burst_lambda) if scenario.burst_lambda is not None else None,
        "burst_interval": int(scenario.burst_interval) if scenario.burst_interval is not None else None,
        "burst_start_offset": int(scenario.burst_start_offset),
        "arrival_mode": str(scenario.arrival_mode),
        "service_model": str(environment.service_model),
        "delay_mode": str(environment.delay_mode),
        "capacity_mode": str(environment.capacity_mode),
    }


def _generate_fixture(
    split_df: pd.DataFrame,
    environment: EnvironmentSpec,
    scenario_name: str,
    scenario: ScenarioSettings,
    split_name: str,
    heavy_tail_config,
    seed: int,
) -> ScenarioFixture:
    shuffled = split_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled_ticket_ids = shuffled["ticket_id"].tolist()
    arrival_counts = generate_arrival_counts(
        total_tickets=len(shuffled_ticket_ids),
        scenario=scenario,
        seed=stable_seed(split_name, scenario_name, seed, "arrivals"),
    )
    if environment.service_model == "empirical":
        if "service_units" not in shuffled.columns:
            raise ValueError("empirical service_model requires service_units column")
        service_units = shuffled["service_units"].astype(int).tolist()
    else:
        service_units = sample_service_units(
            complexity_scores=shuffled["complexity_score"].astype(float).tolist(),
            service_model=environment.service_model,
            heavy_tail=heavy_tail_config,
            seed=stable_seed(split_name, scenario_name, seed, environment.service_model, "service"),
        )
    service_units_by_ticket = dict(zip(shuffled_ticket_ids, service_units))
    return ScenarioFixture(
        scenario=scenario_name,
        seed=seed,
        shuffled_ticket_ids=shuffled_ticket_ids,
        arrival_counts=arrival_counts,
        service_units_by_ticket=service_units_by_ticket,
        fixture_signature=_fixture_signature(environment, scenario),
    )


def ensure_environment_fixtures(config: AppConfig, split_name: str, split_df: pd.DataFrame) -> None:
    for environment in build_environment_specs(config):
        for scenario_name, scenario in config.simulation.scenarios.items():
            for seed in config.simulation.seeds:
                path = fixture_path(
                    config.paths.model_dir,
                    split_name,
                    environment.environment_id,
                    scenario_name,
                    seed,
                )
                expected_signature = _fixture_signature(environment, scenario)
                if path.exists():
                    try:
                        existing_payload = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        existing_payload = {}
                    if existing_payload.get("fixture_signature") == expected_signature:
                        continue
                fixture = _generate_fixture(
                    split_df=split_df,
                    environment=environment,
                    scenario_name=scenario_name,
                    scenario=scenario,
                    split_name=split_name,
                    heavy_tail_config=config.simulation.heavy_tail,
                    seed=seed,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(fixture.to_dict(), handle, indent=2)


def load_scenario_fixture(
    model_dir: str | Path,
    split_name: str,
    environment_id: str,
    scenario_name: str,
    seed: int,
) -> ScenarioFixture:
    path = fixture_path(model_dir, split_name, environment_id, scenario_name, seed)
    payload = json.loads(path.read_text())

    def deserialize_ticket_id(value: object) -> int | str:
        if isinstance(value, int):
            return value
        text = str(value)
        if text.isdigit():
            return int(text)
        return text

    return ScenarioFixture(
        scenario=str(payload["scenario"]),
        seed=int(payload["seed"]),
        shuffled_ticket_ids=[deserialize_ticket_id(value) for value in payload["shuffled_ticket_ids"]],
        arrival_counts=[int(value) for value in payload["arrival_counts"]],
        service_units_by_ticket={
            deserialize_ticket_id(ticket_id): int(units)
            for ticket_id, units in payload["service_units_by_ticket"].items()
        },
        fixture_signature=dict(payload.get("fixture_signature", {})),
    )
