from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from qa_ftopsis.config import ScenarioSettings
from qa_ftopsis.environment import load_scenario_fixture
from qa_ftopsis.policies import BasePolicy
from qa_ftopsis.skill_features import load_macro_groups, load_queue_compatibility
from qa_ftopsis.types import EnvironmentSpec, QueueState, RunMetrics, TicketState


@dataclass(slots=True)
class SimulationRunOutput:
    metrics: dict[str, Any]
    decisions: pd.DataFrame
    backlog_trace: pd.DataFrame


def build_mu_by_queue(
    train_df: pd.DataFrame,
    lambda_base: int,
    rho_target: float,
    capacity_mode: str,
) -> np.ndarray:
    num_queues = int(train_df["true_queue_id"].nunique())
    normalized_mode = capacity_mode.strip().lower()
    if normalized_mode == "uniform_one":
        return np.ones(num_queues, dtype=int)
    if normalized_mode == "proportional_to_pj":
        queue_share = train_df["true_queue_id"].value_counts(normalize=True).sort_index()
        mu = np.zeros(num_queues, dtype=int)
        for queue_id in range(num_queues):
            share = float(queue_share.get(queue_id, 0.0))
            mu[queue_id] = max(1, round((lambda_base * share) / rho_target))
        return mu
    if normalized_mode == "empirical_service_demand":
        if "service_units" not in train_df.columns:
            raise ValueError("empirical_service_demand capacity mode requires service_units in train_df")
        queue_share = train_df["true_queue_id"].value_counts(normalize=True).sort_index()
        service_by_queue = (
            train_df.groupby("true_queue_id", dropna=False)["service_units"].mean().to_dict()
        )
        mu = np.zeros(num_queues, dtype=int)
        for queue_id in range(num_queues):
            share = float(queue_share.get(queue_id, 0.0))
            mean_service = float(service_by_queue.get(queue_id, 1.0))
            mu[queue_id] = max(1, round((lambda_base * share * mean_service) / rho_target))
        return mu
    if normalized_mode == "empirical_service_median":
        if "service_units" not in train_df.columns:
            raise ValueError("empirical_service_median capacity mode requires service_units in train_df")
        queue_share = train_df["true_queue_id"].value_counts(normalize=True).sort_index()
        service_by_queue = (
            train_df.groupby("true_queue_id", dropna=False)["service_units"].median().to_dict()
        )
        mu = np.zeros(num_queues, dtype=int)
        for queue_id in range(num_queues):
            share = float(queue_share.get(queue_id, 0.0))
            median_service = float(service_by_queue.get(queue_id, 1.0))
            mu[queue_id] = max(1, round((lambda_base * share * median_service) / rho_target))
        return mu
    raise ValueError(f"Unsupported capacity_mode: {capacity_mode}")


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("prob_q_")],
        key=lambda value: int(value.split("_")[-1]),
    )


def _kappa_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("kappa_q_")],
        key=lambda value: int(value.split("_")[-1]),
    )


def _delay_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("delay_q_")],
        key=lambda value: int(value.split("_")[-1]),
    )


def _serialize(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _make_ticket_record(
    row: dict[str, Any],
    probability_columns: list[str],
    kappa_columns: list[str],
    delay_columns: list[str],
    macro_group_members: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    ticket = dict(row)
    ticket["prob_vector"] = np.array([float(row[column]) for column in probability_columns], dtype=float)
    ticket["p_max"] = float(row.get("p_max", np.max(ticket["prob_vector"])))
    if kappa_columns:
        ticket["kappa_vector"] = np.array([float(row[column]) for column in kappa_columns], dtype=float)
    if delay_columns:
        ticket["delay_vector"] = np.array([float(row[column]) for column in delay_columns], dtype=float)
    if macro_group_members is not None:
        predicted_queue_id = int(row.get("predicted_queue_id", np.argmax(ticket["prob_vector"])))
        ticket["macro_group_candidate_indices"] = list(
            macro_group_members.get(predicted_queue_id, [predicted_queue_id])
        )
    return ticket


def _serve_queue(queue: deque[TicketState], capacity: int) -> tuple[list[TicketState], deque[TicketState]]:
    if capacity <= 0 or not queue:
        return [], queue
    served_batch: list[TicketState] = []
    while capacity > 0 and queue:
        ticket = queue.popleft()
        ticket.remaining_service -= 1
        served_batch.append(ticket)
        capacity -= 1

    unfinished = [ticket for ticket in served_batch if ticket.remaining_service > 0]
    for ticket in reversed(unfinished):
        queue.appendleft(ticket)
    return served_batch, queue


def simulate_single_run(
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    policy: BasePolicy,
    environment: EnvironmentSpec,
    scenario_name: str,
    scenario: ScenarioSettings,
    seed: int,
    sla_deadlines: dict[str, int],
    split_name: str,
    sla_profile: str,
    model_dir: str,
    include_records: bool = True,
) -> SimulationRunOutput:
    fixture = load_scenario_fixture(
        model_dir=model_dir,
        split_name=split_name,
        environment_id=environment.environment_id,
        scenario_name=scenario_name,
        seed=seed,
    )
    rows_by_ticket_id = {
        row["ticket_id"]: row
        for row in split_df.to_dict(orient="records")
    }
    probability_columns = _probability_columns(split_df)
    kappa_columns = _kappa_columns(split_df) if environment.delay_mode == "embedding_kappa" else []
    delay_columns = _delay_columns(split_df) if environment.delay_mode == "learned_jira_delay" else []
    queue_compatibility = {}
    macro_group_members: dict[int, list[int]] | None = None
    compatibility_metadata = Path(model_dir) / "skill_features" / "embedding_metadata.json"
    if compatibility_metadata.exists():
        queue_compatibility = load_queue_compatibility(model_dir)
        macro_groups = load_macro_groups(model_dir)
        members: dict[int, list[int]] = {}
        for queue_id, group_id in macro_groups.items():
            members.setdefault(group_id, []).append(int(queue_id))
        macro_group_members = {
            queue_id: sorted(members.get(group_id, [queue_id]))
            for queue_id, group_id in macro_groups.items()
        }
    mu_by_queue = build_mu_by_queue(
        train_df,
        scenario.lambda_base,
        scenario.rho_target,
        environment.capacity_mode,
    )
    num_queues = len(mu_by_queue)

    service_queues = [deque() for _ in range(num_queues)]
    pending_wrong = [list() for _ in range(num_queues)]
    decisions_store: dict[int | str, dict[str, Any]] = {}
    completed_rows: list[dict[str, Any]] = []
    backlog_rows: list[dict[str, Any]] = []

    ticket_order = list(fixture.shuffled_ticket_ids)
    arrival_counts = list(fixture.arrival_counts)
    cursor = 0
    slot_index = 0

    while cursor < len(ticket_order) or any(service_queues) or any(pending_wrong):
        served_by_queue = np.zeros(num_queues, dtype=int)

        for queue_id in range(num_queues):
            served_tickets, service_queues[queue_id] = _serve_queue(
                service_queues[queue_id],
                int(mu_by_queue[queue_id]),
            )
            served_by_queue[queue_id] = len(served_tickets)
            for current in served_tickets:
                if current.remaining_service > 0:
                    continue
                wait_slots = slot_index - int(current.arrival_slot) + 1
                deadline = int(sla_deadlines[str(current.priority).lower()])
                completed_rows.append(
                    {
                        **decisions_store[current.ticket_id],
                        "arrival_slot": int(current.arrival_slot),
                        "completion_slot": slot_index,
                        "wait_slots": wait_slots,
                        "sla_deadline": deadline,
                        "sla_violated": bool(wait_slots > deadline),
                        "misrouted": bool(current.misrouted),
                    }
                )

        for queue_id in range(num_queues):
            if pending_wrong[queue_id]:
                for ticket in pending_wrong[queue_id]:
                    service_queues[int(ticket.true_queue_id)].append(ticket)
                pending_wrong[queue_id] = []

        arrivals_this_slot = arrival_counts[slot_index] if slot_index < len(arrival_counts) else 0
        arrivals_this_slot = min(arrivals_this_slot, len(ticket_order) - cursor)
        for _ in range(arrivals_this_slot):
            ticket_id = ticket_order[cursor]
            cursor += 1
            row = rows_by_ticket_id[ticket_id]
            ticket = _make_ticket_record(
                row,
                probability_columns,
                kappa_columns,
                delay_columns,
                macro_group_members=macro_group_members,
            )
            backlog_counts = [len(queue) for queue in service_queues]
            queue_state = QueueState(
                time_slot=slot_index,
                backlog_by_queue=backlog_counts,
                served_by_queue=served_by_queue.astype(int).tolist(),
            )
            decision = policy.route(ticket, queue_state, mu_by_queue)
            true_queue_id = int(ticket["true_queue_id"])
            chosen_queue_id = int(decision.chosen_queue)
            compatible_queues = queue_compatibility.get(true_queue_id, [true_queue_id])
            compatible_route = chosen_queue_id in compatible_queues and chosen_queue_id != true_queue_id
            service_units = int(row.get("service_units", fixture.service_units_by_ticket[ticket["ticket_id"]]))
            if compatible_route and "delay_vector" in ticket:
                chosen_service = int(max(1, round(float(ticket["delay_vector"][chosen_queue_id]))))
            else:
                chosen_service = int(fixture.service_units_by_ticket[ticket["ticket_id"]])
            ticket_state = TicketState(
                ticket_id=ticket["ticket_id"],
                arrival_slot=slot_index,
                true_queue_id=true_queue_id,
                chosen_queue_id=chosen_queue_id,
                priority=str(ticket["priority"]),
                language=str(ticket["language"]),
                remaining_service=chosen_service,
                misrouted=bool(chosen_queue_id != true_queue_id and not compatible_route),
            )
            decisions_store[ticket["ticket_id"]] = {
                "ticket_id": ticket["ticket_id"],
                "policy": policy.name,
                "environment_id": environment.environment_id,
                "service_model": environment.service_model,
                "delay_mode": environment.delay_mode,
                "capacity_mode": environment.capacity_mode,
                "scenario": scenario_name,
                "seed": seed,
                "split": split_name,
                "sla_profile": sla_profile,
                "true_queue_id": true_queue_id,
                "priority": str(ticket["priority"]),
                "language": str(ticket["language"]),
                "chosen_queue_id": chosen_queue_id,
                "predicted_queue_id": int(ticket["predicted_queue_id"]),
                "score": float(decision.score),
                "runner_up_queue_id": decision.runner_up_queue,
                "reason_snippet": decision.reason_snippet,
                "criteria_raw_json": _serialize(decision.criteria_raw),
                "criteria_fuzzy_json": _serialize(decision.criteria_fuzzy),
                "closeness_json": _serialize(decision.closeness),
                "complexity_score": float(ticket["complexity_score"]),
                "p_max": float(ticket["p_max"]),
                "initial_remaining_service": int(ticket_state.remaining_service),
                "compatible_route": bool(compatible_route),
                "true_queue_service_units": int(service_units),
            }
            if ticket_state.misrouted:
                pending_wrong[chosen_queue_id].append(ticket_state)
            else:
                service_queues[chosen_queue_id].append(ticket_state)

        total_backlog = sum(len(queue) for queue in service_queues) + sum(len(queue) for queue in pending_wrong)
        backlog_rows.append(
            {
                "slot": slot_index,
                "policy": policy.name,
                "environment_id": environment.environment_id,
                "service_model": environment.service_model,
                "delay_mode": environment.delay_mode,
                "capacity_mode": environment.capacity_mode,
                "scenario": scenario_name,
                "seed": seed,
                "split": split_name,
                "sla_profile": sla_profile,
                "total_backlog": total_backlog,
            }
        )
        slot_index += 1

    decisions_df = pd.DataFrame(completed_rows)
    backlog_df = pd.DataFrame(backlog_rows)

    metrics = RunMetrics(
        mean_wait=float(decisions_df["wait_slots"].mean()),
        p95_wait=float(np.percentile(decisions_df["wait_slots"], 95)),
        p99_wait=float(np.percentile(decisions_df["wait_slots"], 99)),
        mean_backlog=float(backlog_df["total_backlog"].mean()),
        sla_violation_rate=float(decisions_df["sla_violated"].mean()),
        accuracy=float(accuracy_score(decisions_df["true_queue_id"], decisions_df["chosen_queue_id"])),
        macro_f1=float(
            f1_score(decisions_df["true_queue_id"], decisions_df["chosen_queue_id"], average="macro")
        ),
        misroute_rate=float(decisions_df["misrouted"].mean()),
        avg_cost=float(
            (
                decisions_df["wait_slots"]
                + 5.0 * decisions_df["sla_violated"].astype(float)
                + 2.0 * decisions_df["misrouted"].astype(float)
            ).mean()
        ),
    )

    output_metrics = {
        "policy": policy.name,
        "environment_id": environment.environment_id,
        "service_model": environment.service_model,
        "delay_mode": environment.delay_mode,
        "capacity_mode": environment.capacity_mode,
        "scenario": scenario_name,
        "seed": seed,
        "split": split_name,
        "sla_profile": sla_profile,
        "lambda_base": scenario.lambda_base,
        "rho_target": scenario.rho_target,
        "mu_by_queue_json": json.dumps(mu_by_queue.astype(int).tolist()),
        **metrics.to_dict(),
    }

    if not include_records:
        decisions_df = pd.DataFrame()
        backlog_df = pd.DataFrame()

    return SimulationRunOutput(metrics=output_metrics, decisions=decisions_df, backlog_trace=backlog_df)


def run_policy_across_scenarios(
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    policy: BasePolicy,
    environment: EnvironmentSpec,
    scenarios: dict[str, ScenarioSettings],
    seeds: list[int],
    sla_deadlines: dict[str, int],
    split_name: str,
    sla_profile: str,
    model_dir: str,
    include_records: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    decision_frames: list[pd.DataFrame] = []
    backlog_frames: list[pd.DataFrame] = []

    for scenario_name, scenario in scenarios.items():
        for seed in seeds:
            result = simulate_single_run(
                split_df=split_df,
                train_df=train_df,
                policy=policy,
                environment=environment,
                scenario_name=scenario_name,
                scenario=scenario,
                seed=seed,
                sla_deadlines=sla_deadlines,
                split_name=split_name,
                sla_profile=sla_profile,
                model_dir=model_dir,
                include_records=include_records,
            )
            metric_rows.append(result.metrics)
            if include_records and not result.decisions.empty:
                decision_frames.append(result.decisions)
            if include_records and not result.backlog_trace.empty:
                backlog_frames.append(result.backlog_trace)

    metrics_df = pd.DataFrame(metric_rows)
    decisions_df = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    backlog_df = pd.concat(backlog_frames, ignore_index=True) if backlog_frames else pd.DataFrame()
    return metrics_df, decisions_df, backlog_df
