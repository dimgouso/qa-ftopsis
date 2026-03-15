from __future__ import annotations

import json
import pandas as pd

from qa_ftopsis.baselines import misroute_cost
from qa_ftopsis.config import HeavyTailSettings, ScenarioSettings, load_config
from qa_ftopsis.data import prepare_dataset
from qa_ftopsis.environment import build_environment_specs, ensure_environment_fixtures, fixture_path, generate_arrival_counts
from qa_ftopsis.models import load_feature_split, train_classifier
from qa_ftopsis.policies import ClassifierOnlyPolicy, build_policy
from qa_ftopsis.service import sample_service_units
from qa_ftopsis.sim import build_mu_by_queue, simulate_single_run
from qa_ftopsis.types import EnvironmentSpec
from tests.conftest import build_synthetic_dataset, build_test_config


def test_generate_arrival_counts_is_reproducible():
    scenario = ScenarioSettings(lambda_base=4, rho_target=0.8)
    first = generate_arrival_counts(20, scenario, seed=5)
    second = generate_arrival_counts(20, scenario, seed=5)
    assert first == second


def test_misroute_cost_remains_finite_for_small_probabilities():
    values = misroute_cost(pd.Series([1.0, 1e-12, 0.0]).to_numpy())
    assert pd.Series(values).map(float).apply(pd.notna).all()
    assert all(value >= 0 for value in values)


def test_generate_arrival_counts_respects_burst_start_offset():
    scenario = ScenarioSettings(
        lambda_base=0,
        rho_target=0.8,
        burst_lambda=5,
        burst_interval=3,
        burst_start_offset=2,
    )
    counts = generate_arrival_counts(6, scenario, seed=5)
    assert counts[0] == 0
    assert counts[1] == 0


def test_heavy_tail_service_units_are_reproducible():
    settings = HeavyTailSettings(max_service_units=8)
    first = sample_service_units([0.1, 0.4, 0.9], "heavy_tail", settings, seed=3)
    second = sample_service_units([0.1, 0.4, 0.9], "heavy_tail", settings, seed=3)
    assert first == second
    assert all(unit >= 1 for unit in first)


def test_build_mu_by_queue_supports_uniform_one():
    train_df = pd.DataFrame({"true_queue_id": [0, 0, 1, 2, 2, 2]})
    mu = build_mu_by_queue(train_df, lambda_base=8, rho_target=0.9, capacity_mode="uniform_one")
    assert mu.tolist() == [1, 1, 1]


def test_build_mu_by_queue_supports_empirical_service_demand():
    train_df = pd.DataFrame(
        {
            "true_queue_id": [0, 0, 1, 1, 2, 2],
            "service_units": [2, 4, 3, 3, 1, 1],
        }
    )
    mu = build_mu_by_queue(
        train_df,
        lambda_base=6,
        rho_target=1.0,
        capacity_mode="empirical_service_demand",
    )
    assert mu.tolist() == [6, 6, 2]


def test_build_mu_by_queue_supports_empirical_service_median():
    train_df = pd.DataFrame(
        {
            "true_queue_id": [0, 0, 1, 1, 2, 2],
            "service_units": [2, 10, 3, 3, 1, 9],
        }
    )
    mu = build_mu_by_queue(
        train_df,
        lambda_base=6,
        rho_target=1.0,
        capacity_mode="empirical_service_median",
    )
    assert mu.tolist() == [12, 6, 10]


def test_misroute_bounce_adds_one_slot_penalty(tmp_path):
    train_df = pd.DataFrame(
        {
            "true_queue_id": [0, 0, 1, 1],
        }
    )
    split_df = pd.DataFrame(
        [
            {
                "ticket_id": 1,
                "true_queue_id": 0,
                "priority": "high",
                "language": "en",
                "predicted_queue_id": 1,
                "complexity_score": 0.3,
                "prob_q_0": 0.1,
                "prob_q_1": 0.9,
                "kappa_q_0": 1.0,
                "kappa_q_1": 1.1,
            }
        ]
    )
    scenario = ScenarioSettings(lambda_base=1, rho_target=1.0)
    environment = EnvironmentSpec(
        environment_id="deterministic_redundant",
        service_model="deterministic",
        delay_mode="redundant_baseline",
        capacity_mode="proportional_to_pj",
        slot_order="serve_then_arrive",
        serve_new_same_slot=False,
    )
    model_dir = tmp_path / "model"
    path = fixture_path(model_dir, "test", environment.environment_id, "single", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scenario": "single",
                "seed": 1,
                "shuffled_ticket_ids": [1],
                "arrival_counts": [1],
                "service_units_by_ticket": {"1": 1},
                "fixture_signature": {
                    "lambda_base": 1,
                    "rho_target": 1.0,
                    "burst_lambda": None,
                    "burst_interval": None,
                    "burst_start_offset": 0,
                    "arrival_mode": "poisson",
                    "service_model": "deterministic",
                    "delay_mode": "redundant_baseline",
                    "capacity_mode": "proportional_to_pj",
                },
            }
        ),
        encoding="utf-8",
    )
    result = simulate_single_run(
        split_df=split_df,
        train_df=train_df,
        policy=ClassifierOnlyPolicy(),
        environment=environment,
        scenario_name="single",
        scenario=scenario,
        seed=1,
        sla_deadlines={"high": 3, "medium": 6, "low": 10},
        split_name="test",
        sla_profile="primary",
        model_dir=model_dir,
        include_records=True,
    )
    assert len(result.decisions) == 1
    assert bool(result.decisions.loc[0, "misrouted"]) is True
    assert int(result.decisions.loc[0, "wait_slots"]) == 3


def test_compatible_route_skips_bounce_and_uses_alternative_service(tmp_path):
    train_df = pd.DataFrame({"true_queue_id": [0, 0, 1, 1], "service_units": [2, 2, 1, 1]})
    split_df = pd.DataFrame(
        [
            {
                "ticket_id": "ISSUE-1",
                "true_queue_id": 0,
                "priority": "high",
                "language": "unknown",
                "predicted_queue_id": 1,
                "complexity_score": 0.3,
                "service_units": 4,
                "prob_q_0": 0.4,
                "prob_q_1": 0.6,
                "delay_q_0": 4.0,
                "delay_q_1": 1.0,
            }
        ]
    )
    scenario = ScenarioSettings(lambda_base=1, rho_target=1.0)
    environment = EnvironmentSpec(
        environment_id="empirical_learnedjira",
        service_model="empirical",
        delay_mode="learned_jira_delay",
        capacity_mode="empirical_service_demand",
        slot_order="serve_then_arrive",
        serve_new_same_slot=False,
    )
    model_dir = tmp_path / "model"
    path = fixture_path(model_dir, "test", environment.environment_id, "single", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scenario": "single",
                "seed": 1,
                "shuffled_ticket_ids": ["ISSUE-1"],
                "arrival_counts": [1],
                "service_units_by_ticket": {"ISSUE-1": 4},
                "fixture_signature": {
                    "lambda_base": 1,
                    "rho_target": 1.0,
                    "burst_lambda": None,
                    "burst_interval": None,
                    "burst_start_offset": 0,
                    "arrival_mode": "poisson",
                    "service_model": "empirical",
                    "delay_mode": "learned_jira_delay",
                    "capacity_mode": "empirical_service_demand",
                },
            }
        ),
        encoding="utf-8",
    )
    skill_dir = model_dir / "skill_features"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "embedding_metadata.json").write_text(
        json.dumps({"compatible_queues": {"0": [0, 1], "1": [1, 0]}}),
        encoding="utf-8",
    )
    result = simulate_single_run(
        split_df=split_df,
        train_df=train_df,
        policy=ClassifierOnlyPolicy(),
        environment=environment,
        scenario_name="single",
        scenario=scenario,
        seed=1,
        sla_deadlines={"high": 3, "medium": 6, "low": 10},
        split_name="test",
        sla_profile="primary",
        model_dir=model_dir,
        include_records=True,
    )
    assert len(result.decisions) == 1
    assert bool(result.decisions.loc[0, "compatible_route"]) is True
    assert bool(result.decisions.loc[0, "misrouted"]) is False
    assert int(result.decisions.loc[0, "initial_remaining_service"]) == 1
    assert int(result.decisions.loc[0, "wait_slots"]) == 2


def test_topk_policies_never_choose_outside_candidate_set():
    queue_state = type(
        "QueueStateStub",
        (),
        {"backlog_by_queue": [10, 1, 0], "served_by_queue": [0, 0, 0], "time_slot": 0},
    )()
    ticket = {
        "ticket_id": 1,
        "prob_vector": [0.60, 0.30, 0.10],
        "complexity_score": 0.4,
    }
    mu_by_queue = pd.Series([1, 1, 1]).to_numpy()
    environment = EnvironmentSpec(
        environment_id="deterministic_redundant",
        service_model="deterministic",
        delay_mode="redundant_baseline",
        capacity_mode="uniform_one",
        slot_order="serve_then_arrive",
        serve_new_same_slot=False,
    )

    jsq_topk = build_policy("jsq_topk", topk_k=2)
    jsq_decision = jsq_topk.route(ticket, queue_state, mu_by_queue)
    assert jsq_decision.chosen_queue in {0, 1}

    qa_topk = build_policy(
        "qa_ftopsis_topk",
        environment=environment,
        weights=[0.30, 0.40, 0.30],
        topk_k=2,
    )
    qa_decision = qa_topk.route(ticket, queue_state, mu_by_queue)
    assert qa_decision.chosen_queue in {0, 1}


def test_qa_topk_confidence_gate_keeps_classifier_choice():
    queue_state = type(
        "QueueStateStub",
        (),
        {"backlog_by_queue": [5, 0, 0], "served_by_queue": [0, 0, 0], "time_slot": 0},
    )()
    ticket = {
        "ticket_id": 2,
        "prob_vector": [0.91, 0.05, 0.04],
        "complexity_score": 0.4,
    }
    mu_by_queue = pd.Series([1, 1, 1]).to_numpy()
    environment = EnvironmentSpec(
        environment_id="deterministic_redundant",
        service_model="deterministic",
        delay_mode="redundant_baseline",
        capacity_mode="uniform_one",
        slot_order="serve_then_arrive",
        serve_new_same_slot=False,
    )
    policy = build_policy(
        "qa_ftopsis_topk",
        environment=environment,
        weights=[0.30, 0.40, 0.30],
        confidence_gate=0.85,
        topk_k=2,
    )
    decision = policy.route(ticket, queue_state, mu_by_queue)
    assert decision.chosen_queue == 0


def test_qa_hierarchical_stays_within_macro_group():
    queue_state = type(
        "QueueStateStub",
        (),
        {"backlog_by_queue": [5, 0, 0, 0], "served_by_queue": [0, 0, 0, 0], "time_slot": 0},
    )()
    ticket = {
        "ticket_id": 3,
        "prob_vector": [0.55, 0.25, 0.15, 0.05],
        "complexity_score": 0.4,
        "macro_group_candidate_indices": [0, 1],
    }
    mu_by_queue = pd.Series([1, 1, 1, 1]).to_numpy()
    environment = EnvironmentSpec(
        environment_id="deterministic_redundant",
        service_model="empirical",
        delay_mode="redundant_baseline",
        capacity_mode="empirical_service_demand",
        slot_order="serve_then_arrive",
        serve_new_same_slot=False,
    )
    policy = build_policy(
        "qa_ftopsis_hierarchical",
        environment=environment,
        weights=[0.10, 0.15, 0.75],
        confidence_gate=0.7,
    )
    decision = policy.route(ticket, queue_state, mu_by_queue)
    assert decision.chosen_queue in {0, 1}


def test_qa_hybrid_stays_within_macro_group_and_topk():
    queue_state = type(
        "QueueStateStub",
        (),
        {"backlog_by_queue": [4, 3, 0, 0], "served_by_queue": [0, 0, 0, 0], "time_slot": 0},
    )()
    ticket = {
        "ticket_id": 4,
        "prob_vector": [0.40, 0.35, 0.20, 0.05],
        "complexity_score": 0.4,
        "macro_group_candidate_indices": [1, 2, 3],
    }
    mu_by_queue = pd.Series([1, 1, 1, 1]).to_numpy()
    environment = EnvironmentSpec(
        environment_id="empirical_learnedjira",
        service_model="empirical",
        delay_mode="learned_jira_delay",
        capacity_mode="empirical_service_demand",
        slot_order="serve_then_arrive",
        serve_new_same_slot=False,
    )
    ticket["delay_vector"] = [4.0, 1.0, 2.0, 3.0]
    policy = build_policy(
        "qa_ftopsis_hybrid",
        environment=environment,
        weights=[0.10, 0.15, 0.75],
        alpha=0.5,
        hybrid_mix=0.25,
        topk_k=2,
    )
    decision = policy.route(ticket, queue_state, mu_by_queue)
    assert decision.chosen_queue in {1, 2}


def test_qa_hybrid_is_anchored_to_best_maxweight_band():
    queue_state = type(
        "QueueStateStub",
        (),
        {"backlog_by_queue": [2, 0, 0, 0], "served_by_queue": [0, 0, 0, 0], "time_slot": 0},
    )()
    ticket = {
        "ticket_id": 5,
        "prob_vector": [0.10, 0.44, 0.43, 0.03],
        "complexity_score": 0.5,
        "macro_group_candidate_indices": [1, 2, 3],
        "delay_vector": [4.0, 2.0, 1.0, 5.0],
    }
    mu_by_queue = pd.Series([1, 1, 1, 1]).to_numpy()
    environment = EnvironmentSpec(
        environment_id="empirical_learnedjira",
        service_model="empirical",
        delay_mode="learned_jira_delay",
        capacity_mode="empirical_service_demand",
        slot_order="serve_then_arrive",
        serve_new_same_slot=False,
    )
    policy = build_policy(
        "qa_ftopsis_hybrid",
        environment=environment,
        weights=[0.10, 0.15, 0.75],
        alpha=2.0,
        hybrid_mix=0.15,
        topk_k=2,
    )
    decision = policy.route(ticket, queue_state, mu_by_queue)
    assert decision.chosen_queue in {1, 2}


def test_environment_fixtures_regenerate_when_signature_changes(tmp_path):
    dataset_path = build_synthetic_dataset(tmp_path / "tickets.csv", per_group=12)
    config_path = build_test_config(tmp_path / "config.yaml", dataset_path)
    config = load_config(config_path)
    config.simulation.scenarios["high_load"].burst_lambda = 5
    config.simulation.scenarios["high_load"].burst_interval = 2
    config.simulation.scenarios["high_load"].burst_start_offset = 0

    prepare_dataset(
        config.paths.input_csv,
        config.paths.prepared_dir,
        random_state=config.data.random_state,
        dataset_profile=config.data.dataset_profile,
        priority_mapping=config.data.priority_mapping,
        stratify_fields=config.data.stratify_fields,
        expected_languages=config.data.expected_languages,
    )
    train_classifier(config)
    split_df = load_feature_split(config.paths.model_dir, "val_sim")
    ensure_environment_fixtures(config, "val_sim", split_df)

    environment = build_environment_specs(config)[0]
    path = fixture_path(
        config.paths.model_dir,
        "val_sim",
        environment.environment_id,
        "high_load",
        config.simulation.seeds[0],
    )
    initial_payload = json.loads(path.read_text())
    assert initial_payload["fixture_signature"]["burst_start_offset"] == 0

    config.simulation.scenarios["high_load"].burst_start_offset = 1
    ensure_environment_fixtures(config, "val_sim", split_df)

    updated_payload = json.loads(path.read_text())
    assert updated_payload["fixture_signature"]["burst_start_offset"] == 1
