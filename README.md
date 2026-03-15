# QA-FTOPSIS

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/Focus-Queue--Aware%20Routing-0F766E?style=for-the-badge" alt="Queue-aware routing">
  <img src="https://img.shields.io/badge/Benchmark-Real%20Jira%20Issues-B45309?style=for-the-badge" alt="Real Jira issues">
  <img src="https://img.shields.io/badge/Method-Hierarchical%20Fuzzy%20TOPSIS-7C3AED?style=for-the-badge" alt="Hierarchical fuzzy TOPSIS">
</p>

<p align="center">
  <strong>Research-grade queue-aware routing toolkit for issue triage, simulation, and reproducible policy evaluation.</strong>
</p>

QA-FTOPSIS is a deterministic Python package and CLI for studying when queue-aware routing improves over a strong text classifier. It combines text classification, queue simulation, hierarchical queue families, fuzzy TOPSIS decision logic, and reporting for reproducible experiments on issue-routing datasets.

## What This Project Does

- Builds routing benchmarks from real issue-tracking data, including Jira exports and API pulls.
- Trains calibrated text classifiers for queue prediction.
- Adds queue-aware policies such as `JSQ`, `MaxWeight`, `QA-FTOPSIS`, `Top-K`, `hierarchical`, and hybrid variants.
- Simulates operational load under normal, high-load, bursty, and tail-sensitive scenarios.
- Exports reports, comparisons, confusion summaries, and paper-ready assets.

## Positioning

This repository is best used as:

- a benchmarking toolkit for routing policies,
- a reference implementation for hierarchical queue-aware fuzzy routing,
- an offline experimentation framework for triage and assignment systems.

It is not positioned as a production serving stack or real-time routing service.

## Core Workflow

```bash
python3 -m pip install -e .

# Generic CSV benchmark
qa-ftopsis prepare-data --config configs/german42.yaml
qa-ftopsis train-classifier --config configs/german42.yaml
qa-ftopsis build-skill-features --config configs/german42.yaml
qa-ftopsis run-suite --config configs/german42.yaml

# Jira benchmark workflow
qa-ftopsis fetch-jira-api --config configs/jira_kafka_real_combo.yaml
qa-ftopsis build-jira-benchmark --config configs/jira_kafka_real_combo.yaml
qa-ftopsis prepare-data --config configs/jira_kafka_real_combo.yaml
qa-ftopsis train-classifier --config configs/jira_kafka_real_combo.yaml
qa-ftopsis build-skill-features --config configs/jira_kafka_real_combo.yaml
qa-ftopsis train-delay-model --config configs/jira_kafka_real_combo.yaml
qa-ftopsis run-suite --config configs/jira_kafka_real_combo.yaml
```

## Main Capabilities

### Data and Benchmarking

- Deterministic dataset preparation with reproducible splits.
- Jira benchmark construction from raw issue tables and history logs.
- Queue family construction for hierarchical routing.
- Priority normalization and dataset-profile-specific validation.

### Models and Features

- Word and character TF-IDF classifier with calibrated probabilities.
- Confidence features such as entropy and `p_max`.
- Queue-similarity skill features from text embeddings.
- Learned Jira delay models for queue-specific service risk estimation.

### Routing Policies

- `classifier_only`
- `jsq`
- `jsq_topk`
- `maxweight_delay`
- `maxweight_prob`
- `qa_ftopsis`
- `qa_ftopsis_topk`
- `qa_ftopsis_hierarchical`
- `qa_ftopsis_hybrid`

### Simulation and Reporting

- Scenario-based queue simulation with deterministic seeds.
- Support for empirical and synthetic service abstractions.
- Reporting for `avg_cost`, `macro_f1`, `p95`, `p99`, SLA violations, backlog, and misroutes.
- Paper asset generation for figures, tables, and draft text.

## Repository Layout

```text
qa_ftopsis/      Core package
configs/         Experiment profiles
tests/           Unit and integration tests
sample_data/     Small local smoke-test inputs
paper/           Manuscript assets
```

## Example Configurations

- `configs/german42.yaml`: 42-queue benchmark profile.
- `configs/jira_public.yaml`: Public Jira benchmark profile.
- `configs/jira_kafka_real_combo.yaml`: real Jira combo benchmark with hierarchical routing.
- `configs/jira_kafka_real_combo_tailsim.yaml`: tail-sensitive follow-up configuration.

## Reproducibility

- Fixed seeds for splits and simulation runs.
- Deterministic CLI workflow.
- Artifact-based experiment structure for prepared data, models, runs, and reports.
- Tests cover fuzzy math, simulation logic, data integrity, and Jira pipeline behavior.

Run the test suite with:

```bash
pytest -q
```

## Data Notes

- Large raw datasets and generated artifacts are intentionally excluded from version control.
- The repository expects you to point configs to local datasets or Jira exports.
- Sample files are included only for smoke testing and development.

## Why Use QA-FTOPSIS

Use this project if you need to answer questions like:

- When does queue-aware routing beat classifier-only triage?
- Does hierarchical queue structure matter more than policy tuning?
- How much operational cost can be reduced without collapsing routing quality?
- Which routing policy is most robust under high-load or bursty conditions?

## Research Findings

This repository was developed as part of an empirical study on queue-aware routing for issue triage.

What we found:

- Plain `QA-FTOPSIS` did not consistently beat a strong `classifier_only` baseline.
- The positive result appeared when routing was made hierarchical on real Jira data.
- The strongest gains came when queues formed meaningful families of closely related tasks rather than completely unrelated destinations.
- In that setting, `qa_ftopsis_hierarchical` improved average operational cost and macro-F1 on the real Jira benchmark.
- The method did not consistently improve tail metrics such as `p95` and `p99`, so the result is positive but conditional.

Where it works best:

- when tasks are similar but not identical,
- when the classifier already identifies the right queue family,
- when the final decision is between nearby specialist queues,
- when queue pressure and service risk are useful tie-breakers between close alternatives.

Intuition:

The classifier finds the right neighborhood. The hierarchical queue-aware policy picks the best house inside that neighborhood.

## Citation

If you use this repository in academic work, cite the associated manuscript or link back to this project once the paper record is finalized.
