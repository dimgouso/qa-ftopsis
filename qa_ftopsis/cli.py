from __future__ import annotations

import json
from pathlib import Path

import typer

from qa_ftopsis.config import load_config
from qa_ftopsis.data import prepare_dataset
from qa_ftopsis.experiment import (
    build_jira_benchmark_command,
    build_skill_features_command,
    fetch_jira_api_command,
    run_sim,
    run_suite,
    train_delay_model_command,
)
from qa_ftopsis.models import train_classifier
from qa_ftopsis.paper_assets import generate_paper_assets, write_paper_draft
from qa_ftopsis.reporting import generate_report

app = typer.Typer(no_args_is_help=True)


@app.command("prepare-data")
def prepare_data_command(
    config: str | None = typer.Option(None, help="Optional path to experiment YAML config."),
    input: str | None = typer.Option(None, help="Path to the raw CSV dataset."),
    output: str | None = typer.Option(None, help="Directory for prepared parquet splits."),
    random_state: int | None = typer.Option(None, help="Deterministic split seed."),
) -> None:
    if config is not None:
        config_obj = load_config(config)
        if config_obj.data.dataset_profile == "jira_public" and not config_obj.paths.input_csv.exists():
            build_jira_benchmark_command(config_obj)
        metadata = prepare_dataset(
            input_csv=config_obj.paths.input_csv,
            output_dir=config_obj.paths.prepared_dir,
            random_state=config_obj.data.random_state,
            dataset_profile=config_obj.data.dataset_profile,
            priority_mapping=config_obj.data.priority_mapping,
            stratify_fields=config_obj.data.stratify_fields,
            expected_languages=config_obj.data.expected_languages,
        )
    else:
        if input is None or output is None:
            raise typer.BadParameter("Either --config or both --input and --output are required.")
        metadata = prepare_dataset(
            input_csv=input,
            output_dir=output,
            random_state=random_state or 7,
        )
    typer.echo(f"Prepared dataset in {metadata['prepared_dir']}")


@app.command("train-classifier")
def train_classifier_command(
    config: str = typer.Option(..., help="Path to experiment YAML config."),
) -> None:
    config_obj = load_config(config)
    result = train_classifier(config_obj)
    typer.echo(f"Trained classifier in {result['model_dir']}")


@app.command("build-jira-benchmark")
def build_jira_benchmark_cli(
    config: str = typer.Option(..., help="Path to experiment YAML config."),
) -> None:
    result = build_jira_benchmark_command(config)
    typer.echo(f"Built Jira benchmark in {result['benchmark_dir']}")


@app.command("fetch-jira-api")
def fetch_jira_api_cli(
    config: str = typer.Option(..., help="Path to experiment YAML config."),
) -> None:
    result = fetch_jira_api_command(config)
    typer.echo(f"Fetched {result['num_issues']} Jira issues to {result['issues_path']}")


@app.command("run-sim")
def run_sim_command(
    config: str = typer.Option(..., help="Path to experiment YAML config."),
    policy: str = typer.Option(..., help="Policy name."),
    split: str = typer.Option("test", help="Split name: val_sim or test."),
) -> None:
    run_dir = run_sim(config, policy_name=policy, split=split)
    typer.echo(f"Simulation run saved to {run_dir}")


@app.command("run-suite")
def run_suite_command(
    config: str = typer.Option(..., help="Path to experiment YAML config."),
) -> None:
    run_dir = run_suite(config)
    typer.echo(f"Suite saved to {run_dir}")


@app.command("build-skill-features")
def build_skill_features_cli(
    config: str = typer.Option(..., help="Path to experiment YAML config."),
) -> None:
    result = build_skill_features_command(config)
    typer.echo(f"Skill features saved to {result['skill_dir']}")


@app.command("train-delay-model")
def train_delay_model_cli(
    config: str = typer.Option(..., help="Path to experiment YAML config."),
) -> None:
    result = train_delay_model_command(config)
    typer.echo(f"Delay model saved to {result['delay_model_path']}")


@app.command("report")
def report_command(
    run_dir: str = typer.Option(..., help="Path to run directory produced by run-sim or run-suite."),
) -> None:
    run_path = Path(run_dir).resolve()
    config_snapshot = run_path / "config_snapshot.yaml"
    if config_snapshot.exists():
        config = load_config(config_snapshot)
        report_dir = generate_report(
            run_dir=run_path,
            report_root=config.paths.report_root,
            sample_size=config.reporting.explainability_sample_size,
        )
    else:
        report_dir = generate_report(
            run_dir=run_path,
            report_root=run_path.parent.parent / "reports",
            sample_size=100,
        )
    typer.echo(f"Report written to {report_dir}")


@app.command("paper-assets")
def paper_assets_command(
    old_report: str = typer.Option(..., help="Path to the old benchmark report directory."),
    combo_report: str = typer.Option(..., help="Path to the positive combo benchmark report directory."),
    tail_report: str = typer.Option(..., help="Path to the tail-sensitive follow-up report directory."),
    output_dir: str = typer.Option(..., help="Directory for generated paper figures and tables."),
    draft_path: str | None = typer.Option(None, help="Optional path for a generated markdown paper draft."),
) -> None:
    outputs = generate_paper_assets(
        old_report_dir=old_report,
        combo_report_dir=combo_report,
        tail_report_dir=tail_report,
        output_dir=output_dir,
    )
    if draft_path is not None:
        claims_path = Path(outputs["claims_path"])
        claims = json.loads(claims_path.read_text())
        write_paper_draft(draft_path=draft_path, assets_root=output_dir, claims=claims)
    typer.echo(f"Paper assets written to {outputs['assets_root']}")


if __name__ == "__main__":
    app()
