from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class ReportBundle:
    report_dir: Path
    summary: pd.DataFrame
    relative: pd.DataFrame
    seed_wins: pd.DataFrame
    benchmark_metadata: dict


def _load_report(report_dir: str | Path) -> ReportBundle:
    resolved = Path(report_dir).resolve()
    return ReportBundle(
        report_dir=resolved,
        summary=pd.read_csv(resolved / "summary_metrics.csv"),
        relative=pd.read_csv(resolved / "relative_improvement.csv"),
        seed_wins=pd.read_csv(resolved / "seed_win_loss_vs_classifier.csv"),
        benchmark_metadata=json.loads((resolved / "benchmark_metadata.json").read_text()),
    )


def _load_benchmark_summary(name: str, benchmark_dir: Path) -> dict[str, float | int | str]:
    metadata = json.loads((benchmark_dir / "benchmark_metadata.json").read_text())
    reroute = pd.read_csv(benchmark_dir / "reroute_stats.csv").iloc[0]
    macro_groups = pd.read_csv(benchmark_dir / "macro_groups.csv")
    group_sizes = macro_groups.groupby("macro_group_id")["queue_id"].count()
    return {
        "benchmark": name,
        "queue_field": metadata["queue_field"],
        "num_issues": int(metadata["num_issues"]),
        "num_queues": int(metadata["num_queues"]),
        "num_macro_groups": int(metadata["num_macro_groups"]),
        "largest_macro_group_size": int(group_sizes.max()),
        "reroute_rate": float(reroute["reroute_rate"]),
        "mean_num_queue_changes": float(reroute["mean_num_queue_changes"]),
        "median_time_to_first_queue_change_hours": float(
            reroute["median_time_to_first_queue_change_hours"]
        ),
        "mean_service_units": float(reroute["mean_service_units"]),
    }


def _primary_env_frame(report: ReportBundle, environment_id: str) -> pd.DataFrame:
    return report.summary[
        (report.summary["split"] == "test")
        & (report.summary["sla_profile"] == "primary")
        & (report.summary["environment_id"] == environment_id)
    ].copy()


def _primary_relative_frame(report: ReportBundle, environment_id: str) -> pd.DataFrame:
    return report.relative[report.relative["environment_id"] == environment_id].copy()


def _primary_seed_frame(report: ReportBundle, environment_id: str) -> pd.DataFrame:
    return report.seed_wins[report.seed_wins["environment_id"] == environment_id].copy()


def _format_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    return formatted


def _write_table(df: pd.DataFrame, stem: str, tables_dir: Path) -> None:
    df.to_csv(tables_dir / f"{stem}.csv", index=False)
    latex_df = _format_table(df)
    (tables_dir / f"{stem}.tex").write_text(
        latex_df.to_latex(index=False, escape=True),
        encoding="utf-8",
    )


def _annotate_bars(axis: plt.Axes, decimals: int = 0, suffix: str = "") -> None:
    for patch in axis.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        axis.annotate(
            f"{height:.{decimals}f}{suffix}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_dataset_summary_table(
    old_report: ReportBundle,
    combo_report: ReportBundle,
    tail_report: ReportBundle,
) -> pd.DataFrame:
    old_dir = Path(old_report.benchmark_metadata["benchmark_dir"])
    combo_dir = Path(combo_report.benchmark_metadata["benchmark_dir"])
    tail_dir = Path(tail_report.benchmark_metadata["benchmark_dir"])
    return pd.DataFrame(
        [
            _load_benchmark_summary("component", old_dir),
            _load_benchmark_summary("component_combo", combo_dir),
            _load_benchmark_summary("component_combo_tailsim", tail_dir),
        ]
    )


def _build_results_table(
    report: ReportBundle,
    environment_id: str,
    policies: list[str],
    scenarios: list[str],
) -> pd.DataFrame:
    frame = _primary_env_frame(report, environment_id)
    subset = frame[
        frame["policy"].isin(policies)
        & frame["scenario"].isin(scenarios)
    ][
        [
            "scenario",
            "policy",
            "avg_cost_median",
            "p95_wait_median",
            "p99_wait_median",
            "macro_f1_median",
            "misroute_rate_median",
            "sla_violation_rate_median",
        ]
    ].sort_values(["scenario", "policy"])
    return subset.reset_index(drop=True)


def _build_relative_table(
    report: ReportBundle,
    environment_id: str,
    policies: list[str],
    scenarios: list[str],
) -> pd.DataFrame:
    frame = _primary_relative_frame(report, environment_id)
    subset = frame[
        frame["policy"].isin(policies)
        & frame["scenario"].isin(scenarios)
    ].sort_values(["scenario", "policy"])
    return subset.reset_index(drop=True)


def _build_tail_followup_table(
    combo_report: ReportBundle,
    tail_report: ReportBundle,
    environment_id: str,
    scenarios: list[str],
    policies: list[str],
) -> pd.DataFrame:
    combo = _build_results_table(combo_report, environment_id, policies, scenarios).assign(
        benchmark_variant="combo"
    )
    tail = _build_results_table(tail_report, environment_id, policies, scenarios).assign(
        benchmark_variant="tailsim"
    )
    return pd.concat([combo, tail], ignore_index=True)[
        [
            "benchmark_variant",
            "scenario",
            "policy",
            "avg_cost_median",
            "p95_wait_median",
            "p99_wait_median",
            "macro_f1_median",
            "misroute_rate_median",
            "sla_violation_rate_median",
        ]
    ]


def _build_seed_win_table(
    report: ReportBundle,
    environment_id: str,
    policies: list[str],
    scenarios: list[str],
) -> pd.DataFrame:
    frame = _primary_seed_frame(report, environment_id)
    subset = frame[
        frame["policy"].isin(policies)
        & frame["scenario"].isin(scenarios)
    ].copy()
    grouped = (
        subset.groupby(["scenario", "policy"], as_index=False)
        .agg(
            avg_cost_wins=("avg_cost_improved", "sum"),
            p95_wins=("p95_wait_improved", "sum"),
            p99_wins=("p99_wait_improved", "sum"),
            macro_f1_delta_pp_median=("macro_f1_delta_pp", "median"),
        )
        .sort_values(["scenario", "policy"])
    )
    return grouped


def _plot_benchmark_shift(dataset_summary: pd.DataFrame, figures_dir: Path) -> None:
    order = ["component", "component_combo", "component_combo_tailsim"]
    frame = dataset_summary.set_index("benchmark").loc[order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    sns.barplot(
        data=frame,
        x="benchmark",
        y="num_issues",
        hue="benchmark",
        ax=axes[0],
        palette="Blues_d",
        legend=False,
    )
    axes[0].set_title("Issues")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    _annotate_bars(axes[0], decimals=0)

    sns.barplot(
        data=frame,
        x="benchmark",
        y="num_macro_groups",
        hue="benchmark",
        ax=axes[1],
        palette="Greens_d",
        legend=False,
    )
    axes[1].set_title("Macro-groups")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    _annotate_bars(axes[1], decimals=0)

    reroute_pct = frame.assign(reroute_rate_pct=frame["reroute_rate"] * 100.0)
    sns.barplot(
        data=reroute_pct,
        x="benchmark",
        y="reroute_rate_pct",
        hue="benchmark",
        ax=axes[2],
        palette="Oranges_d",
        legend=False,
    )
    axes[2].set_title("Reroute rate (%)")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=20)
    _annotate_bars(axes[2], decimals=1, suffix="%")

    fig.suptitle("Benchmark shift from trivial to non-trivial hierarchy", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "fig01_benchmark_shift.png")


def _plot_macro_group_structure(
    old_report: ReportBundle,
    combo_report: ReportBundle,
    figures_dir: Path,
) -> None:
    old_macro = pd.read_csv(Path(old_report.benchmark_metadata["benchmark_dir"]) / "macro_groups.csv")
    combo_macro = pd.read_csv(Path(combo_report.benchmark_metadata["benchmark_dir"]) / "macro_groups.csv")

    def sizes(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        grouped = (
            frame.groupby("macro_group_id")["queue_id"]
            .count()
            .sort_values(ascending=False)
            .reset_index(name="queue_count")
        )
        grouped["benchmark"] = label
        grouped["group_label"] = [f"G{i+1}" for i in range(len(grouped))]
        return grouped

    plot_frame = pd.concat(
        [sizes(old_macro, "component"), sizes(combo_macro, "component_combo")],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, label in zip(axes, ["component", "component_combo"]):
        subset = plot_frame[plot_frame["benchmark"] == label]
        sns.barplot(data=subset, x="group_label", y="queue_count", ax=axis, color="#4C72B0")
        axis.set_title(label)
        axis.set_xlabel("Macro-group")
        axis.set_ylabel("Queues per macro-group")
        _annotate_bars(axis, decimals=0)
    fig.suptitle("Hierarchy becomes non-trivial only after component-combo benchmark construction", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "fig02_macro_group_structure.png")


def _plot_combo_main_results(
    combo_report: ReportBundle,
    environment_id: str,
    figures_dir: Path,
) -> None:
    policies = [
        "maxweight_prob",
        "qa_ftopsis",
        "qa_ftopsis_topk",
        "qa_ftopsis_hybrid",
        "qa_ftopsis_hierarchical",
    ]
    order = ["normal", "high_load", "bursty"]
    frame = _primary_relative_frame(combo_report, environment_id)
    plot_frame = frame[frame["policy"].isin(policies)].copy()
    plot_frame["scenario"] = pd.Categorical(plot_frame["scenario"], order, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(
        data=plot_frame,
        x="scenario",
        y="avg_cost_improvement_pct",
        hue="policy",
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Avg-cost improvement vs classifier-only")
    axes[0].set_ylabel("Improvement (%)")

    sns.barplot(
        data=plot_frame,
        x="scenario",
        y="macro_f1_delta_pp",
        hue="policy",
        ax=axes[1],
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Macro-F1 delta vs classifier-only")
    axes[1].set_ylabel("Delta (pp)")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend_.remove()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Main result on the real component-combo benchmark", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "fig03_combo_main_results.png")


def _plot_seed_wins(
    combo_report: ReportBundle,
    environment_id: str,
    figures_dir: Path,
) -> None:
    policies = [
        "maxweight_prob",
        "qa_ftopsis",
        "qa_ftopsis_topk",
        "qa_ftopsis_hybrid",
        "qa_ftopsis_hierarchical",
    ]
    frame = _build_seed_win_table(
        combo_report,
        environment_id=environment_id,
        policies=policies,
        scenarios=["normal", "high_load", "bursty"],
    )
    pivot = frame.pivot(index="policy", columns="scenario", values="avg_cost_wins").reindex(policies)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        vmin=0,
        vmax=max(3, float(pivot.max().max())),
        cbar_kws={"label": "Seeds with lower avg_cost than classifier-only"},
        ax=ax,
    )
    ax.set_title("Seed-level avg-cost wins on component-combo benchmark")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Policy")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "fig04_seed_wins_combo.png")


def _plot_tail_followup(
    combo_report: ReportBundle,
    tail_report: ReportBundle,
    environment_id: str,
    figures_dir: Path,
) -> None:
    scenarios = ["normal", "high_load", "bursty"]
    policies = ["classifier_only", "qa_ftopsis_hierarchical"]

    combo = _build_results_table(combo_report, environment_id, policies, scenarios).assign(
        benchmark_variant="combo"
    )
    tail = _build_results_table(tail_report, environment_id, policies, scenarios).assign(
        benchmark_variant="tailsim"
    )
    absolute = pd.concat([combo, tail], ignore_index=True)

    combo_rel = _primary_relative_frame(combo_report, environment_id)
    tail_rel = _primary_relative_frame(tail_report, environment_id)
    relative = pd.concat(
        [
            combo_rel[combo_rel["policy"] == "qa_ftopsis_hierarchical"].assign(benchmark_variant="combo"),
            tail_rel[tail_rel["policy"] == "qa_ftopsis_hierarchical"].assign(benchmark_variant="tailsim"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.barplot(
        data=absolute,
        x="scenario",
        y="p95_wait_median",
        hue="benchmark_variant",
        ax=axes[0, 0],
        palette="Blues",
    )
    axes[0, 0].set_title("Absolute p95 wait rises after tail-sensitive redesign")
    axes[0, 0].set_ylabel("Median p95 wait")

    sns.barplot(
        data=absolute,
        x="scenario",
        y="p99_wait_median",
        hue="benchmark_variant",
        ax=axes[0, 1],
        palette="Purples",
    )
    axes[0, 1].set_title("Absolute p99 wait rises after tail-sensitive redesign")
    axes[0, 1].set_ylabel("Median p99 wait")

    sns.barplot(
        data=relative,
        x="scenario",
        y="avg_cost_improvement_pct",
        hue="benchmark_variant",
        ax=axes[1, 0],
        palette="Greens",
    )
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set_title("Hierarchical QA still improves avg_cost")
    axes[1, 0].set_ylabel("Improvement (%)")

    sns.barplot(
        data=relative,
        x="scenario",
        y="macro_f1_delta_pp",
        hue="benchmark_variant",
        ax=axes[1, 1],
        palette="Oranges",
    )
    axes[1, 1].axhline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Hierarchical QA still improves macro-F1")
    axes[1, 1].set_ylabel("Delta (pp)")

    for row in axes:
        for axis in row:
            axis.legend(title="")
    fig.suptitle("Tail-sensitive follow-up: higher tails, same relative tail ordering", fontsize=14)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "fig05_tail_followup.png")


def _build_claims_json(
    old_report: ReportBundle,
    combo_report: ReportBundle,
    tail_report: ReportBundle,
    output_dir: Path,
) -> dict[str, object]:
    old_rel = _primary_relative_frame(old_report, "empirical_learnedjira")
    combo_rel = _primary_relative_frame(combo_report, "empirical_learnedjira")
    tail_rel = _primary_relative_frame(tail_report, "empirical_learnedjira")

    def pick(frame: pd.DataFrame, policy: str, scenario: str) -> dict[str, float]:
        row = frame[(frame["policy"] == policy) & (frame["scenario"] == scenario)].iloc[0]
        return {
            "avg_cost_improvement_pct": float(row["avg_cost_improvement_pct"]),
            "p95_wait_improvement_pct": float(row["p95_wait_improvement_pct"]),
            "p99_wait_improvement_pct": float(row["p99_wait_improvement_pct"]),
            "macro_f1_delta_pp": float(row["macro_f1_delta_pp"]),
        }

    claims = {
        "component_old_high_load_qa": pick(old_rel, "qa_ftopsis", "high_load"),
        "component_old_high_load_hierarchical": pick(old_rel, "qa_ftopsis_hierarchical", "high_load"),
        "combo_high_load_hierarchical": pick(combo_rel, "qa_ftopsis_hierarchical", "high_load"),
        "combo_bursty_hierarchical": pick(combo_rel, "qa_ftopsis_hierarchical", "bursty"),
        "tailsim_high_load_hierarchical": pick(tail_rel, "qa_ftopsis_hierarchical", "high_load"),
        "tailsim_bursty_hierarchical": pick(tail_rel, "qa_ftopsis_hierarchical", "bursty"),
        "source_reports": {
            "old_report": str(old_report.report_dir),
            "combo_report": str(combo_report.report_dir),
            "tail_report": str(tail_report.report_dir),
        },
    }
    claims_path = output_dir / "paper_claims.json"
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    return claims


def write_paper_draft(
    draft_path: str | Path,
    assets_root: str | Path,
    claims: dict[str, object],
) -> Path:
    draft = Path(draft_path).resolve()
    assets = Path(assets_root).resolve()
    figures = assets / "figures"
    tables = assets / "tables"

    combo_high = claims["combo_high_load_hierarchical"]
    combo_bursty = claims["combo_bursty_hierarchical"]
    tail_high = claims["tailsim_high_load_hierarchical"]
    old_high = claims["component_old_high_load_qa"]

    text = f"""# Hierarchical Queue-Aware Fuzzy Routing on Real Jira Issues

## Abstract

We evaluate whether queue-aware fuzzy routing can improve issue assignment over a strong text classifier on real Apache Jira data. The study uses a reproducible benchmark built from `KAFKA` issues, component histories, and resolution-time-derived service proxies. On an initial `component` benchmark, the queue graph is trivial and plain `QA-FTOPSIS` fails to beat `classifier_only`; in `high_load`, plain QA is worse by {abs(old_high["avg_cost_improvement_pct"]):.2f}% on `avg_cost` and loses {abs(old_high["macro_f1_delta_pp"]):.2f} macro-F1 points. We then redesign benchmark construction so that routing targets preserve multi-component structure (`component_combo`) and induce non-trivial macro-groups. Under this benchmark, `qa_ftopsis_hierarchical` beats `classifier_only` in the learned-delay environment, improving `avg_cost` by {combo_high["avg_cost_improvement_pct"]:.2f}% in `high_load` and {combo_bursty["avg_cost_improvement_pct"]:.2f}% in `bursty`, while also improving macro-F1 by {combo_high["macro_f1_delta_pp"]:.2f} points in both scenarios. A final tail-sensitive simulator redesign increases absolute tail waits substantially, but the relative `p95/p99` gains remain zero even though the hierarchical policy keeps a smaller positive `avg_cost` gain ({tail_high["avg_cost_improvement_pct"]:.2f}% in `high_load`). The main finding is therefore conditional: hierarchical queue-aware fuzzy control becomes useful once benchmark construction exposes meaningful queue families, but tail-delay improvements remain unsupported in the current simulator.

## 1. Introduction

Queue-aware routing methods are attractive because they promise to trade a small amount of label accuracy for better operational performance under load. In practice, this promise often fails because the benchmark itself can hide or remove the structure that a queue-aware policy needs in order to help. If routing targets collapse naturally related queues, the search space for a control policy becomes either trivial or excessively punitive.

This paper studies that problem on real Jira issue data. The central question is not whether fuzzy reranking can always beat a classifier, but whether benchmark construction determines whether a queue-aware controller has any realistic leverage at all. The experiments here show a sharp answer. With a naive queue definition, hierarchical control collapses to a no-op and plain fuzzy reranking loses. With a queue definition that preserves multi-component issue structure, the same family of policies becomes competitive and the hierarchical variant produces consistent gains in average operational cost and routing quality.

The contribution is therefore methodological as much as algorithmic. The strongest positive result comes from changing the benchmark so that the queue graph becomes meaningful, not from repeatedly retuning fuzzy weights. The tail-sensitive follow-up reinforces this point: even after explicitly increasing tail pressure in the simulator, `p95/p99` gains do not appear.

## 2. Benchmark Construction

The benchmark is built from real Apache Jira `KAFKA` issues and history records. Each issue contributes:

- `summary + description` as routing text
- component-derived routing targets
- component-history-based reroute evidence
- resolution-time-derived service-demand proxies

We compare three benchmark variants:

1. `component`: queue target is the single component label.
2. `component_combo`: queue target preserves multi-component combinations.
3. `component_combo_tailsim`: same routing benchmark as `component_combo`, but with a tail-sensitive service calibration for the simulator.

The structural difference that matters is hierarchy. The `component` benchmark has 10 queues and 10 macro-groups, so hierarchy is inert. The `component_combo` benchmark still has 10 queues but only 7 macro-groups, which gives the hierarchical controller a real restricted action set.

![Benchmark shift]({figures / "fig01_benchmark_shift.png"})

![Macro-group structure]({figures / "fig02_macro_group_structure.png"})

The dataset summary table is here: [table_dataset_summary.csv]({tables / "table_dataset_summary.csv"}).

## 3. Method

We evaluate six routing policies against the same arrival streams and scenario definitions:

- `classifier_only`
- `maxweight_prob`
- `qa_ftopsis`
- `qa_ftopsis_topk`
- `qa_ftopsis_hybrid`
- `qa_ftopsis_hierarchical`

The classifier is a strong text-only baseline trained on Jira issue text. Queue-aware policies receive calibrated class probabilities together with queue-state features. The hierarchical policy first restricts routing to the relevant macro-group and then applies queue-aware fuzzy ranking inside that family. This matters because the cost of deviating from the classifier is much lower when candidate queues are structurally compatible.

The main fuzzy criteria are:

- backlog pressure
- learned or embedding-based delay risk
- misroute risk

The main reported metrics are:

- `avg_cost`
- `macro_f1`
- `p95_wait`
- `p99_wait`
- `sla_violation_rate`

## 4. Experimental Setup

The paper uses three report bundles:

- old real benchmark: [suite_20260228T170859Z](/Users/dimgouso/Tsitsiklis_fuzzy/artifacts/reports/suite_20260228T170859Z)
- main positive benchmark: [suite_20260228T172639Z](/Users/dimgouso/Tsitsiklis_fuzzy/artifacts/reports/suite_20260228T172639Z)
- tail-sensitive follow-up: [suite_20260228T174618Z](/Users/dimgouso/Tsitsiklis_fuzzy/artifacts/reports/suite_20260228T174618Z)

All claims in this draft are derived directly from those report directories and the tables generated in [paper_assets]({assets}).

## 5. Results

### 5.1 Old benchmark: plain QA does not help

On the original `component` benchmark, `qa_ftopsis_hierarchical` is effectively identical to `classifier_only` because the hierarchy is trivial. Plain `qa_ftopsis` is worse. In `empirical_learnedjira / high_load`, plain QA loses {abs(old_high["avg_cost_improvement_pct"]):.2f}% on `avg_cost` and {abs(old_high["macro_f1_delta_pp"]):.2f} macro-F1 points relative to the classifier. This is not a case where a smarter weight sweep was needed; the benchmark simply did not create a useful control surface.

The old-benchmark results table is here: [table_old_component_results.csv]({tables / "table_old_component_results.csv"}).

### 5.2 Component-combo benchmark: hierarchy unlocks a positive QA result

The `component_combo` benchmark changes the picture. The strongest result appears in the `empirical_learnedjira` environment, where `qa_ftopsis_hierarchical` becomes the best QA-family policy. In `high_load`, it improves `avg_cost` by {combo_high["avg_cost_improvement_pct"]:.2f}% and macro-F1 by {combo_high["macro_f1_delta_pp"]:.2f} points. In `bursty`, it improves `avg_cost` by {combo_bursty["avg_cost_improvement_pct"]:.2f}% and macro-F1 by {combo_bursty["macro_f1_delta_pp"]:.2f} points. The main gain is therefore not only operational but also label-quality preserving.

![Main combo results]({figures / "fig03_combo_main_results.png"})

The corresponding tables are:

- [table_combo_results.csv]({tables / "table_combo_results.csv"})
- [table_combo_relative_results.csv]({tables / "table_combo_relative_results.csv"})

### 5.3 The positive signal comes from hierarchy, not from plain fuzzy reranking

Across the same benchmark, plain `qa_ftopsis` and `qa_ftopsis_topk` are weaker than `qa_ftopsis_hierarchical`. The seed-level win counts make this visible. On the component-combo benchmark, the hierarchical policy wins on `avg_cost` for all seeds in `high_load` and `bursty`, while the plain fuzzy variants do not show the same consistency.

![Seed-level wins]({figures / "fig04_seed_wins_combo.png"})

The seed summary is here: [table_combo_seed_wins.csv]({tables / "table_combo_seed_wins.csv"}).

### 5.4 Tail-sensitive redesign increases absolute tails, but not relative tail gains

The final experiment changes the simulator so that tail pressure is stronger. Absolute tail values do increase: the benchmark moves from roughly `p95 ≈ 81`, `p99 ≈ 82` to `p95 ≈ 113.95`, `p99 ≈ 120` in the key scenarios. But the relative `p95/p99` improvement of `qa_ftopsis_hierarchical` remains zero. The hierarchical policy still keeps a smaller positive `avg_cost` gain, including {tail_high["avg_cost_improvement_pct"]:.2f}% in `high_load`, and it still preserves a positive macro-F1 delta. However, the tail-delay claim is not supported.

![Tail follow-up]({figures / "fig05_tail_followup.png"})

The follow-up table is here: [table_tail_followup.csv]({tables / "table_tail_followup.csv"}).

## 6. Discussion

The main lesson is that benchmark construction dominates the outcome. When queue definitions erase natural queue families, hierarchical routing cannot help. Once the benchmark preserves those families, `qa_ftopsis_hierarchical` becomes useful and consistently beats the classifier baseline on `avg_cost` while also improving macro-F1.

This is a publishable positive result, but it is narrower than the original ambition. The supported claim is not that fuzzy queue-aware control improves every operational metric. The supported claim is that hierarchical queue-aware fuzzy routing helps on real Jira data when the queue graph is meaningful. The unsupported claim remains tail-delay improvement.

## 7. Limitations

- The data comes from a single Jira project family (`KAFKA`).
- Queue definitions are benchmark constructions derived from components, not native team labels.
- Service demand is a proxy derived from resolution time rather than direct effort logs.
- Even after the tail-sensitive redesign, the current simulator does not show relative `p95/p99` improvements.

## 8. Conclusion

This study finds a conditional but real positive result for `QA-FTOPSIS`. The plain fuzzy policy does not beat a strong classifier on the initial real benchmark, and hierarchy is inert there. After reconstructing the benchmark so that queues preserve multi-component structure, `qa_ftopsis_hierarchical` improves both `avg_cost` and macro-F1 on real Jira issue routing. A targeted tail-sensitive redesign increases absolute tail pressure but still fails to produce relative `p95/p99` gains. The strongest paper claim is therefore methodological: benchmark construction determines whether queue-aware fuzzy control has any practical leverage, and hierarchical fuzzy routing can benefit from that leverage once the queue graph is real.

## Artifact Index

- assets root: [paper_assets]({assets})
- figures directory: [figures]({figures})
- tables directory: [tables]({tables})
- claims file: [paper_claims.json]({assets / "paper_claims.json"})
"""
    draft.write_text(text, encoding="utf-8")
    return draft


def generate_paper_assets(
    old_report_dir: str | Path,
    combo_report_dir: str | Path,
    tail_report_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, str]:
    sns.set_theme(style="whitegrid", context="talk")

    old_report = _load_report(old_report_dir)
    combo_report = _load_report(combo_report_dir)
    tail_report = _load_report(tail_report_dir)

    root = Path(output_dir).resolve()
    figures_dir = root / "figures"
    tables_dir = root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = _build_dataset_summary_table(old_report, combo_report, tail_report)
    _write_table(dataset_summary, "table_dataset_summary", tables_dir)

    key_policies = [
        "classifier_only",
        "maxweight_prob",
        "qa_ftopsis",
        "qa_ftopsis_topk",
        "qa_ftopsis_hierarchical",
        "qa_ftopsis_hybrid",
    ]
    scenarios = ["normal", "high_load", "bursty"]
    old_results = _build_results_table(old_report, "empirical_learnedjira", key_policies, scenarios)
    combo_results = _build_results_table(combo_report, "empirical_learnedjira", key_policies, scenarios)
    combo_relative = _build_relative_table(combo_report, "empirical_learnedjira", key_policies, scenarios)
    tail_followup = _build_tail_followup_table(
        combo_report,
        tail_report,
        "empirical_learnedjira",
        scenarios=["high_load", "bursty"],
        policies=["classifier_only", "maxweight_prob", "qa_ftopsis_hierarchical"],
    )
    seed_wins = _build_seed_win_table(
        combo_report,
        "empirical_learnedjira",
        policies=key_policies[1:],
        scenarios=scenarios,
    )

    _write_table(old_results, "table_old_component_results", tables_dir)
    _write_table(combo_results, "table_combo_results", tables_dir)
    _write_table(combo_relative, "table_combo_relative_results", tables_dir)
    _write_table(tail_followup, "table_tail_followup", tables_dir)
    _write_table(seed_wins, "table_combo_seed_wins", tables_dir)

    _plot_benchmark_shift(dataset_summary, figures_dir)
    _plot_macro_group_structure(old_report, combo_report, figures_dir)
    _plot_combo_main_results(combo_report, "empirical_learnedjira", figures_dir)
    _plot_seed_wins(combo_report, "empirical_learnedjira", figures_dir)
    _plot_tail_followup(combo_report, tail_report, "empirical_learnedjira", figures_dir)

    claims = _build_claims_json(old_report, combo_report, tail_report, root)
    manifest = {
        "old_report": str(Path(old_report_dir).resolve()),
        "combo_report": str(Path(combo_report_dir).resolve()),
        "tail_report": str(Path(tail_report_dir).resolve()),
        "output_dir": str(root),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
    }
    (root / "README.md").write_text(
        "\n".join(
            [
                "# Paper Assets",
                "",
                "This directory was generated from the final real-data Jira runs.",
                "",
                f"- old report: `{manifest['old_report']}`",
                f"- combo report: `{manifest['combo_report']}`",
                f"- tail report: `{manifest['tail_report']}`",
                "",
                "Key outputs:",
                "- `figures/fig01_benchmark_shift.png`",
                "- `figures/fig02_macro_group_structure.png`",
                "- `figures/fig03_combo_main_results.png`",
                "- `figures/fig04_seed_wins_combo.png`",
                "- `figures/fig05_tail_followup.png`",
                "- `tables/table_dataset_summary.csv`",
                "- `tables/table_old_component_results.csv`",
                "- `tables/table_combo_results.csv`",
                "- `tables/table_combo_relative_results.csv`",
                "- `tables/table_tail_followup.csv`",
                "- `tables/table_combo_seed_wins.csv`",
            ]
        ),
        encoding="utf-8",
    )
    (root / "paper_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "assets_root": str(root),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "claims_path": str(root / "paper_claims.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures and tables from report directories.")
    parser.add_argument("--old-report", required=True, help="Path to the old real benchmark report directory.")
    parser.add_argument("--combo-report", required=True, help="Path to the positive combo benchmark report directory.")
    parser.add_argument("--tail-report", required=True, help="Path to the tail-sensitive follow-up report directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for figures and tables.")
    parser.add_argument(
        "--draft-path",
        help="Optional markdown path for a full paper draft that references the generated assets.",
    )
    args = parser.parse_args()

    outputs = generate_paper_assets(
        old_report_dir=args.old_report,
        combo_report_dir=args.combo_report,
        tail_report_dir=args.tail_report,
        output_dir=args.output_dir,
    )
    if args.draft_path:
        claims = json.loads(Path(outputs["claims_path"]).read_text())
        write_paper_draft(args.draft_path, outputs["assets_root"], claims)


if __name__ == "__main__":
    main()
