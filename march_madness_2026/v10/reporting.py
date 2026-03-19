from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict

from ..models import PortfolioResult
from ..reporting import _candidate_to_dict
from ..tournament import TournamentModel


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _safe_team_name(model: TournamentModel, team_id: str) -> str:
    try:
        return model.team_name(team_id)
    except Exception:
        return team_id


def write_outputs(output_dir: Path, run_bundle: Dict[str, Any]) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: PortfolioResult = run_bundle["result"]
    model: TournamentModel = run_bundle["model"]
    stamp = date.today().isoformat()
    base_name = f"{stamp}_march-madness-bracket-portfolio"
    summary_payload = _json_ready(result.summary)

    finalists_payload = [_candidate_to_dict(candidate, model) for candidate in result.finalists]
    portfolio_payload = {
        "run_id": result.run_id,
        "data_source": result.data_source,
        "candidate_count": result.candidate_count,
        "summary": summary_payload,
        "scenario_summary": _json_ready(result.scenario_summary),
        "sensitivity_summary": _json_ready(result.sensitivity_summary),
        "finalists": finalists_payload,
        "engine_version": run_bundle.get("engine_version", "v10"),
        "simulation_config": _json_ready(run_bundle.get("simulation_config", {})),
        "selection_metadata": _json_ready(run_bundle.get("selection_metadata", {})),
        "release_guardrail_failures": _json_ready(run_bundle.get("release_guardrail_failures", [])),
    }
    # Preserve older top-level consumers that expect summary metrics outside the nested summary object.
    portfolio_payload.update(summary_payload)
    if "v10_artifacts" in run_bundle:
        portfolio_payload["v10_artifacts"] = _json_ready(run_bundle["v10_artifacts"])

    portfolio_json = output_dir / f"{base_name}_finalists.json"
    with portfolio_json.open("w", encoding="utf-8") as handle:
        json.dump(portfolio_payload, handle, indent=2)

    finalist_paths: Dict[str, Path] = {}
    for index, finalist in enumerate(finalists_payload, start=1):
        bracket_path = output_dir / f"{base_name}_bracket-{index:02d}.json"
        with bracket_path.open("w", encoding="utf-8") as handle:
            json.dump(finalist, handle, indent=2)
        finalist_paths[f"bracket_{index:02d}"] = bracket_path

    simulation_config = run_bundle.get("simulation_config", {})
    selection_metadata = run_bundle.get("selection_metadata", {})
    v10_artifacts = run_bundle.get("v10_artifacts", {})
    release_evaluation = v10_artifacts.get("portfolio_release_evaluation")
    naive_release_evaluation = v10_artifacts.get("naive_release_evaluation")
    team_posteriors = (v10_artifacts.get("team_posteriors") or {}).get("team_posteriors", [])
    duplication = v10_artifacts.get("public_duplication", {})
    duplication_rows = sorted(
        (
            {
                "team_id": team_id,
                "team_name": _safe_team_name(model, team_id),
                **metrics,
            }
            for team_id, metrics in duplication.items()
        ),
        key=lambda item: item.get("expected_title_duplicates", 0.0),
        reverse=True,
    )

    report_path = output_dir / f"{base_name}_report.md"
    report_lines = [
        "# March Madness Bracket Portfolio Report",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Data source: `{result.data_source}`",
        f"- Candidate universe: `{result.candidate_count}` brackets",
        f"- Finalists: `{len(result.finalists)}` brackets",
        f"- Engine version: `{run_bundle.get('engine_version', 'v10')}`",
        "",
        "## Portfolio Summary",
        "",
        f"- Weighted portfolio first-place equity: `{result.summary.weighted_portfolio_first_place_equity:.4f}`",
        f"- Naive baseline first-place equity: `{result.summary.naive_baseline_first_place_equity:.4f}`",
        f"- Weighted portfolio capture rate: `{result.summary.weighted_portfolio_capture_rate:.4f}`",
        f"- Naive top-5 baseline capture rate: `{result.summary.naive_baseline_capture_rate:.4f}`",
        f"- Weighted expected payout: `{result.summary.weighted_portfolio_expected_payout:.2f}`",
        f"- Naive expected payout: `{result.summary.naive_baseline_expected_payout:.2f}`",
        f"- Release objective score: `{result.summary.weighted_portfolio_objective:.4f}`",
        f"- Naive release objective score: `{result.summary.naive_baseline_objective:.4f}`",
        f"- Weighted cash rate: `{result.summary.weighted_portfolio_cash_rate:.4f}`",
        f"- Weighted top-3 equity: `{result.summary.weighted_portfolio_top3_equity:.4f}`",
        f"- Payoff correlation score: `{result.summary.payoff_correlation_score:.4f}`",
        f"- Average pairwise overlap: `{result.summary.average_pairwise_overlap:.4f}`",
        f"- Distinct archetypes: `{result.summary.distinct_archetypes}`",
        f"- Unique champions: `{result.summary.unique_champions}`",
        "",
        "## Finalist Brackets",
        "",
    ]

    for candidate in result.finalists:
        champion_name = model.team_name(candidate.champion_team_id)
        report_lines.extend(
            [
                f"### {candidate.bracket_id}",
                "",
                f"- Archetype: `{candidate.archetype}`",
                f"- Risk label: `{candidate.risk_label}`",
                f"- Champion: `{champion_name}`",
                f"- Final Four: `{', '.join(model.team_name(team_id) for team_id in candidate.final_four_team_ids)}`",
                f"- Weighted first-place equity: `{candidate.weighted_first_place_equity:.4f}`",
                f"- Weighted average finish: `{candidate.weighted_average_finish:.2f}`",
                f"- Scenario fit: `{candidate.scenario_fit}`",
                f"- Why it survived: {candidate.why_selected}",
                "",
            ]
        )

    report_lines.extend(
        [
            "## Run Config",
            "",
            f"- Seed: `{simulation_config.get('seed', 'N/A')}`",
            f"- Candidate generation seed: `{simulation_config.get('candidate_generation_seed', 'N/A')}`",
            f"- Tournament simulations: `{simulation_config.get('num_tournament_simulations', 'N/A')}`",
            f"- Candidate brackets: `{simulation_config.get('num_candidate_brackets', 'N/A')}`",
            f"- Portfolio size: `{simulation_config.get('portfolio_size', 'N/A')}`",
            f"- Release evaluation seeds: `{simulation_config.get('release_evaluation_seeds', 'N/A')}`",
            f"- Release evaluation seed count: `{simulation_config.get('release_evaluation_seed_count', 'N/A')}`",
            f"- Training profile: `{simulation_config.get('training_profile_id', 'N/A')}`",
            f"- Game model artifact: `{simulation_config.get('model_artifact_path', 'N/A')}`",
        f"- Public field artifact: `{simulation_config.get('public_field_artifact_path', 'N/A')}`",
        f"- Public field runtime mode: `{simulation_config.get('public_field_runtime_mode', 'N/A')}`",
        f"- Candidate pool size: `{selection_metadata.get('candidate_pool_size', 'N/A')}`",
            f"- Positive equity pool: `{selection_metadata.get('positive_equity_pool_size', 'N/A')}`",
            f"- Practical positive equity pool: `{selection_metadata.get('practical_positive_equity_pool_size', 'N/A')}`",
            f"- Zero equity fallback: `{selection_metadata.get('used_zero_equity_fallback', 'N/A')}`",
            f"- Selection objective: `{selection_metadata.get('selection_objective', 'N/A')}`",
            f"- Local search swaps: `{selection_metadata.get('local_search_swaps', 'N/A')}`",
            "",
            "## Release Contract",
            "",
            f"- Objective name: `{simulation_config.get('release_contract', {}).get('objective_name', 'N/A')}`",
            f"- Blended weight floor: `{simulation_config.get('release_contract', {}).get('blended_weight_floor', 'N/A')}`",
            f"- Release gates passed: `{selection_metadata.get('passes_release_gates', 'N/A')}`",
            f"- Guardrail failures: `{selection_metadata.get('portfolio_guardrail_failures', [])}`",
            f"- Minimum finalist FPE: `{selection_metadata.get('portfolio_min_individual_fpe', 'N/A')}`",
            f"- Release objective score: `{selection_metadata.get('portfolio_release_objective_score', 'N/A')}`",
            f"- Naive release objective score: `{selection_metadata.get('naive_release_objective_score', 'N/A')}`",
            "",
            "## Artifact Provenance",
            "",
            f"- Game model provenance: `{simulation_config.get('artifact_provenance', {}).get('game_model', {})}`",
            f"- Public field provenance: `{simulation_config.get('artifact_provenance', {}).get('public_field', {})}`",
            "",
            "## Scenario Summary",
            "",
        ]
    )
    for contest_id, metrics in result.scenario_summary.items():
        report_lines.append(
            f"- `{contest_id}`: portfolio_fpe=`{metrics.get('portfolio_first_place_equity', 0.0):.4f}`, "
            f"capture=`{metrics.get('portfolio_capture_rate', 0.0):.4f}`, "
            f"expected_payout=`{metrics.get('portfolio_expected_payout', 0.0):.2f}`, "
            f"expected_utility=`{metrics.get('portfolio_expected_utility', 0.0):.4f}`, "
            f"cash_rate=`{metrics.get('portfolio_cash_rate', 0.0):.4f}`, "
            f"top3=`{metrics.get('portfolio_top3_equity', 0.0):.4f}`"
        )
    report_lines.extend(["", "## Sensitivity Summary", ""])
    for contest_id, metrics in result.sensitivity_summary.items():
        report_lines.append(
            f"- `{contest_id}`: portfolio_fpe=`{metrics.get('portfolio_first_place_equity', 0.0):.4f}`, "
            f"capture=`{metrics.get('portfolio_capture_rate', 0.0):.4f}`, "
            f"expected_payout=`{metrics.get('portfolio_expected_payout', 0.0):.2f}`, "
            f"expected_utility=`{metrics.get('portfolio_expected_utility', 0.0):.4f}`"
        )

    report_lines.extend(["", "## Posterior Leaders", ""])
    for team in team_posteriors[:10]:
        report_lines.append(
            f"- `{team['team_name']}`: mean_win_probability=`{team['mean_win_probability']:.4f}`, "
            f"uncertainty=`{team['mean_uncertainty']:.4f}`, "
            f"lower_10=`{team['posterior_lower_10']:.4f}`, upper_90=`{team['posterior_upper_90']:.4f}`"
        )

    report_lines.extend(["", "## Public Duplication Leaders", ""])
    for row in duplication_rows[:10]:
        report_lines.append(
            f"- `{row['team_name']}`: title_dupes=`{row.get('expected_title_duplicates', 0.0):.1f}`, "
            f"final_four_dupes=`{row.get('expected_final_four_duplicates', 0.0):.1f}`, "
            f"path_duplication=`{row.get('path_duplication_proxy', 0.0):.4f}`"
        )
    if release_evaluation is not None:
        report_lines.extend(
            [
                "",
                "## Release Evaluation",
                "",
                f"- Portfolio release objective score: `{getattr(release_evaluation, 'release_objective_score', 0.0):.4f}`",
                f"- Portfolio guardrail failures: `{getattr(release_evaluation, 'guardrail_failures', [])}`",
                f"- Portfolio min individual FPE: `{getattr(release_evaluation, 'min_individual_fpe', 0.0):.6f}`",
                f"- Portfolio average duplication proxy: `{getattr(release_evaluation, 'average_duplication_proxy', 0.0):.4f}`",
                f"- Portfolio Final Four repeat penalty: `{getattr(release_evaluation, 'final_four_repeat_penalty', 0.0):.4f}`",
                f"- Portfolio region winner repeat penalty: `{getattr(release_evaluation, 'region_winner_repeat_penalty', 0.0):.4f}`",
            ]
        )
    if naive_release_evaluation is not None:
        report_lines.extend(
            [
                f"- Naive release objective score: `{getattr(naive_release_evaluation, 'release_objective_score', 0.0):.4f}`",
                f"- Naive guardrail failures: `{getattr(naive_release_evaluation, 'guardrail_failures', [])}`",
            ]
        )
    report_lines.append("")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    dashboard_spec_path = output_dir / f"{base_name}_dashboard-spec.md"
    dashboard_lines = [
        "# Bracket Portfolio Dashboard Spec",
        "",
        "## Purpose",
        "",
        "Local dashboard for comparing V10 candidates, posterior uncertainty, public duplication, and payout-aware portfolio behavior.",
        "",
        "## Required Views",
        "",
    ]
    for section in result.dashboard_spec_sections:
        dashboard_lines.append(f"- {section}")
    dashboard_lines.extend(
        [
            "",
            "## Key Panels",
            "",
            "- Candidate table with posterior probability, uncertainty bands, duplication proxy, and weighted first-place equity.",
            "- Contest payout panel with expected payout, expected utility, cash rate, and top-3 equity.",
            "- Public-field panel showing champion crowding and path duplication estimates.",
            "- Artifact provenance panel with model ids, training profile, and artifact paths.",
            "",
        ]
    )
    with dashboard_spec_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(dashboard_lines))

    return {
        "portfolio_json": portfolio_json,
        "report": report_path,
        "dashboard_spec": dashboard_spec_path,
        **finalist_paths,
    }
