from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Dict

from .models import BracketCandidate, CandidateContestMetrics, PortfolioResult
from .tournament import TournamentModel


def _contest_metric_dict(metric: CandidateContestMetrics) -> Dict[str, float]:
    return {
        "first_place_equity": round(metric.first_place_equity, 6),
        "average_finish": round(metric.average_finish, 4),
        "top_decile_rate": round(metric.top_decile_rate, 6),
    }


def _candidate_to_dict(candidate: BracketCandidate, model: TournamentModel) -> Dict[str, Any]:
    picks_by_round: Dict[str, list] = {}
    for meta, winner_index in zip(model.game_metadata, candidate.pick_indices):
        picks_by_round.setdefault(meta.round_name, []).append(
            {
                "slot": meta.slot_name,
                "region": meta.region,
                "winner_team_id": model.team_id_by_index[winner_index],
                "winner_team_name": model.team_name_by_index[winner_index],
                "winner_seed": int(model.seed_by_index[winner_index]),
            }
        )

    return {
        "bracket_id": candidate.bracket_id,
        "archetype": candidate.archetype,
        "risk_label": candidate.risk_label,
        "champion_team_id": candidate.champion_team_id,
        "champion_team_name": model.team_name(candidate.champion_team_id),
        "final_four_team_ids": candidate.final_four_team_ids,
        "final_four_team_names": [model.team_name(team_id) for team_id in candidate.final_four_team_ids],
        "chalkiness_proxy": candidate.chalkiness_proxy,
        "duplication_proxy": candidate.duplication_proxy,
        "weighted_first_place_equity": round(candidate.weighted_first_place_equity, 6),
        "weighted_average_finish": round(candidate.weighted_average_finish, 4),
        "scenario_fit": candidate.scenario_fit,
        "why_selected": candidate.why_selected,
        "contest_metrics": {
            contest_id: _contest_metric_dict(metric)
            for contest_id, metric in candidate.contest_metrics.items()
        },
        "picks_by_round": picks_by_round,
    }


def write_outputs(output_dir: Path, run_bundle: Dict[str, Any]) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: PortfolioResult = run_bundle["result"]
    model: TournamentModel = run_bundle["model"]
    stamp = date.today().isoformat()
    base_name = f"{stamp}_march-madness-bracket-portfolio"

    finalists_payload = [_candidate_to_dict(candidate, model) for candidate in result.finalists]
    portfolio_payload = {
        "run_id": result.run_id,
        "data_source": result.data_source,
        "candidate_count": result.candidate_count,
        "summary": asdict(result.summary),
        "scenario_summary": result.scenario_summary,
        "sensitivity_summary": result.sensitivity_summary,
        "finalists": finalists_payload,
    }
    # Include engine version
    if "engine_version" in run_bundle:
        portfolio_payload["engine_version"] = run_bundle["engine_version"]
    # Include simulation config for reproducibility if present in run bundle
    if "simulation_config" in run_bundle:
        portfolio_payload["simulation_config"] = run_bundle["simulation_config"]
    if "selection_metadata" in run_bundle:
        portfolio_payload["selection_metadata"] = run_bundle["selection_metadata"]

    portfolio_json = output_dir / f"{base_name}_finalists.json"
    with portfolio_json.open("w", encoding="utf-8") as handle:
        json.dump(portfolio_payload, handle, indent=2)

    finalist_paths: Dict[str, Path] = {}
    for index, finalist in enumerate(finalists_payload, start=1):
        bracket_path = output_dir / f"{base_name}_bracket-{index:02d}.json"
        with bracket_path.open("w", encoding="utf-8") as handle:
            json.dump(finalist, handle, indent=2)
        finalist_paths[f"bracket_{index:02d}"] = bracket_path

    report_path = output_dir / f"{base_name}_report.md"
    report_lines = [
        "# March Madness Bracket Portfolio Report",
        "",
        f"- Run ID: `{result.run_id}`",
        f"- Data source: `{result.data_source}`",
        f"- Candidate universe: `{result.candidate_count}` brackets",
        f"- Finalists: `{len(result.finalists)}` brackets",
        "",
        "## Portfolio Summary",
        "",
        f"- Weighted portfolio first-place equity: `{result.summary.weighted_portfolio_first_place_equity:.4f}`",
        f"- Naive baseline first-place equity: `{result.summary.naive_baseline_first_place_equity:.4f}`",
        f"- Weighted portfolio objective: `{result.summary.weighted_portfolio_objective:.4f}`",
        f"- Naive top-5 baseline objective: `{result.summary.naive_baseline_objective:.4f}`",
        f"- Weighted portfolio capture rate: `{result.summary.weighted_portfolio_capture_rate:.4f}`",
        f"- Naive top-5 baseline capture rate: `{result.summary.naive_baseline_capture_rate:.4f}`",
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

    # Run Config section
    engine_version = run_bundle.get("engine_version", "unknown")
    simulation_config = run_bundle.get("simulation_config", {})
    selection_metadata = run_bundle.get("selection_metadata", {})
    report_lines.extend([
        "## Run Config",
        "",
        f"- Engine version: `{engine_version}`",
        f"- Seed: `{simulation_config.get('seed', 'N/A')}`",
        f"- Tournament simulations: `{simulation_config.get('num_tournament_simulations', 'N/A')}`",
        f"- Candidate brackets: `{simulation_config.get('num_candidate_brackets', 'N/A')}`",
        f"- Portfolio size: `{simulation_config.get('portfolio_size', 'N/A')}`",
        f"- Candidate pool size: `{selection_metadata.get('candidate_pool_size', 'N/A')}`",
        f"- Positive equity pool: `{selection_metadata.get('positive_equity_pool_size', 'N/A')}`",
        f"- Zero equity fallback: `{selection_metadata.get('used_zero_equity_fallback', 'N/A')}`",
        f"- Local search swaps: `{selection_metadata.get('local_search_swaps', 'N/A')}`",
        "",
    ])

    report_lines.extend(["## Scenario Summary", ""])
    for contest_id, metrics in result.scenario_summary.items():
        fpe_str = f", portfolio_first_place_equity=`{metrics['portfolio_first_place_equity']:.4f}`" if "portfolio_first_place_equity" in metrics else ""
        report_lines.append(
            f"- `{contest_id}`: portfolio_capture_rate=`{metrics['portfolio_capture_rate']:.4f}`, best_finalist_equity=`{metrics['best_finalist_equity']:.4f}`{fpe_str}"
        )
    report_lines.extend(["", "## Sensitivity Summary", ""])
    for contest_id, metrics in result.sensitivity_summary.items():
        fpe_str = f", portfolio_first_place_equity=`{metrics['portfolio_first_place_equity']:.4f}`" if "portfolio_first_place_equity" in metrics else ""
        report_lines.append(
            f"- `{contest_id}`: portfolio_capture_rate=`{metrics['portfolio_capture_rate']:.4f}`, best_finalist_equity=`{metrics['best_finalist_equity']:.4f}`{fpe_str}"
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
        "Local dashboard for comparing candidate brackets, scenario performance, and the final five-entry portfolio.",
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
            "- Candidate-universe table with filters for archetype, champion seed, chalkiness, and weighted first-place equity.",
            "- Scenario comparison panel for small, mid, large, flat-score, and upset-bonus environments.",
            "- Overlap matrix for the final five brackets.",
            "- Finalist detail panel with champion thesis, final-four path, and why-selected summary.",
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
