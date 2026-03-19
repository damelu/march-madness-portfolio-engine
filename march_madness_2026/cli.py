from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_selection_sunday_dataset
from .demo import build_demo_dataset
from .engine import BracketPortfolioEngine
from .reporting import write_outputs as write_v9_outputs
from .v10 import V10BracketPortfolioEngine
from .v10 import DEFAULT_V10_INFERENCE_SNAPSHOT, materialize_inference_snapshot
from .v10.portfolio import ReleaseContractConfig
from .v10.reporting import write_outputs as write_v10_outputs
from .v10.search import (
    DEFAULT_BLENDED_WEIGHT_FLOOR,
    V10SearchParams,
    apply_params_to_engine,
    resolve_release_seeds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V10_SOURCE_DATASET = PROJECT_ROOT / "data" / "features" / "selection_sunday" / "snapshot.json"
DEFAULT_V10_DATASET = DEFAULT_V10_INFERENCE_SNAPSHOT
DEFAULT_V10_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v10_test"
DEFAULT_V10_GAME_MODEL = PROJECT_ROOT / "data" / "models" / "v10_6" / "game_model_v10_6_release.pkl"
DEFAULT_V10_PUBLIC_FIELD = PROJECT_ROOT / "data" / "models" / "v10_6" / "public_field_v10_6_release.pkl"


def _build_simulation_overrides(args: argparse.Namespace) -> dict[str, int] | None:
    overrides: dict[str, int] = {}
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "num_tournament_sims", None) is not None:
        overrides["num_tournament_simulations"] = args.num_tournament_sims
    if getattr(args, "num_candidates", None) is not None:
        overrides["num_candidate_brackets"] = args.num_candidates
    return overrides or None


def _print_generated_outputs(output_paths: dict[str, Path]) -> None:
    print("generated_outputs:")
    for key, value in output_paths.items():
        print(f"  {key}: {value}")


def _best_finalist_equity(run_bundle: dict[str, object], contest_id: str) -> float:
    result = run_bundle["result"]
    finalists = getattr(result, "finalists", [])
    best_equity = 0.0
    for finalist in finalists:
        metric = finalist.contest_metrics.get(contest_id)
        if metric is not None and metric.first_place_equity > best_equity:
            best_equity = float(metric.first_place_equity)
    return best_equity


def _normalize_v10_run_bundle(run_bundle: dict[str, object]) -> dict[str, object]:
    result = run_bundle["result"]
    scenario_summary = getattr(result, "scenario_summary", {})
    sensitivity_summary = getattr(result, "sensitivity_summary", {})

    normalized_scenario_summary: dict[str, dict[str, float]] = {}
    for contest_id, metrics in scenario_summary.items():
        payload = dict(metrics)
        payload.setdefault("best_finalist_equity", _best_finalist_equity(run_bundle, contest_id))
        normalized_scenario_summary[contest_id] = payload

    normalized_sensitivity_summary: dict[str, dict[str, float]] = {}
    for contest_id, metrics in sensitivity_summary.items():
        payload = dict(metrics)
        payload.setdefault("best_finalist_equity", _best_finalist_equity(run_bundle, contest_id))
        normalized_sensitivity_summary[contest_id] = payload

    result.scenario_summary = normalized_scenario_summary
    result.sensitivity_summary = normalized_sensitivity_summary
    return run_bundle


def _run_v9(args: argparse.Namespace) -> int:
    overrides = _build_simulation_overrides(args)

    dataset = None
    if args.use_demo:
        dataset = None
    elif args.input_json is not None:
        dataset = load_selection_sunday_dataset(args.input_json)

    engine = BracketPortfolioEngine(simulation_overrides=overrides)
    run_bundle = engine.run(dataset=dataset)
    output_paths = write_v9_outputs(args.output_dir, run_bundle)
    _print_generated_outputs(output_paths)
    return 0


def _run_v10(args: argparse.Namespace) -> int:
    overrides = _build_simulation_overrides(args)

    dataset = None
    if args.use_demo:
        demo_seed = overrides.get("seed") if overrides else None
        dataset = build_demo_dataset(seed=demo_seed or 20260317)
    else:
        dataset_path = args.dataset
        if not dataset_path.exists():
            dataset_path = materialize_inference_snapshot(
                source_path=args.source_snapshot,
                output_path=args.dataset,
            )
        dataset = load_selection_sunday_dataset(dataset_path)

    engine = V10BracketPortfolioEngine(
        simulation_overrides=overrides,
        training_profile_id=args.training_profile_id,
        model_artifact_path=args.game_model_artifact,
        public_field_artifact_path=args.public_field_artifact,
    )
    params_payload = None
    release_contract_payload: dict[str, object] = {}
    if args.params_json is not None:
        with args.params_json.open("r", encoding="utf-8") as handle:
            params_payload = json.load(handle)
        release_contract_payload = dict(params_payload.get("release_contract", {}))

    contest_mode = args.contest_mode
    if params_payload is not None and contest_mode == "blended":
        contest_mode = str(params_payload.get("contest_mode", contest_mode))

    blended_weight_floor = args.blended_weight_floor
    if params_payload is not None and blended_weight_floor == DEFAULT_BLENDED_WEIGHT_FLOOR:
        blended_weight_floor = float(
            params_payload.get("blended_weight_floor")
            or params_payload.get("release_contract", {}).get("blended_weight_floor")
            or blended_weight_floor
        )

    if params_payload is not None:
        params_values = params_payload.get("params", params_payload)
        apply_params_to_engine(
            engine,
            V10SearchParams(**params_values),
            contest_mode=contest_mode,
            blended_weight_floor=blended_weight_floor,
        )

    release_seeds = resolve_release_seeds(
        base_seed=engine.simulation_profile.seed,
        explicit_seeds=params_payload.get("evaluation_seeds") if params_payload and not args.release_seeds else None,
        explicit_seed_string=args.release_seeds,
        seed_count=args.release_seed_count,
    )
    release_contract = ReleaseContractConfig(**release_contract_payload)
    release_contract = ReleaseContractConfig(
            **{
                **release_contract.__dict__,
            "objective_name": "v10_6_release_objective",
            "blended_weight_floor": float(blended_weight_floor),
            "allow_naive_regression": bool(args.allow_naive_regression)
            or bool(args.use_demo)
            or release_contract.allow_naive_regression,
            "allow_zero_fpe_finalist": bool(args.allow_zero_fpe_finalist)
            or bool(args.use_demo)
            or release_contract.allow_zero_fpe_finalist,
            "fail_on_guardrail": not (bool(args.allow_guardrail_failure) or bool(args.use_demo)),
        }
    )

    run_bundle = engine.run(
        dataset=dataset,
        release_seeds=release_seeds,
        release_contract=release_contract,
    )
    if args.payout_profile:
        simulation_config = dict(run_bundle.get("simulation_config", {}))
        simulation_config["requested_payout_profile"] = args.payout_profile
        run_bundle = {**run_bundle, "simulation_config": simulation_config}
    run_bundle = _normalize_v10_run_bundle(run_bundle)
    output_paths = write_v10_outputs(args.output_dir, run_bundle)
    _print_generated_outputs(output_paths)
    guardrail_failures = run_bundle.get("selection_metadata", {}).get("portfolio_guardrail_failures", [])
    if guardrail_failures and not args.allow_guardrail_failure:
        print(f"release_guardrail_failures: {', '.join(guardrail_failures)}")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a March Madness bracket portfolio.")
    parser.add_argument("--input-json", type=Path, help="Selection Sunday dataset in JSON format.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for finalist JSON and report outputs.",
    )
    parser.add_argument("--seed", type=int, help="Override the default simulation seed.")
    parser.add_argument(
        "--num-tournament-sims",
        type=int,
        help="Override the default number of tournament simulations.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        help="Override the default number of candidate brackets to generate.",
    )
    parser.add_argument(
        "--use-demo",
        action="store_true",
        help="Force the synthetic demo dataset even if no input file is supplied.",
    )
    return parser


def build_v10_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an integrated V10 March Madness bracket portfolio.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_V10_DATASET,
        help="Selection Sunday dataset in JSON format.",
    )
    parser.add_argument(
        "--source-snapshot",
        type=Path,
        default=DEFAULT_V10_SOURCE_DATASET,
        help="Source Selection Sunday snapshot used to materialize the isolated V10 inference dataset when needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_V10_OUTPUT_DIR,
        help="Directory for finalist JSON and report outputs.",
    )
    parser.add_argument("--seed", type=int, help="Override the default simulation seed.")
    parser.add_argument(
        "--num-tournament-sims",
        type=int,
        help="Override the default number of tournament simulations.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        help="Override the default number of candidate brackets to generate.",
    )
    parser.add_argument(
        "--training-profile-id",
        default="release_v10_6",
        help="V10 training profile to use when bootstrapping artifacts.",
    )
    parser.add_argument(
        "--contest-mode",
        choices=("blended", "standard_small", "standard_mid", "standard_large"),
        default="blended",
        help="Contest-mode contract to use when applying params.",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        help="Optional V10 optimizer checkpoint or params payload to apply before building.",
    )
    parser.add_argument(
        "--game-model-artifact",
        type=Path,
        default=DEFAULT_V10_GAME_MODEL,
        help="V10 game model artifact path.",
    )
    parser.add_argument(
        "--public-field-artifact",
        type=Path,
        default=DEFAULT_V10_PUBLIC_FIELD,
        help="Optional public field artifact path.",
    )
    parser.add_argument(
        "--payout-profile",
        default="",
        help="Optional payout profile label to preserve in the run bundle.",
    )
    parser.add_argument(
        "--release-seed-count",
        type=int,
        default=3,
        help="Number of release-evaluation seeds to use when explicit release seeds are not supplied.",
    )
    parser.add_argument(
        "--release-seeds",
        default="",
        help="Comma-separated release-evaluation seeds.",
    )
    parser.add_argument(
        "--blended-weight-floor",
        type=float,
        default=DEFAULT_BLENDED_WEIGHT_FLOOR,
        help="Minimum per-contest weight allowed in blended contest mode.",
    )
    parser.add_argument(
        "--allow-guardrail-failure",
        action="store_true",
        help="Do not fail closed when the release contract guardrails fail.",
    )
    parser.add_argument(
        "--allow-naive-regression",
        action="store_true",
        help="Allow release candidates that regress the naive baseline on guarded metrics.",
    )
    parser.add_argument(
        "--allow-zero-fpe-finalist",
        action="store_true",
        help="Allow effectively zero-FPE finalists in the published portfolio.",
    )
    parser.add_argument(
        "--use-demo",
        action="store_true",
        help="Force the synthetic demo dataset even if no input file is supplied.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_v9(args)


def main_v10(argv: list[str] | None = None) -> int:
    args = build_v10_parser().parse_args(argv)
    return _run_v10(args)
