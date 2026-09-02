from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.rewiring.homophily_targeted import homophily_targeted_rewire_graph
from gnn_rashomon.rewiring.local_entanglement import local_entanglement_rewire_graph
from gnn_rashomon.rewiring.random_rewire import random_rewire_graph
from gnn_rashomon.rewiring.structure_preserving import structure_preserving_rewire_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate degree-preserving rewired graph artifacts."
    )
    parser.add_argument("--dataset", default="cora")
    parser.add_argument(
        "--mode",
        choices=[
            "random",
            "homophily_targeted",
            "local_entanglement",
            "structure_preserving",
        ],
        default="random",
    )
    parser.add_argument("--strength", type=float, default=0.05)
    parser.add_argument("--target-homophily", type=float, default=None)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--max-attempts", type=int, default=50000)
    parser.add_argument("--max-feature-drift", type=float, default=None)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--delta-e", type=float, default=0.02)
    parser.add_argument("--treatment-count", type=int, default=50)
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument(
        "--direction",
        choices=["increase", "decrease"],
        default="increase",
        help="For local_entanglement mode: push treatment-node entropy up or down.",
    )
    parser.add_argument(
        "--homophily-tol",
        type=float,
        default=0.0,
        help="Max allowed local-homophily change per endpoint (structure_preserving).",
    )
    parser.add_argument(
        "--entropy-tol",
        type=float,
        default=0.0,
        help="Max allowed neighborhood-entropy change per endpoint (structure_preserving).",
    )
    parser.add_argument(
        "--target-accepted-swaps",
        type=int,
        default=None,
        help="For random mode: stop after this many accepted swaps (matched-budget control).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config = load_config([f"dataset={args.dataset}"], project_root / "configs")
    graph = load_graph(
        args.dataset,
        root=str(project_root / config["dataset_config"]["root"]),
        dataset_config=config["dataset_config"],
    )
    if args.mode == "homophily_targeted" and args.target_homophily is None:
        raise SystemExit("--target-homophily is required for homophily_targeted rewiring.")
    strength_id = str(args.strength).replace(".", "p")
    target_id = (
        ""
        if args.target_homophily is None
        else f"-target{str(args.target_homophily).replace('.', 'p')}"
    )
    if args.mode == "local_entanglement":
        direction_suffix = "" if args.direction == "increase" else "-decrease"
        target_id = f"-tau{str(args.tau).replace('.', 'p')}{direction_suffix}"
    mode_tag = args.mode
    if args.mode == "random" and args.target_accepted_swaps is not None:
        mode_tag = "random_matched_accepted"
        target_id = f"-acc{args.target_accepted_swaps}"
        strength_id = "matched"
    graph_id = f"{args.dataset}-{mode_tag}{target_id}-{strength_id}-seed{args.seed}"
    output_dir = Path(args.output_dir)
    graph_path = output_dir / "graphs" / f"{graph_id}.pt"
    if args.mode == "random":
        result = random_rewire_graph(
            graph=graph,
            output_path=graph_path,
            strength=args.strength,
            seed=args.seed,
            target_accepted_swaps=args.target_accepted_swaps,
        )
    elif args.mode == "homophily_targeted":
        result = homophily_targeted_rewire_graph(
            graph=graph,
            output_path=graph_path,
            target_homophily=float(args.target_homophily),
            tolerance=args.tolerance,
            seed=args.seed,
            max_attempts=args.max_attempts,
            max_feature_drift=args.max_feature_drift,
        )
    elif args.mode == "structure_preserving":
        result = structure_preserving_rewire_graph(
            graph=graph,
            output_path=graph_path,
            strength=args.strength,
            seed=args.seed,
            homophily_tol=args.homophily_tol,
            entropy_tol=args.entropy_tol,
        )
    else:
        result = local_entanglement_rewire_graph(
            graph=graph,
            output_path=graph_path,
            tau=args.tau,
            delta_e=args.delta_e,
            max_feature_drift=(
                args.max_feature_drift if args.max_feature_drift is not None else 0.02
            ),
            seed=args.seed,
            max_attempts=args.max_attempts,
            treatment_count=args.treatment_count,
            min_degree=args.min_degree,
            direction=args.direction,
        )
    result_path = output_dir / "graphs" / f"{graph_id}.json"
    result_payload = {
        **result.to_dict(),
        "seed": args.seed,
        "strength": args.strength,
        "max_attempts": args.max_attempts,
        "max_feature_drift": args.max_feature_drift,
        "target_accepted_swaps": args.target_accepted_swaps,
    }
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote rewired graph to {graph_path}")
    print(f"Wrote rewiring metadata to {result_path}")
    print(json.dumps(result_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
