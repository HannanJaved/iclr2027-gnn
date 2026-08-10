from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn_rashomon.rashomon.filtering import (
    filter_by_training_loss,
    validate_partition,
)
from gnn_rashomon.rashomon.io import discover_runs, write_json
from gnn_rashomon.rashomon.tolerance import loss_threshold

LOSS_DEFINITION = "mean cross-entropy over train_mask nodes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and filter trained runs into a Rashomon set."
    )
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--architecture", default="gcn")
    parser.add_argument("--set-type", default="seed")
    parser.add_argument(
        "--rashomon-type",
        default=None,
        help="Optional run metadata filter. Use hyperparameter for hyperparameter sweeps.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--tolerance-mode", choices=["absolute", "relative"], default="absolute")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--baseline-run-id", default=None)
    parser.add_argument(
        "--deduplicate-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep one record per seed, preferring lower train loss. "
            "Intended for seed Rashomon sets."
        ),
    )
    parser.add_argument(
        "--deduplicate-hyperparameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the lowest-train-loss run for each model/optimization configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    runs = discover_runs(
        metrics_dir=metrics_dir,
        dataset=args.dataset,
        architecture=args.architecture,
        max_epochs=args.max_epochs,
        rashomon_type=args.rashomon_type,
    )
    if not runs:
        raise SystemExit(
            f"No runs found for dataset={args.dataset} architecture={args.architecture} "
            f"max_epochs={args.max_epochs} in {metrics_dir}"
        )
    if args.deduplicate_seeds:
        best_by_seed: dict[int, dict[str, object]] = {}
        for run in runs:
            seed = int(run["seed"])
            current = best_by_seed.get(seed)
            if current is None or float(run["train_loss"]) < float(current["train_loss"]):
                best_by_seed[seed] = run
        runs = [best_by_seed[seed] for seed in sorted(best_by_seed)]
    if args.deduplicate_hyperparameters:
        best_by_configuration: dict[str, dict[str, object]] = {}
        ignored = {"device", "max_epochs", "early_stopping_patience"}
        for run in runs:
            hyperparameters = dict(run.get("hyperparameters", {}))
            configuration = {
                key: value for key, value in hyperparameters.items() if key not in ignored
            }
            key = json.dumps(configuration, sort_keys=True)
            current = best_by_configuration.get(key)
            if current is None or float(run["train_loss"]) < float(current["train_loss"]):
                best_by_configuration[key] = run
        runs = [best_by_configuration[key] for key in sorted(best_by_configuration)]

    by_id = {str(run["run_id"]): run for run in runs}
    if args.baseline_run_id:
        baseline = by_id[args.baseline_run_id]
    else:
        baseline = min(runs, key=lambda run: float(run["train_loss"]))

    baseline_loss = float(baseline["train_loss"])
    losses = {str(run["run_id"]): float(run["train_loss"]) for run in runs}
    retained, rejected = filter_by_training_loss(
        losses=losses,
        baseline_loss=baseline_loss,
        mode=args.tolerance_mode,
        epsilon=args.epsilon,
    )
    validate_partition(losses.keys(), retained, rejected)

    set_id = (
        f"{args.dataset}-{args.architecture}-{args.set_type}-"
        f"{args.tolerance_mode}{args.epsilon:g}-epochs{args.max_epochs}"
    )
    payload = {
        "set_id": set_id,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "set_type": args.set_type,
        "baseline_run_id": baseline["run_id"],
        "baseline_loss": baseline_loss,
        "epsilon": args.epsilon,
        "tolerance_mode": args.tolerance_mode,
        "loss_definition": LOSS_DEFINITION,
        "threshold": loss_threshold(baseline_loss, args.tolerance_mode, args.epsilon),
        "candidate_run_ids": sorted(losses),
        "retained_run_ids": sorted(retained),
        "rejected_run_ids": sorted(rejected),
        "candidate_count": len(losses),
        "retained_count": len(retained),
        "rejected_count": len(rejected),
        "deduplicate_seeds": args.deduplicate_seeds,
        "deduplicate_hyperparameters": args.deduplicate_hyperparameters,
        "runs": {
            run_id: {
                "train_loss": losses[run_id],
                "validation_accuracy": float(by_id[run_id]["validation_accuracy"]),
                "test_accuracy": float(by_id[run_id]["test_accuracy"]),
                "probability_path": by_id[run_id]["probability_path"],
                "record_path": by_id[run_id]["_record_path"],
            }
            for run_id in sorted(losses)
        },
    }
    path = output_dir / "rashomon_sets" / f"{set_id}.json"
    write_json(path, payload)
    print(f"Wrote Rashomon set to {path}")
    print(f"baseline={baseline['run_id']} train_loss={baseline_loss:.6f}")
    print(f"retained={len(retained)} rejected={len(rejected)} threshold={payload['threshold']:.6f}")


if __name__ == "__main__":
    main()
