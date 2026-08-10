from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.multiplicity.rashomon_capacity import probability_diameter
from gnn_rashomon.rashomon.io import discover_runs
from gnn_rashomon.rashomon.tolerance import loss_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Rashomon retention, prediction collapse, and utility across tolerances."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--rashomon-type", required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--expected-candidates", type=int, default=None)
    parser.add_argument("--tolerance-mode", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--epsilon", type=float, nargs="+", required=True)
    parser.add_argument("--collapse-majority-threshold", type=float, default=0.99)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _load_probability(path: str) -> np.ndarray:
    import torch

    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return tensor.detach().cpu().numpy()


def audit_tolerances(
    runs: list[dict[str, object]],
    epsilons: list[float],
    mode: str,
    collapse_majority_threshold: float,
    probability_loader: Callable[[str], np.ndarray] = _load_probability,
) -> pd.DataFrame:
    if not runs:
        raise ValueError("No model runs were supplied for the health audit.")
    baseline_loss = min(float(run["train_loss"]) for run in runs)
    rows: list[dict[str, object]] = []
    probability_cache: dict[str, np.ndarray] = {}

    for epsilon in epsilons:
        threshold = loss_threshold(baseline_loss, mode, epsilon)
        retained = [run for run in runs if float(run["train_loss"]) <= threshold]
        probabilities = []
        collapsed = []
        majority_fractions = []
        prediction_vectors = []
        for run in retained:
            path = str(run["probability_path"])
            if path not in probability_cache:
                probability_cache[path] = probability_loader(path)
            probability = probability_cache[path]
            prediction = probability.argmax(axis=-1)
            counts = np.bincount(prediction, minlength=probability.shape[-1])
            majority_fraction = float(counts.max() / max(1, prediction.size))
            majority_fractions.append(majority_fraction)
            collapsed.append(
                bool(
                    np.count_nonzero(counts) <= 1
                    or majority_fraction >= collapse_majority_threshold
                )
            )
            probabilities.append(probability)
            prediction_vectors.append(prediction.tobytes())

        if probabilities:
            stacked = np.stack(probabilities)
            predictions = stacked.argmax(axis=-1)
            disagreement = float((predictions != predictions[0:1]).any(axis=0).mean())
            capacity_mean = float(
                probability_diameter(stacked, method="chunked", chunk_size=16).mean()
            )
        else:
            disagreement = float("nan")
            capacity_mean = float("nan")

        validation = [float(run["validation_accuracy"]) for run in retained]
        tests = [float(run["test_accuracy"]) for run in retained]
        rows.append(
            {
                "tolerance_mode": mode,
                "epsilon": float(epsilon),
                "baseline_train_loss": baseline_loss,
                "loss_threshold": threshold,
                "candidate_count": len(runs),
                "retained_count": len(retained),
                "collapsed_model_count": int(sum(collapsed)),
                "collapsed_model_fraction": (
                    float(np.mean(collapsed)) if collapsed else float("nan")
                ),
                "max_model_majority_fraction": max(majority_fractions, default=float("nan")),
                "unique_prediction_vector_count": len(set(prediction_vectors)),
                "fraction_prediction_disagreement": disagreement,
                "mean_rashomon_capacity": capacity_mean,
                "validation_accuracy_min": min(validation, default=float("nan")),
                "validation_accuracy_mean": (
                    float(np.mean(validation)) if validation else float("nan")
                ),
                "validation_accuracy_max": max(validation, default=float("nan")),
                "test_accuracy_min_report_only": min(tests, default=float("nan")),
                "test_accuracy_mean_report_only": float(np.mean(tests)) if tests else float("nan"),
                "test_accuracy_max_report_only": max(tests, default=float("nan")),
            }
        )
    return pd.DataFrame(rows)


def audit_model_runs(
    runs: list[dict[str, object]],
    collapse_majority_threshold: float,
    probability_loader: Callable[[str], np.ndarray] = _load_probability,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        probability = probability_loader(str(run["probability_path"]))
        prediction = probability.argmax(axis=-1)
        counts = np.bincount(prediction, minlength=probability.shape[-1])
        majority_fraction = float(counts.max() / max(1, prediction.size))
        hyperparameters = dict(run.get("hyperparameters", {}))
        rows.append(
            {
                "run_id": run["run_id"],
                "seed": run.get("seed"),
                "train_loss": run["train_loss"],
                "validation_accuracy": run["validation_accuracy"],
                "test_accuracy_report_only": run["test_accuracy"],
                "best_epoch": run.get("best_epoch"),
                "epochs_trained": run.get("epochs_trained"),
                "stopped_early": run.get("stopped_early"),
                "learning_rate": hyperparameters.get("learning_rate"),
                "weight_decay": hyperparameters.get("weight_decay"),
                "dropout": hyperparameters.get("dropout"),
                "hidden_channels": hyperparameters.get("hidden_channels"),
                "heads": hyperparameters.get("heads"),
                "predicted_class_count": int(np.count_nonzero(counts)),
                "majority_prediction_fraction": majority_fraction,
                "collapsed_prediction": bool(
                    np.count_nonzero(counts) <= 1
                    or majority_fraction >= collapse_majority_threshold
                ),
                "prediction_class_counts": json.dumps(counts.tolist()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    runs = discover_runs(
        metrics_dir=output_dir / "metrics",
        dataset=args.dataset,
        architecture=args.architecture,
        max_epochs=args.max_epochs,
        rashomon_type=args.rashomon_type,
    )
    if args.expected_candidates is not None and len(runs) != args.expected_candidates:
        raise SystemExit(
            f"Expected {args.expected_candidates} candidates but found {len(runs)} for "
            f"dataset={args.dataset} architecture={args.architecture} "
            f"rashomon_type={args.rashomon_type}."
        )
    probability_cache: dict[str, np.ndarray] = {}

    def cached_probability_loader(path: str) -> np.ndarray:
        if path not in probability_cache:
            probability_cache[path] = _load_probability(path)
        return probability_cache[path]

    models = audit_model_runs(
        runs=runs,
        collapse_majority_threshold=args.collapse_majority_threshold,
        probability_loader=cached_probability_loader,
    )
    table = audit_tolerances(
        runs=runs,
        epsilons=list(args.epsilon),
        mode=args.tolerance_mode,
        collapse_majority_threshold=args.collapse_majority_threshold,
        probability_loader=cached_probability_loader,
    )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = Path(f"{prefix}.csv")
    model_path = Path(f"{prefix}.models.csv")
    summary_path = Path(f"{prefix}.summary.json")
    table.to_csv(table_path, index=False)
    models.to_csv(model_path, index=False)
    summary = {
        "dataset": args.dataset,
        "architecture": args.architecture,
        "rashomon_type": args.rashomon_type,
        "max_epochs": args.max_epochs,
        "collapse_majority_threshold": args.collapse_majority_threshold,
        "candidate_count": len(runs),
        "expected_candidate_count": args.expected_candidates,
        "test_accuracy_is_report_only": True,
        "table": str(table_path),
        "model_table": str(model_path),
        "rows": table.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote Rashomon health audit to {table_path}")
    print(f"Wrote model-level health audit to {model_path}")
    print(f"Wrote Rashomon health summary to {summary_path}")


if __name__ == "__main__":
    main()
