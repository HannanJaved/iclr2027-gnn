from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.multiplicity.rashomon_capacity import probability_diameter
from gnn_rashomon.rashomon.io import read_json, write_json
from gnn_rashomon.training.checkpointing import checkpoint_logits


def _masked_accuracy(probabilities: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-model accuracy on a boolean mask; shape (n_models,)."""
    preds = probabilities.argmax(axis=-1)
    masked_preds = preds[:, mask]
    masked_labels = labels[mask]
    return (masked_preds == masked_labels[None, :]).mean(axis=-1)


def _accuracy_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    val_acc = _masked_accuracy(probabilities, labels, val_mask)
    test_acc = _masked_accuracy(probabilities, labels, test_mask)
    return {
        f"{prefix}_validation_accuracy_mean": float(val_acc.mean()),
        f"{prefix}_validation_accuracy_min": float(val_acc.min()),
        f"{prefix}_validation_accuracy_max": float(val_acc.max()),
        f"{prefix}_test_accuracy_mean_report_only": float(test_acc.mean()),
        f"{prefix}_test_accuracy_min_report_only": float(test_acc.min()),
        f"{prefix}_test_accuracy_max_report_only": float(test_acc.max()),
        f"{prefix}_n_models_within_val_delta_0p02": int(
            (val_acc >= float(val_acc.max()) - 0.02 - 1e-12).sum()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one fixed retained model set on a rewired graph. Model weights and "
            "membership remain fixed, making topology the only changed model input."
        )
    )
    parser.add_argument("--rashomon-set", required=True)
    parser.add_argument("--original-node-table", required=True)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--rewiring-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--node-output", default=None)
    parser.add_argument("--checkpoint-probability-tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_path(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _checkpoint_paths(rashomon: dict[str, object], project_root: Path) -> list[Path]:
    runs = rashomon["runs"]
    assert isinstance(runs, dict)
    retained = rashomon["retained_run_ids"]
    assert isinstance(retained, list)
    paths: list[Path] = []
    for run_id in retained:
        run = runs[str(run_id)]
        assert isinstance(run, dict)
        record_path = _resolve_path(str(run["record_path"]), project_root)
        record = read_json(record_path)
        paths.append(_resolve_path(str(record["checkpoint_path"]), project_root))
    return paths


def _archived_probabilities(
    rashomon: dict[str, object], project_root: Path
) -> np.ndarray:
    import torch

    runs = rashomon["runs"]
    assert isinstance(runs, dict)
    retained = rashomon["retained_run_ids"]
    assert isinstance(retained, list)
    arrays = []
    for run_id in retained:
        run = runs[str(run_id)]
        assert isinstance(run, dict)
        path = _resolve_path(str(run["probability_path"]), project_root)
        arrays.append(torch.load(path, map_location="cpu", weights_only=True).numpy())
    return np.stack(arrays)


def _fixed_set_probabilities(
    checkpoint_paths: list[Path],
    *,
    x: object,
    edge_index: object,
    out_channels: int,
    device: str,
) -> np.ndarray:
    import torch

    x = x.to(device)
    edge_index = edge_index.to(device)
    probabilities = []
    for checkpoint_path in checkpoint_paths:
        logits = checkpoint_logits(
            checkpoint_path,
            x=x,
            edge_index=edge_index,
            out_channels=out_channels,
            map_location=device,
        )
        probabilities.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.stack(probabilities)


def _mean_for_nodes(values: np.ndarray, nodes: list[int]) -> float | None:
    return float(values[np.asarray(nodes, dtype=int)].mean()) if nodes else None


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    rashomon = read_json(Path(args.rashomon_set))
    metadata = read_json(Path(args.rewiring_metadata))
    original = pd.read_csv(args.original_node_table).sort_values("node_id")
    dataset = str(rashomon["dataset"])

    config = load_config([f"dataset={dataset}"], project_root / "configs")
    original_graph = load_graph(
        dataset,
        root=str(project_root / config["dataset_config"]["root"]),
        dataset_config=config["dataset_config"],
    )
    graph = load_graph(
        dataset,
        root=str(project_root / config["dataset_config"]["root"]),
        graph_path=args.graph_path,
        dataset_config=config["dataset_config"],
    )
    if len(original) != int(graph.data.num_nodes):
        raise ValueError("Original node table and rewired graph have different node counts.")
    if not np.array_equal(original["node_id"].to_numpy(dtype=int), np.arange(len(original))):
        raise ValueError("Original node table must contain every node exactly once in node order.")

    checkpoints = _checkpoint_paths(rashomon, project_root)
    archived_probabilities = _archived_probabilities(rashomon, project_root)
    checkpoint_original_probabilities = _fixed_set_probabilities(
        checkpoints,
        x=original_graph.data.x,
        edge_index=original_graph.data.edge_index,
        out_channels=int(original_graph.metadata.num_classes),
        device=args.device,
    )
    checkpoint_probability_max_abs_error = float(
        np.max(np.abs(checkpoint_original_probabilities - archived_probabilities))
    )
    if checkpoint_probability_max_abs_error > args.checkpoint_probability_tolerance:
        raise ValueError(
            "Checkpoint inference does not reproduce archived probabilities: "
            f"max_abs_error={checkpoint_probability_max_abs_error:.6g}, "
            f"tolerance={args.checkpoint_probability_tolerance:.6g}."
        )
    probabilities = _fixed_set_probabilities(
        checkpoints,
        x=graph.data.x,
        edge_index=graph.data.edge_index,
        out_channels=int(graph.metadata.num_classes),
        device=args.device,
    )
    rewired_diameter = probability_diameter(probabilities, method="exact")
    rewired_predictions = probabilities.argmax(axis=-1)
    rewired_disagreement = (rewired_predictions != rewired_predictions[0:1]).any(axis=0)
    original_diameter = original["rashomon_capacity"].to_numpy(dtype=float)
    original_disagreement = original["has_prediction_disagreement"].to_numpy(dtype=bool)
    delta_diameter = rewired_diameter - original_diameter

    labels = original_graph.data.y.detach().cpu().numpy()
    val_mask = original_graph.data.val_mask.detach().cpu().numpy().astype(bool)
    test_mask = original_graph.data.test_mask.detach().cpu().numpy().astype(bool)
    original_acc = _accuracy_summary(
        checkpoint_original_probabilities,
        labels,
        val_mask,
        test_mask,
        prefix="original",
    )
    rewired_acc = _accuracy_summary(
        probabilities,
        labels,
        val_mask,
        test_mask,
        prefix="rewired",
    )
    # Membership on G is fixed; K_delta(G') asks how many retained models would
    # still lie within delta of the best *original-graph* validation accuracy.
    original_val = _masked_accuracy(
        checkpoint_original_probabilities, labels, val_mask
    )
    rewired_val = _masked_accuracy(probabilities, labels, val_mask)
    best_original_val = float(original_val.max())
    n_within_delta_on_rewired = int(
        (rewired_val >= best_original_val - 0.02 - 1e-12).sum()
    )

    treatment_nodes = [int(value) for value in (metadata.get("treatment_nodes") or [])]
    control_nodes = [int(value) for value in (metadata.get("control_nodes") or [])]
    pair_count = min(len(treatment_nodes), len(control_nodes))
    matched_effect: float | None = None
    if pair_count:
        treatment_delta = delta_diameter[np.asarray(treatment_nodes[:pair_count], dtype=int)]
        control_delta = delta_diameter[np.asarray(control_nodes[:pair_count], dtype=int)]
        matched_effect = float(np.mean(treatment_delta - control_delta))

    output = {
        "dataset": dataset,
        "architecture": str(rashomon["architecture"]),
        "set_id": str(rashomon["set_id"]),
        "retained_count": len(checkpoints),
        "mode": str(metadata["mode"]),
        "strength": metadata.get("strength"),
        "target_homophily": metadata.get("target_homophily"),
        "tau": metadata.get("tau"),
        "rewiring_seed": metadata.get("seed"),
        "graph_path": str(args.graph_path),
        "rewiring_metadata": str(args.rewiring_metadata),
        "target_reached": metadata.get("target_reached") is not False,
        "degree_preserved": bool(metadata["degree_preserved"]),
        "features_preserved": bool(metadata["features_preserved"]),
        "labels_preserved": bool(metadata["labels_preserved"]),
        "masks_preserved": bool(metadata["masks_preserved"]),
        "fixed_model_weights": True,
        "fixed_rashomon_membership": True,
        "inference_device": args.device,
        "checkpoint_probability_max_abs_error": checkpoint_probability_max_abs_error,
        "checkpoint_probability_tolerance": args.checkpoint_probability_tolerance,
        "checkpoint_probability_validation_passed": True,
        "test_metrics_used_for_membership": False,
        "inference_unit": "rewired graph realization",
        "valid_for_inference": metadata.get("target_reached") is not False
        and bool(metadata["degree_preserved"])
        and bool(metadata["features_preserved"])
        and bool(metadata["labels_preserved"])
        and bool(metadata["masks_preserved"]),
        "original_mean_probability_diameter": float(original_diameter.mean()),
        "rewired_mean_probability_diameter": float(rewired_diameter.mean()),
        "delta_mean_probability_diameter": float(
            rewired_diameter.mean() - original_diameter.mean()
        ),
        "original_disagreement_fraction": float(original_disagreement.mean()),
        "rewired_disagreement_fraction": float(rewired_disagreement.mean()),
        "delta_disagreement_fraction": float(
            rewired_disagreement.mean() - original_disagreement.mean()
        ),
        **original_acc,
        **rewired_acc,
        "best_original_validation_accuracy": best_original_val,
        "n_models_within_original_val_delta_0p02_on_rewired": n_within_delta_on_rewired,
        "treatment_count": len(treatment_nodes),
        "control_count": len(control_nodes),
        "pair_count": pair_count,
        "mean_treatment_delta_diameter": _mean_for_nodes(delta_diameter, treatment_nodes),
        "mean_control_delta_diameter": _mean_for_nodes(delta_diameter, control_nodes),
        "matched_treatment_control_delta_diameter": matched_effect,
        "delta_treatment_entropy_mean": metadata.get("delta_treatment_entropy_mean"),
        "max_abs_control_entropy_delta": metadata.get("max_abs_control_entropy_delta"),
    }
    write_json(Path(args.output), output)

    if args.node_output:
        node_output = Path(args.node_output)
        node_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "node_id": np.arange(len(original)),
                "original_probability_diameter": original_diameter,
                "rewired_probability_diameter": rewired_diameter,
                "delta_probability_diameter": delta_diameter,
                "rewired_prediction_disagreement": rewired_disagreement,
            }
        ).to_csv(node_output, index=False)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
