from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.cli.evaluate_fixed_set_rewiring import (
    _archived_probabilities,
    _checkpoint_paths,
    _fixed_set_probabilities,
    _masked_accuracy,
)
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.multiplicity.rashomon_capacity import (
    pairwise_total_variation_summaries,
    probability_diameter,
)
from gnn_rashomon.rashomon.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute fixed-ensemble diameter restricted to K_delta(G')-viable "
            "members, alongside the usual full-frozen-set diameter, for one "
            "rewired-graph realization. Answers the question raised in review: "
            "does the structure-preserving-vs-matched-random contrast survive "
            "once both arms are compared on their own viable subset, rather "
            "than the full frozen set (which includes members that have "
            "already fallen out of delta-tolerance on G')?"
        )
    )
    parser.add_argument("--rashomon-set", required=True)
    parser.add_argument("--original-node-table", required=True)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--rewiring-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--checkpoint-probability-tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve_path(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


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

    labels = original_graph.data.y.detach().cpu().numpy()
    val_mask = original_graph.data.val_mask.detach().cpu().numpy().astype(bool)

    original_val = _masked_accuracy(checkpoint_original_probabilities, labels, val_mask)
    rewired_val = _masked_accuracy(probabilities, labels, val_mask)
    best_original_val = float(original_val.max())
    viable_mask = rewired_val >= (best_original_val - args.delta - 1e-12)
    n_viable = int(viable_mask.sum())

    # Full-set diameter, on both G and G', so the delta below matches what
    # Table 3 already reports (sanity check on this independently-run script).
    original_full_diameter = probability_diameter(checkpoint_original_probabilities, method="exact")
    rewired_full_diameter = probability_diameter(probabilities, method="exact")
    mean_full_set_diameter = float(rewired_full_diameter.mean())
    delta_diameter_full = float(rewired_full_diameter.mean() - original_full_diameter.mean())

    _, tv_original_full_mean_per_node, _ = pairwise_total_variation_summaries(
        checkpoint_original_probabilities
    )
    _, tv_rewired_full_mean_per_node, _ = pairwise_total_variation_summaries(probabilities)
    mean_tv_original_full = float(tv_original_full_mean_per_node.mean())
    mean_tv_rewired_full = float(tv_rewired_full_mean_per_node.mean())
    delta_tv_full = mean_tv_rewired_full - mean_tv_original_full

    if n_viable >= 2:
        # Within-subset comparison: the SAME K models on both sides (G and
        # G'), so set size cancels exactly and cannot manufacture an
        # apparent effect on its own (unlike comparing a K-viable subset on
        # G' to a full K-member set's diameter on G).
        original_viable_diameter = probability_diameter(
            checkpoint_original_probabilities[viable_mask], method="exact"
        )
        rewired_viable_diameter = probability_diameter(probabilities[viable_mask], method="exact")
        mean_viable_restricted_diameter = float(rewired_viable_diameter.mean())
        delta_diameter_viable = float(
            rewired_viable_diameter.mean() - original_viable_diameter.mean()
        )

        _, tv_original_viable_mean_per_node, _ = pairwise_total_variation_summaries(
            checkpoint_original_probabilities[viable_mask]
        )
        _, tv_rewired_viable_mean_per_node, _ = pairwise_total_variation_summaries(
            probabilities[viable_mask]
        )
        mean_tv_original_viable = float(tv_original_viable_mean_per_node.mean())
        mean_tv_rewired_viable = float(tv_rewired_viable_mean_per_node.mean())
        delta_tv_viable = mean_tv_rewired_viable - mean_tv_original_viable
    else:
        # Diameter/TV over a single (or empty) model is undefined; leave
        # null rather than reporting a fabricated zero.
        mean_viable_restricted_diameter = None
        delta_diameter_viable = None
        mean_tv_original_viable = None
        mean_tv_rewired_viable = None
        delta_tv_viable = None

    output = {
        "dataset": dataset,
        "architecture": str(rashomon["architecture"]),
        "set_id": str(rashomon["set_id"]),
        "retained_count": len(checkpoints),
        "mode": str(metadata["mode"]),
        "rewiring_seed": metadata.get("seed"),
        "graph_path": str(args.graph_path),
        "checkpoint_probability_max_abs_error": checkpoint_probability_max_abs_error,
        "delta": args.delta,
        "best_original_validation_accuracy": best_original_val,
        "n_viable": n_viable,
        "n_total": len(checkpoints),
        # Full-set (K = n_total on both G and G'; matches Table 3).
        "mean_full_set_diameter": mean_full_set_diameter,
        "delta_diameter_full": delta_diameter_full,
        "mean_tv_original_full": mean_tv_original_full,
        "mean_tv_rewired_full": mean_tv_rewired_full,
        "delta_tv_full": delta_tv_full,
        # Within-subset (K = n_viable on both G and G'; set size cancels).
        "mean_viable_restricted_diameter": mean_viable_restricted_diameter,
        "delta_diameter_viable": delta_diameter_viable,
        "mean_tv_original_viable": mean_tv_original_viable,
        "mean_tv_rewired_viable": mean_tv_rewired_viable,
        "delta_tv_viable": delta_tv_viable,
    }
    write_json(Path(args.output), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
