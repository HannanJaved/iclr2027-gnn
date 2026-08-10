from __future__ import annotations

"""Re-evaluate saved fixed-ensemble checkpoints on saved counterfactual graphs.

No retraining. Adds validation/test accuracy on G and G' to realization sidecars
and writes a paper-ready summary table for Appendix accuracy-under-rewiring.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.cli.evaluate_fixed_set_rewiring import (
    _accuracy_summary,
    _archived_probabilities,
    _checkpoint_paths,
    _fixed_set_probabilities,
    _masked_accuracy,
)
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.rashomon.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realizations-csv",
        action="append",
        default=[],
        help="One or more realizations CSVs (defaults cover GCN + PubMed GAT).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rewiring_accuracy",
    )
    parser.add_argument(
        "--strengths",
        default="0.25",
        help="Comma-separated random-rewiring strengths to evaluate on G'.",
    )
    parser.add_argument(
        "--modes",
        default="random",
        help="Comma-separated rewiring modes (default: random).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-realizations-per-group",
        type=int,
        default=None,
        help="Optional cap for smoke tests.",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip a realization when its accuracy sidecar already exists.",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _rashomon_path_for_set(set_id: str, root: Path) -> Path:
    candidates = [
        root / "outputs" / "rashomon_sets" / f"{set_id}.json",
        root
        / "outputs"
        / "canonical_pubmed_gat"
        / "performance_sensitivity"
        / "rashomon_sets"
        / f"{set_id}.json",
        root
        / "outputs"
        / "canonical_pubmed_gat"
        / "rashomon_sets"
        / f"{set_id}.json",
        root
        / "outputs"
        / "rashomon_membership_sensitivity"
        / "rashomon_sets"
        / f"{set_id}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Strip valaccdelta suffix and try the parent set.
    if "-valaccdelta" in set_id:
        parent = set_id.split("-valaccdelta")[0]
        return _rashomon_path_for_set(parent, root)
    raise FileNotFoundError(f"No rashomon set JSON found for set_id={set_id}")


def main() -> None:
    args = parse_args()
    root = _project_root()
    output_dir = _resolve(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = output_dir / "sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    default_csvs = [
        "outputs/repeated_rewiring/repeated_rewiring_realizations.csv",
        "outputs/non_gcn_rewiring/pubmed_gat/pubmed_gat_repeated_rewiring_realizations.csv",
    ]
    csv_paths = args.realizations_csv or default_csvs
    frames = [pd.read_csv(_resolve(path, root)) for path in csv_paths]
    frame = pd.concat(frames, ignore_index=True)

    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    strengths = {float(s.strip()) for s in args.strengths.split(",") if s.strip()}
    subset = frame[frame["mode"].isin(modes) & frame["strength"].isin(strengths)].copy()
    if args.max_realizations_per_group is not None:
        subset = (
            subset.groupby(["dataset", "architecture", "mode", "strength"], dropna=False)
            .head(args.max_realizations_per_group)
            .reset_index(drop=True)
        )

    cache: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for _, row in subset.iterrows():
        set_id = str(row["set_id"])
        graph_path = _resolve(str(row["graph_path"]), root)
        seed = int(row["rewiring_seed"]) if pd.notna(row.get("rewiring_seed")) else -1
        target = row.get("target_homophily")
        tau = row.get("tau")
        target_tag = (
            f"-t{str(target).replace('.', 'p')}"
            if pd.notna(target)
            else (f"-tau{str(tau).replace('.', 'p')}" if pd.notna(tau) else "")
        )
        sidecar = (
            sidecar_dir
            / f"{row['dataset']}-{row['architecture']}-{row['mode']}-"
            f"s{str(row['strength']).replace('.', 'p')}{target_tag}-seed{seed}.json"
        )
        if args.skip_if_done and sidecar.exists():
            payload = read_json(sidecar)
            rows.append(payload)
            continue

        if set_id not in cache:
            rashomon_path = _rashomon_path_for_set(set_id, root)
            rashomon = read_json(rashomon_path)
            dataset = str(rashomon["dataset"])
            config = load_config([f"dataset={dataset}"], root / "configs")
            original_graph = load_graph(
                dataset,
                root=str(root / config["dataset_config"]["root"]),
                dataset_config=config["dataset_config"],
            )
            checkpoints = _checkpoint_paths(rashomon, root)
            archived = _archived_probabilities(rashomon, root)
            original_probs = _fixed_set_probabilities(
                checkpoints,
                x=original_graph.data.x,
                edge_index=original_graph.data.edge_index,
                out_channels=int(original_graph.metadata.num_classes),
                device=args.device,
            )
            max_abs = float(np.max(np.abs(original_probs - archived)))
            if max_abs > 1e-5:
                raise ValueError(
                    f"Checkpoint/archive mismatch for {set_id}: max_abs={max_abs:.6g}"
                )
            labels = original_graph.data.y.detach().cpu().numpy()
            val_mask = original_graph.data.val_mask.detach().cpu().numpy().astype(bool)
            test_mask = original_graph.data.test_mask.detach().cpu().numpy().astype(bool)
            original_acc = _accuracy_summary(
                original_probs, labels, val_mask, test_mask, prefix="original"
            )
            original_val = _masked_accuracy(original_probs, labels, val_mask)
            cache[set_id] = {
                "rashomon": rashomon,
                "rashomon_path": rashomon_path,
                "dataset": dataset,
                "config": config,
                "original_graph": original_graph,
                "checkpoints": checkpoints,
                "labels": labels,
                "val_mask": val_mask,
                "test_mask": test_mask,
                "original_acc": original_acc,
                "best_original_val": float(original_val.max()),
                "original_val": original_val,
            }

        cached = cache[set_id]
        original_graph = cached["original_graph"]
        config = cached["config"]
        dataset = str(cached["dataset"])
        graph = load_graph(
            dataset,
            root=str(root / config["dataset_config"]["root"]),
            graph_path=str(graph_path),
            dataset_config=config["dataset_config"],
        )
        rewired_probs = _fixed_set_probabilities(
            cached["checkpoints"],  # type: ignore[arg-type]
            x=graph.data.x,
            edge_index=graph.data.edge_index,
            out_channels=int(original_graph.metadata.num_classes),
            device=args.device,
        )
        labels = cached["labels"]  # type: ignore[assignment]
        val_mask = cached["val_mask"]  # type: ignore[assignment]
        test_mask = cached["test_mask"]  # type: ignore[assignment]
        rewired_acc = _accuracy_summary(
            rewired_probs, labels, val_mask, test_mask, prefix="rewired"
        )
        rewired_val = _masked_accuracy(rewired_probs, labels, val_mask)
        best_original_val = float(cached["best_original_val"])  # type: ignore[arg-type]
        n_within = int((rewired_val >= best_original_val - 0.02 - 1e-12).sum())

        payload = {
            "dataset": str(row["dataset"]),
            "architecture": str(row["architecture"]),
            "set_id": set_id,
            "mode": str(row["mode"]),
            "strength": float(row["strength"]),
            "target_homophily": (
                float(row["target_homophily"])
                if "target_homophily" in row and pd.notna(row["target_homophily"])
                else None
            ),
            "tau": float(row["tau"]) if "tau" in row and pd.notna(row["tau"]) else None,
            "rewiring_seed": seed,
            "graph_path": str(graph_path),
            "rashomon_set_path": str(cached["rashomon_path"]),
            "retained_count": int(row["retained_count"]),
            **cached["original_acc"],  # type: ignore[arg-type]
            **rewired_acc,
            "best_original_validation_accuracy": best_original_val,
            "n_models_within_original_val_delta_0p02_on_rewired": n_within,
            "delta_mean_validation_accuracy": float(
                rewired_acc["rewired_validation_accuracy_mean"]
                - cached["original_acc"]["original_validation_accuracy_mean"]  # type: ignore[index]
            ),
        }
        write_json(sidecar, payload)
        rows.append(payload)
        print(
            f"{payload['dataset']} {payload['architecture']} "
            f"{payload['mode']} s={payload['strength']} seed={seed}: "
            f"val {payload['original_validation_accuracy_mean']:.4f} -> "
            f"{payload['rewired_validation_accuracy_mean']:.4f} "
            f"(K_delta={n_within}/{payload['retained_count']})"
        )

    detail = pd.DataFrame(rows)
    # Merge with any previously written sidecars so repeated runs do not drop
    # earlier conditions from the summary table.
    existing = [read_json(path) for path in sidecar_dir.glob("*.json")]
    if existing:
        detail = pd.concat([pd.DataFrame(existing), detail], ignore_index=True)
    for col in ("target_homophily", "tau"):
        if col not in detail.columns:
            detail[col] = None
    subset_cols = [
        "dataset",
        "architecture",
        "mode",
        "strength",
        "target_homophily",
        "tau",
        "rewiring_seed",
    ]
    detail = detail.drop_duplicates(subset=subset_cols, keep="last")
    detail_path = output_dir / "rewiring_accuracy_detail.csv"
    detail.to_csv(detail_path, index=False)

    # Baseline (strength 0) once per set, from original_* fields on sidecars.
    baseline_rows = []
    if not detail.empty:
        for (dataset, architecture, set_id), group in detail.groupby(
            ["dataset", "architecture", "set_id"], dropna=False
        ):
            baseline_rows.append(
                {
                    "dataset": dataset,
                    "architecture": architecture,
                    "set_id": set_id,
                    "mode": "identity",
                    "strength": 0.0,
                    "target_homophily": None,
                    "tau": None,
                    "mean_val_acc": float(group["original_validation_accuracy_mean"].iloc[0]),
                    "min_val_acc": float(group["original_validation_accuracy_min"].iloc[0]),
                    "mean_k_delta": float(
                        group["original_n_models_within_val_delta_0p02"].iloc[0]
                        if "original_n_models_within_val_delta_0p02" in group.columns
                        else group["retained_count"].iloc[0]
                    ),
                    "n_realizations": 1,
                }
            )

    summary_rows = list(baseline_rows)
    if not detail.empty:
        grouped = detail.groupby(
            [
                "dataset",
                "architecture",
                "set_id",
                "mode",
                "strength",
                "target_homophily",
                "tau",
            ],
            dropna=False,
        )
        for keys, group in grouped:
            summary_rows.append(
                {
                    "dataset": keys[0],
                    "architecture": keys[1],
                    "set_id": keys[2],
                    "mode": keys[3],
                    "strength": keys[4],
                    "target_homophily": keys[5],
                    "tau": keys[6],
                    "mean_val_acc": float(group["rewired_validation_accuracy_mean"].mean()),
                    "min_val_acc": float(group["rewired_validation_accuracy_min"].mean()),
                    "mean_k_delta": float(
                        group["n_models_within_original_val_delta_0p02_on_rewired"].mean()
                    ),
                    "n_realizations": int(len(group)),
                    "mean_delta_val_acc": float(
                        group["delta_mean_validation_accuracy"].mean()
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset", "architecture", "mode", "strength"]
    )
    summary_path = output_dir / "rewiring_accuracy_summary.csv"
    summary.to_csv(summary_path, index=False)
    write_json(
        output_dir / "rewiring_accuracy_summary.json",
        {
            "detail_csv": str(detail_path.relative_to(root)),
            "summary_csv": str(summary_path.relative_to(root)),
            "n_detail_rows": int(len(detail)),
            "n_summary_rows": int(len(summary)),
        },
    )
    print(json.dumps({"detail": str(detail_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
