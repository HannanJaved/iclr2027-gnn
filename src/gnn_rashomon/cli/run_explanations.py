from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.explainability.comparison import (
    compare_edge_explanations,
    edge_score_vector,
    instability_regression,
    save_top_edges,
    select_nodes_by_multiplicity,
    summarize_instability,
)
from gnn_rashomon.rashomon.io import read_json
from gnn_rashomon.training.checkpointing import load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded GNN explanation-instability analysis.")
    parser.add_argument("--rashomon-set", required=True)
    parser.add_argument("--node-table", required=True, help="Structure or multiplicity node table.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--max-models", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--explainer-epochs", type=int, default=100)
    parser.add_argument(
        "--pgexplainer-lr",
        type=float,
        default=0.003,
        help="Learning rate for PGExplainer. Ignored by other explainers.",
    )
    parser.add_argument(
        "--explainer",
        choices=["gnnexplainer", "pgexplainer", "gat_attention"],
        default="gnnexplainer",
        help="Explanation method. gat_attention is intended for GAT checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _read_run_record(record_path: str | Path) -> dict[str, object]:
    return json.loads(Path(record_path).read_text(encoding="utf-8"))


def _out(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def _select_runs(rashomon: dict[str, object], max_models: int) -> list[str]:
    retained = list(rashomon["retained_run_ids"])
    runs = rashomon["runs"]
    return sorted(retained, key=lambda run_id: float(runs[run_id]["train_loss"]))[:max_models]


def _run_gnnexplainer(
    model: object,
    x: object,
    edge_index: object,
    node_id: int,
    epochs: int,
) -> np.ndarray:
    from torch_geometric.explain import Explainer, GNNExplainer

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )
    explanation = explainer(x, edge_index, index=int(node_id))
    return explanation.edge_mask.detach().cpu().numpy()


def _make_pgexplainer(
    model: object,
    epochs: int,
    lr: float,
) -> object:
    from torch_geometric.explain import Explainer, PGExplainer

    return Explainer(
        model=model,
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )


def _train_pgexplainer(
    explainer: object,
    model: object,
    x: object,
    edge_index: object,
    target: object,
    train_indices: object,
    epochs: int,
) -> None:
    node_ids = [int(node_id) for node_id in train_indices.detach().cpu().tolist()]
    for epoch in range(epochs):
        losses = []
        for node_id in node_ids:
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=x,
                edge_index=edge_index,
                target=target,
                index=node_id,
            )
            losses.append(float(loss))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"PGExplainer epoch {epoch + 1}/{epochs} mean_loss={mean_loss:.6f}", flush=True)


def _run_pgexplainer(
    explainer: object,
    x: object,
    edge_index: object,
    target: object,
    node_id: int,
) -> np.ndarray:
    explanation = explainer(x, edge_index, target=target, index=int(node_id))
    return explanation.edge_mask.detach().cpu().numpy()


def _run_gat_attention(
    model: object,
    x: object,
    edge_index: object,
    node_id: int,
) -> pd.DataFrame:
    import torch

    if not hasattr(model, "conv1") or not hasattr(model, "conv2"):
        raise ValueError("gat_attention explainer requires a GAT-style model with conv1 and conv2 layers.")
    with torch.no_grad():
        hidden = torch.nn.functional.elu(model.conv1(x, edge_index))
        _, attention = model.conv2(hidden, edge_index, return_attention_weights=True)
    attention_edge_index, alpha = attention
    scores = alpha.detach()
    if scores.ndim > 1:
        scores = scores.mean(dim=-1)
    edge_index_np = attention_edge_index.detach().cpu().numpy().astype(int)
    scores_np = scores.cpu().numpy().astype(float)

    rows: list[dict[str, object]] = []
    for pos, (src, dst) in enumerate(edge_index_np.T):
        if int(dst) != int(node_id):
            continue
        a, b = sorted((int(src), int(dst)))
        rows.append({"edge_key": f"{a}:{b}", "edge_score": float(scores_np[pos])})
    if not rows:
        return pd.DataFrame(columns=["edge_key", "edge_score"])
    return pd.DataFrame(rows).groupby("edge_key", as_index=False)["edge_score"].max()


def _write_plots(summary: pd.DataFrame, output_prefix: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping explanation plots because matplotlib is unavailable: {exc}", flush=True)
        return

    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = ["low", "medium", "high"]
    data = [
        summary.loc[summary["multiplicity_group"] == group, "top_k_jaccard_mean"].dropna().to_numpy(dtype=float)
        for group in groups
    ]
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, showfliers=False)
    colors = ["#4C78A8", "#54A24B", "#F58518"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_xlabel("Multiplicity group")
    ax.set_ylabel("Mean top-k explanation Jaccard")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(_out(output_prefix, ".jaccard_by_group.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        summary["rashomon_capacity"].to_numpy(dtype=float),
        summary["top_k_jaccard_mean"].to_numpy(dtype=float),
        s=28,
        alpha=0.75,
    )
    ax.set_xlabel("Rashomon capacity")
    ax.set_ylabel("Mean top-k explanation Jaccard")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_out(output_prefix, ".capacity_vs_jaccard.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    rashomon_path = Path(args.rashomon_set)
    rashomon = read_json(rashomon_path)
    dataset = args.dataset or str(rashomon["dataset"])
    output_prefix = Path(args.output_prefix or f"outputs/explanations/{rashomon['set_id']}.{args.explainer}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    config = load_config([f"dataset={dataset}"], project_root / "configs")
    graph = load_graph(
        dataset,
        root=str(project_root / config["dataset_config"]["root"]),
        graph_path=args.graph_path,
        dataset_config=config["dataset_config"],
    )
    data = graph.data

    import torch

    device = torch.device(args.device)
    data = data.to(device)
    edge_index_np = _to_numpy(data.edge_index).astype(int)

    node_table = pd.read_csv(args.node_table)
    selected_nodes = select_nodes_by_multiplicity(
        node_table=node_table,
        max_nodes=args.max_nodes,
        metric="rashomon_capacity",
        seed=args.seed,
    )
    selected_path = _out(output_prefix, ".selected_nodes.csv")
    selected_nodes.to_csv(selected_path, index=False)

    run_ids = _select_runs(rashomon, args.max_models)
    run_records = {
        run_id: _read_run_record(rashomon["runs"][run_id]["record_path"])
        for run_id in run_ids
    }
    run_table = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "seed": run_records[run_id]["seed"],
                "train_loss": run_records[run_id]["train_loss"],
                "validation_accuracy": run_records[run_id]["validation_accuracy"],
                "test_accuracy": run_records[run_id]["test_accuracy"],
                "checkpoint_path": run_records[run_id]["checkpoint_path"],
            }
            for run_id in run_ids
        ]
    )
    selected_models_path = _out(output_prefix, ".selected_models.csv")
    run_table.to_csv(selected_models_path, index=False)

    explanations: dict[tuple[int, str], pd.DataFrame] = {}
    for run_pos, run_id in enumerate(run_ids, start=1):
        record = run_records[run_id]
        print(f"Loading model {run_pos}/{len(run_ids)} {run_id}", flush=True)
        model = load_checkpoint_model(
            record["checkpoint_path"],
            in_channels=int(data.num_node_features),
            out_channels=int(graph.metadata.num_classes),
            map_location=args.device,
        ).to(device)
        model.eval()
        pgexplainer = None
        pg_target = None
        if args.explainer == "pgexplainer":
            with torch.no_grad():
                pg_target = model(data.x, data.edge_index).argmax(dim=-1)
            train_index = torch.as_tensor(
                selected_nodes["node_id"].to_numpy(dtype=int),
                dtype=torch.long,
                device=device,
            )
            pgexplainer = _make_pgexplainer(
                model=model,
                epochs=args.explainer_epochs,
                lr=args.pgexplainer_lr,
            )
            _train_pgexplainer(
                explainer=pgexplainer,
                model=model,
                x=data.x,
                edge_index=data.edge_index,
                target=pg_target,
                train_indices=train_index,
                epochs=args.explainer_epochs,
            )
        for node_pos, node in enumerate(selected_nodes.itertuples(index=False), start=1):
            node_id = int(node.node_id)
            print(
                f"Explaining node {node_pos}/{len(selected_nodes)} "
                f"for model {run_pos}/{len(run_ids)}",
                flush=True,
            )
            if args.explainer == "gnnexplainer":
                edge_mask = _run_gnnexplainer(
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    node_id=node_id,
                    epochs=args.explainer_epochs,
                )
                explanations[(node_id, run_id)] = edge_score_vector(edge_index_np, edge_mask)
            elif args.explainer == "pgexplainer":
                edge_mask = _run_pgexplainer(
                    explainer=pgexplainer,
                    x=data.x,
                    edge_index=data.edge_index,
                    target=pg_target,
                    node_id=node_id,
                )
                explanations[(node_id, run_id)] = edge_score_vector(edge_index_np, edge_mask)
            else:
                explanations[(node_id, run_id)] = _run_gat_attention(
                    model=model,
                    x=data.x,
                    edge_index=data.edge_index,
                    node_id=node_id,
                )

    top_edges_path = _out(output_prefix, ".top_edges.csv")
    save_top_edges(
        explanations=explanations,
        run_metadata=run_records,
        selected_nodes=selected_nodes,
        output_path=top_edges_path,
        top_k=args.top_k,
    )
    pairwise = compare_edge_explanations(explanations, top_k=args.top_k)
    pairwise_path = _out(output_prefix, ".pairwise.csv")
    pairwise.to_csv(pairwise_path, index=False)
    summary = summarize_instability(pairwise, selected_nodes)
    summary_path = _out(output_prefix, ".node_summary.csv")
    summary.to_csv(summary_path, index=False)
    regression = instability_regression(summary)
    regression_path = _out(output_prefix, ".regression.csv")
    regression.to_csv(regression_path, index=False)
    _write_plots(summary, output_prefix)

    metadata = {
        "rashomon_set": str(rashomon_path),
        "dataset": dataset,
        "explainer": args.explainer,
        "explainer_epochs": args.explainer_epochs,
        "pgexplainer_lr": args.pgexplainer_lr,
        "max_nodes": args.max_nodes,
        "max_models": args.max_models,
        "selected_node_count": int(len(selected_nodes)),
        "selected_model_count": int(len(run_ids)),
        "top_k": args.top_k,
        "outputs": {
            "selected_nodes": str(selected_path),
            "selected_models": str(selected_models_path),
            "top_edges": str(top_edges_path),
            "pairwise": str(pairwise_path),
            "node_summary": str(summary_path),
            "regression": str(regression_path),
        },
    }
    metadata_path = _out(output_prefix, ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote selected nodes to {selected_path}", flush=True)
    print(f"Wrote selected models to {selected_models_path}", flush=True)
    print(f"Wrote top explanation edges to {top_edges_path}", flush=True)
    print(f"Wrote pairwise explanation metrics to {pairwise_path}", flush=True)
    print(f"Wrote node-level explanation instability summary to {summary_path}", flush=True)
    print(f"Wrote explanation instability regression to {regression_path}", flush=True)
    print(f"Wrote explanation analysis metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
