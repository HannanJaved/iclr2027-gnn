from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.fairness.fairsin import heterogeneous_neighbor_targets
from gnn_rashomon.fairness.metrics import (
    demographic_parity_gap,
    equalized_odds_gap,
    fairness_metrics,
)
from gnn_rashomon.rewiring.random_rewire import degree_sequence
from gnn_rashomon.training.reproducibility import seed_everything


@dataclass(frozen=True)
class FairSINConfig:
    dataset: str
    seeds: list[int]
    deltas: list[float]
    epochs: int
    f3_pretrain_epochs: int
    hidden_channels: int
    dropout: float
    learning_rate: float
    weight_decay: float
    f3_weight: float
    adversary_weight: float
    sensitive_attribute: str
    positive_label: int
    degree_quantile: float
    equalized_odds_combine: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a model-centric FairSIN baseline with validation-only selection."
    )
    parser.add_argument("--dataset", default="nba")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0.1, 0.5, 1.0, 2.0])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--f3-pretrain-epochs", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--f3-weight", type=float, default=1.0)
    parser.add_argument("--adversary-weight", type=float, default=1.0)
    parser.add_argument("--sensitive-attribute", default="country")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--degree-quantile", type=float, default=0.25)
    parser.add_argument(
        "--equalized-odds-combine", choices=["max", "sum", "average"], default="max"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _build_model(in_channels: int, config: FairSINConfig) -> Any:
    import torch
    from torch import nn
    from torch_geometric.nn import GCNConv

    class FairSIN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = config.hidden_channels
            self.f3 = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, in_channels),
            )
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.classifier = nn.Linear(hidden, 1)
            self.adversary = nn.Linear(hidden, 1)

        def encode(self, x: Any, edge_index: Any, delta: float) -> Any:
            x = x + delta * self.f3(x)
            x = self.conv1(x, edge_index).relu()
            x = torch.nn.functional.dropout(x, p=config.dropout, training=self.training)
            return self.conv2(x, edge_index).relu()

    return FairSIN()


def _fairness(
    predicted: np.ndarray,
    labels: np.ndarray,
    sensitive: np.ndarray,
    config: FairSINConfig,
) -> tuple[float, float]:
    return (
        demographic_parity_gap(predicted, sensitive, positive_label=config.positive_label),
        equalized_odds_gap(
            predicted,
            labels,
            sensitive,
            positive_label=config.positive_label,
            combine=config.equalized_odds_combine,
        ),
    )


def _evaluate(
    model: Any,
    data: Any,
    delta: float,
    labels: np.ndarray,
    sensitive: np.ndarray,
    label_mask: np.ndarray,
    degree: np.ndarray,
    config: FairSINConfig,
) -> dict[str, Any]:
    import torch

    model.eval()
    with torch.no_grad():
        hidden = model.encode(data.x, data.edge_index, delta)
        positive = torch.sigmoid(model.classifier(hidden).squeeze(-1)).cpu().numpy()
    predicted = (positive >= 0.5).astype(np.int64)
    masks = {
        "train": _to_numpy(data.train_mask).astype(bool) & label_mask,
        "validation": _to_numpy(data.val_mask).astype(bool) & label_mask,
        "test": _to_numpy(data.test_mask).astype(bool) & label_mask,
    }
    accuracies = {
        name: float(np.mean(predicted[mask] == labels[mask])) for name, mask in masks.items()
    }
    val_dp, val_eo = _fairness(
        predicted[masks["validation"]],
        labels[masks["validation"]],
        sensitive[masks["validation"]],
        config,
    )
    validation_fairness = fairness_metrics(
        predicted,
        labels,
        sensitive,
        degree,
        masks["validation"],
        positive_label=config.positive_label,
        equalized_odds_combine=config.equalized_odds_combine,
        degree_quantile=config.degree_quantile,
    )
    test_fairness = fairness_metrics(
        predicted,
        labels,
        sensitive,
        degree,
        masks["test"],
        positive_label=config.positive_label,
        equalized_odds_combine=config.equalized_odds_combine,
        degree_quantile=config.degree_quantile,
    )
    return {
        "positive_probabilities": positive,
        "train_accuracy": accuracies["train"],
        "validation_accuracy": accuracies["validation"],
        "test_accuracy_report_only": accuracies["test"],
        "validation_demographic_parity_gap": val_dp,
        "validation_equalized_odds_gap": val_eo,
        "validation_fairness_sum": float(val_dp + val_eo),
        "validation_degree_disparity": validation_fairness["degree_disparity"],
        **{f"test_{key}_report_only": value for key, value in test_fairness.items()},
    }


def _train_candidate(
    graph: Any,
    config: FairSINConfig,
    seed: int,
    delta: float,
    device_name: str,
) -> tuple[Any, dict[str, Any], int]:
    import torch

    seed_everything(seed)
    data = graph.data.to(torch.device(device_name))
    sensitive_value = graph.sensitive_attributes.get(config.sensitive_attribute)
    if sensitive_value is None and hasattr(data, "sensitive_attr"):
        sensitive_value = data.sensitive_attr
    if sensitive_value is None:
        raise ValueError(f"No sensitive attribute named {config.sensitive_attribute!r}.")
    sensitive_tensor = torch.as_tensor(
        _to_numpy(sensitive_value), dtype=torch.long, device=data.x.device
    )
    labels = _to_numpy(data.y).astype(np.int64)
    sensitive = _to_numpy(sensitive_tensor).astype(np.int64)
    label_mask = _to_numpy(getattr(data, "label_mask", np.ones_like(labels, dtype=bool))).astype(
        bool
    ) & (labels >= 0)
    degree = degree_sequence(_to_numpy(data.edge_index).astype(int), int(data.num_nodes))
    train_mask = data.train_mask & torch.as_tensor(label_mask, device=data.x.device)
    targets, has_target = heterogeneous_neighbor_targets(data.x, data.edge_index, sensitive_tensor)
    model = _build_model(int(data.num_node_features), config).to(data.x.device)

    f3_optimizer = torch.optim.Adam(
        model.f3.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in range(config.f3_pretrain_epochs):
        model.f3.train()
        f3_optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model.f3(data.x)[has_target], targets[has_target])
        loss.backward()
        f3_optimizer.step()

    main_parameters = (
        list(model.f3.parameters())
        + list(model.conv1.parameters())
        + list(model.conv2.parameters())
        + list(model.classifier.parameters())
    )
    main_optimizer = torch.optim.Adam(
        main_parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    adversary_optimizer = torch.optim.Adam(
        model.adversary.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    labels_tensor = data.y.float()
    best_state = None
    best_metrics = None
    best_key = None
    best_epoch = -1
    for epoch in range(config.epochs):
        model.train()
        model.adversary.requires_grad_(False)
        main_optimizer.zero_grad()
        hidden = model.encode(data.x, data.edge_index, delta)
        label_logits = model.classifier(hidden).squeeze(-1)
        sensitive_logits = model.adversary(hidden).squeeze(-1)
        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            label_logits[train_mask], labels_tensor[train_mask]
        )
        f3_loss = torch.nn.functional.mse_loss(model.f3(data.x)[has_target], targets[has_target])
        adversary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            sensitive_logits, sensitive_tensor.float()
        )
        loss = (
            classification_loss
            + config.f3_weight * f3_loss
            - config.adversary_weight * adversary_loss
        )
        loss.backward()
        main_optimizer.step()

        model.adversary.requires_grad_(True)
        adversary_optimizer.zero_grad()
        with torch.no_grad():
            detached_hidden = model.encode(data.x, data.edge_index, delta).detach()
        adversary_logits = model.adversary(detached_hidden).squeeze(-1)
        discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            adversary_logits, sensitive_tensor.float()
        )
        discriminator_loss.backward()
        adversary_optimizer.step()

        metrics = _evaluate(model, data, delta, labels, sensitive, label_mask, degree, config)
        key = (
            float(metrics["validation_accuracy"]) - float(metrics["validation_fairness_sum"]),
            float(metrics["validation_accuracy"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    assert best_state is not None and best_metrics is not None
    return best_state, best_metrics, best_epoch


def main() -> None:
    import torch

    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config = FairSINConfig(
        dataset=args.dataset,
        seeds=list(args.seeds),
        deltas=list(args.deltas),
        epochs=args.epochs,
        f3_pretrain_epochs=args.f3_pretrain_epochs,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        f3_weight=args.f3_weight,
        adversary_weight=args.adversary_weight,
        sensitive_attribute=args.sensitive_attribute,
        positive_label=args.positive_label,
        degree_quantile=args.degree_quantile,
        equalized_odds_combine=args.equalized_odds_combine,
    )
    project_config = load_config([f"dataset={config.dataset}"], project_root / "configs")
    graph = load_graph(
        config.dataset,
        root=str(project_root / project_config["dataset_config"]["root"]),
        dataset_config=project_config["dataset_config"],
    )
    output_dir = Path(args.output_dir) / "baselines" / "fairsin"
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in config.seeds:
        candidates = []
        for delta in config.deltas:
            _, metrics, best_epoch = _train_candidate(graph, config, seed, delta, args.device)
            candidates.append((delta, metrics, best_epoch))
        delta, metrics, best_epoch = max(
            candidates,
            key=lambda item: (
                float(item[1]["validation_accuracy"]) - float(item[1]["validation_fairness_sum"]),
                float(item[1]["validation_accuracy"]),
            ),
        )
        run_id = f"{config.dataset}-fairsin-{seed}"
        probabilities = np.column_stack(
            [1.0 - metrics["positive_probabilities"], metrics["positive_probabilities"]]
        )
        probability_path = prediction_dir / f"{run_id}.probabilities.pt"
        torch.save(torch.as_tensor(probabilities, dtype=torch.float32), probability_path)
        rows.append(
            {
                "run_id": run_id,
                "baseline": "fairsin",
                "variant": "model_centric",
                "dataset": config.dataset,
                "seed": seed,
                "selected_delta": delta,
                "best_epoch": best_epoch,
                "selection_uses_test_metrics": False,
                "test_accuracy_is_report_only": True,
                "probability_path": str(probability_path),
                **{key: value for key, value in metrics.items() if key != "positive_probabilities"},
            }
        )
    table = pd.DataFrame(rows)
    table_path = output_dir / f"{config.dataset}_fairsin_baseline_models.csv"
    summary_path = output_dir / f"{config.dataset}_fairsin_baseline_summary.json"
    table.to_csv(table_path, index=False)
    summary = {
        "baseline": "fairsin",
        "variant": "model_centric",
        "config": asdict(config),
        "num_runs": len(table),
        "selection_uses_test_metrics": False,
        "test_accuracy_is_report_only": True,
        "model_table": str(table_path),
        "mean_validation_accuracy": float(table["validation_accuracy"].mean()),
        "mean_test_accuracy_report_only": float(table["test_accuracy_report_only"].mean()),
        "best_validation_demographic_parity_gap": float(
            table["validation_demographic_parity_gap"].min()
        ),
        "best_validation_equalized_odds_gap": float(table["validation_equalized_odds_gap"].min()),
        "best_validation_degree_disparity": float(table["validation_degree_disparity"].min()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote FairSIN baseline table to {table_path}")
    print(f"Wrote FairSIN baseline summary to {summary_path}")


if __name__ == "__main__":
    main()
