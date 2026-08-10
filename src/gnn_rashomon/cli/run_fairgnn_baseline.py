from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.fairness.metrics import (
    demographic_parity_gap,
    equalized_odds_gap,
    fairness_metrics,
)
from gnn_rashomon.rewiring.random_rewire import degree_sequence


@dataclass(frozen=True)
class FairGNNConfig:
    dataset: str
    seeds: list[int]
    epochs: int
    pretrain_epochs: int
    hidden_channels: int
    estimator_hidden_channels: int
    dropout: float
    learning_rate: float
    weight_decay: float
    alpha: float
    beta: float
    sensitive_attribute: str
    sensitive_train_count: int
    positive_label: int
    degree_quantile: float
    equalized_odds_combine: str
    min_validation_accuracy: float
    min_validation_roc_auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyG FairGNN-style adversarial GCN baseline on a fairness dataset."
    )
    parser.add_argument("--dataset", default="nba")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--estimator-hidden-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--sensitive-attribute", default="country")
    parser.add_argument("--sensitive-train-count", type=int, default=50)
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--degree-quantile", type=float, default=0.25)
    parser.add_argument(
        "--equalized-odds-combine",
        choices=["max", "sum", "average"],
        default="max",
    )
    parser.add_argument("--min-validation-accuracy", type=float, default=0.70)
    parser.add_argument("--min-validation-roc-auc", type=float, default=0.76)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("FairGNN baseline training requires torch and torch-geometric.") from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(np.mean(predicted == labels))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except ModuleNotFoundError:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _fairness_score(
    predicted: np.ndarray,
    labels: np.ndarray,
    sensitive: np.ndarray,
    positive_label: int,
    equalized_odds_combine: str,
) -> tuple[float, float, float]:
    parity = demographic_parity_gap(predicted, sensitive, positive_label=positive_label)
    equality = equalized_odds_gap(
        predicted,
        labels,
        sensitive,
        positive_label=positive_label,
        combine=equalized_odds_combine,
    )
    total = float(np.nansum([parity, equality]))
    return parity, equality, total


def _select_sensitive_training_nodes(
    sensitive: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    avoid = train_mask | val_mask | test_mask
    candidates = np.where(~avoid & np.isfinite(sensitive))[0]
    if candidates.size < count:
        candidates = np.where(np.isfinite(sensitive))[0]
    rng = np.random.default_rng(seed)
    if candidates.size == 0:
        raise ValueError("No nodes with usable sensitive attributes are available.")
    selected = rng.choice(candidates, size=min(count, candidates.size), replace=False)
    return np.asarray(sorted(selected), dtype=np.int64)


def _build_model(in_channels: int, config: FairGNNConfig) -> Any:
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import GCNConv
    except ModuleNotFoundError as exc:
        raise RuntimeError("FairGNN baseline training requires torch and torch-geometric.") from exc

    class GCNBody(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = float(dropout)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x, edge_index).relu()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return self.conv2(x, edge_index).relu()

    class SensitiveEstimator(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, 1)
            self.dropout = float(dropout)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x, edge_index).relu()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return self.conv2(x, edge_index).squeeze(-1)

    class FairGNNBaseline(nn.Module):
        def __init__(self, input_dim: int, cfg: FairGNNConfig) -> None:
            super().__init__()
            self.estimator = SensitiveEstimator(
                input_dim, cfg.estimator_hidden_channels, cfg.dropout
            )
            self.encoder = GCNBody(input_dim, cfg.hidden_channels, cfg.dropout)
            self.classifier = nn.Linear(cfg.hidden_channels, 1)
            self.adversary = nn.Linear(cfg.hidden_channels, 1)

        def forward(
            self, x: torch.Tensor, edge_index: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            sensitive_logits = self.estimator(x, edge_index)
            hidden = self.encoder(x, edge_index)
            label_logits = self.classifier(hidden).squeeze(-1)
            return label_logits, sensitive_logits

    return FairGNNBaseline(in_channels, config)


def _pretrain_estimator(
    model: Any,
    data: Any,
    sensitive: Any,
    sensitive_train_idx: Any,
    config: FairGNNConfig,
) -> None:
    import torch

    if config.pretrain_epochs <= 0:
        return
    optimizer = torch.optim.Adam(
        model.estimator.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    for _ in range(config.pretrain_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model.estimator(data.x, data.edge_index)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[sensitive_train_idx],
            sensitive[sensitive_train_idx].float(),
        )
        loss.backward()
        optimizer.step()


def _evaluate(
    model: Any,
    data: Any,
    labels_np: np.ndarray,
    sensitive_np: np.ndarray,
    degree_np: np.ndarray,
    label_mask_np: np.ndarray,
    config: FairGNNConfig,
) -> dict[str, float | np.ndarray]:
    import torch

    model.eval()
    with torch.no_grad():
        logits, _ = model(data.x, data.edge_index)
        probabilities_positive = torch.sigmoid(logits).detach().cpu().numpy()
    predicted = (probabilities_positive >= 0.5).astype(np.int64)

    train_mask = _to_numpy(data.train_mask).astype(bool) & label_mask_np
    val_mask = _to_numpy(data.val_mask).astype(bool) & label_mask_np
    test_mask = _to_numpy(data.test_mask).astype(bool) & label_mask_np
    val_parity, val_equality, val_fairness = _fairness_score(
        predicted[val_mask],
        labels_np[val_mask],
        sensitive_np[val_mask],
        config.positive_label,
        config.equalized_odds_combine,
    )
    validation_fairness = fairness_metrics(
        predicted,
        labels_np,
        sensitive_np,
        degree_np,
        val_mask,
        positive_label=config.positive_label,
        equalized_odds_combine=config.equalized_odds_combine,
        degree_quantile=config.degree_quantile,
    )
    test_fairness = fairness_metrics(
        predicted,
        labels_np,
        sensitive_np,
        degree_np,
        test_mask,
        positive_label=config.positive_label,
        equalized_odds_combine=config.equalized_odds_combine,
        degree_quantile=config.degree_quantile,
    )
    return {
        "predicted": predicted,
        "probabilities_positive": probabilities_positive,
        "train_accuracy": _accuracy(predicted[train_mask], labels_np[train_mask]),
        "validation_accuracy": _accuracy(predicted[val_mask], labels_np[val_mask]),
        "test_accuracy": _accuracy(predicted[test_mask], labels_np[test_mask]),
        "validation_roc_auc": _roc_auc(probabilities_positive[val_mask], labels_np[val_mask]),
        "test_roc_auc": _roc_auc(probabilities_positive[test_mask], labels_np[test_mask]),
        "validation_demographic_parity_gap": val_parity,
        "validation_equalized_odds_gap": val_equality,
        "validation_fairness_sum": val_fairness,
        "validation_degree_disparity": validation_fairness["degree_disparity"],
        **{f"test_{key}_report_only": value for key, value in test_fairness.items()},
    }


def _train_one_seed(
    graph: Any,
    config: FairGNNConfig,
    seed: int,
    device_name: str,
    output_dir: Path,
) -> dict[str, object]:
    import torch

    _seed_everything(seed)
    data = graph.data
    device = torch.device(device_name)
    data = data.to(device)
    sensitive_value = graph.sensitive_attributes.get(config.sensitive_attribute)
    if sensitive_value is None and hasattr(data, "sensitive_attr"):
        sensitive_value = data.sensitive_attr
    if sensitive_value is None:
        raise ValueError(f"No sensitive attribute named {config.sensitive_attribute!r}.")

    labels_np = _to_numpy(data.y).astype(np.int64)
    sensitive_np = _to_numpy(sensitive_value).astype(np.int64)
    label_mask_np = _to_numpy(
        getattr(data, "label_mask", np.ones_like(labels_np, dtype=bool))
    ).astype(bool)
    degree_np = degree_sequence(_to_numpy(data.edge_index).astype(int), int(data.num_nodes))
    train_mask_np = _to_numpy(data.train_mask).astype(bool)
    val_mask_np = _to_numpy(data.val_mask).astype(bool)
    test_mask_np = _to_numpy(data.test_mask).astype(bool)
    sensitive_train_idx_np = _select_sensitive_training_nodes(
        sensitive_np,
        train_mask_np,
        val_mask_np,
        test_mask_np,
        count=config.sensitive_train_count,
        seed=seed,
    )
    sensitive = torch.as_tensor(sensitive_np, dtype=torch.float32, device=device)
    labels = data.y.float()
    train_idx = torch.as_tensor(
        np.where(train_mask_np & label_mask_np & (labels_np >= 0))[0],
        dtype=torch.long,
        device=device,
    )
    sensitive_train_idx = torch.as_tensor(sensitive_train_idx_np, dtype=torch.long, device=device)

    model = _build_model(int(data.num_node_features), config).to(device)
    _pretrain_estimator(model, data, sensitive, sensitive_train_idx, config)

    g_params = (
        list(model.encoder.parameters())
        + list(model.classifier.parameters())
        + list(model.estimator.parameters())
    )
    optimizer_g = torch.optim.Adam(
        g_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_a = torch.optim.Adam(
        model.adversary.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float | np.ndarray] | None = None
    best_key: tuple[bool, float, float, float] | None = None
    for epoch in range(config.epochs):
        model.train()
        model.adversary.requires_grad_(False)
        optimizer_g.zero_grad()
        sensitive_logits = model.estimator(data.x, data.edge_index)
        hidden = model.encoder(data.x, data.edge_index)
        label_logits = model.classifier(hidden).squeeze(-1)
        adversary_logits = model.adversary(hidden).squeeze(-1)

        sensitive_score = torch.sigmoid(sensitive_logits.detach())
        sensitive_score[sensitive_train_idx] = sensitive[sensitive_train_idx]
        label_score = torch.sigmoid(label_logits)
        cov = torch.abs(
            torch.mean(
                (sensitive_score - torch.mean(sensitive_score))
                * (label_score - torch.mean(label_score))
            )
        )
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            label_logits[train_idx],
            labels[train_idx],
        )
        adv_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            adversary_logits,
            sensitive_score,
        )
        g_loss = cls_loss + config.alpha * cov - config.beta * adv_loss
        g_loss.backward()
        optimizer_g.step()

        model.adversary.requires_grad_(True)
        optimizer_a.zero_grad()
        with torch.no_grad():
            hidden_detached = model.encoder(data.x, data.edge_index).detach()
            sensitive_logits_detached = model.estimator(data.x, data.edge_index)
            sensitive_score_detached = torch.sigmoid(sensitive_logits_detached.detach())
            sensitive_score_detached[sensitive_train_idx] = sensitive[sensitive_train_idx]
        adversary_logits = model.adversary(hidden_detached).squeeze(-1)
        a_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            adversary_logits,
            sensitive_score_detached,
        )
        a_loss.backward()
        optimizer_a.step()

        metrics = _evaluate(model, data, labels_np, sensitive_np, degree_np, label_mask_np, config)
        passes_thresholds = bool(
            metrics["validation_accuracy"] >= config.min_validation_accuracy
            and (
                np.isnan(metrics["validation_roc_auc"])
                or metrics["validation_roc_auc"] >= config.min_validation_roc_auc
            )
        )
        key = (
            passes_thresholds,
            -float(metrics["validation_fairness_sum"]),
            float(metrics["validation_accuracy"]),
            float(metrics["validation_roc_auc"])
            if np.isfinite(metrics["validation_roc_auc"])
            else -1.0,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    assert best_metrics is not None
    run_id = f"{config.dataset}-fairgnn-{seed}"
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.column_stack(
        [
            1.0 - np.asarray(best_metrics["probabilities_positive"], dtype=float),
            np.asarray(best_metrics["probabilities_positive"], dtype=float),
        ]
    )
    probability_path = prediction_dir / f"{run_id}.probabilities.pt"
    torch.save(torch.as_tensor(probabilities, dtype=torch.float32), probability_path)

    return {
        "run_id": run_id,
        "baseline": "fairgnn",
        "dataset": config.dataset,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "selection_uses_test_metrics": False,
        "test_accuracy_is_report_only": True,
        "probability_path": str(probability_path),
        "train_accuracy": best_metrics["train_accuracy"],
        "validation_accuracy": best_metrics["validation_accuracy"],
        "validation_roc_auc": best_metrics["validation_roc_auc"],
        "test_accuracy_report_only": best_metrics["test_accuracy"],
        "test_roc_auc": best_metrics["test_roc_auc"],
        "validation_demographic_parity_gap": best_metrics["validation_demographic_parity_gap"],
        "validation_equalized_odds_gap": best_metrics["validation_equalized_odds_gap"],
        "validation_degree_disparity": best_metrics["validation_degree_disparity"],
        "test_demographic_parity_gap_report_only": best_metrics[
            "test_demographic_parity_gap_report_only"
        ],
        "test_equalized_odds_gap_report_only": best_metrics["test_equalized_odds_gap_report_only"],
        "test_degree_disparity_report_only": best_metrics["test_degree_disparity_report_only"],
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config_data = FairGNNConfig(
        dataset=str(args.dataset),
        seeds=list(args.seeds),
        epochs=int(args.epochs),
        pretrain_epochs=int(args.pretrain_epochs),
        hidden_channels=int(args.hidden_channels),
        estimator_hidden_channels=int(args.estimator_hidden_channels),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        alpha=float(args.alpha),
        beta=float(args.beta),
        sensitive_attribute=str(args.sensitive_attribute),
        sensitive_train_count=int(args.sensitive_train_count),
        positive_label=int(args.positive_label),
        degree_quantile=float(args.degree_quantile),
        equalized_odds_combine=str(args.equalized_odds_combine),
        min_validation_accuracy=float(args.min_validation_accuracy),
        min_validation_roc_auc=float(args.min_validation_roc_auc),
    )
    project_config = load_config([f"dataset={config_data.dataset}"], project_root / "configs")
    graph = load_graph(
        config_data.dataset,
        root=str(project_root / project_config["dataset_config"]["root"]),
        dataset_config=project_config["dataset_config"],
    )

    output_dir = Path(args.output_dir) / "baselines" / "fairgnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _train_one_seed(graph, config_data, seed, args.device, output_dir)
        for seed in config_data.seeds
    ]
    table = pd.DataFrame(rows)
    table_path = output_dir / f"{config_data.dataset}_fairgnn_baseline_models.csv"
    summary_path = output_dir / f"{config_data.dataset}_fairgnn_baseline_summary.json"
    table.to_csv(table_path, index=False)
    summary = {
        "baseline": "fairgnn",
        "config": asdict(config_data),
        "num_runs": int(len(table)),
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
    print(f"Wrote FairGNN baseline table to {table_path}")
    print(f"Wrote FairGNN baseline summary to {summary_path}")


if __name__ == "__main__":
    main()
