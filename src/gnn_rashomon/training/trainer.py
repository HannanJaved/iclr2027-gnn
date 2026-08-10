from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from gnn_rashomon.data.contracts import ModelRun
from gnn_rashomon.training.evaluation import masked_loss_and_accuracy
from gnn_rashomon.training.reproducibility import seed_everything


def build_model(
    model_config: dict[str, Any],
    in_channels: int,
    out_channels: int,
) -> Any:
    architecture = str(model_config["architecture"])
    kwargs = {
        "in_channels": in_channels,
        "hidden_channels": int(model_config["hidden_channels"]),
        "out_channels": out_channels,
        "dropout": float(model_config["dropout"]),
    }
    if architecture == "gcn":
        from gnn_rashomon.models.gcn import GCN

        return GCN(**kwargs)
    if architecture == "gat":
        from gnn_rashomon.models.gat import GAT

        return GAT(
            **kwargs,
            heads=int(model_config.get("heads", 4)),
        )
    if architecture == "graphsage":
        from gnn_rashomon.models.graphsage import GraphSAGE

        return GraphSAGE(**kwargs)
    if architecture == "appnp":
        from gnn_rashomon.models.appnp import APPNP

        return APPNP(
            **kwargs,
            k=int(model_config.get("k", 10)),
            alpha=float(model_config.get("alpha", 0.1)),
        )
    if architecture == "mlp":
        from gnn_rashomon.models.mlp import MLP

        return MLP(**kwargs)
    if architecture == "topology_only":
        from gnn_rashomon.models.topology_only import TopologyOnlyGCN

        representation = str(model_config.get("input_representation", "constant"))
        if representation != "constant":
            raise ValueError(
                "Only topology_only input_representation='constant' is implemented."
            )
        return TopologyOnlyGCN(**kwargs)
    raise ValueError(f"Unsupported architecture: {architecture}")


def train_model(
    graph: Any,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    dataset: str,
    output_dir: str | Path,
    config_path: str,
    rashomon_type: str = "baseline",
) -> ModelRun:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Training requires torch and torch-geometric.") from exc

    seed_everything(seed)
    data = graph.data
    device = torch.device(training_config.get("device", "cpu"))
    data = data.to(device)

    model = build_model(
        model_config=model_config,
        in_channels=int(data.num_node_features),
        out_channels=int(graph.metadata.num_classes),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config.get("early_stopping_patience", max_epochs))
    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    best_epoch = -1
    last_epoch = -1

    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_loss, _ = masked_loss_and_accuracy(logits, data.y, data.val_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = torch.softmax(logits, dim=-1)
        train_loss, train_acc = masked_loss_and_accuracy(logits, data.y, data.train_mask)
        val_loss, val_acc = masked_loss_and_accuracy(logits, data.y, data.val_mask)
        test_loss, test_acc = masked_loss_and_accuracy(logits, data.y, data.test_mask)

    run_id = f"{dataset}-{model_config['architecture']}-{seed}-{uuid.uuid4().hex[:8]}"
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"{run_id}.pt"
    logits_path = prediction_dir / f"{run_id}.logits.pt"
    probabilities_path = prediction_dir / f"{run_id}.probabilities.pt"
    record_path = metrics_dir / f"{run_id}.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": training_config,
            "metadata": graph.metadata,
        },
        checkpoint_path,
    )
    torch.save(logits.detach().cpu(), logits_path)
    torch.save(probabilities.detach().cpu(), probabilities_path)

    record = ModelRun(
        run_id=run_id,
        dataset=dataset,
        architecture=str(model_config["architecture"]),
        rashomon_type=rashomon_type,
        seed=seed,
        hyperparameters={**model_config, **training_config},
        train_loss=train_loss,
        validation_loss=val_loss,
        test_loss=test_loss,
        train_accuracy=train_acc,
        validation_accuracy=val_acc,
        test_accuracy=test_acc,
        checkpoint_path=str(checkpoint_path),
        prediction_path=str(logits_path),
        probability_path=str(probabilities_path),
        config_path=config_path,
        best_epoch=best_epoch,
        epochs_trained=last_epoch + 1,
        stopped_early=(last_epoch + 1) < max_epochs,
    )
    record_path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return record
