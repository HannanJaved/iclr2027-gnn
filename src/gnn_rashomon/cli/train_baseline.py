from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.training.trainer import train_model


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    project_root = Path(__file__).resolve().parents[3]
    config_dir = project_root / "configs"
    _log(f"Loading config from {config_dir}")
    config = load_config(argv, config_dir)
    output_dir = project_root / str(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output_dir = output_dir / "configs"
    config_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = str(config["dataset"])
    config_id = f"{dataset}-seed{int(config['seed'])}-{uuid.uuid4().hex[:8]}"
    resolved_config_path = config_output_dir / f"{config_id}.json"
    resolved_config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"Wrote resolved config to {resolved_config_path}")

    _log(f"Loading dataset={dataset}")
    graph_path = config.get("graph_path")
    graph = load_graph(
        dataset,
        root=str(project_root / config["dataset_config"]["root"]),
        graph_path=str(graph_path) if graph_path else None,
        dataset_config=config["dataset_config"],
    )
    _log(
        "Loaded graph "
        f"nodes={graph.metadata.num_nodes} edges={graph.metadata.num_edges} "
        f"features={graph.metadata.num_features} classes={graph.metadata.num_classes}"
    )
    _log(
        "Starting training "
        f"architecture={config['model_config']['architecture']} seed={config['seed']} "
        f"device={config['trainer']['device']} max_epochs={config['trainer']['max_epochs']}"
    )
    record = train_model(
        graph=graph,
        model_config=config["model_config"],
        training_config=config["trainer"],
        seed=int(config["seed"]),
        dataset=dataset,
        output_dir=output_dir,
        config_path=str(resolved_config_path),
        rashomon_type=str(config.get("rashomon_type", "baseline")),
    )
    _log(f"Finished training run_id={record.run_id} test_accuracy={record.test_accuracy:.4f}")
    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
