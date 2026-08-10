from __future__ import annotations

import argparse
import json
import time
import uuid
from itertools import product
from pathlib import Path

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.training.trainer import train_model


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiple runs in one Python process.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="gcn")
    parser.add_argument("--sweep", choices=["seed", "hyperparameter"], required=True)
    parser.add_argument(
        "--rashomon-type",
        default=None,
        help="Run metadata rashomon_type. Defaults to the sweep name.",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.02, 0.01, 0.005])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0005, 0.0001])
    parser.add_argument("--hidden-channels", nargs="+", type=int, default=[16, 32])
    return parser.parse_args()


def _write_config(output_dir: Path, dataset: str, seed: int, config: dict[str, object]) -> Path:
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{dataset}-seed{seed}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config_dir = project_root / "configs"
    base = load_config(
        [
            f"dataset={args.dataset}",
            f"model={args.model}",
            f"trainer.device={args.device}",
            f"trainer.max_epochs={args.max_epochs}",
        ],
        config_dir,
    )
    output_dir = project_root / str(base["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Loading dataset={args.dataset} model={args.model}")
    graph = load_graph(
        args.dataset,
        root=str(project_root / base["dataset_config"]["root"]),
        dataset_config=base["dataset_config"],
    )
    _log(
        "Loaded graph "
        f"nodes={graph.metadata.num_nodes} edges={graph.metadata.num_edges} "
        f"features={graph.metadata.num_features} classes={graph.metadata.num_classes}"
    )

    runs: list[tuple[int, dict[str, object], dict[str, object], str]] = []
    if args.sweep == "seed":
        for seed in range(args.seed_start, args.seed_start + args.seed_count):
            runs.append((seed, dict(base["model_config"]), dict(base["trainer"]), args.rashomon_type or "seed"))
    else:
        for lr, dropout, weight_decay, hidden in product(
            args.learning_rates,
            args.dropouts,
            args.weight_decays,
            args.hidden_channels,
        ):
            model_config = dict(base["model_config"])
            trainer = dict(base["trainer"])
            model_config["dropout"] = dropout
            model_config["hidden_channels"] = hidden
            trainer["learning_rate"] = lr
            trainer["weight_decay"] = weight_decay
            runs.append((0, model_config, trainer, args.rashomon_type or "hyperparameter"))

    for index, (seed, model_config, trainer, rashomon_type) in enumerate(runs, start=1):
        resolved = dict(base)
        resolved["seed"] = seed
        resolved["model_config"] = model_config
        resolved["trainer"] = trainer
        resolved["rashomon_type"] = rashomon_type
        config_path = _write_config(output_dir, args.dataset, seed, resolved)
        _log(
            f"Training {index}/{len(runs)} dataset={args.dataset} architecture={model_config['architecture']} "
            f"rashomon_type={rashomon_type} seed={seed}"
        )
        record = train_model(
            graph=graph,
            model_config=model_config,
            training_config=trainer,
            seed=seed,
            dataset=args.dataset,
            output_dir=output_dir,
            config_path=str(config_path),
            rashomon_type=rashomon_type,
        )
        _log(f"Finished run_id={record.run_id} test_accuracy={record.test_accuracy:.4f}")


if __name__ == "__main__":
    main()
