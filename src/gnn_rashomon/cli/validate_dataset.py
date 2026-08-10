from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.data.splits import validate_masks


def main(argv: list[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    config = load_config(args, Path("configs"))
    dataset = str(config["dataset"])
    dataset_config = dict(config.get("dataset_config", {}))
    root = dataset_config.get("root", config["paths"]["data_dir"])
    output_dir = Path(config["paths"]["output_dir"]) / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset={dataset} root={root}", flush=True)
    bundle = load_graph(dataset, root=str(root), dataset_config=dataset_config)
    print(
        "Loaded graph "
        f"nodes={bundle.metadata.num_nodes} edges={bundle.metadata.num_edges} "
        f"features={bundle.metadata.num_features}",
        flush=True,
    )
    split = validate_masks(
        bundle.data.train_mask,
        bundle.data.val_mask,
        bundle.data.test_mask,
        num_nodes=int(bundle.data.num_nodes),
        allow_unassigned=str(dataset_config.get("split", "")).lower() == "public",
    )
    if not split.valid:
        raise SystemExit(
            f"Invalid masks for dataset={dataset}: "
            f"covers_all_nodes={split.covers_all_nodes} mutually_exclusive={split.mutually_exclusive}"
        )

    report = {
        "dataset": dataset,
        "metadata": asdict(bundle.metadata),
        "split_validation": asdict(split),
        "sensitive_attributes": sorted(bundle.sensitive_attributes.keys()),
    }
    path = output_dir / f"{dataset}.metadata.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote dataset validation report to {path}", flush=True)


if __name__ == "__main__":
    main()
