from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn_rashomon.analysis.rewiring_effect_sizes import (
    dose_response_table,
    enrich_realization_effect_sizes,
    summarize_effect_sizes,
    summarize_node_delta_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute practical effect sizes and dose-response summaries for fixed-ensemble "
            "rewiring interventions."
        )
    )
    parser.add_argument(
        "--realizations-csv",
        default="outputs/repeated_rewiring/repeated_rewiring_realizations.csv",
    )
    parser.add_argument(
        "--node-delta-glob",
        action="append",
        default=[],
        help="Optional glob(s) of node-level delta CSVs from evaluate_fixed_set_rewiring --node-output.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/rewiring_effect_sizes/rewiring_effect_sizes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    realizations = pd.read_csv(args.realizations_csv)
    enriched = enrich_realization_effect_sizes(realizations)
    effect_summary = summarize_effect_sizes(enriched)
    dose = dose_response_table(enriched)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    enriched_path = prefix.with_name(f"{prefix.name}_realizations_enriched.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.csv")
    dose_path = prefix.with_name(f"{prefix.name}_dose_response.csv")
    meta_path = prefix.with_name(f"{prefix.name}_summary.json")

    enriched.to_csv(enriched_path, index=False)
    effect_summary.to_csv(summary_path, index=False)
    dose.to_csv(dose_path, index=False)

    node_summary_path = None
    node_paths: list[Path] = []
    for pattern in args.node_delta_glob:
        node_paths.extend(sorted(Path().glob(pattern)))
    if node_paths:
        node_summary = summarize_node_delta_paths(node_paths)
        node_summary_path = prefix.with_name(f"{prefix.name}_node_delta_summary.csv")
        node_summary.to_csv(node_summary_path, index=False)

    meta = {
        "realizations_csv": args.realizations_csv,
        "enriched_csv": str(enriched_path),
        "summary_csv": str(summary_path),
        "dose_response_csv": str(dose_path),
        "node_delta_summary_csv": str(node_summary_path) if node_summary_path else None,
        "n_realizations": int(len(enriched)),
        "n_node_tables": int(len(node_paths)),
        "practical_thresholds": [0.01, 0.05, 0.10],
        "note": (
            "Graph-level relative effects are computed immediately from existing realization "
            "summaries. Node-level quantiles require materializing --node-output CSVs."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {enriched_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {dose_path}")
    if node_summary_path:
        print(f"Wrote {node_summary_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
