from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn_rashomon.analysis.repeated_rewiring import (
    realization_frame,
    summarize_realization_effects,
)
from gnn_rashomon.rashomon.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated rewiring with graph realization as the inference unit."
    )
    parser.add_argument("--realization", action="append", default=[])
    parser.add_argument("--realization-glob", action="append", default=[])
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(value) for value in args.realization]
    for pattern in args.realization_glob:
        paths.extend(sorted(Path().glob(pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No realization summaries were found.")

    payloads = []
    for path in paths:
        payload = read_json(path)
        rewiring = read_json(Path(str(payload["rewiring_metadata"])))
        payload["accepted_swaps"] = rewiring.get("accepted_swaps")
        if payload["mode"] == "homophily_targeted" and int(
            payload["accepted_swaps"] or 0
        ) == 0:
            payload["valid_for_inference"] = False
            payload["exclusion_reason"] = "no-op target already within tolerance"
        payloads.append(payload)
    realizations = realization_frame(payloads)
    valid = realizations.loc[realizations["valid_for_inference"].astype(bool)].copy()
    if valid.empty:
        raise SystemExit("No realization passed the manipulation and invariant checks.")
    inference = summarize_realization_effects(
        valid,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    realization_path = prefix.with_name(f"{prefix.name}_realizations.csv")
    inference_path = prefix.with_name(f"{prefix.name}_inference.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    realizations.to_csv(realization_path, index=False)
    inference.to_csv(inference_path, index=False)
    write_json(
        summary_path,
        {
            "design": (
                "Fixed retained models are evaluated on independently seeded, degree-preserving "
                "counterfactual graphs. Each graph realization contributes one effect estimate."
            ),
            "scope": (
                "Inference quantifies robustness over the rewiring randomization distribution for "
                "the observed graph; it is not a population-of-graphs estimate."
            ),
            "realization_count": len(realizations),
            "valid_realization_count": len(valid),
            "excluded_realization_count": len(realizations) - len(valid),
            "exclusion_rule": (
                "Invariant/manipulation failures and homophily targets requiring zero accepted "
                "swaps are excluded from inferential summaries."
            ),
            "source_files": [str(path) for path in paths],
            "realizations_csv": str(realization_path),
            "inference_csv": str(inference_path),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
    )
    print(json.dumps({"realizations": len(realizations), "tests": len(inference)}, indent=2))


if __name__ == "__main__":
    main()
