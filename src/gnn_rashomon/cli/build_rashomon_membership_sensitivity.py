from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn_rashomon.analysis.multiplicity_from_set import multiplicity_metrics_from_set
from gnn_rashomon.analysis.performance_sensitivity import (
    constrain_by_validation_accuracy,
    constrain_by_validation_accuracy_only,
    train_loss_only_view,
)
from gnn_rashomon.rashomon.io import read_json, write_json

DEFAULT_HEADLINE_SETS = [
    "outputs/rashomon_sets/cora-gcn-seed-relative0.1-epochs200.json",
    "outputs/rashomon_sets/citeseer-gcn-seed-relative0.1-epochs200.json",
    "outputs/rashomon_sets/pubmed-gcn-seed-relative0.1-epochs200.json",
    "outputs/rashomon_sets/cora-gcn-hyperparameter-relative2-epochs200.json",
    "outputs/rashomon_sets/citeseer-gcn-hyperparameter-relative2-epochs200.json",
    "outputs/rashomon_sets/pubmed-gcn-hyperparameter-relative2-epochs200.json",
    "outputs/rashomon_sets/amazon_photo-gcn-seed-relative0.1-epochs200.json",
    "outputs/rashomon_sets/amazon_photo-gcn-hyperparameter-relative2-epochs200.json",
    "outputs/rashomon_sets/ogbn_arxiv-gcn-seed-relative0.1-epochs300.json",
    "outputs/rashomon_sets/nba-gcn-seed-relative0.1-epochs200.json",
    "outputs/rashomon_sets/nba-gcn-hyperparameter-relative2-epochs200.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild headline multiplicity tables under train-loss, validation-only, and "
            "joint train+validation Rashomon membership definitions."
        )
    )
    parser.add_argument("--rashomon-set", action="append", default=[])
    parser.add_argument(
        "--validation-accuracy-delta",
        action="append",
        type=float,
        default=[],
        help="Absolute validation-accuracy tolerances. Defaults to 0.01, 0.02, 0.05.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rashomon_membership_sensitivity",
    )
    parser.add_argument(
        "--primary-delta",
        type=float,
        default=0.02,
        help="Marked as the primary joint-membership condition in summary outputs.",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _row_from_payload(
    payload: dict[str, object],
    summary: dict[str, object],
    *,
    source_path: Path,
    derived_set_path: Path | None,
    project_root: Path,
    primary_delta: float,
) -> dict[str, object]:
    metrics = multiplicity_metrics_from_set(payload, project_root=project_root)
    delta = summary.get("validation_accuracy_delta")
    is_primary = (
        summary.get("membership_definition") == "joint_train_loss_and_validation_accuracy"
        and delta is not None
        and abs(float(delta) - primary_delta) < 1e-12
    )
    return {
        **summary,
        **metrics,
        "source_path": str(source_path),
        "derived_set_path": str(derived_set_path) if derived_set_path else str(source_path),
        "is_primary_condition": bool(is_primary),
    }


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    set_paths = [Path(path) for path in (args.rashomon_set or DEFAULT_HEADLINE_SETS)]
    deltas = sorted(set(args.validation_accuracy_delta or [0.01, 0.02, 0.05]))
    output_dir = Path(args.output_dir)
    set_dir = output_dir / "rashomon_sets"
    rows: list[dict[str, object]] = []

    for source_path in set_paths:
        if not source_path.exists():
            print(f"Skipping missing set: {source_path}")
            continue
        source = read_json(source_path)

        train_payload, train_summary = train_loss_only_view(source)
        rows.append(
            _row_from_payload(
                train_payload,
                train_summary,
                source_path=source_path,
                derived_set_path=source_path,
                project_root=project_root,
                primary_delta=args.primary_delta,
            )
        )

        for delta in deltas:
            joint_payload, joint_summary = constrain_by_validation_accuracy(source, delta)
            joint_path = set_dir / f"{joint_payload['set_id']}.json"
            write_json(joint_path, joint_payload)
            rows.append(
                _row_from_payload(
                    joint_payload,
                    joint_summary,
                    source_path=source_path,
                    derived_set_path=joint_path,
                    project_root=project_root,
                    primary_delta=args.primary_delta,
                )
            )

            val_payload, val_summary = constrain_by_validation_accuracy_only(source, delta)
            val_path = set_dir / f"{val_payload['set_id']}.json"
            write_json(val_path, val_payload)
            rows.append(
                _row_from_payload(
                    val_payload,
                    val_summary,
                    source_path=source_path,
                    derived_set_path=val_path,
                    project_root=project_root,
                    primary_delta=args.primary_delta,
                )
            )

    if not rows:
        raise SystemExit("No Rashomon sets were processed.")

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_csv = output_dir / "membership_sensitivity_all.csv"
    primary_csv = output_dir / "primary_joint_validation_matched.csv"
    ablation_csv = output_dir / "definition_ablation.csv"
    meta_json = output_dir / "membership_sensitivity_summary.json"

    frame.to_csv(all_csv, index=False)
    primary = frame[
        (frame["membership_definition"] == "joint_train_loss_and_validation_accuracy")
        & (frame["validation_accuracy_delta"] == args.primary_delta)
    ].copy()
    primary.to_csv(primary_csv, index=False)

    # Wide ablation view at the primary delta (plus train-loss-only).
    ablation_parts = [
        frame[frame["membership_definition"] == "train_loss_only"],
        frame[
            (frame["membership_definition"] == "validation_accuracy_only")
            & (frame["validation_accuracy_delta"] == args.primary_delta)
        ],
        frame[
            (frame["membership_definition"] == "joint_train_loss_and_validation_accuracy")
            & (frame["validation_accuracy_delta"] == args.primary_delta)
        ],
    ]
    ablation = pd.concat(ablation_parts, ignore_index=True)
    ablation.to_csv(ablation_csv, index=False)

    meta = {
        "primary_delta": args.primary_delta,
        "deltas": deltas,
        "n_source_sets": int(frame["parent_set_id"].nunique()),
        "n_rows": int(len(frame)),
        "all_csv": str(all_csv),
        "primary_csv": str(primary_csv),
        "ablation_csv": str(ablation_csv),
        "note": (
            "Primary headline condition is joint train-loss + validation matching at "
            f"delta={args.primary_delta}. Train-loss-only remains available as secondary."
        ),
    }
    meta_json.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {all_csv}")
    print(f"Wrote {primary_csv}")
    print(f"Wrote {ablation_csv}")
    print(f"Wrote {meta_json}")


if __name__ == "__main__":
    main()
