from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests


CITATION_DATASETS = ("cora", "citeseer", "pubmed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multiple-comparison correction tables from saved analysis artifacts."
    )
    parser.add_argument("--output-prefix", default="outputs/figures/statistical_audit")
    return parser.parse_args()


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _append_rows(rows: list[dict[str, object]], family: str, source: Path, frame: pd.DataFrame) -> None:
    for _, row in frame.iterrows():
        p_value = row.get("p_value", row.get("P>|z|"))
        if pd.isna(p_value):
            continue
        rows.append(
            {
                "family": family,
                "source": str(source),
                "dataset": row.get("dataset"),
                "analysis": row.get("analysis"),
                "outcome": row.get("outcome", row.get("target")),
                "term": row.get("term", row.get("predictor", row.get("comparison"))),
                "metric": row.get("metric"),
                "comparison": row.get("comparison"),
                "coefficient": row.get("coefficient", row.get("Coef.", row.get("spearman"))),
                "effect_mean": row.get("effect_mean"),
                "n": row.get("n", row.get("nobs")),
                "p_value": float(p_value),
            }
        )


def _collect_structure_regressions(root: Path, rows: list[dict[str, object]]) -> None:
    terms = {"log_degree", "local_homophily", "neighborhood_label_entropy"}
    for dataset in CITATION_DATASETS:
        path = root / "outputs" / "metrics" / f"{dataset}-gcn-seed-relative0.1-epochs200.structure_regression.csv"
        frame = _read_if_exists(path)
        if frame is None:
            continue
        frame = frame[frame["term"].isin(terms)].copy()
        frame["dataset"] = dataset
        frame["analysis"] = "citation_seed_structure_regression"
        _append_rows(rows, "citation_structure_regressions", path, frame)


def _collect_local_entanglement(root: Path, rows: list[dict[str, object]]) -> None:
    for dataset in CITATION_DATASETS:
        regression_path = root / "outputs" / "figures" / f"{dataset}_local_entanglement_treatment_control.regression.csv"
        regression = _read_if_exists(regression_path)
        if regression is not None:
            regression = regression[
                (regression["outcome"] == "delta_rashomon_capacity")
                & (regression["term"] == "delta_neighborhood_label_entropy")
            ].copy()
            regression["dataset"] = dataset
            regression["analysis"] = "local_entanglement_capacity_regression"
            _append_rows(rows, "local_entanglement_regressions", regression_path, regression)

        tests_path = root / "outputs" / "figures" / f"{dataset}_local_entanglement_treatment_control.tests.csv"
        tests = _read_if_exists(tests_path)
        if tests is not None:
            tests = tests[
                (tests["metric"] == "delta_neighborhood_label_entropy")
                & (tests["comparison"] == "treatment_minus_matched_control")
            ].copy()
            tests["dataset"] = dataset
            tests["analysis"] = "local_entanglement_manipulation_check"
            _append_rows(rows, "local_entanglement_manipulation_tests", tests_path, tests)


def _collect_explanation_regressions(root: Path, rows: list[dict[str, object]]) -> None:
    prefixes = {
        "cora_gcn": "cora-gcn-seed-relative0.1-epochs200.gnnexplainer",
        "cora_gat": "cora-gat-arch_gat-relative2-epochs200.gnnexplainer",
        "cora_appnp": "cora-appnp-arch_appnp-relative2-epochs200.gnnexplainer",
        "pubmed_gat": "pubmed-gat-arch_gat-relative2-epochs200.gnnexplainer",
        "pubmed_appnp": "pubmed-appnp-arch_appnp-relative2-epochs200.gnnexplainer",
    }
    for label, prefix in prefixes.items():
        path = root / "outputs" / "explanations" / f"{prefix}.regression.csv"
        frame = _read_if_exists(path)
        if frame is None:
            continue
        frame = frame[frame["term"].isin(["rashomon_capacity", "degree", "confidence"])].copy()
        frame["dataset"] = label
        frame["analysis"] = "explanation_controlled_regression"
        _append_rows(rows, "explanation_controlled_regressions", path, frame)


def _apply_corrections(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    corrected_parts: list[pd.DataFrame] = []
    for family, family_frame in frame.groupby("family", sort=True):
        family_frame = family_frame.copy()
        p_values = family_frame["p_value"].astype(float).to_numpy()
        _, fdr, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        _, bonferroni, _, _ = multipletests(p_values, alpha=0.05, method="bonferroni")
        family_frame["p_value_fdr_bh"] = fdr
        family_frame["p_value_bonferroni"] = bonferroni
        family_frame["significant_fdr_bh_0p05"] = family_frame["p_value_fdr_bh"] <= 0.05
        family_frame["significant_bonferroni_0p05"] = family_frame["p_value_bonferroni"] <= 0.05
        corrected_parts.append(family_frame)
    return pd.concat(corrected_parts, ignore_index=True)


def build_statistical_audit(root: Path, output_prefix: Path) -> tuple[Path, Path]:
    rows: list[dict[str, object]] = []
    _collect_structure_regressions(root, rows)
    _collect_local_entanglement(root, rows)
    _collect_explanation_regressions(root, rows)
    frame = _apply_corrections(pd.DataFrame(rows))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tests_path = Path(f"{output_prefix}.tests.csv")
    summary_path = Path(f"{output_prefix}.summary.json")
    frame.to_csv(tests_path, index=False)

    summary = {
        "test_count": int(len(frame)),
        "families": {
            family: {
                "test_count": int(len(family_frame)),
                "fdr_significant_count": int(family_frame["significant_fdr_bh_0p05"].sum()),
                "bonferroni_significant_count": int(
                    family_frame["significant_bonferroni_0p05"].sum()
                ),
            }
            for family, family_frame in frame.groupby("family", sort=True)
        },
        "tests_table": str(tests_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return tests_path, summary_path


def main() -> None:
    args = parse_args()
    tests_path, summary_path = build_statistical_audit(Path("."), Path(args.output_prefix))
    print(f"Wrote statistical audit tests to {tests_path}")
    print(f"Wrote statistical audit summary to {summary_path}")


if __name__ == "__main__":
    main()
