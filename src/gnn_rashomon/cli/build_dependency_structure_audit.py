from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

FOCAL_EXPLANATIONS = [
    "cora-gcn-seed-relative0.1-epochs200.gnnexplainer",
    "cora-gat-arch_gat-relative2-epochs200.gnnexplainer",
    "cora-appnp-arch_appnp-relative2-epochs200.gnnexplainer",
    "pubmed-gat-arch_gat-relative2-epochs200.gnnexplainer",
    "pubmed-appnp-arch_appnp-relative2-epochs200.gnnexplainer",
    "cora-gcn-seed-relative0.1-epochs200.pgexplainer",
    "cora-gat-arch_gat-relative2-epochs200.pgexplainer",
    "cora-appnp-arch_appnp-relative2-epochs200.pgexplainer",
    "pubmed-gat-arch_gat-relative2-epochs200.pgexplainer",
    "pubmed-appnp-arch_appnp-relative2-epochs200.pgexplainer",
]

ROBUSTNESS_EXPLANATIONS = [
    "cora-gat-arch_gat-relative2-epochs200.gat_attention",
    "amazon_photo-gcn-seed-relative0.1-epochs200.gnnexplainer",
    "amazon_photo-gat-arch_gat-relative0.1-epochs200.gnnexplainer",
    "amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200.gnnexplainer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the manuscript dependency-structure audit.")
    parser.add_argument(
        "--fairness-audit",
        default=(
            "outputs/release/adult_corrected_20260715_124823/figures/"
            "fairness_concentration_dependency_audit.csv"
        ),
    )
    parser.add_argument(
        "--output-prefix", default="outputs/figures/dependency_structure_audit"
    )
    parser.add_argument(
        "--repeated-rewiring-inference",
        default="outputs/repeated_rewiring/repeated_rewiring_inference.csv",
    )
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def _community_permutation_p(
    frame: pd.DataFrame, permutations: int, seed: int
) -> tuple[float, int, int]:
    x = frame["rashomon_capacity"].to_numpy(dtype=float)
    y = frame["top_k_jaccard_mean"].to_numpy(dtype=float)
    communities = frame["community_id"].to_numpy()
    observed = float(spearmanr(x, y).statistic)
    indices = [np.flatnonzero(communities == group) for group in np.unique(communities)]
    permutable = [index for index in indices if len(index) >= 2]
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = x.copy()
        for index in permutable:
            shuffled[index] = rng.permutation(shuffled[index])
        statistic = float(spearmanr(shuffled, y).statistic)
        exceed += int(abs(statistic) >= abs(observed))
    return (exceed + 1) / (permutations + 1), len(indices), sum(map(len, permutable))


def _explanation_audit(root: Path, permutations: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prefixes = FOCAL_EXPLANATIONS + ROBUSTNESS_EXPLANATIONS
    for position, prefix in enumerate(prefixes):
        base = root / "outputs" / "explanations" / prefix
        summary = pd.read_csv(Path(f"{base}.node_summary.csv"))
        pairwise = pd.read_csv(Path(f"{base}.pairwise.csv"))
        regression = pd.read_csv(Path(f"{base}.regression.csv"))
        if summary["node_id"].duplicated().any():
            raise ValueError(f"{prefix}: node summary contains repeated node rows")
        capacity = regression.loc[regression["term"] == "rashomon_capacity"].iloc[0]
        if "community-clustered" not in str(capacity["note"]):
            raise ValueError(f"{prefix}: regression is not community-clustered")
        if "small-sample t reference" not in str(capacity["note"]):
            raise ValueError(f"{prefix}: regression lacks finite-cluster inference")
        direct = float(
            spearmanr(summary["rashomon_capacity"], summary["top_k_jaccard_mean"]).statistic
        )
        direct_p, communities, permutable_nodes = _community_permutation_p(
            summary.dropna(
                subset=["rashomon_capacity", "top_k_jaccard_mean", "community_id"]
            ),
            permutations,
            seed + position,
        )
        rows.append(
            {
                "analysis": prefix,
                "family": "focal_10" if prefix in FOCAL_EXPLANATIONS else "robustness",
                "nodes": len(summary),
                "model_pair_rows": len(pairwise),
                "communities": communities,
                "permutable_nodes": permutable_nodes,
                "direct_spearman": direct,
                "direct_community_permutation_p": direct_p,
                "controlled_diameter_coefficient": float(capacity["coefficient"]),
                "controlled_community_clustered_p": float(capacity["p_value"]),
                "cluster_count": int(capacity["cluster_count"]),
                "reference_df": int(capacity["reference_df"]),
                "covariance_note": capacity["note"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["controlled_bh_q_focal_10"] = np.nan
    frame["controlled_bonferroni_p_focal_10"] = np.nan
    focal = frame["family"] == "focal_10"
    focal_p = frame.loc[focal, "controlled_community_clustered_p"].to_numpy(dtype=float)
    frame.loc[focal, "controlled_bh_q_focal_10"] = multipletests(
        focal_p, method="fdr_bh"
    )[1]
    frame.loc[focal, "controlled_bonferroni_p_focal_10"] = multipletests(
        focal_p, method="bonferroni"
    )[1]
    return frame


def _require_modes(path: Path, column: str, modes: set[str]) -> None:
    frame = pd.read_csv(path)
    observed = set(frame[column].astype(str))
    if not modes.issubset(observed):
        raise ValueError(f"{path}: expected modes {sorted(modes)}, observed {sorted(observed)}")


def _claim_inventory(
    root: Path, fairness_path: Path, repeated_rewiring_path: Path
) -> pd.DataFrame:
    repeated = pd.read_csv(repeated_rewiring_path)
    expected_datasets = {"cora", "citeseer", "pubmed"}
    local = repeated[
        repeated["effect"].eq("matched_treatment_control_delta_diameter")
    ]
    if set(local["dataset"]) != expected_datasets:
        raise ValueError("Repeated local-rewiring inference is incomplete")
    if not local["realization_count"].eq(20).all():
        raise ValueError("Repeated local-rewiring inference requires 20 realizations")
    if not local["inference_unit"].eq(
        "independently seeded rewired graph realization"
    ).all():
        raise ValueError("Repeated local-rewiring inference uses the wrong unit")

    global_rewiring = repeated[
        repeated["effect"].isin(
            ["delta_disagreement_fraction", "delta_mean_probability_diameter"]
        )
    ]
    if set(global_rewiring["dataset"]) != expected_datasets:
        raise ValueError("Repeated global-rewiring inference is incomplete")
    if not global_rewiring["realization_count"].eq(20).all():
        raise ValueError("Repeated global-rewiring inference requires 20 realizations")

    _require_modes(
        root / "outputs" / "figures" / "stage1_structural_permutation_audit.csv",
        "mode",
        {"degree", "community"},
    )
    _require_modes(
        root / "outputs" / "figures" / "stage1_middle_link_permutation_audit.csv",
        "mode",
        {"degree", "community"},
    )
    _require_modes(
        root / "outputs" / "figures" / "amazon_photo_middle_link_permutation_audit.csv",
        "mode",
        {"degree", "community"},
    )
    _require_modes(
        root / "outputs" / "figures" / "nba_structural_permutation_audit.csv",
        "permutation_null",
        {"degree", "community"},
    )
    fairness = pd.read_csv(fairness_path)
    if len(fairness) != 4 or fairness["community_permutation_p_two_sided"].isna().any():
        raise ValueError("Fairness dependency audit is incomplete")

    rows = [
        (
            "Repeated local entropy intervention",
            "Matched treatment/control nodes within each rewired graph; 20 graph realizations",
            "One matched mean effect per graph realization; t and signed-rank "
            "inference across realizations",
            "pass",
        ),
        (
            "Repeated global rewiring interventions",
            "The same fixed retained ensemble evaluated on 20 independently seeded "
            "graphs per condition",
            "Graph realization is the inferential unit; model weights and Rashomon "
            "membership remain fixed",
            "pass",
        ),
        (
            "Explanation stability regressions",
            "Many model-pair similarities per node; nodes clustered in communities",
            "Model-pair rows averaged to one row per node; SEs clustered by community",
            "pass",
        ),
        (
            "Citation structural and middle-link claims",
            "One row per graph node with graph dependence",
            "Degree-bin and community-preserving permutation nulls; "
            "asymptotic node p-values secondary",
            "pass_with_stated_limit",
        ),
        (
            "Amazon-Photo middle-link claims",
            "Same nodes appear in separate architecture analyses",
            "No pooled cross-architecture contrast; each node appears once per regression; "
            "graph-aware permutations",
            "pass",
        ),
        (
            "NBA entropy claims",
            "Same nodes appear in separate seed/hyperparameter analyses",
            "No paired cross-set contrast; degree-bin and community-preserving permutations "
            "within each set",
            "pass",
        ),
        (
            "Adult/German multiplicity concentration",
            "One row per node with sensitive-group and community clustering",
            "Sensitive labels permuted within detected communities; "
            "rank-sum p-values treated as secondary",
            "pass_with_stated_limit",
        ),
        (
            "Architecture, Pareto, and baseline comparisons",
            "Repeated seeds, nodes, or baseline runs across descriptive comparisons",
            "No population-level inferential contrast is claimed; "
            "counts and frontiers are descriptive",
            "not_applicable",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["claim_family", "dependency_structure", "handling", "verdict"],
    )


def main() -> None:
    args = parse_args()
    root = Path(".")
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    explanation = _explanation_audit(root, args.permutations, args.seed)
    claims = _claim_inventory(
        root,
        Path(args.fairness_audit),
        Path(args.repeated_rewiring_inference),
    )
    claims_path = Path(f"{prefix}.claims.csv")
    explanation_path = Path(f"{prefix}.explanations.csv")
    summary_path = Path(f"{prefix}.summary.json")
    claims.to_csv(claims_path, index=False)
    explanation.to_csv(explanation_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "claim_families": len(claims),
                "explanation_analyses": len(explanation),
                "explanation_permutations": args.permutations,
                "claims_csv": str(claims_path),
                "explanations_csv": str(explanation_path),
                "all_required_checks_passed": True,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(claims.to_string(index=False))
    print(f"Wrote {claims_path}")
    print(f"Wrote {explanation_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
