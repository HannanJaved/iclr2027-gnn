from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectedNode:
    node_id: int
    multiplicity_group: str
    rashomon_capacity: float
    degree: int | None
    label: int | None
    correct: int | None
    confidence: float | None


def select_nodes_by_multiplicity(
    node_table: pd.DataFrame,
    max_nodes: int = 50,
    metric: str = "rashomon_capacity",
    seed: int = 0,
) -> pd.DataFrame:
    if metric not in node_table.columns:
        raise ValueError(f"Missing multiplicity metric column: {metric}")
    if "node_id" not in node_table.columns:
        raise ValueError("Node table must contain node_id.")

    working = node_table.dropna(subset=[metric]).copy()
    if working.empty:
        raise ValueError(f"No finite values available for {metric}.")
    low, high = working[metric].quantile([1 / 3, 2 / 3])
    working["multiplicity_group"] = "medium"
    working.loc[working[metric] <= low, "multiplicity_group"] = "low"
    working.loc[working[metric] >= high, "multiplicity_group"] = "high"

    rng = np.random.default_rng(seed)
    per_group = max(1, max_nodes // 3)
    selected: list[pd.DataFrame] = []
    for group in ["high", "medium", "low"]:
        subset = working[working["multiplicity_group"] == group]
        if subset.empty:
            continue
        order = rng.permutation(len(subset))
        selected.append(subset.iloc[order[:per_group]])
    out = pd.concat(selected, ignore_index=True) if selected else working.head(0)
    if len(out) < max_nodes:
        remaining = working[~working["node_id"].isin(out["node_id"])]
        if not remaining.empty:
            order = rng.permutation(len(remaining))
            out = pd.concat([out, remaining.iloc[order[: max_nodes - len(out)]]], ignore_index=True)
    return out.head(max_nodes).sort_values(["multiplicity_group", metric], ascending=[True, False])


def edge_score_vector(edge_index: np.ndarray, edge_mask: np.ndarray) -> pd.DataFrame:
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")
    if edge_index.shape[1] != edge_mask.shape[0]:
        raise ValueError("edge_mask length must match edge count.")
    rows = []
    for pos, (src, dst) in enumerate(edge_index.T.astype(int)):
        a, b = sorted((int(src), int(dst)))
        rows.append({"edge_key": f"{a}:{b}", "edge_score": float(edge_mask[pos])})
    frame = pd.DataFrame(rows)
    return frame.groupby("edge_key", as_index=False)["edge_score"].max()


def top_k_edges(scores: pd.DataFrame, k: int) -> set[str]:
    if scores.empty or k <= 0:
        return set()
    top = scores.sort_values("edge_score", ascending=False).head(k)
    return set(top["edge_key"].astype(str))


def _rank_correlation(left: np.ndarray, right: np.ndarray, method: str) -> tuple[float, float]:
    try:
        from scipy.stats import kendalltau, spearmanr
    except Exception:
        return float("nan"), float("nan")
    if method == "spearman":
        result = spearmanr(left, right)
    elif method == "kendall":
        result = kendalltau(left, right)
    else:
        raise ValueError(f"Unknown rank-correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def compare_edge_explanations(
    explanations: dict[tuple[int, str], pd.DataFrame],
    top_k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    nodes = sorted({node_id for node_id, _ in explanations})
    for node_id in nodes:
        run_ids = sorted(run_id for n_id, run_id in explanations if n_id == node_id)
        for left_run, right_run in combinations(run_ids, 2):
            left = explanations[(node_id, left_run)]
            right = explanations[(node_id, right_run)]
            joined = left.merge(
                right, on="edge_key", how="outer", suffixes=("_left", "_right")
            ).fillna(0.0)
            left_scores = joined["edge_score_left"].to_numpy(dtype=float)
            right_scores = joined["edge_score_right"].to_numpy(dtype=float)
            left_top = top_k_edges(left, top_k)
            right_top = top_k_edges(right, top_k)
            union = left_top | right_top
            jaccard = float(len(left_top & right_top) / len(union)) if union else float("nan")
            spearman, spearman_p = _rank_correlation(left_scores, right_scores, "spearman")
            kendall, kendall_p = _rank_correlation(left_scores, right_scores, "kendall")
            rows.append(
                {
                    "node_id": node_id,
                    "left_run_id": left_run,
                    "right_run_id": right_run,
                    "top_k": top_k,
                    "top_k_jaccard": jaccard,
                    "spearman": spearman,
                    "spearman_p_value": spearman_p,
                    "kendall": kendall,
                    "kendall_p_value": kendall_p,
                    "left_explanation_size": int((left["edge_score"] > 0).sum()),
                    "right_explanation_size": int((right["edge_score"] > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_instability(pairwise: pd.DataFrame, selected_nodes: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    grouped = pairwise.groupby("node_id", as_index=False).agg(
        top_k_jaccard_mean=("top_k_jaccard", "mean"),
        top_k_jaccard_median=("top_k_jaccard", "median"),
        top_k_jaccard_min=("top_k_jaccard", "min"),
        spearman_mean=("spearman", "mean"),
        spearman_median=("spearman", "median"),
        kendall_mean=("kendall", "mean"),
        explanation_size_mean=("left_explanation_size", "mean"),
        model_pair_count=("top_k_jaccard", "size"),
    )
    node_cols = [
        col
        for col in [
            "node_id",
            "multiplicity_group",
            "rashomon_capacity",
            "predictive_entropy",
            "variation_ratio",
            "degree",
            "label",
            "correct",
            "confidence",
            "community_id",
        ]
        if col in selected_nodes.columns
    ]
    return selected_nodes[node_cols].merge(grouped, on="node_id", how="left")


def save_top_edges(
    explanations: dict[tuple[int, str], pd.DataFrame],
    run_metadata: dict[str, dict[str, object]],
    selected_nodes: pd.DataFrame,
    output_path: str | Path,
    top_k: int,
) -> None:
    rows: list[dict[str, object]] = []
    groups = selected_nodes.set_index("node_id")["multiplicity_group"].to_dict()
    for (node_id, run_id), scores in explanations.items():
        top = scores.sort_values("edge_score", ascending=False).head(top_k)
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            rows.append(
                {
                    "node_id": node_id,
                    "multiplicity_group": groups.get(node_id),
                    "run_id": run_id,
                    "seed": run_metadata.get(run_id, {}).get("seed"),
                    "rank": rank,
                    "edge_key": row.edge_key,
                    "edge_score": float(row.edge_score),
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def instability_regression(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "outcome": "top_k_jaccard_mean",
                    "term": "statsmodels_unavailable",
                    "coefficient": float("nan"),
                    "std_error": float("nan"),
                    "p_value": float("nan"),
                    "note": str(exc),
                }
            ]
        )
    terms = ["rashomon_capacity"]
    for column in ["degree", "confidence", "correct"]:
        if column in summary.columns:
            terms.append(column)
    required = ["top_k_jaccard_mean", *terms]
    use_community_clusters = (
        "community_id" in summary.columns and summary["community_id"].nunique(dropna=True) >= 2
    )
    if use_community_clusters:
        required.append("community_id")
    working = summary.dropna(subset=required).copy()
    if len(working) < 3:
        return pd.DataFrame()
    fitted = smf.ols(f"top_k_jaccard_mean ~ {' + '.join(terms)}", data=working)
    if use_community_clusters:
        cluster_count = int(working["community_id"].nunique())
        model = fitted.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": working["community_id"],
                "use_correction": True,
                "df_correction": True,
            },
            use_t=True,
        )
        covariance_note = (
            "node-level model-pair means; community-clustered standard errors; "
            "small-sample t reference with df=clusters-1"
        )
        reference_df = cluster_count - 1
    else:
        model = fitted.fit(cov_type="HC1")
        covariance_note = "node-level model-pair means; HC1 standard errors"
        cluster_count = 0
        reference_df = float(model.df_resid)
    return pd.DataFrame(
        [
            {
                "outcome": "top_k_jaccard_mean",
                "term": term,
                "coefficient": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "r_squared": float(model.rsquared),
                "n": int(model.nobs),
                "cluster_count": cluster_count,
                "reference_df": reference_df,
                "note": covariance_note,
            }
            for term in model.params.index
        ]
    )
