import numpy as np
import pandas as pd

from gnn_rashomon.explainability.comparison import (
    compare_edge_explanations,
    edge_score_vector,
    instability_regression,
    select_nodes_by_multiplicity,
    summarize_instability,
    top_k_edges,
)


def test_select_nodes_by_multiplicity_balances_low_medium_high_groups():
    node_table = pd.DataFrame(
        {
            "node_id": np.arange(30),
            "rashomon_capacity": np.linspace(0.0, 1.0, 30),
            "degree": np.ones(30),
            "label": np.zeros(30),
            "correct": np.ones(30),
            "confidence": np.linspace(0.5, 0.9, 30),
        }
    )

    selected = select_nodes_by_multiplicity(node_table, max_nodes=9, seed=0)

    assert len(selected) == 9
    assert set(selected["multiplicity_group"]) == {"low", "medium", "high"}
    counts = selected["multiplicity_group"].value_counts().to_dict()
    assert counts == {"high": 3, "medium": 3, "low": 3}


def test_edge_score_vector_merges_directed_edges_by_undirected_key():
    edge_index = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=int)
    edge_mask = np.asarray([0.1, 0.7, 0.2, 0.4], dtype=float)

    scores = edge_score_vector(edge_index, edge_mask)

    by_edge = dict(zip(scores["edge_key"], scores["edge_score"], strict=True))
    assert by_edge == {"0:1": 0.7, "1:2": 0.4}
    assert top_k_edges(scores, 1) == {"0:1"}


def test_compare_edge_explanations_and_summarize_instability():
    explanations = {
        (5, "run-a"): pd.DataFrame({"edge_key": ["0:1", "1:2"], "edge_score": [1.0, 0.2]}),
        (5, "run-b"): pd.DataFrame({"edge_key": ["0:1", "2:3"], "edge_score": [0.9, 0.4]}),
        (5, "run-c"): pd.DataFrame({"edge_key": ["4:5", "2:3"], "edge_score": [1.0, 0.4]}),
    }
    selected = pd.DataFrame(
        {
            "node_id": [5],
            "multiplicity_group": ["high"],
            "rashomon_capacity": [0.8],
            "community_id": [2],
        }
    )

    pairwise = compare_edge_explanations(explanations, top_k=1)
    summary = summarize_instability(pairwise, selected)

    assert len(pairwise) == 3
    assert pairwise["top_k_jaccard"].between(0.0, 1.0).all()
    assert summary.loc[0, "node_id"] == 5
    assert summary.loc[0, "model_pair_count"] == 3
    assert summary.loc[0, "community_id"] == 2


def test_instability_regression_clusters_standard_errors_by_community():
    rng = np.random.default_rng(0)
    summary = pd.DataFrame(
        {
            "node_id": np.arange(40),
            "top_k_jaccard_mean": rng.uniform(0.2, 0.9, 40),
            "rashomon_capacity": rng.uniform(0.0, 0.8, 40),
            "degree": rng.integers(1, 20, 40),
            "confidence": rng.uniform(0.4, 1.0, 40),
            "correct": rng.integers(0, 2, 40),
            "community_id": np.repeat(np.arange(10), 4),
        }
    )

    regression = instability_regression(summary)

    assert not regression.empty
    assert regression["n"].eq(40).all()
    assert regression["cluster_count"].eq(10).all()
    assert regression["reference_df"].eq(9).all()
    assert regression["note"].str.contains("community-clustered").all()
    assert regression["note"].str.contains("small-sample t reference").all()
