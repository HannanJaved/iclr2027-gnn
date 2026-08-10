import pandas as pd

from gnn_rashomon.cli.build_explanation_report import (
    _correlation_table,
    _group_summary,
    _paper_summary,
)


def test_explanation_report_tables_have_expected_rows():
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3, 4, 5],
            "multiplicity_group": ["low", "low", "medium", "medium", "high", "high"],
            "rashomon_capacity": [0.1, 0.2, 0.4, 0.5, 0.8, 0.9],
            "predictive_entropy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "variation_ratio": [0, 0, 0.1, 0.1, 0.2, 0.2],
            "degree": [1, 2, 3, 4, 5, 6],
            "confidence": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "correct": [1, 1, 1, 0, 0, 0],
            "top_k_jaccard_mean": [0.9, 0.8, 0.6, 0.5, 0.3, 0.2],
            "spearman_mean": [0.9, 0.8, 0.6, 0.5, 0.3, 0.2],
            "kendall_mean": [0.8, 0.7, 0.5, 0.4, 0.2, 0.1],
            "explanation_size_mean": [2, 3, 4, 5, 6, 7],
        }
    )
    pairwise = pd.DataFrame({"node_id": [0, 0, 1, 1, 2, 2]})
    metadata = {
        "dataset": "cora",
        "explainer": "gnnexplainer",
        "selected_model_count": 3,
        "top_k": 10,
        "explainer_epochs": 20,
    }

    group = _group_summary(nodes)
    corr = _correlation_table(nodes)
    paper = _paper_summary(metadata, nodes, pairwise)

    assert set(group["multiplicity_group"]) == {"low", "medium", "high"}
    assert "rashomon_capacity" in set(corr["predictor"])
    assert paper.loc[0, "selected_nodes"] == 6
    assert paper.loc[0, "high_minus_low_jaccard_mean"] < 0
