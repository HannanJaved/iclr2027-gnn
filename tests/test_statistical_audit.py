import json

import pandas as pd

from gnn_rashomon.cli.build_statistical_audit import build_statistical_audit


def test_build_statistical_audit_applies_family_corrections(tmp_path):
    metrics = tmp_path / "outputs" / "metrics"
    figures = tmp_path / "outputs" / "figures"
    explanations = tmp_path / "outputs" / "explanations"
    metrics.mkdir(parents=True)
    figures.mkdir(parents=True)
    explanations.mkdir(parents=True)

    pd.DataFrame(
        {
            "term": ["local_homophily", "neighborhood_label_entropy"],
            "P>|z|": [0.01, 0.20],
            "Coef.": [1.0, 2.0],
            "target": ["rashomon_capacity", "rashomon_capacity"],
            "nobs": [10, 10],
        }
    ).to_csv(metrics / "cora-gcn-seed-relative0.1-epochs200.structure_regression.csv", index=False)
    pd.DataFrame(
        {
            "outcome": ["delta_rashomon_capacity"],
            "term": ["delta_neighborhood_label_entropy"],
            "coefficient": [0.5],
            "p_value": [0.02],
            "n": [20],
        }
    ).to_csv(figures / "cora_local_entanglement_treatment_control.regression.csv", index=False)
    pd.DataFrame(
        {
            "metric": ["delta_neighborhood_label_entropy"],
            "comparison": ["treatment_minus_matched_control"],
            "effect_mean": [0.1],
            "p_value": [0.03],
            "n": [20],
        }
    ).to_csv(figures / "cora_local_entanglement_treatment_control.tests.csv", index=False)
    pd.DataFrame(
        {
            "outcome": ["top_k_jaccard_mean"],
            "term": ["rashomon_capacity"],
            "coefficient": [0.1],
            "p_value": [0.04],
            "n": [30],
        }
    ).to_csv(explanations / "cora-gcn-seed-relative0.1-epochs200.gnnexplainer.regression.csv", index=False)

    tests_path, summary_path = build_statistical_audit(tmp_path, tmp_path / "outputs" / "audit")
    tests = pd.read_csv(tests_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert {"p_value_fdr_bh", "p_value_bonferroni"}.issubset(tests.columns)
    assert summary["test_count"] == 5
    assert "citation_structure_regressions" in summary["families"]
