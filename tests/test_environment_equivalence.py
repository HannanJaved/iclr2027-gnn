import json

import pandas as pd

from gnn_rashomon.cli.compare_artifact_environments import build_report


def test_environment_comparison_accepts_numerically_equivalent_artifacts(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    pd.DataFrame({"name": ["a"], "value": [1.0]}).to_csv(reference / "table.csv", index=False)
    pd.DataFrame({"name": ["a"], "value": [1.0 + 1e-9]}).to_csv(
        candidate / "table.csv", index=False
    )
    (reference / "summary.json").write_text(json.dumps({"score": 2.0}), encoding="utf-8")
    (candidate / "summary.json").write_text(
        json.dumps({"score": 2.0 + 1e-9}), encoding="utf-8"
    )

    report = build_report(reference, candidate, ["*.csv", "*.json"], 1e-6, 1e-8)

    assert report["passed"] is True
    assert report["artifact_count"] == 2


def test_environment_comparison_reports_missing_candidates(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    (reference / "summary.json").write_text(json.dumps({"score": 2.0}), encoding="utf-8")

    report = build_report(reference, candidate, ["*.json"], 1e-6, 1e-8)

    assert report["passed"] is False
    assert report["artifacts"][0]["detail"] == "missing candidate"
