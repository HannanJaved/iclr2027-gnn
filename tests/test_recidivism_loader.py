import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None,
    reason="Recidivism loader test requires torch and torch-geometric.",
)
def test_recidivism_loader_filters_binary_race_and_builds_graph(tmp_path):
    from gnn_rashomon.data.loaders import load_graph

    root = tmp_path / "Recidivism"
    root.mkdir()
    csv_path = root / "compas-scores-two-years.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,name,race,sex,age,age_cat,juv_fel_count,juv_misd_count,juv_other_count,priors_count,c_charge_degree,two_year_recid,is_recid,decile_score,score_text",
                "1,A,African-American,Male,25,Less than 25,0,0,0,2,F,1,1,7,High",
                "2,B,Caucasian,Female,45,Greater than 45,0,1,0,0,M,0,0,3,Low",
                "3,C,African-American,Male,35,25 - 45,1,0,0,5,F,1,1,8,High",
                "4,D,Caucasian,Male,30,25 - 45,0,0,1,1,M,0,0,2,Low",
                "5,E,Hispanic,Female,40,25 - 45,0,0,0,1,F,0,0,4,Low",
                "6,F,Caucasian,Female,29,25 - 45,0,0,0,3,F,1,1,6,Medium",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_graph(
        "recidivism",
        root=str(root),
        dataset_config={
            "csv_path": str(csv_path),
            "k_neighbors": 2,
            "use_cache": False,
            "race_groups": ["Caucasian", "African-American"],
        },
    )

    assert bundle.metadata.dataset == "recidivism"
    assert bundle.metadata.num_nodes == 5
    assert bundle.metadata.num_classes == 2
    assert "race" in bundle.sensitive_attributes
    assert int(bundle.data.y.sum()) == 3
    assert int(bundle.sensitive_attributes["race"].sum()) == 2
