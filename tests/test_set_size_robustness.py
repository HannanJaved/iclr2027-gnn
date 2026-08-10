from __future__ import annotations

import numpy as np

from gnn_rashomon.analysis.set_size_robustness import (
    multiplicity_summary,
    subsample_multiplicity,
)
from gnn_rashomon.cli.summarize_set_size_robustness import flatten_summary
from gnn_rashomon.multiplicity.rashomon_capacity import (
    pairwise_total_variation_summaries,
)


def test_pairwise_tv_summaries_distinguish_maximum_and_average() -> None:
    probabilities = np.asarray(
        [
            [[1.0, 0.0]],
            [[0.5, 0.5]],
            [[0.0, 1.0]],
        ]
    )

    diameter, pairwise_mean, pairwise_q95 = pairwise_total_variation_summaries(
        probabilities
    )

    assert np.allclose(diameter, [1.0])
    assert np.allclose(pairwise_mean, [2.0 / 3.0])
    assert pairwise_q95[0] > pairwise_mean[0]


def test_subsampling_is_deterministic_and_uses_common_size() -> None:
    probabilities = np.asarray(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.6, 0.4], [0.4, 0.6]],
            [[0.2, 0.8], [0.3, 0.7]],
            [[0.1, 0.9], [0.7, 0.3]],
        ]
    )
    first = subsample_multiplicity(
        probabilities, ["a", "b", "c", "d"], sample_size=3, draws=5, seed=7
    )
    second = subsample_multiplicity(
        probabilities, ["a", "b", "c", "d"], sample_size=3, draws=5, seed=7
    )

    assert first.equals(second)
    assert set(first["sample_size"]) == {3}
    assert all(len(value.split(";")) == 3 for value in first["sampled_model_ids"])
    assert multiplicity_summary(probabilities)["mean_probability_diameter"] > 0


def test_summary_flattener_preserves_full_and_subsample_metrics() -> None:
    row = flatten_summary(
        {
            "dataset": "cora",
            "architecture": "gcn",
            "set_id": "set",
            "retained_model_count": 20,
            "sample_size": 10,
            "draw_count": 100,
            "full_set_metrics": {"mean_probability_diameter": 0.2},
            "subsample_metrics": {
                "mean_probability_diameter": {"mean": 0.1, "median": 0.09}
            },
        }
    )

    assert row["full_set_mean_probability_diameter"] == 0.2
    assert row["subsample_mean_probability_diameter_mean"] == 0.1
