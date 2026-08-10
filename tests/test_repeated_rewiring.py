from __future__ import annotations

import numpy as np
import pandas as pd

from gnn_rashomon.analysis.repeated_rewiring import summarize_realization_effects


def test_repeated_rewiring_uses_realizations_as_inference_units() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["cora"] * 6,
            "mode": ["local_entanglement"] * 6,
            "strength": [0.05] * 6,
            "target_homophily": [None] * 6,
            "tau": [0.1] * 6,
            "delta_mean_probability_diameter": np.linspace(0.01, 0.03, 6),
            "delta_disagreement_fraction": np.linspace(0.02, 0.04, 6),
            "matched_treatment_control_delta_diameter": np.linspace(0.005, 0.015, 6),
        }
    )

    summary = summarize_realization_effects(frame, seed=4, bootstrap_samples=200)

    assert len(summary) == 3
    assert set(summary["realization_count"]) == {6}
    assert set(summary["t_reference_df"]) == {5}
    assert set(summary["inference_unit"]) == {
        "independently seeded rewired graph realization"
    }
    assert (summary["bootstrap_ci_low"] > 0).all()
    assert summary["t_bonferroni_p_value"].between(0, 1).all()
    assert (summary["positive_effect_count"] == 6).all()
    assert (summary["negative_effect_count"] == 0).all()
    assert summary["sign_test_bonferroni_p_value"].between(0, 1).all()


def test_multiplicity_correction_is_separate_by_effect_family() -> None:
    rows = []
    for effect_shift in [0.01, 0.02, 0.03]:
        rows.append(
            {
                "dataset": "cora",
                "mode": "random",
                "strength": effect_shift,
                "target_homophily": None,
                "tau": None,
                "delta_mean_probability_diameter": effect_shift,
                "delta_disagreement_fraction": effect_shift,
                "matched_treatment_control_delta_diameter": None,
            }
        )
    frame = pd.DataFrame(rows * 4)

    summary = summarize_realization_effects(frame, seed=3, bootstrap_samples=100)

    assert set(summary.groupby("effect").size()) == {3}
