from __future__ import annotations

import pytest

from gnn_rashomon.analysis.feature_representation_audit import (
    classify_feature_representation,
)


@pytest.mark.parametrize(
    ("raw_error", "normalized_error", "expected"),
    [
        (1e-7, 0.2, "raw_only"),
        (0.2, 1e-7, "normalized_only"),
        (1e-7, 2e-7, "both"),
        (0.2, 0.3, "neither"),
    ],
)
def test_classify_feature_representation(
    raw_error: float, normalized_error: float, expected: str
) -> None:
    assert (
        classify_feature_representation(raw_error, normalized_error, tolerance=1e-5)
        == expected
    )


def test_classification_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        classify_feature_representation(0.0, 0.0, tolerance=-1.0)
