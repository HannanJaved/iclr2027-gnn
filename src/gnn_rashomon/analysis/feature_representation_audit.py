from __future__ import annotations


def classify_feature_representation(
    raw_max_abs_error: float,
    normalized_max_abs_error: float,
    *,
    tolerance: float,
) -> str:
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative.")
    raw_passes = raw_max_abs_error <= tolerance
    normalized_passes = normalized_max_abs_error <= tolerance
    if raw_passes and normalized_passes:
        return "both"
    if raw_passes:
        return "raw_only"
    if normalized_passes:
        return "normalized_only"
    return "neither"
