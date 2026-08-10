from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from gnn_rashomon.data.preprocessing import to_numpy


@dataclass(frozen=True)
class RewiringInvariantReport:
    degree_preserved: bool
    masks_preserved: bool
    labels_preserved: bool
    features_preserved: bool
    sensitive_attributes_preserved: bool
    node_count_preserved: bool
    node_order_preserved: bool

    @property
    def valid(self) -> bool:
        return all(asdict(self).values())

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def degree_vector(edge_index: object, num_nodes: int) -> np.ndarray:
    edges = to_numpy(edge_index).astype(int)
    return np.bincount(edges[0], minlength=num_nodes)


def _same_array(left: object, right: object) -> bool:
    return bool(np.array_equal(to_numpy(left), to_numpy(right)))


def _same_sensitive_attributes(
    original: dict[str, Any],
    rewired: dict[str, Any],
) -> bool:
    if set(original) != set(rewired):
        return False
    return all(_same_array(original[name], rewired[name]) for name in original)


def validate_rewiring_invariants(
    original_data: Any,
    rewired_data: Any,
    original_sensitive_attributes: dict[str, Any] | None = None,
    rewired_sensitive_attributes: dict[str, Any] | None = None,
) -> RewiringInvariantReport:
    original_sensitive_attributes = original_sensitive_attributes or {}
    rewired_sensitive_attributes = rewired_sensitive_attributes or {}
    original_nodes = int(original_data.num_nodes)
    rewired_nodes = int(rewired_data.num_nodes)
    node_count_preserved = original_nodes == rewired_nodes
    num_nodes = original_nodes if node_count_preserved else min(original_nodes, rewired_nodes)

    return RewiringInvariantReport(
        degree_preserved=bool(
            node_count_preserved
            and np.array_equal(
                degree_vector(original_data.edge_index, num_nodes),
                degree_vector(rewired_data.edge_index, num_nodes),
            )
        ),
        masks_preserved=bool(
            _same_array(original_data.train_mask, rewired_data.train_mask)
            and _same_array(original_data.val_mask, rewired_data.val_mask)
            and _same_array(original_data.test_mask, rewired_data.test_mask)
        ),
        labels_preserved=_same_array(original_data.y, rewired_data.y),
        features_preserved=_same_array(original_data.x, rewired_data.x),
        sensitive_attributes_preserved=_same_sensitive_attributes(
            original_sensitive_attributes,
            rewired_sensitive_attributes,
        ),
        node_count_preserved=node_count_preserved,
        node_order_preserved=bool(
            node_count_preserved
            and _same_array(original_data.y, rewired_data.y)
            and _same_array(original_data.x, rewired_data.x)
        ),
    )
