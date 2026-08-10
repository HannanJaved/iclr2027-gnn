from collections import Counter
from types import SimpleNamespace

import numpy as np

from gnn_rashomon.rewiring.edge_swap import degree_preserving_swap
from gnn_rashomon.rewiring.local_entanglement import select_treatment_control_nodes
from gnn_rashomon.rewiring.validation import validate_rewiring_invariants


def _degree(edges):
    counts = Counter()
    for a, b in edges:
        counts[a] += 1
        counts[b] += 1
    return counts


def test_degree_preserving_swap_keeps_degree_sequence():
    edges = {(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)}
    rewired, _ = degree_preserving_swap(edges, seed=0, attempted_swaps=10)
    assert _degree(rewired) == _degree(edges)
    assert len(rewired) == len(edges)


def _graph(edge_index):
    return SimpleNamespace(
        num_nodes=3,
        edge_index=np.asarray(edge_index, dtype=int),
        x=np.asarray([[1.0], [2.0], [3.0]], dtype=float),
        y=np.asarray([0, 1, 0], dtype=int),
        train_mask=np.asarray([True, False, False]),
        val_mask=np.asarray([False, True, False]),
        test_mask=np.asarray([False, False, True]),
    )


def test_rewiring_invariant_validation_accepts_preserved_graph_attributes():
    original = _graph([[0, 1, 1, 2], [1, 0, 2, 1]])
    rewired = _graph([[0, 1, 1, 2], [1, 0, 2, 1]])

    report = validate_rewiring_invariants(
        original,
        rewired,
        original_sensitive_attributes={"group": np.asarray([0, 1, 0])},
        rewired_sensitive_attributes={"group": np.asarray([0, 1, 0])},
    )

    assert report.valid
    assert report.degree_preserved
    assert report.sensitive_attributes_preserved


def test_rewiring_invariant_validation_detects_feature_drift():
    original = _graph([[0, 1, 1, 2], [1, 0, 2, 1]])
    rewired = _graph([[0, 1, 1, 2], [1, 0, 2, 1]])
    rewired.x = np.asarray([[1.0], [2.0], [99.0]], dtype=float)

    report = validate_rewiring_invariants(original, rewired)

    assert not report.valid
    assert not report.features_preserved
    assert report.degree_preserved


def test_select_treatment_control_nodes_matches_count_and_excludes_overlap():
    edge_index = np.array(
        [
            [0, 1, 1, 2, 2, 0, 3, 4, 4, 5, 5, 3],
            [1, 0, 2, 1, 0, 2, 4, 3, 5, 4, 3, 5],
        ],
        dtype=int,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    treatment, control = select_treatment_control_nodes(
        edge_index=edge_index,
        labels=labels,
        num_nodes=6,
        num_classes=2,
        treatment_count=2,
        tau=0.1,
        seed=0,
        min_degree=2,
    )

    assert len(treatment) == 2
    assert len(control) == 2
    assert set(treatment).isdisjoint(set(control))
