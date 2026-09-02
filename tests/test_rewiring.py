from collections import Counter
from types import SimpleNamespace

import numpy as np

from gnn_rashomon.rewiring.edge_swap import degree_preserving_swap
from gnn_rashomon.rewiring.local_entanglement import select_treatment_control_nodes
from gnn_rashomon.rewiring.validation import validate_rewiring_invariants
from gnn_rashomon.structure.neighborhood_entropy import neighborhood_label_entropy


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


def test_select_treatment_control_nodes_decrease_direction_picks_high_entropy_nodes():
    # Two same-label 4-cycles (0-1-2-3 label 0, 4-5-6-7 label 1), plus two
    # cross-label edges (0-4, 1-5) that give nodes 0, 1, 4, 5 mixed-label
    # neighborhoods (nonzero entropy) while 2, 3, 6, 7 stay purely
    # within-label (zero entropy). "increase" direction would only ever
    # pick from the zero-entropy pool; "decrease" needs entropy headroom to
    # fall, so it must pick only from {0, 1, 4, 5}.
    within_label_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
    cross_label_edges = [(0, 4), (1, 5)]
    directed = within_label_edges + cross_label_edges
    directed += [(dst, src) for src, dst in directed]
    edge_index = np.array(directed, dtype=int).T
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)

    entropy = neighborhood_label_entropy(edge_index, labels, num_nodes=8, num_classes=2)
    high_entropy_nodes = {0, 1, 4, 5}
    assert all(entropy[node] > 0 for node in high_entropy_nodes)
    assert all(entropy[node] == 0 for node in {2, 3, 6, 7})

    treatment, control = select_treatment_control_nodes(
        edge_index=edge_index,
        labels=labels,
        num_nodes=8,
        num_classes=2,
        treatment_count=2,
        tau=0.1,
        seed=0,
        min_degree=2,
        direction="decrease",
    )

    assert len(treatment) == 2
    assert len(control) == 2
    assert set(treatment).isdisjoint(set(control))
    assert set(treatment.tolist()).issubset(high_entropy_nodes)


def test_select_treatment_control_nodes_rejects_unknown_direction():
    edge_index = np.array([[0, 1], [1, 0]], dtype=int)
    labels = np.array([0, 0], dtype=int)
    try:
        select_treatment_control_nodes(
            edge_index=edge_index,
            labels=labels,
            num_nodes=2,
            num_classes=1,
            treatment_count=1,
            tau=0.1,
            seed=0,
            direction="sideways",
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for an unknown direction.")
