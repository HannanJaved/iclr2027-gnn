from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from gnn_rashomon.data.metadata import stable_hash
from gnn_rashomon.rewiring.random_rewire import (
    RewiringResult,
    _to_numpy,
    degree_sequence,
    edges_to_symmetric_edge_index,
    global_homophily,
    undirected_edges,
)
from gnn_rashomon.rewiring.validation import validate_rewiring_invariants
from gnn_rashomon.structure.feature_alignment import graph_feature_similarity


def _same_label_count(edges: set[tuple[int, int]], labels: np.ndarray) -> int:
    return sum(int(labels[src] == labels[dst]) for src, dst in edges)


def homophily_targeted_rewire_graph(
    graph: object,
    output_path: Path,
    target_homophily: float,
    tolerance: float,
    seed: int,
    max_attempts: int,
    max_feature_drift: float | None = None,
) -> RewiringResult:
    import torch

    rng = random.Random(seed)
    original_data = graph.data
    data = graph.data.clone()
    labels = _to_numpy(data.y).astype(int)
    features = _to_numpy(data.x).astype(float)
    original_edge_index = _to_numpy(data.edge_index).astype(int)
    original_edges = undirected_edges(original_edge_index)
    current_edges = set(original_edges)
    edge_list = list(current_edges)
    edge_count = len(current_edges)
    target_same = target_homophily * edge_count
    current_same = _same_label_count(current_edges, labels)
    accepted = 0

    original_feature_similarity = graph_feature_similarity(
        original_edge_index,
        features,
        int(data.num_nodes),
    )

    for _ in range(max_attempts):
        current_homophily = current_same / edge_count
        if abs(current_homophily - target_homophily) <= tolerance:
            break
        (a, b), (c, d) = rng.sample(edge_list, 2)
        if len({a, b, c, d}) < 4:
            continue
        proposed_edges = (tuple(sorted((a, d))), tuple(sorted((c, b))))
        if proposed_edges[0][0] == proposed_edges[0][1] or proposed_edges[1][0] == proposed_edges[1][1]:
            continue
        if proposed_edges[0] in current_edges or proposed_edges[1] in current_edges:
            continue

        removed_same = int(labels[a] == labels[b]) + int(labels[c] == labels[d])
        added_same = int(labels[proposed_edges[0][0]] == labels[proposed_edges[0][1]]) + int(
            labels[proposed_edges[1][0]] == labels[proposed_edges[1][1]]
        )
        proposed_same = current_same - removed_same + added_same
        if abs(proposed_same - target_same) >= abs(current_same - target_same):
            continue

        current_edges.remove((a, b))
        current_edges.remove((c, d))
        current_edges.add(proposed_edges[0])
        current_edges.add(proposed_edges[1])
        current_same = proposed_same
        edge_list = list(current_edges)
        accepted += 1

        if max_feature_drift is not None:
            edge_index = edges_to_symmetric_edge_index(current_edges)
            similarity = graph_feature_similarity(edge_index, features, int(data.num_nodes))
            if abs(similarity - original_feature_similarity) > max_feature_drift:
                current_edges.remove(proposed_edges[0])
                current_edges.remove(proposed_edges[1])
                current_edges.add((a, b))
                current_edges.add((c, d))
                current_same = current_same + removed_same - added_same
                edge_list = list(current_edges)
                accepted -= 1

    rewired_edge_index = edges_to_symmetric_edge_index(current_edges)
    original_degree = degree_sequence(original_edge_index, int(data.num_nodes))
    final_degree = degree_sequence(rewired_edge_index, int(data.num_nodes))
    final_homophily = global_homophily(rewired_edge_index, labels)
    final_feature_similarity = graph_feature_similarity(rewired_edge_index, features, int(data.num_nodes))

    data.edge_index = torch.as_tensor(rewired_edge_index, dtype=torch.long)
    invariants = validate_rewiring_invariants(
        original_data=original_data,
        rewired_data=data,
        original_sensitive_attributes=graph.sensitive_attributes,
        rewired_sensitive_attributes=graph.sensitive_attributes,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": data,
            "metadata": graph.metadata,
            "sensitive_attributes": graph.sensitive_attributes,
            "rewiring": {
                "mode": "homophily_targeted",
                "target_homophily": target_homophily,
                "tolerance": tolerance,
                "seed": seed,
                "max_attempts": max_attempts,
                "accepted_swaps": accepted,
                "max_feature_drift": max_feature_drift,
                "invariants": invariants.to_dict(),
            },
        },
        output_path,
    )

    return RewiringResult(
        original_graph_hash=stable_hash(original_edge_index),
        rewired_graph_path=str(output_path),
        mode="homophily_targeted",
        attempted_swaps=max_attempts,
        accepted_swaps=accepted,
        original_homophily=global_homophily(original_edge_index, labels),
        final_homophily=final_homophily,
        original_feature_similarity=original_feature_similarity,
        final_feature_similarity=final_feature_similarity,
        degree_preserved=bool(np.array_equal(original_degree, final_degree))
        and invariants.degree_preserved,
        masks_preserved=invariants.masks_preserved,
        labels_preserved=invariants.labels_preserved,
        features_preserved=invariants.features_preserved,
        sensitive_attributes_preserved=invariants.sensitive_attributes_preserved,
        node_count_preserved=invariants.node_count_preserved,
        node_order_preserved=invariants.node_order_preserved,
        target_homophily=target_homophily,
        tolerance=tolerance,
        target_reached=abs(final_homophily - target_homophily) <= tolerance,
    )
