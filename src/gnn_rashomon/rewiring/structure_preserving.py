from __future__ import annotations

"""Degree-preserving rewiring that holds local structure approximately fixed.

Same attempted-swap budget as random rewiring, but a swap is accepted only when
local homophily and neighborhood label entropy of the affected endpoints change
by at most the stated tolerances. This is the structure-preserving control for
the fixed-ensemble intervention experiments.
"""

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


def _adjacency(edges: set[tuple[int, int]], num_nodes: int) -> list[set[int]]:
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for src, dst in edges:
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return neighbors


def _local_homophily(neighbors: set[int], labels: np.ndarray, node: int) -> float:
    if not neighbors:
        return 0.0
    same = sum(1 for nbr in neighbors if labels[nbr] == labels[node])
    return same / len(neighbors)


def _local_entropy(neighbors: set[int], labels: np.ndarray, num_classes: int) -> float:
    if not neighbors:
        return 0.0
    counts = np.bincount(
        [int(labels[nbr]) for nbr in neighbors], minlength=num_classes
    ).astype(float)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _endpoint_stats(
    neighbors: list[set[int]],
    labels: np.ndarray,
    num_classes: int,
    nodes: list[int],
) -> dict[int, tuple[float, float]]:
    return {
        node: (
            _local_homophily(neighbors[node], labels, node),
            _local_entropy(neighbors[node], labels, num_classes),
        )
        for node in nodes
    }


def structure_preserving_swap(
    edges: set[tuple[int, int]],
    labels: np.ndarray,
    num_nodes: int,
    num_classes: int,
    *,
    seed: int,
    attempted_swaps: int,
    homophily_tol: float,
    entropy_tol: float,
) -> tuple[set[tuple[int, int]], int, int]:
    """Return (rewired_edges, accepted, rejected_structure)."""
    import random

    rng = random.Random(seed)
    current = set(edges)
    neighbors = _adjacency(current, num_nodes)
    edge_list = list(current)
    accepted = 0
    rejected_structure = 0
    for _ in range(attempted_swaps):
        if len(edge_list) < 2:
            break
        (a, b), (c, d) = rng.sample(edge_list, 2)
        if len({a, b, c, d}) < 4:
            continue
        new_ab = tuple(sorted((a, d)))
        new_cd = tuple(sorted((c, b)))
        if new_ab[0] == new_ab[1] or new_cd[0] == new_cd[1]:
            continue
        if new_ab in current or new_cd in current:
            continue

        affected = [a, b, c, d]
        before = _endpoint_stats(neighbors, labels, num_classes, affected)

        # Apply tentatively.
        current.remove((a, b))
        current.remove((c, d))
        current.add(new_ab)
        current.add(new_cd)
        neighbors[a].remove(b)
        neighbors[b].remove(a)
        neighbors[c].remove(d)
        neighbors[d].remove(c)
        neighbors[a].add(d)
        neighbors[d].add(a)
        neighbors[c].add(b)
        neighbors[b].add(c)

        after = _endpoint_stats(neighbors, labels, num_classes, affected)
        ok = True
        for node in affected:
            dh = abs(after[node][0] - before[node][0])
            de = abs(after[node][1] - before[node][1])
            if dh > homophily_tol + 1e-12 or de > entropy_tol + 1e-12:
                ok = False
                break

        if ok:
            edge_list = list(current)
            accepted += 1
        else:
            # Revert.
            current.remove(new_ab)
            current.remove(new_cd)
            current.add((a, b))
            current.add((c, d))
            neighbors[a].remove(d)
            neighbors[d].remove(a)
            neighbors[c].remove(b)
            neighbors[b].remove(c)
            neighbors[a].add(b)
            neighbors[b].add(a)
            neighbors[c].add(d)
            neighbors[d].add(c)
            rejected_structure += 1
    return current, accepted, rejected_structure


def structure_preserving_rewire_graph(
    graph: object,
    output_path: Path,
    strength: float,
    seed: int,
    *,
    homophily_tol: float = 0.0,
    entropy_tol: float = 0.0,
) -> RewiringResult:
    import torch

    original_data = graph.data
    data = graph.data.clone()
    labels = _to_numpy(data.y).astype(int)
    original_edge_index = _to_numpy(data.edge_index).astype(int)
    features = _to_numpy(data.x).astype(float)
    original_edges = undirected_edges(original_edge_index)
    attempted_swaps = max(1, int(round(len(original_edges) * strength)))
    num_classes = int(graph.metadata.num_classes)
    rewired_edges, accepted_swaps, rejected_structure = structure_preserving_swap(
        original_edges,
        labels,
        int(data.num_nodes),
        num_classes,
        seed=seed,
        attempted_swaps=attempted_swaps,
        homophily_tol=homophily_tol,
        entropy_tol=entropy_tol,
    )
    rewired_edge_index = edges_to_symmetric_edge_index(rewired_edges)
    original_degree = degree_sequence(original_edge_index, int(data.num_nodes))
    final_degree = degree_sequence(rewired_edge_index, int(data.num_nodes))

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
                "mode": "structure_preserving",
                "strength": strength,
                "seed": seed,
                "attempted_swaps": attempted_swaps,
                "accepted_swaps": accepted_swaps,
                "rejected_structure": rejected_structure,
                "homophily_tol": homophily_tol,
                "entropy_tol": entropy_tol,
                "invariants": invariants.to_dict(),
            },
        },
        output_path,
    )

    result = RewiringResult(
        original_graph_hash=stable_hash(original_edge_index),
        rewired_graph_path=str(output_path),
        mode="structure_preserving",
        attempted_swaps=attempted_swaps,
        accepted_swaps=accepted_swaps,
        original_homophily=global_homophily(original_edge_index, labels),
        final_homophily=global_homophily(rewired_edge_index, labels),
        original_feature_similarity=graph_feature_similarity(
            original_edge_index,
            features,
            int(data.num_nodes),
        ),
        final_feature_similarity=graph_feature_similarity(
            rewired_edge_index,
            features,
            int(data.num_nodes),
        ),
        degree_preserved=bool(np.array_equal(original_degree, final_degree))
        and invariants.degree_preserved,
        masks_preserved=invariants.masks_preserved,
        labels_preserved=invariants.labels_preserved,
        features_preserved=invariants.features_preserved,
        sensitive_attributes_preserved=invariants.sensitive_attributes_preserved,
        node_count_preserved=invariants.node_count_preserved,
        node_order_preserved=invariants.node_order_preserved,
        rejected_structure=rejected_structure,
        homophily_tol=homophily_tol,
        entropy_tol=entropy_tol,
    )
    return result
