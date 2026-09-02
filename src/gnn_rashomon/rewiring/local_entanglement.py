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
from gnn_rashomon.structure.neighborhood_entropy import neighborhood_label_entropy


def _entropy_for_nodes(
    edges: set[tuple[int, int]],
    labels: np.ndarray,
    nodes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    neighbors: dict[int, list[int]] = {int(node): [] for node in nodes}
    selected = set(neighbors)
    for src, dst in edges:
        if src in selected:
            neighbors[src].append(dst)
        if dst in selected:
            neighbors[dst].append(src)

    out = np.zeros(len(nodes), dtype=float)
    for idx, node in enumerate(nodes):
        values = neighbors[int(node)]
        if not values:
            continue
        counts = np.bincount(labels[values], minlength=num_classes).astype(float)
        probabilities = counts[counts > 0] / counts.sum()
        out[idx] = -float(np.sum(probabilities * np.log(probabilities)))
    return out


def select_treatment_control_nodes(
    edge_index: np.ndarray,
    labels: np.ndarray,
    num_nodes: int,
    num_classes: int,
    treatment_count: int,
    tau: float,
    seed: int,
    min_degree: int = 2,
    direction: str = "increase",
) -> tuple[np.ndarray, np.ndarray]:
    if direction not in {"increase", "decrease"}:
        raise ValueError(f"Unknown direction: {direction!r}")
    rng = np.random.default_rng(seed)
    degree = degree_sequence(edge_index, num_nodes)
    entropy = neighborhood_label_entropy(edge_index, labels, num_nodes, num_classes)
    max_entropy = float(np.log(num_classes))
    if direction == "increase":
        eligible = np.flatnonzero((degree >= min_degree) & (entropy <= max_entropy - tau))
    else:
        eligible = np.flatnonzero((degree >= min_degree) & (entropy >= tau))
    if eligible.size == 0:
        raise ValueError("No eligible treatment nodes have enough entropy headroom.")

    sort_key = entropy[eligible] if direction == "increase" else -entropy[eligible]
    ordered = eligible[np.argsort(sort_key, kind="stable")]
    if ordered.size > treatment_count:
        pool = ordered[: max(treatment_count * 4, treatment_count)]
        treatment = np.sort(rng.choice(pool, size=treatment_count, replace=False))
    else:
        treatment = np.sort(ordered)

    treatment_set = set(int(node) for node in treatment)
    controls: list[int] = []
    used = set(treatment_set)
    for node in treatment:
        candidates = np.flatnonzero((labels == labels[node]) & (degree >= min_degree))
        candidates = np.asarray([int(candidate) for candidate in candidates if int(candidate) not in used])
        if candidates.size == 0:
            candidates = np.asarray(
                [idx for idx in range(num_nodes) if idx not in used and degree[idx] >= min_degree],
                dtype=int,
            )
        if candidates.size == 0:
            break
        distances = np.abs(degree[candidates] - degree[node])
        choice = int(candidates[np.argmin(distances)])
        controls.append(choice)
        used.add(choice)

    if not controls:
        raise ValueError("Could not select matched control nodes.")
    return treatment, np.asarray(controls, dtype=int)


def local_entanglement_rewire_graph(
    graph: object,
    output_path: Path,
    tau: float,
    delta_e: float,
    max_feature_drift: float,
    seed: int,
    max_attempts: int,
    treatment_count: int,
    min_degree: int = 2,
    direction: str = "increase",
) -> RewiringResult:
    import torch

    if direction not in {"increase", "decrease"}:
        raise ValueError(f"Unknown direction: {direction!r}")

    rng = random.Random(seed)
    original_data = graph.data
    data = graph.data.clone()
    labels = _to_numpy(data.y).astype(int)
    features = _to_numpy(data.x).astype(float)
    original_edge_index = _to_numpy(data.edge_index).astype(int)
    num_nodes = int(data.num_nodes)
    num_classes = int(labels.max()) + 1
    original_edges = undirected_edges(original_edge_index)
    current_edges = set(original_edges)
    edge_list = list(current_edges)

    treatment, control = select_treatment_control_nodes(
        edge_index=original_edge_index,
        labels=labels,
        num_nodes=num_nodes,
        num_classes=num_classes,
        treatment_count=treatment_count,
        tau=tau,
        seed=seed,
        min_degree=min_degree,
        direction=direction,
    )
    original_treatment_entropy = _entropy_for_nodes(
        current_edges,
        labels,
        treatment,
        num_classes,
    )
    original_control_entropy = _entropy_for_nodes(current_edges, labels, control, num_classes)
    current_treatment_mean = float(original_treatment_entropy.mean())
    original_treatment_mean = current_treatment_mean
    original_control_mean = float(original_control_entropy.mean())
    original_feature_similarity = graph_feature_similarity(original_edge_index, features, num_nodes)
    accepted = 0

    for _ in range(max_attempts):
        progress = (
            current_treatment_mean - original_treatment_mean
            if direction == "increase"
            else original_treatment_mean - current_treatment_mean
        )
        if progress >= tau:
            break
        (a, b), (c, d) = rng.sample(edge_list, 2)
        if len({a, b, c, d}) < 4:
            continue
        proposed_edges = (tuple(sorted((a, d))), tuple(sorted((c, b))))
        if proposed_edges[0][0] == proposed_edges[0][1] or proposed_edges[1][0] == proposed_edges[1][1]:
            continue
        if proposed_edges[0] in current_edges or proposed_edges[1] in current_edges:
            continue

        candidate_edges = set(current_edges)
        candidate_edges.remove((a, b))
        candidate_edges.remove((c, d))
        candidate_edges.add(proposed_edges[0])
        candidate_edges.add(proposed_edges[1])

        treatment_entropy = _entropy_for_nodes(candidate_edges, labels, treatment, num_classes)
        treatment_mean = float(treatment_entropy.mean())
        if direction == "increase":
            if treatment_mean <= current_treatment_mean:
                continue
        else:
            if treatment_mean >= current_treatment_mean:
                continue

        control_entropy = _entropy_for_nodes(candidate_edges, labels, control, num_classes)
        if float(np.max(np.abs(control_entropy - original_control_entropy))) > delta_e:
            continue

        candidate_edge_index = edges_to_symmetric_edge_index(candidate_edges)
        feature_similarity = graph_feature_similarity(candidate_edge_index, features, num_nodes)
        if abs(feature_similarity - original_feature_similarity) > max_feature_drift:
            continue

        current_edges = candidate_edges
        edge_list = list(current_edges)
        current_treatment_mean = treatment_mean
        accepted += 1

    rewired_edge_index = edges_to_symmetric_edge_index(current_edges)
    final_treatment_entropy = _entropy_for_nodes(current_edges, labels, treatment, num_classes)
    final_control_entropy = _entropy_for_nodes(current_edges, labels, control, num_classes)
    final_treatment_mean = float(final_treatment_entropy.mean())
    final_control_mean = float(final_control_entropy.mean())
    final_feature_similarity = graph_feature_similarity(rewired_edge_index, features, num_nodes)
    original_degree = degree_sequence(original_edge_index, num_nodes)
    final_degree = degree_sequence(rewired_edge_index, num_nodes)

    data.edge_index = torch.as_tensor(rewired_edge_index, dtype=torch.long)
    invariants = validate_rewiring_invariants(
        original_data=original_data,
        rewired_data=data,
        original_sensitive_attributes=graph.sensitive_attributes,
        rewired_sensitive_attributes=graph.sensitive_attributes,
    )
    mode_name = "local_entanglement" if direction == "increase" else "local_entanglement_decrease"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "data": data,
            "metadata": graph.metadata,
            "sensitive_attributes": graph.sensitive_attributes,
            "rewiring": {
                "mode": mode_name,
                "direction": direction,
                "tau": tau,
                "delta_e": delta_e,
                "max_feature_drift": max_feature_drift,
                "seed": seed,
                "max_attempts": max_attempts,
                "accepted_swaps": accepted,
                "treatment_nodes": treatment.tolist(),
                "control_nodes": control.tolist(),
                "original_treatment_entropy_mean": original_treatment_mean,
                "final_treatment_entropy_mean": final_treatment_mean,
                "original_control_entropy_mean": original_control_mean,
                "final_control_entropy_mean": final_control_mean,
                "invariants": invariants.to_dict(),
            },
        },
        output_path,
    )

    final_progress = (
        final_treatment_mean - original_treatment_mean
        if direction == "increase"
        else original_treatment_mean - final_treatment_mean
    )

    result = RewiringResult(
        original_graph_hash=stable_hash(original_edge_index),
        rewired_graph_path=str(output_path),
        mode=mode_name,
        attempted_swaps=max_attempts,
        accepted_swaps=accepted,
        original_homophily=global_homophily(original_edge_index, labels),
        final_homophily=global_homophily(rewired_edge_index, labels),
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
        target_homophily=None,
        tolerance=None,
        target_reached=final_progress >= tau,
        tau=tau,
        delta_e=delta_e,
        direction=direction,
        treatment_count=int(len(treatment)),
        control_count=int(len(control)),
        treatment_nodes=treatment.tolist(),
        control_nodes=control.tolist(),
        original_treatment_entropy_mean=original_treatment_mean,
        final_treatment_entropy_mean=final_treatment_mean,
        delta_treatment_entropy_mean=final_treatment_mean - original_treatment_mean,
        original_control_entropy_mean=original_control_mean,
        final_control_entropy_mean=final_control_mean,
        max_abs_control_entropy_delta=float(
            np.max(np.abs(final_control_entropy - original_control_entropy))
        ),
    )
    return result
