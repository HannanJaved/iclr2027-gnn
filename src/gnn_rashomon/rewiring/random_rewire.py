from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gnn_rashomon.data.metadata import stable_hash
from gnn_rashomon.rewiring.edge_swap import degree_preserving_swap
from gnn_rashomon.rewiring.validation import validate_rewiring_invariants
from gnn_rashomon.structure.feature_alignment import graph_feature_similarity


@dataclass(frozen=True)
class RewiringResult:
    original_graph_hash: str
    rewired_graph_path: str
    mode: str
    attempted_swaps: int
    accepted_swaps: int
    original_homophily: float
    final_homophily: float
    original_feature_similarity: float
    final_feature_similarity: float
    degree_preserved: bool
    masks_preserved: bool
    labels_preserved: bool
    features_preserved: bool
    target_homophily: float | None = None
    tolerance: float | None = None
    target_reached: bool | None = None
    sensitive_attributes_preserved: bool = True
    node_count_preserved: bool = True
    node_order_preserved: bool = True
    tau: float | None = None
    delta_e: float | None = None
    treatment_count: int | None = None
    control_count: int | None = None
    treatment_nodes: list[int] | None = None
    control_nodes: list[int] | None = None
    original_treatment_entropy_mean: float | None = None
    final_treatment_entropy_mean: float | None = None
    delta_treatment_entropy_mean: float | None = None
    original_control_entropy_mean: float | None = None
    final_control_entropy_mean: float | None = None
    max_abs_control_entropy_delta: float | None = None
    rejected_structure: int | None = None
    homophily_tol: float | None = None
    entropy_tol: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def undirected_edges(edge_index: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for src, dst in edge_index.T:
        if src == dst:
            continue
        edges.add(tuple(sorted((int(src), int(dst)))))
    return edges


def edges_to_symmetric_edge_index(edges: set[tuple[int, int]]) -> np.ndarray:
    directed: list[tuple[int, int]] = []
    for src, dst in sorted(edges):
        directed.append((src, dst))
        directed.append((dst, src))
    return np.asarray(directed, dtype=np.int64).T


def global_homophily(edge_index: np.ndarray, labels: np.ndarray) -> float:
    edges = undirected_edges(edge_index)
    if not edges:
        return 0.0
    same = sum(float(labels[src] == labels[dst]) for src, dst in edges)
    return same / len(edges)


def degree_sequence(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    return np.bincount(edge_index[0], minlength=num_nodes)


def random_rewire_graph(
    graph: object,
    output_path: Path,
    strength: float,
    seed: int,
    *,
    target_accepted_swaps: int | None = None,
) -> RewiringResult:
    import torch

    original_data = graph.data
    data = graph.data.clone()
    labels = _to_numpy(data.y).astype(int)
    original_edge_index = _to_numpy(data.edge_index).astype(int)
    features = _to_numpy(data.x).astype(float)
    original_edges = undirected_edges(original_edge_index)
    if target_accepted_swaps is not None:
        # Generous attempt budget so acceptance can reach the matched count.
        attempted_swaps = max(target_accepted_swaps * 20, target_accepted_swaps + 100)
        mode_name = "random_matched_accepted"
    else:
        attempted_swaps = max(1, int(round(len(original_edges) * strength)))
        mode_name = "random"
    rewired_edges, accepted_swaps = degree_preserving_swap(
        original_edges,
        seed=seed,
        attempted_swaps=attempted_swaps,
        stop_at_accepted=target_accepted_swaps,
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
                "mode": mode_name,
                "strength": strength,
                "seed": seed,
                "attempted_swaps": attempted_swaps,
                "accepted_swaps": accepted_swaps,
                "target_accepted_swaps": target_accepted_swaps,
                "invariants": invariants.to_dict(),
            },
        },
        output_path,
    )

    return RewiringResult(
        original_graph_hash=stable_hash(original_edge_index),
        rewired_graph_path=str(output_path),
        mode=mode_name,
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
    )
