from __future__ import annotations

from collections import deque

import numpy as np


def _undirected_adjacency(edge_index: np.ndarray, num_nodes: int) -> list[set[int]]:
    adjacency = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.T:
        src_i = int(src)
        dst_i = int(dst)
        if src_i == dst_i:
            continue
        adjacency[src_i].add(dst_i)
        adjacency[dst_i].add(src_i)
    return adjacency


def clustering_coefficient(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    adjacency = _undirected_adjacency(edge_index, num_nodes)
    values = np.zeros(num_nodes, dtype=float)
    for node, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        if degree < 2:
            continue
        links = 0
        neighbor_list = list(neighbors)
        for i, first in enumerate(neighbor_list):
            for second in neighbor_list[i + 1 :]:
                links += int(second in adjacency[first])
        values[node] = (2.0 * links) / (degree * (degree - 1))
    return values


def _communities_networkx(edge_index: np.ndarray, num_nodes: int, seed: int) -> np.ndarray:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from((int(src), int(dst)) for src, dst in edge_index.T if int(src) != int(dst))
    if hasattr(nx.community, "louvain_communities"):
        communities = nx.community.louvain_communities(graph, seed=seed)
    else:
        communities = nx.community.greedy_modularity_communities(graph)
    community_id = np.full(num_nodes, -1, dtype=int)
    for idx, community in enumerate(communities):
        for node in community:
            community_id[int(node)] = idx
    return community_id


def boundary_features(
    edge_index: np.ndarray,
    num_nodes: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    adjacency = _undirected_adjacency(edge_index, num_nodes)
    community_id = _communities_networkx(edge_index, num_nodes, seed)
    is_boundary = np.zeros(num_nodes, dtype=int)
    cross_fraction = np.zeros(num_nodes, dtype=float)
    for node, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        cross = sum(int(community_id[node] != community_id[neighbor]) for neighbor in neighbors)
        cross_fraction[node] = cross / len(neighbors)
        is_boundary[node] = int(cross > 0)

    distance = np.full(num_nodes, -1, dtype=int)
    queue: deque[int] = deque()
    for node in np.flatnonzero(is_boundary):
        distance[int(node)] = 0
        queue.append(int(node))
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if distance[neighbor] >= 0:
                continue
            distance[neighbor] = distance[node] + 1
            queue.append(neighbor)
    return {
        "community_id": community_id,
        "is_community_boundary": is_boundary,
        "cross_community_fraction": cross_fraction,
        "distance_to_community_boundary": distance,
    }
