from __future__ import annotations

import random


def degree_preserving_swap(
    edges: set[tuple[int, int]],
    seed: int,
    attempted_swaps: int,
    *,
    stop_at_accepted: int | None = None,
) -> tuple[set[tuple[int, int]], int]:
    rng = random.Random(seed)
    current = set(edges)
    accepted = 0
    edge_list = list(current)
    for _ in range(attempted_swaps):
        if stop_at_accepted is not None and accepted >= stop_at_accepted:
            break
        (a, b), (c, d) = rng.sample(edge_list, 2)
        if len({a, b, c, d}) < 4:
            continue
        proposal = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if proposal[0][0] == proposal[0][1] or proposal[1][0] == proposal[1][1]:
            continue
        if proposal[0] in current or proposal[1] in current:
            continue
        current.remove((a, b))
        current.remove((c, d))
        current.add(proposal[0])
        current.add(proposal[1])
        edge_list = list(current)
        accepted += 1
    return current, accepted
