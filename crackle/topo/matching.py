"""Frame-to-frame feature matching for topological event extraction.

Two methods over the same cost model:

- ``greedy``      : Phase-0 heuristic. Sort all pairwise costs ascending and
                    accept pairs first-come-first-served. Locally optimal
                    choices can force spurious born/died pairs (see tests).
- ``wasserstein`` : optimal assignment (Hungarian) on the SAME cost matrix,
                    augmented with per-feature unmatch costs. This is the
                    unbalanced optimal-transport construction: each feature
                    either matches a partner (cost = spatial distance +
                    value_weight * |birth difference|, gated at max_dist) or
                    stays unmatched (cost = max_dist). Globally optimal, so
                    swap-prone configurations resolve correctly.

Choice documented per spec 1.2: persim's wasserstein/bottleneck match in
(birth, death) diagram space and discard birth LOCATION entirely, which is
the wrong geometry for identity tracking of damage hotspots (two features
with similar persistence but far-apart locations would happily match).
We therefore implement the spec's fallback — Hungarian on the Phase-0 cost
matrix — and keep the cost model identical between methods so the matcher
is the only variable in greedy-vs-wasserstein comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

_INF = 1e9  # linear_sum_assignment rejects np.inf


def cost_matrix(
    a_yx: np.ndarray,
    a_val: np.ndarray,
    b_yx: np.ndarray,
    b_val: np.ndarray,
    *,
    max_dist: float,
    value_weight: float,
) -> np.ndarray:
    """Pairwise match cost; entries with spatial distance > max_dist are _INF."""
    d_sp = np.linalg.norm(
        a_yx[:, None, :].astype(np.float64) - b_yx[None, :, :].astype(np.float64),
        axis=2,
    )
    d_val = np.abs(a_val[:, None] - b_val[None, :])
    cost = d_sp + float(value_weight) * d_val
    cost[d_sp > float(max_dist)] = _INF
    return cost


def greedy_match(
    a_yx: np.ndarray,
    a_val: np.ndarray,
    b_yx: np.ndarray,
    b_val: np.ndarray,
    *,
    max_dist: float,
    value_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-0 greedy 1-1 matching. Returns (matched_a_mask, matched_b_mask)."""
    na, nb = a_yx.shape[0], b_yx.shape[0]
    matched_a = np.zeros((na,), dtype=bool)
    matched_b = np.zeros((nb,), dtype=bool)
    if na == 0 or nb == 0:
        return matched_a, matched_b
    cost = cost_matrix(
        a_yx, a_val, b_yx, b_val, max_dist=max_dist, value_weight=value_weight
    )
    flat = np.argsort(cost, axis=None)
    for idx in flat:
        i, j = np.unravel_index(idx, cost.shape)
        if cost[i, j] >= _INF:
            break
        if matched_a[i] or matched_b[j]:
            continue
        matched_a[i] = True
        matched_b[j] = True
    return matched_a, matched_b


def wasserstein_match(
    a_yx: np.ndarray,
    a_val: np.ndarray,
    b_yx: np.ndarray,
    b_val: np.ndarray,
    *,
    max_dist: float,
    value_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Optimal unbalanced assignment. Returns (matched_a_mask, matched_b_mask).

    Augmented square matrix of side (na + nb):

        [ pair cost (na x nb)      | unmatch a (diag, na x na) ]
        [ unmatch b (diag, nb x nb)| zeros     (nb x na)       ]

    A feature pays max_dist to stay unmatched, so a pair is preferred over
    two unmatches exactly when its cost is below 2 * max_dist.
    """
    na, nb = a_yx.shape[0], b_yx.shape[0]
    matched_a = np.zeros((na,), dtype=bool)
    matched_b = np.zeros((nb,), dtype=bool)
    if na == 0 or nb == 0:
        return matched_a, matched_b
    pair = cost_matrix(
        a_yx, a_val, b_yx, b_val, max_dist=max_dist, value_weight=value_weight
    )
    unmatch = float(max_dist)
    big = np.full((na + nb, na + nb), _INF, dtype=np.float64)
    big[:na, :nb] = pair
    big[:na, nb:] = _INF
    big[:na, nb:][np.diag_indices(na)] = unmatch
    big[na:, :nb][np.diag_indices(nb)] = unmatch
    big[na:, nb:] = 0.0
    rows, cols = linear_sum_assignment(big)
    for i, j in zip(rows, cols):
        if i < na and j < nb and big[i, j] < _INF:
            matched_a[i] = True
            matched_b[j] = True
    return matched_a, matched_b


def match_features(
    a_yx: np.ndarray,
    a_val: np.ndarray,
    b_yx: np.ndarray,
    b_val: np.ndarray,
    *,
    method: str = "wasserstein",
    max_dist: float,
    value_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "greedy":
        fn = greedy_match
    elif method == "wasserstein":
        fn = wasserstein_match
    else:
        raise ValueError(f"unknown matching method {method!r}")
    return fn(
        a_yx, a_val, b_yx, b_val, max_dist=max_dist, value_weight=value_weight
    )
