from __future__ import annotations

import numpy as np

from crackle.data.features import update_history_trigger


def hawkes_history_by_step(labels: dict[str, np.ndarray], *, decay: float = 0.85) -> list[np.ndarray]:
    bonds = labels["bonds"].astype(np.int64)
    alive = labels["bond_alive"].astype(bool)
    history = np.zeros((bonds.shape[0],), dtype=np.float64)
    out: list[np.ndarray] = []
    for step in range(alive.shape[0] - 1):
        out.append(history.copy())
        events = alive[step] & ~alive[step + 1]
        history = update_history_trigger(history, events, bonds, decay=decay)
    return out
