"""Track C — message passing on the peridynamic bond graph.

Native-representation hazard model: nodes = material points, edges =
bonds. For a (case, t) snapshot, predict per-bond breaking within
horizon H, for all H simultaneously (one head per horizon).

Bond input features (raw, per at-risk bond):
  ratio (stretch_t / critical), gc_bond, rest/horizon, |orientation_y|,
  boundary distance (normalized).
Node input features (local state, computable without message passing):
  broken fraction of incident bonds, mean and max ratio over incident
  alive bonds.

The tabular referee gets THE SAME raw bond features PLUS engineered
neighborhood aggregates (endpoint broken fractions, endpoint max ratio),
so the GNN's only edge is learned multi-hop message passing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

BOND_FEATURES = ["ratio", "gc_bond", "rest_norm", "orient_y", "bdist"]
NODE_FEATURES = ["broken_frac", "mean_ratio", "max_ratio"]
REFEREE_EXTRA = ["ep_broken_mean", "ep_broken_max", "ep_ratio_max"]


@dataclass
class GraphSample:
    """One (case, t) snapshot restricted to at-risk (alive) bonds."""

    bonds: np.ndarray        # (B, 2) int64 node ids (all bonds incl. dead)
    alive: np.ndarray        # (B,) bool at-risk mask at t
    bond_x: np.ndarray       # (B, 5) float32 raw bond features (dead rows 0)
    node_x: np.ndarray       # (N, 3) float32 node features
    labels: np.ndarray       # (B, n_horizons) int8; -1 censored, only alive valid
    n_nodes: int


def build_node_features(bonds: np.ndarray, alive: np.ndarray,
                        ratio: np.ndarray, n_nodes: int) -> np.ndarray:
    deg = np.zeros(n_nodes)
    np.add.at(deg, bonds[:, 0], 1.0)
    np.add.at(deg, bonds[:, 1], 1.0)
    deg = np.maximum(deg, 1.0)
    broken = np.zeros(n_nodes)
    dead = ~alive
    np.add.at(broken, bonds[dead, 0], 1.0)
    np.add.at(broken, bonds[dead, 1], 1.0)
    mean_r = np.zeros(n_nodes)
    max_r = np.zeros(n_nodes)
    r_alive = np.where(alive, ratio, 0.0)
    np.add.at(mean_r, bonds[:, 0], r_alive)
    np.add.at(mean_r, bonds[:, 1], r_alive)
    np.maximum.at(max_r, bonds[:, 0], r_alive)
    np.maximum.at(max_r, bonds[:, 1], r_alive)
    return np.stack([broken / deg, mean_r / deg, max_r], axis=1
                    ).astype(np.float32)


def referee_features(sample: GraphSample) -> np.ndarray:
    """Raw bond features + engineered endpoint aggregates (at-risk rows)."""
    i, j = sample.bonds[:, 0], sample.bonds[:, 1]
    broken = sample.node_x[:, 0]
    maxr = sample.node_x[:, 2]
    extra = np.stack([
        0.5 * (broken[i] + broken[j]),
        np.maximum(broken[i], broken[j]),
        np.maximum(maxr[i], maxr[j]),
    ], axis=1).astype(np.float32)
    feats = np.concatenate([sample.bond_x, extra], axis=1)
    return feats[sample.alive]


class BondGNN(nn.Module):
    def __init__(self, n_horizons: int, d: int = 64, rounds: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.rounds = rounds
        self.edge_in = nn.Linear(len(BOND_FEATURES), d)
        self.node_in = nn.Linear(len(NODE_FEATURES), d)
        self.node_upd = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout))
             for _ in range(rounds)])
        self.edge_upd = nn.ModuleList(
            [nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout))
             for _ in range(rounds)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, n_horizons))

    def forward(self, bonds: torch.Tensor, alive: torch.Tensor,
                bond_x: torch.Tensor, node_x: torch.Tensor,
                n_nodes: int) -> torch.Tensor:
        """bonds (B,2) long; alive (B,) bool; returns logits (B, H)
        (rows for dead bonds are meaningless — mask downstream)."""
        e = self.edge_in(bond_x)
        h = self.node_in(node_x)
        live = alive.unsqueeze(-1).float()
        for r in range(self.rounds):
            agg = torch.zeros((n_nodes, e.shape[1]), device=e.device)
            agg.index_add_(0, bonds[:, 0], e * live)
            agg.index_add_(0, bonds[:, 1], e * live)
            h = h + self.node_upd[r](torch.cat([h, agg], dim=-1))
            e = e + self.edge_upd[r](
                torch.cat([e, h[bonds[:, 0]], h[bonds[:, 1]]], dim=-1))
        return self.head(e)
