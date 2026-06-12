"""Track A — neural temporal point process over topological events.

Discrete-time marked spatio-temporal point process. Load steps are integer
ticks; per case the model factorizes the likelihood of the event stream as

    LL = sum_t Poisson(N_t | lambda_t)                       (count)
       + sum_events log p(kind | history, t)                 (mark: kind)
       + sum_events log p(tile | history, t)                 (mark: location)

lambda_t, p(kind), p(tile) are read from a causal transformer over the
event history (kind + tile + step encodings), conditioned additionally on
the global topological summary curves at step t-1 (frames <= t-1 only, so
forecasting step t is causal). This is the discrete-time analogue of a
transformer-Hawkes intensity model, matched to a world where events carry
integer timestamps and can co-occur.

The parametric referee (same likelihood decomposition, addendum v1.1 B):
exponential-kernel Hawkes lambda_t = mu + alpha * sum_i exp(-beta (t-t_i))
fit by MLE on train cases, with empirical train distributions for kind and
tile. A neural model that cannot beat this referee by more than seed std
has produced a negative result.

Goodness-of-fit: time-rescaling — intervals between consecutive events
rescaled by the cumulative intensity must be Exp(1); tested with
crackle.metrics.point_process.ks_exp_pvalue.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from crackle.topo.catalog import EVENT_KINDS


@dataclass
class NTPPConfig:
    n_tiles_y: int = 5
    n_tiles_x: int = 8
    tile: int = 6
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    n_curve_features: int = 6
    max_steps: int = 96
    lr: float = 1e-3
    epochs: int = 12
    batch_cases: int = 64

    @property
    def n_tiles(self) -> int:
        return self.n_tiles_y * self.n_tiles_x


class CaseTensors:
    """Pack one case's event stream into step-aligned tensors."""

    def __init__(self, steps: np.ndarray, kinds: np.ndarray, tiles: np.ndarray,
                 curves: np.ndarray, n_steps: int):
        self.steps = steps          # (E,) int event step, ascending
        self.kinds = kinds          # (E,) int in [0, 4)
        self.tiles = tiles          # (E,) int tile index
        self.curves = curves        # (T, C) float, curves[t] uses frames <= t
        self.n_steps = n_steps


def pack_case(events: "object", n_steps: int, curves: np.ndarray,
              config: NTPPConfig) -> CaseTensors:
    """events: DataFrame-like with step/kind/y/x columns (catalog rows)."""
    kind_index = {k: i for i, k in enumerate(EVENT_KINDS)}
    order = np.argsort(np.asarray(events["step"], dtype=np.int64), kind="stable")
    steps = np.asarray(events["step"], dtype=np.int64)[order]
    kinds = np.array([kind_index[k] for k in np.asarray(events["kind"])[order]],
                     dtype=np.int64)
    ty = np.asarray(events["y"], dtype=np.int64)[order] // config.tile
    tx = np.asarray(events["x"], dtype=np.int64)[order] // config.tile
    ty = np.clip(ty, 0, config.n_tiles_y - 1)
    tx = np.clip(tx, 0, config.n_tiles_x - 1)
    return CaseTensors(steps, kinds, ty * config.n_tiles_x + tx, curves, n_steps)


class DiscreteTHP(nn.Module):
    def __init__(self, config: NTPPConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.kind_emb = nn.Embedding(len(EVENT_KINDS) + 1, d)  # +1 = BOS
        self.tile_emb = nn.Embedding(config.n_tiles + 1, d)
        self.step_emb = nn.Embedding(config.max_steps + 1, d)
        self.curve_proj = nn.Linear(config.n_curve_features, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.n_heads, dim_feedforward=4 * d,
            dropout=config.dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.head_intensity = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.head_kind = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(EVENT_KINDS)))
        self.head_tile = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, config.n_tiles))

    def encode_history(self, ev_kind: torch.Tensor, ev_tile: torch.Tensor,
                       ev_step: torch.Tensor, ev_mask: torch.Tensor
                       ) -> torch.Tensor:
        """(B, E) event tensors -> (B, E, d) causal encodings (BOS at 0)."""
        x = (self.kind_emb(ev_kind) + self.tile_emb(ev_tile)
             + self.step_emb(ev_step))
        attn_mask = nn.Transformer.generate_square_subsequent_mask(
            x.shape[1], device=x.device)
        return self.encoder(x, mask=attn_mask,
                            src_key_padding_mask=~ev_mask)

    def query(self, enc: torch.Tensor, ev_step: torch.Tensor,
              ev_mask: torch.Tensor, t: torch.Tensor,
              curves_tm1: torch.Tensor) -> torch.Tensor:
        """Context for predicting step t: last event encoding with step < t,
        concatenated with the projected curves at t-1. t: (B,)."""
        # index of last event with step < t (BOS counts, step 0 sentinel)
        usable = (ev_step < t[:, None]) & ev_mask
        idx = usable.float().cumsum(dim=1).argmax(dim=1)
        last = enc[torch.arange(enc.shape[0], device=enc.device), idx]
        return torch.cat([last, self.curve_proj(curves_tm1)], dim=-1)

    def step_terms(self, context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        lam = nn.functional.softplus(self.head_intensity(context)).squeeze(-1)
        return lam, self.head_kind(context), self.head_tile(context)


def case_log_likelihood(
    model: DiscreteTHP, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Vectorized over (B cases x T steps). Returns per-batch LL components.

    batch tensors:
      ev_kind/ev_tile/ev_step (B, E+1) with BOS at slot 0 (step 0, ids = pad)
      ev_mask (B, E+1) bool; counts (B, T) float; curves (B, T, C);
      step_kind/step_tile (B, T, K) one-hot sums NOT used — see ev targets
      tgt_mask (B, E+1) bool: real events to score marks on
    """
    enc = model.encode_history(batch["ev_kind"], batch["ev_tile"],
                               batch["ev_step"], batch["ev_mask"])
    b, t_max = batch["counts"].shape
    device = enc.device
    ll_count = torch.zeros((), device=device)
    ll_kind = torch.zeros((), device=device)
    ll_tile = torch.zeros((), device=device)
    n_events = torch.zeros((), device=device)
    for t in range(1, t_max):
        tt = torch.full((b,), t, device=device, dtype=torch.long)
        context = model.query(enc, batch["ev_step"], batch["ev_mask"], tt,
                              batch["curves"][:, t - 1])
        lam, kind_logits, tile_logits = model.step_terms(context)
        n_t = batch["counts"][:, t]
        valid = batch["step_valid"][:, t]
        ll_count = ll_count + ((n_t * torch.log(lam + 1e-9) - lam
                                - torch.lgamma(n_t + 1.0)) * valid).sum()
        # mark terms: events at step t
        ev_here = (batch["ev_step"] == t) & batch["tgt_mask"]
        if ev_here.any():
            rows, cols = ev_here.nonzero(as_tuple=True)
            ll_kind = ll_kind + torch.log_softmax(kind_logits, -1)[
                rows, batch["ev_kind"][rows, cols]].sum()
            ll_tile = ll_tile + torch.log_softmax(tile_logits, -1)[
                rows, batch["ev_tile"][rows, cols]].sum()
            n_events = n_events + ev_here.sum()
    return {"count": ll_count, "kind": ll_kind, "tile": ll_tile,
            "n_events": n_events,
            "n_steps": batch["step_valid"][:, 1:].sum()}


def fit_parametric_hawkes(
    event_steps: list[np.ndarray], n_steps_list: list[int],
    *, grid_mu=np.geomspace(0.005, 2.0, 40),
    grid_alpha=np.geomspace(0.01, 2.0, 30),
    grid_beta=np.geomspace(0.05, 3.0, 25),
) -> tuple[float, float, float]:
    """MLE by grid search for discrete-step Poisson-Hawkes (3 params).

    lambda_t = mu + alpha * sum_{s_i < t} exp(-beta (t - s_i)); the
    discrete-time LL is sum_t [N_t log lambda_t - lambda_t]. Grid search is
    robust and exactly reproducible; the surface is smooth and 3D.
    """
    counts_list = []
    for steps, n_steps in zip(event_steps, n_steps_list):
        c = np.zeros(n_steps)
        np.add.at(c, steps[steps < n_steps], 1.0)
        counts_list.append(c)

    best = (-np.inf, 0.0, 0.0, 0.0)
    for beta in grid_beta:
        decay = np.exp(-beta)
        # kernel sums K_t per case for this beta, flattened across cases
        big_c, big_k = [], []
        for c in counts_list:
            k = np.zeros_like(c)
            acc = 0.0
            for t in range(1, c.size):
                acc = (acc + c[t - 1]) * decay
                k[t] = acc
            big_c.append(c[1:])
            big_k.append(k[1:])
        flat_c = np.concatenate(big_c)
        flat_k = np.concatenate(big_k)
        for mu in grid_mu:
            for alpha in grid_alpha:
                lam = mu + alpha * flat_k
                ll = float(np.sum(flat_c * np.log(lam + 1e-12) - lam))
                if ll > best[0]:
                    best = (ll, float(mu), float(alpha), float(beta))
    return best[1], best[2], best[3]


def hawkes_log_likelihood(
    steps: np.ndarray, n_steps: int, mu: float, alpha: float, beta: float
) -> tuple[float, np.ndarray]:
    """Count LL and per-step intensity for one case."""
    c = np.zeros(n_steps)
    np.add.at(c, steps[steps < n_steps], 1.0)
    lam = np.zeros(n_steps)
    acc, decay = 0.0, np.exp(-beta)
    for t in range(1, n_steps):
        acc = (acc + c[t - 1]) * decay
        lam[t] = mu + alpha * acc
    ll = float(np.sum(c[1:] * np.log(lam[1:] + 1e-12) - lam[1:]
                      - [float(np.sum(np.log(np.arange(1, int(n) + 1))))
                         if n > 1 else 0.0 for n in c[1:]]))
    return ll, lam


def rescaled_intervals(steps: np.ndarray, lam: np.ndarray) -> list[float]:
    """Time-rescaling with piecewise-constant intensity between events."""
    out = []
    prev = 0
    for s in steps:
        s = int(s)
        if s <= prev:
            continue
        out.append(float(lam[prev + 1 : s + 1].sum()))
        prev = s
    return out
