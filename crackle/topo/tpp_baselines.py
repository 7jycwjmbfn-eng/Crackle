"""Published neural temporal-point-process baselines for Track A.

These let Track A benchmark its transformer intensity model against the
field's standard TPP history encoders rather than only the parametric
Hawkes referee. Each baseline reuses the SAME marks/intensity heads, the
SAME query mechanism, the SAME discrete marked-Poisson likelihood and
metrics as crackle.topo.ntpp.DiscreteTHP — only the history ENCODER
changes, so the comparison isolates the encoder:

  parametric Hawkes (classical exp kernel)
    < RMTPP  — RNN/GRU history       (Du et al., KDD 2016)
    < NHP    — continuous-time LSTM  (Mei & Eisner, NeurIPS 2017)
    < THP    — self-attention        (ours; Zuo et al.-style, ICML 2020)

Faithful reimplementations adapted to our discrete, co-occurring,
doubly-marked (kind x tile) event stream under the shared likelihood — not
the official single-mark continuous-time repos, which do not apply directly
here. The report states this.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from crackle.topo.ntpp import DiscreteTHP, NTPPConfig


def _step_gaps(ev_step: torch.Tensor) -> torch.Tensor:
    """Non-negative gap to the previous event (0 at the first slot)."""
    gaps = torch.zeros_like(ev_step, dtype=torch.float32)
    gaps[:, 1:] = (ev_step[:, 1:] - ev_step[:, :-1]).clamp(min=0).float()
    return gaps


class RMTPPModel(DiscreteTHP):
    """Recurrent Marked TPP (Du et al., KDD 2016): a GRU over the event
    history, with an explicit time-gap input feature (RMTPP conditions its
    intensity on h and dt). Recurrent Markovian memory vs the transformer."""

    def __init__(self, config: NTPPConfig):
        super().__init__(config)
        d = config.d_model
        self.gru = nn.GRU(d + 1, d, num_layers=config.n_layers,
                          batch_first=True,
                          dropout=config.dropout if config.n_layers > 1 else 0.0)
        self.encoder = None  # disable the inherited transformer

    def encode_history(self, ev_kind, ev_tile, ev_step, ev_mask):
        x = (self.kind_emb(ev_kind) + self.tile_emb(ev_tile)
             + self.step_emb(ev_step))
        gap = _step_gaps(ev_step) / float(self.config.max_steps)
        x = torch.cat([x, gap.unsqueeze(-1)], dim=-1)
        out, _ = self.gru(x)            # GRU is causal by construction
        return out * ev_mask.unsqueeze(-1).float()


class _CTLSTMCell(nn.Module):
    """Continuous-time LSTM cell (Mei & Eisner, NeurIPS 2017): the cell
    state decays exponentially toward a target between events; gates fire at
    each event. We decay by the integer step gap to each event."""

    def __init__(self, d_in: int, d: int):
        super().__init__()
        self.lin = nn.Linear(d_in + d, 7 * d)
        self.d = d

    def forward(self, x_seq: torch.Tensor, gaps: torch.Tensor) -> torch.Tensor:
        b, e, _ = x_seq.shape
        d = self.d
        c = x_seq.new_zeros(b, d)
        c_bar = x_seq.new_zeros(b, d)
        o = x_seq.new_zeros(b, d)
        delta = x_seq.new_zeros(b, d)
        outs = []
        for t in range(e):
            decay = torch.exp(-delta * gaps[:, t:t + 1])
            c = c_bar + (c - c_bar) * decay
            h = o * torch.tanh(c)
            z = self.lin(torch.cat([x_seq[:, t], h], dim=-1))
            i, f, z_g, o_g, i_bar, f_bar, dlt = z.chunk(7, dim=-1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o_g)
            i_bar, f_bar = torch.sigmoid(i_bar), torch.sigmoid(f_bar)
            z_g = torch.tanh(z_g)
            c = f * c + i * z_g
            c_bar = f_bar * c_bar + i_bar * z_g
            delta = nn.functional.softplus(dlt)
            outs.append(o * torch.tanh(c))
        return torch.stack(outs, dim=1)


class NHPModel(DiscreteTHP):
    """Neural Hawkes Process (Mei & Eisner, NeurIPS 2017): a continuous-time
    LSTM whose memory decays between events — a self-modulating,
    history-decaying intensity. Defining feature vs RMTPP is the explicit
    exponential cell-state decay with the inter-event gap."""

    def __init__(self, config: NTPPConfig):
        super().__init__(config)
        self.ctlstm = _CTLSTMCell(config.d_model, config.d_model)
        self.encoder = None

    def encode_history(self, ev_kind, ev_tile, ev_step, ev_mask):
        x = (self.kind_emb(ev_kind) + self.tile_emb(ev_tile)
             + self.step_emb(ev_step))
        gaps = _step_gaps(ev_step) / float(self.config.max_steps)
        out = self.ctlstm(x, gaps)
        return out * ev_mask.unsqueeze(-1).float()


BASELINES = {"rmtpp": RMTPPModel, "nhp": NHPModel, "thp": DiscreteTHP}
