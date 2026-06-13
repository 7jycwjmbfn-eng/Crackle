"""Neural operators and traditional baselines for damage-field forecasting.

Phase 4. The task is autoregressive rollout of a SOLVED-peridynamics damage
field: given the field d_t (and the static toughness g), predict the
NON-NEGATIVE increment to d_{t+1}. Damage is monotone non-decreasing
(broken bonds stay broken), so the increment is relu-gated and the next
field is clip(d_t + increment, 0, 1). Persistence is exactly "increment =
0"; the operator's job is to predict WHERE the crack front advances next,
which traditional extrapolation cannot do — so the lead appears at long
rollout horizon.

Models (all map (d_t, g) -> increment field, except where noted):
- Persistence / Linear: parameter-free traditional extrapolation.
- PerPixelGRU: a GRU over each pixel's damage history independently — a
  temporal model with NO spatial coupling (the "traditional" neural
  baseline that lacks operator structure).
- FNO2d: Fourier Neural Operator (Li et al., 2021).
- ConvNet: a residual dilated CNN (a strong local operator; sharp crack
  fronts are high-frequency, where FNO's spectral bias can struggle).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- neural operators ----------------------------

class SpectralConv2d(nn.Module):
    """FNO spectral convolution: linear transform on the lowest Fourier modes."""

    def __init__(self, c_in: int, c_out: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (c_in * c_out)
        self.w1 = nn.Parameter(
            scale * torch.rand(c_in, c_out, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(
            scale * torch.rand(c_in, c_out, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2d(nn.Module):
    def __init__(self, c_in: int = 2, width: int = 32, modes: int = 12,
                 n_layers: int = 4):
        super().__init__()
        self.lift = nn.Conv2d(c_in, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.local = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                  nn.Conv2d(width, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for sp, lc in zip(self.spectral, self.local):
            h = h + F.gelu(sp(h) + lc(h))
        return self.proj(h).squeeze(1)  # (B, H, W) increment (pre-relu)


class ConvNet(nn.Module):
    """Residual dilated CNN operator — strong on sharp local crack fronts."""

    def __init__(self, c_in: int = 2, width: int = 48, n_blocks: int = 5):
        super().__init__()
        self.lift = nn.Conv2d(c_in, width, 3, padding=1)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            d = 2 ** (i % 3)
            self.blocks.append(nn.Sequential(
                nn.Conv2d(width, width, 3, padding=d, dilation=d), nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1)))
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                  nn.Conv2d(width, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.lift(x))
        for blk in self.blocks:
            h = h + F.gelu(blk(h))
        return self.proj(h).squeeze(1)


class DeepONet(nn.Module):
    """DeepONet (Lu et al., 2021) for field->field operator learning on a
    fixed grid. Branch net encodes the input function (d_t, g) to p latent
    coefficients; trunk net maps grid coordinates to p basis functions; the
    increment field is their inner product. This is the canonical neural
    operator alongside FNO — included so the neural-operator family is not
    represented by a single architecture (no cherry-picking a weak one)."""

    def __init__(self, c_in: int = 2, p: int = 64, branch_width: int = 48):
        super().__init__()
        self.p = p
        # branch: CNN encoder over the input field -> global latent (B, p)
        self.branch = nn.Sequential(
            nn.Conv2d(c_in, branch_width, 3, padding=1), nn.GELU(),
            nn.Conv2d(branch_width, branch_width, 3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(branch_width, branch_width, 3, padding=1, stride=2),
            nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.branch_head = nn.Linear(branch_width, p)
        # trunk: per-coordinate MLP -> (H*W, p)
        self.trunk = nn.Sequential(
            nn.Linear(2, 96), nn.GELU(), nn.Linear(96, 96), nn.GELU(),
            nn.Linear(96, p))
        self.bias = nn.Parameter(torch.zeros(1))
        self._coords_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _coords(self, h: int, w: int, device) -> torch.Tensor:
        key = (h, w)
        c = self._coords_cache.get(key)
        if c is None or c.device != device:
            ys = torch.linspace(0, 1, h, device=device)
            xs = torch.linspace(0, 1, w, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            c = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)
            self._coords_cache[key] = c
        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        coef = self.branch_head(self.branch(x).flatten(1))   # (B, p)
        basis = self.trunk(self._coords(h, w, x.device))      # (H*W, p)
        out = torch.einsum("bp,np->bn", coef, basis) + self.bias
        return out.reshape(b, h, w)


class PerPixelGRU(nn.Module):
    """GRU over each pixel's damage history, no spatial coupling. Operates on
    a (B, T, H, W) damage tensor + static g; predicts per-pixel increments.
    Here used in a 1-step form: hidden carried during rollout externally is
    avoided — instead it sees a short history window stacked as input."""

    def __init__(self, hist: int = 3, hidden: int = 16):
        super().__init__()
        self.hist = hist
        self.net = nn.Sequential(
            nn.Linear(hist + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, hist+1, H, W) — last channel is g, first `hist` are damage
        b, c, h, w = x.shape
        feat = x.permute(0, 2, 3, 1).reshape(-1, c)
        return self.net(feat).reshape(b, h, w)


class MLPPixel(nn.Module):
    """Pointwise (1x1) MLP: neural but with NO spatial coupling — isolates the
    value of operator structure. Maps (d_t, g) per pixel -> increment."""

    def __init__(self, c_in: int = 2, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


NEURAL = {"mlp_pixel": MLPPixel, "fno": FNO2d, "deeponet": DeepONet,
          "convnet": ConvNet}


# --------------------- crackle: topology-aware forecaster ------------------

class GraphForecaster(nn.Module):
    """The 'crackle' competitor for Phase 4: predicts the per-NODE damage
    increment by message passing on the native peridynamic BOND GRAPH, given
    the same information the grid operators get (current damage d_t and static
    toughness g). No rasterization — connectivity IS the representation, so a
    crack front advances along bonds, not across a Cartesian grid the operator
    has to resample onto. Static edge attributes (toughness, rest length)
    enter the message, so the model knows where bonds are weak.

    Batched over the S time-snapshots of ONE case at a time (the bond graph is
    constant within a case), so a whole movie is one forward pass.

    forward(node_x (S,N,Fn), edge_x (M,Fe), bonds (M,2)) -> increment (S,N).
    """

    def __init__(self, f_node: int = 2, f_edge: int = 2, d: int = 64,
                 rounds: int = 4):
        super().__init__()
        self.rounds = rounds
        self.node_in = nn.Linear(f_node, d)
        self.edge_in = nn.Linear(f_edge, d)
        self.msg = nn.ModuleList(
            [nn.Sequential(nn.Linear(3 * d, d), nn.GELU()) for _ in range(rounds)])
        self.node_upd = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * d, d), nn.GELU()) for _ in range(rounds)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, node_x: torch.Tensor, edge_x: torch.Tensor,
                bonds: torch.Tensor) -> torch.Tensor:
        s, n, _ = node_x.shape
        src, dst = bonds[:, 0], bonds[:, 1]
        h = self.node_in(node_x)                       # (S, N, d)
        e = self.edge_in(edge_x).unsqueeze(0).expand(s, -1, -1)  # (S, M, d)
        for r in range(self.rounds):
            hi, hj = h[:, src, :], h[:, dst, :]        # (S, M, d)
            m = self.msg[r](torch.cat([hi, hj, e], dim=-1))      # (S, M, d)
            agg = torch.zeros_like(h)
            agg.index_add_(1, src, m)
            agg.index_add_(1, dst, m)                  # undirected bonds
            h = h + self.node_upd[r](torch.cat([h, agg], dim=-1))
        return self.head(h).squeeze(-1)                # (S, N) pre-relu


# ----------------------------- rollout helpers -----------------------------

def neural_step(model: nn.Module, d_t: torch.Tensor, g: torch.Tensor,
                hist: torch.Tensor | None = None) -> torch.Tensor:
    """One forecast step: returns d_{t+1} = clip(d_t + relu(increment), 0, 1).

    hist (B, k, H, W) optional damage history for PerPixelGRU (else (d_t, g))."""
    if isinstance(model, PerPixelGRU):
        x = torch.cat([hist, g.unsqueeze(1)], dim=1)
    else:
        x = torch.stack([d_t, g], dim=1)
    inc = F.relu(model(x))
    return torch.clamp(d_t + inc, 0.0, 1.0)


def persistence_step(d_t: torch.Tensor) -> torch.Tensor:
    return d_t


def linear_step(d_t: torch.Tensor, d_prev: torch.Tensor) -> torch.Tensor:
    # monotone, clamped linear extrapolation of the last increment
    return torch.clamp(d_t + F.relu(d_t - d_prev), 0.0, 1.0)
