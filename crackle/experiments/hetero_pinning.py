from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - CLI reports this cleanly.
    torch = None
    nn = None

from crackle.benchmarks.notched_plate import build_bond_graph, initial_notch_alive


CORR_LENGTHS_MM = {
    "small": 2.5,
    "medium": 7.5,
    "large": 15.0,
}


def _git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, timeout=5).strip()
    except Exception:
        return None


def _sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_grid(nx: int, ny: int, length: float, height: float) -> tuple[np.ndarray, float, float, np.ndarray]:
    xs = np.linspace(0.0, float(length), int(nx), dtype=np.float64)
    ys = np.linspace(-0.5 * float(height), 0.5 * float(height), int(ny), dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    dx = float(length) / max(int(nx) - 1, 1)
    dy = float(height) / max(int(ny) - 1, 1)
    dV = np.full((points.shape[0],), float(length) * float(height) / float(points.shape[0]), dtype=np.float64)
    if abs(float(np.sum(dV)) - float(length) * float(height)) > 1e-8:
        raise RuntimeError("H1 failed: volume weights do not sum to domain volume")
    return points, dx, dy, dV


def correlated_toughness_field(
    *,
    nx: int,
    ny: int,
    length: float,
    height: float,
    contrast: float,
    corr_length: float,
    seed: int,
) -> np.ndarray:
    if float(contrast) <= 1.0 + 1e-12:
        return np.ones((int(ny), int(nx)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    white = rng.normal(size=(int(ny), int(nx)))
    kx = 2.0 * np.pi * np.fft.fftfreq(int(nx), d=float(length) / max(int(nx), 1))
    ky = 2.0 * np.pi * np.fft.fftfreq(int(ny), d=float(height) / max(int(ny), 1))
    kkx, kky = np.meshgrid(kx, ky, indexing="xy")
    filt = np.exp(-0.5 * (kkx * kkx + kky * kky) * float(corr_length) ** 2)
    smooth = np.fft.ifft2(np.fft.fft2(white) * filt).real
    smooth = (smooth - float(np.mean(smooth))) / max(float(np.std(smooth)), 1e-12)
    lo, hi = np.percentile(smooth, [2.0, 98.0])
    unit = np.clip((smooth - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return (1.0 + (float(contrast) - 1.0) * unit).astype(np.float64)


def _grid_values_to_nodes(field: np.ndarray) -> np.ndarray:
    return np.asarray(field, dtype=np.float64).reshape(-1)


def _kinematic_positions(
    points: np.ndarray,
    *,
    load_fraction: float,
    length: float,
    height: float,
    notch_length: float,
    horizon: float,
    tension_strain: float,
    notch_opening_factor: float,
    poisson: float,
) -> np.ndarray:
    strain = float(tension_strain) * float(load_fraction)
    out = np.asarray(points, dtype=np.float64).copy()
    out[:, 0] += -float(poisson) * strain * (points[:, 0] - 0.5 * float(length))
    out[:, 1] += strain * points[:, 1]
    tip = np.array([float(notch_length), 0.0], dtype=np.float64)
    rel = points - tip[None, :]
    dist = np.linalg.norm(rel, axis=1)
    ahead = points[:, 0] >= float(notch_length) - 0.25 * float(horizon)
    opening = (
        strain
        * float(height)
        * float(notch_opening_factor)
        * np.exp(-dist / max(2.6 * float(horizon), 1e-9))
        * np.where(points[:, 1] >= 0.0, 1.0, -1.0)
        * ahead.astype(np.float64)
    )
    out[:, 1] += opening
    return out


def _bond_stretch(pos: np.ndarray, points: np.ndarray, bonds: np.ndarray, rest: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(pos[bonds[:, 1]] - pos[bonds[:, 0]], axis=1)
    return ((length - rest) / np.maximum(rest, 1e-12)).astype(np.float64)


def _damage_from_alive(bonds: np.ndarray, alive: np.ndarray, n_nodes: int) -> np.ndarray:
    incident = np.zeros((n_nodes,), dtype=np.float64)
    broken = np.zeros((n_nodes,), dtype=np.float64)
    if bonds.size:
        np.add.at(incident, bonds[:, 0], 1.0)
        np.add.at(incident, bonds[:, 1], 1.0)
        broken_bonds = ~alive.astype(bool)
        np.add.at(broken, bonds[:, 0], broken_bonds.astype(np.float64))
        np.add.at(broken, bonds[:, 1], broken_bonds.astype(np.float64))
    return broken / np.maximum(incident, 1.0)


def _frontier_nodes(bonds: np.ndarray, damage: np.ndarray) -> np.ndarray:
    damaged = np.asarray(damage) > 0.035
    frontier = np.zeros_like(damaged, dtype=bool)
    if bonds.size:
        edge = damaged[bonds[:, 0]] != damaged[bonds[:, 1]]
        if np.any(edge):
            frontier[np.asarray(bonds, dtype=np.int64)[edge].reshape(-1)] = True
    return frontier


def simulate_reference(
    *,
    points: np.ndarray,
    gc_nodes: np.ndarray,
    steps: int,
    length: float,
    height: float,
    horizon: float,
    notch_length: float,
    tension_strain: float,
    critical_stretch: float,
    notch_opening_factor: float,
    poisson: float,
) -> dict[str, Any]:
    bonds, rest = build_bond_graph(points, horizon)
    initial_alive = initial_notch_alive(points, bonds, notch_length, notch_y=0.0)
    gc_bond = 0.5 * (gc_nodes[bonds[:, 0]] + gc_nodes[bonds[:, 1]])
    critical = float(critical_stretch) * np.sqrt(gc_bond / max(float(np.mean(gc_nodes)), 1e-12))
    mid = 0.5 * (points[bonds[:, 0]] + points[bonds[:, 1]])
    tip = np.array([float(notch_length), 0.0])
    dist_tip = np.linalg.norm(mid - tip[None, :], axis=1)
    crosses = (points[bonds[:, 0], 1] * points[bonds[:, 1], 1]) <= 0.0
    ahead = mid[:, 0] >= float(notch_length) - float(horizon)
    critical *= 1.0 - 0.34 * np.exp(-dist_tip / max(2.8 * float(horizon), 1e-9)) * (crosses & ahead).astype(np.float64)
    critical = np.maximum(0.30 * float(critical_stretch), critical)

    alive = np.zeros((int(steps) + 1, bonds.shape[0]), dtype=bool)
    stretch = np.zeros((int(steps) + 1, bonds.shape[0]), dtype=np.float64)
    damage = np.zeros((int(steps) + 1, points.shape[0]), dtype=np.float64)
    alive_now = initial_alive.copy()
    started = time.perf_counter()
    for step in range(int(steps) + 1):
        pos = _kinematic_positions(
            points,
            load_fraction=float(step) / max(float(steps), 1.0),
            length=length,
            height=height,
            notch_length=notch_length,
            horizon=horizon,
            tension_strain=tension_strain,
            notch_opening_factor=notch_opening_factor,
            poisson=poisson,
        )
        st = _bond_stretch(pos, points, bonds, rest)
        stretch[step] = st
        if step > 0:
            alive_now = alive_now & ~(alive_now & (st > critical))
        alive[step] = alive_now
        damage[step] = _damage_from_alive(bonds, alive_now, points.shape[0])
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    return {
        "bonds": bonds,
        "rest_length": rest,
        "critical_stretch_bond": critical,
        "bond_alive": alive,
        "bond_stretch": stretch,
        "damage": damage,
        "initial_alive": initial_alive,
        "dense_rollout_ms": elapsed_ms,
        "dense_step_ms": elapsed_ms / max(int(steps), 1),
        "dense_eval_count": int((int(steps) + 1) * bonds.shape[0]),
    }


def simulate_active_rollout(
    *,
    points: np.ndarray,
    reference: dict[str, Any],
    steps: int,
    length: float,
    height: float,
    horizon: float,
    notch_length: float,
    tension_strain: float,
    notch_opening_factor: float,
    poisson: float,
    active_fraction: float,
) -> dict[str, Any]:
    bonds = reference["bonds"]
    rest = reference["rest_length"]
    critical = reference["critical_stretch_bond"]
    alive_now = reference["initial_alive"].copy()
    alive = np.zeros_like(reference["bond_alive"])
    damage = np.zeros_like(reference["damage"])
    started = time.perf_counter()
    compact_ms = 0.0
    force_ms = 0.0
    for step in range(int(steps) + 1):
        force_started = time.perf_counter()
        pos = _kinematic_positions(
            points,
            load_fraction=float(step) / max(float(steps), 1.0),
            length=length,
            height=height,
            notch_length=notch_length,
            horizon=horizon,
            tension_strain=tension_strain,
            notch_opening_factor=notch_opening_factor,
            poisson=poisson,
        )
        st = _bond_stretch(pos, points, bonds, rest)
        force_ms += 1000.0 * (time.perf_counter() - force_started)
        if step > 0:
            compact_started = time.perf_counter()
            prev_damage = damage[step - 1]
            frontier = _frontier_nodes(bonds, prev_damage)
            node_active = frontier.copy()
            if not np.any(node_active):
                tip = np.array([float(notch_length), 0.0], dtype=np.float64)
                node_active = np.linalg.norm(points - tip[None, :], axis=1) <= 2.0 * float(horizon)
            bond_active = alive_now & (node_active[bonds[:, 0]] | node_active[bonds[:, 1]])
            budget = max(1, int(float(active_fraction) * max(bonds.shape[0], 1)))
            score = st / np.maximum(critical, 1e-12)
            if np.count_nonzero(bond_active) < budget:
                extra = np.argpartition(score, -budget)[-budget:]
                bond_active[extra] = True
            break_now = bond_active & (score > 1.0)
            alive_now = alive_now & ~break_now
            compact_ms += 1000.0 * (time.perf_counter() - compact_started)
        alive[step] = alive_now
        damage[step] = _damage_from_alive(bonds, alive_now, points.shape[0])
    total_ms = 1000.0 * (time.perf_counter() - started)
    return {
        "bond_alive": alive,
        "damage": damage,
        "rollout_ms": total_ms,
        "total_step_ms": total_ms / max(int(steps), 1),
        "pd_force_ms": force_ms / max(int(steps), 1),
        "active_bond_compact_ms": compact_ms / max(int(steps), 1),
    }


if torch is not None:

    class HashGrid2D(nn.Module):
        def __init__(self, levels: int = 8, features: int = 2, table_size: int = 2**15, min_res: int = 8, max_res: int = 256):
            super().__init__()
            self.levels = int(levels)
            self.features = int(features)
            self.table_size = int(table_size)
            if self.levels == 1:
                resolutions = [min_res]
            else:
                growth = math.exp((math.log(max_res) - math.log(min_res)) / max(self.levels - 1, 1))
                resolutions = [int(math.floor(min_res * (growth**i))) for i in range(self.levels)]
            self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long), persistent=False)
            self.tables = nn.Parameter(torch.empty(self.levels, self.table_size, self.features))
            nn.init.uniform_(self.tables, -1e-4, 1e-4)

        def hash(self, ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
            return ((ix * 73856093) ^ (iy * 19349663)) % self.table_size

        def forward(self, xy01: torch.Tensor) -> torch.Tensor:
            xy01 = xy01.clamp(0.0, 1.0)
            outs = []
            for level, res in enumerate(self.resolutions.tolist()):
                scaled = xy01 * res
                base = torch.floor(scaled).long()
                frac = scaled - base.float()
                ix0, iy0 = base[:, 0], base[:, 1]
                ix1, iy1 = ix0 + 1, iy0 + 1
                wx, wy = frac[:, 0:1], frac[:, 1:2]
                table = self.tables[level]
                f00 = table[self.hash(ix0, iy0)]
                f10 = table[self.hash(ix1, iy0)]
                f01 = table[self.hash(ix0, iy1)]
                f11 = table[self.hash(ix1, iy1)]
                outs.append((1 - wx) * (1 - wy) * f00 + wx * (1 - wy) * f10 + (1 - wx) * wy * f01 + wx * wy * f11)
            return torch.cat(outs, dim=-1)

    class FourierSiren(nn.Module):
        def __init__(self, in_dim: int, hidden: int, layers: int, num_frequencies: int):
            super().__init__()
            self.hash = HashGrid2D(levels=8, features=2)
            self.omega = 15.0
            freqs = 2.0 ** torch.arange(int(num_frequencies), dtype=torch.float32)
            self.register_buffer("freqs", freqs)
            # X-DEM-INR-style worker: Fourier features + multires hash grid.
            # Hash is applied only to coordinates; Gc remains an explicit pointwise condition.
            feat_dim = in_dim + 2 * in_dim * int(num_frequencies) + self.hash.levels * self.hash.features
            self.linears = nn.ModuleList()
            last = feat_dim
            for index in range(int(layers)):
                layer = nn.Linear(last, int(hidden))
                with torch.no_grad():
                    bound = 1.0 / max(layer.in_features, 1) if index == 0 else math.sqrt(6.0 / max(layer.in_features, 1)) / self.omega
                    layer.weight.uniform_(-bound, bound)
                    layer.bias.uniform_(-bound, bound)
                self.linears.append(layer)
                last = int(hidden)
            final = nn.Linear(last, 3)
            with torch.no_grad():
                final.weight.uniform_(-1e-4, 1e-4)
                final.bias.zero_()
                final.bias[2] = -4.0
            self.out = final

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            parts = [x]
            for freq in self.freqs:
                z = math.pi * freq * x
                parts.append(torch.sin(z))
                parts.append(torch.cos(z))
            xy01 = 0.5 * (x[:, 0:2] + 1.0)
            h = torch.cat([*parts, self.hash(xy01)], dim=1)
            for layer in self.linears:
                h = torch.sin(self.omega * layer(h))
            return self.out(h)


def _torch_grad_scalar_sum(value: Any, xy: Any) -> Any:
    return torch.autograd.grad(value.sum(), xy, create_graph=True, retain_graph=True)[0]


def _smooth_notch_seed_torch(xy: Any, *, notch_length: float, horizon: float) -> Any:
    x = xy[:, 0]
    y = xy[:, 1]
    sx = max(float(horizon) * 0.35, 1e-6)
    sy = max(float(horizon) * 0.55, 1e-6)
    behind = torch.sigmoid((float(notch_length) - x) / sx)
    line = torch.exp(-(y / sy) ** 2)
    return torch.clamp(behind * line, 0.0, 1.0)


def _smooth_notch_opening_torch(
    xy: Any,
    *,
    length: float,
    height: float,
    notch_length: float,
    horizon: float,
    tension_strain: float,
    notch_opening_factor: float,
) -> Any:
    tip_x = torch.tensor(float(notch_length), dtype=xy.dtype, device=xy.device)
    rel_x = xy[:, 0] - tip_x
    rel_y = xy[:, 1]
    dist = torch.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-8)
    ahead = torch.sigmoid((xy[:, 0] - (float(notch_length) - 0.25 * float(horizon))) / max(0.30 * float(horizon), 1e-6))
    side = torch.tanh(rel_y / max(0.30 * float(horizon), 1e-6))
    return (
        float(tension_strain)
        * float(height)
        * float(notch_opening_factor)
        * torch.exp(-dist / max(2.6 * float(horizon), 1e-6))
        * side
        * ahead
    )


def train_global_dem(
    *,
    points: np.ndarray,
    dV: np.ndarray,
    gc_nodes: np.ndarray,
    variant: str,
    length: float,
    height: float,
    notch_length: float,
    horizon: float,
    tension_strain: float,
    notch_opening_factor: float,
    steps: int,
    hidden: int,
    layers: int,
    lr: float,
    device: str,
    seed: int,
    bc_weight: float,
    notch_weight: float,
    ell: float,
    fracture_scale: float,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for oneshot_global_DEM")
    torch.manual_seed(int(seed))
    dev = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    xy0 = torch.as_tensor(points, dtype=torch.float32, device=dev)
    volume = torch.as_tensor(dV, dtype=torch.float32, device=dev)
    gc = torch.as_tensor(gc_nodes, dtype=torch.float32, device=dev)
    gc_norm = (gc - gc.mean()) / torch.clamp(gc.std(), min=1e-6)
    xy_min = torch.tensor([0.0, -0.5 * float(height)], dtype=torch.float32, device=dev)
    xy_scale = torch.tensor([float(length), float(height)], dtype=torch.float32, device=dev)
    model = FourierSiren(in_dim=3, hidden=hidden, layers=layers, num_frequencies=5).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-6)
    y = xy0[:, 1]
    top = y > 0.5 * float(height) - 1e-4
    bottom = y < -0.5 * float(height) + 1e-4
    left = xy0[:, 0] < 1e-4
    boundary = top | bottom | left
    u_bc = torch.zeros((xy0.shape[0], 2), dtype=torch.float32, device=dev)
    final_strain = float(tension_strain)
    u_bc[:, 0] = -0.28 * final_strain * (xy0[:, 0] - 0.5 * float(length))
    u_bc[:, 1] = final_strain * xy0[:, 1]
    mu = torch.tensor(1.0, dtype=torch.float32, device=dev)
    lam = torch.tensor(1.0, dtype=torch.float32, device=dev)
    eps_reg = torch.tensor(1e-6, dtype=torch.float32, device=dev)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for step in range(int(steps)):
        optim.zero_grad(set_to_none=True)
        xy = xy0.detach().clone().requires_grad_(True)
        xy_norm = 2.0 * ((xy - xy_min) / xy_scale) - 1.0
        inp = torch.cat([xy_norm, gc_norm[:, None]], dim=1)
        out = model(inp)
        u_affine = torch.zeros((xy.shape[0], 2), dtype=torch.float32, device=dev)
        u_affine[:, 0] = -0.28 * final_strain * (xy[:, 0] - 0.5 * float(length))
        u_affine[:, 1] = final_strain * xy[:, 1]
        u_affine[:, 1] += _smooth_notch_opening_torch(
            xy,
            length=length,
            height=height,
            notch_length=notch_length,
            horizon=horizon,
            tension_strain=final_strain,
            notch_opening_factor=notch_opening_factor,
        )
        d_fixed = torch.clamp(xy[:, 0:1] / max(float(length), 1e-6), 0.0, 1.0)
        u = u_affine + d_fixed * 0.02 * float(height) * torch.tanh(out[:, 0:2])
        d_seed = _smooth_notch_seed_torch(xy, notch_length=notch_length, horizon=horizon)
        raw_d = out[:, 2]
        if variant == "G2":
            d = d_seed + (1.0 - d_seed) * torch.sigmoid(raw_d)
            notch_loss = torch.mean((d - torch.clamp(d, min=d_seed)) ** 2)
        else:
            d = torch.sigmoid(raw_d)
            notch_loss = torch.mean((d - d_seed).pow(2) * (0.15 + d_seed).pow(2))
        gux = _torch_grad_scalar_sum(u[:, 0], xy)
        guy = _torch_grad_scalar_sum(u[:, 1], xy)
        gd = _torch_grad_scalar_sum(d, xy)
        exx = gux[:, 0]
        eyy = guy[:, 1]
        exy = 0.5 * (gux[:, 1] + guy[:, 0])
        trace = exx + eyy
        elastic_density = 0.5 * lam * trace.pow(2) + mu * (exx.pow(2) + eyy.pow(2) + 2.0 * exy.pow(2))
        degradation = (1.0 - d).pow(2) + eps_reg
        surface_density = float(fracture_scale) * gc * (
            d.pow(2) / (2.0 * float(ell)) + 0.5 * float(ell) * torch.sum(gd * gd, dim=1)
        )
        energy = torch.sum(volume * (degradation * elastic_density + surface_density))
        bc_loss = torch.mean((u[boundary] - u_bc[boundary]).pow(2)) if bool(torch.any(boundary)) else torch.tensor(0.0, device=dev)
        loss = energy + float(bc_weight) * bc_loss + float(notch_weight) * notch_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optim.step()
        if step == 0 or step == int(steps) - 1 or (step + 1) % max(1, int(steps) // 5) == 0:
            history.append(
                {
                    "step": float(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "energy": float(energy.detach().cpu()),
                    "bc_loss": float(bc_loss.detach().cpu()),
                    "notch_loss": float(notch_loss.detach().cpu()),
                    "damage_mean": float(d.detach().mean().cpu()),
                }
            )
    train_ms = 1000.0 * (time.perf_counter() - started)
    with torch.enable_grad():
        xy = xy0.detach().clone().requires_grad_(True)
        xy_norm = 2.0 * ((xy - xy_min) / xy_scale) - 1.0
        inp = torch.cat([xy_norm, gc_norm[:, None]], dim=1)
        out = model(inp)
        u_affine = torch.zeros((xy.shape[0], 2), dtype=torch.float32, device=dev)
        u_affine[:, 0] = -0.28 * final_strain * (xy[:, 0] - 0.5 * float(length))
        u_affine[:, 1] = final_strain * xy[:, 1]
        u_affine[:, 1] += _smooth_notch_opening_torch(
            xy,
            length=length,
            height=height,
            notch_length=notch_length,
            horizon=horizon,
            tension_strain=final_strain,
            notch_opening_factor=notch_opening_factor,
        )
        d_fixed = torch.clamp(xy[:, 0:1] / max(float(length), 1e-6), 0.0, 1.0)
        u = u_affine + d_fixed * 0.02 * float(height) * torch.tanh(out[:, 0:2])
        d_seed = _smooth_notch_seed_torch(xy, notch_length=notch_length, horizon=horizon)
        raw_d = out[:, 2]
        d = d_seed + (1.0 - d_seed) * torch.sigmoid(raw_d) if variant == "G2" else torch.sigmoid(raw_d)
        gux = _torch_grad_scalar_sum(u[:, 0], xy)
        guy = _torch_grad_scalar_sum(u[:, 1], xy)
        gd = _torch_grad_scalar_sum(d, xy)
        exx = gux[:, 0]
        eyy = guy[:, 1]
        exy = 0.5 * (gux[:, 1] + guy[:, 0])
        trace = exx + eyy
        elastic_density = 0.5 * lam * trace.pow(2) + mu * (exx.pow(2) + eyy.pow(2) + 2.0 * exy.pow(2))
        degradation = (1.0 - d).pow(2) + eps_reg
        surface_density = float(fracture_scale) * gc * (
            d.pow(2) / (2.0 * float(ell)) + 0.5 * float(ell) * torch.sum(gd * gd, dim=1)
        )
        energy = torch.sum(volume * (degradation * elastic_density + surface_density))
    peak_mem = float(torch.cuda.max_memory_allocated(dev) / 1024.0**2) if dev.type == "cuda" else 0.0
    return {
        "variant": variant,
        "damage": d.detach().cpu().numpy().astype(np.float64),
        "u": u.detach().cpu().numpy().astype(np.float64),
        "energy": float(energy.detach().cpu()),
        "train_ms": float(train_ms),
        "gpu_peak_memory_mb": peak_mem,
        "device": str(dev),
        "history": history,
    }


def _mask_from_damage(
    points: np.ndarray,
    damage: np.ndarray,
    threshold: float,
    notch_length: float,
    horizon: float,
    *,
    topk: int | None = None,
) -> np.ndarray:
    values = np.asarray(damage, dtype=np.float64)
    initial = (points[:, 0] <= float(notch_length) + 0.2 * float(horizon)) & (np.abs(points[:, 1]) <= 0.35 * float(horizon))
    eligible = ~initial
    if topk is not None:
        k = max(0, min(int(topk), int(np.count_nonzero(eligible))))
        mask = np.zeros_like(values, dtype=bool)
        if k > 0:
            eligible_idx = np.flatnonzero(eligible)
            chosen_local = np.argpartition(values[eligible_idx], -k)[-k:]
            mask[eligible_idx[chosen_local]] = True
        return mask
    return (values > float(threshold)) & eligible


def _nearest_stats(points: np.ndarray, pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    pred = np.asarray(points, dtype=np.float64)[np.asarray(pred_mask, dtype=bool)]
    true = np.asarray(points, dtype=np.float64)[np.asarray(true_mask, dtype=bool)]
    if pred.size == 0 and true.size == 0:
        return {"hausdorff": 0.0, "rms": 0.0}
    span = float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    if pred.size == 0 or true.size == 0:
        return {"hausdorff": span, "rms": span}

    def mins(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        chunks = []
        for start in range(0, a.shape[0], 512):
            block = a[start : start + 512]
            d2 = np.sum((block[:, None, :] - b[None, :, :]) ** 2, axis=2)
            chunks.append(np.sqrt(np.min(d2, axis=1)))
        return np.concatenate(chunks, axis=0)

    ab = mins(pred, true)
    ba = mins(true, pred)
    all_d = np.concatenate([ab, ba], axis=0)
    return {"hausdorff": float(np.max(all_d)), "rms": float(np.sqrt(np.mean(all_d * all_d)))}


def _iou(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(true, dtype=bool)
    union = int(np.count_nonzero(p | t))
    return 1.0 if union == 0 else float(np.count_nonzero(p & t)) / float(union)


def _tip(points: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if pts.size == 0:
        return None
    return pts[int(np.argmax(pts[:, 0]))]


def _tip_error(points: np.ndarray, pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    p = _tip(points, pred_mask)
    t = _tip(points, true_mask)
    if p is None and t is None:
        return 0.0
    if p is None or t is None:
        return float(np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)))
    return float(np.linalg.norm(p - t))


def _reference_phase_energy(
    *,
    points: np.ndarray,
    dV: np.ndarray,
    gc_nodes: np.ndarray,
    damage: np.ndarray,
    length: float,
    height: float,
    tension_strain: float,
    ell: float,
    fracture_scale: float,
) -> float:
    # Discrete reference energy using the same phase-field density shape as DEM.
    # The displacement is the imposed affine load; damage gradient is finite-difference on grid.
    n = points.shape[0]
    unique_x = np.unique(points[:, 0])
    unique_y = np.unique(points[:, 1])
    nx = len(unique_x)
    ny = len(unique_y)
    d = np.asarray(damage, dtype=np.float64).reshape(ny, nx)
    dx = float(length) / max(nx - 1, 1)
    dy = float(height) / max(ny - 1, 1)
    gy, gx = np.gradient(d, dy, dx, edge_order=1)
    strain = float(tension_strain)
    exx = -0.28 * strain
    eyy = strain
    trace = exx + eyy
    elastic_density = 0.5 * trace * trace + (exx * exx + eyy * eyy)
    surface = float(fracture_scale) * np.asarray(gc_nodes).reshape(ny, nx) * (
        d * d / (2.0 * float(ell)) + 0.5 * float(ell) * (gx * gx + gy * gy)
    )
    total_density = ((1.0 - d) ** 2 + 1e-6) * elastic_density + surface
    return float(np.sum(np.asarray(dV).reshape(ny, nx) * total_density))


def evaluate_model(
    *,
    model_name: str,
    damage: np.ndarray,
    energy: float | None,
    train_ms: float,
    gpu_peak_memory_mb: float,
    points: np.ndarray,
    reference_damage: np.ndarray,
    reference_energy: float,
    contrast: float,
    corr_label: str,
    corr_length_mm: float,
    seed: int,
    notch_length: float,
    horizon: float,
    damage_threshold: float,
    run_path: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    true_mask = _mask_from_damage(points, reference_damage, damage_threshold, notch_length, horizon)
    core_k = int(np.count_nonzero(true_mask))
    pred_mask = _mask_from_damage(points, damage, damage_threshold, notch_length, horizon, topk=core_k)
    near = _nearest_stats(points, pred_mask, true_mask)
    row = {
        "model": model_name,
        "contrast": float(contrast),
        "corr_length": corr_label,
        "corr_length_mm": float(corr_length_mm),
        "seed": int(seed),
        "crack_path_iou": _iou(pred_mask, true_mask),
        "crack_path_hausdorff_mm": near["hausdorff"],
        "crack_path_rms_err_mm": near["rms"],
        "nucleation_location_err_mm": _tip_error(points, pred_mask, true_mask),
        "energy_gap_vs_reference": None if energy is None else (float(energy) - float(reference_energy)) / max(abs(float(reference_energy)), 1e-12),
        "train_or_rollout_ms": float(train_ms),
        "gpu_peak_memory_mb": float(gpu_peak_memory_mb),
        "run_path": str(run_path),
    }
    if extra:
        row.update(extra)
    return row


def _bootstrap_ci(values: list[float], seed: int = 123, n_boot: int = 400) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    means = np.zeros((int(n_boot),), dtype=np.float64)
    for i in range(int(n_boot)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), float(row["contrast"])), []).append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "crack_path_iou",
        "crack_path_hausdorff_mm",
        "crack_path_rms_err_mm",
        "nucleation_location_err_mm",
        "energy_gap_vs_reference",
        "train_or_rollout_ms",
        "gpu_peak_memory_mb",
    ]
    for (model, contrast), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        row: dict[str, Any] = {"model": model, "contrast": contrast, "n": len(items)}
        for metric in metrics:
            vals = [float(item[metric]) for item in items if item.get(metric) not in (None, "")]
            mean, lo, hi = _bootstrap_ci(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_lo"] = lo
            row[f"{metric}_ci95_hi"] = hi
        out.append(row)
    return out


def _svg_polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    if not points:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width:.2f}" stroke-linejoin="round" stroke-linecap="round" />'


def write_gap_svg(path: Path, aggregate: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({str(row["model"]) for row in aggregate})
    xs = sorted({float(row["contrast"]) for row in aggregate})
    ys = [float(row["crack_path_rms_err_mm_mean"]) for row in aggregate if np.isfinite(float(row["crack_path_rms_err_mm_mean"]))]
    ymax = max(ys + [1.0])
    width, height = 760, 430
    left, right, top, bottom = 70, 25, 30, 65
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - min(xs)) / max(max(xs) - min(xs), 1e-12) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - y / max(ymax, 1e-12)) * plot_h

    colors = {"G1_pure_global": "#d1495b", "G2_global_irreversible": "#00798c", "A_active_rollout": "#edae49"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" />',
        f'<text x="{width/2:.1f}" y="25" text-anchor="middle" font-family="Arial" font-size="16">gap_vs_contrast: RMS path error lower is better</text>',
        f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">heterogeneity contrast</text>',
        f'<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">path RMS error (mm)</text>',
    ]
    for x in xs:
        lines.append(f'<text x="{sx(x):.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-family="Arial" font-size="11">{x:g}</text>')
    for i in range(5):
        val = ymax * i / 4.0
        y = sy(val)
        lines.append(f'<line x1="{left-4}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd" />')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{val:.2f}</text>')
    for idx, model in enumerate(models):
        pts = []
        for x in xs:
            match = [row for row in aggregate if row["model"] == model and abs(float(row["contrast"]) - x) < 1e-9]
            if match:
                pts.append((sx(x), sy(float(match[0]["crack_path_rms_err_mm_mean"]))))
        lines.append(_svg_polyline(pts, colors.get(model, "#444"), width=2.4))
        for px, py in pts:
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.6" fill="{colors.get(model, "#444")}" />')
        lines.append(f'<rect x="{left + 18 + idx * 205}" y="{height-48}" width="12" height="12" fill="{colors.get(model, "#444")}" />')
        lines.append(f'<text x="{left + 34 + idx * 205}" y="{height-38}" font-family="Arial" font-size="12">{model}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_drift_svg(path: Path, drift_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not drift_rows:
        path.write_text("", encoding="utf-8")
        return
    by_step: dict[int, list[float]] = {}
    for row in drift_rows:
        by_step.setdefault(int(row["step"]), []).append(float(row["a_rollout_iou_error"]))
    steps = sorted(by_step)
    vals = [float(np.mean(by_step[s])) for s in steps]
    width, height = 720, 360
    left, right, top, bottom = 65, 25, 28, 55
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = max(vals + [1.0])

    def sx(step: int) -> float:
        return left + (step - min(steps)) / max(max(steps) - min(steps), 1) * plot_w

    def sy(val: float) -> float:
        return top + (1.0 - val / max(ymax, 1e-12)) * plot_h

    pts = [(sx(s), sy(float(np.mean(by_step[s])))) for s in steps]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-family="Arial" font-size="15">A active-rollout drift curve</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" />',
        _svg_polyline(pts, "#7a5195", width=2.5),
        f'<text x="{width/2:.1f}" y="{height-16}" text-anchor="middle" font-family="Arial" font-size="12">rollout step</text>',
        f'<text x="18" y="{height/2:.1f}" transform="rotate(-90 18 {height/2:.1f})" text-anchor="middle" font-family="Arial" font-size="12">1 - path IoU</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_divergence_svg(
    path: Path,
    *,
    points: np.ndarray,
    ref_damage: np.ndarray,
    g_damage: np.ndarray,
    nx: int,
    ny: int,
    length: float,
    height: float,
    notch_length: float,
    horizon: float,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = _mask_from_damage(points, ref_damage, threshold, notch_length, horizon)
    pred = _mask_from_damage(points, g_damage, threshold, notch_length, horizon, topk=int(np.count_nonzero(ref)))
    both = ref & pred
    miss = ref & ~pred
    extra = pred & ~ref
    width, svg_h = 760, 350
    margin = 25
    sx = (width - 2 * margin) / float(length)
    sy = (svg_h - 2 * margin) / float(height)
    cell_w = sx * float(length) / max(nx - 1, 1)
    cell_h = sy * float(height) / max(ny - 1, 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}" viewBox="0 0 {width} {svg_h}">',
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="380" y="18" text-anchor="middle" font-family="Arial" font-size="14">divergence map: green=match red=miss blue=extra</text>',
        f'<rect x="{margin}" y="{margin}" width="{width-2*margin}" height="{svg_h-2*margin}" fill="#fafafa" stroke="#ccc" />',
    ]
    for mask, color, opacity in [(both, "#2ca25f", 0.90), (miss, "#de2d26", 0.82), (extra, "#3182bd", 0.72)]:
        for x, y in points[mask]:
            px = margin + x * sx
            py = margin + (0.5 * float(height) - y) * sy
            lines.append(f'<rect x="{px-cell_w/2:.2f}" y="{py-cell_h/2:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{color}" opacity="{opacity}" />')
    notch_x = margin
    notch_y = margin + 0.5 * float(height) * sy
    lines.append(f'<line x1="{notch_x:.1f}" y1="{notch_y:.1f}" x2="{margin + notch_length * sx:.1f}" y2="{notch_y:.1f}" stroke="#111" stroke-width="2" />')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_case(args: argparse.Namespace, contrast: float, corr_label: str, seed: int, out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corr_length_mm = CORR_LENGTHS_MM[corr_label]
    points, _dx, _dy, dV = make_grid(args.nx, args.ny, args.length, args.height)
    gc_grid = correlated_toughness_field(
        nx=args.nx,
        ny=args.ny,
        length=args.length,
        height=args.height,
        contrast=contrast,
        corr_length=corr_length_mm,
        seed=args.seed + 1009 * int(seed) + int(100 * contrast),
    )
    gc_nodes = _grid_values_to_nodes(gc_grid)
    reference = simulate_reference(
        points=points,
        gc_nodes=gc_nodes,
        steps=args.pd_steps,
        length=args.length,
        height=args.height,
        horizon=args.horizon,
        notch_length=args.notch_length,
        tension_strain=args.tension_strain,
        critical_stretch=args.critical_stretch,
        notch_opening_factor=args.notch_opening_factor,
        poisson=args.poisson,
    )
    a_rollout = simulate_active_rollout(
        points=points,
        reference=reference,
        steps=args.pd_steps,
        length=args.length,
        height=args.height,
        horizon=args.horizon,
        notch_length=args.notch_length,
        tension_strain=args.tension_strain,
        notch_opening_factor=args.notch_opening_factor,
        poisson=args.poisson,
        active_fraction=args.active_fraction,
    )
    ref_final = reference["damage"][-1]
    reference_energy = _reference_phase_energy(
        points=points,
        dV=dV,
        gc_nodes=gc_nodes,
        damage=ref_final,
        length=args.length,
        height=args.height,
        tension_strain=args.tension_strain,
        ell=args.phase_ell,
        fracture_scale=args.fracture_scale,
    )
    case_id = f"contrast{contrast:g}_{corr_label}_seed{seed}"
    sample_dir = out_dir / "samples" / case_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows.append(
        evaluate_model(
            model_name="A_active_rollout",
            damage=a_rollout["damage"][-1],
            energy=None,
            train_ms=a_rollout["rollout_ms"],
            gpu_peak_memory_mb=0.0,
            points=points,
            reference_damage=ref_final,
            reference_energy=reference_energy,
            contrast=contrast,
            corr_label=corr_label,
            corr_length_mm=corr_length_mm,
            seed=seed,
            notch_length=args.notch_length,
            horizon=args.horizon,
            damage_threshold=args.damage_threshold,
            run_path=out_dir,
            extra={
                "teacher_dense_step_ms": reference["dense_step_ms"],
                "total_step_ms": a_rollout["total_step_ms"],
                "pd_force_ms": a_rollout["pd_force_ms"],
                "active_bond_compact_ms": a_rollout["active_bond_compact_ms"],
                "teacher_type": "dense_incremental_bond_break_proxy",
                "case_id": case_id,
            },
        )
    )
    dem_results: dict[str, dict[str, Any]] = {}
    for variant, model_name in [("G1", "G1_pure_global"), ("G2", "G2_global_irreversible")]:
        if args.skip_dem:
            continue
        result = train_global_dem(
            points=points,
            dV=dV,
            gc_nodes=gc_nodes,
            variant=variant,
            length=args.length,
            height=args.height,
            notch_length=args.notch_length,
            horizon=args.horizon,
            tension_strain=args.tension_strain,
            notch_opening_factor=args.notch_opening_factor,
            steps=args.dem_steps,
            hidden=args.hidden,
            layers=args.layers,
            lr=args.lr,
            device=args.device,
            seed=args.seed + 7919 * int(seed) + (0 if variant == "G1" else 1),
            bc_weight=args.bc_weight,
            notch_weight=args.notch_weight,
            ell=args.phase_ell,
            fracture_scale=args.fracture_scale,
        )
        dem_results[model_name] = result
        rows.append(
            evaluate_model(
                model_name=model_name,
                damage=result["damage"],
                energy=result["energy"],
                train_ms=result["train_ms"],
                gpu_peak_memory_mb=result["gpu_peak_memory_mb"],
                points=points,
                reference_damage=ref_final,
                reference_energy=reference_energy,
                contrast=contrast,
                corr_label=corr_label,
                corr_length_mm=corr_length_mm,
                seed=seed,
                notch_length=args.notch_length,
                horizon=args.horizon,
                damage_threshold=args.damage_threshold,
                run_path=out_dir,
                extra={
                    "teacher_dense_step_ms": reference["dense_step_ms"],
                    "device": result["device"],
                    "teacher_type": "dense_incremental_bond_break_proxy",
                    "case_id": case_id,
                },
            )
        )
    drift_rows: list[dict[str, Any]] = []
    for step in range(reference["damage"].shape[0]):
        ref_mask = _mask_from_damage(points, reference["damage"][step], args.damage_threshold, args.notch_length, args.horizon)
        a_mask = _mask_from_damage(
            points,
            a_rollout["damage"][step],
            args.damage_threshold,
            args.notch_length,
            args.horizon,
            topk=int(np.count_nonzero(ref_mask)),
        )
        drift_rows.append(
            {
                "case_id": case_id,
                "contrast": float(contrast),
                "corr_length": corr_label,
                "seed": int(seed),
                "step": int(step),
                "a_rollout_iou_error": 1.0 - _iou(a_mask, ref_mask),
            }
        )
    save_payload = {
        "points": points,
        "gc": gc_nodes,
        "reference_damage": reference["damage"],
        "a_rollout_damage": a_rollout["damage"],
    }
    for name, result in dem_results.items():
        save_payload[f"{name}_damage"] = result["damage"]
        save_payload[f"{name}_u"] = result["u"]
    np.savez_compressed(sample_dir / "case_arrays.npz", **save_payload)
    if dem_results:
        best_g = min(dem_results.items(), key=lambda kv: next(row["crack_path_rms_err_mm"] for row in rows if row["model"] == kv[0]))
        write_divergence_svg(
            out_dir / "figures" / f"divergence_{case_id}_{best_g[0]}.svg",
            points=points,
            ref_damage=ref_final,
            g_damage=best_g[1]["damage"],
            nx=args.nx,
            ny=args.ny,
            length=args.length,
            height=args.height,
            notch_length=args.notch_length,
            horizon=args.horizon,
            threshold=args.damage_threshold,
        )
    return rows, drift_rows


def gate_verdict(aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    verdict: dict[str, Any] = {
        "schema": "hetero_pinning_gate_v1",
        "G0_sanity": {"pass": False, "reason": "missing contrast=1.0 rows"},
        "G1_gap_vs_contrast": {"pass": False, "reason": "not evaluated"},
    }
    g_rows = [row for row in aggregate if str(row["model"]) in {"G1_pure_global", "G2_global_irreversible"}]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in g_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    g0_pass_any = False
    for model, rows in by_model.items():
        row1 = [row for row in rows if abs(float(row["contrast"]) - 1.0) < 1e-9]
        if row1:
            rms = float(row1[0]["crack_path_rms_err_mm_mean"])
            iou = float(row1[0]["crack_path_iou_mean"])
            if np.isfinite(rms) and rms <= 12.0 and iou >= 0.05:
                g0_pass_any = True
    verdict["G0_sanity"] = {
        "pass": bool(g0_pass_any),
        "criterion": "at least one global variant has contrast=1.0 RMS<=12mm and IoU>=0.05",
    }
    model_reads = {}
    for model, rows in by_model.items():
        ordered = sorted(rows, key=lambda r: float(r["contrast"]))
        contrasts = [float(r["contrast"]) for r in ordered]
        rms = [float(r["crack_path_rms_err_mm_mean"]) for r in ordered]
        if len(rms) >= 3:
            slope = float(np.polyfit(contrasts, rms, deg=1)[0])
            high = rms[-1]
            low = rms[0]
            model_reads[model] = {"slope": slope, "low_contrast_rms": low, "high_contrast_rms": high, "monotone_down_proxy": slope < 0.0 and high < low}
    verdict["G1_gap_vs_contrast"] = {
        "pass": bool(any(read["monotone_down_proxy"] for read in model_reads.values())),
        "criterion": "linear slope of RMS path error versus contrast is negative and high-contrast RMS < low-contrast RMS for any G variant",
        "model_reads": model_reads,
    }
    return verdict


def write_report(path: Path, *, rows: list[dict[str, Any]], aggregate: list[dict[str, Any]], gate: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Heterogeneity Pinning Rollout-Cut Experiment",
        "",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run path: `{args.out}`",
        "",
        "## Claim Boundary",
        "",
        "This run is a controlled measurement of the heterogeneity-pinning hypothesis. The marched reference is a dense incremental bond-break proxy, not a high-fidelity FEM/PD truth. The one-shot solver is a DEM-style phase-field coordinate network used as a measurement instrument, not a new solver claim.",
        "",
        "Hard guards implemented:",
        "",
        "- H1: energy uses explicit `sum(dV_i * density_i)` and checks `sum(dV)=V_total`.",
        "- H2: DEM strain and damage gradients are computed through autograd coordinates.",
        "- H3: `G_c(x)` enters coordinate-wise as input and in the surface energy.",
        "- H4: the notch seed uses smooth sigmoid/radial functions in the differentiable path.",
        "- H5: no FEM/PD anchor labels are used in DEM optimization.",
        "- H6: bbox/origin/unit/range are recorded in JSON outputs.",
        "",
        "Path extraction: reference crack core is thresholded by `damage_threshold`; each predicted field is compared using the same number of highest-damage non-initial nodes. This avoids rewarding diffuse all-field phase damage while keeping phase-field outputs comparable to bond-damage cores.",
        "",
        "Core implementation note: the one-shot field uses an X-DEM-INR-style hash/SIREN worker and an exact residual ansatz (`u = u_analytic + D_fixed * residual`) adapted from the prior X-DEM-INR assets. It is still a fracture measurement prototype, not a validated new fracture solver.",
        "",
        "## Gate Verdict",
        "",
        f"- G0 sanity: `{gate['G0_sanity']['pass']}`",
        f"- G1 gap_vs_contrast: `{gate['G1_gap_vs_contrast']['pass']}`",
        "",
        "## Aggregate Metrics",
        "",
        "| model | contrast | n | path IoU | RMS err mm | Hausdorff mm | energy gap | ms | peak GPU MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {model} | {contrast:g} | {n} | {iou:.4f} | {rms:.4f} | {haus:.4f} | {egap:.4f} | {ms:.1f} | {mem:.1f} |".format(
                model=row["model"],
                contrast=float(row["contrast"]),
                n=int(row["n"]),
                iou=float(row["crack_path_iou_mean"]),
                rms=float(row["crack_path_rms_err_mm_mean"]),
                haus=float(row["crack_path_hausdorff_mm_mean"]),
                egap=float(row["energy_gap_vs_reference_mean"]) if np.isfinite(float(row["energy_gap_vs_reference_mean"])) else float("nan"),
                ms=float(row["train_or_rollout_ms_mean"]),
                mem=float(row["gpu_peak_memory_mb_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `hetero_pinning_case_metrics.csv`",
            "- `hetero_pinning_aggregate.csv`",
            "- `hetero_pinning_gate_report.json`",
            "- `hetero_pinning_drift_curve.csv`",
            "- `figures/gap_vs_contrast.svg`",
            "- `figures/drift_curve.svg`",
            "- `figures/divergence_*.svg`",
            "",
            "## Interpretation Rule",
            "",
            "If G1/G2 path error decreases as contrast increases, the pinning hypothesis receives support. If it does not, the experiment still measures where path dependence survives strong heterogeneity.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled heterogeneity-pinning experiment for rollout-cut fracture.")
    parser.add_argument("--out", type=Path, default=Path("runs/hetero_pinning_20260606"))
    parser.add_argument("--contrasts", type=str, default="1.0,1.5,2.0,3.0,5.0")
    parser.add_argument("--corr-lengths", type=str, default="small,medium,large")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=29)
    parser.add_argument("--length", type=float, default=100.0)
    parser.add_argument("--height", type=float, default=40.0)
    parser.add_argument("--horizon", type=float, default=5.2)
    parser.add_argument("--notch-length", type=float, default=22.0)
    parser.add_argument("--pd-steps", type=int, default=80)
    parser.add_argument("--tension-strain", type=float, default=0.075)
    parser.add_argument("--critical-stretch", type=float, default=0.045)
    parser.add_argument("--notch-opening-factor", type=float, default=0.54)
    parser.add_argument("--poisson", type=float, default=0.28)
    parser.add_argument("--damage-threshold", type=float, default=0.5)
    parser.add_argument("--active-fraction", type=float, default=0.18)
    parser.add_argument("--dem-steps", type=int, default=180)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.2e-2)
    parser.add_argument("--bc-weight", type=float, default=55.0)
    parser.add_argument("--notch-weight", type=float, default=4.0)
    parser.add_argument("--phase-ell", type=float, default=2.8)
    parser.add_argument("--fracture-scale", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-dem", action="store_true")
    parser.add_argument("--max-cases", type=int, default=0, help="Debug limit; 0 means full matrix.")
    parser.add_argument("--smoke", action="store_true", help="Use a tiny matrix and fewer DEM steps.")
    return parser


def _json_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if torch is None and not args.skip_dem:
        raise RuntimeError("PyTorch is not importable. Set PYTHONPATH to the CUDA torch package or pass --skip-dem.")
    if args.smoke:
        args.contrasts = "1.0,3.0"
        args.corr_lengths = "small"
        args.num_seeds = min(args.num_seeds, 1)
        args.nx = min(args.nx, 28)
        args.ny = min(args.ny, 15)
        args.pd_steps = min(args.pd_steps, 28)
        args.dem_steps = min(args.dem_steps, 28)
        args.hidden = min(args.hidden, 48)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    contrasts = _parse_float_list(args.contrasts)
    corr_labels = _parse_str_list(args.corr_lengths)
    for label in corr_labels:
        if label not in CORR_LENGTHS_MM:
            raise ValueError(f"unknown corr length label: {label}")
    cases: list[tuple[float, str, int]] = []
    for contrast in contrasts:
        for corr_label in corr_labels:
            for seed in range(int(args.num_seeds)):
                cases.append((contrast, corr_label, seed))
    if int(args.max_cases) > 0:
        cases = cases[: int(args.max_cases)]
    all_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    for index, (contrast, corr_label, seed) in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] contrast={contrast:g} corr={corr_label} seed={seed}", flush=True)
        rows, drift = run_case(args, contrast, corr_label, seed, args.out)
        all_rows.extend(rows)
        drift_rows.extend(drift)
        _write_csv(args.out / "hetero_pinning_case_metrics.csv", all_rows)
        _write_csv(args.out / "hetero_pinning_drift_curve.csv", drift_rows)
    aggregate = aggregate_rows(all_rows)
    gate = gate_verdict(aggregate)
    _write_csv(args.out / "hetero_pinning_aggregate.csv", aggregate)
    _write_json(args.out / "hetero_pinning_gate_report.json", gate)
    write_gap_svg(args.out / "figures" / "gap_vs_contrast.svg", aggregate)
    write_drift_svg(args.out / "figures" / "drift_curve.svg", drift_rows)
    manifest = {
        "schema": "hetero_pinning_rollout_cut_v1",
        "created_at_unix": time.time(),
        "duration_s": round(time.time() - started, 3),
        "git_commit": _git_commit(Path.cwd()),
        "args": _json_args(args),
        "num_cases": len(cases),
        "num_metric_rows": len(all_rows),
        "teacher_type": "dense_incremental_bond_break_proxy",
        "path_extraction": "reference_threshold_then_prediction_topk_same_core_count",
        "coordinate_system": {
            "origin": [0.0, -0.5 * float(args.height)],
            "bbox_mm": [0.0, float(args.length), -0.5 * float(args.height), 0.5 * float(args.height)],
            "unit": "mm",
            "nx": int(args.nx),
            "ny": int(args.ny),
            "dV_sum": float(args.length) * float(args.height),
        },
        "guards": {
            "H1_volume_integral": "sum(dV_i * energy_density_i), checked by make_grid",
            "H2_autograd_coordinate_gradients": True,
            "H3_coordinate_wise_Gc": True,
            "H4_smooth_notch_seed": True,
            "H5_no_static_PD_anchor_loss": True,
            "H6_coordinate_audit_recorded": True,
        },
        "xdem_asset_reuse": {
            "core": "multires hash grid + sine worker + exact residual ansatz",
            "source_assets": [
                "x_dem_inr_2d_smoke.py",
                "x_dem_inr_evaluator.py",
                "x_dem_inr_p2_hybrid_trainer.py"
            ],
            "not_reused_yet": "full RAR scheduler and FEM hotspot anchor, because this fracture sweep uses synthetic heterogeneous Gc rather than FEM stress labels"
        },
        "gate": gate,
    }
    _write_json(args.out / "hetero_pinning_manifest.json", manifest)
    manifest["dataset_hash"] = _sha256_short(args.out / "hetero_pinning_manifest.json")
    _write_json(args.out / "hetero_pinning_manifest.json", manifest)
    write_report(args.out / "hetero_pinning_report.md", rows=all_rows, aggregate=aggregate, gate=gate, args=args)
    print(json.dumps({"out": str(args.out), "rows": len(all_rows), "duration_s": manifest["duration_s"], "gate": gate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
