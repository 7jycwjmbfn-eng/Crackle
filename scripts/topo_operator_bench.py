"""Phase 4 - neural operator vs traditional damage-field forecasting.

PRE-REGISTERED CLAIM (fixed before any training run; see git history):
on held-out SOLVED-peridynamics cases, a spatial neural operator (FNO or
ConvNet) reduces LONG-horizon (h >= 20 steps) autoregressive rollout
relative-L2 AND topological bottleneck distance vs the BEST traditional
baseline (persistence / linear / mean-rate) by a large margin (target:
>= 40% reduction), and a pointwise neural net (mlp_pixel, no spatial
coupling) does NOT — isolating spatial operator structure as the cause.
Reported honestly either way; FNO's spectral bias may lose to ConvNet on
sharp crack fronts.

Why a large lead is structural, not tuned: damage is monotone; persistence
(frozen field) cannot predict crack-front advance, so its error grows with
horizon by construction. The question is whether the operator's error
stays bounded and whether it keeps the crack TOPOLOGY right (where
per-pixel error can be misleading).

Data: solved-PD damage movies + static toughness (gaussmoe archive).
Run on the kinematic proxy is intentionally NOT supported (its field is a
deterministic function of load -> forecasting is circular).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from crackle.data.common import write_json
from crackle.experiments.hetero_pinning import _git_commit
from crackle.topo.cubical import superlevel_persistence
from crackle.topo.io import load_case_npz


def _rasterize(points: np.ndarray, ncols: int, nrows: int):
    """Map a (subset of a) regular lattice to (row, col) indices. Solved-PD
    notches remove interior nodes, so the point set is irregular; we scatter
    onto the full lattice and leave removed cells at 0."""
    xs = np.unique(np.round(points[:, 0], 5))
    ys = np.unique(np.round(points[:, 1], 5))
    ix = np.searchsorted(xs, np.round(points[:, 0], 5))
    iy = np.searchsorted(ys, np.round(points[:, 1], 5))
    return iy, ix, ys.size, xs.size


def load_movies(dataset: Path, key: str = "damage"):
    """Returns list of (damage (T,ny,nx) f32, toughness (ny,nx) f32),
    rasterized onto the full lattice (notch cells = 0). Cases with differing
    lattice sizes are padded to the dataset max so a single grid model fits."""
    raws = []
    maxny = maxnx = 0
    sample_dir = dataset / "samples"
    for case in sorted(sample_dir.iterdir()):
        npz = case / "crack_labels.npz"
        if not npz.exists():
            continue
        data = dict(np.load(npz))
        pts = data["reference_x"]
        dmg = data[key].astype(np.float32)         # (T, N)
        n_nodes = pts.shape[0]
        if "material_toughness" in data and data["material_toughness"].size == n_nodes:
            g_nodes = data["material_toughness"].astype(np.float64)
        elif "material_toughness" in data and "bonds" in data:
            bonds = data["bonds"].astype(np.int64)
            tb = data["material_toughness"].astype(np.float64)
            acc = np.zeros(n_nodes); cnt = np.zeros(n_nodes)
            for c in (0, 1):
                np.add.at(acc, bonds[:, c], tb); np.add.at(cnt, bonds[:, c], 1.0)
            g_nodes = acc / np.maximum(cnt, 1.0)
        else:
            g_nodes = np.ones(n_nodes)
        iy, ix, ny, nx = _rasterize(pts, 0, 0)
        raws.append((dmg, g_nodes, iy, ix, ny, nx))
        maxny, maxnx = max(maxny, ny), max(maxnx, nx)

    out = []
    for dmg, g_nodes, iy, ix, ny, nx in raws:
        T = dmg.shape[0]
        dgrid = np.zeros((T, maxny, maxnx), dtype=np.float32)
        dgrid[:, iy, ix] = dmg
        ggrid = np.zeros((maxny, maxnx), dtype=np.float32)
        ggrid[iy, ix] = g_nodes
        out.append((dgrid, ggrid))
    return out


def _diag_sig(field: np.ndarray, sig_tau: float):
    d = superlevel_persistence(field, maxdim=1, connectivity="8")
    h0, h1 = d.split()
    h0s = h0.significant(sig_tau, include_essential=False)
    bd0 = np.column_stack([h0s.death, h0s.birth]) if h0s.dim.size else \
        np.zeros((0, 2))
    return bd0, int(h0s.dim.size)


def topo_fidelity(pred: np.ndarray, true: np.ndarray, sig_tau: float):
    from persim import bottleneck
    bd_p, b0_p = _diag_sig(pred, sig_tau)
    bd_t, b0_t = _diag_sig(true, sig_tau)
    try:
        bn = float(bottleneck(bd_p, bd_t))
    except Exception:
        bn = float("nan")
    return bn, abs(b0_p - b0_t)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key", type=str, default="damage")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--models", type=str, nargs="*",
                        default=["mlp_pixel", "fno", "deeponet", "convnet"])
    parser.add_argument("--t0-frac", type=float, default=0.25)
    parser.add_argument("--horizons", type=int, nargs="*",
                        default=[1, 3, 5, 10, 20, 30, 40])
    parser.add_argument("--sig-tau", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from crackle.operators import NEURAL, neural_step

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    movies = load_movies(args.dataset, key=args.key)
    print(f"{len(movies)} solved-PD cases; grid {movies[0][0].shape[1:]}; "
          f"device {device}", flush=True)
    rng0 = np.random.default_rng(0)
    idx = rng0.permutation(len(movies))
    n_test = max(1, int(args.test_frac * len(movies)))
    test_ids, train_ids = idx[:n_test], idx[n_test:]

    # toughness standardization on train
    g_all = np.stack([movies[i][1] for i in train_ids])
    g_mean, g_std = float(g_all.mean()), float(g_all.std() + 1e-6)

    def gten(g):
        return torch.as_tensor((g - g_mean) / g_std, device=device)

    # ---- training pairs (d_t, g) -> d_{t+1} from train cases ----
    def make_pairs(ids):
        dt, dn, gg = [], [], []
        for i in ids:
            d, g = movies[i]
            dt.append(d[:-1]); dn.append(d[1:])
            gg.append(np.broadcast_to(g, d[:-1].shape))
        return (np.concatenate(dt), np.concatenate(dn), np.concatenate(gg))

    tr_dt, tr_dn, tr_g = make_pairs(train_ids)
    tr_dt = torch.as_tensor(tr_dt, device=device)
    tr_dn = torch.as_tensor(tr_dn, device=device)
    tr_g = torch.as_tensor((tr_g - g_mean) / g_std, device=device)
    n_pairs = tr_dt.shape[0]
    print(f"train pairs {n_pairs}, test cases {n_test}", flush=True)

    def train_model(mkey, seed):
        torch.manual_seed(seed)
        model = NEURAL[mkey]().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        rng = np.random.default_rng(seed)
        bs = 64
        for ep in range(args.epochs):
            order = rng.permutation(n_pairs)
            for i in range(0, n_pairs, bs):
                b = order[i:i + bs]
                x = torch.stack([tr_dt[b], tr_g[b]], dim=1)
                inc = F.relu(model(x))
                pred = torch.clamp(tr_dt[b] + inc, 0.0, 1.0)
                loss = F.mse_loss(pred, tr_dn[b])
                opt.zero_grad(); loss.backward(); opt.step()
        return model

    # ---- rollout evaluation ----
    def eval_model(step_fn):
        per_h = {h: {"rel_l2": [], "bottleneck": [], "b0_err": []}
                 for h in args.horizons}
        for i in test_ids:
            d, g = movies[i]
            T = d.shape[0]
            t0 = int(args.t0_frac * T)
            hmax = max(args.horizons)
            if t0 + hmax >= T:
                hmax = T - t0 - 1
            preds = step_fn(d, g, t0, hmax)
            for h in args.horizons:
                if h > hmax:
                    continue
                p = preds[h - 1]; tru = d[t0 + h]
                denom = np.linalg.norm(tru) + 1e-8
                per_h[h]["rel_l2"].append(float(np.linalg.norm(p - tru) / denom))
                bn, b0e = topo_fidelity(p, tru, args.sig_tau)
                per_h[h]["bottleneck"].append(bn)
                per_h[h]["b0_err"].append(float(b0e))
        return {h: {k: float(np.nanmean(v)) if v else float("nan")
                    for k, v in d.items()} for h, d in per_h.items()}

    rows = []
    started = time.perf_counter()

    # traditional baselines (parameter-free / fit-free)
    def persistence_roll(d, g, t0, hmax):
        d0 = d[t0]
        return [d0.copy() for _ in range(hmax)]

    def linear_roll(d, g, t0, hmax):
        d0 = d[t0]
        rate = np.clip(d[t0] - d[t0 - 1], 0.0, None) if t0 > 0 else \
            np.zeros_like(d0)
        return [np.clip(d0 + (k + 1) * rate, 0.0, 1.0) for k in range(hmax)]

    inc_mean = (tr_dn - tr_dt).clamp(min=0).mean(dim=0).cpu().numpy()

    def meanrate_roll(d, g, t0, hmax):
        out = []; d_t = d[t0].copy()
        for _ in range(hmax):
            d_t = np.clip(d_t + inc_mean, 0.0, 1.0); out.append(d_t.copy())
        return out

    for name, fn in (("persistence", persistence_roll),
                     ("linear", linear_roll), ("mean_rate", meanrate_roll)):
        res = eval_model(fn)
        for h, m in res.items():
            rows.append({"model": name, "seed": -1, "horizon": h, **m})
        print(f"{name} done", flush=True)

    # neural operators
    for mkey in args.models:
        for seed in args.seeds:
            model = train_model(mkey, seed)
            model.eval()

            def step_fn(d, g, t0, hmax, _m=model):
                with torch.no_grad():
                    out = []
                    d_t = torch.as_tensor(d[t0], device=device)
                    gt = gten(g)
                    for _ in range(hmax):
                        x = torch.stack([d_t, gt], dim=0).unsqueeze(0)
                        inc = F.relu(_m(x)).squeeze(0)
                        d_t = torch.clamp(d_t + inc, 0.0, 1.0)
                        out.append(d_t.cpu().numpy())
                    return out
            res = eval_model(step_fn)
            for h, m in res.items():
                rows.append({"model": mkey, "seed": seed, "horizon": h, **m})
            print(f"{mkey} seed {seed} done ({time.perf_counter()-started:.0f}s)",
                  flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "operator_results.csv", index=False)
    write_json(args.out / "config.json", {
        "dataset": str(args.dataset), "n_cases": len(movies),
        "n_test": n_test, "horizons": args.horizons, "epochs": args.epochs,
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "wall_s": round(time.perf_counter() - started, 1)})
    print(json.dumps({"out": str(args.out), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
