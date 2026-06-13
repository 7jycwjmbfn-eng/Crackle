"""Phase 4 HEAD-TO-HEAD: crackle (bond-graph forecaster) vs neural operators
vs traditional extrapolation, on the SAME damage-field rollout task, the SAME
case split, and the SAME grid-space metrics (rel-L2 + topological bottleneck).

PRE-REGISTERED CLAIM (fixed before any training run here; see git history):
on held-out AND on genuinely-new (--eval-dataset) solved-peridynamics cases,
the crackle GraphForecaster cuts long-horizon (h>=20) autoregressive rollout
rel-L2 vs BOTH (a) the best traditional baseline (persistence/linear/
mean_rate) AND (b) the best neural operator (FNO/DeepONet/ConvNet), by more
than the across-seed std. If it does not beat both families, that is a
NEGATIVE result and is reported as such — no rerun-until-it-wins.

Fairness rules:
- every model gets the SAME inputs (current damage d_t + static toughness g);
  the operators see them rasterized to a grid, the graph sees them on the
  native bond graph. No model sees a future frame.
- normalization statistics (toughness, edge features) come from TRAIN ONLY.
- all learned models: identical epochs, optimizer, seeds.
- graph predictions are scattered back to the grid and scored with the
  identical rel-L2 + bottleneck code the operators are scored with.
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


# ------------------------------- data loading ------------------------------

def _node_toughness(data: dict, n_nodes: int, bonds: np.ndarray) -> np.ndarray:
    mt = data.get("material_toughness")
    if mt is None:
        return np.ones(n_nodes)
    mt = mt.astype(np.float64)
    if mt.size == n_nodes:
        return mt
    if mt.size == bonds.shape[0]:
        acc = np.zeros(n_nodes); cnt = np.zeros(n_nodes)
        for c in (0, 1):
            np.add.at(acc, bonds[:, c], mt); np.add.at(cnt, bonds[:, c], 1.0)
        return acc / np.maximum(cnt, 1.0)
    return np.ones(n_nodes)


def _edge_toughness(data: dict, n_nodes: int, bonds: np.ndarray,
                    g_node: np.ndarray) -> np.ndarray:
    mt = data.get("material_toughness")
    if mt is not None and mt.astype(np.float64).size == bonds.shape[0]:
        return mt.astype(np.float64)
    return 0.5 * (g_node[bonds[:, 0]] + g_node[bonds[:, 1]])


def load_cases(dataset: Path, key: str = "damage"):
    """Per case: native node arrays (graph) + rasterized grid (operators) +
    the node->cell map so graph predictions can be scored on the grid."""
    sample_dir = dataset / "samples"
    raws = []
    maxny = maxnx = 0
    for case in sorted(sample_dir.iterdir()):
        npz = case / "crack_labels.npz"
        if not npz.exists():
            continue
        data = dict(np.load(npz))
        pts = data["reference_x"].astype(np.float64)
        dmg = data[key].astype(np.float32)                 # (T, N)
        bonds = data["bonds"].astype(np.int64)
        n_nodes = pts.shape[0]
        g_node = _node_toughness(data, n_nodes, bonds)
        e_tough = _edge_toughness(data, n_nodes, bonds, g_node)
        rest = np.linalg.norm(pts[bonds[:, 0]] - pts[bonds[:, 1]], axis=1)
        # node -> grid cell
        xs = np.unique(np.round(pts[:, 0], 5)); ys = np.unique(np.round(pts[:, 1], 5))
        ix = np.searchsorted(xs, np.round(pts[:, 0], 5))
        iy = np.searchsorted(ys, np.round(pts[:, 1], 5))
        ny, nx = ys.size, xs.size
        maxny, maxnx = max(maxny, ny), max(maxnx, nx)
        raws.append(dict(dmg=dmg, bonds=bonds, g_node=g_node, e_tough=e_tough,
                         rest=rest, iy=iy, ix=ix, ny=ny, nx=nx))
    cases = []
    for r in raws:
        T = r["dmg"].shape[0]
        dgrid = np.zeros((T, maxny, maxnx), dtype=np.float32)
        dgrid[:, r["iy"], r["ix"]] = r["dmg"]
        ggrid = np.zeros((maxny, maxnx), dtype=np.float32)
        ggrid[r["iy"], r["ix"]] = r["g_node"]
        r.update(dgrid=dgrid, ggrid=ggrid, maxny=maxny, maxnx=maxnx)
        cases.append(r)
    return cases


# ------------------------------- topo metric -------------------------------

def _diag_sig(field: np.ndarray, sig_tau: float):
    d = superlevel_persistence(field, maxdim=1, connectivity="8")
    h0, _ = d.split()
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


def scatter_to_grid(node_vals, case):
    g = np.zeros((case["maxny"], case["maxnx"]), dtype=np.float32)
    g[case["iy"], case["ix"]] = node_vals
    return g


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-dataset", type=Path, default=None,
                        help="train on --dataset, test on this never-seen set")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key", type=str, default="damage")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--models", type=str, nargs="*",
                        default=["mlp_pixel", "fno", "deeponet", "convnet",
                                 "graph"])
    parser.add_argument("--t0-frac", type=float, default=0.25)
    parser.add_argument("--horizons", type=int, nargs="*",
                        default=[1, 3, 5, 10, 20, 30, 40])
    parser.add_argument("--sig-tau", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--gchunk", type=int, default=24,
                        help="frame-chunk for graph fwd (memory bound)")
    parser.add_argument("--rollout-k", type=int, default=3,
                        help="pushforward depth: unroll k steps during training "
                             "and backprop through the rollout, so every learned "
                             "model is trained on its OWN compounding error "
                             "(the thing we evaluate). 1-step teacher forcing "
                             "either collapses to persistence or explodes under "
                             "autoregressive rollout. Same k for every model.")
    parser.add_argument("--graph-starts", type=int, default=64,
                        help="random rollout starts sampled per case per epoch "
                             "for the graph (controls graph train cost)")
    parser.add_argument("--graph-rounds", type=int, default=3,
                        help="message-passing rounds in the bond-graph "
                             "forecaster (fewer = far fewer kernel launches)")
    parser.add_argument("--graph-dim", type=int, default=64,
                        help="hidden width of the bond-graph forecaster")
    parser.add_argument("--front-weight", type=float, default=50.0,
                        help="extra loss weight on advancing-front cells (cells "
                             "where the true increment is non-zero). Applied "
                             "IDENTICALLY to every learned model. Unweighted "
                             "MSE (front-weight 0) makes the 5%%-sparse task "
                             "degenerate: all learned models collapse to "
                             "persistence. Traditional baselines are unaffected.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from crackle.operators import NEURAL, GraphForecaster

    fw = args.front_weight

    # bias so the gate STARTS at the physical per-step increment scale
    # (~softplus(GATE_BIAS) ~ 0.004). Without it softplus(0)=0.69 is ~170x too
    # big, the static 95% of cells punish that, and every spatial model flees
    # to the deep-negative zero-increment basin == persistence (observed).
    GATE_BIAS = -5.5

    def gate(raw):
        # non-negative increment gate. softplus (not relu) so the sparse front
        # cells keep a non-zero gradient: under relu, once a front cell's
        # pre-activation goes negative the gradient dies and every learned
        # model freezes at zero-increment == persistence (observed directly).
        return F.softplus(raw + GATE_BIAS)

    def front_weighted_mse(pred, target, prev):
        # upweight cells where the crack actually advances (true increment !=0)
        # so the loss is not dominated by the static background. Same weight
        # for every learned model -> fair.
        w = 1.0 + fw * (torch.abs(target - prev) > 1e-4).float()
        return (w * (pred - target) ** 2).sum() / (w.sum() + 1e-8)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    all_cases = load_cases(args.dataset, key=args.key)
    if args.eval_dataset is not None:
        train_cases = all_cases
        test_cases = load_cases(args.eval_dataset, key=args.key)
        split = "ood"
    else:
        rng0 = np.random.default_rng(0)
        idx = rng0.permutation(len(all_cases))
        n_te = max(1, int(args.test_frac * len(all_cases)))
        test_cases = [all_cases[i] for i in idx[:n_te]]
        train_cases = [all_cases[i] for i in idx[n_te:]]
        split = "heldout"
    print(f"split={split}; train {len(train_cases)} / test {len(test_cases)}; "
          f"train grid {train_cases[0]['dgrid'].shape[1:]} / "
          f"test grid {test_cases[0]['dgrid'].shape[1:]}; device {device}",
          flush=True)

    # --- normalization (TRAIN ONLY) ---
    g_all = np.concatenate([c["g_node"] for c in train_cases])
    g_mean, g_std = float(g_all.mean()), float(g_all.std() + 1e-6)
    et_all = np.concatenate([c["e_tough"] for c in train_cases])
    et_m, et_s = float(et_all.mean()), float(et_all.std() + 1e-6)
    rest_all = np.concatenate([c["rest"] for c in train_cases])
    rest_m, rest_s = float(rest_all.mean()), float(rest_all.std() + 1e-6)

    def edge_feats(c):
        return np.stack([(c["e_tough"] - et_m) / et_s,
                         (c["rest"] - rest_m) / rest_s], axis=1).astype(np.float32)

    # ===================== grid operators (pushforward) =======================
    # all train cases share the padded grid, so stack into one (C,T,H,W) tensor
    k = args.rollout_k
    T_tr = min(c["dgrid"].shape[0] for c in train_cases)
    mv = torch.as_tensor(
        np.stack([c["dgrid"][:T_tr] for c in train_cases]), device=device)
    g_t = torch.as_tensor(
        np.stack([(c["ggrid"] - g_mean) / g_std for c in train_cases]),
        device=device, dtype=torch.float32)
    C = mv.shape[0]
    op_starts = np.array([(ci, t0) for ci in range(C)
                          for t0 in range(0, T_tr - k)], dtype=np.int64)

    def train_operator(mkey, seed):
        torch.manual_seed(seed)
        model = NEURAL[mkey]().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        rng = np.random.default_rng(seed); bs = 16
        for ep in range(args.epochs):
            order = rng.permutation(len(op_starts))
            for i in range(0, len(order), bs):
                sel = op_starts[order[i:i + bs]]
                ci = torch.as_tensor(sel[:, 0], device=device)
                t0 = torch.as_tensor(sel[:, 1], device=device)
                g_b = g_t[ci]
                cur = mv[ci, t0]
                loss = 0.0
                for j in range(k):       # pushforward: feed own predictions
                    x = torch.stack([cur, g_b], dim=1)
                    cur = torch.clamp(cur + gate(model(x)), 0.0, 1.0)
                    loss = loss + front_weighted_mse(
                        cur, mv[ci, t0 + j + 1], mv[ci, t0 + j])
                opt.zero_grad(); (loss / k).backward(); opt.step()
        model.eval(); return model

    def operator_rollout(model, c, t0, hmax):
        with torch.no_grad():
            out = []
            d_t = torch.as_tensor(c["dgrid"][t0], device=device)
            gt = torch.as_tensor((c["ggrid"] - g_mean) / g_std, device=device)
            for _ in range(hmax):
                x = torch.stack([d_t, gt], dim=0).unsqueeze(0)
                d_t = torch.clamp(d_t + gate(model(x)).squeeze(0), 0.0, 1.0)
                out.append(d_t.cpu().numpy())
            return out

    # ========================= crackle graph forecaster =======================
    def train_graph(seed):
        torch.manual_seed(seed)
        model = GraphForecaster(d=args.graph_dim,
                                rounds=args.graph_rounds).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
        rng = np.random.default_rng(seed)
        # pre-move per-case tensors
        prepped = []
        for c in train_cases:
            bonds = torch.as_tensor(c["bonds"], device=device)
            ex = torch.as_tensor(edge_feats(c), device=device)
            dmg = torch.as_tensor(c["dgrid"][:, c["iy"], c["ix"]], device=device)
            gn = torch.as_tensor((c["g_node"] - g_mean) / g_std,
                                 device=device, dtype=torch.float32)
            prepped.append((bonds, ex, dmg, gn))
        for ep in range(args.epochs):
            for ci in rng.permutation(len(prepped)):
                bonds, ex, dmg, gn = prepped[ci]
                T = dmg.shape[0]
                # sample rollout starts in this case, unroll k steps each
                starts = rng.permutation(T - k)[:args.graph_starts]
                for s in range(0, len(starts), args.gchunk):
                    idx = torch.as_tensor(starts[s:s + args.gchunk],
                                          device=device)
                    cur = dmg[idx]                          # (S, N)
                    loss = 0.0
                    for j in range(k):       # pushforward on the bond graph
                        S = cur.shape[0]
                        node_x = torch.stack(
                            [cur, gn.unsqueeze(0).expand(S, -1)], dim=-1)
                        cur = torch.clamp(
                            cur + gate(model(node_x, ex, bonds)), 0.0, 1.0)
                        loss = loss + front_weighted_mse(
                            cur, dmg[idx + j + 1], dmg[idx + j])
                    opt.zero_grad(); (loss / k).backward(); opt.step()
        model.eval(); return model

    def graph_rollout(model, c, t0, hmax):
        with torch.no_grad():
            bonds = torch.as_tensor(c["bonds"], device=device)
            ex = torch.as_tensor(edge_feats(c), device=device)
            gn = torch.as_tensor((c["g_node"] - g_mean) / g_std,
                                 device=device, dtype=torch.float32)
            d_t = torch.as_tensor(c["dgrid"][t0][c["iy"], c["ix"]],
                                  device=device, dtype=torch.float32)
            out = []
            for _ in range(hmax):
                node_x = torch.stack([d_t, gn], dim=-1).unsqueeze(0)  # (1,N,2)
                inc = gate(model(node_x, ex, bonds)).squeeze(0)
                d_t = torch.clamp(d_t + inc, 0.0, 1.0)
                out.append(scatter_to_grid(d_t.cpu().numpy(), c))
            return out

    # =============================== evaluation ===============================
    def eval_rollout(roll_fn):
        per_h = {h: {"rel_l2": [], "rel_l2_front": [], "bottleneck": [],
                     "b0_err": []} for h in args.horizons}
        for c in test_cases:
            d = c["dgrid"]; T = d.shape[0]
            t0 = int(args.t0_frac * T)
            hmax = max(args.horizons)
            if t0 + hmax >= T:
                hmax = T - t0 - 1
            d0 = d[t0]
            preds = roll_fn(c, t0, hmax)
            for h in args.horizons:
                if h > hmax:
                    continue
                p = preds[h - 1]; tru = d[t0 + h]
                denom = np.linalg.norm(tru) + 1e-8
                per_h[h]["rel_l2"].append(float(np.linalg.norm(p - tru) / denom))
                # front-restricted: only the cells the crack ACTUALLY advanced
                # into between t0 and t0+h (the rest is static background that
                # dilutes the full-field number and rewards persistence).
                front = np.abs(tru - d0) > 1e-4
                if front.any():
                    fd = np.linalg.norm(tru[front]) + 1e-8
                    per_h[h]["rel_l2_front"].append(
                        float(np.linalg.norm((p - tru)[front]) / fd))
                bn, b0e = topo_fidelity(p, tru, args.sig_tau)
                per_h[h]["bottleneck"].append(bn)
                per_h[h]["b0_err"].append(float(b0e))
        return {h: {k: float(np.nanmean(v)) if v else float("nan")
                    for k, v in dd.items()} for h, dd in per_h.items()}

    rows = []
    started = time.perf_counter()

    # --- traditional baselines (grid) ---
    def persistence_roll(c, t0, hmax):
        return [c["dgrid"][t0].copy() for _ in range(hmax)]

    def linear_roll(c, t0, hmax):
        d = c["dgrid"]; d0 = d[t0]
        rate = np.clip(d[t0] - d[t0 - 1], 0.0, None) if t0 > 0 else np.zeros_like(d0)
        return [np.clip(d0 + (k + 1) * rate, 0.0, 1.0) for k in range(hmax)]

    inc_mean = (mv[:, 1:] - mv[:, :-1]).clamp(min=0).mean(dim=(0, 1)).cpu().numpy()
    inc_scalar = float(inc_mean.mean())

    def meanrate_roll(c, t0, hmax):
        inc = inc_mean if c["dgrid"].shape[1:] == inc_mean.shape else inc_scalar
        out = []; d_t = c["dgrid"][t0].copy()
        for _ in range(hmax):
            d_t = np.clip(d_t + inc, 0.0, 1.0); out.append(d_t.copy())
        return out

    for name, fn in (("persistence", persistence_roll),
                     ("linear", linear_roll), ("mean_rate", meanrate_roll)):
        for h, m in eval_rollout(fn).items():
            rows.append({"model": name, "seed": -1, "horizon": h, **m})
        print(f"{name} done", flush=True)

    # --- learned models ---
    for mkey in args.models:
        for seed in args.seeds:
            if mkey == "graph":
                model = train_graph(seed)
                res = eval_rollout(lambda c, t0, h, _m=model: graph_rollout(_m, c, t0, h))
            else:
                model = train_operator(mkey, seed)
                res = eval_rollout(lambda c, t0, h, _m=model: operator_rollout(_m, c, t0, h))
            for h, m in res.items():
                rows.append({"model": mkey, "seed": seed, "horizon": h, **m})
            print(f"{mkey} seed {seed} done ({time.perf_counter()-started:.0f}s)",
                  flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "graph_forecast_results.csv", index=False)
    write_json(args.out / "config.json", {
        "dataset": str(args.dataset),
        "eval_dataset": str(args.eval_dataset) if args.eval_dataset else None,
        "split": split, "n_train": len(train_cases), "n_test": len(test_cases),
        "horizons": args.horizons, "epochs": args.epochs, "models": args.models,
        "g_mean": g_mean, "g_std": g_std,
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "wall_s": round(time.perf_counter() - started, 1)})
    print(json.dumps({"out": str(args.out), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
