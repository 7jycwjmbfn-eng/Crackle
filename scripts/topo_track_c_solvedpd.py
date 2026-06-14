"""Phase 4c — solved-PD replication of the per-bond HAZARD head-to-head.

Removes the kinematic-proxy caveat of Phase 4b: same task (does an at-risk
bond break within (t, t+H]?) but on SOLVED peridynamics, using the real
per-bond alive/stretch time series. Crackle bond-GNN vs neural operators
(FNO/ConvNet, rasterized node field -> per-cell hazard -> per-bond) vs a
no-message-passing tabular logistic referee, all on identical
snapshots/labels/metric.

Split: train/val/test by case on the heterogeneous hard_bench; OOD = a
DIFFERENT solved-PD dataset (notched_plate) never trained on -- genuinely
new geometry/loading. No leakage: standardization stats from train only.

PRE-REGISTERED (before any training run here): crackle beats the best neural
operator on TEST and OOD per-bond NLL by > seed std on >= 2 of 3 horizons.
Honest either way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from crackle.data.common import write_json
from crackle.experiments.hetero_pinning import _git_commit
from crackle.metrics.point_process import binary_nll, topk_precision_recall
from crackle.topo.bondgnn import (BondGNN, GraphSample, build_node_features,
                                  referee_features)

HORIZONS = (3, 5, 10)


def grid_map(points):
    xs = np.unique(np.round(points[:, 0], 5))
    ys = np.unique(np.round(points[:, 1], 5))
    ix = np.searchsorted(xs, np.round(points[:, 0], 5))
    iy = np.searchsorted(ys, np.round(points[:, 1], 5))
    return iy, ix, ys.size, xs.size


def load_case(npz_path: Path, n_steps: int = 14):
    d = np.load(npz_path)
    bonds = d["bonds"].astype(np.int64)
    alive = d["bond_alive"].astype(bool)            # (T, M)
    stretch = d["bond_stretch"].astype(np.float64)  # (T, M)
    pts = d["reference_x"].astype(np.float64)
    n_nodes = pts.shape[0]
    T, M = alive.shape
    gc = d["material_toughness"].astype(np.float64) if "material_toughness" in d \
        else np.ones(M)
    rest = d["bond_rest_length"].astype(np.float64) if "bond_rest_length" in d \
        else np.linalg.norm(pts[bonds[:, 1]] - pts[bonds[:, 0]], axis=1)
    vec = pts[bonds[:, 1]] - pts[bonds[:, 0]]
    orient_y = np.abs(vec[:, 1]) / np.maximum(np.linalg.norm(vec, axis=1), 1e-9)
    centers = 0.5 * (pts[bonds[:, 0]] + pts[bonds[:, 1]])
    h = pts[:, 1].max() - pts[:, 1].min() + 1e-9
    bdist = np.minimum(centers[:, 1] - pts[:, 1].min(),
                       pts[:, 1].max() - centers[:, 1]) / (0.5 * h)
    gc_norm = gc / max(gc.mean(), 1e-9)
    rest_norm = rest / max(rest.mean(), 1e-9)
    iy, ix, ny, nx = grid_map(pts)

    # sample timesteps where there is still action (some bonds alive and some
    # breaking ahead); spread across the trajectory.
    last = T - max(HORIZONS) - 1
    if last < 2:
        return None
    steps = np.unique(np.linspace(1, last, n_steps).astype(int))
    samples = []
    for t in steps:
        alive_t = alive[t]
        s_t = np.clip(stretch[t], -3.0, 3.0)
        bond_x = np.stack([np.where(alive_t, s_t, 0.0), gc_norm, rest_norm,
                           orient_y, bdist], axis=1).astype(np.float32)
        labels = np.full((M, len(HORIZONS)), -1, dtype=np.int8)
        for hi, hh in enumerate(HORIZONS):
            labels[:, hi] = (alive_t & ~alive[t + hh]).astype(np.int8)
        labels[~alive_t] = -1
        node_x = build_node_features(bonds, alive_t, np.where(alive_t, s_t, 0.0),
                                     n_nodes)
        samples.append(GraphSample(bonds=bonds, alive=alive_t, bond_x=bond_x,
                                    node_x=node_x, labels=labels,
                                    n_nodes=n_nodes))
    return dict(samples=samples, iy=iy, ix=ix, ny=ny, nx=nx,
                bi=bonds[:, 0], bj=bonds[:, 1])


def load_dataset(root: Path, limit=None):
    sd = root / "samples"
    cases = {}
    for i, case in enumerate(sorted(sd.iterdir())):
        if limit and i >= limit:
            break
        npz = case / "crack_labels.npz"
        if not npz.exists():
            continue
        c = load_case(npz)
        if c is not None:
            cases[case.name] = c
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)   # hard_bench
    ap.add_argument("--ood-dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from crackle.operators import FNO2d, ConvNet

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}; loading ...", flush=True)
    base = load_dataset(args.dataset)
    ood = load_dataset(args.ood_dataset)
    ids = sorted(base)
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(len(ids))
    n_test = max(1, len(ids) // 5); n_val = max(1, len(ids) // 5)
    test_ids = [ids[i] for i in perm[:n_test]]
    val_ids = [ids[i] for i in perm[n_test:n_test + n_val]]
    train_ids = [ids[i] for i in perm[n_test + n_val:]]
    splits = {"train": [(base, c) for c in train_ids],
              "val": [(base, c) for c in val_ids],
              "test": [(base, c) for c in test_ids],
              "ood": [(ood, c) for c in sorted(ood)]}
    print({k: len(v) for k, v in splits.items()}, flush=True)

    # standardize node + bond features on train
    nx_all = np.concatenate([g.node_x for ds, c in splits["train"]
                             for g in ds[c]["samples"]])
    bx_all = np.concatenate([g.bond_x[g.alive] for ds, c in splits["train"]
                             for g in ds[c]["samples"]])
    nmean, nstd = nx_all.mean(0), nx_all.std(0) + 1e-6
    bmean, bstd = bx_all.mean(0), bx_all.std(0) + 1e-6

    # ---------------- crackle bond-GNN ----------------
    def gnn_eval(model, split):
        model.eval()
        probs = {h: [] for h in HORIZONS}; ys = {h: [] for h in HORIZONS}
        with torch.no_grad():
            for ds, c in splits[split]:
                for g in ds[c]["samples"]:
                    bonds = torch.as_tensor(g.bonds, device=device)
                    alive = torch.as_tensor(g.alive, device=device)
                    bx = torch.as_tensor((g.bond_x - bmean) / bstd, device=device)
                    nx = torch.as_tensor((g.node_x - nmean) / nstd, device=device)
                    lab = torch.as_tensor(g.labels, device=device)
                    logits = model(bonds, alive, bx, nx, g.n_nodes)
                    p = torch.sigmoid(logits)
                    for hi, h in enumerate(HORIZONS):
                        m = alive & (lab[:, hi] >= 0)
                        probs[h].append(p[m, hi].cpu().numpy())
                        ys[h].append(lab[m, hi].cpu().numpy())
        rows = []
        for h in HORIZONS:
            prob = np.concatenate(probs[h]); y = np.concatenate(ys[h]).astype(np.int8)
            k = max(1, int(0.01 * y.size))
            prec, rec = topk_precision_recall(prob, y, k)
            rows.append(dict(horizon=h, split=split, nll=binary_nll(prob, y),
                             top1pct_recall=rec, top1pct_precision=prec,
                             n_rows=int(y.size), pos_rate=float(y.mean())))
        return rows

    # ---------------- neural operators ----------------
    def case_grid(ds, c):
        cc = ds[c]
        grids, labels, alive = [], [], []
        for g in cc["samples"]:
            f = ((g.node_x - nmean) / nstd).astype(np.float32)
            grid = np.zeros((3, cc["ny"], cc["nx"]), dtype=np.float32)
            grid[:, cc["iy"], cc["ix"]] = f.T
            grids.append(grid); labels.append(g.labels.astype(np.float32))
            alive.append(g.alive)
        return (torch.as_tensor(np.stack(grids), device=device),
                torch.as_tensor(cc["iy"], device=device),
                torch.as_tensor(cc["ix"], device=device),
                torch.as_tensor(cc["bi"], device=device),
                torch.as_tensor(cc["bj"], device=device),
                torch.as_tensor(np.stack(labels), device=device),
                torch.as_tensor(np.stack(alive), device=device))

    def bond_logits(out, iy, ix, bi, bj):
        node = out[:, :, iy, ix]
        bl = 0.5 * (node[:, :, bi] + node[:, :, bj])
        return bl.permute(0, 2, 1)

    grid_cache = {s: [(ds, c, case_grid(ds, c)) for ds, c in splits[s]]
                  for s in splits}

    def op_eval(model, split):
        model.eval()
        probs = {h: [] for h in HORIZONS}; ys = {h: [] for h in HORIZONS}
        with torch.no_grad():
            for ds, c, (grids, iy, ix, bi, bj, labels, alive) in grid_cache[split]:
                bl = bond_logits(model(grids), iy, ix, bi, bj)
                p = torch.sigmoid(bl)
                for hi, h in enumerate(HORIZONS):
                    m = alive & (labels[:, :, hi] >= 0)
                    probs[h].append(p[:, :, hi][m].cpu().numpy())
                    ys[h].append(labels[:, :, hi][m].cpu().numpy())
        rows = []
        for h in HORIZONS:
            prob = np.concatenate(probs[h]); y = np.concatenate(ys[h]).astype(np.int8)
            k = max(1, int(0.01 * y.size))
            prec, rec = topk_precision_recall(prob, y, k)
            rows.append(dict(horizon=h, split=split, nll=binary_nll(prob, y),
                             top1pct_recall=rec, top1pct_precision=prec,
                             n_rows=int(y.size), pos_rate=float(y.mean())))
        return rows

    results = []
    started = time.perf_counter()

    # ---------------- traditional referee (XGBoost on bond feats + 1-hop) ----
    try:
        import xgboost as xgb

        def tab(split):
            xs, ys = [], []
            for ds, c in splits[split]:
                for g in ds[c]["samples"]:
                    gg = GraphSample(bonds=g.bonds, alive=g.alive,
                                     bond_x=(g.bond_x - bmean) / bstd,
                                     node_x=(g.node_x - nmean) / nstd,
                                     labels=g.labels, n_nodes=g.n_nodes)
                    xs.append(referee_features(gg)); ys.append(g.labels[g.alive])
            return np.concatenate(xs), np.concatenate(ys)

        xtr, ytr = tab("train")
        ev = {s: tab(s) for s in ("test", "ood")}
        for hi, h in enumerate(HORIZONS):
            keep = ytr[:, hi] >= 0
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                    learning_rate=0.1, subsample=0.8,
                                    colsample_bytree=0.8, tree_method="hist",
                                    n_jobs=0, random_state=0,
                                    eval_metric="logloss")
            clf.fit(xtr[keep], ytr[keep][:, hi])
            for s, (xe, ye) in ev.items():
                mm = ye[:, hi] >= 0
                prob = clf.predict_proba(xe[mm])[:, 1]; y = ye[mm][:, hi]
                k = max(1, int(0.01 * y.size))
                prec, rec = topk_precision_recall(prob, y, k)
                results.append(dict(model="gbm_referee", seed=-1, horizon=h,
                                    split=s, nll=binary_nll(prob, y),
                                    top1pct_recall=rec, top1pct_precision=prec,
                                    n_rows=int(y.size), pos_rate=float(y.mean())))
        print("gbm referee done", flush=True)
    except Exception as exc:  # xgboost optional; operators+gnn still run
        print(f"gbm referee skipped: {exc}", flush=True)

    op_builders = {
        "op_fno": lambda: FNO2d(c_in=3, width=32, modes=12, n_layers=4, c_out=3),
        "op_convnet": lambda: ConvNet(c_in=3, width=48, n_blocks=5, c_out=3),
    }
    for seed in args.seeds:
        # crackle GNN
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        gnn = BondGNN(n_horizons=len(HORIZONS)).to(device)
        opt = torch.optim.AdamW(gnn.parameters(), lr=1e-3)
        best, best_state = np.inf, None
        for ep in range(args.epochs):
            gnn.train()
            order = rng.permutation(len(train_ids))
            for j in order:
                ds, c = base, train_ids[j]
                for g in ds[c]["samples"]:
                    bonds = torch.as_tensor(g.bonds, device=device)
                    alive = torch.as_tensor(g.alive, device=device)
                    bx = torch.as_tensor((g.bond_x - bmean) / bstd, device=device)
                    nx = torch.as_tensor((g.node_x - nmean) / nstd, device=device)
                    lab = torch.as_tensor(g.labels.astype(np.float32), device=device)
                    logits = gnn(bonds, alive, bx, nx, g.n_nodes)
                    mask = alive.unsqueeze(-1) & (lab >= 0)
                    loss = F.binary_cross_entropy_with_logits(
                        logits, lab.clamp(min=0.0), reduction="none")
                    loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
                    opt.zero_grad(); loss.backward(); opt.step()
            vnll = float(np.mean([r["nll"] for r in gnn_eval(gnn, "val")]))
            if vnll < best:
                best = vnll
                best_state = {k: v.detach().clone() for k, v in gnn.state_dict().items()}
        gnn.load_state_dict(best_state)
        for s in ("test", "ood"):
            for r in gnn_eval(gnn, s):
                results.append({"model": "bond_gnn", "seed": seed, **r})
        print(f"gnn seed {seed} done ({time.perf_counter()-started:.0f}s, "
              f"val {best:.5f})", flush=True)

        # operators
        for mkey, build in op_builders.items():
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = build().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
            best, best_state = np.inf, None
            train_cache = [x for x in grid_cache["train"]]
            for ep in range(args.epochs):
                model.train()
                for k in rng.permutation(len(train_cache)):
                    _, _, (grids, iy, ix, bi, bj, labels, alive) = train_cache[k]
                    bl = bond_logits(model(grids), iy, ix, bi, bj)
                    mask = alive.unsqueeze(-1) & (labels >= 0)
                    loss = F.binary_cross_entropy_with_logits(
                        bl, labels.clamp(min=0.0), reduction="none")
                    loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1)
                    opt.zero_grad(); loss.backward(); opt.step()
                vnll = float(np.mean([r["nll"] for r in op_eval(model, "val")]))
                if vnll < best:
                    best = vnll
                    best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            for s in ("test", "ood"):
                for r in op_eval(model, s):
                    results.append({"model": mkey, "seed": seed, **r})
            print(f"{mkey} seed {seed} done ({time.perf_counter()-started:.0f}s, "
                  f"val {best:.5f})", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(args.out / "solvedpd_hazard_results.csv", index=False)
    write_json(args.out / "config.json", {
        "dataset": str(args.dataset), "ood_dataset": str(args.ood_dataset),
        "split": {k: len(v) for k, v in splits.items()},
        "horizons": list(HORIZONS), "seeds": list(args.seeds),
        "epochs": args.epochs,
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "wall_s": round(time.perf_counter() - started, 1)})
    print(json.dumps({"out": str(args.out), "rows": len(results)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
