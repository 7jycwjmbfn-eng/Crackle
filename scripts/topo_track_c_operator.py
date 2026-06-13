"""Phase 4b — NEURAL OPERATOR baseline on the Track-C per-bond HAZARD task.

This is the legitimate venue for "crackle > neural operator": crackle's edge
is discrete crack-event / topology prediction (Track C: the bond-graph GNN
already beats a same-feature GBM referee on per-bond breaking NLL), NOT the
smooth full-field rollout where a tuned FNO dominates (Phase 4 readout).

Task (identical to Track C): for each at-risk bond, does it break within
(t, t+H]? horizons {3,5,10}; by-case split with the 4-notch geometry stratum
held out as OOD; metric = per-bond binary NLL + top-1% recall/precision.

Neural-operator baseline (fair analog of an operator on this task): the SAME
per-node state field the GNN sees (broken fraction, mean/max incident
stretch-ratio) is rasterized onto the 48x29 lattice; an FNO / ConvNet maps
it to a per-cell hazard-logit field (one channel per horizon); each bond's
hazard logit is the mean of its two endpoint cells. Trained with the SAME
masked BCE on at-risk bonds, scored with the SAME NLL/recall code. Run the
GNN+GBM with scripts/topo_track_c_bondgnn.py on the same dataset for the full
table.

PRE-REGISTERED (before any training run here): a fair claim is that the
crackle bond-GNN beats the BEST neural operator (FNO/ConvNet) on TEST and on
OOD per-bond NLL by more than seed std on >= 2 of the 3 horizons. Reported
honestly either way.
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
from crackle.topo.catalog import split_of_case
from scripts.topo_track_c_bondgnn import HORIZONS, load_case_samples


def grid_map(points: np.ndarray):
    xs = np.unique(np.round(points[:, 0], 5))
    ys = np.unique(np.round(points[:, 1], 5))
    ix = np.searchsorted(xs, np.round(points[:, 0], 5))
    iy = np.searchsorted(ys, np.round(points[:, 1], 5))
    return iy, ix, ys.size, xs.size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--models", type=str, nargs="*", default=["fno", "convnet"])
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from crackle.operators import FNO2d, ConvNet, DeepONet

    builders = {
        "fno": lambda: FNO2d(c_in=4, width=32, modes=12, n_layers=4, c_out=3),
        "convnet": lambda: ConvNet(c_in=4, width=48, n_blocks=5, c_out=3),
        "deeponet": lambda: DeepONet(c_in=4, p=64, branch_width=48, c_out=3),
    }
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    manifest = pd.read_csv(args.dataset / "manifest.csv")
    # per case: list of snapshots, each a dict of arrays; + static grid/bond map
    cases, split_of = {}, {}
    for _, row in manifest.iterrows():
        cid = row["case_id"]
        samples = load_case_samples(args.dataset / "shards" / f"{cid}.npz")
        data = np.load(args.dataset / "shards" / f"{cid}.npz")
        pts = data["points"].astype(np.float64)
        iy, ix, ny, nx = grid_map(pts)
        bonds = data["bonds"].astype(np.int64)
        # static per-node material toughness (avg of incident bonds) so the
        # operator gets the SAME material info the GNN/GBM have -- no
        # baseline starvation.
        gc = data["gc_bond"].astype(np.float64)
        n_nodes = pts.shape[0]
        acc = np.zeros(n_nodes); cnt = np.zeros(n_nodes)
        for cc in (0, 1):
            np.add.at(acc, bonds[:, cc], gc); np.add.at(cnt, bonds[:, cc], 1.0)
        gc_node = (acc / np.maximum(cnt, 1.0)).astype(np.float32)
        cases[cid] = dict(samples=samples, iy=iy, ix=ix, ny=ny, nx=nx,
                          bi=bonds[:, 0], bj=bonds[:, 1], gc_node=gc_node)
        split_of[cid] = split_of_case(cid, int(row["n_notches"]))
    splits = {s: [c for c, sp in split_of.items() if sp == s]
              for s in ("train", "val", "test", "ood")}
    print({s: len(c) for s, c in splits.items()}, flush=True)

    def node_feat(g, c):  # (N,4): 3 dynamic state feats + static toughness
        return np.concatenate(
            [g.node_x, c["gc_node"][:, None]], axis=1).astype(np.float32)

    # node-feature standardization on TRAIN only
    nxall = np.concatenate([node_feat(g, cases[cid]) for cid in splits["train"]
                            for g in cases[cid]["samples"]])
    nmean, nstd = nxall.mean(0), nxall.std(0) + 1e-6

    def case_tensors(cid):
        c = cases[cid]
        iy = torch.as_tensor(c["iy"], device=device)
        ix = torch.as_tensor(c["ix"], device=device)
        bi = torch.as_tensor(c["bi"], device=device)
        bj = torch.as_tensor(c["bj"], device=device)
        grids, labels, alive = [], [], []
        for g in c["samples"]:
            f = ((node_feat(g, c) - nmean) / nstd).astype(np.float32)  # (N,4)
            grid = np.zeros((4, c["ny"], c["nx"]), dtype=np.float32)
            grid[:, c["iy"], c["ix"]] = f.T
            grids.append(grid)
            labels.append(g.labels.astype(np.float32))           # (n_bonds,H)
            alive.append(g.alive)
        return (torch.as_tensor(np.stack(grids), device=device),    # (S,3,ny,nx)
                iy, ix, bi, bj,
                torch.as_tensor(np.stack(labels), device=device),   # (S,nb,H)
                torch.as_tensor(np.stack(alive), device=device))    # (S,nb)

    def bond_logits(out, iy, ix, bi, bj):
        # out (S,H,ny,nx) -> per-bond logits (S,nb,H)
        node = out[:, :, iy, ix]                 # (S,H,N)
        bl = 0.5 * (node[:, :, bi] + node[:, :, bj])   # (S,H,nb)
        return bl.permute(0, 2, 1)               # (S,nb,H)

    train_ct = {cid: case_tensors(cid) for cid in splits["train"]}
    eval_ct = {s: {cid: case_tensors(cid) for cid in splits[s]}
               for s in ("val", "test", "ood")}

    def eval_split(model, s):
        model.eval()
        probs = {h: [] for h in HORIZONS}; ys = {h: [] for h in HORIZONS}
        with torch.no_grad():
            for cid in splits[s]:
                grids, iy, ix, bi, bj, labels, alive = eval_ct[s][cid]
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
            rows.append(dict(horizon=h, split=s, nll=binary_nll(prob, y),
                             top1pct_recall=rec, top1pct_precision=prec,
                             n_rows=int(y.size), pos_rate=float(y.mean())))
        return rows

    results = []
    started = time.perf_counter()
    train_ids = splits["train"]
    for mkey in args.models:
        for seed in args.seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = builders[mkey]().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
            best_val, best_state = np.inf, None
            for ep in range(args.epochs):
                model.train()
                for cid in [train_ids[i] for i in rng.permutation(len(train_ids))]:
                    grids, iy, ix, bi, bj, labels, alive = train_ct[cid]
                    bl = bond_logits(model(grids), iy, ix, bi, bj)
                    mask = alive.unsqueeze(-1) & (labels >= 0)
                    loss = F.binary_cross_entropy_with_logits(
                        bl, labels.clamp(min=0.0), reduction="none")
                    loss = (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
                    opt.zero_grad(); loss.backward(); opt.step()
                vrows = eval_split(model, "val")
                vnll = float(np.mean([r["nll"] for r in vrows]))
                if vnll < best_val:
                    best_val = vnll
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            for s in ("test", "ood"):
                for r in eval_split(model, s):
                    results.append({"model": f"op_{mkey}", "seed": seed, **r})
            print(f"{mkey} seed {seed} done ({time.perf_counter()-started:.0f}s, "
                  f"best_val_nll {best_val:.5f})", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(args.out / "operator_hazard_results.csv", index=False)
    write_json(args.out / "config.json", {
        "dataset": str(args.dataset), "horizons": list(HORIZONS),
        "seeds": list(args.seeds), "epochs": args.epochs, "models": args.models,
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "wall_s": round(time.perf_counter() - started, 1)})
    print(json.dumps({"out": str(args.out), "rows": len(results)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
