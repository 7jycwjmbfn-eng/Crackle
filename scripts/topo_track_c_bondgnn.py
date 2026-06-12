"""Track C — bond-graph GNN vs strong tabular referee (Phase 2.2, optional).

PRE-REGISTERED SUCCESS CRITERION (fixed before any training run; see git
history of this file): the bond GNN counts as a positive result iff it
beats the XGBoost referee on TEST per-bond binary NLL by a margin
exceeding the 3-seed std on >= 2 of the 3 horizons {3, 5, 10}. The
referee receives THE SAME raw bond features PLUS engineered one-hop
neighborhood aggregates, so the GNN's only advantage is learned
multi-hop message passing. Anything else is a negative readout.

Task: per-bond hazard — does this at-risk bond break within (t, t+H]? —
on (case, t) snapshots of the native peridynamic bond graph.

Example (PowerShell):
  python -m scripts.topo_track_c_bondgnn `
    --dataset .\\datasets\\topo_bonds_v1 --out .\\runs_topo\\phase2_track_c
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from crackle.data.common import write_json
from crackle.experiments.hetero_pinning import _git_commit
from crackle.metrics.point_process import binary_nll, topk_precision_recall
from crackle.topo.bondgnn import GraphSample, build_node_features, referee_features
from crackle.topo.catalog import split_of_case

HORIZONS = (3, 5, 10)


def load_case_samples(path: Path, height: float = 40.0) -> list[GraphSample]:
    data = np.load(path)
    bonds = data["bonds"].astype(np.int64)
    n_bonds = int(data["n_bonds"])
    points = data["points"].astype(np.float64)
    alive = np.unpackbits(data["alive_packed"], axis=1,
                          count=n_bonds).astype(bool)
    rest = data["rest"].astype(np.float64)
    gc_bond = data["gc_bond"].astype(np.float64)
    meta = json.loads(str(data["meta"]))
    n_nodes = points.shape[0]
    t_total = alive.shape[0]

    centers = 0.5 * (points[bonds[:, 0]] + points[bonds[:, 1]])
    d = points[bonds[:, 1]] - points[bonds[:, 0]]
    orient_y = np.abs(d[:, 1]) / np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    bdist = np.minimum(centers[:, 1] - points[:, 1].min(),
                       points[:, 1].max() - centers[:, 1]) / (0.5 * height)
    rest_norm = rest / float(meta["horizon"])
    gc_norm = gc_bond / max(float(gc_bond.mean()), 1e-9)

    samples = []
    for s_idx, t in enumerate(data["sample_steps"].astype(np.int64)):
        alive_t = alive[t]
        ratio = data["ratio"][s_idx].astype(np.float64)
        bond_x = np.stack([
            np.clip(np.where(alive_t, ratio, 0.0), -2.0, 2.0),
            gc_norm, rest_norm, orient_y, bdist,
        ], axis=1).astype(np.float32)
        labels = np.full((n_bonds, len(HORIZONS)), -1, dtype=np.int8)
        for h_i, h in enumerate(HORIZONS):
            if t + h < t_total:
                labels[:, h_i] = (alive_t & ~alive[t + h]).astype(np.int8)
        labels[~alive_t] = -1  # only at-risk bonds carry labels
        samples.append(GraphSample(
            bonds=bonds, alive=alive_t, bond_x=bond_x,
            node_x=build_node_features(bonds, alive_t, ratio, n_nodes),
            labels=labels, n_nodes=n_nodes))
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-graphs", type=int, default=8)
    parser.add_argument("--referee-train-rows", type=int, default=4_000_000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    import xgboost as xgb

    from crackle.topo.bondgnn import BondGNN

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    manifest = pd.read_csv(args.dataset / "manifest.csv")
    cases: dict[str, list[GraphSample]] = {}
    split_of: dict[str, str] = {}
    for _, row in manifest.iterrows():
        cid = row["case_id"]
        cases[cid] = load_case_samples(args.dataset / "shards" / f"{cid}.npz")
        split_of[cid] = split_of_case(cid, int(row["n_notches"]))
    splits = {s: [c for c, sp in split_of.items() if sp == s]
              for s in ("train", "val", "test", "ood")}
    print({s: len(c) for s, c in splits.items()}, flush=True)

    # ---------------- referee: XGBoost on features + 1-hop aggregates ----
    def tabular(split: str, max_rows: int | None = None, seed: int = 0):
        xs, ys = [], []
        for cid in splits[split]:
            for g in cases[cid]:
                xs.append(referee_features(g))
                ys.append(g.labels[g.alive])
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        if max_rows and x.shape[0] > max_rows:
            idx = np.random.default_rng(seed).choice(
                x.shape[0], max_rows, replace=False)
            x, y = x[idx], y[idx]
        return x, y

    print("building referee tables ...", flush=True)
    x_train, y_train = tabular("train", args.referee_train_rows)
    eval_tables = {s: tabular(s) for s in ("test", "ood")}
    results = []
    for h_i, h in enumerate(HORIZONS):
        keep = y_train[:, h_i] >= 0
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=6,
                                learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist",
                                n_jobs=0, random_state=0,
                                eval_metric="logloss")
        clf.fit(x_train[keep], y_train[keep][:, h_i])
        for split, (x_ev, y_ev) in eval_tables.items():
            m = y_ev[:, h_i] >= 0
            prob = clf.predict_proba(x_ev[m])[:, 1]
            y = y_ev[m][:, h_i]
            k = max(1, int(0.01 * y.size))
            prec, rec = topk_precision_recall(prob, y, k)
            results.append({"model": "gbm_referee", "horizon": h, "seed": -1,
                            "split": split, "nll": binary_nll(prob, y),
                            "top1pct_recall": rec, "top1pct_precision": prec,
                            "n_rows": int(y.size),
                            "pos_rate": float(y.mean())})
        print(f"referee H{h} done", flush=True)

    # ---------------- GNN ------------------------------------------------
    all_graphs = {s: [g for cid in splits[s] for g in cases[cid]]
                  for s in splits}
    bx = np.concatenate([g.bond_x[g.alive] for g in all_graphs["train"]])
    nx_ = np.concatenate([g.node_x for g in all_graphs["train"]])
    bx_mean, bx_std = bx.mean(0), bx.std(0) + 1e-6
    nx_mean, nx_std = nx_.mean(0), nx_.std(0) + 1e-6

    def to_batch(graphs: list[GraphSample]):
        offset = 0
        bonds, alive, bond_x, node_x, labels = [], [], [], [], []
        for g in graphs:
            bonds.append(g.bonds + offset)
            alive.append(g.alive)
            bond_x.append((g.bond_x - bx_mean) / bx_std)
            node_x.append((g.node_x - nx_mean) / nx_std)
            labels.append(g.labels)
            offset += g.n_nodes
        return (torch.as_tensor(np.concatenate(bonds), device=device),
                torch.as_tensor(np.concatenate(alive), device=device),
                torch.as_tensor(np.concatenate(bond_x), device=device),
                torch.as_tensor(np.concatenate(node_x), device=device),
                torch.as_tensor(np.concatenate(labels).astype(np.float32),
                                device=device),
                offset)

    def eval_gnn(model, split: str, seed: int):
        model.eval()
        probs = {h: [] for h in HORIZONS}
        ys = {h: [] for h in HORIZONS}
        graphs = all_graphs[split]
        with torch.no_grad():
            for i in range(0, len(graphs), args.batch_graphs * 4):
                chunk = graphs[i : i + args.batch_graphs * 4]
                bonds, alive, bond_x, node_x, labels, n = to_batch(chunk)
                logits = model(bonds, alive, bond_x, node_x, n)
                p = torch.sigmoid(logits)
                for h_i, h in enumerate(HORIZONS):
                    m = alive & (labels[:, h_i] >= 0)
                    probs[h].append(p[m, h_i].cpu().numpy())
                    ys[h].append(labels[m, h_i].cpu().numpy())
        out = []
        for h in HORIZONS:
            prob = np.concatenate(probs[h])
            y = np.concatenate(ys[h]).astype(np.int8)
            k = max(1, int(0.01 * y.size))
            prec, rec = topk_precision_recall(prob, y, k)
            out.append({"model": "bond_gnn", "horizon": h, "seed": seed,
                        "split": split, "nll": binary_nll(prob, y),
                        "top1pct_recall": rec, "top1pct_precision": prec,
                        "n_rows": int(y.size), "pos_rate": float(y.mean())})
        return out

    started = time.perf_counter()
    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = BondGNN(n_horizons=len(HORIZONS)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_graphs = all_graphs["train"]
        best_val, best_state = np.inf, None
        for epoch in range(args.epochs):
            model.train()
            order = rng.permutation(len(train_graphs))
            total, nb = 0.0, 0
            for i in range(0, order.size, args.batch_graphs):
                chunk = [train_graphs[j]
                         for j in order[i : i + args.batch_graphs]]
                bonds, alive, bond_x, node_x, labels, n = to_batch(chunk)
                logits = model(bonds, alive, bond_x, node_x, n)
                mask = alive.unsqueeze(-1) & (labels >= 0)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.clamp(min=0.0), reduction="none")
                loss = (loss * mask.float()).sum() \
                    / mask.float().sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach())
                nb += 1
            val_rows = eval_gnn(model, "val", seed)
            val_nll = float(np.mean([r["nll"] for r in val_rows]))
            print(f"seed {seed} epoch {epoch}: loss {total/nb:.4f} "
                  f"val mean NLL {val_nll:.5f}", flush=True)
            if val_nll < best_val:
                best_val = val_nll
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        for split in ("test", "ood"):
            results += eval_gnn(model, split, seed)
        print(f"seed {seed} done ({time.perf_counter()-started:.0f}s)",
              flush=True)

    res = pd.DataFrame(results)
    res.to_csv(args.out / "track_c_results.csv", index=False)
    write_json(args.out / "config.json", {
        "dataset": str(args.dataset), "horizons": list(HORIZONS),
        "seeds": list(args.seeds), "epochs": args.epochs,
        "git_commit": _git_commit(Path.cwd()),
        "wall_s": round(time.perf_counter() - started, 1),
    })
    print(json.dumps({"out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
