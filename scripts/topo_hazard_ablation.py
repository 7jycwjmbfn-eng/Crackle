"""Phase 2.2 — hazard-model feature ablation on the tile risk sets.

PRE-REGISTERED SUCCESS CRITERION (fixed before any model run; see the
git history of this file): topological features count as useful iff
ablation (b) or (c) improves TEST binary NLL or top-1% recall over (a)
by a margin exceeding the GBM seed std on >= 2 of the 3 horizons.
Otherwise the readout is negative and is reported as such.

Ablations (the experiment):
  (a) local damage only          : dmg_mean, dmg_max, dmg_grad, dmg_mean_d3
  (b) + global topological curves: + per-step summaries and 3-step deltas
  (c) + event history            : + Hawkes-style decayed counts

Referee baselines (must appear in every table, addendum v1.1 B):
  base-rate          : constant train-split positive rate
  carry-forward      : two-bucket rate by recent tile event history
  hawkes-logistic    : logistic regression on history features only

Models: logistic regression (deterministic) and XGBoost (3 seeds,
mean +/- std). Metrics (crackle.metrics.point_process): binary_nll,
brier, calibration_ece, top-1% precision/recall. Splits are case-level
train/val/test plus the held-out all-4-notch OOD stratum.
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
from crackle.metrics.point_process import (
    binary_nll,
    brier_score,
    calibration_ece,
    topk_precision_recall,
)
from crackle.topo.catalog import (
    GLOBAL_FEATURES,
    HISTORY_FEATURES,
    LOCAL_FEATURES,
)

ABLATIONS: dict[str, list[str]] = {
    "a_local": LOCAL_FEATURES,
    "b_local+topo": LOCAL_FEATURES + GLOBAL_FEATURES,
    "c_local+topo+hist": LOCAL_FEATURES + GLOBAL_FEATURES + HISTORY_FEATURES,
}
GBM_SEEDS = (0, 1, 2)
TOPK_FRAC = 0.01


def _metrics(prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    k = max(1, int(TOPK_FRAC * y.size))
    prec, rec = topk_precision_recall(prob, y, k)
    return {
        "nll": binary_nll(prob, y),
        "brier": brier_score(prob, y),
        "ece": calibration_ece(prob, y),
        "top1pct_precision": prec,
        "top1pct_recall": rec,
    }


def _eval_rows(name: str, ablation: str, horizon: int, seed: int | None,
               prob: dict[str, np.ndarray], y: dict[str, np.ndarray]
               ) -> list[dict[str, Any]]:
    rows = []
    for split in ("test", "ood"):
        rows.append({
            "model": name, "ablation": ablation, "horizon": horizon,
            "seed": -1 if seed is None else seed, "split": split,
            **_metrics(prob[split], y[split]),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True,
                        help="catalog dir containing risksets/")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="*", default=[3, 5, 10])
    parser.add_argument("--gbm-rounds", type=int, default=300)
    parser.add_argument("--lr-max-iter", type=int, default=200)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    splits = {}
    for split in ("train", "test", "ood"):
        path = args.catalog / "risksets" / f"riskset_{split}.parquet"
        splits[split] = pd.read_parquet(path)
        print(f"{split}: {len(splits[split])} rows", flush=True)

    all_feats = ABLATIONS["c_local+topo+hist"]
    results: list[dict[str, Any]] = []
    started = time.perf_counter()

    for horizon in args.horizons:
        label = f"label_any_H{horizon}"
        data = {}
        for split, df in splits.items():
            keep = df[label] >= 0
            data[split] = (
                df.loc[keep, all_feats].to_numpy(dtype=np.float32),
                df.loc[keep, label].to_numpy(dtype=np.int8),
                df.loc[keep],
            )
        y = {s: data[s][1] for s in data}
        base_rate = float(y["train"].mean())
        print(f"H{horizon}: train pos rate {base_rate:.4f}", flush=True)

        # referee: constant base rate
        prob = {s: np.full(y[s].shape, base_rate) for s in ("test", "ood")}
        results += _eval_rows("ref_base_rate", "-", horizon, None, prob, y)

        # referee: carry-forward two-bucket (recent tile event history)
        hist_cols = [c for c in HISTORY_FEATURES if not c.startswith("hist_global")]
        recent_train = (data["train"][2][hist_cols].sum(axis=1) > 0.5).to_numpy()
        rate_recent = float(y["train"][recent_train].mean()) \
            if recent_train.any() else base_rate
        rate_quiet = float(y["train"][~recent_train].mean()) \
            if (~recent_train).any() else base_rate
        prob = {}
        for s in ("test", "ood"):
            recent = (data[s][2][hist_cols].sum(axis=1) > 0.5).to_numpy()
            prob[s] = np.where(recent, rate_recent, rate_quiet)
        results += _eval_rows("ref_carry_forward", "-", horizon, None, prob, y)

        # referee: hawkes-style logistic on history features only
        scaler = StandardScaler().fit(data["train"][2][HISTORY_FEATURES])
        lr = LogisticRegression(max_iter=args.lr_max_iter)
        lr.fit(scaler.transform(data["train"][2][HISTORY_FEATURES]), y["train"])
        prob = {s: lr.predict_proba(
            scaler.transform(data[s][2][HISTORY_FEATURES]))[:, 1]
            for s in ("test", "ood")}
        results += _eval_rows("ref_hawkes_logistic", "-", horizon, None, prob, y)

        for ablation, feats in ABLATIONS.items():
            idx = [all_feats.index(f) for f in feats]
            x_train = data["train"][0][:, idx]
            scaler = StandardScaler().fit(x_train)
            lr = LogisticRegression(max_iter=args.lr_max_iter)
            lr.fit(scaler.transform(x_train), y["train"])
            prob = {s: lr.predict_proba(
                scaler.transform(data[s][0][:, idx]))[:, 1]
                for s in ("test", "ood")}
            results += _eval_rows("logistic", ablation, horizon, None, prob, y)
            print(f"  H{horizon} {ablation} logistic done", flush=True)

            for seed in GBM_SEEDS:
                clf = xgb.XGBClassifier(
                    n_estimators=args.gbm_rounds, max_depth=6,
                    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist", n_jobs=0, random_state=seed,
                    eval_metric="logloss",
                )
                clf.fit(x_train, y["train"])
                prob = {s: clf.predict_proba(data[s][0][:, idx])[:, 1]
                        for s in ("test", "ood")}
                results += _eval_rows("gbm", ablation, horizon, seed, prob, y)
            print(f"  H{horizon} {ablation} gbm x{len(GBM_SEEDS)} done", flush=True)

    res = pd.DataFrame(results)
    res.to_csv(args.out / "ablation_results.csv", index=False)
    write_json(args.out / "config.json", {
        "catalog": str(args.catalog), "horizons": args.horizons,
        "gbm_rounds": args.gbm_rounds, "gbm_seeds": list(GBM_SEEDS),
        "topk_frac": TOPK_FRAC, "git_commit": _git_commit(Path.cwd()),
        "wall_s": round(time.perf_counter() - started, 1),
    })
    print(json.dumps({"rows": len(res), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
