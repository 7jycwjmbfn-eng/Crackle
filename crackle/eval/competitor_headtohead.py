from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from crackle.baselines.hawkes import hawkes_history_by_step
from crackle.data.common import load_json, load_labels, ok_samples, safe_div, write_csv, write_json
from crackle.data.features import material_toughness
from crackle.eval.eval_crackle import _load_model, _predict_proba
from crackle.eval.wallclock_benchmark import causal_candidate_bonds


REGIME_MAP = {
    "single_notch_baseline": "clean_single_notch",
    "heterogeneous_toughness": "heterogeneous_toughness",
    "single_edge_notch_mixed_mode": "mixed_mode",
    "off_center_impact": "mixed_mode",
    "arrest_candidate": "arrest_candidate",
    "hole_plus_notch": "hole_plus_notch",
    "branching_candidate": "branching",
    "double_notch_competing": "branching",
}


def regime_of(sample: dict[str, Any]) -> str:
    return REGIME_MAP.get(str(sample.get("benchmark_case_id") or ""), str(sample.get("benchmark_case_id") or "unknown"))


def crack_curve(labels: dict[str, np.ndarray]) -> np.ndarray:
    alive = labels["bond_alive"].astype(bool)
    initial = alive[0]
    broken = np.count_nonzero(initial[None, :] & ~alive, axis=1)
    return broken.astype(np.float64) / max(float(np.count_nonzero(initial)), 1.0)


def driver_curve(labels: dict[str, np.ndarray]) -> np.ndarray:
    bonds = labels["bonds"].astype(np.int64)
    toughness = material_toughness(labels, bonds)
    stretch = labels["bond_stretch"] / np.maximum(toughness[None, :], 1e-12)
    alive = labels["bond_alive"].astype(bool)
    out = np.zeros((stretch.shape[0],), dtype=np.float64)
    for step in range(stretch.shape[0]):
        vals = stretch[step, alive[step]]
        out[step] = float(np.percentile(vals, 95)) if vals.size else 0.0
    return out


def paris_fit(curve: np.ndarray, driver: np.ndarray, prefix_len: int) -> tuple[float, float]:
    rate = np.diff(curve[:prefix_len], prepend=curve[0])
    x = np.log(np.maximum(driver[:prefix_len], 1e-8))
    y = np.log(np.maximum(rate, 1e-8))
    valid = np.isfinite(x) & np.isfinite(y) & (rate > 0)
    if np.count_nonzero(valid) < 3:
        return 1e-5, 2.0
    a = np.stack([np.ones(np.count_nonzero(valid)), x[valid]], axis=1)
    coef, *_ = np.linalg.lstsq(a, y[valid], rcond=None)
    c = float(np.exp(coef[0]))
    m = float(np.clip(coef[1], 0.2, 8.0))
    return c, m


def build_curve_tensor(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    regimes = sorted(set(REGIME_MAP.values()))
    rows: list[dict[str, Any]] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sample in samples:
        labels = load_labels(sample)
        curve = crack_curve(labels)
        driver = driver_curve(labels)
        t = np.linspace(0.0, 1.0, curve.size, dtype=np.float64)
        prefix_len = max(4, int(math.ceil(0.30 * curve.size)))
        c, m = paris_fit(curve, driver, prefix_len)
        rate = np.diff(curve, prepend=curve[0])
        known_curve = np.full_like(curve, curve[prefix_len - 1])
        known_rate = np.zeros_like(rate)
        known_driver = np.full_like(driver, driver[prefix_len - 1])
        known_curve[:prefix_len] = curve[:prefix_len]
        known_rate[:prefix_len] = rate[:prefix_len]
        known_driver[:prefix_len] = driver[:prefix_len]
        one_hot = np.zeros((curve.size, len(regimes)), dtype=np.float64)
        one_hot[:, regimes.index(regime_of(sample))] = 1.0
        x = np.concatenate(
            [
                known_curve[:, None],
                known_rate[:, None],
                known_driver[:, None],
                t[:, None],
                np.full((curve.size, 1), c, dtype=np.float64),
                np.full((curve.size, 1), m, dtype=np.float64),
                one_hot,
            ],
            axis=1,
        )
        xs.append(x)
        ys.append(curve)
        rows.append({"sample": sample, "prefix_len": prefix_len, "paris_C": c, "paris_m": m})
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), rows


def bootstrap_ci(values: list[float], seed: int = 20260605, reps: int = 1000) -> tuple[float | None, float | None, float | None]:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return None, None, None
    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = np.mean(vals[rng.integers(0, vals.size, size=(int(reps), vals.size))], axis=1)
    return mean, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def train_paris_bilstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    epochs: int,
    hidden: int,
) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    torch.set_num_threads(max(1, min(24, torch.get_num_threads())))
    prefix_len = max(4, int(math.ceil(0.30 * y_train.shape[1])))

    class ParisBiLSTMSA(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden_dim, batch_first=True, bidirectional=True)
            width = hidden_dim * 2
            self.q = nn.Linear(width, width)
            self.k = nn.Linear(width, width)
            self.v = nn.Linear(width, width)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            self.log_rate_scale = nn.Parameter(torch.tensor(-9.2))

        def forward(self, x: Any) -> Any:
            h, _ = self.lstm(x)
            q = self.q(h)
            k = self.k(h)
            v = self.v(h)
            attn = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(float(q.shape[-1])), dim=-1)
            ctx = attn @ v
            raw_delta = self.head(ctx).squeeze(-1)
            curve = x[:, :, 0]
            rate = x[:, :, 1]
            out = curve.clone()
            last_curve = curve[:, prefix_len - 1]
            last_rate = torch.clamp(rate[:, prefix_len - 1], min=0.0)
            delta_rate = torch.relu(raw_delta[:, prefix_len:]) * torch.exp(self.log_rate_scale)
            future_rate = last_rate[:, None] + torch.cumsum(delta_rate, dim=1)
            future_curve = last_curve[:, None] + torch.cumsum(future_rate, dim=1)
            out[:, prefix_len:] = future_curve
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParisBiLSTMSA(x_train.shape[-1], hidden).to(device)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    mask = torch.zeros_like(y)
    mask[:, prefix_len:] = 1.0
    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.sum(mask * (pred - y) ** 2) / torch.clamp(torch.sum(mask), min=1.0)
        loss.backward()
        opt.step()
    return model.cpu()


def predict_paris_model(model: Any, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        pred = model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    return np.maximum.accumulate(np.clip(pred, 0.0, 1.0), axis=1)


def frozen_active_rollout_curve(
    payload: dict[str, Any],
    sample: dict[str, Any],
    labels: dict[str, np.ndarray],
    *,
    prefix_len: int,
    active_topk: int,
) -> np.ndarray:
    true = crack_curve(labels)
    alive0 = labels["bond_alive"].astype(bool)
    bonds = labels["bonds"].astype(np.int64)
    toughness = material_toughness(labels, bonds)
    work = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in labels.items()}
    current_alive = alive0[prefix_len - 1].copy()
    for key in ("bond_stretch", "damage", "strain_energy", "crack_tip_mask"):
        if key in work:
            work[key][prefix_len:] = work[key][prefix_len - 1]
    histories = hawkes_history_by_step(work)
    curve = true.copy()
    curve[prefix_len:] = curve[prefix_len - 1]
    observed_counts = np.diff(true[:prefix_len], prepend=true[0]) * max(float(np.count_nonzero(alive0[0])), 1.0)
    max_events = max(1, int(np.ceil(max(np.percentile(observed_counts, 90), np.mean(observed_counts) * 2.0, 1.0))))
    for step in range(prefix_len - 1, alive0.shape[0] - 1):
        work["bond_alive"][step] = current_alive
        if not np.any(current_alive):
            curve[step + 1 :] = curve[step]
            break
        candidates = causal_candidate_bonds(work, step, active_topk=active_topk, bonds=bonds, toughness=toughness)
        if candidates.size == 0:
            curve[step + 1] = curve[step]
            continue
        prob = _predict_proba(payload, work, step, candidates, histories[step], sample=sample)
        expected = int(np.clip(round(float(np.sum(prob))), 0, min(max_events, candidates.size)))
        if expected > 0:
            local = np.argpartition(prob, -expected)[-expected:]
            current_alive[candidates[local]] = False
        broken = np.count_nonzero(alive0[0] & ~current_alive)
        curve[step + 1] = float(broken) / max(float(np.count_nonzero(alive0[0])), 1.0)
    return np.maximum.accumulate(curve)


def evaluate_curves(
    samples: list[dict[str, Any]],
    predictions: dict[str, dict[str, np.ndarray]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float]] = {}
    arrest_buckets: dict[tuple[str, str], list[float]] = {}
    for sample in samples:
        labels = load_labels(sample)
        true = crack_curve(labels)
        prefix_len = max(4, int(math.ceil(0.30 * true.size)))
        regime = regime_of(sample)
        future = slice(prefix_len, None)
        true_tail = true[future]
        true_final_rate = float(np.mean(np.diff(true[-max(4, true.size // 8) :], prepend=true[-max(4, true.size // 8)])))
        for model, by_sample in predictions.items():
            pred = by_sample[str(sample["id"])]
            rmse = float(np.sqrt(np.mean((pred[future] - true_tail) ** 2)))
            buckets.setdefault((regime, model), []).append(rmse)
            if regime == "arrest_candidate":
                pred_final_rate = float(np.mean(np.diff(pred[-max(4, pred.size // 8) :], prepend=pred[-max(4, pred.size // 8)])))
                arrest_buckets.setdefault((regime, model), []).append(abs(pred_final_rate - true_final_rate))
    rows: list[dict[str, Any]] = []
    for (regime, model), vals in sorted(buckets.items()):
        mean, lo, hi = bootstrap_ci(vals, seed=seed)
        arrest_vals = arrest_buckets.get((regime, model), [])
        arrest_mean, arrest_lo, arrest_hi = bootstrap_ci(arrest_vals, seed=seed) if arrest_vals else (None, None, None)
        rows.append(
            {
                "regime": regime,
                "model": model,
                "metric": "curve_forecast_rmse",
                "mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "n_cases": len(vals),
                "arrest_rate_error_mean": arrest_mean,
                "arrest_rate_error_ci95_low": arrest_lo,
                "arrest_rate_error_ci95_high": arrest_hi,
                "protocol": "first30_known_predict70",
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(args.data / "metadata.json")
    manifest = load_json(Path(meta["source_manifest"]))
    train_samples = ok_samples(manifest, split=args.train_split)
    test_samples = ok_samples(manifest, split=args.test_split)
    x_train, y_train, _ = build_curve_tensor(train_samples)
    x_test, _, test_rows = build_curve_tensor(test_samples)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for seed in args.seed:
        model = train_paris_bilstm(x_train, y_train, seed=seed, epochs=args.epochs, hidden=args.hidden)
        pred = predict_paris_model(model, x_test)
        predictions[f"paris_prior_relu_bilstm_sa_v1_seed{seed}"] = {
            str(row["sample"]["id"]): pred[index] for index, row in enumerate(test_rows)
        }
    for model_dir in args.model_dir or []:
        payload = _load_model(model_dir)
        variant = payload.get("run_variant") or ""
        suffix = f"_{variant}" if variant else ""
        key = f"{payload['model']}{suffix}_seed{payload.get('seed', 0)}_frozen_active_rollout"
        by_sample: dict[str, np.ndarray] = {}
        for sample in test_samples:
            labels = load_labels(sample)
            prefix_len = max(4, int(math.ceil(0.30 * labels["bond_alive"].shape[0])))
            by_sample[str(sample["id"])] = frozen_active_rollout_curve(payload, sample, labels, prefix_len=prefix_len, active_topk=args.active_topk)
        predictions[key] = by_sample
    rows = evaluate_curves(test_samples, predictions, seed=args.seed[0] if args.seed else 20260605)
    write_csv(args.out_dir / "crackle_competitor_headtohead.csv", rows)
    efficiency_rows = []
    target = min((float(row["mean"]) for row in rows if row.get("mean") is not None), default=None)
    for row in rows:
        efficiency_rows.append(
            {
                "model": row["model"],
                "regime": row["regime"],
                "target_metric": "curve_forecast_rmse",
                "target_value": target,
                "real_samples_to_target": len(train_samples) if target is not None and row.get("mean") is not None and float(row["mean"]) <= target * 1.05 else None,
                "available_real_train_samples": len(train_samples),
                "note": "single expanded-data run; subset learning-curve sweep not yet run",
            }
        )
    write_csv(args.out_dir / "crackle_data_efficiency.csv", efficiency_rows)
    payload = {
        "schema": "crackle_competitor_headtohead_v1",
        "dataset_hash": meta.get("dataset_hash") or manifest.get("dataset_hash"),
        "num_train_samples": len(train_samples),
        "num_test_samples": len(test_samples),
        "regime_counts": {regime: sum(1 for sample in test_samples if regime_of(sample) == regime) for regime in sorted(set(REGIME_MAP.values()))},
        "paris_competitor_fidelity": "architecture_implemented; RI auxiliary pretrain unavailable in current synthetic event dataset",
        "survival_rollout_protocol": "prefix frozen active rollout, no future state labels",
        "rows": rows,
    }
    write_json(args.out_dir / "crackle_competitor_headtohead.json", payload)
    print(json.dumps({"out_dir": str(args.out_dir), "rows": rows}, ensure_ascii=False, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crackle first30-to-last70 competitor head-to-head.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, action="append")
    parser.add_argument("--out", "--out-dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--active-topk", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed is None:
        args.seed = [20260605, 20260606, 20260607]
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
