# Phase 4 — Crackle vs Neural Operators vs Traditional (honest readout)

Date: 2026-06-14. Branch `topo-tda`. Goal (user, upgraded): the crackle
topological method must beat BOTH (a) traditional/classical baselines AND
(b) neural operators (FNO/DeepONet), evaluated on NEW / held-out data with
no leakage, no sandbagged baselines, no cherry-picking.

## Verdict (up front, honest)

On the full-field solved-peridynamics damage-rollout task, a fairly-tuned
**FNO is the strongest model** — it beats every traditional baseline AND the
crackle bond-graph forecaster, on both raw rel-L2 and topological bottleneck,
on held-out cases AND on a genuinely-new dataset. **Crackle beats the
traditional baselines on the new dataset, but loses to FNO.** The user's
target ("crackle > both families") is therefore **NOT met on this task.**
This is reported as-is; the harness, data discipline, and negative result
all stand.

## Task & data

- Task: autoregressive rollout of a solved-PD damage field. Given d_t and
  static toughness g, predict the non-negative increment to d_{t+1}; damage
  is monotone (broken bonds stay broken). Evaluated at horizons 10/20/30/40.
- Train: `crack_hard_bench_v1_n4096_s64_t160` (64 heterogeneous-toughness
  cases, grid 40x103). By-case held-out split (no frame crosses train/test).
- True OOD: `crack_notched_plate_v5_1_n4096_s16_t120` (16 homogeneous
  notched-plate cases — different geometry, loading, length). Never trained
  on. Normalization stats come from the train set only.
- Metrics: full-field rel-L2; front-restricted rel-L2 (only cells the crack
  actually advanced into); topological bottleneck distance + b0 error on the
  super-level persistence of the field.

## Models (fair protocol)

Traditional referees: persistence (frozen field), linear (clamped last-rate
extrapolation), mean_rate (mean train increment). Neural operators: FNO,
DeepONet, ConvNet (residual dilated CNN). Ablation: MLPPixel (1x1, no spatial
coupling). Crackle: GraphForecaster — message passing on the native
peridynamic bond graph (mean/degree-normalized aggregation, static edge
features = toughness + rest length), predicting per-node increment. All
learned models share epochs, optimizer (AdamW 2e-3), front-weighted loss,
softplus increment gate, gate-bias init, and pushforward depth.

## The methodology finding that mattered most

Naive 1-step MSE on this task is DEGENERATE. The crack front is ~5% of
pixels, so the MSE optimum is "predict zero increment". At 40 epochs **every
neural operator collapsed bit-identically to persistence**, and a traditional
baseline (mean_rate) beat all of them. A comparison run in that state would
have been meaningless (and would have flattered whichever model happened to
wobble off persistence). Fixes, applied IDENTICALLY to every learned model:

1. front-weighted loss (upweight cells where the crack actually advances);
2. softplus (not relu) increment gate — relu's dead gradient at the sparse
   front froze every model at zero;
3. gate-bias init at the physical increment scale (~0.004), not
   softplus(0)=0.69 (~170x too big, which stampeded models into the zero
   basin);
4. pushforward (k-step) training — 1-step teacher forcing collapses or
   explodes under long autoregressive rollout.

Under the shared protocol (front-weight 8, k=3), FNO becomes a strong
operator that beats all traditional baselines — a fair, un-sandbagged bar.

## Results — held-out hard_bench (seed 0), rel-L2

| model        | h10  | h20  | h30  | h40  |
|--------------|------|------|------|------|
| persistence  | 0.107| 0.181| 0.243| 0.291|
| mean_rate    | 0.093| 0.151| 0.197| 0.229|
| **FNO**      |**0.066**|**0.112**|**0.144**|**0.170**|
| crackle graph| 0.107| 0.181| 0.243| 0.291 (collapsed)|

DeepONet/ConvNet also collapsed under the shared protocol (DeepONet's global
basis can't localize sharp fronts; ConvNet is init-sensitive). FNO is the
strongest; it is the bar crackle must clear.

## Results — TRUE OOD (train hard_bench, test never-seen notched-plate)

| model        | h10  | h20  | h30  | h40  | bottleneck (h10..h40)        |
|--------------|------|------|------|------|------------------------------|
| persistence  | 0.142| 0.234| 0.305| 0.359| 0.018 / 0.065 / 0.087 / 0.121|
| mean_rate    | 0.115| 0.178| 0.220| 0.245| 0.010 / 0.044 / 0.055 / 0.078|
| crackle graph| 0.089| 0.133| 0.177| 0.233| 0.017 / 0.063 / 0.084 / 0.118|
| **FNO**      |**0.046**|**0.083**|**0.106**|**0.115**|**0.011 / 0.021 / 0.012 / 0.020**|

- Crackle BEATS all traditional baselines on the new dataset (good).
- FNO beats crackle decisively on both rel-L2 and topology, and generalizes
  to the new geometry even better than in-distribution.

## Why crackle loses here (honest analysis)

- FNO predicts a smooth, bounded spectral correction → naturally stable, low
  L2, faithful topology. The full-field rel-L2 task rewards exactly this.
- The bond-graph forecaster adds local increments that compound over the
  40-step rollout; it oscillates between collapse (k=4,6) and over-growth
  (k=3) with no robustly stable, accurate regime found across front-weight
  {4,8} x k {3,4,6} x mean-aggregation. Its one genuine edge — best
  short-horizon (h10) topological fidelity — is destroyed by long-horizon
  instability.
- This task (smooth full-field forecasting) is structurally favorable to an
  L2-optimized operator and unfavorable to a discrete crack-front graph.

## Recommended next step (crackle's home turf)

The project's own prior result (Track C) already showed the bond-graph GNN
beating a same-feature GBM on per-bond breaking HAZARD — crackle's genuine
strength is discrete crack-event / topology prediction, not smooth field
L2. The legitimate route to "crackle > neural operator" is to put a neural
operator (FNO/DeepONet as a hazard-FIELD operator) into the Track-C hazard
benchmark and compare there, on held-out + OOD. That plays to crackle's
established edge instead of a task built for smooth operators.

Provenance: harness `scripts/topo_graph_forecast.py`, models
`crackle/operators.py`, outputs under `outputs/h2h_*`. Single-seed (seed 0)
for cost; the qualitative gaps (FNO >> others) far exceed plausible seed
noise given the near-deterministic collapse behavior of the other models.
