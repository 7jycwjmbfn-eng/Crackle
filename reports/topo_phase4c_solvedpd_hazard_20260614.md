# Phase 4c — solved-PD replication: crackle beats neural operators on real mechanics

Date: 2026-06-14. Branch `topo-tda`. Removes the kinematic-proxy caveat of
Phase 4b by replicating the per-bond crack-hazard head-to-head on SOLVED
peridynamics (real per-bond alive/stretch time series).

## Result: crackle wins on real mechanics, test AND OOD

Task: does an at-risk bond break within (t, t+H]? horizons {3,5,10}. Data:
`crack_hard_bench_v1` (64 heterogeneous solved-PD cases) split by case into
train(40)/val(12)/test(12); OOD = `crack_notched_plate_v5_1` (16 cases,
different geometry/loading) — a DIFFERENT solved-PD dataset, never trained
on. Features standardized on train only. 3 seeds. Metric: per-bond binary
NLL + top-1% recall.

### SOLVED-PD TEST (NLL ↓ / top-1% recall ↑)

| model               | H3            | H5            | H10           |
|---------------------|---------------|---------------|---------------|
| op_fno (operator)   | 0.00146/0.775 | 0.00221/0.760 | 0.00526/0.620 |
| op_convnet          | 0.00165/0.620 | 0.00248/0.626 | 0.00610/0.490 |
| **bond_gnn (crackle)** | **0.00083/0.999** | **0.00115/0.994** | **0.00307/0.950** |

crackle NLL is 43/48/42% below the best operator; it recovers ~95-100% of
breaking bonds in the top 1% vs the operator's 62-78%.

### SOLVED-PD OOD — different solved-PD dataset, never trained on

| model               | H3            | H5            | H10           |
|---------------------|---------------|---------------|---------------|
| op_fno              | 0.00171/0.825 | 0.00258/0.817 | 0.00455/0.769 |
| op_convnet          | 0.00194/0.651 | 0.00292/0.650 | 0.00523/0.612 |
| **bond_gnn (crackle)** | **0.00101/0.995** | **0.00137/0.992** | **0.00284/0.964** |

crackle NLL is 41/47/37% below the best operator; recall 0.995/0.992/0.964
vs 0.77-0.83. Dominates on genuinely new geometry too.

## Why this strengthens the claim

- Real solved peridynamics, not the kinematic proxy. The per-bond labels are
  actual bond-breaking events from the simulator.
- The operators are a MUCH stronger baseline here than on the kinematic set
  (recall 0.62-0.83 vs 0.04-0.28 in Phase 4b) — solved-PD stretch is a
  smooth, informative field the operator can exploit. crackle still beats
  them clearly, so the win is not an artifact of weak baselines.
- Same split/labels/metric discipline; OOD is a separate dataset (no leakage,
  no shared geometry).

## Bottom line (the user's goal, on real data)

On solved-peridynamics crack-event prediction — the project's actual subject
— the topological bond-graph method beats neural operators (FNO/ConvNet) on
both held-out and out-of-distribution data, by 37-48% per-bond NLL and with
near-perfect top-1% recall. Combined with Phase 4b (kinematic) and the honest
Phase 4 negative on smooth field rollout (where FNO wins), the complete
picture: operators win smooth full-field forecasting; crackle wins discrete
crack-event / hazard prediction, on proxy AND real mechanics, in- and
out-of-distribution.

Provenance: `scripts/topo_track_c_solvedpd.py`,
`outputs/solvedpd_hazard_full/`. Caveat: small n (64+16 cases), single OOD
dataset; horizons short. The qualitative gap (crackle NLL ~halved, recall
~0.99 vs ~0.7) far exceeds the 3-seed spread.
