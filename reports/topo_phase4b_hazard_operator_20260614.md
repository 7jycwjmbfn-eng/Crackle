# Phase 4b — Crackle beats neural operators AND traditional on crack-hazard

Date: 2026-06-14. Branch `topo-tda`. Goal (user): crackle must be stronger
than BOTH (a) traditional methods AND (b) neural operators, on new data, with
honest, un-sandbagged baselines and no cherry-picking.

## Result (up front): GOAL MET on the crack-event task

On the per-bond breaking-HAZARD task (the project's actual subject), the
crackle bond-graph GNN beats BOTH the neural operators (FNO, ConvNet) and the
traditional GBM referee, on the held-out TEST split AND on the OOD split
(4-notch geometry stratum held out entirely — genuinely unseen), across all
three horizons {3,5,10}.

### TEST (per-bond NLL ↓ / top-1% recall ↑, mean of 3 seeds)

| model                 | H3 NLL/rec | H5 NLL/rec | H10 NLL/rec |
|-----------------------|-----------|-----------|------------|
| op_fno (operator)     | 0.0360/0.255 | 0.0593/0.168 | 0.1156/0.089 |
| op_convnet (operator) | 0.0360/0.275 | 0.0596/0.183 | 0.1164/0.097 |
| op_deeponet (operator)| 0.0639/0.037 | 0.0947/0.038 | 0.1605/0.035 |
| gbm_referee (traditional) | 0.0074/0.761 | 0.0117/0.474 | 0.0219/0.236 |
| **bond_gnn (crackle)**| **0.0047/0.774** | **0.0059/0.474** | **0.0107/0.236** |

crackle NLL is 87/90/91% below the best neural operator and 37/49/51% below
the GBM at H3/H5/H10.

### OOD — 4-notch geometry held out (genuinely new)

| model                 | H3 NLL/rec | H5 NLL/rec | H10 NLL/rec |
|-----------------------|-----------|-----------|------------|
| op_fno                | 0.0293/0.289 | 0.0481/0.192 | 0.0899/0.108 |
| op_convnet            | 0.0290/0.313 | 0.0478/0.210 | 0.0899/0.121 |
| op_deeponet           | 0.0551/0.044 | 0.0824/0.045 | 0.1385/0.044 |
| gbm_referee           | 0.0105/0.793 | 0.0168/0.587 | 0.0303/0.306 |
| **bond_gnn (crackle)**| **0.0051/0.892** | **0.0070/0.595** | **0.0127/0.306** |

crackle NLL is 82/85/86% below the best neural operator and 51/58/58% below
the GBM. On OOD, crackle's top-1% recall (0.892 at H3) is the strongest of
all models — it generalizes to unseen geometry best.

## Why this is honest, not cherry-picked

- **Same everything.** Identical dataset (`datasets/topo_bonds_v1`, 400
  cases), identical by-case split (train/val/test + 4-notch OOD via the
  pre-registered `split_of_case`), identical labels (does an at-risk bond
  break within (t,t+H]?), identical metric code (per-bond `binary_nll`,
  `topk_precision_recall`). 3 seeds; operator variance is ~1e-4 (negligible).
- **Operators NOT sandbagged.** They receive the SAME per-node state field
  the GNN sees (broken fraction, mean/max incident stretch-ratio) PLUS the
  static material-toughness field as a 4th channel. Adding toughness did not
  move their val NLL (0.0691 -> 0.0691): the gap is structural, not feature
  starvation. The GBM referee is also a strong per-bond tabular model (and
  crackle beats it too, by 37-58%).
- **The structural reason crackle wins:** a field operator predicts a smooth
  per-CELL hazard; two distinct bonds sharing endpoint cells get identical
  predictions, so it cannot resolve which individual bond breaks — exactly
  what per-bond hazard requires. The native bond graph represents each bond,
  so message passing localizes the hazard. This is the mirror image of the
  full-field rollout task (Phase 4 readout) where the smooth FNO dominates.

## The complete, honest picture across both Phase-4 tasks

- **Full-field damage rollout** (smooth field forecasting): FNO wins, beating
  traditional baselines AND crackle (held-out + OOD). See
  `topo_phase4_operator_headtohead_20260614.md`. Honest negative for crackle
  there; that task structurally favours an L2-optimised operator.
- **Per-bond crack hazard** (discrete crack-event prediction — the project's
  actual goal): crackle wins, beating traditional AND neural operators
  (held-out + OOD). This report.

Conclusion: on the task the project is about — predicting WHERE/WHEN cracks
advance — the topological bond-graph method is decisively stronger than both
classical methods and neural operators, in- and out-of-distribution.

## Provenance & reproduction

- operators: `crackle/operators.py` (FNO2d/ConvNet now take `c_out`).
- harness: `scripts/topo_track_c_operator.py` (rasterise node field ->
  per-cell hazard -> per-bond via endpoint mean -> masked BCE; same NLL/recall
  as Track C). Operator family = FNO, ConvNet, DeepONet (DeepONet's global
  basis is worst — top-1% recall ~0.04, near chance). outputs
  `outputs/hazard_operator_v2/`.
- crackle + GBM: `scripts/topo_track_c_bondgnn.py`, outputs
  `outputs/track_c_fresh/`.
- Caveat: kinematic-proxy bond dataset (no quantitative mechanics claim); the
  comparison is methodological (representation matters), consistent with the
  dataset card's scope note. Solved-PD replication is future work.
