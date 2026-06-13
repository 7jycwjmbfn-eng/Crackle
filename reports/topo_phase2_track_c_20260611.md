# Topo Phase 2.2 Track C — bond-graph GNN vs strong tabular referee

Date: 2026-06-11. Branch: topo-tda. Spec: addendum v1.1 B, Track C
(optional, highest novelty). Artifacts: runs_topo/phase2_track_c/
(track_c_results.csv, config.json), datasets/topo_bonds_v1 (400 cases).

## Pre-registered criterion (committed before the run; see git history of
scripts/topo_track_c_bondgnn.py)

The bond GNN must beat the XGBoost referee on TEST per-bond binary NLL by
more than the 3-seed std on >= 2 of 3 horizons. The referee receives the
SAME raw bond features PLUS engineered one-hop neighborhood aggregates,
so learned multi-hop message passing is the only thing being tested.

VERDICT: **PASS — 3/3 horizons, in-distribution AND on the held-out
4-notch OOD stratum.** This is the strongest of the three frontier
tracks: the only one that beats its referee on the model's native
representation rather than a resampled grid.

## Setup

Task: per at-risk bond, does it break within (t, t+H]? — on (case, t)
snapshots of the native peridynamic bond graph (no grid resampling).
400 cases (train 207 / val 53 / test 48 / OOD 92 = all 4-notch).
Model: 3-round edge/node message passing (d=64), one hazard head per
horizon. Referee: XGBoost (300 trees) on raw bond features + endpoint
broken-fraction and max-ratio aggregates. 3 seeds, val-NLL selection.

## TEST (mean ± std over 3 seeds)

| horizon | GNN NLL | referee NLL | NLL reduction | GNN rec@1% | ref rec@1% |
|---|---|---|---|---|---|
| 3 | **0.00481 ± 0.0001** | 0.00743 | −35% | 0.776 | 0.761 |
| 5 | **0.00597 ± 0.0002** | 0.01175 | −49% | 0.474 | 0.474 |
| 10 | **0.01051 ± 0.0004** | 0.02187 | −52% | 0.236 | 0.236 |

OOD (held-out 4-notch geometry):

| horizon | GNN NLL | referee NLL | NLL reduction | GNN rec@1% | ref rec@1% |
|---|---|---|---|---|---|
| 3 | **0.00521 ± 0.0001** | 0.01053 | −51% | 0.893 | 0.793 |
| 5 | **0.00705 ± 0.0001** | 0.01683 | −58% | 0.595 | 0.587 |
| 10 | **0.01258 ± 0.0005** | 0.03029 | −58% | 0.306 | 0.306 |

## Reading

- The NLL gap (35–58%) is large and grows with horizon: at longer
  horizons a bond's fate depends on damage propagating from farther
  away, which is exactly what multi-hop message passing captures and a
  one-hop tabular aggregate cannot. The referee is not weak — it beats
  the base rate comfortably — the GNN is simply the right inductive bias.
- top-1% recall is tied at H5/H10 (the highest-ratio bonds are easy for
  both); the GNN's edge is in calibrated probability across the bulk of
  at-risk bonds (NLL), and in recall at H3 OOD (0.89 vs 0.79).
- The advantage STRENGTHENS out-of-distribution (−51…−58% vs −35…−52%
  in-distribution): message passing on the native graph transfers across
  geometry better than tabular features keyed to it. This is the cleanest
  positive transfer result in the branch.

## Claim boundary

Allowed: "On the synthetic multi-notch world, a 3-round message-passing
GNN on the native peridynamic bond graph predicts per-bond breaking
hazard with 35–58% lower NLL than a gradient-boosted referee given the
same features plus one-hop neighborhood aggregates, and the advantage
persists (strengthens) on a held-out geometry stratum."

Not allowed: any real-data claim; any mechanics claim (kinematic proxy);
the absolute recall numbers as an operational capability. The result is
about representation/inductive bias, which was the pre-registered
question.
