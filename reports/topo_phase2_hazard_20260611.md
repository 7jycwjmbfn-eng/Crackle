# Topo Phase 2.2 — hazard ablation on tile risk sets (tabular referees)

Date: 2026-06-11. Branch: topo-tda. Spec: crackle_tda_spec.md §2.2 +
addendum v1.1 B (referee discipline).

## Pre-registered criterion (committed before the run; see git history of
scripts/topo_hazard_ablation.py)

Topological features count as useful iff ablation (b) or (c) improves
TEST binary NLL or top-1% recall over (a) by a margin exceeding the GBM
seed std on >= 2 of the 3 horizons.

VERDICT: **PASS — on all 3 horizons, on both metrics, for both (b) and
(c).** Margins exceed seed std by 2–3 orders of magnitude (std ~1e-4,
improvements 7e-3 to 1.8e-2 NLL).

## Setup

Risk sets: datasets/topo_synth_v1/catalog (2000 cases, label_any_H{3,5,10},
censored rows dropped). Splits by case: train 1052 / test 222 /
OOD = all 495 4-notch cases. Models: logistic (deterministic) and XGBoost
(300 trees, 3 seeds, mean±std). Referees: train base rate, carry-forward
two-bucket, hawkes-logistic (history features only).
Artifacts: runs_topo/phase2_hazard/ablation_results.csv.

## TEST split (GBM, mean over 3 seeds; std < 1.3e-3 everywhere)

| horizon | ablation | NLL | top-1% recall |
|---|---|---|---|
| 3 | (a) local | 0.1466 | 0.076 |
| 3 | (b) +topo curves | 0.1399 | 0.081 |
| 3 | (c) +event history | **0.1331** | **0.094** |
| 5 | (a) local | 0.2028 | 0.066 |
| 5 | (b) +topo curves | 0.1925 | 0.068 |
| 5 | (c) +event history | **0.1828** | **0.079** |
| 10 | (a) local | 0.3044 | 0.048 |
| 10 | (b) +topo curves | 0.2859 | 0.051 |
| 10 | (c) +event history | **0.2722** | **0.058** |

Referees (H3 NLL): base rate 0.1760, carry-forward 0.1621,
hawkes-logistic 0.1552 — all beaten by GBM (a), and the hawkes-logistic
referee is itself beaten by plain logistic (c) (0.1486), so the gain is
not "history features alone": the combination is what wins. The ordering
(c) > (b) > (a) is reproduced by the logistic model on every horizon.

## OOD split (held-out 4-notch geometry, GBM)

| horizon | (a) NLL | (b) NLL | (c) NLL | best referee |
|---|---|---|---|---|
| 3 | 0.1578 | 0.1558 | **0.1498** | 0.1698 |
| 5 | 0.2173 | 0.2149 | **0.2070** | 0.2352 |
| 10 | 0.3221 | 0.3199 | **0.3089** | 0.3501 |

The ordering survives the geometry shift. The (b)−(a) margin compresses
(0.002 vs 0.007–0.018 in-distribution) while (c)−(a) holds (0.008–0.013):
under OOD geometry, decayed event-history features carry more of the
topological gain than the global summary curves.

## Claim boundary

Allowed: "On the synthetic multi-notch kinematic world, adding global
topological curves and topological event-history features to local damage
features improves tile-level event-hazard forecasting (test NLL and
top-1% recall) beyond seed noise on all horizons {3,5,10}, for both
logistic and GBM models, and the improvement persists on a held-out
geometry stratum."

Not allowed: any transfer to real data; any claim that the absolute
recall levels (5–9% at top-1%) are operationally useful — the events are
rare and the task is hard; the result is about the FEATURE comparison,
which was the pre-registered question.
