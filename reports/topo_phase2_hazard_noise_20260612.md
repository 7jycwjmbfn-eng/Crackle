# Topo Phase 2.2 robustness — does the topology beat traditional methods UNDER noise?

Date: 2026-06-12. Branch: topo-tda. Spec: crackle_tda_spec.md §2.2
(strengthening). This re-runs the hazard feature ablation on risk sets
whose FEATURES are computed from a DIC-noised field (σ = 0.05, comparable
to sig_tau = 0.08) while the LABELS remain the clean ground-truth events
(forecast true failure from a noisy observation). Catalog:
datasets/topo_synth_v1/catalog_noise05 (2000 cases). Artifacts:
runs_topo/phase2_hazard_noise/.

The clean ablation (topo_phase2_hazard_20260611.md) already passed; the
open question is whether the topological advantage over TRADITIONAL
methods survives measurement noise, which contaminates the topological
features (the event stream inflates ~5× under this noise). If the
topology only wins on clean simulation data, it is not a real method.

## Pre-registered criterion (committed BEFORE running the noisy ablation)

Under σ = 0.05 noise, topological features count as beating the
traditional baseline iff ablation (c) local+topo+history improves test
binary NLL over BOTH (a) local-damage-only AND every classical referee
(base rate, carry-forward two-bucket, hawkes-logistic) by a margin
exceeding the GBM 3-seed std, on ≥ 2 of the 3 horizons {3, 5, 10}.

Traditional method = local damage features + GBM/logistic, plus the
classical referees. New method = (c) with global topological curves and
topological event-history features. Same model classes, same protocol,
same seeds as the clean run; only the feature field is noised.

## VERDICT: PASS — 3/3 horizons, test AND OOD, advantage preserved under noise

The topological feature advantage over traditional methods survives
measurement noise essentially intact. My prior expectation (that noisy
topological features would erode the advantage more than the
noise-averaged local features) was WRONG — the data shows the gap is
preserved.

### GBM under σ=0.05 noise, TEST (mean over 3 seeds; std ≈ 1e-4)

| horizon | (a) local | (b) +topo curves | (c) +event history | best referee (hawkes-log) |
|---|---|---|---|---|
| 3 | 0.16058 | 0.15489 | **0.14744** | 0.16152 |
| 5 | 0.22140 | 0.21260 | **0.20190** | 0.22241 |
| 10 | 0.33089 | 0.31608 | **0.29871** | 0.32967 |

(c) beats (a) and the strongest classical referee on every horizon by a
margin 2–3 orders of magnitude above the seed std (1e-4). Ordering
(c) > (b) > (a) is preserved. top-1% recall: (c) 0.069/0.062/0.049 vs
(a) 0.047/0.043/0.035 — (c) better on all. OOD (held-out 4-notch) passes
identically: (c) 0.16183/0.22334/0.33167 vs (a) 0.17012/0.23435/0.34630
vs best referee 0.17132/0.23691/0.35211.

### The key finding — the advantage is preserved, not eroded

Topological advantage = NLL(a) − NLL(c), clean vs noised features:

| split | H | gap clean | gap noised |
|---|---|---|---|
| test | 3 | 0.01348 | 0.01313 |
| test | 5 | 0.01997 | 0.01950 |
| test | 10 | 0.03213 | 0.03218 |
| ood | 3 | 0.00797 | 0.00829 |
| ood | 5 | 0.01034 | 0.01101 |
| ood | 10 | 0.01325 | 0.01463 |

Both (a) and (c) degrade by the same amount under noise (all absolute
NLLs rise ~0.01–0.03), so the gap stays. On the test split it is
essentially identical clean-vs-noise; on the harder OOD geometry it even
widens slightly. The topological features carry signal the GBM still
extracts from a noisy field — the win is not a clean-simulation artifact.

Why this holds despite the event stream inflating ~9× under noise: the
global topological curves (counts/sums/entropy) partly average out
per-cell noise, and the GBM down-weights the noisier event-history
features rather than being misled by them. The honest mechanism is "the
robust part of the topological signal survives," not "noise doesn't
touch the topology" (it does — see Phase 2.1 noise report, where the
cumulative event signal failed outright).

## Claim boundary

Allowed: "On the synthetic multi-notch world, adding global topological
curves and event-history features to local damage features improves
tile-level hazard forecasting over local-only features AND over classical
referees (base rate, carry-forward, Hawkes-logistic), and this advantage
is preserved under DIC-like measurement noise (σ ≈ sig_tau), on all
horizons {3,5,10}, test and held-out-geometry OOD."

Not allowed: any real-data transfer; any operational-capability claim
from the absolute recall (5–7%); a claim that the topological FEATURES
are noise-immune (they are not — only their contribution to this GBM
hazard model is robust).

