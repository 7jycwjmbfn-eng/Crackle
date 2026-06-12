# Topo Phase 2.2 Track A — neural TPP vs parametric Hawkes

Date: 2026-06-11. Branch: topo-tda. Spec: addendum v1.1 B, Track A.
Artifacts: runs_topo/phase2_track_a/ (track_a_results.csv, config.json,
curves_cache.npz).

## Pre-registered criterion (committed before the run; see git history of
scripts/topo_track_a_ntpp.py)

The discrete-time transformer-Hawkes model must beat the parametric
Hawkes referee on TEST total log-likelihood per case AND on >= 2 of the
3 component LLs (count, kind, tile), each by more than the 3-seed std.

VERDICT: **PASS — all four LLs beat the referee, margins 5 to 140x the
seed std.**

## Setup

Event streams from the Phase 1.3 catalog (65k events, 2000 cases; splits
train 1052 / val 231 / test 222 / OOD 495 = all 4-notch). Model:
2-layer causal transformer (d=64) over (kind, tile, step) embeddings,
conditioned on global topo curves at t-1; heads emit per-step Poisson
intensity + kind/tile multinomials. Referee: exponential-kernel Hawkes
(grid MLE on train: mu=0.093, alpha=0.186, beta=0.196 — genuine
self-excitation in the event stream) + empirical train mark
distributions, identical LL decomposition. 3 seeds, 12 epochs,
val-LL model selection (~80 s/seed on the RTX 4080).

## Results (mean ± std over 3 seeds)

TEST:

| metric | discrete THP | parametric Hawkes |
|---|---|---|
| total LL/case | **−198.68 ± 0.29** | −213.54 |
| count LL/case | **−57.08 ± 0.02** | −58.58 |
| kind LL/case | **−35.20 ± 0.06** | −43.53 |
| tile LL/case | **−106.41 ± 0.23** | −111.43 |
| kind accuracy | **0.507 ± 0.001** | 0.379 (majority) |
| tile accuracy | **0.133 ± 0.002** | 0.076 (majority) |
| KS frac (p>0.05) | 0.413 ± 0.024 | 0.310 |

OOD (held-out 4-notch geometry):

| metric | discrete THP | parametric Hawkes |
|---|---|---|
| total LL/case | **−215.82 ± 0.41** | −225.99 |
| count LL/case | −67.56 ± 0.38 | **−66.31** |
| kind LL/case | **−39.69 ± 0.19** | −45.06 |
| tile LL/case | **−108.57 ± 0.14** | −114.63 |

## Reading

- The biggest neural gain is in the MARKS (kind +8.3 nats/case, tile
  +5.0 nats/case): event history + topo covariates carry real
  information about what kind of topological change comes next and
  where. Count timing improves modestly in-distribution (+1.5).
- OOD caveat (report-worthy, not criterion-relevant): under the unseen
  4-notch geometry the neural COUNT intensity degrades below the
  Hawkes referee (−67.6 vs −66.3) while the mark heads keep their lead.
  The learned intensity is partly geometry-tuned; the parametric
  referee's three global parameters transfer better for raw counts.
- Time-rescaling KS: both models fail goodness-of-fit in most cases
  (THP 41%, Hawkes 31% of cases with p>0.05); neither intensity model
  fully captures the burstiness around t*. Honest open issue for any
  follow-up (e.g., per-kind intensities, count distributions beyond
  Poisson).

## Claim boundary

Allowed: "On the synthetic multi-notch world, a small causal transformer
TPP beats a fitted exponential-kernel Hawkes referee on held-out total
event-stream likelihood and on mark prediction (kind/location), beyond
seed noise; the count-timing advantage does not survive the OOD
geometry stratum."

Not allowed: any real-data claim; any claim of well-calibrated
intensities (KS rejects both models in most cases); comparisons against
stronger neural baselines that were not run.
