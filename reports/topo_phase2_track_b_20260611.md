# Topo Phase 2.2 Track B — learned diagram vectorization (NEGATIVE)

Date: 2026-06-11. Branch: topo-tda. Spec: addendum v1.1 B, Track B.
Artifacts: runs_topo/phase2_track_b/ (track_b_results.csv,
diagram_cache.npz, config.json).

## Pre-registered criterion (committed before the run; see git history of
scripts/topo_track_b_perslay.py)

The PersLay-style learned vectorization must beat BOTH the fixed
persistence image AND the hand-crafted scalar curves on TEST binary NLL
by more than the 3-seed std on >= 2 of the 3 horizons.

VERDICT: **FAIL on all 3 horizons — negative result, reported as such.**

## Setup

Identical hazard head (MLP), local features, optimizer, epochs, seeds
across variants; the per-frame topology representation is the only
variable. scalar = 12 hand-crafted curve features; pi_fixed = 8×8×2
persistence image; perslay = sum-pooled learned point transform with
persistence-dependent weights, trained end-to-end.

## TEST NLL (mean ± std over 3 seeds)

| horizon | scalar | pi_fixed | perslay |
|---|---|---|---|
| 3 | 0.15274 ± 0.0009 | **0.15254 ± 0.0002** | 0.15331 ± 0.0009 |
| 5 | **0.20858 ± 0.0008** | 0.20952 ± 0.0003 | 0.21181 ± 0.0014 |
| 10 | **0.31031 ± 0.0010** | 0.31141 ± 0.0005 | 0.31706 ± 0.0014 |

top-1% recall shows the same picture (scalar 0.061/0.055/0.042 vs
perslay 0.060/0.051/0.040). The three representations sit within ~2% of
each other; the learned one is never ahead.

## Reading

For THIS task (tile-level hazard with the diagram entering as a global
per-frame covariate) and THIS world, the hand-crafted Phase-0 summaries
already extract what the significant diagram carries; a learned
permutation-invariant encoder finds nothing extra and pays a small
optimization tax. The Phase-0 design choice (scalar curves) is
vindicated as the default representation going forward.

Scope notes, per the no-torture rule: no tuning beyond the
pre-committed protocol was attempted; the result licenses "no free win
from learned vectorization here", NOT "PersLay cannot help anywhere"
(e.g., per-tile local diagrams or real noisy data could change the
picture — untested).

Cross-reference: the MLP head used here is weaker than the GBM of the
main 2.2 ablation (scalar-MLP 0.153 vs GBM(b) 0.140 at H3) — model
class and representation questions are orthogonal; Track B holds the
model class fixed by design.

## Claim boundary

Allowed: "Under a matched protocol on the synthetic world, learned
diagram vectorizations (PersLay-style and fixed persistence images) do
not improve tile-hazard NLL over the hand-crafted scalar summaries."

Not allowed: generalization of this negative to other tasks, encoders,
tuning budgets, or real data.
