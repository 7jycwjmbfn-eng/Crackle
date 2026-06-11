# Topo Phase 1.1/1.2 — ROI mask + matching upgrade audit

Date: 2026-06-11. Branch: topo-tda. Spec: crackle_tda_spec.md §1.1–1.2.
Matrix: Phase 0 synthetic reference (contrasts 1/3/5 × corr small/medium ×
2 seeds, 48×29, 80 steps, sig_tau=0.08, match_dist=6).
Runs: `runs_topo/phase1_roi_{off,on}{,_wass}` (not committed, regenerable via
`python -m scripts.phase0_topo_audit --synthetic --out ... [--roi-margin-k 1.5]
[--matcher greedy|wasserstein]`).

## 1. ROI mask (§1.1)

ROI = exclude 1.5 × horizon (6 rows at ny=29) from top/bottom edges, at the
diagram level (birth-cell filter; field untouched; essential H0 kept).

| case | events off (bnd/int) | events on (bnd/int) |
|---|---|---|
| c1_*_s* (×4) | 15 (6/9) | 9 (0/9) |
| c3_small_s0 | 127 (68/59) | 44 (1/43) |
| c3_small_s1 | 133 (62/71) | 61 (1/60) |
| c3_medium_s0 | 62 (34/28) | 27 (1/26) |
| c3_medium_s1 | 51 (30/21) | 21 (2/19) |
| c5_small_s0 | 130 (71/59) | 50 (1/49) |
| c5_small_s1 | 118 (56/62) | 52 (0/52) |
| c5_medium_s0 | 62 (38/24) | 23 (0/23) |
| c5_medium_s1 | 56 (27/29) | 27 (2/25) |

Acceptance check 1 — boundary-hugging events drop to ~0: PASS.
6–71 boundary events/case → 0–2 (the stragglers are death-located rows of
interior-born features, which is the documented attribution convention).

Acceptance check 2 — interior counts change <15%: FAIL as literally written,
and the failure is informative. Interior totals move −4% to −27%. Splitting
by event class explains it:

| case | born-interior off→on | died-interior off→on |
|---|---|---|
| c1 (×4) | 5 → 5 (0%) | 4 → 4 (0%) |
| c3_small_s0 | 21 → 23 (+10%) | 38 → 20 (−47%) |
| c3_small_s1 | 30 → 32 (+7%) | 41 → 28 (−32%) |
| c3_medium_s0 | 8 → 14 (+75%) | 20 → 12 (−40%) |
| c3_medium_s1 | 8 → 11 (+38%) | 13 → 8 (−38%) |
| c5_small_s0 | 24 → 26 (+8%) | 35 → 23 (−34%) |
| c5_small_s1 | 25 → 27 (+8%) | 37 → 25 (−32%) |
| c5_medium_s0 | 5 → 12 (+140%) | 19 → 11 (−42%) |
| c5_medium_s1 | 12 → 14 (+17%) | 17 → 11 (−35%) |

Two mechanisms, both of which are the artifact being removed rather than
collateral damage:

1. died-interior collapses (−32…−47%): boundary-BORN features merge (die)
   in the interior; their h0_died rows carry interior death coordinates, so
   without ROI they contaminate the interior count. Birth-location filtering
   removes them wholesale.
2. born-interior RISES (+7…+140%, strongest at corr=medium): without ROI,
   genuine interior nucleations were getting matched to boundary-band
   features of the previous frame (within match_dist of the margin), so
   their birth events were swallowed. ROI unmasks them.

Verdict: ROI-on becomes the default for all downstream phases. The <15%
expectation in the spec was written assuming interior counts were clean;
they were not — that is the point of the fix.

Side observation (matters for Phase 2): in the four homogeneous c1 cases,
best lead drops 23 → 5 steps under ROI. A large part of the Phase 0
"precursor lead" in the homogeneous world was boundary artifact. The
heterogeneous best leads survive (15–40 steps, retrospective detector).

Event density after ROI (greedy or wasserstein): heterogeneous mean
0.48 ev/step (range 0.26–0.76). Kill rule 1 threshold (0.3 ev/step) is
cleared on this matrix, pending the generated multi-notch dataset.

## 2. Matching upgrade (§1.2)

Implementation: `crackle/topo/matching.py`. method="wasserstein" = optimal
assignment (scipy Hungarian) on the Phase-0 cost matrix (spatial distance +
value_weight·|Δbirth|, gated at max_dist) augmented with per-feature
unmatch cost = max_dist — the unbalanced-OT construction. persim was NOT
used: its wasserstein/bottleneck match in (birth, death) diagram space and
discard birth location, which is the wrong geometry for spatial identity
tracking (documented choice per spec). Greedy retained as method="greedy";
default switched to "wasserstein".

Unit tests (tests/test_topo_matching.py): merge event, loop birth, moving
feature (must match, not die+born), and a swap-prone pair on hand-built
fields where greedy provably produces a spurious died+born pair and
wasserstein matches both. All pass (22 tests total in the suite).

Disagreement on the Phase 0 matrix (events keyed by step/kind/y/x):

| regime | greedy events | wasserstein events | disagreeing rows |
|---|---|---|---|
| ROI on | 341 | 341 | 2 (1 pair, c5_small_s0) |
| ROI off | 799 | 799 | 2 (1 pair, c5_small_s0) |

The spec's expectation ("greedy over-counts born/died pairs") does NOT
materialize on this matrix: disagreement rate ≈ 0.3% of events, totals
identical. At 48×29 with max_dist=6 and ROI-filtered (sparse, well
separated) features, greedy is almost always already optimal. The failure
mode is real (unit test) but rare in this world. Wasserstein stays the
default because it is never worse and costs microseconds at these diagram
sizes; conclusions are matcher-robust either way.

## Claim boundary

Allowed: boundary artifact events are eliminated by diagram-level ROI
filtering; the Phase-0 interior event population was contaminated in both
directions (death-located boundary features added, interior births
swallowed); event extraction is robust to greedy-vs-optimal matcher choice
on this matrix; homogeneous-case precursor leads were substantially
boundary artifact.

Not allowed: any forecasting claim (onset detector is still retrospective
here); transfer of these counts to the multi-notch generated dataset or
real data (audited next).
