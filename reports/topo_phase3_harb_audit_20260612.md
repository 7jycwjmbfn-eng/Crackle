# Topo Phase 3.2 — real multi-crack RC data: topological grammar reproduced

Date: 2026-06-12. Branch: topo-tda. Spec: crackle_tda_spec.md §3.2 +
addendum v1.1. **First real-data result for the branch.**

Dataset: Harb, Jiang, Lazaros et al., *Multi-temporal Crack Segmentation
Dataset* (Zenodo 15187675, CC-BY-4.0; ISPRS Annals 2025 / arXiv:2411.04620).
One reinforced-concrete slab loaded incrementally to failure; high-res
binary crack-segmentation masks at 25 load epochs (`target_epoch_00_01`
… `target_epoch_30_31`, 11264×8192 px each, crack = white). Loader:
`crackle.topo.load_mask_sequence` (block-average downscale → crack-density
field; topology is scale-robust, see the res96 probe). Audit:
`scripts/topo_realdata_audit.py`. Artifacts: runs_topo/phase3_harb/
(summary.json, topo_events.csv, audit figure committed as
reports/topo_phase3_harb_audit_20260612.png).

## The question (spec §3.2)

"Is the topological event grammar observed in synthetic heterogeneous
fields (nucleation → coalescence/loop ordering, h0_born-dominated)
reproduced in real multi-crack RC data?" Either verdict is a result.

## VERDICT: reproduced.

Headline run (downscale 16 → 512×704, sig_tau 0.05, wasserstein matcher,
25 epochs, 19 s):

| quantity | value |
|---|---|
| total topological events | 3517 |
| h0_born / h0_died | **1888 / 1427** |
| h1_born / h1_died | 126 / 76 |
| # sig H0 (crack components) first → last | 492 → 953 |
| # sig H1 (enclosed loops) first → last | 3 → 53 |
| crack (white) area fraction first → last | 0.94% → 2.23% |

Three predictions from the synthetic world, all confirmed on the real
slab:

1. **h0_born dominates** (1888, the largest category by 1.3–25×) — the
   stream is sequential crack nucleation, exactly the synthetic
   prediction ("h0_born-dominated sequential nucleation").
2. **Nucleation precedes loop formation.** Event timing (epoch index
   1–24): h0_born median epoch **13.0**, h1_born median epoch **19.0** —
   loops close ~6 epochs after the bulk of nucleation. H1 events: 90%
   after epoch 4. This is the nucleation → coalescence → loop-formation
   ordering, in real data.
3. **Both Betti numbers grow monotonically with load** — H0 492→953 (the
   network fragments and densifies), H1 3→53 (cracks coalesce into a net
   enclosing regions). The final crack-density field (figure, bottom
   right) is a recognisable RC flexural-cracking network: a central
   vertical crack feeding branching horizontal cracks.

## Honest scope and caveats

- **n = 1 specimen.** This is ONE real loading sequence reproducing the
  qualitative grammar — not a distribution over cases like the 2000-case
  synthetic claim. It establishes reproduction-in-principle, not
  cross-specimen statistics.
- **Event RATES are not comparable to synthetic** (147 ev/step here vs
  ~0.4 ev/step synthetic). The real masks are far larger, higher-
  resolution dense networks with thousands of fragments; only the KIND
  distribution and temporal ORDERING transfer, and those are what the
  claim is about.
- **The masks are deep-learning segmentations, not ground truth.** The
  topology inherits any segmentation fragmentation (which inflates H0)
  and registration jitter (the crack-fraction dip at epoch ~19–20 is most
  likely re-segmentation variation, not crack healing). The qualitative
  grammar is robust to this; absolute counts are not to be over-read.
- **Parameter choice** (downscale 16, sig_tau 0.05) is to make the 92-MP
  masks tractable. Robustness to this choice is reported below.

## Robustness to the grid/threshold choice

| config | grid | h0_born/died | h1_born/died | H0 first→last | H1 first→last | h0_born med epoch | h1_born med epoch | nucl<loop |
|---|---|---|---|---|---|---|---|---|
| ds16 τ0.05 (headline) | 512×704 | 1888/1427 | 126/76 | 492→953 | 3→53 | 13.0 | 19.0 | ✓ |
| ds8 τ0.05 (greedy) | 1024×1408 | 5619/4655 | 378/270 | 1110→2074 | 4→112 | 13.0 | 19.0 | ✓ |
| ds24 τ0.05 | 341×469 | 1068/809 | 59/37 | 258→517 | 1→23 | 13.0 | 19.0 | ✓ |
| ds16 τ0.08 | 512×704 | 1732/1350 | 76/41 | 377→759 | 0→35 | 13.0 | 19.0 | ✓ |

Every setting reproduces all three findings: h0_born dominance, monotone
growth of both Betti numbers, and nucleation-before-loop ordering — with
the h0_born / h1_born median epochs identically 13.0 / 19.0 across
downscale ∈ {8, 16, 24} and sig_tau ∈ {0.05, 0.08}, and across both
matchers (greedy at ds8, wasserstein elsewhere). The result is not an
artifact of one parameter setting. (Absolute counts scale with grid
resolution, as expected; the grammar does not.)

## Claim boundary

Allowed: "The topological event grammar characterised in the synthetic
heterogeneous multi-notch world — h0_born-dominated sequential nucleation
followed by H1 loop formation as the crack network closes — IS reproduced
in a real reinforced-concrete slab cracking sequence (Harb et al., 25
load epochs), and is robust to the mask-downscaling and significance
threshold."

Not allowed: cross-specimen statistical claims (n=1); any forecasting
claim on real data (this is a descriptive topological audit); reading the
absolute event counts as physical (they reflect DL-segmentation
granularity); any mechanics claim.
