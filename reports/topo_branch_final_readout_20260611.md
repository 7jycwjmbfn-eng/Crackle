# Crackle TDA branch — consolidated readout (spec v1.0 + addendum v1.1)

Date: 2026-06-11. Branch: topo-tda (12 commits on top of main d6d655e).
This consolidates all claim boundaries; per-phase detail lives in the
individual reports referenced below.

## Execution ledger

| spec item | status | report |
|---|---|---|
| §0.1 repo repair (.gitignore data/, crackle/data, import smoke test) | DONE | (commit history; tests/test_imports.py) |
| Phase 0 reverification | DONE — 12-case matrix reproduced digit-for-digit | runs_topo/phase0_noroi |
| Dataset: 2000 multi-notch + 200 @ 96×58 | DONE | datasets/* (dataset_card.md) |
| Phase 1.1 ROI | DONE — boundary events → ~0 | topo_phase1_roi_matching_audit_20260611.md |
| Phase 1.2 matching | DONE — optimal assignment; greedy disagreement ~0.3% | same |
| Phase 1.3 catalog + risk sets [KILL RULE 1] | **PASS** (0.407 ev/step) | topo_phase1_catalog_audit_20260611.md |
| Phase 2.1 causal onset | DONE — selectivity-crossover result | topo_phase2_causal_onset_20260611.md |
| Phase 2.1 noise robustness [pre-registered] | **PASS (qualified)** — topo leads control at matched FA under noise, all 3 σ; cum_events fails | topo_phase2_onset_noise_20260612.md |
| Phase 2.2 tabular ablation [pre-registered] | **PASS** | topo_phase2_hazard_20260611.md |
| Phase 2.2 ablation UNDER NOISE [pre-registered] | **PASS 3/3 test+OOD** — topo advantage over traditional methods preserved under noise | topo_phase2_hazard_noise_20260612.md |
| Track A neural TPP [pre-registered] | **PASS** (with OOD count caveat) | topo_phase2_track_a_20260611.md |
| Track B learned vectorization [pre-registered] | **FAIL — negative readout** | topo_phase2_track_b_20260611.md |
| Track C bond-graph GNN [pre-registered] | **PASS 3/3** — −35…−58% NLL vs strong referee, strengthens OOD | topo_phase2_track_c_20260611.md |
| Phase 3.1 DLR robustness | BLOCKED (no DLR run artifacts on this machine) | topo_phase3_realdata_20260611.md |
| Phase 3.2 real multi-crack data | INSPECTED — Rimkus retrieved (load–strain curves, not crack fields); no field data in any public source | same |

## What the branch now claims (synthetic multi-notch kinematic world only)

1. EVENT STREAM EXISTS AND IS ROBUST. Heterogeneous damage movies carry
   a dense topological event stream (0.41 ev/step over 2000 cases) that
   survives boundary-artifact removal, is insensitive to the matcher
   choice (~0.3% disagreement), and is resolution-stable (same 200
   worlds at 2× grid: mean events 33.1 vs 32.5, per-case r=0.76).
2. PHASE-0 LEADS WERE PARTLY ARTIFACT. Homogeneous-case precursor leads
   collapse 23 → 5 steps under ROI; died-interior events were
   contaminated by boundary-born features (−32…−47%); genuine interior
   births were being swallowed by boundary matches.
3. CAUSAL PRECURSOR CLAIM, QUALIFIED — AND NOW NOISE-TESTED. Under
   acceleration-selective causal detection at matched scale-free
   settings, topological signals alarm in ≥93% of cases with median
   leads 23–37 steps before t*; the macroscopic control detects ≤20% of
   cases there. Under DIC-like measurement noise (σ up to/above sig_tau)
   with a conservative instability-referenced false-alarm definition,
   the instantaneous persistence signal total_pers_h0 still beats the
   macroscopic control at MATCHED false-alarm rate on all three noise
   levels (median lead 12 vs 5/11.5/10 steps; 2–13× higher detection
   rate), under the rolling-z detector. Honest limits: the cumulative
   event count fails under noise; the advantage is detector-specific
   (CUSUM cannot make the macroscopic control selective at low noise);
   absolute detection rates are low. This is the strengthening the
   earlier degenerate-FA result needed.
4. TOPOLOGICAL FEATURES IMPROVE HAZARD FORECASTING — AND BEAT TRADITIONAL
   METHODS UNDER NOISE. Pre-registered PASS: local+topo+history >
   local+topo > local on test NLL and top-1% recall, every horizon
   {3,5,10}, GBM and logistic, margins 2–3 orders above seed std; beats
   every classical referee (base rate, carry-forward, Hawkes-logistic);
   ordering survives the held-out 4-notch OOD stratum. Re-run under
   DIC-like measurement noise (σ ≈ sig_tau, features noised, labels clean
   ground truth): the topological advantage over the traditional baseline
   is PRESERVED — the NLL gap (a)−(c) is essentially unchanged clean-vs-
   noise on test (0.013/0.020/0.032) and slightly widens on OOD. The win
   over traditional methods is not a clean-simulation artifact.
5. NEURAL TPP BEATS PARAMETRIC HAWKES (Track A). Pre-registered PASS on
   all LL components in-distribution; mark prediction (kind 0.51 vs 0.38
   acc) is the main gain. Honest caveats: count intensity loses to the
   referee OOD; time-rescaling KS rejects both models in most cases.
6. LEARNED DIAGRAM VECTORIZATION DOES NOT HELP (Track B, NEGATIVE).
   PersLay-style and fixed persistence images do not beat the
   hand-crafted Phase-0 scalar summaries under a matched protocol — the
   scalar curves remain the default representation.
7. BOND-GRAPH GNN BEATS TABULAR REFEREE (Track C, strongest positive).
   Pre-registered PASS 3/3: message passing on the native peridynamic
   bond graph cuts per-bond hazard NLL 35–58% vs a GBM given the same
   features plus one-hop aggregates; the advantage STRENGTHENS on the
   held-out 4-notch geometry — the cleanest positive transfer in the
   branch.

## What the branch does NOT claim

- Nothing about real data (Phase 3 blocked: registry leads were
  inspected and all three lack usable movie content; Rimkus DiB
  supplement is the manual-retrieval priority).
- No quantitative mechanics (the generator is a kinematic proxy).
- No operational forecasting utility (top-1% recall is 5–9%).
- No well-calibrated event-stream intensities (KS rejects).

## Next actions (in spec order)

1. Rimkus RC ties: public supplement is load–strain curves, not crack
   fields (now confirmed). To run the topological pipeline on it, request
   the original DIC field exports from the authors; alternatively use the
   retrieved curves as a real scalar control signal for an onset-only
   check (no topology).
2. Restore/regenerate DLR UNet mask artifacts → §3.1 bottleneck-distance
   robustness check.
3. Frontier line complete (Tracks A/B/C all run). Optional extensions:
   per-kind NTPP intensities to fix the KS goodness-of-fit; GNN on a
   larger bond dataset (current Track C used 400 cases).
