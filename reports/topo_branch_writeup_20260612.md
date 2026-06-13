# Topological event forecasting for crack damage — branch writeup

Date: 2026-06-12. Branch: topo-tda. This is the consolidated narrative of
the TDA branch: what it does, every pre-registered result (positive and
negative), and the consolidated claim boundary. Per-phase detail and the
exact run commands live in the individual `reports/topo_*` files; every
number here is script-derived from a committed run.

## Abstract

Crackle's original forecasting target — crack-tip coordinates — is
saturated by a trivial persistence baseline. We replace it with the
**topological events** of the damage field: discrete jumps in the
superlevel-set topology (hotspot nucleation `h0_born`, coalescence
`h0_died`, loop birth/death `h1_*`). On a 2000-case synthetic multi-notch
world we show the event stream is dense and robust; that topological
features improve event-hazard forecasting over non-topological baselines
and classical referees, **and that this advantage survives measurement
noise**; that a neural temporal point process beats a parametric Hawkes
referee; and that a bond-graph GNN beats a strong gradient-boosted referee
on the native representation. One pre-registered idea (learned diagram
vectorization) returns a clean negative. Finally, on a **real**
reinforced-concrete slab (Harb et al., 25 load epochs) the synthetic
event grammar — h0_born-dominated nucleation, then H1 loop formation —
is reproduced. The work is a methods + synthetic-study + first-real-data
archive with an explicit, honest claim boundary.

## 1. Pipeline

Damage-field movie `(T, ny, nx)` →
- **superlevel cubical persistence** per frame (cripser/tcripser,
  T-construction: 8-connectivity for cracks, 4 for material islands);
- **per-frame summaries** (Betti curves, persistence entropy, total
  persistence) — `crackle/topo/features.py`;
- **ROI filtering** at the diagram level (drop boundary-band artifacts by
  birth location; never zero the field) — `crackle/topo/roi.py`;
- **frame-to-frame matching** → event stream; default `wasserstein`
  (Hungarian on a spatial+value cost with unmatch cost = max_dist),
  `greedy` retained — `crackle/topo/matching.py`;
- **event catalog + tile risk sets** with strictly past-only features and
  ground-truth labels — `crackle/topo/catalog.py`.

Generator: a multi-notch randomized **kinematic proxy** (prescribed
displacement + bond breaking; not a solved mechanics model — valid for
topology/event mining, invalid for quantitative mechanics).

## 2. Phase 1 — the event stream exists and is robust

On 2000 generated cases (`datasets/topo_synth_v1`): 65,138 events,
**0.407 events/step mean** (kill rule ≥0.3 → PASS). Composition: h0_born
24,741 / h0_died 20,625 / h1_born 10,884 / h1_died 8,888.

- **ROI** removes the boundary artifacts: boundary-born events 6–71/case
  → 0–2. (It also corrected a Phase-0 over-claim: a large part of the
  homogeneous-case "precursor lead" was boundary artifact, 23 → 5 steps.)
- **Matcher** robustness: greedy vs wasserstein disagree on ~0.3% of
  events — conclusions are matcher-independent at this scale.
- **Resolution** robustness: the same 200 worlds at 48×29 vs 96×58 give
  mean events 33.1 vs 32.5 and per-case correlation r = 0.76.

## 3. Phase 2 — does the topology beat traditional methods?

### 3.1 Causal precursor (onset detection)

Noiseless world: under acceleration-selective causal detection at matched
scale-free settings, topological signals alarm in ≥93% of cases with
median leads 23–37 steps before instability, vs the macroscopic control's
≤20% detection there. (At permissive settings the control's earlier alarm
is ramp-following — 97.6% within 3 steps of growth onset — and carries no
timing content.)

**Under DIC-like measurement noise** (σ ∈ {0.02, 0.05, 0.10} vs
sig_tau 0.08; corrected, conservative instability-referenced false-alarm
definition): at matched false-alarm rate the instantaneous persistence
signal `total_pers_h0` beats the macroscopic control on **all three**
noise levels — median lead 12 vs 5 / 11.5 / 10 steps, detection rate
13× / 5× / 2.3× the control. Honest negative within the same test: the
**cumulative** event count fails under noise (flicker accumulates);
the advantage is detector-specific (CUSUM cannot make the macroscopic
control selective at low noise).

### 3.2 Hazard ablation — the headline "beats traditional" result

Tile-level event-hazard forecasting, GBM and logistic, vs classical
referees (base rate, carry-forward, Hawkes-logistic). Ablations:
(a) local damage only, (b) + global topo curves, (c) + event history.

Clean test NLL (GBM, 3-seed mean; std ~1e-4):

| H | (a) | (b) | (c) | best referee |
|---|---|---|---|---|
| 3 | 0.1466 | 0.1399 | **0.1331** | 0.1552 |
| 5 | 0.2028 | 0.1925 | **0.1828** | — |
| 10 | 0.3044 | 0.2859 | **0.2722** | — |

(c) > (b) > (a) on every horizon and both metrics, margins 2–3 orders
above seed std; all referees beaten; ordering survives the held-out
4-notch OOD stratum. **Pre-registered PASS.**

**Under σ=0.05 noise** (features noised, labels = clean ground truth):
the advantage is *preserved*. The gap NLL(a)−NLL(c) is essentially
unchanged clean→noise on test (0.0135→0.0131, 0.0200→0.0195,
0.0321→0.0322 at H3/5/10) and widens slightly OOD; (c) still beats the
strongest referee on all horizons. **The win over traditional methods is
not a clean-simulation artifact.**

### 3.3 Frontier tracks

- **Track A — neural temporal point process** (discrete-time
  transformer-Hawkes) vs parametric Hawkes referee. Test: total LL
  −198.7 vs −213.5; kind LL −35.2 vs −43.5; kind accuracy 0.507 vs 0.379;
  tile accuracy 0.133 vs 0.076. **Pre-registered PASS.** Honest caveats:
  count intensity loses to the referee OOD; time-rescaling KS rejects both
  models in most cases. [Noise rerun: see §3.4.]
- **Track B — learned diagram vectorization** (PersLay-style). Test NLL
  vs hand-crafted scalar curves and fixed persistence images: perslay
  0.1533/0.2118/0.3171 never beats scalar 0.1527/0.2086/0.3103.
  **Pre-registered FAIL — clean negative.** Hand-crafted Phase-0 summaries
  suffice; learned vectorization adds nothing here.
- **Track C — bond-graph GNN** vs XGBoost referee (same features + 1-hop
  aggregates) on the native peridynamic graph. Per-bond hazard test NLL
  cut **35–58%** (H3 0.0048 vs 0.0074, H5 0.0060 vs 0.0118, H10 0.0105 vs
  0.0219); advantage **strengthens OOD** (−51…−58%). **Pre-registered
  PASS — the strongest positive; the only track winning on the model's
  native representation.**

### 3.4 Noise robustness of Track A

Re-run on the σ=0.05 noisy catalog (noisy event stream AND noisy curve
covariates): the neural TPP still beats the parametric Hawkes referee.
Test total LL −1450.0 vs −1517.3, count −151.4 vs −155.6, kind −298.7 vs
−353.0, tile −999.9 vs −1008.7 — all four beaten by > seed std; kind
accuracy 0.509 vs 0.409. (Per-case LL is far more negative than the clean
run because noise inflates the event stream ~9×; the comparison is what
matters.) The same honest OOD caveat as the clean run survives: the
neural count intensity does not beat the referee OOD (−156.4 vs −155.1)
while the mark heads keep their lead. So **all three Phase-2 method
families — hazard features (2.2), onset (2.1), and the neural TPP — keep
their advantage over the classical/traditional baselines under
measurement noise.**

## 4. Phase 3 — real data

**Reproduced.** Harb et al. multi-temporal crack segmentation dataset
(Zenodo 15187675, CC-BY-4.0): one RC slab, 25 load epochs of high-res
binary crack masks. The synthetic event grammar reproduces:
- **h0_born dominates** (1888 of 3517 events at the headline setting);
- **nucleation precedes loop formation** (h0_born median epoch 13.0 vs
  h1_born 19.0; stable across downscale {8,16,24}, sig_tau {0.05,0.08},
  both matchers);
- **both Betti numbers grow monotonically** with load (H0 492→953,
  H1 3→53); the final field is a recognisable RC flexural-crack network.

n = 1 specimen (reproduction-in-principle, not cross-specimen statistics);
masks are DL segmentations; event *rates* are not comparable to synthetic
(only the kind distribution and ordering transfer). Phase 3.1 (DLR mask
robustness) remains blocked for lack of paired pred/ref artifacts; other
public registry leads (KTH, craquelure, desiccation, Rimkus) were
inspected and lack per-frame crack FIELD data — Rimkus's authors were
emailed for the DIC field exports.

### 4.1 Solved-mechanics cross-check (addresses the kinematic-proxy caveat)

To test whether the h0 nucleation grammar is an artifact of the kinematic
proxy, we ran the audit on **solved peridynamics** damage fields (the
gaussmoe dense-PD archive; `load_case_npz` now reads `reference_x`):
- single-notch dense-PD (8 cases, 71×29, damage→1.0): **0 significant
  events** — a single propagating crack is one growing component with no
  nucleation or loops. This confirms the "single crack is topologically
  trivial" boundary in *solved* mechanics, not just the kinematic proxy.
- heterogeneous hard-bench (16 cases, 103×40, toughness contrast 1.7):
  h0_born events DO appear in every case, and scale with crack branching
  — `branching_candidate` cases show 3 h0_born vs `arrest_candidate`
  cases' 1 (more crack tips → more nucleation, the topological reading's
  physical prediction). But they are sparse (mean 2.5 events/case, no H1
  loops) because these are few-crack, mildly-heterogeneous scenarios.

Takeaway: the h0_born nucleation grammar is **not** a kinematic-proxy
artifact (it appears in solved PD and scales sensibly with branching);
the *rich* grammar (H1 loops, coalescence cascades, dense event streams)
requires many simultaneous cracks — which the multi-notch synthetic world
and the real RC slab provide, but these few-crack solved-PD benchmarks do
not.

## 5. Consolidated claim boundary

**Allowed.** On a synthetic multi-notch kinematic world: a dense,
boundary-clean, resolution- and matcher-robust topological event stream
exists; topological features improve event-hazard forecasting over
non-topological features AND classical referees, on clean data and under
measurement noise, in- and out-of-distribution; a neural TPP beats a
parametric Hawkes referee; a bond-graph GNN beats a strong tabular referee
on the native representation; learned diagram vectorization does not help.
On real RC data: the synthetic event grammar (h0_born-dominated
nucleation → loop formation) is reproduced on one slab and is robust to
the mask-processing parameters.

**Not allowed.** Quantitative mechanics (kinematic proxy); operational
forecasting capability (top-1% recall 5–9%); calibrated event intensities
(KS rejects); cross-specimen real-data statistics (n=1); any real-data
forecasting claim (Phase 3 is a descriptive audit).

## 6. What would change the verdict / next steps

- Real DIC **field** exports (Rimkus, requested) → run Phase 0/1 on a
  second real specimen; multi-specimen reproduction would lift the n=1
  caveat.
- A solved-mechanics generator (true peridynamics) would remove the
  kinematic-proxy limitation on the synthetic claims.
- Per-kind NTPP intensities / non-Poisson counts to address the
  time-rescaling KS failure.
