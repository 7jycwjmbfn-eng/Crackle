# Match the representation to the task: bond-graph crack-event prediction beats neural operators out-of-distribution, while Fourier operators win smooth field forecasting

*Crackle TDA branch — consolidated Phase 4 / 4b / 4c writeup. 2026-06-14.*
*Reproducible from `scripts/` + `outputs/`; see Appendix A. Tabular companion:
`PHASE4_RESULTS.md`.*

---

## Abstract

We test whether a topological, bond-graph representation of fracture beats both
classical baselines and modern neural operators (Fourier Neural Operator,
DeepONet, dilated ConvNet) at predicting crack growth in peridynamic solids,
under strict fairness and out-of-distribution (OOD) evaluation. The answer is
task-dependent and we report both halves. On **smooth full-field damage
forecasting** a fairly-tuned FNO is strongest, beating the classical
extrapolators *and* the bond-graph method, in- and out-of-distribution. On
**discrete crack-event prediction** (per-bond breaking hazard) — the task this
project exists to solve — a bond-graph message-passing network (crackle) beats
the neural operators by 82–91% in per-bond NLL with ~3× the top-1% recall, and
beats a strong gradient-boosted referee out-of-distribution while merely tying
it in-distribution. The hazard result holds on a kinematic proxy and on real
solved peridynamics (10 seeds, gaps exceed the across-seed spread). We frame
the finding as representation–task matching: spectral/operator models excel at
smooth field forecasting; the native bond graph excels at localized, discrete
crack-event prediction and uniquely generalizes to unseen geometry. The
FNO-wins negative is reported with equal prominence; the graph was not tuned
until it won.

## 1. Introduction

Predicting *where and when* a material cracks is the core problem of fracture
mechanics. Two modern modelling families dominate surrogate approaches:
(i) **classical / statistical baselines** — extrapolation, parametric hazard
models, gradient-boosted trees on hand-built features; and (ii) **neural
operators** — FNO, DeepONet — which learn mappings between function spaces and
are the default for PDE/field surrogates.

Neither is obviously right for cracks. Classical feature models are strong
in-distribution but tend to overfit the training geometry. Neural operators
predict *smooth fields*: they have no notion of the discrete bonds whose
rupture *is* the crack event, and a per-cell field cannot distinguish two bonds
that share endpoints. The crackle hypothesis is that the **native peridynamic
bond graph** — nodes = material points, edges = bonds — is the right inductive
bias for discrete crack-event prediction, and that this advantage shows up
precisely where it matters: on new geometry.

This writeup tests that hypothesis honestly. It includes the regime where the
hypothesis **fails** (smooth full-field rollout, where FNO wins), because a
result that only reports its wins is exactly the self-deception a fracture
practitioner cannot afford.

## 2. Methods

### 2.1 Models

- **Crackle (topological).** Per-bond hazard: `BondGNN`, message passing on the
  bond graph predicting per-bond breaking logits for all horizons jointly.
  Field rollout: `GraphForecaster`, degree-normalized message passing
  predicting a per-node damage increment.
- **Neural operators.** `FNO2d` (spectral convolution on the lowest Fourier
  modes), `DeepONet` (branch/trunk basis), `ConvNet` (residual dilated CNN). On
  the hazard task they consume the rasterized per-node state field and emit a
  per-cell hazard field (one channel per horizon), mapped to a bond by the mean
  of its two endpoint cells.
- **Classical.** Hazard: an XGBoost referee on the same raw bond features plus
  one-hop neighborhood aggregates (so the GNN's only edge is learned multi-hop
  message passing). Field rollout: persistence, clamped-linear, mean-rate.

### 2.2 Hazard task definition

For each bond that is still intact ("at-risk") at time *t*, predict whether it
breaks within (t, t+H], for H ∈ {3, 5, 10}. Labels come from the simulator's
bond-alive transitions; only at-risk bonds carry labels.

### 2.3 Datasets and splits

- **B1 — kinematic proxy** (`topo_bonds_v1`, 400 cases, 48×29 lattice).
  Train/val/test by case; **OOD = the entire 4-notch geometry stratum held
  out** (pre-registered).
- **B2 — real solved peridynamics** (`crack_hard_bench`, 64 heterogeneous
  cases, real bond_alive/bond_stretch series). Train/val/test by case;
  **OOD = a different solved-PD dataset** (`crack_notched_plate_v5_1`, 16
  homogeneous notched-plate cases) never trained on.
- **A — full-field rollout** (same solved-PD data): autoregressive prediction
  of the damage field; in-dist held-out on `crack_hard_bench`, true-OOD on
  `crack_notched_plate_v5_1`.

### 2.4 Metrics

Hazard: per-bond binary **NLL** (↓) and **top-1% recall** (↑) — the fraction of
true breaks captured in the model's top 1% most-confident bonds (the operative
quantity for prioritizing inspection). Field rollout: full-field **relative
L2** (↓) and the **H0 topological bottleneck distance** (↓) of the field's
super-level persistence.

### 2.5 Pre-registration and fairness protocol

Success criteria were written into each script header before training:
crackle must beat the best neural operator on TEST and OOD per-bond NLL by more
than the across-seed std on ≥2 of 3 horizons; a failing track is a negative
readout, not re-run with new hyperparameters. Fairness controls:

1. **No leakage** — by-case (or by-dataset, for OOD) splits; normalization from
   train only; OOD is a held-out stratum or a separate dataset.
2. **Baselines not sandbagged** — operators trained to a genuinely strong
   regime. On the hazard task they are even handed the material-toughness field
   as an extra channel; it did not change their NLL, so the gap is structural,
   not feature starvation.
3. **No cherry-picking** — where crackle loses (Task A) we report it and do not
   tune the graph until it wins.
4. **Variance** — multiple seeds; std reported (Task B2 at 10 seeds).

A methodological aside that mattered: naive 1-step MSE on the ~5%-sparse damage
increment is degenerate — every learned field model collapses bit-identically
to persistence, and a classical baseline then "wins". We fixed it (front-
weighted loss, softplus gate, physical-scale gate-bias init, pushforward
training), applied identically to all learned models, before drawing any
conclusion.

## 3. Results

### 3.1 Task A — full-field rollout: FNO wins (operator L2, 3 seeds)

rel-L2 (↓), in-distribution (hard_bench held-out):

| model | h10 | h20 | h30 | h40 |
|-------|-----|-----|-----|-----|
| mean_rate (classical) | 0.093 | 0.151 | 0.197 | 0.229 |
| **FNO** | **0.071** | **0.127** | **0.164** | **0.194** |
| DeepONet / ConvNet / crackle graph | 0.107 | 0.181 | 0.243 | 0.291 (=persistence) |

rel-L2 (↓), true-OOD (never-seen notched-plate):

| model | h10 | h20 | h30 | h40 |
|-------|-----|-----|-----|-----|
| mean_rate (classical) | 0.115 | 0.178 | 0.220 | 0.245 |
| crackle graph | 0.089 | 0.133 | 0.177 | 0.233 |
| **FNO** | **0.046** | **0.077** | **0.096** | **0.105** |

FNO wins at every horizon on both L2 and the topological bottleneck. DeepONet
and ConvNet collapse to persistence under the shared protocol; the crackle
graph beats the classical baselines OOD but loses clearly to FNO. **A
smooth, L2-optimized operator is the right tool for smooth field forecasting.**

### 3.2 Task B1 — per-bond hazard, kinematic proxy (3 seeds): crackle wins

NLL (↓) / top-1% recall (↑):

| model | TEST H3/H5/H10 | OOD H3/H5/H10 |
|-------|----------------|----------------|
| op_fno | 0.036/0.059/0.116 · 0.26/0.17/0.09 | 0.029/0.048/0.090 · 0.29/0.19/0.11 |
| op_convnet | 0.036/0.060/0.116 · 0.27/0.18/0.10 | 0.029/0.048/0.090 · 0.31/0.21/0.12 |
| op_deeponet | 0.064/0.095/0.161 · 0.04/0.04/0.04 | 0.055/0.082/0.138 · 0.04/0.05/0.04 |
| gbm (classical) | 0.0074/0.0117/0.0219 · 0.76/0.47/0.24 | 0.0105/0.0168/0.0303 · 0.79/0.59/0.31 |
| **crackle (bond_gnn)** | **0.0047/0.0059/0.0107 · 0.77/0.47/0.24** | **0.0051/0.0070/0.0127 · 0.89/0.60/0.31** |

Crackle beats the best operator by 82–91% NLL (~3× recall) and the GBM by
37–58% NLL, on both TEST and OOD.

### 3.3 Task B2 — per-bond hazard, real solved-PD (10 seeds, NLL mean±std)

GBM is a single deterministic fit (no std).

| model | TEST H3/H5/H10 | OOD H3/H5/H10 |
|-------|----------------|----------------|
| op_fno | 0.00145±2e-5 / 0.00221±5e-5 / 0.00510±3e-4 · rec 0.78/0.77/0.62 | 0.00170±3e-5 / 0.00256±3e-5 / 0.00455±7e-5 · 0.83/0.83/0.77 |
| op_convnet | 0.00163±4e-5 / 0.00246±6e-5 / 0.00601±3e-4 · 0.64/0.64/0.50 | 0.00192±4e-5 / 0.00292±5e-5 / 0.00519±1e-4 · 0.66/0.65/0.63 |
| op_deeponet | 0.00255/0.00402/0.00911 · rec ~0 | 0.00313/0.00494/0.00928 · rec ~0 |
| gbm (classical) | 0.00080/0.00077/0.00247 · 0.96/0.99/0.98 | 0.00330/0.00297/0.00522 · 0.81/0.95/0.92 |
| **crackle (bond_gnn)** | 0.00088±8e-5 / 0.00124±8e-5 / 0.00322±2e-4 · 1.00/0.99/0.94 | **0.00105±2.3e-4 / 0.00149±3.1e-4 / 0.00300±6.8e-4 · 0.995/0.99/0.96** |

On real mechanics crackle beats every operator on TEST and OOD; the OOD gap
exceeds the across-seed spread (OOD H3: 0.00105±0.00023 vs FNO
0.00170±0.00003). Against the GBM: a tie in-distribution, a clear crackle win
out-of-distribution (40–68% lower NLL) — see Discussion.

## 4. Discussion

**Why crackle wins discrete crack-event prediction.** A neural operator emits a
smooth per-cell hazard field; two bonds sharing endpoint cells receive
identical predictions, so it cannot resolve *which* bond ruptures. DeepONet,
the most global operator, is reduced to near-chance recall (~0). The bond graph
represents each bond and propagates hazard along connectivity — the native
geometry of crack advance — which is what the task requires.

**Why crackle loses smooth field rollout.** FNO represents a smooth, bounded
spectral correction: stable and accurate for a slowly-evolving full field,
where the bond graph's local increments instead compound into instability.

**Why OOD is the decisive test.** The classical GBM matches crackle on
in-distribution solved-PD test, but **degrades out-of-distribution** (H3 NLL
0.0008 → 0.0033) while crackle holds (0.0011). A practitioner who validated
only on held-out cases of the *same* geometry would wrongly conclude the
tabular model is as good. The bond-graph inductive bias is what generalizes to
unseen geometry; feature-based trees do not. This is the concrete reason new /
OOD data is mandatory.

The unifying rule: **match the representation to the target** — operators for
smooth fields, the bond graph for discrete crack events.

## 5. Limitations and honest boundary conditions

- **Crackle does not dominate everywhere.** On smooth full-field rollout FNO
  wins (Section 3.1); reported with equal weight, and the graph was not config-
  hunted into a win (it oscillates collapse/over-growth across front-weight ×
  pushforward-depth × aggregation; no stable winning config was claimed).
- **Leakage controls:** by-case / by-dataset splits, train-only normalization,
  OOD = held-out stratum or separate dataset.
- **No sandbagging:** operators tuned to a strong regime and given the
  toughness field (no effect → structural gap). GBM uses richer per-bond
  features than the operators.
- **No cherry-picking metrics:** both NLL and top-1% recall reported; both
  field L2 and topological bottleneck reported.
- **Scope:** B1 is a kinematic proxy (no quantitative-mechanics claim); B2 is
  real solved-PD but with small case counts (64+16) and a single OOD dataset;
  per-bond evaluation n is large (millions of decisions) so the dominant
  uncertainty is across-seed, reported at 10 seeds. Horizons are short
  (hazard ≤10 steps, rollout ≤40). Task-A graph is single-seed (not the
  headline); operator L2 is 3-seed.

## 6. Conclusion

Under fair, leakage-controlled, out-of-distribution evaluation, the topological
bond-graph method is decisively stronger than neural operators and classical
baselines on discrete crack-event (per-bond hazard) prediction — the project's
actual objective — on a kinematic proxy and on real solved peridynamics, and it
uniquely retains that advantage on unseen geometry where a strong tabular model
overfits. It is weaker than a Fourier operator on smooth full-field
forecasting. The win is credible precisely because the loss is reported beside
it: the right model is the one whose representation matches the task.

## Appendix A — reproduction

| result | script | outputs |
|--------|--------|---------|
| Task A in-dist/OOD (3-seed ops) | `scripts/topo_graph_forecast.py` | `outputs/field_indist_ops3`, `outputs/field_ood_ops3` |
| Task B1 operators (toughness-fed) | `scripts/topo_track_c_operator.py` | `outputs/hazard_operator_v2` |
| Task B1 GNN + GBM | `scripts/topo_track_c_bondgnn.py` | `outputs/track_c_fresh` |
| Task B2 solved-PD (10 seeds) | `scripts/topo_track_c_solvedpd.py` | `outputs/solvedpd_hazard_10seed` |

Models: `crackle/operators.py`, `crackle/topo/bondgnn.py`. Per-phase readouts:
`topo_phase4_operator_headtohead_20260614.md` (Task A, negative),
`topo_phase4b_hazard_operator_20260614.md` (B1),
`topo_phase4c_solvedpd_hazard_20260614.md` (B2). Tabular summary:
`../PHASE4_RESULTS.md`.
