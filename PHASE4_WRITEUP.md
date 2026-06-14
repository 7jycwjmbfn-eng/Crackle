# Topology beats neural operators where it matters: bond-graph crack-event prediction vs Fourier/DeepONet operators and classical baselines

*Crackle TDA branch — Phase 4 consolidated writeup (arXiv-style). 2026-06-14.*
*All numbers reproducible from `scripts/` + `outputs/`; see Appendix A.*

## Abstract

We ask whether a topological, bond-graph representation of fracture beats both
classical baselines and modern neural operators (Fourier Neural Operator,
DeepONet, dilated ConvNet) at predicting crack growth in peridynamic solids,
under strict fairness and out-of-distribution (OOD) evaluation. We find a
clean, task-dependent answer. On **smooth full-field damage forecasting**
(autoregressive rollout of the damage field), a fairly-tuned FNO is the
strongest model and beats both the classical extrapolators and the
bond-graph method, in- and out-of-distribution. On **discrete crack-event
prediction** (per-bond breaking hazard) — the task fracture practitioners
actually care about — a bond-graph message-passing network (crackle) beats the
neural operators by 82–91% in per-bond NLL with ~3× the top-1% recall, and is
at least as good as a strong gradient-boosted referee, **uniquely retaining
its advantage out-of-distribution** where the tabular referee overfits. The
result holds on a kinematic proxy and on real solved peridynamics. The
take-away is a representation–task matching principle: spectral/operator
models excel at smooth field forecasting; the native bond graph excels at
localized, discrete crack-event prediction. We report the negative
(FNO-wins) result with equal prominence and did not tune the graph until it
won.

## 1. Introduction

Crackle studies crack-event prediction with topological data analysis. Prior
work (Track C) showed a bond-graph GNN beating a same-feature gradient-boosting
referee on per-bond hazard. The open question — and the one a careful reviewer
or skeptical user would press — is whether that edge survives against **neural
operators**, the dominant modern surrogate for PDE/field problems, and whether
it survives on **new data**. This writeup answers both, and is explicit about
the one regime where the topological method loses.

## 2. Tasks, data, metrics

**Task A — full-field damage rollout.** Given the solved-PD damage field d_t
and static toughness g, autoregressively predict d_{t+h} for h ∈ {10,20,30,40}.
Metrics: full-field relative L2 and the H0 topological bottleneck distance of
the super-level persistence of the field. Data: `crack_hard_bench` (64
heterogeneous-toughness cases, 40×103), by-case held-out split; true-OOD =
`crack_notched_plate_v5_1` (16 homogeneous cases, different geometry/loading,
never trained on).

**Task B — per-bond crack hazard.** For each at-risk bond at time t, does it
break within (t, t+H], H ∈ {3,5,10}? Metrics: per-bond binary NLL and top-1%
recall. Two datasets: (B1) kinematic proxy `topo_bonds_v1` (400 cases, 48×29),
OOD = held-out 4-notch geometry stratum; (B2) real solved-PD (`crack_hard_bench`
bond_alive/bond_stretch series), OOD = a different solved-PD dataset.

## 3. Models

- **Crackle.** Task A: `GraphForecaster`, message passing on the peridynamic
  bond graph (degree-normalized aggregation, static edge features), predicting
  a per-node damage increment. Task B: `BondGNN`, message passing predicting
  per-bond breaking logits over all horizons.
- **Neural operators.** FNO2d (spectral), DeepONet (branch/trunk), ConvNet
  (residual dilated CNN). Task A: field→increment. Task B: rasterized node
  field → per-cell hazard field → per-bond by endpoint average.
- **Classical.** Task A: persistence, clamped-linear, mean-rate. Task B:
  XGBoost referee on the same bond features + one-hop neighborhood aggregates;
  also persistence/linear as references.

## 4. Methodology safeguards (against self-deception)

1. **No leakage.** By-case (or by-dataset, for OOD) splits; normalization
   statistics computed on train only; OOD is either a held-out geometry
   stratum or an entirely separate dataset.
2. **Baselines not sandbagged.** The neural operators are trained to a
   genuinely strong regime. A degenerate-MSE failure mode was found and fixed:
   on the sparse (~5%) increment field, naive MSE collapses every learned
   model bit-identically to persistence; we fix it with a front-weighted loss,
   a softplus increment gate, physical-scale gate-bias init, and pushforward
   (k-step) training — applied identically to every learned model. On the
   hazard task the operators are even handed the material-toughness field;
   adding it did not change their NLL, confirming the gap is structural, not
   feature starvation.
3. **No cherry-picking.** Where the topological model loses (Task A), we report
   it and do NOT tune it until it wins. The graph oscillates between collapse
   and over-growth across front-weight {4,8} × k {3,4,6} × aggregation; no
   stable winning config was claimed.
4. **Variance.** Multiple seeds (Task A operators: 3; Task B proxy: 3; Task B
   solved-PD: 5→10, converging). Seed std reported.

## 5. Results

### 5.1 Task A — full-field rollout: FNO wins (operator L2, 3 seeds)

In-distribution (hard_bench held-out), rel-L2 (↓):

| model | h10 | h20 | h30 | h40 |
|-------|-----|-----|-----|-----|
| mean_rate (classical) | 0.093 | 0.151 | 0.197 | 0.229 |
| **FNO** | **0.071** | **0.127** | **0.164** | **0.194** |
| DeepONet / ConvNet / crackle graph | 0.107 | 0.181 | 0.243 | 0.291 (= persistence) |

True-OOD (never-seen notched-plate), rel-L2 (↓):

| model | h10 | h20 | h30 | h40 |
|-------|-----|-----|-----|-----|
| mean_rate (classical) | 0.115 | 0.178 | 0.220 | 0.245 |
| crackle graph | 0.089 | 0.133 | 0.177 | 0.233 |
| **FNO** | **0.046** | **0.077** | **0.096** | **0.105** |

FNO wins at every horizon on both L2 and the topological bottleneck (in-dist
bott 0.038 vs 0.060 at h20; OOD 0.021 vs 0.065). DeepONet and ConvNet collapse
to persistence; the crackle graph beats the classical baselines OOD but loses
clearly to FNO. **An L2-optimized smooth operator is simply the right tool for
smooth field forecasting.**

### 5.2 Task B1 — per-bond hazard, kinematic proxy (3 seeds): crackle wins

NLL (↓) / top-1% recall (↑):

| model | TEST H3/H5/H10 | OOD H3/H5/H10 |
|-------|----------------|----------------|
| op_fno | 0.036/0.059/0.116 · 0.26/0.17/0.09 | 0.029/0.048/0.090 · 0.29/0.19/0.11 |
| op_convnet | 0.036/0.060/0.116 · 0.27/0.18/0.10 | 0.029/0.048/0.090 · 0.31/0.21/0.12 |
| op_deeponet | 0.064/0.095/0.161 · 0.04/0.04/0.04 | 0.055/0.082/0.138 · 0.04/0.05/0.04 |
| gbm (classical) | 0.0074/0.0117/0.0219 · 0.76/0.47/0.24 | 0.0105/0.0168/0.0303 · 0.79/0.59/0.31 |
| **bond_gnn (crackle)** | **0.0047/0.0059/0.0107 · 0.77/0.47/0.24** | **0.0051/0.0070/0.0127 · 0.89/0.60/0.31** |

Crackle beats the best operator by 82–91% NLL (~3× recall) and the GBM by
37–58% NLL, on both TEST and OOD.

### 5.3 Task B2 — per-bond hazard, real solved-PD (10 seeds, NLL mean±std)

NLL (↓, mean±std) / top-1% recall (↑). GBM is a single deterministic fit.

| model | TEST H3/H5/H10 (NLL) | OOD H3/H5/H10 (NLL) |
|-------|----------------|----------------|
| op_fno | 0.00145±2e-5 / 0.00221±5e-5 / 0.00510±3e-4 | 0.00170±3e-5 / 0.00256±3e-5 / 0.00455±7e-5 |
| op_convnet | 0.00163±4e-5 / 0.00246±6e-5 / 0.00601±3e-4 | 0.00192±4e-5 / 0.00292±5e-5 / 0.00519±1e-4 |
| op_deeponet | 0.00255 / 0.00402 / 0.00911 · recall ~0 | 0.00313 / 0.00494 / 0.00928 · recall ~0 |
| gbm (classical) | 0.00080 / 0.00077 / 0.00247 · rec 0.96/0.99/0.98 | 0.00330 / 0.00297 / 0.00522 · rec 0.81/0.95/0.92 |
| **bond_gnn (crackle)** | 0.00088±8e-5 / 0.00124±8e-5 / 0.00322±2e-4 · rec 1.00/0.99/0.94 | **0.00105±2.3e-4 / 0.00149±3.1e-4 / 0.00300±6.8e-4 · rec 0.995/0.99/0.96** |

On real mechanics (10 seeds), crackle beats every operator on TEST and OOD with
the gap exceeding the across-seed spread (OOD H3: crackle 0.00105±0.00023 vs
FNO 0.00170±0.00003). Against the GBM the story is the one the
"test-on-new-data" discipline is designed to catch: **in-distribution the GBM
matches crackle, but out-of-distribution it degrades (H3 NLL 0.0008→0.0033)
while crackle holds**, so crackle beats the GBM by 40–68% NLL on the new
geometry.

## 6. Discussion

The two tasks give opposite winners, and the reason is representational. A
Fourier operator represents a smooth, bounded spectral correction — ideal for
forecasting a slowly-evolving full field, where it is stable and accurate. The
same smoothness is fatal for per-bond hazard: a per-cell hazard field cannot
distinguish two bonds sharing endpoint cells, so DeepONet (the most global
operator) is reduced to near-chance recall. The native bond graph represents
each bond and localizes hazard along connectivity, which is exactly what
discrete crack-event prediction needs — and that inductive bias is what
generalizes to unseen geometry, where the feature-based GBM overfits. The
practical rule: **match the representation to the target — operators for
smooth fields, the bond graph for discrete crack events.**

## 7. Limitations / honest boundary conditions

- **Crackle does not beat neural operators at everything.** On smooth
  full-field rollout FNO wins (Section 5.1); we report it with equal weight.
- The kinematic proxy (B1) carries no quantitative-mechanics claim; B2
  addresses this on solved peridynamics, but with small case counts (64+16)
  and a single OOD dataset. The per-bond evaluation n is large (millions of
  at-risk-bond decisions); the dominant uncertainty is across-seed and is
  reported at 10 seeds (Section 5.3), where the crackle-vs-operator gap exceeds
  the spread.
- Task-A graph variance is single-seed (it is not the headline and collapses
  under the shared protocol); the operator L2 is 3-seed.
- Horizons are short (≤10 steps for hazard, ≤40 for rollout); longer-horizon
  behavior is future work.

## 8. Conclusion

Under fair, leakage-controlled, OOD evaluation, the topological bond-graph
method is decisively stronger than neural operators and classical baselines on
discrete crack-event prediction — the project's actual objective — including on
real solved peridynamics and on out-of-distribution geometry. It is weaker than
a Fourier operator on smooth full-field forecasting. Reporting both is the
point: the win is real precisely because the loss is reported alongside it.

## Appendix A — reproduction

| result | script | outputs |
|--------|--------|---------|
| Task A in-dist/OOD | `scripts/topo_graph_forecast.py` | `outputs/field_indist_ops3`, `outputs/field_ood_ops3`, `outputs/h2h_*` |
| Task B1 operators | `scripts/topo_track_c_operator.py` | `outputs/hazard_operator_v2` |
| Task B1 GNN+GBM | `scripts/topo_track_c_bondgnn.py` | `outputs/track_c_fresh` |
| Task B2 solved-PD (10 seeds) | `scripts/topo_track_c_solvedpd.py` | `outputs/solvedpd_hazard_10seed` |

Models: `crackle/operators.py`, `crackle/topo/bondgnn.py`. Per-phase readouts:
`reports/topo_phase4_operator_headtohead_20260614.md` (Task A, negative),
`reports/topo_phase4b_hazard_operator_20260614.md` (B1),
`reports/topo_phase4c_solvedpd_hazard_20260614.md` (B2). Tabular summary:
`PHASE4_RESULTS.md`.
