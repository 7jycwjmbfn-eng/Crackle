# Crackle v1.6 Real-Data + GPU Final Push

Date: 2026-06-05

## What Was Downloaded

Real data landed under:

`<workspace>\external_datasets\real_fracture_20260605`

| Dataset | Local status | Useful for | Not useful for |
|---|---:|---|---|
| NASA PHMDC2019 aluminum lap joint | downloaded, 1.8 MB zip | real crack-length curve validation; Lamb-wave side information | bond-level spatial hazard, event location |
| DLR DIC `S_950_1.6` | downloaded, 399.8 MB zip | real DIC displacement/strain nodemaps | crack-tip/path supervised labels are not in this sub-dataset |
| DLR DIC `S_160_4.7` | downloaded and MD5 verified, 1.7 GB zip | real DIC + crack tip/path GroundTruth labels | AE event timing/energy |
| EPFL bonded-composite FCG | metadata/readme/script downloaded; 11 GB group zips not started in this pass | da/dN / crack growth curve validation after large zip | bond-level hazard |
| Dataverse AE single-cycle crack growth | metadata page found; core AE zips are restricted | source citation only | open training data |

DLR `S_160_4.7` verification:

```text
size: 1,702,341,077 bytes
md5:  E9B4A43770B32B9E255D17F45EEA5A3C
zip entries: 2509
GroundTruth files/dirs: 1671
Nodemaps files/dirs: 836
uncompressed size: ~7.06 GB
sample GroundTruth shape: 256 x 256
sample labels: {0: background, 1: crack path, 2: crack tip}
sample nodemap header includes: Force, Potential, Crack length [mm]
```

Important: the best public real files acquired so far are NASA PHMDC2019 for crack-length curves and DLR `S_160_4.7` for real DIC crack tip/path spatial labels. They still do not remove the need for a true AE/event table if the goal is real hazard-event validation with event time/energy.

## Hardware / GPU Reality

Current bundled PyTorch is CPU-only, so `paris_prior_relu_bilstm_sa_v1` trained on CPU in the real NASA curve test.

XGBoost CUDA did run:

| Model | Seeds | Rows per seed | Backend |
|---|---:|---:|---|
| `gbm_survival_v1_cuda_big` | 20260608 / 20260609 / 20260610 | 8,614,243 | `xgboost_binary_discrete_hazard_cuda` |

But inference still warns about CPU input vs CUDA booster, and active-sweep is dominated by CPU-side candidate/features/predictor transfer. This is why CUDA training did not become a fast wall-clock solver.

## Synthetic Hazard Re-Run

Dataset:

`<workspace>\crackle_runs\crackle_v1_2_hard64_20260605\data`

Hash: `900775233e6adaff`

Output:

`<workspace>\crackle_runs\crackle_v1_6_realdata_gpu_20260605\synthetic_hazard_active_sweep\synthetic_hazard_aggregate.csv`

No-threshold active sweep, mean across seeds:

| Model | active_topk | TopK recall | loc err | ms/step | speedup vs dense |
|---|---:|---:|---:|---:|---:|
| `crackle_cox_mechanics_ranker_v1` | 1024 | 0.96813 | 0.18982 | 8.733 | 0.603 |
| `gbm_survival_v1_cuda_big` | 512 | 0.96805 | 0.16729 | 11.314 | 0.469 |
| `gbm_survival_v1_cuda_big` | 1024 | 0.96805 | 0.16743 | 12.543 | 0.420 |
| `crackle_cox_fast_ranker_v1` | 1024 | 0.96243 | 0.20072 | 5.986 | 0.879 |
| `crackle_cox_fast_ranker_v1` | 512 | 0.94615 | 0.25529 | 5.730 | 0.920 |
| `cox_discrete_time_v1` | 512 | 0.91704 | 0.52118 | 4.791 | 1.102 |
| `mechanics_coupled_survival_v1` | 512 | 0.91361 | 0.54392 | 4.887 | 1.078 |
| `catalog_only_hawkes_v1` | 512 | 0.15644 | 3.34950 | 3.804 | 1.384 |
| `parametric_hawkes_etas_graph_v1` | 512 | 0.15455 | 3.20219 | 4.593 | 1.146 |

Dense reference step time in this run: about `5.263 ms/step`.

Verdict:

High-recall models are not faster than dense in full-overhead wall-clock. Cox is fast enough but loses too much recall. Hawkes is not competitive. GBM improves localization but is too slow because the active-sweep inference path is still CPU/transfer bound.

## Real NASA Curve Head-to-Head

Output:

`<workspace>\crackle_runs\crackle_v1_6_realdata_gpu_20260605\real_nasa_phm_curve_v2\real_curve_headtohead.csv`

Protocol: first 30% known, predict last 70%.

Valid curves: train `T1,T3,T4,T5,T6`; validation `T7,T8`. `T2` was too short; one non-monotone cycle typo in `T4` was dropped rather than guessed.

| Model | Curve RMSE mm | 95% CI | Notes |
|---|---:|---|---|
| `cox_discrete_time_curve_proxy_v1` | 2.1061 | [1.2681, 2.9441] | best on this tiny real split |
| `gbm_survival_curve_proxy_v1` | 2.3746 | [0.7214, 4.0278] | second; CUDA backend used |
| `traditional_last_rate` | 2.5797 | [1.1035, 4.0560] | simple traditional baseline |
| `paris_prior_relu_bilstm_sa_v1` | 3.4945 | [2.9330, 4.0560] | CPU torch; small-data unstable |
| `hawkes_curve_intensity_proxy_v1` | 4.2995 | [4.0560, 4.5431] | weak |
| `traditional_paris_powerlaw` | 24.7203 | [4.0560, 45.3847] | unstable under sparse prefix |

Verdict:

This is a real-data positive signal for Cox/GBM-style survival features, but it is not spatial hazard validation. It only validates curve forecasting.

## Competitor Curve Head-to-Head on Synthetic Hard Bench

Output:

`<workspace>\crackle_runs\crackle_v1_6_realdata_gpu_20260605\synthetic_competitor_headtohead\synthetic_competitor_family_aggregate.csv`

Lower curve RMSE is better:

| Regime | Best | Our best / note |
|---|---|---|
| `arrest_candidate` | `paris_prior_relu_bilstm_sa_v1` = 0.000616 | Cox = 0.001554, fast ranker = 0.001620 |
| `heterogeneous_toughness` | `crackle_cox_fast_ranker_v1` = 0.001548 | beats Paris = 0.002926 |
| `hole_plus_notch` | `paris_prior_relu_bilstm_sa_v1` = 0.001141 | Cox = 0.001487 |
| `mixed_mode` | `paris_prior_relu_bilstm_sa_v1` = 0.001348 | fast ranker = 0.001664 |

Answer to the arrest question:

On this synthetic split, the competitor wins arrest curve forecasting. It is not just a mean artifact. Our advantage should be framed around spatial event localization / branch-risk / full-domain hazard, not 1D Paris-style curve RMSE.

## Decision

Main spine for spatial hazard:

`crackle_cox_fast_ranker_v1`

Quality/diagnostic upper bounds:

`crackle_cox_mechanics_ranker_v1`, `gbm_survival_v1_cuda_big`

Cheap but lower-quality fallback:

`cox_discrete_time_v1`

Downgrade:

`catalog_only_hawkes_v1`, `parametric_hawkes_etas_graph_v1`, `traditional_paris_powerlaw`

Claim boundary:

Do not claim this beats dense PD/FEM at matched physical fidelity yet. Do claim that the event/hazard direction is alive, that real curve validation is now wired in, and that the next decisive work is GPU-side active-set/features/predictor fusion plus true AE/event-table acquisition.

